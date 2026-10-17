"""질문과 가까운 글을 고른다. 답은 answer.py가 만든다.

  .venv/bin/python retrieve.py "식물은 햇빛을 어떻게 에너지로 바꾸나?"
  .venv/bin/python retrieve.py --check
  .venv/bin/python retrieve.py --lang en "Why does it rain?"
  .venv/bin/python retrieve.py --method dense --check
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, assert_never

os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.disable(logging.WARNING)

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from bm25 import BM25

logging.disable(logging.NOTSET)

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL: Final = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
CACHE_DIR = ROOT / ".cache"

Method = Literal["dense", "hybrid", "rerank"]
METHODS: Final = ("dense", "hybrid", "rerank")
# hybrid에서 임베딩 점수에 주는 비중. 나머지는 BM25 몫이다.
DENSE_WEIGHT: Final = 0.7
# rerank는 hybrid 상위 몇 편만 다시 채점한다.
RERANK_POOL: Final = 5


@dataclass(frozen=True, slots=True)
class Searcher:
    """질문을 받기 전에 한 번만 준비해 두는 것들."""

    chunks: list[dict]
    embedder: SentenceTransformer
    doc_vectors: np.ndarray
    bm25: BM25
    method: Method


def load_chunks(lang: str) -> list[dict]:
    folder = ROOT / "docs" / lang
    chunks = []
    for path in sorted(folder.glob("*.md")):
        title, body = parse_doc(path.read_text(encoding="utf-8"))
        # 이 샘플은 글이 짧아서 파일 하나가 조각 하나다.
        # 글이 길어지면 이 자리에서 문단이나 토큰 수로 나눈다.
        chunks.append(
            {"file": path.name, "title": title, "text": f"{title}\n{body}", "body": body}
        )
    if not chunks:
        sys.exit(f"문서가 없습니다: {folder}")
    return chunks


def parse_doc(raw: str) -> tuple[str, str]:
    text = raw.strip()
    if text.startswith("---"):
        _, _, text = text.split("---", 2)
        text = text.strip()
    title = ""
    body_lines = []
    for line in text.splitlines():
        if not title and line.startswith("# "):
            title = line[2:].strip()
            continue
        body_lines.append(line)
    return title, "\n".join(body_lines).strip()


def fingerprint(chunks: list[dict]) -> str:
    # 모델과 글이 같으면 임베딩도 같다. 그 둘을 해시해 캐시의 이름표로 쓴다.
    digest = hashlib.sha256(MODEL_NAME.encode())
    for chunk in chunks:
        digest.update(f"\0{chunk['file']}\0{chunk['text']}".encode())
    return digest.hexdigest()


def load_doc_vectors(chunks: list[dict], embedder: SentenceTransformer, lang: str) -> np.ndarray:
    """글 임베딩을 캐시에서 읽는다. 이름표가 다르거나 파일이 상했으면 다시 계산해 저장한다."""
    path = CACHE_DIR / f"embeddings-{lang}.npz"
    expected = fingerprint(chunks)
    reason = "캐시 없음"
    if path.exists():
        try:
            with np.load(path) as saved:
                if str(saved["fingerprint"]) == expected:
                    print(f"글 임베딩 캐시 사용: {path.relative_to(ROOT)}")
                    return saved["vectors"]
            reason = "글이나 모델이 바뀜"
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            reason = "캐시를 읽을 수 없음"
    vectors = embedder.encode([chunk["text"] for chunk in chunks], normalize_embeddings=True)
    CACHE_DIR.mkdir(exist_ok=True)
    # 다 쓴 다음에 이름을 바꾼다. 쓰다 죽어도 반쯤 쓴 파일이 캐시 자리에 남지 않는다.
    partial = path.with_suffix(".partial")
    with partial.open("wb") as file:
        np.savez(file, fingerprint=expected, vectors=vectors)
    partial.replace(path)
    print(f"글 임베딩 계산 후 저장 ({reason}): {path.relative_to(ROOT)}")
    return vectors


def build_searcher(lang: str, method: Method) -> Searcher:
    chunks = load_chunks(lang)
    print(f"모델 준비: {MODEL_NAME}  (검색 방식: {method})")
    embedder = SentenceTransformer(MODEL_NAME)
    vectors = load_doc_vectors(chunks, embedder, lang)
    return Searcher(chunks, embedder, vectors, BM25([chunk["text"] for chunk in chunks]), method)


@functools.cache
def load_reranker() -> CrossEncoder:
    print(f"재순위화 모델 준비: {RERANK_MODEL}")
    return CrossEncoder(RERANK_MODEL)


def minmax(scores: np.ndarray) -> np.ndarray:
    # 점수를 0~1로 편다. 코사인과 BM25는 자가 달라서 그대로 더하면 BM25가 덮어 버린다.
    spread = scores.max() - scores.min()
    return (scores - scores.min()) / spread if spread > 0 else np.zeros_like(scores)


def search(question: str, searcher: Searcher, k: int):
    # 질문만 새로 숫자 벡터로 바꾼다. 글 벡터는 미리 만들어 둔 것을 쓴다.
    # 벡터 길이를 1로 맞추면, 서로 곱한 값이 코사인 유사도다. 1에 가까울수록 뜻이 가깝다.
    query = searcher.embedder.encode([question], normalize_embeddings=True)[0]
    dense = searcher.doc_vectors @ query
    match searcher.method:
        case "dense":
            scores = dense
        case "hybrid":
            scores = hybrid(question, searcher, dense)
        case "rerank":
            return rerank(question, searcher, hybrid(question, searcher, dense), k)
        case unreachable:
            assert_never(unreachable)
    order = np.argsort(-scores)[:k]
    return [(searcher.chunks[int(i)], float(scores[int(i)])) for i in order]


def hybrid(question: str, searcher: Searcher, dense: np.ndarray) -> np.ndarray:
    # 뜻이 가까운지(임베딩)와 같은 조각이 있는지(BM25)를 같은 자로 맞춰 섞는다.
    lexical = searcher.bm25.scores(question)
    return DENSE_WEIGHT * minmax(dense) + (1 - DENSE_WEIGHT) * minmax(lexical)


def rerank(question: str, searcher: Searcher, base: np.ndarray, k: int):
    # cross-encoder는 질문과 글을 한 입력으로 같이 읽고 둘이 맞는 정도를 바로 낸다.
    # 느리니 전부가 아니라 base 점수로 고른 후보만 다시 채점한다.
    pool = np.argsort(-base)[: max(RERANK_POOL, k)]
    scores = load_reranker().predict([(question, searcher.chunks[int(i)]["text"]) for i in pool])
    order = np.argsort(-scores)[:k]
    return [(searcher.chunks[int(pool[j])], float(scores[j])) for j in order]


def show(hits) -> None:
    for rank, (chunk, score) in enumerate(hits, start=1):
        preview = chunk["body"].replace("\n", " ")
        if len(preview) > 140:
            preview = preview[:140] + "…"
        print(f"{rank}. {chunk['title']}  ({score:.3f})  [{chunk['file']}]")
        print(f"   {preview}")


def check(searcher: Searcher, lang: str) -> int:
    key = json.loads((ROOT / "data" / "answer_key.json").read_text(encoding="utf-8"))
    ok = 0
    for item in key["items"]:
        hits = search(item[lang], searcher, k=1)
        got = hits[0][0]["file"]
        passed = got == item["file"]
        ok += int(passed)
        mark = "맞음" if passed else "다름"
        print(f"[{mark}] {item[lang]}")
        print(f"      기대: {item['file']}  결과: {got}  ({hits[0][1]:.3f})")
    print(f"\n{ok}/{len(key['items'])}")
    return 0 if ok == len(key["items"]) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="질문과 가까운 글을 찾는다.")
    parser.add_argument("question", nargs="?", help="찾을 질문")
    parser.add_argument("--lang", choices=("ko", "en"), default="ko")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--method", choices=METHODS, default="rerank", help="검색 방식")
    parser.add_argument("--check", action="store_true", help="정답표 15문항을 채점한다")
    args = parser.parse_args()
    if not args.check and not args.question:
        parser.error("질문을 넘기거나 --check 를 쓰세요.")

    searcher = build_searcher(args.lang, args.method)
    if args.check:
        raise SystemExit(check(searcher, args.lang))
    show(search(args.question, searcher, args.k))


if __name__ == "__main__":
    main()
