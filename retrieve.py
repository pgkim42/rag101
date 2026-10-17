"""질문과 가까운 글을 고른다. 답은 answer.py가 만든다.

  .venv/bin/python retrieve.py "식물은 햇빛을 어떻게 에너지로 바꾸나?"
  .venv/bin/python retrieve.py --check
  .venv/bin/python retrieve.py --lang en "Why does it rain?"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.disable(logging.WARNING)

import numpy as np
from sentence_transformers import SentenceTransformer

logging.disable(logging.NOTSET)

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CACHE_DIR = ROOT / ".cache"


@dataclass(frozen=True, slots=True)
class Searcher:
    """질문을 받기 전에 한 번만 준비해 두는 것들."""

    chunks: list[dict]
    embedder: SentenceTransformer
    doc_vectors: np.ndarray


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


def build_searcher(lang: str) -> Searcher:
    chunks = load_chunks(lang)
    print(f"모델 준비: {MODEL_NAME}")
    embedder = SentenceTransformer(MODEL_NAME)
    return Searcher(chunks, embedder, load_doc_vectors(chunks, embedder, lang))


def search(question: str, searcher: Searcher, k: int):
    # 질문만 새로 숫자 벡터로 바꾼다. 글 벡터는 미리 만들어 둔 것을 쓴다.
    # 벡터 길이를 1로 맞추면, 서로 곱한 값이 코사인 유사도다. 1에 가까울수록 뜻이 가깝다.
    query = searcher.embedder.encode([question], normalize_embeddings=True)[0]
    scores = searcher.doc_vectors @ query
    order = np.argsort(-scores)[:k]
    return [(searcher.chunks[int(i)], float(scores[int(i)])) for i in order]


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
    parser.add_argument("--check", action="store_true", help="정답표 15문항을 채점한다")
    args = parser.parse_args()
    if not args.check and not args.question:
        parser.error("질문을 넘기거나 --check 를 쓰세요.")

    searcher = build_searcher(args.lang)
    if args.check:
        raise SystemExit(check(searcher, args.lang))
    show(search(args.question, searcher, args.k))


if __name__ == "__main__":
    main()
