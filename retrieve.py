"""질문과 가까운 글을 고른다. 답은 아직 만들지 않는다.

  .venv/bin/python retrieve.py "식물은 햇빛을 어떻게 에너지로 바꾸나?"
  .venv/bin/python retrieve.py --check
  .venv/bin/python retrieve.py --lang en "Why does it rain?"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
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


def search(question: str, chunks: list[dict], model: SentenceTransformer, k: int):
    # 질문과 글을 같은 모델로 숫자 벡터로 바꾼다.
    # 벡터 길이를 1로 맞추면, 서로 곱한 값이 코사인 유사도다. 1에 가까울수록 뜻이 가깝다.
    vectors = model.encode(
        [question, *[chunk["text"] for chunk in chunks]],
        normalize_embeddings=True,
    )
    scores = np.asarray(vectors[0]) @ np.asarray(vectors[1:]).T
    order = np.argsort(-scores)[:k]
    return [(chunks[int(i)], float(scores[int(i)])) for i in order]


def show(hits) -> None:
    for rank, (chunk, score) in enumerate(hits, start=1):
        preview = chunk["body"].replace("\n", " ")
        if len(preview) > 140:
            preview = preview[:140] + "…"
        print(f"{rank}. {chunk['title']}  ({score:.3f})  [{chunk['file']}]")
        print(f"   {preview}")


def check(lang: str, chunks: list[dict], model: SentenceTransformer) -> int:
    key = json.loads((ROOT / "data" / "answer_key.json").read_text(encoding="utf-8"))
    ok = 0
    for item in key["items"]:
        hits = search(item[lang], chunks, model, k=1)
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

    chunks = load_chunks(args.lang)
    print(f"모델 준비: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    if args.check:
        raise SystemExit(check(args.lang, chunks, model))
    show(search(args.question, chunks, model, args.k))


if __name__ == "__main__":
    main()
