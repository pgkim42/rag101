"""질문과 가까운 글을 골라 질문 앞에 붙이고, 생성 모델이 그 글만 보고 답을 쓰게 한다.

  .venv/bin/python answer.py "식물은 햇빛을 어떻게 에너지로 바꾸나?"
  .venv/bin/python answer.py --k 1 "허가되지 않은 네트워크 통신을 무엇이 막나?"
  .venv/bin/python answer.py --show-prompt "왜 비가 내리나?"

생성 모델은 OpenAI 호환 엔드포인트면 무엇이든 된다. 주소, 키, 모델은 .env에 둔다.
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from retrieve import ROOT, build_searcher, search

from openai import APIError, OpenAI

DEFAULT_LLM = "gpt-6-luna"

INSTRUCTION = (
    "아래 글만 근거로 답하세요. 글에 없는 내용은 모른다고 하세요.\n"
    "답에 쓴 글의 번호를 [1]처럼 붙이고, 질문과 같은 언어로 답하세요."
)


def load_settings() -> str:
    # .env가 있으면 읽는다. 이미 셸에 있는 값이 우선이다.
    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY가 없습니다. .env.example을 .env로 복사해서 채우세요.")
    return os.environ.get("RAG_LLM_MODEL", DEFAULT_LLM)


def build_prompt(question: str, hits) -> str:
    # 증강: 검색한 글을 질문 앞에 붙여 입력을 키운다.
    docs = "\n\n".join(
        f"[{i}] {chunk['title']}\n{chunk['body']}" for i, (chunk, _) in enumerate(hits, start=1)
    )
    return f"{INSTRUCTION}\n\n글:\n{docs}\n\n질문: {question}"


def generate(prompt: str, model: str) -> str:
    # 생성: 붙인 글을 읽은 모델이 답을 쓴다.
    # OpenAI()는 OPENAI_BASE_URL과 OPENAI_API_KEY를 환경에서 읽는다.
    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
    except APIError as error:
        sys.exit(f"생성 모델 호출에 실패했습니다 ({model}): {error}")
    return (response.choices[0].message.content or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="가까운 글을 붙여 답을 만든다.")
    parser.add_argument("question", help="물어볼 질문")
    parser.add_argument("--lang", choices=("ko", "en"), default="ko")
    parser.add_argument("--k", type=int, default=3, help="답의 근거로 붙일 글 개수")
    parser.add_argument("--show-prompt", action="store_true", help="모델에게 보낸 입력을 보여 준다")
    args = parser.parse_args()

    llm = load_settings()
    hits = search(args.question, build_searcher(args.lang), args.k)

    print("\n근거로 붙인 글:")
    for rank, (chunk, score) in enumerate(hits, start=1):
        print(f"[{rank}] {chunk['title']}  ({score:.3f})  [{chunk['file']}]")

    prompt = build_prompt(args.question, hits)
    if args.show_prompt:
        print(f"\n--- 모델에게 보낸 입력 ---\n{prompt}\n---")

    print(f"\n답 ({llm}):")
    print(generate(prompt, llm))


if __name__ == "__main__":
    main()
