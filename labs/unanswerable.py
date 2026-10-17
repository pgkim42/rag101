"""글에 답이 없는 질문을 받으면 검색과 생성이 어떻게 구는지 본다.

  .venv/bin/python -m labs.unanswerable             # 검색 점수 + 생성 모델 답
  .venv/bin/python -m labs.unanswerable --skip-llm  # 키 없이 검색 점수만

질문은 세 묶음이다.
- 답 있음: 정답표 15문항
- 주제 밖: 15편 어디에도 없는 주제
- 주제는 같지만 답이 없음: 15편의 주제를 건드리지만 글에 답이 적혀 있지 않은 질문
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import numpy as np

from answer import build_prompt, generate, load_settings
from retrieve import ROOT, build_searcher, search

OFF_TOPIC: Final = (
    "김치찌개는 어떻게 끓이나?",
    "축구에서 오프사이드는 무엇인가?",
    "파이썬에서 리스트를 정렬하는 방법은?",
    "블랙홀은 어떻게 생기나?",
    "고양이는 왜 골골거리나?",
    "조선을 세운 사람은 누구인가?",
    "커피를 마시면 왜 잠이 안 오나?",
    "비행기는 어떻게 하늘을 나나?",
)
SAME_TOPIC_NO_ANSWER: Final = (
    "방화벽 제품은 어떤 회사가 만드나?",
    "VPN을 쓰면 인터넷 속도가 느려지나?",
    "광합성을 처음 발견한 사람은 누구인가?",
    "백신 부작용에는 어떤 것이 있나?",
    "인플레이션이 가장 심했던 나라는 어디인가?",
    "DNS 서버 주소는 어떻게 바꾸나?",
)
# 생성 모델의 답이 「모른다」 쪽인지 대충 가르는 말들. 사람이 답 전문을 같이 읽어야 한다.
REFUSAL_WORDS: Final = ("모릅니다", "모르", "알 수 없", "나와 있지 않", "나와있지 않", "찾을 수 없", "정보가 없", "언급", "포함되어 있지 않", "없어서")


def top_score(question: str, searcher) -> tuple[float, str]:
    chunk, score = search(question, searcher, 1)[0]
    return score, chunk["title"]


def describe(scores: list[float]) -> str:
    return f"최소 {min(scores):7.3f}  평균 {np.mean(scores):7.3f}  최대 {max(scores):7.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="답이 없는 질문에서 검색 점수와 생성 모델의 답을 본다.")
    parser.add_argument("--skip-llm", action="store_true", help="생성 모델을 부르지 않는다")
    args = parser.parse_args()
    llm = None if args.skip_llm else load_settings()

    answerable = [item["ko"] for item in json.loads((ROOT / "data" / "answer_key.json").read_text(encoding="utf-8"))["items"]]
    groups = {"답 있음": answerable, "주제 밖": list(OFF_TOPIC), "주제는 같지만 답 없음": list(SAME_TOPIC_NO_ANSWER)}
    dense = build_searcher("ko", "dense")
    rerank = dataclasses.replace(dense, method="rerank")

    print("\n1등 글의 점수 (dense는 코사인, rerank는 재순위화 모델 점수)")
    scores: dict[str, dict[str, list[float]]] = {}
    for name, questions in groups.items():
        scores[name] = {"dense": [top_score(q, dense)[0] for q in questions], "rerank": [top_score(q, rerank)[0] for q in questions]}
        print(f"  {name:14} ({len(questions):2}개)  dense  {describe(scores[name]['dense'])}")
        print(f"  {'':14}        rerank {describe(scores[name]['rerank'])}")

    for method in ("dense", "rerank"):
        lowest_answerable = min(scores["답 있음"][method])
        above = {name: sum(s >= lowest_answerable for s in scores[name][method]) for name in groups if name != "답 있음"}
        print(f"\n{method}: 답 있음 중 가장 낮은 점수 {lowest_answerable:.3f}. 이 값 이상이면 통과시키는 문턱을 두면 "
              + ", ".join(f"{name} {n}/{len(groups[name])}개가 통과" for name, n in above.items()))

    print("\n질문별 rerank 1등")
    for name in ("주제 밖", "주제는 같지만 답 없음"):
        for q in groups[name]:
            score, title = top_score(q, rerank)
            print(f"  [{name}] {score:7.3f}  {title:12}  {q}")

    if llm is None:
        return
    print(f"\n생성 모델({llm})에 rerank 위 3편을 붙여 물었다. 「모른다」 쪽 판단은 낱말로 대충 가른 것이다.")
    # 검색은 여기서 차례로 끝낸다. 임베딩 모델은 GPU(MPS)를 여러 스레드가 동시에 쓰면 죽는다.
    # 기다리는 시간이 긴 생성 모델 호출만 동시에 보낸다.
    jobs = [(name, q, build_prompt(q, search(q, rerank, 3))) for name, questions in groups.items() for q in questions]

    def ask(job: tuple[str, str, str]) -> tuple[str, str, str]:
        name, q, prompt = job
        return name, q, generate(prompt, llm)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(ask, jobs))
    for name in groups:
        rows = [(q, a) for n, q, a in results if n == name]
        refused = [any(word in a for word in REFUSAL_WORDS) for _, a in rows]
        print(f"\n== {name}: {sum(refused)}/{len(rows)}개가 「모른다」 쪽")
        for (q, a), r in zip(rows, refused, strict=True):
            print(f"  {'모름' if r else '답함'} | {q}\n         → {a.replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
