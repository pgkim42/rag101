"""검색이 얼마나 잘하는지를 1등 하나보다 자세히 잰다.

  .venv/bin/python -m labs.evaluate
  .venv/bin/python -m labs.evaluate --lang ko

정답표 15문항마다 기대한 글이 위에서 몇 등에 나왔는지 보고, 그 순위로 지표 네 개를 낸다.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from typing import Final

from retrieve import METHODS, ROOT, Searcher, build_searcher, search

CUTOFF: Final = 5


@dataclasses.dataclass(frozen=True, slots=True)
class Scores:
    hit_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr_at_5: float


def rank_of(expected_file: str, hits) -> int | None:
    for rank, (chunk, _) in enumerate(hits, start=1):
        if chunk["file"] == expected_file:
            return rank
    return None


def summarize(ranks: list[int | None]) -> Scores:
    # 순위가 None이면 위에서 CUTOFF편 안에 없었다는 뜻이다. 어느 지표에서도 0점이다.
    n = len(ranks)

    def within(k: int) -> float:
        return sum(rank is not None and rank <= k for rank in ranks) / n

    mrr = sum(1 / rank for rank in ranks if rank is not None) / n
    return Scores(within(1), within(3), within(CUTOFF), mrr)


def evaluate(searcher: Searcher, questions: list[str], expected: list[str]) -> list[int | None]:
    return [rank_of(file, search(q, searcher, CUTOFF)) for q, file in zip(questions, expected, strict=True)]


def main() -> None:
    parser = argparse.ArgumentParser(description="검색 방식별 평가 지표를 낸다.")
    parser.add_argument("--lang", choices=("ko", "en"), help="한 언어만 잴 때")
    args = parser.parse_args()
    items = json.loads((ROOT / "data" / "answer_key.json").read_text(encoding="utf-8"))["items"]

    for lang in [args.lang] if args.lang else ["ko", "en"]:
        base = build_searcher(lang, "dense")
        questions = [item[lang] for item in items]
        expected = [item["file"] for item in items]
        ranks = {m: evaluate(dataclasses.replace(base, method=m), questions, expected) for m in METHODS}

        print(f"\n== {lang}: 질문 {len(items)}개, 위에서 {CUTOFF}편까지 본다")
        print(f"{'방식':8}{'Hit@1':>8}{'Recall@3':>10}{'Recall@5':>10}{'MRR@5':>8}")
        for method, method_ranks in ranks.items():
            s = summarize(method_ranks)
            print(f"{method:8}{s.hit_at_1:>8.3f}{s.recall_at_3:>10.3f}{s.recall_at_5:>10.3f}{s.mrr_at_5:>8.3f}")

        print(f"\n문항별로 기대한 글이 나온 순위 (-는 {CUTOFF}등 안에 없음)")
        print("  #  " + "".join(f"{m:>8}" for m in METHODS) + "  질문")
        for i, item in enumerate(items):
            cells = "".join(f"{ranks[m][i] or '-':>8}" for m in METHODS)
            print(f"{item['id']:>3}  {cells}  {item[lang]}")


if __name__ == "__main__":
    main()
