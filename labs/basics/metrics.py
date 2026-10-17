"""F장 실습: 검색 지표를 손 계산과 맞추고, 작은 평가 데이터의 불확실성과 과적합을 시뮬레이션한다.

  .venv/bin/python labs/basics/metrics.py

표준 라이브러리만 쓴다. 1~2부의 순위는 설명용 예시다. 4부의 12/15, 14/15, 15/15는 이 저장소에서 잰 값이다(guide/04).
5~6부는 난수 씨앗을 고정한 시뮬레이션이라 몇 번 돌려도 같은 값이 나온다.
"""

from __future__ import annotations

import math
import random

RANKING = ["d3", "d1", "d5", "d2", "d4"]
GRADE = {"d1": 2, "d2": 1}


def hit_at(ranking: list[str], relevant: set[str], k: int) -> float:
    return float(any(d in relevant for d in ranking[:k]))


def precision_at(ranking: list[str], relevant: set[str], k: int) -> float:
    return sum(d in relevant for d in ranking[:k]) / k


def recall_at(ranking: list[str], relevant: set[str], k: int) -> float:
    return sum(d in relevant for d in ranking[:k]) / len(relevant)


def reciprocal_rank(ranking: list[str], relevant: set[str]) -> float:
    for r, d in enumerate(ranking, start=1):
        if d in relevant:
            return 1 / r
    return 0.0


def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(r + 1) for r, g in enumerate(gains, start=1))


def ndcg_at(ranking: list[str], grade: dict[str, int], k: int) -> float:
    gains = [2 ** grade.get(d, 0) - 1 for d in ranking[:k]]
    ideal = sorted((2**g - 1 for g in grade.values()), reverse=True)[:k]
    return dcg(gains) / dcg(ideal)


def close(a: float, b: float) -> bool:
    return abs(a - b) < 5e-4


def part1() -> None:
    print("=== 1부. 질문 하나, 정답 글 두 편 (d1은 완전한 답, d2는 부분적인 답) ===")
    rel = set(GRADE)
    print(f"검색 순위: {RANKING}, 정답: {sorted(rel)}")
    print(f"{'k':>3}{'Hit@k':>8}{'P@k':>8}{'R@k':>8}")
    for k in (1, 2, 3, 5):
        print(f"{k:>3}{hit_at(RANKING, rel, k):>8.3f}{precision_at(RANKING, rel, k):>8.3f}{recall_at(RANKING, rel, k):>8.3f}")
    rr = reciprocal_rank(RANKING, rel)
    binary = ndcg_at(RANKING, {d: 1 for d in rel}, 5)
    graded = ndcg_at(RANKING, GRADE, 5)
    print(f"RR = 1/{RANKING.index('d1') + 1} = {rr:.4f}")
    print(f"nDCG@5 (맞다/틀리다만) = {binary:.4f},  nDCG@5 (d1=2점, d2=1점) = {graded:.4f}")
    assert close(precision_at(RANKING, rel, 3), 1 / 3) and close(recall_at(RANKING, rel, 3), 0.5)
    assert close(binary, 0.6509) and close(graded, 0.6399)

    print("\n순서를 바꾸면: d2를 2등, d1을 4등으로")
    swapped = ["d3", "d2", "d5", "d1", "d4"]
    print(f"Recall@5 {recall_at(swapped, rel, 5):.3f} (같음), RR {reciprocal_rank(swapped, rel):.3f} (같음), "
          f"등급 nDCG@5 {ndcg_at(swapped, GRADE, 5):.4f} (낮아짐)")
    assert ndcg_at(swapped, GRADE, 5) < graded


def part2() -> None:
    print("\n=== 2부. 질문 여러 개: MRR ===")
    first = [1, 3, None]
    rrs = [1 / r if r else 0.0 for r in first]
    print(f"첫 정답 순위 {first} → RR {[round(x, 4) for x in rrs]} → MRR {sum(rrs) / len(rrs):.4f}")
    assert close(sum(rrs) / 3, 0.4444)

    print("\n=== 3부. k를 늘리면 Recall은 오르고 Precision은 내려간다 (정답 1편인 질문) ===")
    print(f"{'k':>3}{'Recall@k':>10}{'Precision@k':>13}")
    for k in (1, 3, 5, 10):
        print(f"{k:>3}{1.0:>10.3f}{1 / k:>13.3f}   (정답이 1등일 때)")


def wilson(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = x / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def part3() -> None:
    print("\n=== 4부. 15문항의 숫자는 얼마나 흔들리나 (95% 윌슨 구간) ===")
    for x, n in ((12, 15), (14, 15), (15, 15), (120, 150), (140, 150)):
        lo, hi = wilson(x, n)
        print(f"  {x:>3}/{n:<3} = {x / n:.3f} → 대략 {lo:.3f} ~ {hi:.3f}")
    lo12, hi12 = wilson(12, 15)
    lo14, _ = wilson(14, 15)
    assert lo14 < hi12, "12/15와 14/15의 구간이 겹친다"
    assert close(lo12, 0.5481) and close(hi12, 0.9295)

    print("\n같은 15문항에서 두 방식이 갈린 문항만 본다 (부호 검정)")
    print("  ko dense 12/15 vs hybrid 14/15: 결과가 다른 문항은 2개, 둘 다 hybrid가 맞힘")
    p = 0.5**2
    print(f"  두 방식이 사실 똑같이 좋은데 우연히 2개 모두 hybrid 쪽으로 갈릴 확률 = 0.5 × 0.5 = {p:.2f}")
    assert p == 0.25


def part4() -> None:
    print("\n=== 5부. 실력이 똑같은 두 방식(정답률 80%)을 15문항으로 비교하면 ===")
    rng = random.Random(42)
    trials, gaps = 20_000, 0
    for _ in range(trials):
        a = sum(rng.random() < 0.8 for _ in range(15))
        b = sum(rng.random() < 0.8 for _ in range(15))
        gaps += abs(a - b) >= 2
    print(f"  {trials:,}번 중 {gaps / trials:.1%}에서 2문항 이상 차이가 났다 (서로 독립된 문항 가정)")

    print("\n=== 6부. 설정 10개를 15문항으로 고르면 생기는 착시 (모두 실제 정답률 80%) ===")
    rng = random.Random(7)
    dev_best, test_of_best = 0.0, 0.0
    trials = 5_000
    for _ in range(trials):
        dev = [sum(rng.random() < 0.8 for _ in range(15)) / 15 for _ in range(10)]
        dev_best += max(dev)
        test_of_best += sum(rng.random() < 0.8 for _ in range(15)) / 15
    print(f"  개발용 15문항에서 1등 설정의 평균 점수: {dev_best / trials:.3f}")
    print(f"  그 설정을 새 15문항(시험용)에서 잰 평균: {test_of_best / trials:.3f}")
    assert dev_best / trials > 0.88 > test_of_best / trials


if __name__ == "__main__":
    part1()
    part2()
    part3()
    part4()
    print("\n모든 손 계산 검사를 통과했다.")
