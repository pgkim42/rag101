"""D장 실습: 점수 섞기(min-max, z-점수, RRF)와 정확한/근사 최근접 이웃 검색.

  .venv/bin/python labs/basics/fusion.py

표준 라이브러리만 쓴다. 1~3부의 점수는 설명용 예시 값이다. 실제 15편의 점수는 labs/score_fusion.py에서 본다.
4부는 무작위로 만든 벡터 4,000개로 근사 검색(IVF)을 직접 만들어 본다.
"""

from __future__ import annotations

import math
import random
import statistics
import time

DOCS = ["방화벽", "VPN", "HTTP", "광합성"]
DENSE = [0.80, 0.82, 0.55, 0.10]
LEXICAL = [12.0, 3.1, 2.9, 0.0]


def minmax(xs: list[float]) -> list[float]:
    lo, hi = min(xs), max(xs)
    return [(x - lo) / (hi - lo) if hi > lo else 0.0 for x in xs]


def zscore(xs: list[float]) -> list[float]:
    mu, sd = statistics.mean(xs), statistics.pstdev(xs)
    return [(x - mu) / sd if sd else 0.0 for x in xs]


def ranks(xs: list[float]) -> list[int]:
    order = sorted(range(len(xs)), key=lambda i: -xs[i])
    out = [0] * len(xs)
    for r, i in enumerate(order, start=1):
        out[i] = r
    return out


def rrf(*score_lists: list[float], k: int = 60) -> list[float]:
    total = [0.0] * len(score_lists[0])
    for xs in score_lists:
        for i, r in enumerate(ranks(xs)):
            total[i] += 1 / (k + r)
    return total


def close(a: float, b: float) -> bool:
    return abs(a - b) < 5e-4


def table(title: str, cols: dict[str, list[float]]) -> None:
    print(f"\n{title}")
    print(f"{'글':8}" + "".join(f"{c:>12}" for c in cols))
    for i, d in enumerate(DOCS):
        print(f"{d:8}" + "".join(f"{cols[c][i]:>12.4f}" for c in cols))


def part1() -> None:
    print("=== 1부. 자가 다른 두 점수를 그대로 더하면 ===")
    raw = [d + l for d, l in zip(DENSE, LEXICAL, strict=True)]
    table("예시 점수", {"dense": DENSE, "BM25": LEXICAL, "그냥 더함": raw})
    print("그냥 더하면 BM25의 자(0~12)가 dense의 자(0~1)를 덮는다. 순서가 BM25 순서와 같다.")
    assert ranks(raw) == ranks(LEXICAL)

    print("\n=== 2부. 자 맞추기: min-max, z-점수, 그리고 순위만 쓰는 RRF ===")
    mm_d, mm_l = minmax(DENSE), minmax(LEXICAL)
    hyb = [0.7 * a + 0.3 * b for a, b in zip(mm_d, mm_l, strict=True)]
    table("min-max 후 0.7·dense + 0.3·BM25", {"dense→0~1": mm_d, "BM25→0~1": mm_l, "hybrid": hyb})
    assert close(mm_d[0], 0.9722) and close(hyb[0], 0.9806) and close(hyb[1], 0.7775)

    z = [0.7 * a + 0.3 * b for a, b in zip(zscore(DENSE), zscore(LEXICAL), strict=True)]
    table("z-점수 후 0.7·dense + 0.3·BM25", {"dense z": zscore(DENSE), "BM25 z": zscore(LEXICAL), "hybrid": z})

    fused = rrf(DENSE, LEXICAL)
    table("RRF (k=60): 1/(60+dense 순위) + 1/(60+BM25 순위)",
          {"dense 순위": [float(r) for r in ranks(DENSE)], "BM25 순위": [float(r) for r in ranks(LEXICAL)], "RRF": fused})
    assert close(fused[0], 1 / 62 + 1 / 61) and close(fused[1], 1 / 61 + 1 / 62)
    print("RRF에서 방화벽과 VPN은 1/61 + 1/62로 정확히 같다. dense의 0.02 차이와 BM25의 8.9 차이가 똑같이 「한 칸」이 되었다.")


def part2() -> None:
    print("\n=== 3부. min-max는 튀는 값 하나에 흔들린다 ===")
    spiky = [40.0, 3.1, 2.9, 0.0]
    print(f"BM25가 {LEXICAL} → min-max {[round(x, 3) for x in minmax(LEXICAL)]}")
    print(f"BM25가 {spiky} → min-max {[round(x, 3) for x in minmax(spiky)]}")
    print("1등이 하나 튀면 나머지가 0 근처로 눌린다. 2~3등 사이의 차이가 섞기에서 거의 사라진다.")
    assert minmax(spiky)[1] < 0.08 < minmax(LEXICAL)[1]


def make_vectors(rng: random.Random, n: int, dim: int, clusters: int) -> list[list[float]]:
    # 실제 임베딩처럼 주제별로 뭉친 점들을 만든다: 중심 몇 개 + 작은 흔들림, 그리고 길이 1.
    centers = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(clusters)]
    out = []
    for _ in range(n):
        c = rng.choice(centers)
        v = [x + rng.gauss(0, 0.6) for x in c]
        s = math.sqrt(sum(x * x for x in v))
        out.append([x / s for x in v])
    return out


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def kmeans(data: list[list[float]], k: int, rng: random.Random, rounds: int = 4) -> tuple[list[list[float]], list[int]]:
    cents = [list(v) for v in rng.sample(data, k)]
    assign = [0] * len(data)
    for _ in range(rounds):
        assign = [max(range(k), key=lambda j: dot(v, cents[j])) for v in data]
        for j in range(k):
            members = [data[i] for i, a in enumerate(assign) if a == j]
            if members:
                cents[j] = [sum(col) / len(members) for col in zip(*members)]
    return cents, assign


def part3() -> None:
    print("\n=== 4부. 정확한 최근접 이웃(전부 비교) vs 근사 최근접 이웃(IVF) ===")
    rng = random.Random(7)
    n, dim, lists, k = 4_000, 16, 40, 10
    data = make_vectors(rng, n, dim, clusters=25)
    queries = make_vectors(random.Random(8), 40, dim, clusters=25)
    t0 = time.perf_counter()
    cents, assign = kmeans(data, lists, rng)
    build = time.perf_counter() - t0
    buckets: dict[int, list[int]] = {j: [] for j in range(lists)}
    for i, a in enumerate(assign):
        buckets[a].append(i)
    print(f"벡터 {n:,}개, {dim}차원. k-means로 묶음 {lists}개를 미리 만든다 (색인 만들기 {build:.1f}초, 이 컴퓨터 기준 참고값)")

    exact = [set(sorted(range(n), key=lambda i: -dot(q, data[i]))[:k]) for q in queries]
    print(f"\n{'nprobe':>7}{'비교한 벡터 수':>14}{'전체 대비':>10}{'Recall@10':>11}")
    print(f"{'전부':>7}{n:>14,}{1:>10.0%}{1:>11.3f}")
    results = {}
    for nprobe in (1, 2, 4, 8, 16):
        compared, recall = 0, 0.0
        for q, truth in zip(queries, exact):
            near = sorted(range(lists), key=lambda j: -dot(q, cents[j]))[:nprobe]
            cand = [i for j in near for i in buckets[j]]
            compared += len(cand) + lists
            got = set(sorted(cand, key=lambda i: -dot(q, data[i]))[:k])
            recall += len(got & truth) / k
        results[nprobe] = recall / len(queries)
        print(f"{nprobe:>7}{compared / len(queries):>14,.0f}{compared / len(queries) / n:>10.0%}{results[nprobe]:>11.3f}")
    assert results[1] <= results[4] <= results[16]
    print("묶음을 적게 열면 빠르지만 진짜 이웃을 놓친다. 많이 열수록 정확해지고 느려진다.")


if __name__ == "__main__":
    part1()
    part2()
    part3()
    print("\n모든 손 계산 검사를 통과했다.")
