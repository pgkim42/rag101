"""C장 실습: 벡터의 크기와 방향, 내적과 코사인, 정규화, 평균 풀링을 작은 숫자로 확인한다.

  .venv/bin/python labs/basics/vectors.py

표준 라이브러리만 쓴다. 4~5부의 3차원 「뜻 벡터」는 설명용으로 사람이 지어낸 값이다.
실제 모델의 값은 labs/embedding_steps.py에서 본다.
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lexical import DOCS, QUERY, strip_particles  # noqa: E402  B장과 같은 장난감 글을 쓴다


def dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def norm(a) -> float:
    return math.sqrt(dot(a, a))


def cosine(a, b) -> float:
    return dot(a, b) / (norm(a) * norm(b))


def unit(a) -> list[float]:
    n = norm(a)
    return [x / n for x in a]


def mean(vectors) -> list[float]:
    return [sum(col) / len(vectors) for col in zip(*vectors, strict=True)]


def close(a: float, b: float) -> bool:
    return abs(a - b) < 5e-4


def part1() -> None:
    print("=== 1부. 내적은 길이에 끌려가고, 코사인은 방향만 본다 ===")
    q, a, b = (1.0, 1.0), (1.0, 0.9), (5.0, 1.0)
    for name, v in (("A", a), ("B", b)):
        print(f"{name}={v}: 길이 {norm(v):.4f}, q·{name} = {dot(q, v):.4f}, cos(q,{name}) = {cosine(q, v):.4f}")
    assert dot(q, b) > dot(q, a) and cosine(q, a) > cosine(q, b)
    assert close(cosine(q, a), 0.9986) and close(cosine(q, b), 0.8321)
    print("내적으로 줄 세우면 B가 1등, 코사인으로 줄 세우면 A가 1등이다.")

    print("\n같은 방향, 다른 길이: A와 3A")
    a3 = tuple(3 * x for x in a)
    print(f"q·A = {dot(q, a):.4f}, q·3A = {dot(q, a3):.4f} / cos(q,A) = {cosine(q, a):.4f}, cos(q,3A) = {cosine(q, a3):.4f}")
    assert close(cosine(q, a), cosine(q, a3))

    print("\n길이를 1로 맞추면(정규화) 내적이 곧 코사인이다")
    qu, au, bu = unit(q), unit(a), unit(b)
    print(f"단위 q = ({qu[0]:.4f}, {qu[1]:.4f}), 길이 {norm(qu):.4f}")
    print(f"단위 q · 단위 A = {dot(qu, au):.4f} (= cos {cosine(q, a):.4f}),  단위 q · 단위 B = {dot(qu, bu):.4f}")
    assert close(dot(qu, au), cosine(q, a)) and close(dot(qu, bu), cosine(q, b))


def part2() -> None:
    print("\n=== 2부. 글도 벡터다: TF 벡터(희소 벡터)와 코사인 ===")
    tokens = {name: strip_particles(text) for name, text in DOCS.items()}
    vocab = sorted({t for toks in tokens.values() for t in toks} | set(strip_particles(QUERY)))
    print(f"차원 = 어휘 수 = {len(vocab)}. 칸 하나가 토큰 하나다.")
    vec = {name: [Counter(toks)[t] for t in vocab] for name, toks in tokens.items()}
    qv = [Counter(strip_particles(QUERY))[t] for t in vocab]
    nonzero = sum(x != 0 for x in vec["D1"])
    print(f"D1 벡터에서 0이 아닌 칸: {nonzero}/{len(vocab)} — 대부분이 0이라 「희소(sparse)」 벡터라 부른다")
    for name, v in vec.items():
        c = cosine(qv, v) if norm(v) else 0.0
        print(f"cos(질문, {name}) = {c:.4f}")
    assert cosine(qv, vec["D1"]) > cosine(qv, vec["D2"]) > 0

    print("\n같은 뜻, 다른 글자: 「햇빛을 전기로」와 「태양광 발전」")
    a, b = strip_particles("햇빛을 전기로 바꾼다"), strip_particles("태양광 발전")
    print(f"토큰 {a} / {b} → 겹치는 칸이 없으니 TF 벡터의 코사인은 0이다")
    assert not set(a) & set(b)


# 설명용으로 지어낸 3차원 토큰 벡터. 축은 (통신·보안, 빛·에너지, 돈·경제)라고 상상한다.
TOY = {
    "햇빛": (0.0, 1.0, 0.0),
    "전기": (0.3, 0.8, 0.1),
    "태양광": (0.1, 0.9, 0.1),
    "발전": (0.2, 0.7, 0.3),
    "통신": (1.0, 0.0, 0.1),
    "차단": (0.8, 0.1, 0.0),
    "않음": (0.2, 0.2, 0.2),
    "물가": (0.0, 0.1, 1.0),
}


def sentence(words: list[str]) -> list[float]:
    return mean([TOY[w] for w in words])


def part3() -> None:
    print("\n=== 3부. (지어낸) 뜻 벡터의 평균: 글자가 안 겹쳐도 가깝다 ===")
    a, b, c = sentence(["햇빛", "전기"]), sentence(["태양광", "발전"]), sentence(["통신", "차단"])
    print(f"평균(햇빛, 전기) = {[round(x, 3) for x in a]}")
    print(f"평균(태양광, 발전) = {[round(x, 3) for x in b]}")
    print(f"cos(햇빛 전기, 태양광 발전) = {cosine(a, b):.4f}")
    print(f"cos(햇빛 전기, 통신 차단)   = {cosine(a, c):.4f}")
    assert cosine(a, b) > 0.95 > 0.5 > cosine(a, c)

    print("\n=== 4부. 평균의 약점: 부정어 하나는 평균에 묻힌다 ===")
    yes = sentence(["통신", "차단"])
    no = sentence(["통신", "차단", "않음"])
    print(f"cos(통신 차단, 통신 차단 않음) = {cosine(yes, no):.4f}  ← 뜻은 반대인데 거의 같다")
    assert cosine(yes, no) > 0.95


def part4() -> None:
    print("\n=== 5부. 384차원 벡터를 저장하면 (float32 = 4바이트) ===")
    per = 384 * 4
    for n in (15, 10_000, 1_000_000):
        print(f"  {n:>9,}편 × 384 × 4바이트 = {n * per:>13,}바이트 ≈ {n * per / 2**20:,.1f} MiB")
    assert 15 * per == 23_040


if __name__ == "__main__":
    part1()
    part2()
    part3()
    part4()
    print("\n모든 손 계산 검사를 통과했다.")
