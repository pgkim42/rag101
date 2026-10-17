"""A장 실습: 순차 검색과 역색인(inverted index)이 「몇 편을 들여다보는지」 비교한다.

  .venv/bin/python labs/basics/inverted_index.py

표준 라이브러리만 쓴다. 1부는 실제 15편, 2부는 무작위로 만든 가짜 코퍼스로 크기를 키운다.
"""

from __future__ import annotations

import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def bigrams(text: str) -> list[str]:
    out = []
    for word in re.findall(r"\w+", text.lower()):
        out.extend(word[i : i + 2] for i in range(max(1, len(word) - 1)))
    return out


def build_index(docs: list[list[str]]) -> dict[str, list[tuple[int, int]]]:
    # 토큰 → [(글 번호, 그 글에서 나온 횟수), ...]. 이 목록을 포스팅 목록(postings)이라 부른다.
    index: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for doc_id, tokens in enumerate(docs):
        for token, tf in Counter(tokens).items():
            index[token].append((doc_id, tf))
    return index


def scan(docs: list[list[str]], query: list[str]) -> tuple[set[int], int]:
    # 순차 검색: 글을 처음부터 끝까지 하나씩 열어 질문 토큰이 있는지 본다.
    found, opened = set(), 0
    wanted = set(query)
    for doc_id, tokens in enumerate(docs):
        opened += 1
        if wanted & set(tokens):
            found.add(doc_id)
    return found, opened


def lookup(index: dict[str, list[tuple[int, int]]], query: list[str]) -> tuple[set[int], int]:
    # 역색인: 질문 토큰의 포스팅 목록만 꺼내 합친다. 질문 토큰이 없는 글은 아예 보지 않는다.
    found, touched = set(), 0
    for token in set(query):
        postings = index.get(token, [])
        touched += len(postings)
        found.update(doc_id for doc_id, _ in postings)
    return found, touched


def part1() -> None:
    print("=== 1부. 실제 한국어 15편 ===")
    paths = sorted((ROOT / "docs" / "ko").glob("*.md"))
    docs = [bigrams(p.read_text(encoding="utf-8").split("---", 2)[2]) for p in paths]
    index = build_index(docs)
    print(f"글 {len(docs)}편, 서로 다른 토큰 {len(index)}개, 포스팅 {sum(len(v) for v in index.values())}개")
    for token in ("방화", "통신", "는다"):
        names = [paths[d].name[:2] + f"(tf={tf})" for d, tf in index[token]]
        print(f"  {token!r:6} → {len(names):>2}편: {', '.join(names)}")
    query = bigrams("허가되지 않은 네트워크 통신을 무엇이 막나?")
    a, opened = scan(docs, query)
    b, touched = lookup(index, query)
    assert a == b, "두 방법이 찾은 글이 같아야 한다"
    print(f"질문 토큰 {len(set(query))}개. 순차 검색은 글 {opened}편을 열었고, 역색인은 포스팅 {touched}개를 읽었다.")
    print(f"질문 토큰이 하나라도 있는 글: {len(a)}편 — 이것은 「후보」일 뿐 순위가 아니다")


def fake_corpus(n_docs: int, vocab: int, length: int, rng: random.Random) -> list[list[str]]:
    # 실제 언어처럼 몇몇 단어는 아주 흔하고 대부분은 드물게(지프 법칙) 나오게 만든다.
    weights = [1 / (rank + 1) for rank in range(vocab)]
    cum, total = [], 0.0
    for w in weights:
        total += w
        cum.append(total)
    words = [f"w{i}" for i in range(vocab)]
    return [rng.choices(words, cum_weights=cum, k=length) for _ in range(n_docs)]


def part2() -> None:
    print("\n=== 2부. 가짜 코퍼스로 크기를 키우면 (글당 30토큰, 어휘 20,000개) ===")
    print("「연 글」과 「읽은 포스팅」은 몇 번 돌려도 같은 값이다. 시간은 이 컴퓨터에서 잰 참고값이다.")
    print(f"{'글 수':>9}{'순차: 연 글':>12}{'역색인: 읽은 포스팅':>18}{'순차 시간':>11}{'역색인 시간':>12}")
    rng = random.Random(0)
    queries = [[f"w{rng.randint(50, 5000)}", f"w{rng.randint(50, 5000)}"] for _ in range(50)]
    for n in (150, 1_500, 15_000, 150_000):
        docs = fake_corpus(n, 20_000, 30, random.Random(n))
        index = build_index(docs)
        t0 = time.perf_counter()
        opened = sum(scan(docs, q)[1] for q in queries) / len(queries)
        t1 = time.perf_counter()
        touched = sum(lookup(index, q)[1] for q in queries) / len(queries)
        t2 = time.perf_counter()
        for q in queries[:5]:
            assert scan(docs, q)[0] == lookup(index, q)[0]
        print(f"{n:>9,}{opened:>12,.0f}{touched:>18,.1f}{(t1 - t0) / len(queries) * 1000:>9.2f}ms{(t2 - t1) / len(queries) * 1000:>10.3f}ms")

    common = [["w0", "w1"]]
    docs = fake_corpus(15_000, 20_000, 30, random.Random(1))
    index = build_index(docs)
    print(f"\n가장 흔한 두 단어(w0, w1)로 물으면 15,000편 중 포스팅 {lookup(index, common[0])[1]:,}개를 읽는다.")
    print("흔한 단어는 색인으로도 비싸고, 정보도 적다. B장의 IDF가 흔한 단어를 깎는 이유와 같은 뿌리다.")


if __name__ == "__main__":
    part1()
    part2()
