"""종합 실습 참고 답안. 직접 해 본 다음에 열자. 정답이 이것 하나만 있는 것은 아니다."""

from __future__ import annotations

import math
import re
from collections import Counter

K1, B = 1.5, 0.75


def tokenize(text: str) -> list[str]:
    tokens = []
    for word in re.findall(r"\w+", text.lower()):
        tokens.extend(word[i : i + 2] for i in range(max(1, len(word) - 1)))
    return tokens


def build_index(docs: dict[str, str]) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    postings: dict[str, dict[str, int]] = {}
    lengths: dict[str, int] = {}
    for name, text in docs.items():
        tokens = tokenize(text)
        lengths[name] = len(tokens)
        for token, tf in Counter(tokens).items():
            postings.setdefault(token, {})[name] = tf
    return postings, lengths


def search(query: str, postings: dict[str, dict[str, int]], lengths: dict[str, int], k: int = 5) -> list[tuple[str, float]]:
    n = len(lengths)
    avg = sum(lengths.values()) / n
    scores: dict[str, float] = {}
    for token in tokenize(query):
        docs = postings.get(token)
        if not docs:
            continue
        idf = math.log(1 + (n - len(docs) + 0.5) / (len(docs) + 0.5))
        for name, tf in docs.items():
            norm = K1 * (1 - B + B * lengths[name] / avg)
            scores[name] = scores.get(name, 0.0) + idf * tf * (K1 + 1) / (tf + norm)
    return sorted(scores.items(), key=lambda kv: -kv[1])[:k]


def evaluate(results: list[list[str]], expected: list[str]) -> dict[str, float]:
    n = len(expected)
    hit1 = recall3 = mrr = 0.0
    for ranking, gold in zip(results, expected):
        top = ranking[:5]
        if gold in top:
            rank = top.index(gold) + 1
            hit1 += rank == 1
            recall3 += rank <= 3
            mrr += 1 / rank
    return {"hit@1": hit1 / n, "recall@3": recall3 / n, "mrr@5": mrr / n}
