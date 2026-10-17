"""단어가 얼마나 겹치는지로 글에 점수를 매기는 BM25. 라이브러리 없이 공식을 그대로 옮겼다."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Final

import numpy as np

# k1은 같은 조각이 여러 번 나올 때 점수가 얼마나 더 오르는지, b는 긴 글을 얼마나 깎는지 정한다.
# 둘 다 널리 쓰는 기본값이다.
K1: Final = 1.5
B: Final = 0.75


def tokenize(text: str) -> list[str]:
    # 한국어는 조사가 붙어 「방화벽은」과 「방화벽」이 다른 단어가 된다.
    # 단어를 두 글자씩 겹쳐 잘라 「방화」「화벽」처럼 조사와 상관없는 조각을 센다.
    tokens: list[str] = []
    for word in re.findall(r"\w+", text.lower()):
        tokens.extend(word[i : i + 2] for i in range(max(1, len(word) - 1)))
    return tokens


class BM25:
    """글 묶음을 한 번 읽어 조각 빈도를 세 두고, 질문마다 글별 점수를 낸다."""

    def __init__(self, texts: list[str]) -> None:
        self.docs = [Counter(tokenize(text)) for text in texts]
        self.lengths = [sum(doc.values()) for doc in self.docs]
        self.avg_length = sum(self.lengths) / len(self.docs)
        n = len(self.docs)
        doc_freq = Counter(term for doc in self.docs for term in doc)
        # 드문 조각일수록 크다. 모든 글에 있는 조각은 0에 가깝다.
        self.idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}

    def scores(self, query: str) -> np.ndarray:
        terms = tokenize(query)
        out = np.zeros(len(self.docs))
        for i, (doc, length) in enumerate(zip(self.docs, self.lengths, strict=True)):
            saturation = K1 * (1 - B + B * length / self.avg_length)
            out[i] = sum(
                self.idf[term] * doc[term] * (K1 + 1) / (doc[term] + saturation)
                for term in terms
                if term in doc
            )
        return out
