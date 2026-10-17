"""B장 실습: 문자열 일치에서 BM25까지, 한 단계씩 손 계산과 맞춰 본다.

  .venv/bin/python labs/basics/lexical.py           # 1~6부: 장난감 글 3편으로 손 계산
  .venv/bin/python labs/basics/lexical.py --real    # 7부까지: 실제 15편과 정답표로 토큰화 비교

표준 라이브러리만 쓴다. 숫자가 B장 본문의 손 계산과 다르면 assert가 멈춘다.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 장난감 코퍼스. 일부러 D2가 「통신」을 여러 번 말하게 만들었다.
DOCS = {
    "D1": "방화벽은 규칙에 따라 통신을 차단한다. 방화벽은 내부 네트워크를 지킨다.",
    "D2": "VPN은 통신을 암호화한다. VPN은 통신 내용과 통신 경로를 숨긴다. 공공 망에서 통신을 지킨다.",
    "D3": "광합성은 햇빛으로 포도당을 만든다.",
}
QUERY = "통신을 차단하는 방화벽"

# 아주 거친 「조사 떼기」. 진짜 형태소 분석기가 아니다. 긴 것부터 검사한다.
PARTICLES = ("에서", "으로", "은", "는", "이", "가", "을", "를", "에", "로", "과", "와", "의")


def words(text: str) -> list[str]:
    """띄어쓰기와 문장 부호로만 자른다."""
    return re.findall(r"\w+", text.lower())


def strip_particles(text: str) -> list[str]:
    out = []
    for word in words(text):
        for p in PARTICLES:
            if word.endswith(p) and len(word) > len(p) + 1:
                word = word[: -len(p)]
                break
        out.append(word)
    return out


def bigrams(text: str) -> list[str]:
    """bm25.py의 tokenize()와 같은 규칙: 단어를 두 글자씩 겹쳐 자른다."""
    out = []
    for word in words(text):
        out.extend(word[i : i + 2] for i in range(max(1, len(word) - 1)))
    return out


def idf_plain(n_docs: int, df: int) -> float:
    """교과서 TF-IDF의 IDF: ln(N / df)."""
    return math.log(n_docs / df)


def idf_bm25(n_docs: int, df: int) -> float:
    """BM25의 IDF. bm25.py와 같다. 모든 글에 있어도 0보다 조금 크다."""
    return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))


def bm25_term(tf: int, length: int, avg_length: float, idf: float, k1: float = 1.5, b: float = 0.75) -> float:
    norm = k1 * (1 - b + b * length / avg_length)
    return idf * tf * (k1 + 1) / (tf + norm)


def show(title: str) -> None:
    print(f"\n=== {title} ===")


def close(a: float, b: float) -> bool:
    return abs(a - b) < 5e-4


def toy() -> None:
    show("1부. 문자열 일치")
    for name, text in DOCS.items():
        print(f"{name}: 질문 전체가 글에 그대로 있나? {QUERY in text}")
    assert not any(QUERY in t for t in DOCS.values())

    show("2부. 띄어쓰기로 자르기")
    q = words(QUERY)
    print("질문 단어:", q)
    for name, text in DOCS.items():
        shared = sorted(set(q) & set(words(text)))
        print(f"{name}: 겹치는 단어 {shared}")
    # 「방화벽」과 「방화벽은」은 다른 문자열이라 D1과 질문의 「방화벽」이 안 겹친다.
    assert set(q) & set(words(DOCS["D1"])) == {"통신을"}

    show("3부. 조사 떼기 + 단어 빈도(TF)")
    tokens = {name: strip_particles(text) for name, text in DOCS.items()}
    q = strip_particles(QUERY)
    print("질문 토큰:", q)
    for name, toks in tokens.items():
        print(f"{name} ({len(toks)}개): {toks}")
    tf = {name: Counter(toks) for name, toks in tokens.items()}
    raw = {name: sum(tf[name][t] for t in q) for name in DOCS}
    print("질문 토큰의 TF 합:", raw)
    assert raw == {"D1": 3, "D2": 4, "D3": 0}, raw  # D2가 이긴다. 「통신」을 많이 말했을 뿐인데.

    show("4부. 문서 빈도(DF)와 역문서 빈도(IDF)")
    n = len(DOCS)
    df = Counter(t for toks in tokens.values() for t in set(toks))
    print(f"{'토큰':8}{'DF':>4}{'ln(N/df)':>10}{'BM25 IDF':>10}")
    for t in q:
        if df[t]:
            print(f"{t:8}{df[t]:>4}{idf_plain(n, df[t]):>10.4f}{idf_bm25(n, df[t]):>10.4f}")
        else:
            print(f"{t:8}{0:>4}{'(없음)':>10}{'(없음)':>10}")
    assert close(idf_plain(3, 1), 1.0986) and close(idf_plain(3, 2), 0.4055)
    assert close(idf_bm25(3, 1), 0.9808) and close(idf_bm25(3, 2), 0.4700)

    show("5부. TF-IDF = TF × IDF")
    tfidf = {}
    for name in DOCS:
        parts = [(t, tf[name][t], idf_plain(n, df[t])) for t in q if tf[name][t]]
        tfidf[name] = sum(f * i for _, f, i in parts)
        detail = " + ".join(f"{f}×{i:.4f}({t})" for t, f, i in parts) or "0"
        print(f"{name}: {detail} = {tfidf[name]:.4f}")
    assert close(tfidf["D1"], 2.6027) and close(tfidf["D2"], 1.6219)

    show("6부. BM25 (k1=1.5, b=0.75)")
    lengths = {name: len(toks) for name, toks in tokens.items()}
    avg = sum(lengths.values()) / n
    print(f"글 길이 {lengths}, 평균 {avg:.4f}")
    bm = {}
    for name in DOCS:
        total = 0.0
        for t in q:
            if not tf[name][t]:
                continue
            s = bm25_term(tf[name][t], lengths[name], avg, idf_bm25(n, df[t]))
            print(f"  {name} {t}: tf={tf[name][t]}, |d|={lengths[name]} → {s:.4f}")
            total += s
        bm[name] = total
        print(f"{name} 합계: {total:.4f}")
    assert close(bm["D1"], 1.8460) and close(bm["D2"], 0.7753), bm

    print("\n포화(k1): 평균 길이 글에서 tf가 늘 때 IDF에 곱하는 값")
    for f in (1, 2, 4, 10, 100):
        print(f"  tf={f:>3}: TF-IDF는 ×{f:<4}  BM25는 ×{bm25_term(f, 1, 1, 1.0):.3f}")
    assert close(bm25_term(1, 1, 1, 1.0), 1.0) and close(bm25_term(10, 1, 1, 1.0), 2.1739)

    print("\n길이 보정(b): tf=2인 같은 토큰, 글 길이만 다를 때 (IDF=1)")
    for ratio in (0.5, 1, 2, 4):
        cells = "  ".join(f"b={b}: {bm25_term(2, ratio, 1, 1.0, b=b):.3f}" for b in (0.0, 0.75, 1.0))
        print(f"  평균의 {ratio}배 길이 → {cells}")

    show("6부 덧붙임. 두 글자 조각이면 「차단하는」과 「차단한다」가 만난다")
    qb = bigrams(QUERY)
    print("질문 두 글자 조각:", qb)
    for name, text in DOCS.items():
        print(f"{name}: 겹치는 조각 {sorted(set(qb) & set(bigrams(text)))}")
    assert "차단" in set(qb) & set(bigrams(DOCS["D1"]))


def load_corpus(lang: str) -> list[tuple[str, str]]:
    docs = []
    for path in sorted((ROOT / "docs" / lang).glob("*.md")):
        raw = path.read_text(encoding="utf-8").split("---", 2)[2].strip()
        docs.append((path.name, raw.replace("# ", "", 1)))
    return docs


def bm25_scores(docs: list[tuple[str, str]], tokenize, query: str, k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
    tfs = [Counter(tokenize(text)) for _, text in docs]
    lengths = [sum(c.values()) for c in tfs]
    avg = sum(lengths) / len(docs)
    df = Counter(t for c in tfs for t in c)
    return {
        name: sum(bm25_term(c[t], length, avg, idf_bm25(len(docs), df[t]), k1, b) for t in tokenize(query) if c[t])
        for (name, _), c, length in zip(docs, tfs, lengths)
    }


def bm25_rank(docs: list[tuple[str, str]], tokenize, query: str, k1: float = 1.5, b: float = 0.75) -> list[str]:
    scores = bm25_scores(docs, tokenize, query, k1, b)
    return sorted(scores, key=lambda name: -scores[name])


def real() -> None:
    items = json.loads((ROOT / "data" / "answer_key.json").read_text(encoding="utf-8"))["items"]
    tokenizers = {"띄어쓰기": words, "조사 떼기": strip_particles, "두 글자 조각": bigrams}
    show("7부. 실제 15편 + 정답표: BM25 혼자, 토큰화만 바꿔서 (Hit@1)")
    print(f"{'토큰화':10}{'ko':>6}{'en':>6}")
    result = {}
    for tname, tok in tokenizers.items():
        row = []
        for lang in ("ko", "en"):
            docs = load_corpus(lang)
            hit = sum(bm25_rank(docs, tok, it[lang])[0] == it["file"] for it in items)
            row.append(hit)
        result[tname] = row
        print(f"{tname:10}{row[0]:>4}/15{row[1]:>4}/15")

    show("7부 덧붙임. 한국어 두 글자 조각에서 k1과 b를 바꾸면 (Hit@1)")
    docs = load_corpus("ko")
    for k1 in (0.5, 1.5, 3.0):
        cells = "  ".join(
            f"b={b}: {sum(bm25_rank(docs, bigrams, it['ko'], k1, b)[0] == it['file'] for it in items):>2}/15"
            for b in (0.0, 0.75, 1.0)
        )
        print(f"  k1={k1}: {cells}")

    try:  # 저장소의 bm25.py와 같은 점수를 내는지 맞춰 본다. numpy가 있는 .venv에서만.
        sys.path.insert(0, str(ROOT))
        from bm25 import BM25
    except ImportError:
        print("\n(numpy가 없어 bm25.py와의 비교는 건너뛴다. .venv/bin/python으로 돌리면 비교한다)")
        return
    docs = load_corpus("ko")
    repo = BM25([text for _, text in docs])
    worst = 0.0
    for it in items:
        mine = bm25_scores(docs, bigrams, it["ko"])
        theirs = repo.scores(it["ko"])
        worst = max(worst, max(abs(mine[name] - float(theirs[i])) for i, (name, _) in enumerate(docs)))
    print(f"\n이 파일의 BM25와 bm25.py의 점수 차이(15문항×15편 중 최대): {worst:.2e}")
    assert worst < 1e-9


if __name__ == "__main__":
    toy()
    if "--real" in sys.argv:
        real()
    print("\n모든 손 계산 검사를 통과했다.")
