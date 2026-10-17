"""종합 실습(G장): 라이브러리 없이 역색인 + BM25 검색기와 평가기를 직접 만든다.

  .venv/bin/python labs/basics/capstone.py              # 내가 채운 TODO를 검사한다
  .venv/bin/python labs/basics/capstone.py --solution   # 참고 답안으로 검사한다 (먼저 직접 해 보자)

TODO 네 개를 위에서부터 채운다. 검사는 앞 단계가 통과해야 다음으로 넘어간다.
표준 라이브러리만 쓴다. 기대값은 저장소의 bm25.py와 labs/score_fusion.py로 잰 값이다.
(score_fusion의 MRR은 15등까지 다 보고, 여기는 5등까지만 본다. en은 그래서 0.843과 0.839로 다르다.)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
K1, B = 1.5, 0.75


def tokenize(text: str) -> list[str]:
    # TODO 1: 소문자로 바꾸고 \w+ 로 단어를 찾은 뒤, 단어마다 두 글자씩 겹쳐 자른다.
    #         한 글자 단어는 그대로 둔다. 예) "방화벽은 차단" → ["방화", "화벽", "벽은", "차단"]
    raise NotImplementedError("TODO 1: tokenize")


def build_index(docs: dict[str, str]) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    # TODO 2: 역색인을 만든다.
    #   postings[토큰][파일 이름] = 그 글에서 토큰이 나온 횟수(tf)
    #   lengths[파일 이름] = 그 글의 토큰 수
    raise NotImplementedError("TODO 2: build_index")


def search(query: str, postings: dict[str, dict[str, int]], lengths: dict[str, int], k: int = 5) -> list[tuple[str, float]]:
    # TODO 3: BM25로 점수를 매겨 상위 k편을 (파일 이름, 점수)로 돌려준다.
    #   - 글 전체를 훑지 말고 질문 토큰의 포스팅 목록만 읽는다.
    #   - IDF = ln(1 + (N - df + 0.5) / (df + 0.5)),  df = 그 토큰이 있는 글 수
    #   - 질문에 같은 토큰이 두 번 나오면 두 번 더한다 (bm25.py와 같게).
    raise NotImplementedError("TODO 3: search")


def evaluate(results: list[list[str]], expected: list[str]) -> dict[str, float]:
    # TODO 4: 질문마다 검색 결과(파일 이름 목록, 위에서부터)와 기대한 파일이 주어진다.
    #   {"hit@1": ..., "recall@3": ..., "mrr@5": ...}를 0~1 사이 값으로 돌려준다.
    raise NotImplementedError("TODO 4: evaluate")


def load(lang: str) -> tuple[dict[str, str], list[dict]]:
    docs = {}
    for path in sorted((ROOT / "docs" / lang).glob("*.md")):
        docs[path.name] = path.read_text(encoding="utf-8").split("---", 2)[2].strip().replace("# ", "", 1)
    items = json.loads((ROOT / "data" / "answer_key.json").read_text(encoding="utf-8"))["items"]
    return docs, items


def check(impl) -> None:
    print("[1] tokenize")
    got = impl.tokenize("방화벽은 차단한다. VPN!")
    want = ["방화", "화벽", "벽은", "차단", "단한", "한다", "vp", "pn"]
    assert got == want, f"기대 {want}, 결과 {got}"
    assert impl.tokenize("a 가") == ["a", "가"]
    print("    통과")

    print("[2] build_index")
    docs, items = load("ko")
    postings, lengths = impl.build_index(docs)
    assert postings["방화"] == {"03-firewalls.md": 3}, postings.get("방화")
    assert len(lengths) == 15 and sum(lengths.values()) == sum(len(impl.tokenize(t)) for t in docs.values())
    print(f"    통과: 서로 다른 토큰 {len(postings)}개, 글 길이 합 {sum(lengths.values())}")

    print("[3] search")
    hits = impl.search("허가되지 않은 네트워크 통신을 무엇이 막나?", postings, lengths, k=3)
    print(f"    {[(f, round(s, 2)) for f, s in hits]}")
    assert hits[0][0] == "03-firewalls.md" and abs(hits[0][1] - 13.28) < 0.01
    assert hits[1][0] == "15-vpns.md" and abs(hits[1][1] - 9.19) < 0.01
    print("    통과: bm25.py와 같은 점수")

    print("[4] evaluate")
    toy = impl.evaluate([["a", "b"], ["b", "a", "c"], ["c", "d", "e", "f", "g"]], ["a", "a", "z"])
    assert abs(toy["hit@1"] - 1 / 3) < 1e-9 and abs(toy["recall@3"] - 2 / 3) < 1e-9 and abs(toy["mrr@5"] - 0.5) < 1e-9, toy
    expected = {"ko": (14, 0.967), "en": (12, 0.839)}
    for lang in ("ko", "en"):
        docs, items = load(lang)
        postings, lengths = impl.build_index(docs)
        results = [[f for f, _ in impl.search(it[lang], postings, lengths, k=5)] for it in items]
        m = impl.evaluate(results, [it["file"] for it in items])
        print(f"    {lang}: Hit@1 {m['hit@1'] * 15:.0f}/15, Recall@3 {m['recall@3']:.3f}, MRR@5 {m['mrr@5']:.3f}")
        hit, mrr = expected[lang]
        assert round(m["hit@1"] * 15) == hit and abs(m["mrr@5"] - mrr) < 0.001
        for it, res in zip(items, results):
            if res[0] != it["file"]:
                print(f"      틀린 질문: {it[lang]}  기대 {it['file']}  결과 {res[:3]}")
    print("    통과")
    print("\n다 통과했다. 이제 guide/G-capstone.md의 2부(분석 질문)로 넘어가자.")


if __name__ == "__main__":
    if "--solution" in sys.argv:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "solutions"))
        import capstone_solution as impl
    else:
        impl = sys.modules[__name__]
    try:
        check(impl)
    except NotImplementedError as todo:
        sys.exit(f"아직 채우지 않았다: {todo}")
