"""D·F장 실습: 실제 15편에서 dense와 BM25 점수가 어떻게 생겼는지, 섞는 방법마다 결과가 어떻게 바뀌는지,
그리고 각 단계가 얼마나 걸리는지 잰다.

  .venv/bin/python -m labs.score_fusion
  .venv/bin/python -m labs.score_fusion --no-timing   # 재순위화 모델 없이 (지연 시간 측정 생략)

retrieve.py의 검색 부품(임베딩, bm25.py)을 그대로 쓰고, 섞는 방법만 여기서 바꿔 본다.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from retrieve import ROOT, build_searcher, minmax


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std() if x.std() > 0 else np.zeros_like(x)


def rank_positions(x: np.ndarray) -> np.ndarray:
    pos = np.empty(len(x), dtype=int)
    pos[np.argsort(-x)] = np.arange(1, len(x) + 1)
    return pos


def rrf(*xs: np.ndarray, k: int = 60) -> np.ndarray:
    return sum(1.0 / (k + rank_positions(x)) for x in xs)


def methods(dense: np.ndarray, lexical: np.ndarray) -> dict[str, np.ndarray]:
    out = {"dense만": dense, "BM25만": lexical}
    for w in (0.3, 0.5, 0.7, 0.9):
        out[f"min-max {w}"] = w * minmax(dense) + (1 - w) * minmax(lexical)
    out["z-점수 0.7"] = 0.7 * zscore(dense) + 0.3 * zscore(lexical)
    out["RRF k=60"] = rrf(dense, lexical)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-timing", action="store_true")
    args = parser.parse_args()
    items = json.loads((ROOT / "data" / "answer_key.json").read_text(encoding="utf-8"))["items"]

    for lang in ("ko", "en"):
        s = build_searcher(lang, "dense")
        files = [c["file"] for c in s.chunks]
        q_vecs = s.embedder.encode([it[lang] for it in items], normalize_embeddings=True)
        per_method: dict[str, list[int]] = {}
        top_dense, gap_dense, top_lex = [], [], []
        for it, q in zip(items, q_vecs, strict=True):
            dense = s.doc_vectors @ q
            lexical = s.bm25.scores(it[lang])
            d_sorted, l_sorted = np.sort(dense)[::-1], np.sort(lexical)[::-1]
            top_dense.append(d_sorted[0])
            gap_dense.append(d_sorted[0] - d_sorted[1])
            top_lex.append(l_sorted[0])
            for name, scores in methods(dense, lexical).items():
                per_method.setdefault(name, []).append(int(rank_positions(scores)[files.index(it["file"])]))
            if lang == "ko" and it["id"] == 3:
                print("\nko Q3 「허가되지 않은 네트워크 통신을 무엇이 막나?」 상위 3편")
                print(f"{'글':22}{'dense':>8}{'BM25':>8}{'min-max 0.7':>13}{'RRF':>9}")
                fused, r = methods(dense, lexical)["min-max 0.7"], rrf(dense, lexical)
                for i in np.argsort(-fused)[:3]:
                    print(f"{files[i]:22}{dense[i]:>8.3f}{lexical[i]:>8.2f}{fused[i]:>13.3f}{r[i]:>9.4f}")

        print(f"\n== {lang}: 점수의 모양 (질문 15개)")
        print(f"  dense 1등 점수 {min(top_dense):.3f} ~ {max(top_dense):.3f}, 1등과 2등 차이 평균 {np.mean(gap_dense):.3f}")
        print(f"  BM25 1등 점수 {min(top_lex):.2f} ~ {max(top_lex):.2f}")
        print(f"\n== {lang}: 섞는 방법별 결과")
        print(f"{'방법':14}{'Hit@1':>7}{'MRR':>8}  정답 순위(Q1~Q15)")
        for name, ranks in per_method.items():
            hit = sum(r == 1 for r in ranks)
            mrr = float(np.mean([1 / r for r in ranks]))
            print(f"{name:14}{hit:>4}/15{mrr:>8.3f}  {' '.join(str(r) for r in ranks)}")

    if args.no_timing:
        return
    print("\n== 지연 시간 (ko, 질문 15개 평균, 이 컴퓨터에서 잰 참고값)")
    s = build_searcher("ko", "rerank")
    from retrieve import load_reranker, search

    load_reranker()
    questions = [it["ko"] for it in items]
    for q in questions[:2]:
        search(q, s, 3)
    timings = {}
    t = time.perf_counter()
    for q in questions:
        s.embedder.encode([q], normalize_embeddings=True)
    timings["질문 임베딩"] = time.perf_counter() - t
    q_vecs = s.embedder.encode(questions, normalize_embeddings=True)
    t = time.perf_counter()
    for q in q_vecs:
        np.argsort(-(s.doc_vectors @ q))
    timings["15편 내적 + 정렬"] = time.perf_counter() - t
    t = time.perf_counter()
    for q in questions:
        s.bm25.scores(q)
    timings["BM25 점수"] = time.perf_counter() - t
    reranker = load_reranker()
    for pool in (5, 15):
        t = time.perf_counter()
        for q in questions:
            reranker.predict([(q, c["text"]) for c in s.chunks[:pool]])
        timings[f"재순위화 {pool}편"] = time.perf_counter() - t
    for name, total in timings.items():
        print(f"  {name:16} {total / len(questions) * 1000:8.2f} ms")


if __name__ == "__main__":
    main()
