"""임베딩 384개를 눈으로 볼 수 있게 줄여서 그린다.

  .venv/bin/python -m labs.embedding_map

한국어 글 15편과 정답표 질문 15개를 2차원 지도로 그리고(guide/img/embedding-map.svg),
글끼리의 코사인 유사도를 표로 그린다(guide/img/similarity-heatmap.svg).
"""

from __future__ import annotations

import json
from html import escape
from typing import Final

import numpy as np

from retrieve import ROOT, build_searcher

OUT_DIR: Final = ROOT / "guide" / "img"
FONT: Final = "Apple SD Gothic Neo, Noto Sans KR, Malgun Gothic, sans-serif"


def pca_2d(docs: np.ndarray, questions: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    # 글 15편이 가장 넓게 퍼지는 두 방향을 찾아 그 위에 그림자를 떨어뜨린다(주성분 분석).
    # 특잇값 분해(SVD)의 오른쪽 벡터 앞 두 개가 그 방향이다.
    center = docs.mean(axis=0)
    _, singular, directions = np.linalg.svd(docs - center, full_matrices=False)
    axes = directions[:2].T
    kept = float((singular[:2] ** 2).sum() / (singular**2).sum())
    return (docs - center) @ axes, (questions - center) @ axes, kept


def to_canvas(points: np.ndarray, size: tuple[int, int], margin: int) -> np.ndarray:
    low, high = points.min(axis=0), points.max(axis=0)
    scale = (np.array(size) - 2 * margin) / (high - low)
    flipped = (points - low) * scale + margin
    flipped[:, 1] = size[1] - flipped[:, 1]
    return flipped


Box = tuple[float, float, float, float]


def text_width(text: str, size: int) -> float:
    return sum(size * (0.95 if ord(ch) > 0x2E80 else 0.6) for ch in text)


def overlaps(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def place_label(x: float, y: float, text: str, size: int, taken: list[Box]) -> tuple[float, float]:
    # 점 둘레 몇 자리를 차례로 대 보고, 이미 그린 글자나 점과 가장 덜 겹치는 자리에 글자를 둔다.
    w = text_width(text, size)
    gap = 9
    candidates = [(x + gap, y - gap), (x + gap, y + size + 2), (x - gap - w, y - gap), (x - gap - w, y + size + 2), (x - w / 2, y - gap - 4), (x - w / 2, y + size + gap)]
    best = min(candidates, key=lambda c: sum(overlaps((c[0], c[1] - size, c[0] + w, c[1] + 2), box) for box in taken))
    taken.append((best[0], best[1] - size, best[0] + w, best[1] + 2))
    return best


def map_svg(titles: list[str], doc_xy: np.ndarray, q_xy: np.ndarray, links: list[tuple[int, bool]]) -> str:
    width, height = 960, 680
    both = to_canvas(np.vstack([doc_xy, q_xy]), (width, height - 50), 110)
    docs, questions = both[: len(titles)], both[len(titles) :]
    taken: list[Box] = [(x - 9, y - 9, x + 9, y + 9) for x, y in both]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="{FONT}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-size="18" font-weight="bold">한국어 글 15편과 질문 15개의 임베딩 지도 (384차원 → 2차원)</text>',
    ]
    for q_index, (doc_index, correct) in enumerate(links):
        (x1, y1), (x2, y2) = questions[q_index], docs[doc_index]
        style = 'stroke="#bbbbbb"' if correct else 'stroke="#d62728" stroke-width="2" stroke-dasharray="6 4"'
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {style}/>')
    for title, (x, y) in zip(titles, docs, strict=True):
        tx, ty = place_label(x, y, title, 14, taken)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="#1f77b4"/>')
        parts.append(f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="14">{escape(title)}</text>')
    for q_index, (x, y) in enumerate(questions):
        color = "#d62728" if not links[q_index][1] else "#555555"
        tx, ty = place_label(x, y, f"Q{q_index + 1}", 11, taken)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="white" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="11" fill="{color}">Q{q_index + 1}</text>')
    legend_y = height - 22
    parts += [
        f'<circle cx="30" cy="{legend_y}" r="8" fill="#1f77b4"/><text x="44" y="{legend_y + 5}" font-size="13">글</text>',
        f'<circle cx="100" cy="{legend_y}" r="5" fill="white" stroke="#555" stroke-width="2"/>',
        f'<text x="112" y="{legend_y + 5}" font-size="13">질문 (선은 기대한 글로 이어짐)</text>',
        f'<line x1="330" y1="{legend_y}" x2="370" y2="{legend_y}" stroke="#d62728" stroke-width="2" stroke-dasharray="6 4"/>',
        f'<text x="378" y="{legend_y + 5}" font-size="13" fill="#d62728">임베딩만 쓰면(dense) 1등을 놓친 질문</text>',
        "</svg>",
    ]
    return "\n".join(parts)


def heatmap_svg(titles: list[str], sims: np.ndarray) -> str:
    cell, left, top = 40, 170, 70
    n = len(titles)
    off_diagonal = sims[~np.eye(n, dtype=bool)]
    low, high = off_diagonal.min(), off_diagonal.max()
    width, height = left + n * cell + 20, top + n * cell + 50
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="{FONT}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-size="18" font-weight="bold">한국어 글끼리의 코사인 유사도</text>',
        f'<text x="20" y="50" font-size="12" fill="#555">진할수록 비슷하다. 대각선은 자기 자신(1.00)이라 회색이다. 색 범위 {low:.2f} ~ {high:.2f}</text>',
    ]
    for i in range(n):
        y = top + i * cell
        parts.append(f'<text x="{left - 8}" y="{y + cell / 2 + 4}" font-size="12" text-anchor="end">{i + 1}. {escape(titles[i])}</text>')
        parts.append(f'<text x="{left + i * cell + cell / 2}" y="{top - 8}" font-size="12" text-anchor="middle">{i + 1}</text>')
        for j in range(n):
            x = left + j * cell
            if i == j:
                fill, ink = "#dddddd", "#777777"
            else:
                strength = (sims[i, j] - low) / (high - low)
                shade = int(255 - strength * 200)
                fill, ink = f"rgb({shade},{shade + int(strength * 30)},255)", "white" if strength > 0.6 else "#222222"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="white"/>')
            parts.append(f'<text x="{x + cell / 2}" y="{y + cell / 2 + 4}" font-size="10" text-anchor="middle" fill="{ink}">{sims[i, j]:.2f}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    searcher = build_searcher("ko", "dense")
    titles = [chunk["title"] for chunk in searcher.chunks]
    files = [chunk["file"] for chunk in searcher.chunks]
    items = json.loads((ROOT / "data" / "answer_key.json").read_text(encoding="utf-8"))["items"]
    questions = searcher.embedder.encode([item["ko"] for item in items], normalize_embeddings=True)

    sims = searcher.doc_vectors @ searcher.doc_vectors.T
    q_scores = questions @ searcher.doc_vectors.T
    links = [(files.index(item["file"]), files[int(np.argmax(row))] == item["file"]) for item, row in zip(items, q_scores, strict=True)]
    doc_xy, q_xy, kept = pca_2d(searcher.doc_vectors, questions)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "embedding-map.svg").write_text(map_svg(titles, doc_xy, q_xy, links), encoding="utf-8")
    (OUT_DIR / "similarity-heatmap.svg").write_text(heatmap_svg(titles, sims), encoding="utf-8")

    print(f"\n2차원에 남은 정보: 전체 퍼짐의 {kept:.1%}")
    print("\n글마다 가장 가까운 다른 글")
    for i, title in enumerate(titles):
        order = [j for j in np.argsort(-sims[i]) if j != i]
        print(f"  {title:12} → {titles[order[0]]} ({sims[i, order[0]]:.3f}), {titles[order[1]]} ({sims[i, order[1]]:.3f})")
    pairs = sorted(((sims[i, j], i, j) for i in range(len(titles)) for j in range(i + 1, len(titles))), reverse=True)
    print("\n가장 비슷한 글 쌍 5개")
    for score, i, j in pairs[:5]:
        print(f"  {score:.3f}  {titles[i]} — {titles[j]}")
    print(f"\n가장 덜 비슷한 쌍: {pairs[-1][0]:.3f}  {titles[pairs[-1][1]]} — {titles[pairs[-1][2]]}")
    print(f"\n그림 저장: {(OUT_DIR / 'embedding-map.svg').relative_to(ROOT)}, {(OUT_DIR / 'similarity-heatmap.svg').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
