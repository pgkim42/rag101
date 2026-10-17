"""긴 문서를 어떻게 쪼개느냐에 따라 검색이 어떻게 달라지는지 본다.

  .venv/bin/python -m labs.chunking

한국어 글 15편을 한 파일처럼 이어 붙여 긴 문서 하나를 만든다(약 1,600조각).
그 문서를 여러 방식으로 쪼갠 뒤, 정답표 15문항으로 임베딩 검색(dense)을 해 본다.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Final

import numpy as np
from transformers.utils import logging as hf_logging

from retrieve import ROOT, build_searcher

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
MODEL_LIMIT: Final = 128


@dataclasses.dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    file: str


@dataclasses.dataclass(frozen=True, slots=True)
class Chunk:
    start: int
    end: int
    tokens: int


def build_long_document(chunks: list[dict]) -> tuple[str, list[Span]]:
    text, spans = "", []
    for chunk in chunks:
        if text:
            text += "\n\n"
        spans.append(Span(len(text), len(text) + len(chunk["text"]), chunk["file"]))
        text += chunk["text"]
    return text, spans


def by_tokens(text: str, tokenizer, size: int, overlap: int) -> list[Chunk]:
    # 조각 번호마다 원문 몇 번째 글자부터 몇 번째 글자까지인지(offset)를 받아, 조각 size개씩 자른다.
    # 다음 조각은 size − overlap만큼 뒤에서 시작하니 overlap개가 겹친다.
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    out, step = [], size - overlap
    for first in range(0, len(offsets), step):
        last = min(first + size, len(offsets))
        out.append(Chunk(offsets[first][0], offsets[last - 1][1], last - first))
        if last == len(offsets):
            break
    return out


def by_sentences(text: str, tokenizer, budget: int) -> list[Chunk]:
    # 문장 끝(. ? !)이나 줄바꿈에서 자른 뒤, 조각 수가 budget을 넘지 않을 때까지 이어 붙인다.
    sentences = [m for m in re.finditer(r"[^\n.?!]+[.?!]?", text) if m.group().strip()]
    out: list[Chunk] = []
    start, end, used = None, 0, 0
    for match in sentences:
        n = len(tokenizer(match.group(), add_special_tokens=False)["input_ids"])
        if start is not None and used + n > budget:
            out.append(Chunk(start, end, used))
            start, used = None, 0
        start = match.start() if start is None else start
        end, used = match.end(), used + n
    if start is not None:
        out.append(Chunk(start, end, used))
    return out


def sources(chunk: Chunk, spans: list[Span]) -> dict[str, int]:
    shared = {s.file: min(chunk.end, s.end) - max(chunk.start, s.start) for s in spans}
    return {file: n for file, n in shared.items() if n > 0}


def main() -> None:
    searcher = build_searcher("ko", "dense")
    tokenizer, embedder = searcher.embedder.tokenizer, searcher.embedder
    text, spans = build_long_document(searcher.chunks)
    items = json.loads((ROOT / "data" / "answer_key.json").read_text(encoding="utf-8"))["items"]
    questions = embedder.encode([item["ko"] for item in items], normalize_embeddings=True)

    total = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    seen = tokenizer.decode(tokenizer(text, truncation=True, max_length=MODEL_LIMIT)["input_ids"], skip_special_tokens=True)
    print(f"\n긴 문서: {len(text):,}글자, {total:,}조각. 임베딩 모델은 앞의 {MODEL_LIMIT}조각(시작·끝 표시 포함)만 읽는다.")
    print(f"통째로 넣으면 모델이 읽는 마지막 부분: …{seen[-40:]}")

    strategies = {
        "통째로 (1개)": [Chunk(0, len(text), total)],
        "글 단위 (원래 15편)": [Chunk(s.start, s.end, len(tokenizer(text[s.start : s.end], add_special_tokens=False)["input_ids"])) for s in spans],
        "고정 256조각": by_tokens(text, tokenizer, 256, 0),
        "고정 128조각": by_tokens(text, tokenizer, 128, 0),
        "고정 64조각": by_tokens(text, tokenizer, 64, 0),
        "고정 32조각": by_tokens(text, tokenizer, 32, 0),
        "고정 64조각, 16 겹침": by_tokens(text, tokenizer, 64, 16),
        "고정 128조각, 32 겹침": by_tokens(text, tokenizer, 128, 32),
        "문장 묶음 (최대 100조각)": by_sentences(text, tokenizer, 100),
        "문장 묶음 (최대 50조각)": by_sentences(text, tokenizer, 50),
    }

    print(f"\n{'방식':24}{'조각 수':>6}{'평균 길이':>8}{'잘림':>5}{'섞임':>5}{'1등이 정답 글':>12}{'1등에 정답 포함':>14}{'1등 중 정답 비율':>14}")
    for name, chunk_list in strategies.items():
        vectors = embedder.encode([text[c.start : c.end] for c in chunk_list], normalize_embeddings=True)
        top = np.argmax(questions @ vectors.T, axis=1)
        owners = [sources(c, spans) for c in chunk_list]
        hit = contains = 0
        purity = 0.0
        for item, best in zip(items, top, strict=True):
            owner = owners[int(best)]
            hit += max(owner, key=owner.__getitem__) == item["file"]
            contains += item["file"] in owner
            purity += owner.get(item["file"], 0) / (chunk_list[int(best)].end - chunk_list[int(best)].start)
        cut = sum(c.tokens + 2 > MODEL_LIMIT for c in chunk_list)
        mixed = sum(len(o) > 1 for o in owners)
        avg = sum(c.tokens for c in chunk_list) / len(chunk_list)
        n = len(items)
        print(f"{name:24}{len(chunk_list):>6}{avg:>8.0f}{cut:>5}{mixed:>5}{hit:>9}/{n}{contains:>11}/{n}{purity / n:>14.0%}")


if __name__ == "__main__":
    main()
