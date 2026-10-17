"""1장 실습: RAG 세 단계를 표준 라이브러리만으로 작게 만든다.

검색은 두 글자 조각이 겹치는 개수를 센다. 증강은 고른 글을 지시, 질문과 함께 묶는다.
생성은 1등 글에서 질문과 가장 많이 겹치는 문장을 베낀다. 학습된 모델은 없다.
겹치는 조각이 없으면 글을 붙이지 않고 「모른다」고 한다.

  .venv/bin/python labs/basics/tiny_rag.py
  .venv/bin/python labs/basics/tiny_rag.py --real
  .venv/bin/python labs/basics/tiny_rag.py --question "김치찌개는 어떻게 끓이나?"

손 계산은 assert로 검사한다. 모델과 키는 필요 없다.
설명은 guide/01-beginner.md에 있다. 저장소의 실제 검색은 retrieve.py, 답은 answer.py가 한다.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, assert_never

ROOT: Final = Path(__file__).resolve().parents[2]
CHAPTER_QUESTION: Final = "식물은 햇빛을 어떻게 에너지로 바꾸나?"
REFUSAL: Final = "모른다. 붙인 글에 질문과 겹치는 조각이 없다."
INSTRUCTION: Final = (
    "아래 글만 근거로 답하세요. 글에 없는 내용은 모른다고 하세요.\n"
    "답에 쓴 글의 번호를 [1]처럼 붙이고, 질문과 같은 언어로 답하세요."
)
Mode = Literal["toy", "question", "real"]


@dataclass(frozen=True, slots=True)
class Passage:
    file: str
    title: str
    body: str


TOY: Final = (
    Passage(
        "01-photosynthesis.md",
        "광합성",
        "광합성은 식물이 햇빛을 화학 에너지로 바꾸는 과정이다. 잎의 엽록소가 빛을 흡수한다.",
    ),
    Passage(
        "08-solar-panels.md",
        "태양광 패널",
        "태양광 패널은 햇빛을 전기로 바꾼다.",
    ),
    Passage(
        "03-firewalls.md",
        "방화벽",
        "방화벽은 허락되지 않은 통신을 막는다.",
    ),
)


def tokenize(text: str) -> list[str]:
    """단어를 두 글자씩 겹쳐 자른다. bm25.py의 tokenize와 같은 조각이다."""
    tokens: list[str] = []
    for word in re.findall(r"\w+", text.lower()):
        tokens.extend(word[i : i + 2] for i in range(max(1, len(word) - 1)))
    return tokens


def shared(query: str, text: str) -> set[str]:
    """질문과 글에 둘 다 있는 조각. 횟수는 세지 않는다."""
    return set(tokenize(query)) & set(tokenize(text))


def rank(question: str, passages: tuple[Passage, ...] | list[Passage]) -> list[tuple[Passage, int]]:
    """검색: 겹치는 조각이 많은 글을 위로. 동점이면 파일 이름 순이다."""
    scored = [
        (passage, len(shared(question, f"{passage.title}\n{passage.body}"))) for passage in passages
    ]
    return sorted(scored, key=lambda item: (-item[1], item[0].file))


def keep(ranked: list[tuple[Passage, int]], k: int) -> list[tuple[Passage, int]]:
    """점수 0은 버린다. 가장 가깝다는 이유만으로 상관없는 글을 붙이지 않는다."""
    return [item for item in ranked if item[1] > 0][:k]


def augment(question: str, hits: list[tuple[Passage, int]]) -> str:
    """증강: 지시, 고른 글, 질문을 입력 하나로 묶는다."""
    blocks = "\n\n".join(
        f"[{i}] {passage.title}\n{passage.body}" for i, (passage, _) in enumerate(hits, start=1)
    )
    return f"지시:\n{INSTRUCTION}\n\n글:\n{blocks}\n\n질문: {question}"


def generate(question: str, hits: list[tuple[Passage, int]]) -> str:
    """생성: 1등 글에서 가장 많이 겹치는 문장을 베낀다. 모델이 아니다."""
    if not hits:
        return REFUSAL
    passage, _ = hits[0]
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", passage.body) if part.strip()]
    if not sentences:
        return REFUSAL
    chosen = max(sentences, key=lambda sentence: len(shared(question, sentence)))
    if not shared(question, chosen):
        return REFUSAL
    return f"{chosen} [1]"


def parse_doc(raw: str) -> tuple[str, str]:
    """retrieve.py의 parse_doc과 같이, 머리말을 버리고 제목과 본문을 나눈다."""
    text = raw.strip()
    if text.startswith("---"):
        _, _, text = text.split("---", 2)
        text = text.strip()
    title = ""
    body_lines: list[str] = []
    for line in text.splitlines():
        if not title and line.startswith("# "):
            title = line[2:].strip()
            continue
        body_lines.append(line)
    return title, "\n".join(body_lines).strip()


def load_corpus() -> list[Passage]:
    folder = ROOT / "docs" / "ko"
    passages: list[Passage] = []
    for path in sorted(folder.glob("*.md")):
        title, body = parse_doc(path.read_text(encoding="utf-8"))
        passages.append(Passage(path.name, title, body))
    if not passages:
        sys.exit(f"문서가 없습니다: {folder}")
    return passages


def option(name: str) -> str | None:
    if name not in sys.argv:
        return None
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv) or sys.argv[index + 1].startswith("--"):
        sys.exit(f"{name} 다음에 질문을 적으세요.")
    return sys.argv[index + 1]


def resolve_mode() -> tuple[Mode, str]:
    question = option("--question")
    if "--real" in sys.argv:
        return "real", question or CHAPTER_QUESTION
    if question is not None:
        return "question", question
    return "toy", CHAPTER_QUESTION


def passages_for(mode: Mode) -> tuple[Passage, ...] | list[Passage]:
    match mode:
        case "toy" | "question":
            return TOY
        case "real":
            return load_corpus()
        case unreachable:
            assert_never(unreachable)


def show(question: str, passages: tuple[Passage, ...] | list[Passage]) -> None:
    ranked = rank(question, passages)
    hits = keep(ranked, k=3)
    print("1. 검색")
    if hits:
        print(f"1등 파일: {hits[0][0].file}")
    else:
        print("1등 파일: 없음")
    for passage, score in ranked:
        print(f"- {passage.title}  ({score})  [{passage.file}]")
    if any(score == 0 for _, score in ranked):
        print("점수 0은 붙이지 않는다. 아무 글이나 가까운 척하지 않기 위해서다.")
    print("\n2. 증강")
    print(augment(question, hits))
    print("\n3. 생성")
    print(generate(question, hits))


def check_toy() -> None:
    """손으로 센 조각이 코드와 같은지. 틀리면 여기서 멈춘다."""
    photo, solar, firewall = TOY
    assert shared(CHAPTER_QUESTION, photo.body) == {
        "식물",
        "햇빛",
        "빛을",
        "에너",
        "너지",
        "지로",
        "바꾸",
    }
    assert [score for _, score in rank(CHAPTER_QUESTION, TOY)] == [7, 2, 0]
    assert solar.file == "08-solar-panels.md"
    assert firewall.title == "방화벽"
    hits = keep(rank(CHAPTER_QUESTION, TOY), k=3)
    assert generate(CHAPTER_QUESTION, hits) == (
        "광합성은 식물이 햇빛을 화학 에너지로 바꾸는 과정이다. [1]"
    )
    refused = generate("김치찌개는 어떻게 끓이나?", keep(rank("김치찌개는 어떻게 끓이나?", TOY), k=3))
    assert refused == REFUSAL
    assert "엽록소" not in refused


def main() -> None:
    check_toy()
    mode, question = resolve_mode()
    passages = passages_for(mode)
    if mode == "real" and question == CHAPTER_QUESTION:
        top, score = rank(question, passages)[0]
        assert top.file == "01-photosynthesis.md"
        assert score == 7
    show(question, passages)


if __name__ == "__main__":
    main()
