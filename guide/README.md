# rag101 교재

RAG를 처음부터 손으로 만들어 보며 익히는 교재다. 장마다 「왜 필요한가 → 원리 → 이 저장소에서는 → 직접 해 보기 → 결과 읽기」 순서로 쓰여 있고, 끝에 연습 문제가 있다. 표에 적힌 숫자는 모두 이 저장소의 코드를 실제로 돌려서 나온 값이다.

## 읽는 순서

| 장 | 다루는 것 | 직접 돌려 볼 것 |
| --- | --- | --- |
| [1. RAG 한눈에 보기](01-rag.md) | 검색 → 증강 → 생성, 이 저장소의 지도 | `retrieve.py`, `answer.py` |
| [2. 임베딩: 문장을 숫자로](02-embedding.md) | 문장이 숫자 384개가 되는 과정, 코사인 유사도, 임베딩 지도 | `labs.embedding_map` |
| [3. 검색: 세 가지 방법](03-search.md) | 임베딩 검색, BM25, 섞기, 재순위화 | `retrieve.py --method` |
| [4. 평가: 검색을 숫자로 재기](04-evaluation.md) | Hit@1, Recall@k, MRR, 과적합 | `labs.evaluate` |
| [5. 조각 내기: 긴 문서 다루기](05-chunking.md) | 왜 쪼개나, 고정 길이·겹침·문장 단위 | `labs.chunking` |
| [6. 생성: 글을 붙여 답하기](06-generation.md) | 프롬프트, 몇 편 붙일까, 답이 없는 질문 | `answer.py`, `labs.unanswerable` |
| [7. 임베딩 캐시](07-cache.md) | 미리 계산해 두기, 언제 다시 계산하나 | `retrieve.py --check` |

1장부터 순서대로 읽는 것을 권한다. 2장과 3장은 뒤의 장들이 기대는 기초다. 4~7장은 서로 덜 엮여 있어서 관심 가는 것부터 읽어도 된다.

## 준비

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

실험은 저장소 맨 위 폴더에서 `-m`으로 돌린다. `labs/` 안의 파일이 `retrieve.py`를 불러 쓰기 때문이다.

```bash
.venv/bin/python -m labs.evaluate
.venv/bin/python -m labs.embedding_map
.venv/bin/python -m labs.chunking
.venv/bin/python -m labs.unanswerable --skip-llm   # 생성 모델 없이
```

`answer.py`와 `labs.unanswerable`(옵션 없이)은 생성 모델이 필요하다. 설정은 [6장](06-generation.md#생성-모델-연결하기)에 있다.

## 용어집

| 말 | 이 교재에서의 뜻 |
| --- | --- |
| RAG | 질문에 맞는 글을 먼저 찾아(검색) 질문 앞에 붙이고(증강) 그걸 읽은 모델이 답을 쓰게(생성) 하는 절차 |
| 코퍼스 | 검색할 글 묶음. 여기서는 `docs/ko`(또는 `docs/en`)의 15편 |
| 조각(chunk) | 검색 단위. 지금은 파일 하나가 조각 하나다. 5장에서 더 잘게 쪼갠다 |
| 토큰 | 모델이 문장을 자르는 단위. 이 교재에서는 「조각」과 헷갈리지 않게 「모델 조각」이라고도 쓴다 |
| 임베딩 | 문장을 숫자 목록(여기서는 384개)으로 바꾼 것. 뜻이 비슷하면 방향이 비슷하다 |
| 코사인 유사도 | 두 숫자 목록의 방향이 얼마나 같은지. 1이면 같고, 0이면 무관하다 |
| bi-encoder | 질문과 글을 따로 임베딩해서 비교하는 모델. 지금 쓰는 MiniLM |
| cross-encoder | 질문과 글을 한 입력으로 같이 읽고 점수 하나를 내는 모델. 재순위화에 쓴다 |
| BM25 | 같은 낱말 조각이 얼마나, 얼마나 드물게 겹치는지로 매기는 점수 |
| 재순위화(rerank) | 빠른 검색으로 추린 후보만 느리고 정확한 모델로 다시 줄 세우는 일 |
| 정답표 | `data/answer_key.json`. 질문 15개와 각 질문에 맞는 글 |
| Hit@1, Recall@k, MRR | 검색을 재는 지표. 4장 |
| 증강 | 고른 글을 질문 앞에 붙여 모델 입력을 키우는 일 |
| 생성 모델 | 글을 읽고 답을 쓰는 모델. 여기서는 `gpt-6-luna` |

## 다른 문서

- [저장소 README](../README.md): 무엇이 있고 어떻게 돌리는지 짧게
- [study.md](../study.md): 이 저장소를 만들면서 나눈 이야기를 순서대로 남긴 일지. 교재의 내용 상당수가 여기서 나왔다
