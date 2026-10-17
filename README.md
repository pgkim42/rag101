# rag101

RAG(retrieval-augmented generation)를 처음부터 손으로 만들어 보며 익히는 공부 저장소다. 짧은 글 15편을 코퍼스로, 검색 → 증강 → 생성을 한 단계씩 구현했고, 각 단계를 실험으로 재 보는 교재를 [`guide/`](guide/README.md)에 정리했다.

- **검색**: 임베딩, BM25, 둘을 섞기, 재순위화(cross-encoder)
- **증강과 생성**: 고른 글을 지시와 함께 붙여 생성 모델이 답하게 하기
- **실험 4개**: 검색 평가 지표, 임베딩 지도, 긴 문서 조각 내기, 답이 없는 질문

## 빠르게 돌려 보기

Python 3.12와 [uv](https://docs.astral.sh/uv/)를 쓴다.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 검색: 질문에 맞는 글 3편
.venv/bin/python retrieve.py "식물은 햇빛을 어떻게 에너지로 바꾸나?"

# 정답표 15문항으로 채점 (방식: dense, hybrid, rerank)
.venv/bin/python retrieve.py --check
.venv/bin/python retrieve.py --method dense --check

# 검색 + 증강 + 생성 (생성 모델 설정이 필요하다)
cp .env.example .env
.venv/bin/python answer.py --show-prompt "식물은 햇빛을 어떻게 에너지로 바꾸나?"
```

처음 실행하면 임베딩 모델(약 400MB)을, 재순위화를 처음 쓸 때 재순위화 모델(약 471MB)을 내려받는다. 생성 모델은 OpenAI 형식 API면 무엇이든 되고, 설정은 [6장](guide/06-generation.md#생성-모델-연결하기)에 있다.

## 교재

| 장 | 내용 |
| --- | --- |
| [1. RAG 한눈에 보기](guide/01-rag.md) | 세 단계, 임베딩과 RAG의 차이, 이 저장소의 지도 |
| [2. 임베딩: 문장을 숫자로](guide/02-embedding.md) | 문장이 숫자 384개가 되는 과정, 코사인 유사도, 임베딩 지도 그림 |
| [3. 검색: 세 가지 방법](guide/03-search.md) | 임베딩 검색이 옆 주제를 집는 이유, BM25, 섞기, 재순위화 |
| [4. 평가: 검색을 숫자로 재기](guide/04-evaluation.md) | Hit@1, Recall@k, MRR, 과적합 |
| [5. 조각 내기: 긴 문서 다루기](guide/05-chunking.md) | 왜 쪼개나, 고정 길이·겹침·문장 단위 비교 |
| [6. 생성: 글을 붙여 답하기](guide/06-generation.md) | 프롬프트, 몇 편 붙일까, 답이 없는 질문 |
| [7. 임베딩 캐시](guide/07-cache.md) | 미리 계산하기, 언제 다시 계산하나 |

[용어집과 읽는 순서](guide/README.md)도 교재 첫 페이지에 있다.

## 실험

저장소 맨 위 폴더에서 `-m`으로 돌린다.

| 명령 | 보여 주는 것 | 장 |
| --- | --- | --- |
| `.venv/bin/python -m labs.embedding_map` | 384차원 임베딩을 2차원 지도와 유사도 표로 그린다 (`guide/img/`) | [2](guide/02-embedding.md) |
| `.venv/bin/python -m labs.evaluate` | 세 검색 방식의 Hit@1, Recall@3/5, MRR과 문항별 순위 | [4](guide/04-evaluation.md) |
| `.venv/bin/python -m labs.chunking` | 긴 문서를 쪼개는 방식 10가지의 검색 결과 | [5](guide/05-chunking.md) |
| `.venv/bin/python -m labs.unanswerable` | 답이 없는 질문의 검색 점수와 생성 모델의 「모른다」 (`--skip-llm`이면 키 없이) | [6](guide/06-generation.md) |

## 폴더 구성

```text
retrieve.py          검색과 채점 (임베딩 캐시, dense·hybrid·rerank, --check)
bm25.py              두 글자 조각 BM25 (라이브러리 없이 공식만)
answer.py            증강과 생성
labs/                공부용 실험 4개
guide/               교재 (1~7장, 그림은 guide/img/)
docs/ko/, docs/en/   코퍼스 15편 (공부용 한국어 번역, 영어 원문)
data/answer_key.json 정답표: 질문(ko/en)과 기대하는 글
study.md             만들면서 나눈 이야기를 순서대로 남긴 일지
.env.example         생성 모델 주소, 키, 모델 이름 본보기
```

## 출처와 모델

- 글 15편과 정답표는 [TestingURL RAG 코퍼스](https://testingurl.dev/ai/corpus)에서 가져왔다. `docs/en`이 원문이다. 이 코퍼스에는 공식 한국어판이 없다. `docs/ko`와 정답표의 한국어 질문은 공부하려고 직접 옮긴 것이고, TestingURL의 공식 번역이 아니다.
- 임베딩 모델: [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- 재순위화 모델: [`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
- 생성 모델: `gpt-6-luna` (로컬 [CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI)를 거쳐 부른다. 다른 OpenAI 형식 엔드포인트로 바꿀 수 있다)
