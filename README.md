# rag101

RAG(retrieval-augmented generation)를 처음부터 손으로 만들어 보며 익히는 공부 저장소다. 짧은 글 15편을 코퍼스로, 검색 → 증강 → 생성을 한 단계씩 구현했고, 각 단계를 실험으로 재 보는 교재를 [`guide/`](guide/README.md)에 정리했다.

- **검색**: 임베딩, BM25, 둘을 섞기, 재순위화(cross-encoder)
- **증강과 생성**: 고른 글을 지시와 함께 붙여 생성 모델이 답하게 하기
- **실험**: 검색 평가 지표, 임베딩 지도, 긴 문서 조각 내기, 답이 없는 질문, 임베딩 단계 재현, 점수 섞기, 프롬프트 주입
- **기초 실습**: BM25, 벡터, RRF, ANN, 평가 지표를 표준 라이브러리만으로 손 계산과 맞춰 보기

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

기초 장(A~G)은 작은 예제와 손 계산으로 원리를 처음부터 세우고, 실측 장(1~7)은 이 저장소의 코드로 15편을 재 본 결과를 다룬다. 읽는 순서는 [교재 목차](guide/README.md)에 있다.

| 기초 장 | 내용 |
| --- | --- |
| [A. RAG 이전에 알아야 할 것](guide/A-before-rag.md) | 파라미터와 컨텍스트, RAG vs Fine-tuning, 역색인, 관련성 |
| [B. 전통적인 정보 검색](guide/B-lexical.md) | 문자열 일치에서 BM25까지 손 계산, 한국어 n-gram |
| [C. 임베딩의 기초](guide/C-vectors.md) | 내적과 코사인, 문장이 벡터가 되는 다섯 단계를 직접 재현, 실패 사례 |
| [D. RAG 검색 구조](guide/D-retrieval.md) | Hybrid, 점수 정규화와 RRF, 재순위화, 벡터 DB와 ANN |
| [E. 검색 결과에서 답변까지](guide/E-grounding.md) | 프롬프트 배치, 인용 vs 근거, Prompt Injection, 실패 구분 |
| [F. 평가와 검증](guide/F-evaluation.md) | nDCG, 작은 데이터셋의 함정, 과적합, 재현성 |
| [G. 종합 실습](guide/G-capstone.md) | 처음부터 만드는 검색기, 전체 데이터 흐름, 다음에 배울 것 |

| 실측 장 | 내용 |
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

`labs/basics/`는 표준 라이브러리만 쓰는 기초 실습이다. 파일 경로로 돌리고, 모델도 키도 필요 없다. 손 계산 값을 `assert`로 검사한다.

| 명령 | 보여 주는 것 | 장 |
| --- | --- | --- |
| `.venv/bin/python labs/basics/inverted_index.py` | 순차 검색과 역색인, 데이터를 키우면 | [A](guide/A-before-rag.md) |
| `.venv/bin/python labs/basics/lexical.py --real` | TF, IDF, TF-IDF, BM25 손 계산, 토큰화 비교 | [B](guide/B-lexical.md) |
| `.venv/bin/python labs/basics/vectors.py` | 내적, 코사인, 정규화, 평균의 약점 | [C](guide/C-vectors.md) |
| `.venv/bin/python labs/basics/fusion.py` | min-max, z-점수, RRF, 직접 만든 IVF 근사 검색 | [D](guide/D-retrieval.md) |
| `.venv/bin/python labs/basics/prompting.py` | 프롬프트 조립, 주입 문장, 인용 검사의 한계, 실패 구분 | [E](guide/E-grounding.md) |
| `.venv/bin/python labs/basics/metrics.py` | nDCG 손 계산, 작은 표본의 흔들림, 과적합 시뮬레이션 | [F](guide/F-evaluation.md) |
| `.venv/bin/python labs/basics/capstone.py` | 종합 실습: TODO를 채워 검색기를 처음부터 | [G](guide/G-capstone.md) |

모델을 쓰는 실험은 저장소 맨 위 폴더에서 `-m`으로 돌린다.

| 명령 | 보여 주는 것 | 장 |
| --- | --- | --- |
| `.venv/bin/python -m labs.embedding_steps` | 문장이 384개가 되는 다섯 단계를 직접 밟기, 임베딩이 약한 곳 | [C](guide/C-vectors.md) |
| `.venv/bin/python -m labs.score_fusion` | 실제 점수 분포, 섞는 방법별 결과, 단계별 지연 시간 | [D](guide/D-retrieval.md), [F](guide/F-evaluation.md) |
| `.venv/bin/python -m labs.injection` | 검색 문서 속 지시를 생성 모델이 따르는지 (`--skip-llm`이면 키 없이 프롬프트만) | [E](guide/E-grounding.md) |
| `.venv/bin/python -m labs.embedding_map` | 384차원 임베딩을 2차원 지도와 유사도 표로 그린다 (`guide/img/`) | [2](guide/02-embedding.md) |
| `.venv/bin/python -m labs.evaluate` | 세 검색 방식의 Hit@1, Recall@3/5, MRR과 문항별 순위 | [4](guide/04-evaluation.md) |
| `.venv/bin/python -m labs.chunking` | 긴 문서를 쪼개는 방식 10가지의 검색 결과 | [5](guide/05-chunking.md) |
| `.venv/bin/python -m labs.unanswerable` | 답이 없는 질문의 검색 점수와 생성 모델의 「모른다」 (`--skip-llm`이면 키 없이) | [6](guide/06-generation.md) |

## 폴더 구성

```text
retrieve.py          검색과 채점 (임베딩 캐시, dense·hybrid·rerank, --check)
bm25.py              두 글자 조각 BM25 (라이브러리 없이 공식만)
answer.py            증강과 생성
labs/                모델을 쓰는 공부용 실험 7개
labs/basics/         표준 라이브러리만 쓰는 기초 실습 7개 (종합 실습 참고 답안은 solutions/)
guide/               교재 (기초 A~G장, 실측 1~7장, 해설은 guide/answers/, 그림은 guide/img/)
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
