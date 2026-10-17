# rag101 교재

RAG를 기초 원리부터 이해하고, 각 기술이 왜 필요한지 설명하며, 작은 구현을 스스로 다시 만들 수 있게 하는 교재다. 두 갈래로 되어 있다.

- **기초 장(A~G)**: 작은 예제와 손 계산으로 원리를 처음부터 세운다. 각 장은 「학습 목표 → 선수 지식 → 문제 상황 → 가장 단순한 접근 → 실패 사례 → 원리 → 최소 구현 → 기존 코드와 연결 → 한계와 타협 → 이해도 점검」 순서를 따른다. 점검 문제의 해설은 [`answers/`](answers/)에 따로 있다.
- **실측 장(1~7)**: 이 저장소의 코드로 실제 15편을 재 본 결과를 다룬다. 기초 장의 원리가 실제 데이터에서 어떻게 드러나는지 보여 준다.

같은 주제를 두 번 설명하지 않도록 나눴다. 기초 장은 「왜, 어떻게 계산하나」를, 실측 장은 「이 저장소에서 재 보니 어땠나」를 맡고, 서로 링크로 넘긴다.

## 읽는 순서

선수 지식을 따라 정한 순서다. 화살표는 「먼저 읽어야 하는 것」이다.

```text
A. RAG 이전에 ─▶ 1. RAG 한눈에 보기
      │
      ▼
B. 전통적인 정보 검색 (가장 깊게)
      │
      ▼
C. 임베딩의 기초 ─▶ 2. 임베딩 (실측)
      │
      ▼
D. RAG 검색 구조 ─▶ 3. 검색 (실측) ─▶ 5. 조각 내기 (실측) ─▶ 7. 임베딩 캐시 (실측)
      │
      ▼
E. 검색 결과에서 답변까지 ─▶ 6. 생성 (실측)
      │
      ▼
F. 평가와 검증 ─▶ 4. 평가 (실측)
      │
      ▼
G. 종합 실습, 전체 데이터 흐름, 다음에 배울 것
```

흐름은 이렇다. 단순한 글자 일치에서 출발해(B) 그 한계를 임베딩으로 넘고(C), 임베딩의 실패 사례에서 Hybrid Search와 재순위화가 필요한 이유를 발견한다(D). 그다음 검색한 글을 답으로 바꾸는 단계의 함정을 보고(E), 마지막으로 그 모든 숫자를 얼마나 믿을 수 있는지 따진다(F).

## 목차

| 장 | 다루는 것 | 실습 | 실행 비용 |
| --- | --- | --- | --- |
| [A. RAG 이전에 알아야 할 것](A-before-rag.md) | 학습 데이터·파라미터·컨텍스트, RAG vs Fine-tuning, 순차 검색과 역색인, 코퍼스·문서·청크·토큰, 관련성, 검색이 어려운 이유 | `labs/basics/inverted_index.py` | 무료 |
| [B. 전통적인 정보 검색](B-lexical.md) | 문자열 일치 → 토큰화 → TF → DF·IDF → TF-IDF → BM25(포화, 길이 보정), 한국어 형태소 분석과 n-gram, 한계 | `labs/basics/lexical.py` | 무료 |
| [C. 임베딩의 기초](C-vectors.md) | 벡터·크기·방향, 내적과 코사인, 정규화, 토큰화 → 토큰 임베딩 → 문맥 표현 → 풀링 → 정규화, 대조 학습, bi-encoder, 의미 유사성 vs 질문 적합성, 길이 한도, 실패 사례 | `labs/basics/vectors.py`, `labs.embedding_steps` | 무료 (`embedding_steps`는 모델 약 400MB) |
| [D. RAG 검색 구조](D-retrieval.md) | Sparse vs Dense, Hybrid가 필요한 이유, 점수 분포와 정규화, RRF, Cross-encoder와 2단계 검색, Recall·Precision, 벡터 DB, Exact vs ANN | `labs/basics/fusion.py`, `labs.score_fusion` | 무료 (`score_fusion`은 모델 필요) |
| [E. 검색 결과에서 답변까지](E-grounding.md) | R·A·G 연결, 프롬프트 배치, 관련성 vs 근거, Hallucination이 남는 이유, 인용 vs 검증, Prompt Injection, 검색 실패와 생성 실패 구분 | `labs/basics/prompting.py`, `labs.injection` | `prompting`은 무료, `injection`은 **생성 모델 키 필요** |
| [F. 평가와 검증](F-evaluation.md) | Ground Truth, Hit·Precision·Recall@k, MRR, nDCG, 정답이 여럿일 때, 작은 데이터셋의 함정, 과적합, 품질과 지연 시간, 재현 가능한 실험 | `labs/basics/metrics.py` | 무료 |
| [G. 종합 실습](G-capstone.md) | 역색인 + BM25 + 평가기를 처음부터, 오답의 근본 원인 찾기, 기술 고르기, 전체 데이터 흐름, 다음에 배울 것 | `labs/basics/capstone.py` | 무료 |
| [1. RAG 한눈에 보기](01-rag.md) | 세 단계, 이 저장소의 지도. 처음이면 [보충](01-beginner.md) | `labs/basics/tiny_rag.py`, `retrieve.py`, `answer.py` | tiny_rag는 무료, `answer.py`는 키 필요 |
| [2. 임베딩: 문장을 숫자로](02-embedding.md) | 15편의 임베딩 지도와 유사도 표 | `labs.embedding_map` | 무료 (모델) |
| [3. 검색: 세 가지 방법](03-search.md) | dense·hybrid·rerank의 실제 오답과 결과 | `retrieve.py --method` | 무료 (모델) |
| [4. 평가: 검색을 숫자로 재기](04-evaluation.md) | 세 방식의 Hit@1, Recall@k, MRR | `labs.evaluate` | 무료 (모델) |
| [5. 조각 내기: 긴 문서 다루기](05-chunking.md) | 쪼개는 방식 10가지 비교 | `labs.chunking` | 무료 (모델) |
| [6. 생성: 글을 붙여 답하기](06-generation.md) | 프롬프트, 몇 편 붙일까, 답이 없는 질문 | `answer.py`, `labs.unanswerable` | 키 필요 (`--skip-llm`이면 무료) |
| [7. 임베딩 캐시](07-cache.md) | 미리 계산하기, 언제 다시 계산하나 | `retrieve.py --check` | 무료 (모델) |

**전체 RAG 파이프라인의 데이터 흐름**은 [G장 3부](G-capstone.md#3부-전체-rag-파이프라인의-데이터-흐름)에, **다음에 배울 주제와 우선순위**는 [G장 4부](G-capstone.md#4부-다음에-배울-것과-우선순위)에 있다.

## 숫자를 읽는 규칙

교재의 숫자에는 출처를 붙였다.

| 표시 | 뜻 |
| --- | --- |
| **[측정]** | 이 저장소에서 명령을 돌려 얻은 값. 바로 옆이나 같은 절에 그 명령이 있다 |
| **[예시]** | 원리를 보이려고 지어낸 값. 실제 모델이나 데이터에서 나온 값이 아니다 |
| **[기존 측정]** | 앞선 실측 장(1~7)에 기록된 값. 기초 장을 쓰면서 다시 잰 것은 「다시 재서 같은 값」이라고 따로 적었다 |

시간(ms, 초)은 이 교재를 쓴 컴퓨터(Apple M5 Pro)에서 잰 참고값이라 다른 컴퓨터에서는 다르다. 생성 모델의 답은 같은 입력에도 매번 달라질 수 있다.

## 실습 돌리기

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

**표준 라이브러리만 쓰는 실습**(`labs/basics/`)은 파일 경로로 돌린다. `.venv` 없이 `python3`(3.10 이상)로도 된다. 각 실습은 손 계산 값을 `assert`로 검사해서, 값이 다르면 그 줄에서 멈춘다.

```bash
.venv/bin/python labs/basics/inverted_index.py   # A
.venv/bin/python labs/basics/lexical.py --real   # B
.venv/bin/python labs/basics/vectors.py          # C
.venv/bin/python labs/basics/fusion.py           # D
.venv/bin/python labs/basics/prompting.py        # E
.venv/bin/python labs/basics/metrics.py          # F
.venv/bin/python labs/basics/capstone.py         # G (TODO를 채운 뒤)
```

**모델을 쓰는 실습**은 저장소 맨 위 폴더에서 `-m`으로 돌린다. `retrieve.py`를 불러 쓰기 때문이다.

```bash
.venv/bin/python -m labs.embedding_steps         # C
.venv/bin/python -m labs.score_fusion            # D, F
.venv/bin/python -m labs.evaluate                # 4
.venv/bin/python -m labs.embedding_map           # 2
.venv/bin/python -m labs.chunking                # 5
.venv/bin/python -m labs.unanswerable --skip-llm # 6 (생성 모델 없이)
```

**생성 모델이 필요한 실습**: `answer.py`, `labs.unanswerable`(옵션 없이), `labs.injection`. 설정은 [6장](06-generation.md#생성-모델-연결하기)에 있다. 키가 없으면 안내 문구를 내고 종료 코드 1로 끝난다.

## 용어집

| 말 | 이 교재에서의 뜻 | 처음 나오는 곳 |
| --- | --- | --- |
| RAG | 질문에 맞는 글을 먼저 찾아(검색) 질문 앞에 붙이고(증강) 그걸 읽은 모델이 답을 쓰게(생성) 하는 절차 | A, 1 |
| 파라미터 | 학습으로 정해진 모델 안의 숫자들. 모델의 「기억」 | A 6.1 |
| 컨텍스트 | 이번 요청에 모델에게 함께 보낸 글 | A 6.1 |
| 코퍼스 | 검색할 글 묶음. 여기서는 `docs/ko`(또는 `docs/en`)의 15편 | A 6.4 |
| 청크(조각) | 검색 단위. 지금은 파일 하나가 청크 하나다 | A 6.4, 5 |
| 토큰 | 텍스트를 자르는 최소 단위. **누가 자르느냐에 따라 다르다**(BM25는 두 글자 조각, 임베딩 모델은 자기 토크나이저 조각) | A 6.4 |
| 역색인 | 「토큰 → 그 토큰이 나오는 문서 목록」 표 | A 6.3 |
| TF / DF / IDF | 토큰이 문서에 나온 횟수 / 토큰이 들어 있는 문서 수 / 드물수록 큰 무게 | B 6.2~6.3 |
| BM25 | TF와 IDF에 포화(k1)와 길이 보정(b)을 더한 점수 | B 6.6 |
| n-gram | 글자를 n개씩 겹쳐 자른 조각. `bm25.py`는 2-gram | B 6.8 |
| 벡터 / 차원 | 숫자 목록 / 그 길이 | C 6.1 |
| 내적 / 코사인 유사도 | 같은 자리끼리 곱해 더한 값 / 내적을 두 길이로 나눈 값. 방향만 본다 | C 6.3~6.4 |
| 정규화 | 벡터를 길이 1로 만들기. 그러면 내적 = 코사인 | C 6.5 |
| 임베딩 | 텍스트를 뜻이 방향에 담기도록 학습된 숫자 목록(여기서는 384개)으로 바꾼 것 | C 6.6 |
| 풀링 | 조각별 벡터를 하나로 합치기. 이 모델은 평균 | C 6.7 |
| bi-encoder | 질문과 글을 따로 임베딩해 비교. 글 쪽을 미리 계산할 수 있다 | C 6.11 |
| cross-encoder | 질문과 글을 한 입력으로 같이 읽고 점수 하나를 낸다. 재순위화에 쓴다 | D 6.8 |
| Hybrid Search | 글자 기반(BM25)과 뜻 기반(임베딩) 점수를 섞는 것 | D 6.1 |
| min-max / RRF | 점수를 0~1로 펴서 섞기 / 순위만으로 섞기 | D 6.4, 6.6 |
| 재순위화(rerank) | 빠른 검색으로 추린 후보만 느리고 정확한 모델로 다시 줄 세우기 | D 6.9 |
| ANN | 근사 최근접 이웃 검색. 조금 덜 정확한 대신 훨씬 적게 비교한다 | D 6.12 |
| 증강 | 고른 글을 지시와 질문과 함께 프롬프트 하나로 묶는 일 | E 6.1 |
| 근거성(groundedness) | 답의 주장이 붙인 문서에 실제로 적혀 있는 정도 | E 6.3 |
| Prompt Injection | 입력(특히 검색된 문서) 안에 모델을 조종하려는 지시를 심는 공격 | E 6.7 |
| 정답표(Ground Truth) | `data/answer_key.json`. 질문 15개와 각 질문에 맞는 글 | F 6.1 |
| Hit@k, Precision@k, Recall@k, MRR, nDCG | 검색을 재는 지표 | F 6.2~6.3 |
| 과적합 | 설정을 고를 때 쓴 질문으로 채점해 점수가 부풀려지는 것 | F 6.6 |

## 다른 문서

- [저장소 README](../README.md): 무엇이 있고 어떻게 돌리는지 짧게
- [1장 보충](01-beginner.md): 처음 읽는 사람용 설명, 모델 없는 세 단계 구현, 1장을 돌릴 때 막히는 팁
- [study.md](../study.md): 이 저장소를 만들면서 나눈 이야기를 순서대로 남긴 일지. 정리된 설명은 이 교재가 맡고, 일지는 당시의 생각 흐름을 남긴다
