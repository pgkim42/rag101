# rag101

RAG(retrieval-augmented generation)를 한 단계씩 직접 만들어 보며 익히는 공부 저장소다.

지금은 RAG의 세 단계가 한 번씩 이어져 있다.

- **검색** (`retrieve.py`, `bm25.py`): 글 15편 중에서 질문에 맞는 글을 골라 줄 세운다. 임베딩과 단어 검색(BM25)으로 후보를 추리고, 재순위화 모델로 다시 줄 세운다.
- **증강과 생성** (`answer.py`): 고른 글을 「아래 글만 근거로 답하세요」 지시와 함께 질문 앞에 붙이고, 생성 모델이 그 글만 보고 답을 쓰게 한다.

만들면서 정리한 설명은 [study.md](study.md)에 있다.

## 하는 일

1. `docs/ko`(또는 `docs/en`)의 글 15편을 읽는다. 파일 하나가 조각 하나다.
2. 질문과 글을 같은 임베딩 모델로 숫자 384개짜리 벡터로 바꾼다. 글 쪽 결과는 `.cache/`에 저장해 두고, 글이나 모델이 바뀌었을 때만 다시 계산한다.
3. 벡터 길이를 1로 맞추고 서로 곱해 코사인 유사도를 낸다.
4. 같은 두 글자 조각이 얼마나 겹치는지로 BM25 점수를 낸다. 두 점수를 0~1로 펴서 0.7 : 0.3으로 섞는다.
5. 섞은 점수 상위 5편을 재순위화 모델이 질문과 함께 읽고 다시 채점한다.
6. 점수가 큰 순서로 위에서 k개를 보여 준다.

`--method`로 어디까지 할지 고른다. `dense`는 1~3단계만, `hybrid`는 4단계까지, `rerank`(기본값)는 5단계까지 한다. 각 방식의 원리는 study.md의 「옆 주제를 집는 검색 고치기」에 있다.

- 임베딩 모델: [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2). 한국어와 영어를 같은 공간에 놓는 다국어 모델이다.
- 재순위화 모델: [`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1). 질문과 글을 한 입력으로 같이 읽는 다국어 cross-encoder다.

## 실행

Python 3.12와 [uv](https://docs.astral.sh/uv/)를 쓴다.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 질문 하나로 검색한다. 기본은 위에서 3개다.
.venv/bin/python retrieve.py "식물은 햇빛을 어떻게 에너지로 바꾸나?"

# 영어 원문으로 검색한다.
.venv/bin/python retrieve.py --lang en "Why does it rain?"

# 정답표 15문항으로 1등이 기대한 글인지 채점한다.
.venv/bin/python retrieve.py --check

# 검색 방식을 바꿔 채점한다. dense, hybrid, rerank
.venv/bin/python retrieve.py --method dense --check
```

처음 실행하면 임베딩 모델(약 400MB)을, `rerank`를 처음 쓸 때는 재순위화 모델(약 471MB)을 내려받는다. `--k`로 보여 줄 개수를 바꿀 수 있다. `--check`는 15문항을 다 맞히지 못하면 종료 코드 1로 끝난다.

`--method dense "식물은 햇빛을 어떻게 에너지로 바꾸나?"`의 결과다. 점수는 코사인 유사도다.

```text
1. 광합성  (0.847)  [01-photosynthesis.md]
2. 태양광 패널의 원리  (0.577)  [08-solar-panels.md]
3. 탄소 순환  (0.478)  [14-carbon-cycle.md]
```

질문에 「광합성」이라는 단어가 없는데도 광합성 글이 1등이다. 단어가 겹치는지를 세는 게 아니라, 문장 전체의 뜻이 가까운지를 본 결과다.

## 지금 결과

`--check`로 잰, 15문항 중 1등을 맞힌 수다.

| 방식 | 한국어 | 영어 |
| --- | --- | --- |
| `dense` (임베딩만) | 12/15 | 12/15 |
| `hybrid` (임베딩 + BM25) | 14/15 | 15/15 |
| `rerank` (hybrid 상위 5편 재채점, 기본값) | 15/15 | 14/15 |

`rerank`가 틀린 하나는 영어 「Why does it rain?」이다. 물의 순환 대신 밀물과 썰물을 골랐다. 15문항은 적어서 한두 문항 차이는 운일 수 있고, hybrid의 0.7 : 0.3은 이 15문항을 보고 고른 비율이다.

임베딩만 쓸 때(`dense`) 틀린 것은 모두 주제가 가까운 글끼리 점수가 붙은 경우였다.

| 언어 | 질문 | 기대한 글 | 1등으로 나온 글 |
| --- | --- | --- | --- |
| ko | 허가되지 않은 네트워크 통신을 무엇이 막나? | 방화벽 | VPN (0.541 대 0.537) |
| ko | 바닷물은 왜 하루에 두 번 올랐다 내리나? | 밀물과 썰물 | 물의 순환 (0.559 대 0.508) |
| ko | 돈으로 살 수 있는 양이 시간이 갈수록 줄어드는 이유는? | 인플레이션 | 복리 (0.631 대 0.618) |
| en | What blocks unauthorized network traffic? | Firewalls | VPNs |
| en | How does sunlight get turned into electricity? | How solar panels work | Photosynthesis |
| en | Why do browsers say a site is not secure? | HTTP vs HTTPS | VPNs |

생성 모델은 붙여 준 글만 보고 답을 쓴다. 그래서 이 검색이 어떤 글을 골랐느냐가 답의 질을 크게 가른다. 아래 「답 붙이기」에서 이 세 질문을 다시 본다.

## 답 붙이기

`answer.py`는 OpenAI 호환 엔드포인트면 어느 생성 모델이든 쓴다. 주소, 키, 모델은 `.env`에 둔다.

```bash
cp .env.example .env   # OPENAI_BASE_URL, OPENAI_API_KEY, RAG_LLM_MODEL을 채운다

# 위에서 3편을 붙여 답한다.
.venv/bin/python answer.py "식물은 햇빛을 어떻게 에너지로 바꾸나?"

# 1등 글만 붙인다.
.venv/bin/python answer.py --k 1 "허가되지 않은 네트워크 통신을 무엇이 막나?"

# 모델에게 보낸 입력(증강된 프롬프트)을 같이 본다.
.venv/bin/python answer.py --show-prompt "왜 비가 내리나?"
```

여기서는 API 키 대신 로컬 [CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI)(`http://127.0.0.1:8317/v1`)를 거쳐 `gpt-6-luna`를 부른다. OpenAI 키가 있으면 `OPENAI_BASE_URL` 줄을 지우고 키만 넣으면 된다. 키가 없으면 바로 안내 문구를 내고 종료 코드 1로 끝난다.

```text
근거로 붙인 글:
[1] 광합성  (9.944)  [01-photosynthesis.md]
[2] 태양광 패널의 원리  (-3.316)  [08-solar-panels.md]
[3] 탄소 순환  (-5.139)  [14-carbon-cycle.md]

답 (gpt-6-luna):
식물은 잎의 엽록소로 햇빛을 흡수하고, 그 에너지로 이산화탄소와 물을 결합해 포도당을 만듭니다.
이 과정에서 산소도 부산물로 나옵니다.[1]
```

기본 검색 방식이 `rerank`라서 괄호 안 점수는 재순위화 모델의 점수다. 코사인과 달리 범위가 정해져 있지 않고 음수도 나오며, 같은 질문 안에서 순서를 가르는 데만 쓴다. `answer.py`도 `--method`를 받는다.

### 몇 편을 붙일까

임베딩만 쓰던 때(`--method dense`) 검색이 틀렸던 한국어 세 질문을 `gpt-6-luna`로 돌려 봤다.

| 질문 | `--k 1` (1등 글만) | `--k 3` |
| --- | --- | --- |
| 허가되지 않은 네트워크 통신을 무엇이 막나? | VPN 글로 「통신을 암호화한다」고 답함 (틀림) | 「방화벽이 기준에 맞지 않는 연결을 차단한다 [2]」 |
| 바닷물은 왜 하루에 두 번 올랐다 내리나? | 「글에 나와 있지 않다」 | 「달의 중력과 지구 자전 때문 [2]」 |
| 돈으로 살 수 있는 양이 시간이 갈수록 줄어드는 이유는? | 「글만으로는 알 수 없다」 | 「인플레이션으로 구매력이 줄기 때문 [2]」 |

1등 글만 붙이면 모델은 모른다고 하거나, 방화벽 질문처럼 글에 있는 말로 질문과 다른 답을 만든다. 위에서 3편을 붙이면 기대한 글이 2등에 들어 있어서 셋 다 맞았다. 그래서 `--k` 기본값은 3이다. 같은 질문이라도 생성 모델의 답 문장은 실행할 때마다 조금씩 다를 수 있다.

검색을 `rerank`로 바꾼 뒤에는 방화벽 질문도 `--k 1`에서 방화벽 글을 1등으로 받고 「방화벽이 막습니다. [1]」이라고 답한다.

## 구성

```text
retrieve.py          검색과 채점 (load_doc_vectors, search, hybrid, rerank, check)
bm25.py              두 글자 조각 BM25 (라이브러리 없이 공식만)
answer.py            증강과 생성 (build_prompt, generate)
.env.example         생성 모델 주소, 키, 모델 이름 본보기
docs/en/             영어 원문 15편
docs/ko/             공부용 한국어 번역 15편
data/answer_key.json 질문(ko/en)과 기대하는 글
study.md             만들면서 정리한 학습 노트
.cache/              글 임베딩 캐시 (실행하면 생기고, 커밋하지 않는다)
```

## 다음 단계

- 숫자를 고를 때 쓸 질문과 채점할 때 쓸 질문을 나눠, 더 큰 질문 모음으로 세 방식을 다시 재기
- 그다음: 벡터 DB, 긴 문서 조각 내기, 더 큰 한국어 데이터셋([allganize/RAG-Evaluation-Dataset-KO](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO))

## 출처

- 글 15편과 정답표는 [TestingURL RAG 코퍼스](https://testingurl.dev/ai/corpus)에서 가져왔다. `docs/en`이 원문이다.
- 이 코퍼스에는 공식 한국어판이 없다. `docs/ko`와 정답표의 한국어 질문은 공부하려고 직접 옮긴 것이고, TestingURL의 공식 번역이 아니다.
