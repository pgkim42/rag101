# rag101

RAG(retrieval-augmented generation)를 한 단계씩 직접 만들어 보며 익히는 공부 저장소다.

지금은 RAG의 세 단계가 한 번씩 이어져 있다.

- **검색** (`retrieve.py`): 글 15편 중에서 질문과 뜻이 가까운 글을 골라 줄 세운다.
- **증강과 생성** (`answer.py`): 고른 글을 「아래 글만 근거로 답하세요」 지시와 함께 질문 앞에 붙이고, 생성 모델이 그 글만 보고 답을 쓰게 한다.

만들면서 정리한 설명은 [study.md](study.md)에 있다.

## 하는 일

1. `docs/ko`(또는 `docs/en`)의 글 15편을 읽는다. 파일 하나가 조각 하나다.
2. 질문과 글을 같은 임베딩 모델로 숫자 384개짜리 벡터로 바꾼다.
3. 벡터 길이를 1로 맞추고 서로 곱해 코사인 유사도를 낸다.
4. 점수가 큰 순서로 위에서 k개를 보여 준다.

임베딩 모델은 [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)다. 한국어와 영어를 같은 공간에 놓는 다국어 모델이다.

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
```

처음 실행하면 임베딩 모델(약 400MB)을 내려받는다. `--k`로 보여 줄 개수를 바꿀 수 있다. `--check`는 15문항을 다 맞히지 못하면 종료 코드 1로 끝난다.

```text
1. 광합성  (0.847)  [01-photosynthesis.md]
2. 태양광 패널의 원리  (0.577)  [08-solar-panels.md]
3. 탄소 순환  (0.478)  [14-carbon-cycle.md]
```

질문에 「광합성」이라는 단어가 없는데도 광합성 글이 1등이다. 단어가 겹치는지를 세는 게 아니라, 문장 전체의 뜻이 가까운지를 본 결과다.

## 지금 결과

`--check` 기준으로 한국어 12/15, 영어 12/15가 1등을 맞혔다. 틀린 것은 모두 주제가 가까운 글끼리 점수가 붙은 경우다.

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
[1] 광합성  (0.847)  [01-photosynthesis.md]
[2] 태양광 패널의 원리  (0.577)  [08-solar-panels.md]
[3] 탄소 순환  (0.478)  [14-carbon-cycle.md]

답 (gpt-6-luna):
식물은 잎의 엽록소가 햇빛을 흡수하고, 그 에너지로 이산화탄소와 물을 결합해 포도당을 만들면서
빛 에너지를 화학 에너지로 바꿉니다. 이 과정에서 산소도 부산물로 나옵니다. [1]
```

### 몇 편을 붙일까

검색이 틀렸던 한국어 세 질문을 `gpt-6-luna`로 돌려 봤다.

| 질문 | `--k 1` (1등 글만) | `--k 3` |
| --- | --- | --- |
| 허가되지 않은 네트워크 통신을 무엇이 막나? | VPN 글로 「통신을 암호화한다」고 답함 (틀림) | 「방화벽이 기준에 맞지 않는 연결을 차단한다 [2]」 |
| 바닷물은 왜 하루에 두 번 올랐다 내리나? | 「글에 나와 있지 않다」 | 「달의 중력과 지구 자전 때문 [2]」 |
| 돈으로 살 수 있는 양이 시간이 갈수록 줄어드는 이유는? | 「글만으로는 알 수 없다」 | 「인플레이션으로 구매력이 줄기 때문 [2]」 |

1등 글만 붙이면 모델은 모른다고 하거나, 방화벽 질문처럼 글에 있는 말로 질문과 다른 답을 만든다. 위에서 3편을 붙이면 기대한 글이 2등에 들어 있어서 셋 다 맞았다. 그래서 `--k` 기본값은 3이다. 같은 질문이라도 생성 모델의 답 문장은 실행할 때마다 조금씩 다를 수 있다.

## 구성

```text
retrieve.py          검색과 채점 (load_chunks, search, check)
answer.py            증강과 생성 (build_prompt, generate)
.env.example         생성 모델 주소, 키, 모델 이름 본보기
docs/en/             영어 원문 15편
docs/ko/             공부용 한국어 번역 15편
data/answer_key.json 질문(ko/en)과 기대하는 글
study.md             만들면서 정리한 학습 노트
```

## 다음 단계

- 글 임베딩을 미리 계산해 저장하기. 지금은 질문할 때마다 15편을 다시 계산한다.
- 옆 주제를 집는 검색을 재순위화(rerank)나 단어 검색(BM25)과 섞어서 고치기
- 그다음: 벡터 DB, 긴 문서 조각 내기, 더 큰 한국어 데이터셋([allganize/RAG-Evaluation-Dataset-KO](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO))

## 출처

- 글 15편과 정답표는 [TestingURL RAG 코퍼스](https://testingurl.dev/ai/corpus)에서 가져왔다. `docs/en`이 원문이다.
- 이 코퍼스에는 공식 한국어판이 없다. `docs/ko`와 정답표의 한국어 질문은 공부하려고 직접 옮긴 것이고, TestingURL의 공식 번역이 아니다.
