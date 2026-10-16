# rag101

RAG를 한 단계씩 익히는 폴더다. 지금은 검색까지만 한다. 질문과 가까운 글을 고르고, 답은 아직 만들지 않는다. 여기까지 나눈 설명은 [study.md](study.md)에 적어 두었다.

글 15편은 [TestingURL RAG 코퍼스](https://testingurl.dev/ai/corpus)의 영어 원문이다. 이 묶음의 공식 한국어판은 없어서, `docs/ko`에 공부용 번역을 넣어 두었다.

## 실행

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python retrieve.py "식물은 햇빛을 어떻게 에너지로 바꾸나?"
.venv/bin/python retrieve.py --check
```

처음 실행하면 임베딩 모델(약 400MB)을 받는다. `--check`는 정답표 15문항에서 1등이 기대 글인지를 센다. 방화벽 질문이 VPN 글을 집는 것처럼, 주제가 가까운 쌍은 점수가 비슷해서 1등이 빗나가기도 한다.

영어 원문으로 보려면 `--lang en`을 붙인다.
