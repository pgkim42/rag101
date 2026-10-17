"""C장 실습: 문장이 숫자 384개가 되는 다섯 단계를 직접 밟아 보고, 임베딩이 약한 곳을 잰다.

  .venv/bin/python -m labs.embedding_steps

retrieve.py와 같은 모델(paraphrase-multilingual-MiniLM-L12-v2)을 쓴다. 처음이면 약 400MB를 내려받는다.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers.utils import logging as hf_logging

from retrieve import MODEL_NAME, ROOT

hf_logging.set_verbosity_error()
QUESTION = "식물은 햇빛을 어떻게 에너지로 바꾸나?"


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def five_steps(model, tokenizer, bert) -> None:
    print(f"=== 1부. 「{QUESTION}」가 임베딩이 되기까지 ===")
    enc = tokenizer(QUESTION, return_tensors="pt").to(bert.device)
    ids = enc["input_ids"][0].tolist()
    print(f"① 자르기: {len(ids)}조각 {tokenizer.convert_ids_to_tokens(ids)}")
    print(f"   조각 번호: {ids}")

    table = bert.embeddings.word_embeddings.weight
    print(f"② 표에서 꺼내기: 표 크기 {tuple(table.shape)}. '{tokenizer.convert_ids_to_tokens(ids[1])}' 행의 앞 4칸 = "
          f"{[round(x, 3) for x in table[ids[1], :4].tolist()]}")
    with torch.no_grad():
        start = bert.embeddings(input_ids=enc["input_ids"], token_type_ids=enc.get("token_type_ids"))[0]
    print(f"   자리 번호 벡터를 더하고 정규화(LayerNorm)한 뒤 같은 조각의 앞 4칸 = "
          f"{[round(x, 3) for x in start[1, :4].tolist()]}  → 이것이 1층에 들어가는 값")

    with torch.no_grad():
        out = bert(**enc).last_hidden_state[0].cpu()
    print(f"③ 12층을 지난 문맥 표현: {tuple(out.shape)} (조각마다 384개)")

    mask = enc["attention_mask"][0].unsqueeze(-1).float().cpu()
    pooled = (out * mask).sum(0) / mask.sum()
    print(f"④ 평균 풀링: {tuple(pooled.shape)}, 앞 4칸 = {[round(x, 3) for x in pooled[:4].tolist()]}, 길이 {pooled.norm():.3f}")

    unit = (pooled / pooled.norm()).numpy()
    print(f"⑤ 길이를 1로: 길이 {np.linalg.norm(unit):.3f}")
    ref = model.encode([QUESTION], normalize_embeddings=True)[0]
    diff = float(np.abs(unit - ref).max())
    print(f"직접 밟은 결과와 model.encode()의 차이(가장 큰 칸): {diff:.2e}")
    assert diff < 1e-5


def context(tokenizer, bert) -> None:
    print("\n=== 2부. 같은 조각, 다른 문맥: 「배」 ===")
    sents = ["배를 타고 바다로 나갔다.", "배를 깎아 먹었다.", "배가 아파서 병원에 갔다."]
    rows, inputs = [], []
    for s in sents:
        enc = tokenizer(s, return_tensors="pt").to(bert.device)
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        pos = tokens.index("▁배")
        with torch.no_grad():
            rows.append(bert(**enc).last_hidden_state[0, pos].cpu().numpy())
        inputs.append(bert.embeddings.word_embeddings.weight[enc["input_ids"][0, pos]].detach().cpu().numpy())
        print(f"  {s}  → {tokens}")
    print(f"  ② 표에서 꺼낸 「▁배」끼리 코사인: {cos(inputs[0], inputs[1]):.3f} (같은 행이니 1)")
    print(f"  ③ 문맥을 거친 「▁배」: 타는 배 vs 먹는 배 {cos(rows[0], rows[1]):.3f}, "
          f"타는 배 vs 아픈 배 {cos(rows[0], rows[2]):.3f}, 먹는 배 vs 아픈 배 {cos(rows[1], rows[2]):.3f}")
    assert cos(inputs[0], inputs[1]) > 0.9999 and cos(rows[0], rows[1]) < 0.99


def probes(model) -> None:
    print("\n=== 3부. 임베딩이 약한 곳: 문장 쌍의 코사인 (이 모델에서 잰 값) ===")
    pairs = [
        ("뜻 같음, 글자 다름", "식물은 햇빛을 어떻게 에너지로 바꾸나?", "광합성의 원리"),
        ("부정", "방화벽은 허가되지 않은 연결을 차단한다.", "방화벽은 허가되지 않은 연결을 차단하지 않는다."),
        ("숫자", "예금 금리는 연 3%다.", "예금 금리는 연 30%다."),
        ("주어와 목적어 바꿈", "고양이가 개를 쫓았다.", "개가 고양이를 쫓았다."),
        ("주제 같음, 묻는 것 다름", "DNS 서버 주소는 어떻게 바꾸나?", "DNS는 도메인 이름을 IP 주소로 바꾸는 시스템이다."),
        ("무관", "식물은 햇빛을 어떻게 에너지로 바꾸나?", "김치찌개는 어떻게 끓이나?"),
    ]
    vecs = model.encode([s for _, a, b in pairs for s in (a, b)], normalize_embeddings=True)
    for i, (name, a, b) in enumerate(pairs):
        print(f"  {float(vecs[2 * i] @ vecs[2 * i + 1]):.3f}  {name:14} {a} / {b}")


def truncation(model, tokenizer) -> None:
    print("\n=== 4부. 128조각 한도: 끝에 붙인 문장은 임베딩을 바꾸지 못한다 ===")
    docs = sorted((ROOT / "docs" / "ko").glob("*.md"))
    long = " ".join(p.read_text(encoding="utf-8").split("---", 2)[2].strip() for p in docs[:2])
    extra = long + " 결론: 이 글의 진짜 주제는 화산 폭발이다."
    n = len(tokenizer(long)["input_ids"])
    a, b = model.encode([long, extra], normalize_embeddings=True)
    print(f"  max_seq_length = {model.max_seq_length}. 앞의 글은 {n}조각(시작·끝 표시 포함).")
    print(f"  끝에 「결론: 이 글의 진짜 주제는 화산 폭발이다.」를 붙이기 전후 코사인: {float(a @ b):.6f}")
    assert n > model.max_seq_length and float(a @ b) > 0.999999


def main() -> None:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    tokenizer, bert = model.tokenizer, model[0].auto_model.eval()
    print(f"모델: {MODEL_NAME}\n구성: {[type(m).__name__ for m in model]}\n")
    five_steps(model, tokenizer, bert)
    context(tokenizer, bert)
    probes(model)
    truncation(model, tokenizer)
    print("\n모든 검사를 통과했다.")


if __name__ == "__main__":
    main()
