from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI

from src.rag_service import load_train_rows
from src.settings import load_settings

settings = load_settings()
client = OpenAI(api_key=settings.openai_api_key, timeout=30.0)
dev_rows = load_train_rows(Path("data/dev.csv"))

correct = 0
for row in dev_rows[:50]:
    prompt = (
        "다음 객관식 문제의 정답을 고르세요.\n"
        "문장에 '옳지 않은 것', '아닌 것', '틀린 것', '예외' 표현이 있으면 해당 조건을 반영하세요.\n"
        "반드시 A, B, C, D 중 한 글자만 출력하세요.\n\n"
        f"문제: {row.question}\n"
        f"A: {row.options['A']}\n"
        f"B: {row.options['B']}\n"
        f"C: {row.options['C']}\n"
        f"D: {row.options['D']}\n\n"
        f"정답:"
    )
    resp = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0.0,
        max_tokens=1,
        messages=[
            {
                "role": "system",
                "content": (
                    "한국 법률 객관식 문제를 푸는 전문가다. "
                    "반드시 A, B, C, D 중 하나의 알파벳 한 글자만 출력하라. "
                    "설명, 근거, 문장, 기호, 공백을 추가하지 마라. "
                    "문제에 옳지 않은/아닌/틀린/예외를 묻는 표현이 있으면 해당 조건을 반영해 선택하라."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    pred = (resp.choices[0].message.content or "").strip().upper()
    pred = pred[0] if pred and pred[0] in ("A", "B", "C", "D") else "A"
    if pred == row.answer:
        correct += 1

print(f"Zero-shot 50문제: {correct}/50 = {correct / 50:.4f}")
