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
                "content": "한국 형사법 및 법률 전문가. A, B, C, D 중 정답 알파벳 하나만 출력.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    pred = (resp.choices[0].message.content or "").strip().upper()
    pred = pred[0] if pred and pred[0] in ("A", "B", "C", "D") else "A"
    if pred == row.answer:
        correct += 1

print(f"Zero-shot 50문제: {correct}/50 = {correct / 50:.4f}")
