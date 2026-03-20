from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_service import OpenAIEmbedder, load_train_rows
from src.settings import load_settings


def main() -> None:
    settings = load_settings()
    train_rows = load_train_rows(Path("data/train.csv"))
    texts = [row.retrieval_text() for row in train_rows]

    embedder = OpenAIEmbedder(settings)
    embeddings = embedder.embed_texts(texts)

    index_dir = Path(settings.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    np.save(index_dir / "embeddings.npy", embeddings)
    payload = [
        {
            "question": row.question,
            "options": row.options,
            "answer": row.answer,
            "category": row.category,
        }
        for row in train_rows
    ]
    (index_dir / "rows.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"index_saved={index_dir}")
    print(f"rows={len(train_rows)}")


if __name__ == "__main__":
    main()
