from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from src.rag_service import OpenAIEmbedder, load_train_rows
from src.settings import load_settings

def build_index(
    train_path: Path = Path("data/train.csv"), 
    output_dir: Path = Path("data/index")
) -> None:
    settings = load_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    print("Loading training data...")

    train_rows = load_train_rows(train_path)
    # 데이터 개수
    print(f"Loaded {len(train_rows)} examples")

    # 임베더 생성
    print("Initializing embedder...")
    embedder = OpenAIEmbedder(settings)

    # 임베딩 생성
    print("Generating embeddings...")
    texts = [row.retrieval_text() for row in train_rows]
    # 모든 텍스트를 벡터로 변환
    embeddings = embedder.embed_texts(texts)
    print(f"Generated embeddings shape: {embeddings.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 임베딩과 행 데이터를 각각 저장
    matrix_path = output_dir / "embeddings.npy"
    rows_path = output_dir / "rows.json"

    np.save(matrix_path, embeddings)
    print(f"Saved embeddings to {matrix_path}")

    # retrival 후 다시 문제 복원해야되니깐 따로 저장
    rows_data = [
        {
            "question": row.question,
            "options": row.options,
            "answer": row.answer,
            "category": row.category,
        }
        for row in train_rows
    ]
    rows_path.write_text(
        json.dumps(rows_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved row data to {rows_path}")


if __name__ == "__main__":
    build_index()
    print("Index built successfully!")
