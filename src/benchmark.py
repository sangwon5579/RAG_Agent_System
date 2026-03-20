from __future__ import annotations
import argparse
import csv
from pathlib import Path
from src.rag_service import RagRuntime, load_train_rows
from src.settings import load_settings

def benchmark(
    dev_path: Path = Path("data/dev.csv"),
    output_path: Path = Path("outputs/benchmark_results.csv"),
) -> float:
    
    settings = load_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    print("Initializing runtime...")
    runtime = RagRuntime(settings)
    runtime.load_index()

    print("Loading dev data...")
    dev_rows = load_train_rows(dev_path)
    print(f"Loaded {len(dev_rows)} dev examples")

    # 평가루프
    print("Running inference...")
    correct = 0
    output_rows: list[dict[str, str]] = []

    for idx, row in enumerate(dev_rows, start=1):
        if idx % 25 == 0:
            print(f"  Processed {idx}/{len(dev_rows)}")

        pred = runtime.infer_mcq(row.question, row.options, row.category)
        is_correct = pred == row.answer

        # 정답 맞으면 카운트 추가
        if is_correct:
            correct += 1

        # 결과 저장
        output_rows.append(
            {
                "question": row.question,
                "category": row.category,
                "A": row.options["A"],
                "B": row.options["B"],
                "C": row.options["C"],
                "D": row.options["D"],
                "answer": row.answer,
                "prediction": pred,
                "is_correct": "1" if is_correct else "0",
            }
        )

    accuracy = correct / len(dev_rows)
    print("\n=== Benchmark Results ===")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(dev_rows)})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Results saved to {output_path}")

    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark on dev dataset")
    parser.add_argument("--dev", type=Path, default=Path("data/dev.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmark_results.csv")
    )
    args = parser.parse_args()

    benchmark(args.dev, args.output)


if __name__ == "__main__":
    main()
