from __future__ import annotations

import csv
import json
import os
import urllib.error
import urllib.request
from pathlib import Path


def format_query(question: str, a: str, b: str, c: str, d: str) -> str:
    return "\n".join(
        [
            question,
            f"A. {a}",
            f"B. {b}",
            f"C. {c}",
            f"D. {d}",
        ]
    )


def main() -> None:
    base_url = os.getenv("AGENT_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    endpoint = f"{base_url}/inference"

    dev_path = Path("data/dev.csv")
    total = 0
    correct = 0

    with dev_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = format_query(
                str(row.get("question", "")).strip(),
                str(row.get("A", "")).strip(),
                str(row.get("B", "")).strip(),
                str(row.get("C", "")).strip(),
                str(row.get("D", "")).strip(),
            )

            payload = {"query": query}
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except urllib.error.URLError as exc:
                raise RuntimeError(
                    "Failed to call inference server. Start server first: make run"
                ) from exc

            pred_raw = str(data.get("answer", "A")).strip().upper()
            pred = pred_raw if pred_raw in {"A", "B", "C", "D"} else "A"

            answer_raw = str(row.get("answer", "1")).strip()
            answer_idx = min(max(int(answer_raw), 1), 4) - 1
            gold = ["A", "B", "C", "D"][answer_idx]
            total += 1
            correct += int(pred == gold)

    accuracy = (correct / total) if total else 0.0
    print(f"dev_accuracy={accuracy:.6f}")
    print(f"correct={correct}")
    print(f"total={total}")


if __name__ == "__main__":
    main()
