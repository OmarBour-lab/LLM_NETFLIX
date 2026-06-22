import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chain import FALLBACK_MESSAGE, ask_netflix
from src.config import DATASET_PATH
from src.router import classify_intent


QUESTIONS_PATH = ROOT / "evaluation" / "questions.json"
EXPORTS_DIR = ROOT / "exports"


def partial_match(answer: str, expected: str) -> bool:
    return expected.lower() in answer.lower()


def write_catalog_stats() -> None:
    df = pd.read_csv(DATASET_PATH)
    EXPORTS_DIR.mkdir(exist_ok=True)
    stats_path = EXPORTS_DIR / "netflix_stats.csv"
    rows = [
        {"metric": "total_titles", "value": len(df)},
        {"metric": "movies", "value": int((df["type"] == "Movie").sum())},
        {"metric": "tv_shows", "value": int((df["type"] == "TV Show").sum())},
        {"metric": "min_release_year", "value": int(df["release_year"].min())},
        {"metric": "max_release_year", "value": int(df["release_year"].max())},
    ]
    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def evaluate() -> None:
    EXPORTS_DIR.mkdir(exist_ok=True)
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    results = []

    for item in questions:
        question = item["question"]
        expected = item["expected"]
        started = time.perf_counter()
        answer = ask_netflix(question)
        latency_ms = (time.perf_counter() - started) * 1000
        route = classify_intent(question)
        results.append(
            {
                "question": question,
                "expected": expected,
                "intent_expected": item["intent"],
                "intent_predicted": route.intent,
                "partial_match": partial_match(answer, expected),
                "fallback": FALLBACK_MESSAGE.lower() in answer.lower(),
                "latency_ms": round(latency_ms, 2),
                "answer": answer,
            }
        )

    output_path = EXPORTS_DIR / "evaluation_results.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    write_catalog_stats()
    partial_score = sum(row["partial_match"] for row in results) / len(results)
    fallback_rate = sum(row["fallback"] for row in results) / len(results)
    print(f"Evaluated {len(results)} questions.")
    print(f"Partial match: {partial_score:.2%}")
    print(f"Fallback rate: {fallback_rate:.2%}")
    print(f"Results exported to {output_path}")


if __name__ == "__main__":
    evaluate()

