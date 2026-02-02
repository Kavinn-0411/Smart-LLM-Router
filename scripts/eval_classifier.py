"""
Evaluate the query classifier against the hand-labeled test dataset.

Measures:
  - Per-query classification correctness
  - Overall accuracy (total, per-category)
  - Average / p50 / p95 classification latency

Usage:
    python scripts/eval_classifier.py
"""

import json
import sys
from pathlib import Path
from statistics import mean, median, quantiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classifier import classify, get_classifier_client
from config import MODELS

DATASET_PATH = Path(__file__).resolve().parent.parent / "test_queries.json"


def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


def run_eval():
    dataset = load_dataset()
    client = get_classifier_client()

    print("Smart LLM Router — Classifier Evaluation")
    print("=" * 70)
    print(f"Dataset: {len(dataset)} queries")
    print(f"Classifier model: qwen-coder ({MODELS['qwen-coder']['name']})")
    print("=" * 70)

    results = []
    correct = 0
    category_stats = {"SIMPLE": {"total": 0, "correct": 0}, "COMPLEX": {"total": 0, "correct": 0}}
    latencies = []

    for i, item in enumerate(dataset):
        query = item["query"]
        expected = item["expected"]

        result = classify(query, client=client)
        predicted = result["category"]
        match = predicted == expected
        if match:
            correct += 1

        category_stats[expected]["total"] += 1
        if match:
            category_stats[expected]["correct"] += 1

        latencies.append(result["latency_ms"])

        status = "OK" if match else "MISS"
        model_name = MODELS[result["model"]]["name"]
        print(f"  [{status:4s}] {i+1:2d}. [{predicted:7s}] → {model_name:20s} | "
              f"{result['latency_ms']:6.1f}ms | {query[:60]}")

        results.append({**item, "predicted": predicted, "model": result["model"],
                        "match": match, "latency_ms": result["latency_ms"]})

    # Summary
    total = len(dataset)
    accuracy = correct / total * 100

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Overall accuracy : {correct}/{total} ({accuracy:.1f}%)")

    for cat, stats in category_stats.items():
        if stats["total"] > 0:
            cat_acc = stats["correct"] / stats["total"] * 100
            print(f"  {cat:7s} accuracy : {stats['correct']}/{stats['total']} ({cat_acc:.1f}%)")

    print(f"\n  Latency (ms):")
    print(f"    Mean   : {mean(latencies):.1f}")
    print(f"    Median : {median(latencies):.1f}")
    if len(latencies) >= 4:
        p95 = quantiles(latencies, n=20)[18]
        print(f"    P95    : {p95:.1f}")
    print(f"    Min    : {min(latencies):.1f}")
    print(f"    Max    : {max(latencies):.1f}")

    target_met = accuracy >= 85
    latency_ok = mean(latencies) < 500
    print(f"\n  Accuracy target (>=85%) : {'PASS' if target_met else 'FAIL'}")
    print(f"  Latency target (<500ms) : {'PASS' if latency_ok else 'FAIL'}")

    # Save results
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_file = results_dir / "classifier_eval.json"
    with open(out_file, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "category_stats": category_stats,
            "avg_latency_ms": round(mean(latencies), 1),
            "median_latency_ms": round(median(latencies), 1),
            "details": results,
        }, f, indent=2)
    print(f"\n  Full results saved to {out_file}")


if __name__ == "__main__":
    run_eval()
