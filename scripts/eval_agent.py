"""
End-to-end evaluation of the Smart LLM Router agent pipeline.

Measures:
  - Routing correctness (did the agent pick the right tool for SIMPLE vs COMPLEX?)
  - Quality scores from the built-in eval chain (1-5)
  - Escalation rate (how often qwen_coder answers get bumped to llama_8b)
  - End-to-end latency per query
  - Error rate (queries that failed entirely)

Usage:
    python scripts/eval_agent.py
"""

import json
import sys
import time
import traceback
from pathlib import Path
from statistics import mean, median, quantiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MODELS, QUALITY_THRESHOLD
from router.graph_router import RouteResult, run_routed_query

DATASET_PATH = Path(__file__).resolve().parent.parent / "test_queries.json"

EXPECTED_TOOL = {
    "SIMPLE": "qwen-coder",
    "COMPLEX": "llama-8b",
}


def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


def run_eval():
    dataset = load_dataset()

    print("Smart LLM Router — End-to-End Agent Evaluation")
    print("=" * 80)
    print(f"Dataset         : {len(dataset)} queries")
    print(f"Quality threshold: {QUALITY_THRESHOLD}")
    print("=" * 80)

    results: list[dict] = []
    route_correct = 0
    errors = 0
    escalations = 0
    scores: list[int] = []
    latencies: list[float] = []
    category_stats = {
        "SIMPLE": {"total": 0, "correct": 0},
        "COMPLEX": {"total": 0, "correct": 0},
    }

    for i, item in enumerate(dataset):
        query = item["query"]
        expected = item["expected"]
        expected_tool = EXPECTED_TOOL[expected]

        start = time.perf_counter()
        try:
            result: RouteResult = run_routed_query(query)
            elapsed_ms = (time.perf_counter() - start) * 1000

            routed_tool = result.first_tool
            route_ok = routed_tool == expected_tool
            if route_ok:
                route_correct += 1

            category_stats[expected]["total"] += 1
            if route_ok:
                category_stats[expected]["correct"] += 1

            score = result.quality_score or 0
            scores.append(score)
            latencies.append(elapsed_ms)

            if result.escalated:
                escalations += 1

            esc_tag = " ESC" if result.escalated else "    "
            route_tag = "OK" if route_ok else "MISS"
            print(
                f"  [{route_tag:4s}] {i + 1:2d}. "
                f"[{routed_tool or 'NONE':11s}] "
                f"score={score} {esc_tag} | "
                f"{elapsed_ms:7.0f}ms | "
                f"{query[:55]}"
            )

            results.append({
                "query": query,
                "expected": expected,
                "expected_tool": expected_tool,
                "routed_tool": routed_tool,
                "route_correct": route_ok,
                "quality_score": score,
                "quality_reason": result.quality_reason,
                "escalated": result.escalated,
                "answer_preview": result.answer[:200],
                "latency_ms": round(elapsed_ms, 1),
                "error": None,
            })

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            errors += 1
            category_stats[expected]["total"] += 1
            latencies.append(elapsed_ms)
            print(
                f"  [ERR ] {i + 1:2d}. "
                f"{elapsed_ms:7.0f}ms | "
                f"{query[:55]}"
            )
            traceback.print_exc()
            results.append({
                "query": query,
                "expected": expected,
                "expected_tool": expected_tool,
                "routed_tool": None,
                "route_correct": False,
                "quality_score": None,
                "quality_reason": None,
                "escalated": False,
                "answer_preview": None,
                "latency_ms": round(elapsed_ms, 1),
                "error": str(e),
            })

    total = len(dataset)
    evaluated = total - errors
    route_accuracy = (route_correct / total * 100) if total else 0
    escalation_rate = (escalations / evaluated * 100) if evaluated else 0

    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")

    print(f"\n  Routing")
    print(f"    Overall accuracy : {route_correct}/{total} ({route_accuracy:.1f}%)")
    for cat, stats in category_stats.items():
        if stats["total"] > 0:
            cat_acc = stats["correct"] / stats["total"] * 100
            print(f"    {cat:7s} accuracy : {stats['correct']}/{stats['total']} ({cat_acc:.1f}%)")

    if scores:
        avg_score = mean(scores)
        print(f"\n  Quality Scores (1-5)")
        print(f"    Mean   : {avg_score:.2f}")
        print(f"    Median : {median(scores):.1f}")
        if len(scores) >= 4:
            p95 = quantiles(scores, n=20)[18]
            print(f"    P95    : {p95:.1f}")
        print(f"    Min    : {min(scores)}")
        print(f"    Max    : {max(scores)}")

    print(f"\n  Escalation")
    print(f"    Escalated : {escalations}/{evaluated} ({escalation_rate:.1f}%)")

    print(f"\n  Errors")
    print(f"    Failed : {errors}/{total}")

    if latencies:
        print(f"\n  Latency (ms)")
        print(f"    Mean   : {mean(latencies):.0f}")
        print(f"    Median : {median(latencies):.0f}")
        if len(latencies) >= 4:
            p95 = quantiles(latencies, n=20)[18]
            print(f"    P95    : {p95:.0f}")
        print(f"    Min    : {min(latencies):.0f}")
        print(f"    Max    : {max(latencies):.0f}")

    route_pass = route_accuracy >= 85
    quality_pass = mean(scores) >= 3.5 if scores else False
    print(f"\n  Targets")
    print(f"    Routing accuracy (>=85%)    : {'PASS' if route_pass else 'FAIL'}")
    print(f"    Avg quality score (>=3.5)   : {'PASS' if quality_pass else 'FAIL'}")

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_file = results_dir / "agent_eval.json"
    with open(out_file, "w") as f:
        json.dump({
            "route_accuracy": round(route_accuracy, 1),
            "route_correct": route_correct,
            "total": total,
            "category_stats": category_stats,
            "avg_quality_score": round(mean(scores), 2) if scores else None,
            "median_quality_score": round(median(scores), 1) if scores else None,
            "escalation_rate": round(escalation_rate, 1),
            "escalations": escalations,
            "errors": errors,
            "avg_latency_ms": round(mean(latencies), 1) if latencies else None,
            "median_latency_ms": round(median(latencies), 1) if latencies else None,
            "details": results,
        }, f, indent=2)
    print(f"\n  Full results saved to {out_file}")


if __name__ == "__main__":
    run_eval()
