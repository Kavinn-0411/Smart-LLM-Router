"""
Benchmark each model: measure latency, time-to-first-token, and tokens/sec.

Usage:
    python scripts/benchmark_models.py
    python scripts/benchmark_models.py --runs 5
"""

import argparse
import sys
import time
import json
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from tabulate import tabulate
from config import MODELS

BENCHMARK_PROMPTS = [
    "Explain what a hash table is in three sentences.",
    "Write a Python function to reverse a linked list.",
    "What are the pros and cons of microservice architecture?",
]


def benchmark_single(client: OpenAI, model_key: str, prompt: str, max_tokens: int = 256):
    """Send a non-streaming request and measure end-to-end latency."""
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model_key,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    elapsed = time.perf_counter() - start

    usage = response.usage
    completion_tokens = usage.completion_tokens
    tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0

    return {
        "latency_s": round(elapsed, 3),
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_per_sec": round(tokens_per_sec, 1),
    }


def benchmark_streaming(client: OpenAI, model_key: str, prompt: str, max_tokens: int = 256):
    """Send a streaming request and measure time-to-first-token + total time."""
    start = time.perf_counter()
    ttft = None
    token_count = 0

    stream = client.chat.completions.create(
        model=model_key,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = time.perf_counter() - start
            token_count += 1

    total = time.perf_counter() - start
    tokens_per_sec = token_count / total if total > 0 else 0

    return {
        "ttft_s": round(ttft, 3) if ttft else None,
        "total_s": round(total, 3),
        "completion_tokens": token_count,
        "tokens_per_sec": round(tokens_per_sec, 1),
    }


def run_benchmark(key: str, runs: int):
    cfg = MODELS[key]
    client = OpenAI(base_url=f"http://localhost:{cfg['port']}/v1", api_key="unused")

    print(f"\n{'─'*60}")
    print(f"Benchmarking: {cfg['name']} (port {cfg['port']})")
    print(f"{'─'*60}")

    latencies = []
    tps_values = []
    ttfts = []

    for i in range(runs):
        prompt = BENCHMARK_PROMPTS[i % len(BENCHMARK_PROMPTS)]
        try:
            result = benchmark_single(client, key, prompt)
            latencies.append(result["latency_s"])
            tps_values.append(result["tokens_per_sec"])
            print(f"  Run {i+1}: {result['latency_s']}s | "
                  f"{result['completion_tokens']} tokens | "
                  f"{result['tokens_per_sec']} tok/s")

            stream_result = benchmark_streaming(client, key, prompt)
            if stream_result["ttft_s"]:
                ttfts.append(stream_result["ttft_s"])

        except Exception as e:
            print(f"  Run {i+1}: ERROR — {e}")

    if not latencies:
        return None

    stats = {
        "model": cfg["name"],
        "key": key,
        "avg_latency_s": round(mean(latencies), 3),
        "std_latency_s": round(stdev(latencies), 3) if len(latencies) > 1 else 0,
        "avg_tokens_per_sec": round(mean(tps_values), 1),
        "avg_ttft_s": round(mean(ttfts), 3) if ttfts else None,
        "runs": len(latencies),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM model servers")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per model")
    args = parser.parse_args()

    print("Smart LLM Router — Model Benchmark")
    print("=" * 60)

    all_stats = []
    for key in MODELS:
        stats = run_benchmark(key, args.runs)
        if stats:
            all_stats.append(stats)

    if not all_stats:
        print("\nNo successful benchmarks. Are the servers running?")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}\n")

    table = []
    for s in all_stats:
        table.append([
            s["model"],
            f"{s['avg_latency_s']:.3f}s ± {s['std_latency_s']:.3f}",
            f"{s['avg_tokens_per_sec']:.1f}",
            f"{s['avg_ttft_s']:.3f}s" if s["avg_ttft_s"] else "N/A",
            s["runs"],
        ])

    print(tabulate(table,
                   headers=["Model", "Avg Latency", "Avg Tok/s", "Avg TTFT", "Runs"],
                   tablefmt="grid"))

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_file = results_dir / "day1_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
