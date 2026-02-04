"""
Quick smoke test: send one prompt to each model and confirm a response.

Usage:
    python scripts/test_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from config import MODELS

TEST_PROMPTS = {
    "phi3-mini": "What is the capital of France? Answer in one sentence.",
    "qwen-coder": "Write a Python function that checks if a number is prime.",
    "llama-8b": "Explain the difference between TCP and UDP in simple terms.",
}


def test_model(key: str):
    cfg = MODELS[key]
    client = OpenAI(base_url=f"http://localhost:{cfg['port']}/v1", api_key="unused")

    prompt = TEST_PROMPTS.get(key, "Say hello.")
    print(f"\n{'='*60}")
    print(f"Model : {cfg['name']} ({key})")
    print(f"Port  : {cfg['port']}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")

    try:
        response = client.chat.completions.create(
            model=key,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.7,
        )
        text = response.choices[0].message.content.strip()
        tokens = response.usage
        print(f"\nResponse:\n{text}")
        print(f"\nTokens — prompt: {tokens.prompt_tokens}, "
              f"completion: {tokens.completion_tokens}, "
              f"total: {tokens.total_tokens}")
        print(f"Result: PASS")
        return True
    except Exception as e:
        print(f"\nERROR: {e}")
        print(f"Result: FAIL")
        return False


def main():
    print("Smart LLM Router — Model Smoke Test")
    print("=" * 60)

    results = {}
    for key in MODELS:
        results[key] = test_model(key)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for key, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {MODELS[key]['name']:30s} [{status}]")

    total = len(results)
    passed = sum(results.values())
    print(f"\n  {passed}/{total} models responding.")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
