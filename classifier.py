"""
Query classifier that routes user prompts to the appropriate model.

Uses Qwen Coder (the lighter model) to classify queries into SIMPLE vs COMPLEX,
then maps to the correct model key.
"""

import time
from openai import OpenAI
from config import MODELS, MODEL_TIERS

CLASSIFIER_MODEL = "qwen-coder"

CLASSIFICATION_PROMPT = """Classify this user query into exactly one category:
- SIMPLE: greetings, factual lookups, short answers, translations, code generation, debugging, code explanation, basic math, definitions, simple how-to questions
- COMPLEX: multi-step reasoning, comparative analysis, creative writing, math proofs, philosophical questions, nuanced debates, system design, research-level questions

Query: "{query}"

Respond with ONLY the category name (SIMPLE or COMPLEX), nothing else."""

CATEGORY_TO_MODEL = {
    "SIMPLE": "qwen-coder",
    "COMPLEX": "llama-8b",
}


def get_classifier_client() -> OpenAI:
    cfg = MODELS[CLASSIFIER_MODEL]
    return OpenAI(base_url=f"http://localhost:{cfg['port']}/v1", api_key="unused")


def classify(query: str, client: OpenAI | None = None) -> dict:
    """
    Classify a query and return the target model key along with metadata.

    Returns:
        {
            "query": str,
            "category": "SIMPLE" | "COMPLEX",
            "model": str,           # model key from config
            "latency_ms": float,
            "raw_response": str,     # raw LLM output (for debugging)
        }
    """
    if client is None:
        client = get_classifier_client()

    prompt = CLASSIFICATION_PROMPT.format(query=query)

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=CLASSIFIER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.0,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    raw = response.choices[0].message.content.strip().upper()

    # Parse — be lenient with model output
    if "COMPLEX" in raw:
        category = "COMPLEX"
    elif "SIMPLE" in raw:
        category = "SIMPLE"
    else:
        # Default to the heavier model if classification is unclear
        category = "COMPLEX"

    model_key = CATEGORY_TO_MODEL[category]

    return {
        "query": query,
        "category": category,
        "model": model_key,
        "latency_ms": round(elapsed_ms, 1),
        "raw_response": raw,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        q = "Write a Python function to sort a list"

    print(f"Query: {q}")
    result = classify(q)
    print(f"Category : {result['category']}")
    print(f"Model    : {result['model']} ({MODELS[result['model']]['name']})")
    print(f"Latency  : {result['latency_ms']} ms")
    print(f"Raw      : {result['raw_response']}")
