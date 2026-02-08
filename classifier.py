"""
Query classifier that routes user prompts to the appropriate model.

Uses Qwen Coder (the lighter model) to classify queries into SIMPLE vs COMPLEX,
then maps to the correct model key. Calls vLLM through LangChain ChatOpenAI
(see router.langchain_llms) instead of the raw OpenAI SDK.
"""

import time

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from config import MODELS
from router.langchain_llms import get_classifier_llm

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


def classify(query: str, llm: ChatOpenAI | None = None) -> dict:
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
    if llm is None:
        llm = get_classifier_llm()

    prompt = CLASSIFICATION_PROMPT.format(query=query)

    start = time.perf_counter()
    msg = llm.invoke([HumanMessage(content=prompt)])
    elapsed_ms = (time.perf_counter() - start) * 1000

    raw = (msg.content or "").strip().upper()

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
