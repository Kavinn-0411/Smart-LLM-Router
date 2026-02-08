"""
OpenAI compatible LangChain LLMs served with vLLM offline models.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from config import MODELS, ROUTER_LLM_KEY


def get_chat_model(model_key: str, temperature: float = 0.2, max_tokens: int | None = 1024) -> ChatOpenAI:
    cfg = MODELS[model_key]
    kwargs: dict = {
        "base_url": f"http://localhost:{cfg['port']}/v1",
        "api_key": "unused",
        "model": model_key,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


def get_classifier_llm() -> ChatOpenAI:
    """Small / fast model used for classification and (by default) quality judging."""
    return get_chat_model(ROUTER_LLM_KEY, temperature=0.0, max_tokens=10)
