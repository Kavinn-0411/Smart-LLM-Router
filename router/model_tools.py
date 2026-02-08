"""
Step 2 — Tools: one LangChain Tool per backend model

A *Tool* is a named, described callable the agent can invoke. The LLM (with tool calling)
picks a tool + arguments; LangChain runs your function and feeds the *observation* back.

Here each tool forwards the user's text to a different ChatOpenAI (different vLLM port).
The router LLM decides *which* tool to use based on the query and (via memory) prior turns.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from router.langchain_llms import get_chat_model


def _answer_with_model(model_key: str, question: str) -> str:
    llm = get_chat_model(model_key, temperature=0.3, max_tokens=512)
    msg = llm.invoke([HumanMessage(content=question)])
    return (msg.content or "").strip()


@tool
def qwen_coder(question: str) -> str:
    """Fast 3B coder model. Use for simple questions, code generation, debugging, short factual answers, greetings."""
    return _answer_with_model("qwen-coder", question)


@tool
def llama_8b(question: str) -> str:
    """Larger 8B model. Use for multi-step reasoning, deep analysis, creative writing, proofs, nuanced comparisons."""
    return _answer_with_model("llama-8b", question)


def get_model_tools():
    return [qwen_coder, llama_8b]
