"""
Step 4 — Evaluation as an LLM *chain*

A *chain* in LangChain is a pipeline: prompt → model → (optional) parser.
We use a small judge model (same fast endpoint as the classifier) to score the
assistant answer on 1–5; then application code decides whether to escalate.

This replaces ad-hoc string scoring with a reusable Runnable you can swap or extend
(e.g. swap in LangChain's evaluators later without changing the gateway).
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI


EVAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Rate the assistant response for the user question.
Score 1-5 where 5 is excellent (correct, complete, coherent) and 1 is poor.

Question: {question}
Response: {answer}

Reply with ONLY valid JSON, no markdown: {{"score": <int>, "reason": "<brief>"}}""",
        )
    ]
)


def _parse_eval_json(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {"score": 3, "reason": "unparseable judge output", "raw": text}
    try:
        data = json.loads(m.group())
        score = int(data.get("score", 3))
        reason = str(data.get("reason", ""))[:500]
        return {"score": score, "reason": reason, "raw": text}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"score": 3, "reason": "json parse error", "raw": text}


def build_quality_evaluator(judge_llm: ChatOpenAI):
    """
    Returns a Runnable: input {question, answer} -> {score, reason, raw}.
    """

    def invoke_judge(payload: dict) -> dict:
        prompt_value = EVAL_PROMPT.format_messages(
            question=payload["question"],
            answer=payload["answer"],
        )
        out = judge_llm.invoke(prompt_value)
        content = out.content if hasattr(out, "content") else str(out)
        parsed = _parse_eval_json(content)
        return parsed

    return RunnablePassthrough() | RunnableLambda(invoke_judge)
