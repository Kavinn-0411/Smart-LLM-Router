"""
LangGraph State-Graph Router
=============================

Routing logic as an explicit state graph with nodes for classification,
model inference, and quality evaluation.

The key agentic behavior is the cyclic evaluation loop: if a model's
response falls below a confidence threshold, LangGraph routes it back
to the stronger model automatically.

    observe (classify) -> reason (route) -> act (infer)
        -> evaluate (score) -> self-correct (escalate if needed)

Graph:

    START -> classify -> [route]
        SIMPLE  -> infer_qwen  -> evaluate -> [check_quality]
        COMPLEX -> infer_llama -> evaluate -> [check_quality]
            score >= threshold              -> END
            score < threshold AND was qwen  -> infer_llama (escalate)
            otherwise                       -> END

Usage:
    python -m router.graph_router
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from classifier import classify
from config import (
    QUALITY_SCORE_MAX,
    QUALITY_SCORE_MIN,
    QUALITY_THRESHOLD,
    ROUTER_LLM_KEY,
)
from router.eval_chain import build_quality_evaluator
from router.langchain_llms import get_chat_model


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class RouterState(TypedDict, total=False):
    query: str
    category: str           # "SIMPLE" or "COMPLEX"
    model_key: str          # "qwen-coder" or "llama-8b"
    answer: str
    quality_score: int
    quality_reason: str
    escalated: bool


# ---------------------------------------------------------------------------
# Node functions  (each returns a partial state update)
# ---------------------------------------------------------------------------

def classify_node(state: RouterState) -> dict:
    """Observe: classify the query as SIMPLE or COMPLEX."""
    result = classify(state["query"])
    return {
        "category": result["category"],
        "model_key": result["model"],
    }


def infer_qwen(state: RouterState) -> dict:
    """Act: generate an answer with qwen-coder."""
    llm = get_chat_model("qwen-coder", temperature=0.3, max_tokens=512)
    msg = llm.invoke([HumanMessage(content=state["query"])])
    return {"answer": (msg.content or "").strip(), "model_key": "qwen-coder"}


def infer_llama(state: RouterState) -> dict:
    """Act: generate an answer with llama-8b (or escalate to it)."""
    llm = get_chat_model("llama-8b", temperature=0.3, max_tokens=512)
    msg = llm.invoke([HumanMessage(content=state["query"])])
    escalated = state.get("answer", "") != ""
    return {
        "answer": (msg.content or "").strip(),
        "model_key": "llama-8b",
        "escalated": escalated,
    }


def evaluate_node(state: RouterState) -> dict:
    """Evaluate: score the answer quality 1-5."""
    judge = get_chat_model(ROUTER_LLM_KEY, temperature=0.0, max_tokens=256)
    evaluator = build_quality_evaluator(judge)
    result = evaluator.invoke({
        "question": state["query"],
        "answer": state["answer"],
    })
    score = max(QUALITY_SCORE_MIN, min(QUALITY_SCORE_MAX, int(result.get("score", 3))))
    return {
        "quality_score": score,
        "quality_reason": result.get("reason", ""),
    }


# ---------------------------------------------------------------------------
# Routing (conditional edge) functions
# ---------------------------------------------------------------------------

def route_by_category(state: RouterState) -> str:
    """Reason: pick the inference node based on classification."""
    if state["category"] == "SIMPLE":
        return "infer_qwen"
    return "infer_llama"


def check_quality(state: RouterState) -> str:
    """Self-correct: escalate to llama if score is low and we used qwen."""
    if (
        state["quality_score"] < QUALITY_THRESHOLD
        and state["model_key"] == "qwen-coder"
        and not state.get("escalated", False)
    ):
        return "escalate"
    return "done"


# ---------------------------------------------------------------------------
# Build the compiled graph
# ---------------------------------------------------------------------------

def build_router_graph():
    """Construct and compile the LangGraph state graph."""
    graph = StateGraph(RouterState)

    graph.add_node("classify", classify_node)
    graph.add_node("infer_qwen", infer_qwen)
    graph.add_node("infer_llama", infer_llama)
    graph.add_node("evaluate", evaluate_node)

    graph.set_entry_point("classify")

    graph.add_conditional_edges("classify", route_by_category, {
        "infer_qwen": "infer_qwen",
        "infer_llama": "infer_llama",
    })
    graph.add_edge("infer_qwen", "evaluate")
    graph.add_edge("infer_llama", "evaluate")
    graph.add_conditional_edges("evaluate", check_quality, {
        "done": END,
        "escalate": "infer_llama",
    })

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class RouteResult:
    answer: str
    escalated: bool
    quality_score: int | None
    quality_reason: str | None
    first_tool: str | None
    intermediate_steps: list = field(default_factory=list)


def run_routed_query(
    user_input: str,
    *,
    quality_threshold: int = QUALITY_THRESHOLD,
) -> RouteResult:
    """
    Run the LangGraph router: classify -> infer -> evaluate (-> escalate).
    """
    app = build_router_graph()

    final_state = app.invoke({"query": user_input, "escalated": False})

    return RouteResult(
        answer=final_state.get("answer", ""),
        escalated=final_state.get("escalated", False),
        quality_score=final_state.get("quality_score"),
        quality_reason=final_state.get("quality_reason"),
        first_tool=final_state.get("model_key"),
    )


if __name__ == "__main__":
    r = run_routed_query("Write a one-line hello world in Python")
    print(r)
    r2 = run_routed_query("Explain the CAP theorem and how it applies to distributed databases")
    print(r2)
