"""
Steps 2–4 — AgentExecutor, memory, evaluation, escalation

*Agent*: a loop where the LLM decides actions (here: which Tool to call).
*AgentExecutor* runs that loop until the agent returns a final answer or hits max iterations.

*ConversationBufferMemory* keeps the last turns of the conversation so the same executor
can resolve follow-ups ("fix that function") without resending the whole thread yourself.

Flow:
  1) AgentExecutor picks qwen_coder or llama_8b and returns an answer.
  2) Quality evaluator chain scores (question, answer).
  3) If score < threshold and the first pass used the smaller model, call llama once more (escalation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

from config import (
    AGENT_MAX_ITERATIONS,
    MODELS,
    QUALITY_SCORE_MAX,
    QUALITY_SCORE_MIN,
    QUALITY_THRESHOLD,
    ROUTER_LLM_KEY,
)
from router.eval_chain import build_quality_evaluator
from router.langchain_llms import get_chat_model
from router.model_tools import get_model_tools


REACT_TEMPLATE = """You are a query router. You MUST call exactly one tool to answer the user.

Choose qwen_coder for: simple questions, code, debugging, short facts, translations, greetings.
Choose llama_8b for: multi-step reasoning, deep analysis, creative writing, proofs, system design, nuanced debate.

Do not answer from your own knowledge; always use a tool.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

{chat_history}
Question: {input}
Thought:{agent_scratchpad}"""


def _router_llm() -> Any:
    return get_chat_model(ROUTER_LLM_KEY, temperature=0.0, max_tokens=256)


def build_agent_executor(memory: ConversationBufferMemory | None = None) -> AgentExecutor:
    """
    Build an AgentExecutor with a ReAct (text-based) agent and optional conversation memory.
    Uses ReAct instead of tool-calling to avoid requiring vLLM --enable-auto-tool-choice.
    """
    llm = _router_llm()
    tools = get_model_tools()

    prompt = PromptTemplate.from_template(REACT_TEMPLATE)

    agent = create_react_agent(llm, tools, prompt)

    if memory is None:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False, input_key="input")

    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=False,
        max_iterations=AGENT_MAX_ITERATIONS,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


def _last_tool_name(steps: list) -> str | None:
    if not steps:
        return None
    action, _obs = steps[-1]
    return getattr(action, "tool", None)


def _clamp_score(score: int) -> int:
    return max(QUALITY_SCORE_MIN, min(QUALITY_SCORE_MAX, score))


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
    memory: ConversationBufferMemory | None = None,
    *,
    quality_threshold: int = QUALITY_THRESHOLD,
) -> RouteResult:
    """
    Run the agent, evaluate quality, optionally escalate to llama-8b.

    Pass the same `memory` instance across requests in a session to preserve context.
    """
    executor = build_agent_executor(memory=memory)
    judge = get_chat_model(ROUTER_LLM_KEY, temperature=0.0, max_tokens=256)
    evaluator = build_quality_evaluator(judge)

    out = executor.invoke({"input": user_input})
    answer = (out.get("output") or "").strip()
    steps = out.get("intermediate_steps") or []
    first_tool = _last_tool_name(steps)

    eval_out = evaluator.invoke({"question": user_input, "answer": answer})
    score = _clamp_score(int(eval_out.get("score", 0)))
    reason = eval_out.get("reason")

    escalated = False
    if score < quality_threshold and first_tool == "qwen_coder":
        llm = get_chat_model("llama-8b", temperature=0.3, max_tokens=512)
        msg = llm.invoke([HumanMessage(content=user_input)])
        answer = (msg.content or "").strip()
        escalated = True
        eval_out2 = evaluator.invoke({"question": user_input, "answer": answer})
        score = _clamp_score(int(eval_out2.get("score", score)))
        reason = eval_out2.get("reason", reason)

    return RouteResult(
        answer=answer,
        escalated=escalated,
        quality_score=score,
        quality_reason=reason,
        first_tool=first_tool,
        intermediate_steps=steps,
    )


if __name__ == "__main__":
    mem = ConversationBufferMemory(memory_key="chat_history", return_messages=False, input_key="input")
    r = run_routed_query("Write a one-line hello world in Python", memory=mem)
    print(r)
    r2 = run_routed_query("Make it print twice", memory=mem)
    print(r2)
