"""LangGraph assembly for the IT operations assistant.

Four lanes leave the supervisor. Three of them are short; the fourth is the one
that needs an agent rather than a single model call:

    START -> supervisor
      selamlama  -> greeting                                        -> END
      politika   -> policy  <-> policy_tools                        -> END
      operasyon  -> ops     <-> ops_tools                           -> END
      talep      -> planner -> policy <-> policy_tools
                            -> ops    <-> ops_tools
                            -> decision -> approval -> action_tools -> reporter -> END

The graph is built at runtime because the operational tools depend on which
transport is available (MCP or the local fallback).
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agents.approval_node import approval_node
from agents.budget_guard_node import budget_guard_node
from agents.decision_agent import make_decision_node
from agents.greeting_agent import greeting_node
from agents.invalid_tool_node import invalid_tool_node
from agents.ops_agent import make_ops_node
from agents.planner_agent import planner_node
from agents.policy_agent import policy_node
from agents.reporter_agent import reporter_node
from agents.supervisor_agent import supervisor_node
from config import log
from state import AssistantState
from tools import WRITE_TOOL_NAMES, OpsToolset
from tools.policy_tools import POLICY_TOOLS

# A small model can get stuck re-calling the same tool. The guard lets an agent
# use its tools a few times and then forces the graph forward with whatever has
# been gathered, instead of looping until the context window runs out.
#
# The budgets differ because the work differs: the policy agent usually needs
# one or two searches, while the operations agent legitimately chains four
# lookups (employee -> asset -> failure history -> stock), each depending on the
# previous result.
MAX_POLICY_LOOPS = 3
MAX_OPS_LOOPS = 6


def _wants_tools(state: AssistantState, counter: str, limit: int) -> bool:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)

    if not tool_calls:
        return False

    if state.get(counter, 0) > limit:
        log(
            f"[GUARD] Tool dongu siniri asildi ({counter} > {limit}), "
            "toplanan veriyle devam ediliyor."
        )
        return False

    return True


def supervisor_router(state: AssistantState) -> str:
    return state.get("route", "politika")


def _has_pending_calls(state: AssistantState) -> bool:
    return bool(getattr(state["messages"][-1], "tool_calls", None))


def policy_router(state: AssistantState) -> str:
    if _wants_tools(state, "policy_loops", MAX_POLICY_LOOPS):
        return "tools"

    # Budget spent but the agent still asked for tools: the pending call has to
    # be answered before the lane moves on.
    if _has_pending_calls(state):
        return "butce"

    return "devam" if state.get("route") == "talep" else "bitir"


def ops_router(state: AssistantState) -> str:
    if _wants_tools(state, "ops_loops", MAX_OPS_LOOPS):
        return "tools"

    if _has_pending_calls(state):
        return "butce"

    return "devam" if state.get("route") == "talep" else "bitir"


def budget_router(state: AssistantState) -> str:
    """Send the lane onward after the budget guard has closed the calls."""

    if state.get("route") != "talep":
        return "bitir"

    return "ops" if state.get("last_agent") == "politika" else "karar"


def decision_router(state: AssistantState) -> str:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    if not tool_calls:
        return "rapor"

    # Only a genuine write call may reach the approval gate. Anything else is an
    # invented tool name and is answered as an error instead of being approved.
    if all(call["name"] in WRITE_TOOL_NAMES for call in tool_calls):
        return "onay"

    return "gecersiz"


def approval_router(state: AssistantState) -> str:
    return "aksiyon" if state.get("approved") else "rapor"


def build_graph(toolset: OpsToolset):
    """Compile the assistant graph for a given operational toolset."""

    builder = StateGraph(AssistantState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("greeting_agent", greeting_node)
    builder.add_node("planner_agent", planner_node)
    builder.add_node("policy_agent", policy_node)
    builder.add_node("policy_tools", ToolNode(POLICY_TOOLS))
    builder.add_node("ops_agent", make_ops_node(toolset.read_tools))
    builder.add_node("ops_tools", ToolNode(toolset.read_tools))
    builder.add_node("decision_agent", make_decision_node(toolset.write_tools))
    builder.add_node("approval_gate", approval_node)
    builder.add_node("invalid_tool", invalid_tool_node)
    builder.add_node("budget_guard", budget_guard_node)
    builder.add_node("action_tools", ToolNode(toolset.write_tools))
    builder.add_node("reporter_agent", reporter_node)

    builder.add_edge(START, "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "selamlama": "greeting_agent",
            "politika": "policy_agent",
            "operasyon": "ops_agent",
            "talep": "planner_agent",
        },
    )

    builder.add_edge("greeting_agent", END)
    builder.add_edge("planner_agent", "policy_agent")

    builder.add_conditional_edges(
        "policy_agent",
        policy_router,
        {
            "tools": "policy_tools",
            "butce": "budget_guard",
            "devam": "ops_agent",
            "bitir": END,
        },
    )
    builder.add_edge("policy_tools", "policy_agent")

    builder.add_conditional_edges(
        "ops_agent",
        ops_router,
        {
            "tools": "ops_tools",
            "butce": "budget_guard",
            "devam": "decision_agent",
            "bitir": END,
        },
    )
    builder.add_edge("ops_tools", "ops_agent")

    builder.add_conditional_edges(
        "decision_agent",
        decision_router,
        {
            "onay": "approval_gate",
            "gecersiz": "invalid_tool",
            "rapor": "reporter_agent",
        },
    )

    builder.add_edge("invalid_tool", "reporter_agent")

    builder.add_conditional_edges(
        "budget_guard",
        budget_router,
        {"ops": "ops_agent", "karar": "decision_agent", "bitir": END},
    )

    builder.add_conditional_edges(
        "approval_gate",
        approval_router,
        {"aksiyon": "action_tools", "rapor": "reporter_agent"},
    )

    builder.add_edge("action_tools", "reporter_agent")
    builder.add_edge("reporter_agent", END)

    # The checkpointer gives the assistant its memory: conversation history per
    # thread_id, and the ability to pause at the approval gate and resume later.
    return builder.compile(checkpointer=InMemorySaver())
