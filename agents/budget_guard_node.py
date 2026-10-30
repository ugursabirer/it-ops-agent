"""Closes off tool calls that the loop budget refuses to run.

When a gathering agent still wants tools after its budget is spent, the graph
moves on. It cannot simply leave the pending call hanging: a tool call with no
matching result is an incomplete exchange, and the next agent in the lane was
observed returning an empty answer because of it. This node answers every
pending call explicitly, so the next agent sees a consistent history and knows
why no more data is coming.
"""

from __future__ import annotations

from langchain_core.messages import ToolMessage

from config import log
from state import AssistantState


def budget_guard_node(state: AssistantState) -> dict:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    log(f"[GUARD] {len(tool_calls)} bekleyen tool cagrisi butce nedeniyle kapatildi.")

    return {
        "messages": [
            ToolMessage(
                content=(
                    '{"sonuc": "Tool cagri butcesi doldu, bu cagri '
                    "uygulanmadi. Elindeki veriyle devam et; eksik kalan "
                    'bilgiyi eksik olarak bildir."}'
                ),
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
            )
            for tool_call in tool_calls
        ]
    }
