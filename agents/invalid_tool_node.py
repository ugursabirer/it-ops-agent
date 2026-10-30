"""Answers tool calls that name a tool the agent does not actually have.

A small model occasionally invents a plausible-sounding tool name. Two things
must then happen, and neither can be skipped: the pending call has to be
answered (an unanswered tool call corrupts the message history for the next
model call), and the invented call must not be mistaken for the real write
action waiting at the approval gate.
"""

from __future__ import annotations

from langchain_core.messages import ToolMessage

from config import log
from state import AssistantState


def invalid_tool_node(state: AssistantState) -> dict:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    answers = []

    for tool_call in tool_calls:
        log(f"[GUARD] Tanimsiz tool cagrisi reddedildi: {tool_call['name']}")

        answers.append(
            ToolMessage(
                content=(
                    '{"hata": "Boyle bir tool bulunmuyor, cagri '
                    'uygulanmadi.", "istenen_tool": '
                    f'"{tool_call["name"]}"}}'
                ),
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
            )
        )

    return {"messages": answers, "approved": False}
