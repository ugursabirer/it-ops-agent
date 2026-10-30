"""Human approval gate in front of every write action.

The assistant may read freely, but it may not change the corporate system
without a person saying yes. The gate uses LangGraph's ``interrupt``: the graph
pauses, the pending tool call is shown to the user, and execution resumes only
after an answer arrives. A rejection still has to answer the pending tool call,
otherwise the model would be left with an unresolved call in its history.
"""

from __future__ import annotations

from langchain_core.messages import ToolMessage
from langgraph.types import interrupt

from config import log, settings
from state import AssistantState

APPROVED = {"onay", "onayla", "evet", "e", "y", "yes", "ok", "tamam"}


def approval_node(state: AssistantState) -> dict:
    """Pause for approval, then record the outcome in state."""

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    if not tool_calls:
        return {"approved": False}

    tool_call = tool_calls[0]

    if not settings.require_approval:
        log("[ONAY] IT_OPS_REQUIRE_APPROVAL=false, aksiyon onaysiz calisiyor.")
        return {"approved": True}

    answer = interrupt(
        {
            "tur": "onay_gerekiyor",
            "tool": tool_call["name"],
            "parametreler": tool_call["args"],
        }
    )

    approved = str(answer).strip().lower() in APPROVED

    if approved:
        log("[ONAY] Kullanici onayladi, aksiyon uygulaniyor.")
        return {"approved": True}

    log("[ONAY] Kullanici reddetti, aksiyon uygulanmadi.")

    return {
        "approved": False,
        "messages": [
            ToolMessage(
                content=(
                    '{"sonuc": "Kullanici onayi verilmedigi icin talep '
                    'acilmadi. Sistemde hicbir degisiklik yapilmadi."}'
                ),
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
            )
        ],
    }
