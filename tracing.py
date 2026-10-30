"""Terminal tracing helpers.

The assistant's value is in its intermediate steps, so the steps are shown
rather than hidden: which agent ran, which tool it called, and what it concluded
before handing over to the next agent.
"""

from __future__ import annotations

from typing import Any

from config import log

PREVIEW_CHARS = 220


def preview(message: Any) -> None:
    """Log a short preview of an agent's own output.

    Tool calls are shown by name; free text is truncated, because the point is
    to make the handover visible, not to reprint the whole answer.
    """

    tool_calls = getattr(message, "tool_calls", None) or []

    if tool_calls:
        names = ", ".join(call["name"] for call in tool_calls)
        log(f"         -> tool cagrisi: {names}")
        return

    content = str(getattr(message, "content", "") or "").strip()

    if not content:
        log("         -> (bos cevap)")
        return

    flattened = " ".join(content.split())
    shown = (
        flattened
        if len(flattened) <= PREVIEW_CHARS
        else f"{flattened[:PREVIEW_CHARS]}..."
    )

    log(f"         -> {shown}")
