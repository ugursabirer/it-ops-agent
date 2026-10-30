"""Tool loading, including the MCP transport and its local fallback.

The assistant reads policy documents through in-process RAG tools and reaches
operational data through the custom MCP server in ``ops_mcp/server.py``.
``ops_toolset`` is the single place that decides which transport is used, so the
rest of the application never has to care which one is active.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from config import log, settings
from tools.ops_tools import OPS_TOOLS, READ_TOOLS, WRITE_TOOLS
from tools.policy_tools import POLICY_TOOLS

SERVER_SCRIPT = Path(__file__).resolve().parent.parent / "ops_mcp" / "server.py"

# The agent must never reach a write without passing the approval gate, so the
# write tool is kept in a node of its own and identified by name.
WRITE_TOOL_NAMES = {"create_ticket"}


@dataclass
class OpsToolset:
    """Operational tools plus a human-readable note on where they came from."""

    read_tools: list[BaseTool]
    write_tools: list[BaseTool]
    transport: str
    note: str

    @property
    def all_tools(self) -> list[BaseTool]:
        return [*self.read_tools, *self.write_tools]


def _flatten(result: Any) -> str:
    """Normalise an MCP result into plain text.

    MCP returns a list of typed content blocks. Handing that structure straight
    to a 4B model costs tokens and invites confusion, so the text blocks are
    joined into the same flat JSON string the local tools already return.
    """

    if isinstance(result, str):
        return result

    if isinstance(result, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in result
        ]
        return "\n".join(part for part in parts if part)

    return str(result)


def _wrap(mcp_tool: BaseTool) -> BaseTool:
    """Re-expose an MCP tool with flattened output and a trace line."""

    async def _call(**kwargs: Any) -> str:
        shown = ", ".join(f"{key}={value!r}" for key, value in kwargs.items())
        log(f"  [MCP] {mcp_tool.name}({shown})")

        try:
            raw = await mcp_tool.ainvoke(kwargs)
        except Exception as error:  # noqa: BLE001 - surfaced to the model
            return (
                f'{{"hata": "MCP tool cagrisi basarisiz oldu.", "ayrinti": "{error}"}}'
            )

        return _flatten(raw)

    return StructuredTool(
        name=mcp_tool.name,
        description=mcp_tool.description,
        args_schema=mcp_tool.args_schema,
        coroutine=_call,
    )


def _local_toolset(note: str) -> OpsToolset:
    return OpsToolset(
        read_tools=list(READ_TOOLS),
        write_tools=list(WRITE_TOOLS),
        transport="dogrudan SQLite (yedek yol)",
        note=note,
    )


@asynccontextmanager
async def ops_toolset() -> AsyncIterator[OpsToolset]:
    """Yield the operational toolset, preferring MCP over the local fallback.

    The MCP session is held open for the lifetime of the context so the server
    runs as a single child process instead of being restarted on every call.
    A failure to start it is not fatal: the assistant degrades to the direct
    SQLite path and says so, rather than refusing to answer at all.

    The ``yield`` sits outside the ``try`` on purpose. Inside it, an exception
    raised by the caller's own body would travel back into the generator, be
    caught here, and be misread as an MCP failure.
    """

    if not settings.use_mcp:
        yield _local_toolset("IT_OPS_USE_MCP=false oldugu icin MCP atlandi.")
        return

    async with AsyncExitStack() as stack:
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            from langchain_mcp_adapters.tools import load_mcp_tools

            client = MultiServerMCPClient(
                {
                    "ops": {
                        "command": sys.executable,
                        "args": [str(SERVER_SCRIPT)],
                        "transport": "stdio",
                    }
                }
            )

            session = await stack.enter_async_context(client.session("ops"))
            mcp_tools = [_wrap(tool) for tool in await load_mcp_tools(session)]

            write_tools = [tool for tool in mcp_tools if tool.name in WRITE_TOOL_NAMES]

            if not write_tools:
                raise RuntimeError("MCP server create_ticket tool'unu sunmuyor.")

            toolset = OpsToolset(
                read_tools=[
                    tool for tool in mcp_tools if tool.name not in WRITE_TOOL_NAMES
                ],
                write_tools=write_tools,
                transport="MCP / stdio (ops_mcp/server.py)",
                note=f"{len(mcp_tools)} MCP tool yuklendi.",
            )

        except Exception as error:  # noqa: BLE001 - degradation is intentional
            log(f"[UYARI] MCP kullanilamadi ({error}). Yedek yola geciliyor.")
            toolset = _local_toolset(f"MCP baslatilamadi: {error}")

        yield toolset


__all__ = [
    "OPS_TOOLS",
    "POLICY_TOOLS",
    "WRITE_TOOL_NAMES",
    "OpsToolset",
    "ops_toolset",
]
