"""Test fixtures.

Everything in this suite runs offline: the graph's routing logic and the data
layer are plain Python, so they can be tested without Ollama, without the MCP
server and without touching the demo database.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

BASE_PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_PATH))


@pytest.fixture(scope="session")
def ops_database(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Seed a throwaway database and point the data layer at it.

    IT_OPS_DB_PATH is set before seeding, so the project's own
    ops_mcp/it_ops.db is never touched by a test run.
    """

    target = tmp_path_factory.mktemp("ops") / "it_ops.db"
    os.environ["IT_OPS_DB_PATH"] = str(target)

    from ops_mcp.create_database import create

    create()

    return target
