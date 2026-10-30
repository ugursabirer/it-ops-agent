"""Graph routing and the guards around the write action.

These are the rules that must hold regardless of what the model produces, so
they are tested directly rather than through a model call.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

import graph as graph_module
from graph import (
    MAX_OPS_LOOPS,
    MAX_POLICY_LOOPS,
    approval_router,
    budget_router,
    decision_router,
    ops_router,
    policy_router,
    supervisor_router,
)


def _ai(content: str = "", tool_calls: list[dict] | None = None) -> AIMessage:
    return AIMessage(content=content, tool_calls=tool_calls or [])


def _call(name: str, **args: object) -> dict:
    return {"name": name, "args": args, "id": f"call-{name}"}


# ------------------------------------------------------------------ supervisor


def test_supervisor_router_uses_the_chosen_lane():
    assert supervisor_router({"route": "operasyon", "messages": []}) == "operasyon"


def test_supervisor_router_falls_back_to_policy():
    assert supervisor_router({"messages": []}) == "politika"


# ------------------------------------------------------------ gathering lanes


def test_policy_lane_runs_tools_when_requested():
    state = {
        "messages": [_ai(tool_calls=[_call("search_it_policy", query="x")])],
        "route": "politika",
        "policy_loops": 1,
    }

    assert policy_router(state) == "tools"


def test_policy_lane_ends_a_simple_question():
    state = {"messages": [_ai("cevap")], "route": "politika"}

    assert policy_router(state) == "bitir"


def test_policy_lane_continues_to_ops_for_a_request():
    state = {"messages": [_ai("kurallar")], "route": "talep"}

    assert policy_router(state) == "devam"


def test_spent_policy_budget_closes_the_pending_call():
    """A pending call must be answered, not carried into the next agent."""

    state = {
        "messages": [_ai(tool_calls=[_call("search_it_policy", query="x")])],
        "route": "talep",
        "policy_loops": MAX_POLICY_LOOPS + 1,
    }

    assert policy_router(state) == "butce"


def test_spent_ops_budget_closes_the_pending_call():
    state = {
        "messages": [_ai(tool_calls=[_call("get_employee_assets", employee_id="E-1")])],
        "route": "talep",
        "ops_loops": MAX_OPS_LOOPS + 1,
    }

    assert ops_router(state) == "butce"


def test_ops_budget_is_independent_of_policy_budget():
    """The shared counter was the bug: it cut data gathering short."""

    state = {
        "messages": [_ai(tool_calls=[_call("find_employee", query="Mert")])],
        "route": "talep",
        "policy_loops": MAX_POLICY_LOOPS + 5,
        "ops_loops": 1,
    }

    assert ops_router(state) == "tools"


# ------------------------------------------------------------- budget guard


def test_budget_guard_hands_policy_lane_to_ops():
    state = {"route": "talep", "last_agent": "politika", "messages": []}

    assert budget_router(state) == "ops"


def test_budget_guard_hands_ops_lane_to_decision():
    state = {"route": "talep", "last_agent": "operasyon", "messages": []}

    assert budget_router(state) == "karar"


def test_budget_guard_ends_a_simple_lane():
    state = {"route": "operasyon", "last_agent": "operasyon", "messages": []}

    assert budget_router(state) == "bitir"


# ---------------------------------------------------------- the write action


def test_a_real_write_call_goes_to_the_approval_gate():
    state = {
        "messages": [_ai(tool_calls=[_call("create_ticket", employee_id="E-1003")])]
    }

    assert decision_router(state) == "onay"


def test_an_invented_tool_name_never_reaches_the_approval_gate():
    """Observed in a real run: the model called a tool it did not have."""

    state = {"messages": [_ai(tool_calls=[_call("search_employee", query="x")])]}

    assert decision_router(state) == "gecersiz"


def test_a_write_call_mixed_with_an_invented_one_is_refused():
    state = {
        "messages": [
            _ai(
                tool_calls=[
                    _call("create_ticket", employee_id="E-1003"),
                    _call("search_employee", query="x"),
                ]
            )
        ]
    }

    assert decision_router(state) == "gecersiz"


def test_a_decision_without_a_tool_call_goes_straight_to_the_report():
    state = {"messages": [_ai("KARAR: UYGUN DEGIL")]}

    assert decision_router(state) == "rapor"


def test_approval_router_only_acts_when_approved():
    assert approval_router({"approved": True, "messages": []}) == "aksiyon"
    assert approval_router({"approved": False, "messages": []}) == "rapor"
    assert approval_router({"messages": []}) == "rapor"


# ------------------------------------------------------------- guard nodes


def test_invalid_tool_node_answers_every_pending_call():
    from agents.invalid_tool_node import invalid_tool_node

    state = {
        "messages": [
            HumanMessage(content="soru"),
            _ai(tool_calls=[_call("search_employee", query="x")]),
        ]
    }

    result = invalid_tool_node(state)

    assert len(result["messages"]) == 1
    assert result["messages"][0].tool_call_id == "call-search_employee"
    assert result["approved"] is False


def test_budget_guard_node_answers_every_pending_call():
    from agents.budget_guard_node import budget_guard_node

    state = {
        "messages": [
            _ai(
                tool_calls=[
                    _call("get_employee_assets", employee_id="E-1"),
                    _call("get_spare_stock", asset_type="dizustu"),
                ]
            )
        ]
    }

    result = budget_guard_node(state)

    assert len(result["messages"]) == 2
    assert {message.tool_call_id for message in result["messages"]} == {
        "call-get_employee_assets",
        "call-get_spare_stock",
    }


def test_write_tool_names_are_known_to_the_router():
    """The approval gate is keyed on this set; an empty set would open it."""

    assert "create_ticket" in graph_module.WRITE_TOOL_NAMES
    assert len(graph_module.WRITE_TOOL_NAMES) == 1
