"""Shared LangGraph state for the IT operations assistant.

Built on ``MessagesState`` so the conversation history is appended
automatically; every other field is optional and written by exactly one node.
"""

from __future__ import annotations

from typing import NotRequired

from langgraph.graph import MessagesState


class AssistantState(MessagesState):
    """State carried through the graph.

    route:       supervisor's routing decision for this turn.
    plan:        planner's step list, used by the gathering agents as a brief.
    last_agent:  which specialist answered last, so the supervisor can resolve
                 short follow-up messages against the previous turn.
    policy_loops / ops_loops:
                 per-agent tool-call counters. They are separate on purpose: a
                 single shared counter let the policy agent's turns use up the
                 operations agent's budget, and the operations agent was cut off
                 before it could fetch the data the decision depends on.
    approved:    result of the human approval gate for a write action.
    """

    route: NotRequired[str]
    plan: NotRequired[list[str]]
    last_agent: NotRequired[str]
    policy_loops: NotRequired[int]
    ops_loops: NotRequired[int]
    approved: NotRequired[bool]
