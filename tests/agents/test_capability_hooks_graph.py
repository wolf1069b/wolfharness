"""Tests verifying pdai Capability hooks fire on graph (BaseTeam) run path.

Phase 6 of thin-wrapper refactor: verifies that Capability node-level hooks
(``wrap_node_run``, ``after_node_run``, ``before_model_request``) fire when
an agent with capabilities runs as part of a BaseTeam.

Node-level hooks are used instead of tool-level hooks (``wrap_tool_execute``)
because ``TestModel`` does not call tools, so tool-level hooks would never
fire in these test fixtures.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
import pytest

from wolfharness.agents.native_agent.agent import Agent
from wolfharness.delegation import BaseTeam


class _NodeHookTrackerCapability(AbstractCapability[Any]):
    """Capability that records when node-level hooks are called."""

    def __init__(self) -> None:
        super().__init__()
        self.wrap_node_run_called = False
        self.after_node_run_called = False
        self.before_model_request_called = False

    async def wrap_node_run(self, ctx: Any, *, node: Any, handler: Any) -> Any:
        self.wrap_node_run_called = True
        return await handler(node)

    async def after_node_run(self, ctx: Any, *, node: Any, result: Any) -> Any:
        self.after_node_run_called = True
        return result

    async def before_model_request(self, ctx: Any, request_context: Any) -> Any:
        self.before_model_request_called = True
        return request_context


pytestmark = [pytest.mark.anyio, pytest.mark.unit]


async def test_capability_hooks_fire_on_graph_run() -> None:
    """Node-level Capability hooks SHALL fire when agent runs inside a BaseTeam."""
    tracker = _NodeHookTrackerCapability()
    model = TestModel(custom_output_text="result")
    agent: Agent[Any, Any] = Agent(
        name="member-agent",
        model=model,
        capabilities=[tracker],
    )
    team = BaseTeam(agents=[agent], mode="sequential")

    async with agent:
        await team.run("hello")

    assert tracker.wrap_node_run_called, (
        "wrap_node_run hook did not fire on graph (BaseTeam) run path"
    )
    assert tracker.after_node_run_called, (
        "after_node_run hook did not fire on graph (BaseTeam) run path"
    )
    assert tracker.before_model_request_called, (
        "before_model_request hook did not fire on graph (BaseTeam) run path"
    )


async def test_capability_hooks_fire_on_sequential_chain() -> None:
    """Node-level hooks SHALL fire on each member of a sequential chain."""
    tracker_a = _NodeHookTrackerCapability()
    tracker_b = _NodeHookTrackerCapability()
    model = TestModel(custom_output_text="result")

    agent_a: Agent[Any, Any] = Agent(
        name="agent-a",
        model=model,
        capabilities=[tracker_a],
    )
    agent_b: Agent[Any, Any] = Agent(
        name="agent-b",
        model=model,
        capabilities=[tracker_b],
    )
    team = BaseTeam(agents=[agent_a, agent_b], mode="sequential")

    async with agent_a, agent_b:
        await team.run("hello")

    assert tracker_a.wrap_node_run_called, "agent-a wrap_node_run not called in chain"
    assert tracker_a.after_node_run_called, "agent-a after_node_run not called in chain"
    assert tracker_b.wrap_node_run_called, "agent-b wrap_node_run not called in chain"
    assert tracker_b.after_node_run_called, "agent-b after_node_run not called in chain"
