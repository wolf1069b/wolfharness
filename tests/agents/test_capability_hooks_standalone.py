"""Tests verifying pdai Capability hooks fire on standalone run path.

Phase 2 of thin-wrapper refactor: BaseAgent.run_stream() now delegates
directly to _stream_events() → NativeTurn.execute()
which calls agent_run.next(node) explicitly, ensuring all pdai Capability
hooks fire on every run path.

This test verifies both run-level (`wrap_run`) and node-level
(`wrap_node_run`, `before_model_request`, `after_node_run`) hooks fire.
The node-level hooks are the core differentiator of Phase 2: they require
`agent_run.next(node)` to be called explicitly (rather than a bare
`async for` over the agent run, which would skip node-level hooks).
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.capabilities import AbstractCapability
import pytest

from wolfharness.agents.native_agent.agent import Agent
from wolfharness.models.agents import NativeAgentConfig


class HookTrackerCapability(AbstractCapability[Any]):
    """Capability that records when run- and node-level hooks are called."""

    def __init__(self) -> None:
        super().__init__()
        self.wrap_run_called = False
        self.wrap_node_run_called = False
        self.before_model_request_called = False
        self.after_node_run_called = False

    async def wrap_run(self, ctx: Any, *, handler: Any) -> Any:
        self.wrap_run_called = True
        return await handler()

    async def wrap_node_run(self, ctx: Any, *, node: Any, handler: Any) -> Any:
        self.wrap_node_run_called = True
        return await handler(node)

    async def before_model_request(self, ctx: Any, request_context: Any) -> Any:
        self.before_model_request_called = True
        return request_context

    async def after_node_run(self, ctx: Any, *, node: Any, result: Any) -> Any:
        self.after_node_run_called = True
        return result


pytestmark = [pytest.mark.unit, pytest.mark.anyio]


async def test_capability_hooks_fire_on_standalone_run() -> None:
    """All Capability hooks SHALL fire on standalone run_stream().

    Verifies the core Phase 2 invariant: `agent_run.next(node)` is invoked
    on the standalone path, triggering node-level hooks. A bare
    `async for` over the agent run would skip these hooks.
    """
    tracker = HookTrackerCapability()
    config = NativeAgentConfig(
        name="test-agent",
        model="test:test",
        system_prompt="You are a test agent.",
        capabilities=[tracker],
    )
    agent: Agent[Any, Any] = Agent(
        name="test-agent",
        model="test:test",
        agent_config=config,
    )

    async with agent:
        async for _event in agent.run_stream("Hello"):
            pass

    assert tracker.wrap_run_called, "wrap_run hook did not fire on standalone run_stream() path"
    assert tracker.wrap_node_run_called, (
        "wrap_node_run hook did not fire on standalone run_stream() path — "
        "agent_run.next(node) may not be called on this path"
    )
    assert tracker.before_model_request_called, (
        "before_model_request hook did not fire on standalone run_stream() path"
    )
    assert tracker.after_node_run_called, (
        "after_node_run hook did not fire on standalone run_stream() path"
    )
