"""Unit tests for Task 12: Deprecation + migration.

Tests that DeprecationWarning is emitted by:
- RunLoopDelegationService.spawn_subagent()
- RunLoopDelegationService.get_available_agents()

And that SubagentCapability uses run_agent() when session_pool
is available, falling back to delegation when it is None.

Note: ``receive_request()`` deprecation tests have been removed since
the method was deleted in Phase 6.4 of session-debt-cleanup.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
import warnings

import pytest

from wolfharness.capabilities.runloop_delegation import RunLoopDelegationService


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. RunLoopDelegationService.spawn_subagent() deprecation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_delegation_service_spawn_subagent_deprecation() -> None:
    """RunLoopDelegationService.spawn_subagent() emits DeprecationWarning."""
    from wolfharness.agents.events.events import StreamCompleteEvent
    from wolfharness.orchestrator.core import EventBus

    registry = MagicMock()
    registry.exists = MagicMock(return_value=True)
    host = MagicMock()
    host.session_pool = MagicMock()
    host.session_pool.sessions = MagicMock()
    # Use a real EventBus so subscribe/unsubscribe work properly.
    event_bus = EventBus()
    host.session_pool.event_bus = event_bus
    # Make send_message return a truthy message_id so spawn proceeds.
    host.session_pool.send_message = AsyncMock(return_value="mid")

    service = RunLoopDelegationService(registry, host, "parent-sess")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gen = service.spawn_subagent("agent1", "do something")

        # Drain the generator in a task so we can push an event.
        import asyncio

        drained: list[Any] = []
        task = asyncio.create_task(_drain(gen, drained))
        await asyncio.sleep(0.05)

        # Push a terminal event so the generator completes.
        child_session_id = "parent-sess::child::agent1"
        complete = StreamCompleteEvent(
            message=MagicMock(),
            session_id=child_session_id,
        )
        await event_bus.publish(child_session_id, complete)
        await asyncio.wait_for(task, timeout=5.0)

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) >= 1
    assert "spawn_subagent" in str(dep_warnings[0].message).lower()


async def _drain(gen: Any, collected: list[Any]) -> None:
    """Drain an async generator into a list."""
    collected.extend([event async for event in gen])


# ---------------------------------------------------------------------------
# 2. RunLoopDelegationService.get_available_agents() deprecation
# ---------------------------------------------------------------------------


def test_delegation_service_get_available_agents_deprecation() -> None:
    """RunLoopDelegationService.get_available_agents() emits DeprecationWarning."""
    registry = MagicMock()
    registry.list_names = MagicMock(return_value=["agent1", "agent2"])
    host = MagicMock()
    service = RunLoopDelegationService(registry, host, "sess-1")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = service.get_available_agents()

    assert result == ["agent1", "agent2"]
    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) >= 1
    assert "get_available_agents" in str(dep_warnings[0].message).lower()


# ---------------------------------------------------------------------------
# 3. SubagentCapability uses run_agent() when session_pool available
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_subagent_capability_uses_run_agent() -> None:
    """SubagentCapability.spawn_subagent calls run_agent() when session_pool is available."""
    from wolfharness.capabilities.agent_context import AgentContextDeps
    from wolfharness.capabilities.subagent_capability import SubagentCapability

    # Build a mock AgentContextDeps with a non-None session_pool.
    session_pool = MagicMock()
    session_pool.run_agent = AsyncMock(return_value="subagent result")

    agent_registry = MagicMock()
    agent_registry.list_names = MagicMock(return_value=["a", "b"])

    host = MagicMock()
    host.session_pool = session_pool

    session = MagicMock()
    session.session_id = "parent-session"

    delegation = MagicMock()

    agent_ctx = AgentContextDeps(
        agent_registry=agent_registry,
        delegation=delegation,
        session=session,
        scope=MagicMock(),
        host=host,
    )

    # Build a mock RunContext.
    ctx = MagicMock()
    ctx.deps = agent_ctx

    result = await SubagentCapability.spawn_subagent(ctx, "worker", "do task")

    assert result == "subagent result"
    session_pool.run_agent.assert_awaited_once_with(
        "worker",
        "do task",
        parent_session_id="parent-session",
    )
    # DelegationService should NOT be called.
    delegation.spawn_subagent.assert_not_called()


# ---------------------------------------------------------------------------
# 4. SubagentCapability falls back to delegation when session_pool is None
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_subagent_capability_fallback() -> None:
    """SubagentCapability falls back to delegation when session_pool is None."""

    async def _mock_stream(name: str, prompt: str) -> Any:
        yield "chunk1"
        yield "chunk2"

    from wolfharness.capabilities.agent_context import AgentContextDeps
    from wolfharness.capabilities.subagent_capability import SubagentCapability

    host = MagicMock()
    host.session_pool = None

    session = MagicMock()
    session.session_id = "parent-session"

    delegation = MagicMock()
    delegation.spawn_subagent = _mock_stream

    agent_registry = MagicMock()
    agent_registry.list_names = MagicMock(return_value=["a"])

    agent_ctx = AgentContextDeps(
        agent_registry=agent_registry,
        delegation=delegation,
        session=session,
        scope=MagicMock(),
        host=host,
    )

    ctx = MagicMock()
    ctx.deps = agent_ctx

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await SubagentCapability.spawn_subagent(ctx, "worker", "do task")

    assert result == "chunk1\nchunk2"
    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) >= 1
    assert "fell back" in str(dep_warnings[0].message).lower()
