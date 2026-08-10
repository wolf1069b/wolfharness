"""Tests for RunLoopDelegationService race condition fix (2nd round review).

Verifies that spawn_subagent() subscribes to the EventBus instead of
calling run_handle.start("") a second time, which would race with the
background _consume_run task started by send_message().
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock
import warnings

import pytest

from wolfharness.agents.events.events import StreamCompleteEvent
from wolfharness.capabilities.runloop_delegation import RunLoopDelegationService
from wolfharness.host.context import HostContext
from wolfharness.host.registry import AgentRegistry
from wolfharness.orchestrator.core import EventBus
from wolfharness.orchestrator.run import RunHandle


pytestmark = pytest.mark.unit


def _make_host_context(session_pool: Any = None) -> HostContext:
    """Build a HostContext with minimal stubs for testing."""
    return HostContext(
        manifest=MagicMock(),
        storage=MagicMock(),
        vfs_registry=MagicMock(),
        connection_registry=MagicMock(),
        mcp=MagicMock(),
        skills_registry=MagicMock(),
        prompt_manager=MagicMock(),
        process_manager=MagicMock(),
        file_ops=MagicMock(),
        todos=MagicMock(),
        session_pool=session_pool,
        config_file_path=None,
    )


async def _drain_generator(gen: Any, collected: list[Any]) -> None:
    """Drain an async generator into a list."""
    collected.extend([event async for event in gen])


@pytest.mark.anyio
async def test_spawn_subagent_uses_event_bus_not_start() -> None:
    """spawn_subagent must subscribe to EventBus, not call run_handle.start().

    The bug: send_message() starts a background _consume_run that calls
    run_handle.start(content). Then spawn_subagent calls run_handle.start("")
    on the SAME RunHandle — two concurrent start() calls corrupt state.

    The fix: subscribe to the EventBus and yield events from the queue
    instead of calling start("").
    """
    # --- Arrange ---
    event_bus = EventBus()

    # Mock session_pool with send_message returning a message_id
    session_pool = MagicMock()
    session_pool.event_bus = event_bus
    session_pool.send_message = AsyncMock(return_value="msg-123")

    # Mock controller with get_session returning a session with current_run_id
    mock_session = MagicMock()
    mock_session.current_run_id = "run-abc"
    controller = MagicMock()
    controller.get_session = MagicMock(return_value=mock_session)

    # Mock RunHandle — we do NOT want start() to be called
    mock_run_handle = MagicMock(spec=RunHandle)
    mock_run_handle.run_id = "run-abc"
    mock_run_handle.start = MagicMock(side_effect=AssertionError("start() must not be called"))
    controller._runs = {"run-abc": mock_run_handle}
    session_pool.sessions = controller

    host = _make_host_context(session_pool=session_pool)
    registry = AgentRegistry({"test-agent": MagicMock()})
    svc = RunLoopDelegationService(registry, host, "parent-session")

    # --- Act ---
    # We need to run spawn_subagent as an async generator and push an event
    # onto the EventBus so it can complete.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        gen = svc.spawn_subagent("test-agent", "do something")

    # Start consuming in a task so we can push events concurrently
    collected_events: list[Any] = []
    task = asyncio.create_task(_drain_generator(gen, collected_events))

    # Give the subscription a moment to register
    await asyncio.sleep(0.05)

    # Push a StreamCompleteEvent onto the EventBus for the child session
    child_session_id = "parent-session::child::test-agent"
    complete_event = StreamCompleteEvent(
        message=MagicMock(),
        session_id=child_session_id,
    )
    await event_bus.publish(child_session_id, complete_event)

    # Wait for the task to complete
    await asyncio.wait_for(task, timeout=5.0)

    # --- Assert ---
    # 1. start() was never called on the RunHandle
    mock_run_handle.start.assert_not_called()

    # 2. Events were received from the EventBus
    assert len(collected_events) >= 1
    assert any(isinstance(e, StreamCompleteEvent) for e in collected_events)

    # 3. send_message was called with the correct child session ID
    session_pool.send_message.assert_awaited_once()
    call_kwargs = session_pool.send_message.await_args
    assert call_kwargs.kwargs["session_id"] == child_session_id


@pytest.mark.anyio
async def test_spawn_subagent_no_start_when_no_message_id() -> None:
    """When send_message returns None, spawn_subagent returns early."""
    session_pool = MagicMock()
    session_pool.event_bus = EventBus()
    session_pool.send_message = AsyncMock(return_value=None)
    session_pool.sessions = MagicMock()

    host = _make_host_context(session_pool=session_pool)
    registry = AgentRegistry({"test-agent": MagicMock()})
    svc = RunLoopDelegationService(registry, host, "parent-session")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        gen = svc.spawn_subagent("test-agent", "do something")

    events: list[Any] = [event async for event in gen]
    assert events == []


@pytest.mark.anyio
async def test_spawn_subagent_unsubscribes_after_completion() -> None:
    """spawn_subagent must unsubscribe from EventBus after terminal event."""
    event_bus = EventBus()

    session_pool = MagicMock()
    session_pool.event_bus = event_bus
    session_pool.send_message = AsyncMock(return_value="msg-123")

    mock_session = MagicMock()
    mock_session.current_run_id = "run-abc"
    controller = MagicMock()
    controller.get_session = MagicMock(return_value=mock_session)
    mock_run_handle = MagicMock(spec=RunHandle)
    mock_run_handle.run_id = "run-abc"
    controller._runs = {"run-abc": mock_run_handle}
    session_pool.sessions = controller

    host = _make_host_context(session_pool=session_pool)
    registry = AgentRegistry({"test-agent": MagicMock()})
    svc = RunLoopDelegationService(registry, host, "parent-session")

    child_session_id = "parent-session::child::test-agent"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        gen = svc.spawn_subagent("test-agent", "do something")

    collected: list[Any] = []
    task = asyncio.create_task(_drain_generator(gen, collected))
    await asyncio.sleep(0.05)

    complete_event = StreamCompleteEvent(
        message=MagicMock(),
        session_id=child_session_id,
    )
    await event_bus.publish(child_session_id, complete_event)
    await asyncio.wait_for(task, timeout=5.0)

    # After completion, the subscriber queue should be removed from EventBus
    assert (
        child_session_id not in event_bus._subscribers
        or len(event_bus._subscribers[child_session_id]) == 0
    )
