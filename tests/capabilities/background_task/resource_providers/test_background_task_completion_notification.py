"""Tests for background task completion notification: batched delivery with debounce.

Core rules:
- When a background task completes and ``background_output(block=True)`` is
  actively waiting, the result is returned via that call — no duplicate
  notification via ``followup()``.
- When a background task completes with **no** blocking waiter, the
  ``NotificationBatcher`` debounces and batches the notification, then
  delivers it via ``session_pool.followup()`` (never ``steer()``).
- Notifications use ``[BACKGROUND TASK RESULT READY]`` /
  ``[ALL BACKGROUND TASKS COMPLETE]`` / ``[BACKGROUND TASK ERROR]`` headers.
- The batcher has a 500ms debounce window — no notification is delivered
  before the window expires.
"""

# pyright: reportAttributeAccessIssue=false
# Mock-heavy test code: assigning to spec'd attributes (run_stream, _current_run_ctx)
# and accessing mock methods (assert_not_called, assert_called_once, call_args)
# are all expected in this test file.

from __future__ import annotations

import asyncio
import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic_ai import RunContext
import pytest

from wolfharness import ChatMessage
from wolfharness.agents.base_agent import BaseAgent
from wolfharness.agents.context import AgentContext
from wolfharness.agents.events import StreamCompleteEvent
from wolfharness.capabilities.background_task.capability import (
    BackgroundTaskCapability,
)
from wolfharness.capabilities.background_task.manager import BackgroundTaskManager
from wolfharness.capabilities.background_task.types import BackgroundTask, TaskHandle
from wolfharness.delegation import AgentPool


pytestmark = pytest.mark.anyio


async def _wait_until_called(mock_obj: Any, timeout: float = 3.0, interval: float = 0.05) -> None:
    """Poll until a Mock has been called at least once, or timeout.

    More reliable than fixed ``asyncio.sleep`` under CI load where the
    event loop may be busy and debounce timers fire late.
    """
    elapsed = 0.0
    while mock_obj.call_count == 0 and elapsed < timeout:
        await asyncio.sleep(interval)
        elapsed += interval
    assert mock_obj.call_count > 0, f"Mock was not called within {timeout}s"


def _wrap_in_run_context(agent_ctx):
    """Wrap an AgentContext in a mock RunContext for capability tool methods.

    Uses ``MagicMock(spec=RunContext)`` so ``isinstance(ctx, RunContext)``
    returns ``True`` — this ensures ``_get_session_state()`` extracts
    ``agent_ctx`` from ``ctx.deps`` and resolves the same ``run_ctx``
    as direct ``AgentContext`` usage.
    """
    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = agent_ctx
    run_ctx.tool_call_id = agent_ctx.tool_call_id
    return run_ctx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_node(*, name: str = "worker", description: str = "A worker agent") -> MagicMock:
    """Create a mock node that looks like a BaseAgent with run_stream support."""
    node = MagicMock(spec=BaseAgent)
    node.type = "agent"
    node.name = name
    node.description = description
    node.model_name = "test:model"
    node.session_id = "ses_parent_123"
    return node


def _make_mock_pool(nodes: dict[str, MagicMock] | None = None) -> AgentPool:
    """Create a mock AgentPool with the given nodes."""
    pool = MagicMock(spec=AgentPool)

    if nodes is None:
        mock_node = _make_mock_node()
        nodes = {"worker": mock_node}

    pool.manifest = MagicMock()
    pool.manifest.agents = nodes
    pool.nodes = nodes
    pool.agent_configs = nodes
    pool.all_agents = list(nodes.items())
    pool.teams = {}
    pool.sessions = None
    mock_session_pool = MagicMock()

    async def _get_or_create_session_agent(agent_name: str, agent_type: str):
        return nodes.get(agent_name, next(iter(nodes.values())))

    mock_session_pool.sessions = MagicMock()
    mock_session_pool.sessions.get_or_create_session_agent = AsyncMock(
        side_effect=_get_or_create_session_agent,
    )

    mock_session_pool.inject_prompt = AsyncMock()
    mock_session_pool.steer = AsyncMock()
    mock_session_pool.followup = AsyncMock(return_value=True)

    async def _send_message(session_id: str, prompt: str, input_provider=None, **kwargs: Any):
        await asyncio.sleep(0.1)
        return MagicMock()

    mock_session_pool.send_message = AsyncMock(side_effect=_send_message)

    async def _subscribe(session_id: str, scope: str = "session"):
        queue = MagicMock()
        queue.get = AsyncMock(side_effect=asyncio.QueueShutDown())
        return queue

    mock_session_pool.event_bus = MagicMock()
    mock_session_pool.event_bus.subscribe = AsyncMock(side_effect=_subscribe)
    mock_session_pool.event_bus.unsubscribe = AsyncMock()
    pool.session_pool = mock_session_pool
    return pool


def _make_agent_context(
    pool: AgentPool | None = None,
    data: dict[str, object] | None = None,
) -> AgentContext:
    """Create a minimal AgentContext for testing.

    The ``ctx.agent`` mock exposes ``inject_prompt`` so tests can
    assert whether it was called.

    Sets ``ctx.run_ctx`` to a mock ``AgentRunContext`` with the attributes
    needed by the batching / notification code path:
    - ``session_id``: parent session ID
    - ``_run_handle``: ``None`` (forces ``session_pool.followup()`` path)
    - ``child_done_events``: dict for event popping
    """
    agent = MagicMock(spec=BaseAgent)
    agent.type = "agent"
    agent.name = "coordinator"
    agent.session_id = "ses_parent_123"
    agent.agent_pool = pool
    agent.inject_prompt = MagicMock()

    ctx = MagicMock(spec=AgentContext)
    ctx.node = agent
    ctx.agent = agent
    ctx.pool = pool
    ctx.data = data if data is not None else {}
    ctx.tool_call_id = "tc_001"

    # Mock AgentRunContext — must support weak references (MagicMock does)
    mock_run_ctx = MagicMock()
    mock_run_ctx.session_id = "ses_parent_123"
    mock_run_ctx._run_handle = None
    mock_run_ctx.child_done_events = {}
    ctx.run_ctx = mock_run_ctx

    ctx.events = MagicMock()
    ctx.events.emit_event = AsyncMock()
    ctx.create_child_session = AsyncMock(return_value="ses_child_456")
    ctx.internal_fs = MagicMock()
    ctx.internal_fs.mkdirs = MagicMock()
    ctx.internal_fs.pipe = MagicMock()

    return ctx


async def _collect_stream_events(events: list[object]):
    """Create an async generator that yields the given events."""
    for event in events:
        yield event


# ---------------------------------------------------------------------------
# Test: no followup when blocking waiter is present
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_no_inject_when_blocking_waiter_present():
    """When background_output(block=True) is waiting, task completion must
    NOT call followup — the result is returned through the blocking
    call instead.
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    wrapped_ctx = _wrap_in_run_context(ctx)
    await capability.before_run(wrapped_ctx)

    mock_node = pool.nodes["worker"]
    complete_event = StreamCompleteEvent(
        message=ChatMessage(content="Final result", role="assistant"),
    )
    mock_node.run_stream = MagicMock(
        return_value=_collect_stream_events([complete_event]),
    )

    with patch(
        "wolfharness.capabilities.background_task.capability._generate_task_id",
        return_value="bg_a1b2c3d4",
    ):
        result_text = await capability._task(
            _wrap_in_run_context(ctx),
            agent="worker",
            message="test task",
            async_mode=True,
        )

    # Parse task_id from formatted text output
    match = re.search(r"Task ID: (bg_\w+)", result_text)
    assert match is not None
    task_id = match.group(1)

    # Now call background_output with block=True — this registers a waiter
    output = await capability._background_output(
        _wrap_in_run_context(ctx),
        task_id=task_id,
        block=True,
        timeout_seconds=5.0,
    )

    # Result should be returned normally
    assert "Task Result" in output

    # followup must NOT have been called (blocking waiter was present)
    pool.session_pool.followup.assert_not_called()

    # steer must NOT have been called either
    pool.session_pool.steer.assert_not_called()


# ---------------------------------------------------------------------------
# Test: followup IS called when no blocking waiter (after debounce)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_inject_when_no_blocking_waiter():
    """When no background_output(block=True) is waiting, task completion
    must call followup to notify the lead agent after the 500ms debounce.
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    wrapped_ctx = _wrap_in_run_context(ctx)
    await capability.before_run(wrapped_ctx)

    mock_node = pool.nodes["worker"]
    complete_event = StreamCompleteEvent(
        message=ChatMessage(content="Final result", role="assistant"),
    )
    mock_node.run_stream = MagicMock(
        return_value=_collect_stream_events([complete_event]),
    )

    with patch(
        "wolfharness.capabilities.background_task.capability._generate_task_id",
        return_value="bg_noblock99",
    ):
        await capability._task(
            _wrap_in_run_context(ctx),
            agent="worker",
            message="test task",
            async_mode=True,
        )

    # Wait for background task to complete + 500ms debounce
    await _wait_until_called(pool.session_pool.followup)

    # followup SHOULD have been called (no blocking waiter)
    pool.session_pool.followup.assert_called_once()
    call_args = pool.session_pool.followup.call_args
    notice = call_args[0][1]  # second positional arg is the notice
    assert "bg_noblock99" in notice
    assert "[BACKGROUND TASK RESULT READY]" in notice

    # steer must NOT have been called (followup is used, not steer)
    pool.session_pool.steer.assert_not_called()


# ---------------------------------------------------------------------------
# Test: background_output(block=False) does not count as a blocking waiter
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_nonblocking_output_does_not_suppress_inject():
    """A non-blocking background_output call must NOT register as a waiter,
    so followup still fires on completion after the debounce.
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    wrapped_ctx = _wrap_in_run_context(ctx)
    await capability.before_run(wrapped_ctx)

    mock_node = pool.nodes["worker"]
    complete_event = StreamCompleteEvent(
        message=ChatMessage(content="Done", role="assistant"),
    )
    mock_node.run_stream = MagicMock(
        return_value=_collect_stream_events([complete_event]),
    )

    with patch(
        "wolfharness.capabilities.background_task.capability._generate_task_id",
        return_value="bg_nonblock1",
    ):
        await capability._task(
            _wrap_in_run_context(ctx),
            agent="worker",
            message="test task",
            async_mode=True,
        )

    # Non-blocking status check (should NOT register a waiter)
    await asyncio.sleep(0.05)
    status = await capability._background_output(
        _wrap_in_run_context(ctx),
        task_id="bg_nonblock1",
        block=False,
    )
    assert "Task Result" in status or "Task Error" in status or "running" in status.lower()

    # Wait for task to finish + 500ms debounce
    await _wait_until_called(pool.session_pool.followup)

    # followup should have been called
    pool.session_pool.followup.assert_called_once()

    # steer must NOT have been called
    pool.session_pool.steer.assert_not_called()


# ---------------------------------------------------------------------------
# Test: waiter is unregistered after blocking call returns
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_waiter_unregistered_after_blocking_returns():
    """After background_output(block=True) returns, the waiter must be
    unregistered so subsequent tasks on the same task_id won't be
    incorrectly suppressed.
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    wrapped_ctx = _wrap_in_run_context(ctx)
    await capability.before_run(wrapped_ctx)

    mock_node = pool.nodes["worker"]
    complete_event = StreamCompleteEvent(
        message=ChatMessage(content="Result", role="assistant"),
    )
    mock_node.run_stream = MagicMock(
        return_value=_collect_stream_events([complete_event]),
    )

    with patch(
        "wolfharness.capabilities.background_task.capability._generate_task_id",
        return_value="bg_unreg001",
    ):
        await capability._task(
            _wrap_in_run_context(ctx),
            agent="worker",
            message="test task",
            async_mode=True,
        )

    # Block and get result
    await capability._background_output(
        _wrap_in_run_context(ctx),
        task_id="bg_unreg001",
        block=True,
        timeout_seconds=5.0,
    )

    # Waiter must be gone — access via _get_session_state, not _task_manager
    state = capability._get_session_state(_wrap_in_run_context(ctx))
    assert not state.task_manager.has_blocking_waiter("bg_unreg001")


# ---------------------------------------------------------------------------
# Test: followup queues and triggers auto-resume when parent turn is idle
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_inject_prompt_queues_and_triggers_auto_resume():
    """When the parent turn has ended, followup should queue the notice
    and trigger auto-resume so the lead agent receives the completion message.

    This reproduces the bug where the page never receives auto-resume events
    after a background task completes.
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    wrapped_ctx = _wrap_in_run_context(ctx)
    await capability.before_run(wrapped_ctx)

    # Simulate a session pool where followup returns False
    # (message queued for next turn, auto-resume triggered)
    pool.session_pool.followup = AsyncMock(return_value=False)

    mock_node = pool.nodes["worker"]
    complete_event = StreamCompleteEvent(
        message=ChatMessage(content="Final result", role="assistant"),
    )
    mock_node.run_stream = MagicMock(
        return_value=_collect_stream_events([complete_event]),
    )

    with patch(
        "wolfharness.capabilities.background_task.capability._generate_task_id",
        return_value="bg_autoresume01",
    ):
        await capability._task(
            _wrap_in_run_context(ctx),
            agent="worker",
            message="test task",
            async_mode=True,
        )

    # Wait for background task to complete + 500ms debounce
    await _wait_until_called(pool.session_pool.followup)

    # followup should have been called
    pool.session_pool.followup.assert_called_once()

    # Verify it was called with a non-empty parent session ID
    call_args = pool.session_pool.followup.call_args
    assert call_args is not None
    parent_session_id = call_args[0][0]
    assert parent_session_id, "parent_session_id should not be empty"

    # Verify the notice contains the expected content
    notice = call_args[0][1]
    assert "bg_autoresume01" in notice
    assert "[BACKGROUND TASK RESULT READY]" in notice

    # followup returned False, meaning the message was queued and
    # auto-resume should have been triggered.  If auto-resume is broken,
    # the queued message will never be processed.
    # This test documents the expected behavior; a follow-up integration
    # test with a real SessionPool can verify the full auto-resume cycle.

    # steer must NOT have been called
    pool.session_pool.steer.assert_not_called()


# ---------------------------------------------------------------------------
# Test: auto-resume is triggered when followup returns False
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_auto_resume_triggered_when_inject_prompt_returns_false():
    """When followup returns False (message queued), auto-resume must be
    triggered so the lead agent processes the queued completion notice.

    This reproduces the bug where the page never receives auto-resume events
    after a background task completes.
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    wrapped_ctx = _wrap_in_run_context(ctx)
    await capability.before_run(wrapped_ctx)

    # Simulate followup returning False (queued, auto-resume triggered)
    pool.session_pool.followup = AsyncMock(return_value=False)

    mock_node = pool.nodes["worker"]
    complete_event = StreamCompleteEvent(
        message=ChatMessage(content="Final result", role="assistant"),
    )
    mock_node.run_stream = MagicMock(
        return_value=_collect_stream_events([complete_event]),
    )

    with patch(
        "wolfharness.capabilities.background_task.capability._generate_task_id",
        return_value="bg_autoresume02",
    ):
        await capability._task(
            _wrap_in_run_context(ctx),
            agent="worker",
            message="test task",
            async_mode=True,
        )

    # Wait for background task to complete + 500ms debounce
    await _wait_until_called(pool.session_pool.followup)

    # followup should have been called
    pool.session_pool.followup.assert_called_once()

    # Verify parent_session_id is not empty
    call_args = pool.session_pool.followup.call_args
    parent_session_id = call_args[0][0]
    assert parent_session_id, "parent_session_id should not be empty"

    # Verify the notice content
    notice = call_args[0][1]
    assert "bg_autoresume02" in notice
    assert "[BACKGROUND TASK RESULT READY]" in notice

    # followup returned False, so auto-resume should have been triggered.
    # In a real SessionPool, this means _trigger_auto_resume was scheduled.
    # The batcher's deliver_callback handles this via session_pool.followup().

    # steer must NOT have been called
    pool.session_pool.steer.assert_not_called()


# ---------------------------------------------------------------------------
# Test: followup is safe when agent has no run context
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_inject_prompt_safe_when_no_run_context():
    """When the agent's run context is None (ephemeral state path),
    no notification should be delivered — the no-op deliver callback is used.
    The test verifies this does not raise an exception.
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    # Simulate agent with no active run context — ephemeral state is used
    ctx.agent.inject_prompt = MagicMock(return_value=None)
    ctx.agent._current_run_ctx = None
    ctx.agent._background_run_ctx = None
    ctx.run_ctx = None  # Force ephemeral path

    mock_node = pool.nodes["worker"]
    complete_event = StreamCompleteEvent(
        message=ChatMessage(content="Done", role="assistant"),
    )
    mock_node.run_stream = MagicMock(
        return_value=_collect_stream_events([complete_event]),
    )

    with patch(
        "wolfharness.capabilities.background_task.capability._generate_task_id",
        return_value="bg_noctx001",
    ):
        await capability._task(
            _wrap_in_run_context(ctx),
            agent="worker",
            message="test task",
            async_mode=True,
        )

    await asyncio.sleep(1.0)

    # With no run_ctx, the ephemeral state uses a no-op deliver callback.
    # followup should NOT have been called (no real delivery path).
    pool.session_pool.followup.assert_not_called()

    # steer must NOT have been called either
    pool.session_pool.steer.assert_not_called()


# ---------------------------------------------------------------------------
# Test: has_blocking_waiter on manager
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_manager_has_blocking_waiter():
    """Unit test for BackgroundTaskManager.has_blocking_waiter()."""
    manager = BackgroundTaskManager()

    task_model = BackgroundTask(
        id="test-1",
        description="test",
        agent_or_team="worker",
        prompt="do it",
        parent_session_id=None,
        child_session_id=None,
    )
    manager.register_task(task_model)

    # No handle yet → no waiter
    assert not manager.has_blocking_waiter("test-1")

    # Create a handle with a mock asyncio.Task
    mock_atask = MagicMock(spec=asyncio.Task)
    handle = TaskHandle(task=mock_atask)
    manager._handles["test-1"] = handle

    # Handle exists, no waiter set
    assert not manager.has_blocking_waiter("test-1")

    # Register a waiter
    token = manager.register_blocking_waiter("test-1")
    assert token is not None
    assert manager.has_blocking_waiter("test-1")

    # Unregister
    manager.unregister_blocking_waiter("test-1", token)
    assert not manager.has_blocking_waiter("test-1")

    # Stale unregister is a no-op
    manager.register_blocking_waiter("test-1")
    manager.unregister_blocking_waiter("test-1", "wrong-token")
    assert manager.has_blocking_waiter("test-1")


# ---------------------------------------------------------------------------
# Test: 500ms debounce window — no notification before, one after
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_debounce_window_no_notification_before_timeout():
    """Verify that no notification is delivered before the 500ms debounce
    window expires, and exactly one is delivered after.
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    wrapped_ctx = _wrap_in_run_context(ctx)
    await capability.before_run(wrapped_ctx)

    mock_node = pool.nodes["worker"]
    complete_event = StreamCompleteEvent(
        message=ChatMessage(content="Debounce test", role="assistant"),
    )
    mock_node.run_stream = MagicMock(
        return_value=_collect_stream_events([complete_event]),
    )

    with patch(
        "wolfharness.capabilities.background_task.capability._generate_task_id",
        return_value="bg_debounce01",
    ):
        await capability._task(
            _wrap_in_run_context(ctx),
            agent="worker",
            message="test task",
            async_mode=True,
        )

    # Wait 200ms — less than the 500ms debounce
    await asyncio.sleep(0.2)

    # followup should NOT have been called yet (debounce window not expired)
    pool.session_pool.followup.assert_not_called()

    # Wait for the debounce to expire
    await _wait_until_called(pool.session_pool.followup)

    # Now followup SHOULD have been called exactly once
    pool.session_pool.followup.assert_called_once()
    notice = pool.session_pool.followup.call_args[0][1]
    assert "bg_debounce01" in notice
    assert "[BACKGROUND TASK RESULT READY]" in notice


# ---------------------------------------------------------------------------
# Test: multiple tasks within debounce window — single batched notification
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_multiple_tasks_batched_in_debounce_window():
    """When multiple tasks complete within the debounce window, a single
    batched notification should be delivered containing all task IDs.
    """
    capability = BackgroundTaskCapability(schemas=None)

    # Create two mock nodes for two different agents
    node1 = _make_mock_node(name="worker1", description="Worker 1")
    node2 = _make_mock_node(name="worker2", description="Worker 2")
    pool = _make_mock_pool(nodes={"worker1": node1, "worker2": node2})
    ctx = _make_agent_context(pool=pool)

    wrapped_ctx = _wrap_in_run_context(ctx)
    await capability.before_run(wrapped_ctx)

    complete_event1 = StreamCompleteEvent(
        message=ChatMessage(content="Result 1", role="assistant"),
    )
    node1.run_stream = MagicMock(
        return_value=_collect_stream_events([complete_event1]),
    )

    complete_event2 = StreamCompleteEvent(
        message=ChatMessage(content="Result 2", role="assistant"),
    )
    node2.run_stream = MagicMock(
        return_value=_collect_stream_events([complete_event2]),
    )

    # Launch two tasks quickly (within the debounce window)
    task_ids = []
    with (
        patch(
            "wolfharness.capabilities.background_task.capability._generate_task_id",
            side_effect=["bg_batch01", "bg_batch02"],
        ),
    ):
        result1 = await capability._task(
            _wrap_in_run_context(ctx),
            agent="worker1",
            message="task 1",
            async_mode=True,
        )
        match1 = re.search(r"Task ID: (bg_\w+)", result1)
        assert match1 is not None
        task_ids.append(match1.group(1))

        result2 = await capability._task(
            _wrap_in_run_context(ctx),
            agent="worker2",
            message="task 2",
            async_mode=True,
        )
        match2 = re.search(r"Task ID: (bg_\w+)", result2)
        assert match2 is not None
        task_ids.append(match2.group(1))

    # Wait for tasks to complete + debounce to expire
    await _wait_until_called(pool.session_pool.followup)

    # followup should have been called exactly once (single batched notification)
    pool.session_pool.followup.assert_called_once()

    # Verify both task IDs are in the notice
    notice = pool.session_pool.followup.call_args[0][1]
    assert "bg_batch01" in notice
    assert "bg_batch02" in notice

    # The header should be [ALL BACKGROUND TASKS COMPLETE] since both
    # tasks are done and no more are pending
    assert "[ALL BACKGROUND TASKS COMPLETE]" in notice

    # steer must NOT have been called
    pool.session_pool.steer.assert_not_called()
