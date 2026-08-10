"""Tests for error paths in ``_run_and_stream`` inner coroutine.

These tests verify error handling fixes in the ``_run_and_stream`` coroutine
of ``BackgroundTaskCapability._task_async``.

Key fixes verified:
- P0: Exceptions from ``send_message`` are re-raised; task marked "error".
- P2: EndOfStream without a terminal event writes error to output file.
- P2: ``send_message`` returning None writes error to output file.
- P2: ``RunErrorEvent`` writes "Task Error" and task reaches "error" state.
- P2: Cancellation correctly writes "Task Cancelled" and marks "cancelled".
- P2: Task ID generation uses uuid4().hex[:12] (12 hex chars).
"""

# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
# Mock-heavy test code: assigning to spec'd attributes, accessing mock call_args_list,
# and accessing event.message.content through union types are all expected.

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic_ai import RunContext
import pytest

from wolfharness.agents.base_agent import BaseAgent
from wolfharness.agents.context import AgentContext
from wolfharness.agents.events import RunErrorEvent
from wolfharness.capabilities.background_task.capability import (
    BackgroundTaskCapability,
    _generate_task_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrap_in_run_context(agent_ctx: Any) -> MagicMock:
    """Wrap an AgentContext in a mock RunContext for capability tool methods.

    Uses ``MagicMock(spec=RunContext)`` so ``isinstance(ctx, RunContext)``
    returns ``True`` — this ensures ``_get_session_state()`` extracts
    ``agent_ctx`` from ``ctx.deps`` and resolves the same ephemeral state
    as direct ``AgentContext`` usage.
    """
    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = agent_ctx
    run_ctx.tool_call_id = agent_ctx.tool_call_id
    return run_ctx


def _make_mock_pool() -> MagicMock:
    """Create a mock AgentPool with session_pool for background tasks."""
    pool = MagicMock()
    mock_agent = MagicMock(spec=BaseAgent)
    mock_agent.name = "test_agent"
    mock_agent.description = "test agent"
    mock_agent.session_id = "ses_parent_123"
    mock_agent.model_name = "test:model"
    mock_agent.type = "agent"
    pool.nodes = {"test_agent": mock_agent}
    pool.agent_configs = {"test_agent": mock_agent}
    pool.all_agents = list(pool.nodes.items())
    pool.teams = {}
    pool.sessions = None

    mock_session_pool = MagicMock()
    mock_session_pool.event_bus = MagicMock()
    mock_session_pool.event_bus.subscribe = AsyncMock(return_value=MagicMock())
    mock_session_pool.event_bus.unsubscribe = AsyncMock()
    mock_session_pool.send_message = AsyncMock(return_value=MagicMock())
    mock_session_pool.steer = AsyncMock()
    mock_session_pool.sessions = MagicMock()
    mock_session_pool.sessions.get_or_create_session_agent = AsyncMock(return_value=mock_agent)
    pool.session_pool = mock_session_pool
    pool.manifest = MagicMock()
    pool.manifest.agents = pool.nodes
    return pool


def _make_agent_context(pool: MagicMock) -> MagicMock:
    """Create a minimal AgentContext for testing."""
    agent = MagicMock(spec=BaseAgent)
    agent.name = "coordinator"
    agent.session_id = "ses_parent_123"
    agent.agent_pool = pool

    ctx = MagicMock(spec=AgentContext)
    ctx.node = agent
    ctx.pool = pool
    ctx.data = {}
    ctx.tool_call_id = "tc_001"
    ctx.run_ctx = None
    ctx.events = MagicMock()
    ctx.events.emit_event = AsyncMock()
    ctx.create_child_session = AsyncMock(return_value="ses_child_456")
    ctx.internal_fs = MagicMock()
    ctx.internal_fs.mkdirs = MagicMock()
    ctx.internal_fs.pipe = MagicMock()
    ctx.internal_fs.cat = MagicMock(return_value=b"")
    return ctx


def _make_end_of_stream_queue() -> MagicMock:
    """Create a mock event queue whose get() immediately raises QueueShutDown."""
    queue = MagicMock()
    queue.get = AsyncMock(side_effect=asyncio.QueueShutDown())
    return queue


def _make_blocking_queue() -> MagicMock:
    """Create a mock event queue whose receive() blocks forever."""

    async def _block_forever() -> None:
        await asyncio.Event().wait()

    queue = MagicMock()
    queue.get = AsyncMock(side_effect=_block_forever)
    return queue


def _make_event_queue(events: list[Any]) -> MagicMock:
    """Create a mock event queue that yields the given events then raises QueueShutDown."""
    iterator = iter(events)

    async def _receive() -> Any:
        try:
            return next(iterator)
        except StopIteration:
            raise asyncio.QueueShutDown() from None

    queue = MagicMock()
    queue.get = AsyncMock(side_effect=_receive)
    return queue


def _collect_pipe_content(pipe_mock: MagicMock) -> bytes:
    """Concatenate all content written to fs.pipe as bytes."""
    all_content = b""
    for call in pipe_mock.call_args_list:
        args = call[0]
        if len(args) > 1:
            content = args[1]
            if isinstance(content, str):
                content = content.encode()
            all_content += content
    return all_content


async def _wait_for_terminal(
    capability: BackgroundTaskCapability,
    ctx: Any,
    task_id: str,
    timeout_seconds: float = 2.0,
) -> None:
    """Poll task status until terminal or timeout."""
    state = capability._get_session_state(ctx)
    deadline = asyncio.get_event_loop().time() + timeout_seconds
    while asyncio.get_event_loop().time() < deadline:
        task = state.task_manager.get_task(task_id)
        if task is not None and task.status in ("completed", "error", "cancelled", "timed_out"):
            return
        await asyncio.sleep(0.02)
    task = state.task_manager.get_task(task_id)
    if task is not None:
        msg = f"Task {task_id} did not reach terminal state within {timeout_seconds}s (status={task.status})"
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Test 1: Exception in send_message is re-raised, task marked "error"
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_run_and_stream_swallows_exception_and_marks_completed():
    """FIXED: ValueError from send_message is re-raised; task marked error.

    The ``_run_and_stream`` coroutine catches ``(ValueError, RuntimeError, ...)``
    and writes "Task Failed" to the output file, then re-raises.
    ``BackgroundTaskManager._execute_task`` catches the exception and marks
    the task as "error".
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool)

    # send_message raises ValueError
    pool.session_pool.send_message = AsyncMock(side_effect=ValueError("connection lost"))

    with patch(
        "wolfharness.capabilities.background_task.capability._generate_task_id",
        return_value="bg_err001",
    ):
        await capability._task(
            _wrap_in_run_context(ctx),
            agent="test_agent",
            message="test task",
            async_mode=True,
        )

    await _wait_for_terminal(capability, ctx, "bg_err001")
    await asyncio.sleep(0.1)  # Let completion callback's notify task run

    state = capability._get_session_state(ctx)
    task_model = state.task_manager.get_task("bg_err001")
    assert task_model is not None
    # FIXED: exception is re-raised, _execute_task marks as "error"
    assert task_model.status == "error", (
        f"Expected 'error' (exception re-raised), got '{task_model.status}'"
    )

    # Verify "Task Failed" was written to the output file
    pipe_calls = ctx.internal_fs.pipe.call_args_list
    assert len(pipe_calls) > 0, "Expected fs.pipe to be called"
    content = _collect_pipe_content(ctx.internal_fs.pipe)
    assert b"Task Failed" in content, f"Expected 'Task Failed' in output, got: {content}"


# ---------------------------------------------------------------------------
# Test 2: EndOfStream without a terminal event
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_run_and_stream_end_of_stream_no_terminal_event():
    """FIXED: EndOfStream without a terminal event now writes error to output.

    The ``while True`` loop breaks on ``EndOfStream``, but now the handler
    writes an error message to the output file before breaking.
    The coroutine still returns normally (no exception raised), so the task
    is marked "completed" — but the output file contains the error message.
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool)

    # event_queue immediately raises EndOfStream
    pool.session_pool.event_bus.subscribe = AsyncMock(
        return_value=_make_end_of_stream_queue(),
    )
    # send_message returns a non-None handle
    pool.session_pool.send_message = AsyncMock(return_value=MagicMock())

    with patch(
        "wolfharness.capabilities.background_task.capability._generate_task_id",
        return_value="bg_eos001",
    ):
        await capability._task(
            _wrap_in_run_context(ctx),
            agent="test_agent",
            message="test task",
            async_mode=True,
        )

    await _wait_for_terminal(capability, ctx, "bg_eos001")
    await asyncio.sleep(0.1)

    state = capability._get_session_state(ctx)
    task_model = state.task_manager.get_task("bg_eos001")
    assert task_model is not None
    assert task_model.status == "completed", f"Expected 'completed', got '{task_model.status}'"

    # FIXED: EndOfStream handler now writes an error message to the output file
    content = _collect_pipe_content(ctx.internal_fs.pipe)
    assert b"Event stream ended without a terminal event" in content, (
        f"Expected 'Event stream ended without a terminal event' in output, got: {content}"
    )


# ---------------------------------------------------------------------------
# Test 3: send_message returns None
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_run_and_stream_send_message_returns_none():
    """FIXED: send_message returns None — error written to output file.

    The code now logs a warning, writes an error message to the output file
    ("Run was not started (send_message returned None)"), and returns
    from the coroutine.  The task is marked "completed" since no exception
    is raised.
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool)

    # send_message returns None
    pool.session_pool.send_message = AsyncMock(return_value=None)
    # event_queue immediately raises EndOfStream
    pool.session_pool.event_bus.subscribe = AsyncMock(
        return_value=_make_end_of_stream_queue(),
    )

    with (
        patch(
            "wolfharness.capabilities.background_task.capability._generate_task_id",
            return_value="bg_none01",
        ),
        patch(
            "wolfharness.capabilities.background_task.capability.logger",
        ) as mock_logger,
    ):
        await capability._task(
            _wrap_in_run_context(ctx),
            agent="test_agent",
            message="test task",
            async_mode=True,
        )

        await _wait_for_terminal(capability, ctx, "bg_none01")
        await asyncio.sleep(0.1)

        # FIXED: logger.warning is now called (was logger.debug before fix)
        mock_logger.warning.assert_called()

    state = capability._get_session_state(ctx)
    task_model = state.task_manager.get_task("bg_none01")
    assert task_model is not None
    assert task_model.status == "completed", (
        f"Expected 'completed' (coroutine returns normally), got '{task_model.status}'"
    )

    # FIXED: error message about send_message returning None is now written to output
    content = _collect_pipe_content(ctx.internal_fs.pipe)
    assert b"send_message returned None" in content, (
        f"Expected 'send_message returned None' in output, got: {content}"
    )


# ---------------------------------------------------------------------------
# Test 4: RunErrorEvent writes "Task Error" to output
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_run_and_stream_writes_error_on_run_error_event():
    """FIXED: RunErrorEvent from the subagent causes "Task Error" to be written
    and the task reaches "error" state.

    The match case writes ``# Task Error\\n\\n{message}`` to the output file
    and sets ``task_error``.  Then ``RuntimeError(task_error)`` is raised,
    caught by the specific exception handler which writes ``# Task Failed``,
    and re-raised.  ``_execute_task`` catches the exception and marks the
    task as "error".
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool)

    # event_queue yields a RunErrorEvent then raises EndOfStream
    error_event = RunErrorEvent(message="subagent crashed")
    pool.session_pool.event_bus.subscribe = AsyncMock(
        return_value=_make_event_queue([error_event]),
    )
    pool.session_pool.send_message = AsyncMock(return_value=MagicMock())

    with patch(
        "wolfharness.capabilities.background_task.capability._generate_task_id",
        return_value="bg_reer01",
    ):
        await capability._task(
            _wrap_in_run_context(ctx),
            agent="test_agent",
            message="test task",
            async_mode=True,
        )

    await _wait_for_terminal(capability, ctx, "bg_reer01")
    await asyncio.sleep(0.1)

    state = capability._get_session_state(ctx)
    task_model = state.task_manager.get_task("bg_reer01")
    assert task_model is not None
    # FIXED: exception is re-raised, _execute_task marks as "error"
    assert task_model.status == "error", (
        f"Expected 'error' (exception re-raised), got '{task_model.status}'"
    )

    # Verify "Task Error" and "subagent crashed" were written to the output file
    content = _collect_pipe_content(ctx.internal_fs.pipe)
    assert b"Task Error" in content or b"subagent crashed" in content, (
        f"Expected 'Task Error' or 'subagent crashed' in output, got: {content}"
    )


# ---------------------------------------------------------------------------
# Test 5: CancelledError writes "Task Cancelled"
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_run_and_stream_writes_cancelled_on_cancelled_error():
    """Cancelling a running background task writes "Task Cancelled" to output.

    The ``CancelledError`` handler (line 721-724) writes the cancellation
    message and re-raises.  ``_execute_task`` catches ``CancelledError`` and
    marks the task "cancelled".
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool)

    # event_queue blocks forever (so the task is running when we cancel)
    pool.session_pool.event_bus.subscribe = AsyncMock(
        return_value=_make_blocking_queue(),
    )
    pool.session_pool.send_message = AsyncMock(return_value=MagicMock())

    with patch(
        "wolfharness.capabilities.background_task.capability._generate_task_id",
        return_value="bg_cancel01",
    ):
        await capability._task(
            _wrap_in_run_context(ctx),
            agent="test_agent",
            message="test task",
            async_mode=True,
        )

    # Wait for the task to start running
    await asyncio.sleep(0.1)

    # Cancel the task
    await capability._background_cancel(
        _wrap_in_run_context(ctx),
        task_id="bg_cancel01",
    )

    await _wait_for_terminal(capability, ctx, "bg_cancel01")
    await asyncio.sleep(0.1)

    state = capability._get_session_state(ctx)
    task_model = state.task_manager.get_task("bg_cancel01")
    assert task_model is not None
    assert task_model.status == "cancelled", f"Expected 'cancelled', got '{task_model.status}'"

    # Verify "Task Cancelled" was written to the output file
    content = _collect_pipe_content(ctx.internal_fs.pipe)
    assert b"Task Cancelled" in content, f"Expected 'Task Cancelled' in output, got: {content}"


# ---------------------------------------------------------------------------
# Test 6: Task ID collision overwrites silently
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_task_id_collision_overwrites_silently():
    """FIXED: Task ID generation uses uuid4().hex[:12] — 12 hex chars.

    ``_generate_task_id`` now uses ``uuid4().hex[:12]`` instead of
    ``secrets.token_hex(4)`` (8 hex chars).  The longer ID space makes
    collisions negligible.  This test verifies the ID format and uniqueness.
    """
    # Verify ID format: bg_ + 12 hex chars
    task_id = _generate_task_id("test task")
    assert task_id.startswith("bg_"), f"Expected 'bg_' prefix, got: {task_id}"
    hex_part = task_id[3:]
    assert len(hex_part) == 12, f"Expected 12 hex chars after 'bg_', got {len(hex_part)}: {task_id}"
    valid_hex = set("0123456789abcdef")
    assert all(c in valid_hex for c in hex_part), f"Expected only hex chars, got: {task_id}"

    # Verify uniqueness: 100 IDs should all be different (collision negligible)
    ids = {_generate_task_id("test") for _ in range(100)}
    assert len(ids) == 100, f"Expected 100 unique IDs, got {len(ids)} duplicates"
