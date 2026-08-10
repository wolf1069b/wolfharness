"""Tests for BackgroundTaskProvider background_output() and background_cancel() tools."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import MagicMock

from pydantic_ai import RunContext
import pytest
from upathtools.filesystems.isolated_memory_fs import IsolatedMemoryFileSystem

from wolfharness.agents.base_agent import BaseAgent
from wolfharness.agents.context import AgentContext
from wolfharness.capabilities.background_task.capability import (
    BackgroundTaskCapability,
)
from wolfharness.capabilities.background_task.types import BackgroundTask, TaskStatus
from wolfharness.tools.exceptions import ToolError


def _wrap_in_run_context(agent_ctx):
    """Wrap an AgentContext in a mock RunContext for capability tool methods.

    Uses ``MagicMock(spec=RunContext)`` so ``isinstance(ctx, RunContext)``
    returns ``True`` — this ensures ``_get_session_state()`` resolves the
    same ``run_ctx`` as direct ``AgentContext`` usage.
    """
    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = agent_ctx
    run_ctx.tool_call_id = agent_ctx.tool_call_id
    return run_ctx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_context() -> AgentContext:
    """Create a minimal AgentContext for testing."""
    agent = MagicMock(spec=BaseAgent)
    agent.name = "coordinator"
    agent.session_id = "ses_parent_001"

    ctx = MagicMock(spec=AgentContext)
    ctx.node = agent
    ctx.pool = None
    ctx.data = {}
    ctx.tool_call_id = "tc_output_001"
    # Provide a real internal_fs for output_file reading tests
    ctx.internal_fs = IsolatedMemoryFileSystem()

    # Mock AgentRunContext for session state resolution
    mock_run_ctx = MagicMock()
    mock_run_ctx.session_id = "ses_parent_001"
    mock_run_ctx._run_handle = None
    mock_run_ctx.child_done_events = {}
    ctx.run_ctx = mock_run_ctx

    return ctx


def _get_task_manager(capability: BackgroundTaskCapability, ctx: AgentContext):
    """Get the task manager from the capability's session state.

    Replaces direct ``_get_task_manager(capability, ctx)`` access (which no longer
    exists — task manager is per-session via ``_get_session_state()``).
    """
    return capability._get_session_state(_wrap_in_run_context(ctx)).task_manager


def _make_terminal_task(
    task_id: str = "bg_test000",
    status: TaskStatus = "completed",
    result: str | None = None,
    error: str | None = None,
    completed_at: datetime | None = None,
    description: str = "test task",
) -> BackgroundTask:
    """Create a BackgroundTask in a terminal state, pre-registered in the manager."""
    return BackgroundTask(
        id=task_id,
        description=description,
        agent_or_team="worker",
        prompt="do something",
        parent_session_id="ses_parent",
        child_session_id="ses_child",
        status=status,
        result=result,
        error=error,
        completed_at=completed_at or datetime.now(tz=UTC),
        started_at=datetime.now(tz=UTC),
    )


# ---------------------------------------------------------------------------
# background_output: non-existent task
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_background_output_nonexistent_task():
    """Test that background_output returns 'not found' for non-existent task_id."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()

    result = await capability._background_output(
        _wrap_in_run_context(ctx), task_id="nonexistent-id"
    )

    assert "not found" in result
    assert "nonexistent-id" in result


# ---------------------------------------------------------------------------
# background_output: running task with block=False
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_background_output_running_task_block_false():
    """Test that background_output returns status-only for running task with block=False."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()

    # Register a task and start it with a long-running coroutine
    task_model = BackgroundTask(
        id="bg_running1",
        description="running task",
        agent_or_team="worker",
        prompt="do something slow",
        parent_session_id="ses_parent",
        child_session_id="ses_child",
    )
    _get_task_manager(capability, ctx).register_task(task_model)

    # Start the task with a long-running coroutine
    async def _slow_coro() -> None:
        await asyncio.sleep(100)

    _get_task_manager(capability, ctx).start_task("bg_running1", _slow_coro())

    # Give the event loop a chance to start the task
    await asyncio.sleep(0.1)

    result = await capability._background_output(
        _wrap_in_run_context(ctx), task_id="bg_running1", block=False
    )

    # Should be status-only, no partial output
    assert "running" in result
    assert "Use block=True" in result
    # Should NOT contain "Result:" (no partial output)
    assert "Result:" not in result

    # Clean up
    await _get_task_manager(capability, ctx).cancel_all()


# ---------------------------------------------------------------------------
# background_output: pending task with block=False
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_background_output_pending_task_block_false():
    """Test that background_output returns status-only for pending task with block=False."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()

    # Register a task but don't start it (stays pending)
    task_model = BackgroundTask(
        id="bg_pending1",
        description="pending task",
        agent_or_team="worker",
        prompt="do something",
        parent_session_id="ses_parent",
        child_session_id="ses_child",
        status="pending",
    )
    _get_task_manager(capability, ctx).register_task(task_model)

    result = await capability._background_output(
        _wrap_in_run_context(ctx), task_id="bg_pending1", block=False
    )

    assert "pending" in result
    assert "Not yet started" in result
    assert "Use block=True" in result


# ---------------------------------------------------------------------------
# background_output: completed task
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_background_output_completed_task():
    """Test that background_output returns result for completed task."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()

    completed_at = datetime(2025, 4, 25, 12, 5, 0, tzinfo=UTC)
    task_model = _make_terminal_task(
        task_id="bg_done0011",
        status="completed",
        result="Analysis complete: motor bearing worn",
        completed_at=completed_at,
    )
    _get_task_manager(capability, ctx).register_task(task_model)

    result = await capability._background_output(_wrap_in_run_context(ctx), task_id="bg_done0011")

    assert "Task Result" in result
    assert "Analysis complete: motor bearing worn" in result


# ---------------------------------------------------------------------------
# background_output: error task
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_background_output_error_task():
    """Test that background_output returns error info for errored task."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()

    task_model = _make_terminal_task(
        task_id="bg_error001",
        status="error",
        error="ValueError: invalid input",
        completed_at=datetime.now(tz=UTC),
    )
    _get_task_manager(capability, ctx).register_task(task_model)

    result = await capability._background_output(_wrap_in_run_context(ctx), task_id="bg_error001")

    assert "error" in result
    assert "ValueError: invalid input" in result


# ---------------------------------------------------------------------------
# background_output: cancelled task
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_background_output_cancelled_task():
    """Test that background_output returns cancellation info for cancelled task."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()

    task_model = _make_terminal_task(
        task_id="bg_cancel01",
        status="cancelled",
        completed_at=datetime.now(tz=UTC),
    )
    _get_task_manager(capability, ctx).register_task(task_model)

    result = await capability._background_output(_wrap_in_run_context(ctx), task_id="bg_cancel01")

    assert "Task Cancelled" in result


# ---------------------------------------------------------------------------
# background_output: timed out task
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_background_output_timed_out_task():
    """Test that background_output returns timeout info for timed-out task."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()

    task_model = _make_terminal_task(
        task_id="bg_timeout1",
        status="timed_out",
        error="Task timed out after 1800s",
        completed_at=datetime.now(tz=UTC),
    )
    _get_task_manager(capability, ctx).register_task(task_model)

    result = await capability._background_output(_wrap_in_run_context(ctx), task_id="bg_timeout1")

    assert "Task Error" in result
    assert "Task timed out after 1800s" in result


# ---------------------------------------------------------------------------
# background_output: block=True waits for completion
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_background_output_block_true_waits_for_completion():
    """Test that background_output with block=True waits and returns result."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()

    # Register a task that completes quickly
    task_model = BackgroundTask(
        id="bg_quick001",
        description="quick task",
        agent_or_team="worker",
        prompt="do something quick",
        parent_session_id="ses_parent",
        child_session_id="ses_child",
    )
    _get_task_manager(capability, ctx).register_task(task_model)

    # Start with a fast coroutine
    async def _fast_coro() -> None:
        pass  # Completes immediately

    _get_task_manager(capability, ctx).start_task("bg_quick001", _fast_coro())

    # Use block=True with a reasonable timeout
    result = await capability._background_output(
        _wrap_in_run_context(ctx), task_id="bg_quick001", block=True
    )

    assert "Task Result" in result


# ---------------------------------------------------------------------------
# background_output: completed task with no result
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_background_output_completed_task_no_result():
    """Test that background_output handles completed task with no result."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()

    task_model = _make_terminal_task(
        task_id="bg_norslt1",
        status="completed",
        result=None,
        completed_at=datetime.now(tz=UTC),
    )
    _get_task_manager(capability, ctx).register_task(task_model)

    result = await capability._background_output(_wrap_in_run_context(ctx), task_id="bg_norslt1")

    assert "Task Result" in result
    assert "No result available" in result


# ---------------------------------------------------------------------------
# background_cancel: no arguments raises ToolError
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_background_cancel_no_args_raises_tool_error():
    """Test that background_cancel raises ToolError when neither task_id nor cancel_all is provided."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()

    with pytest.raises(ToolError, match="Either task_id or cancel_all=True must be provided"):
        await capability._background_cancel(
            _wrap_in_run_context(ctx), task_id=None, cancel_all=False
        )


# ---------------------------------------------------------------------------
# background_cancel: both task_id and cancel_all raises ToolError
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_background_cancel_both_args_raises_tool_error():
    """Test that background_cancel raises ToolError when both task_id and cancel_all are provided."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()

    with pytest.raises(ToolError, match="Cannot specify both task_id and cancel_all=True"):
        await capability._background_cancel(
            _wrap_in_run_context(ctx), task_id="some-task", cancel_all=True
        )


# ---------------------------------------------------------------------------
# background_cancel: cancel_all cancels non-terminal tasks
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_background_cancel_cancel_all():
    """Test that background_cancel with cancel_all=True cancels all non-terminal tasks."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()

    # Register and start multiple tasks
    for i in range(3):
        task_model = BackgroundTask(
            id=f"bg_multi{i:02d}",
            description=f"task {i}",
            agent_or_team="worker",
            prompt=f"do something {i}",
            parent_session_id="ses_parent",
            child_session_id="ses_child",
        )
        _get_task_manager(capability, ctx).register_task(task_model)

        async def _long_coro() -> None:
            await asyncio.sleep(100)

        _get_task_manager(capability, ctx).start_task(f"bg_multi{i:02d}", _long_coro())

    # Give the event loop a chance to start the tasks
    await asyncio.sleep(0.1)

    result = await capability._background_cancel(_wrap_in_run_context(ctx), cancel_all=True)

    assert "Cancelled 3 background task(s)" in result


# ---------------------------------------------------------------------------
# background_cancel: cancel single task
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_background_cancel_single_task():
    """Test that background_cancel cancels a single task by ID."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()

    # Register and start a task
    task_model = BackgroundTask(
        id="bg_single01",
        description="single task",
        agent_or_team="worker",
        prompt="do something",
        parent_session_id="ses_parent",
        child_session_id="ses_child",
    )
    _get_task_manager(capability, ctx).register_task(task_model)

    async def _long_coro() -> None:
        await asyncio.sleep(100)

    _get_task_manager(capability, ctx).start_task("bg_single01", _long_coro())

    # Give the event loop a chance to start the task
    await asyncio.sleep(0.1)

    result = await capability._background_cancel(_wrap_in_run_context(ctx), task_id="bg_single01")

    # Verify formatted output
    assert isinstance(result, str)
    assert len(result) > 0
    assert "bg_single01" in result


# ---------------------------------------------------------------------------
# background_cancel: cancel non-existent task
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_background_cancel_nonexistent_task():
    """Test that background_cancel returns 'not found' for non-existent task."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()

    result = await capability._background_cancel(
        _wrap_in_run_context(ctx), task_id="nonexistent-id"
    )

    assert "not found" in result
    assert "nonexistent-id" in result


# ---------------------------------------------------------------------------
# background_cancel: cancel already terminal task
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_background_cancel_already_completed_task():
    """Test that background_cancel handles already-completed task gracefully."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()

    # Register a completed task
    task_model = _make_terminal_task(
        task_id="bg_alrdydn1",
        status="completed",
        result="Done already",
        completed_at=datetime.now(tz=UTC),
    )
    _get_task_manager(capability, ctx).register_task(task_model)

    result = await capability._background_cancel(_wrap_in_run_context(ctx), task_id="bg_alrdydn1")

    # Manager returns a message about the task already being in terminal state
    assert "already" in result


# ---------------------------------------------------------------------------
# background_cancel: cancel_all with mixed terminal and non-terminal tasks
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_background_cancel_cancel_all_mixed_states():
    """Test that cancel_all only cancels non-terminal tasks, not terminal ones."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()

    # Register a completed task (terminal)
    completed_task = _make_terminal_task(
        task_id="bg_comp0001",
        status="completed",
        result="Already done",
        completed_at=datetime.now(tz=UTC),
    )
    _get_task_manager(capability, ctx).register_task(completed_task)

    # Register and start a running task
    running_task = BackgroundTask(
        id="bg_run00001",
        description="running task",
        agent_or_team="worker",
        prompt="do something",
        parent_session_id="ses_parent",
        child_session_id="ses_child",
    )
    _get_task_manager(capability, ctx).register_task(running_task)

    async def _long_coro() -> None:
        await asyncio.sleep(100)

    _get_task_manager(capability, ctx).start_task("bg_run00001", _long_coro())

    await asyncio.sleep(0.1)

    result = await capability._background_cancel(_wrap_in_run_context(ctx), cancel_all=True)

    # Should only cancel the 1 running task, not the completed one
    assert "Cancelled 1 background task(s)" in result


# ---------------------------------------------------------------------------
# background_cancel: pending task cancellation
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_background_cancel_pending_task():
    """Test that background_cancel cancels a pending task (not yet started)."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()

    # Register a task but don't start it (stays pending)
    task_model = BackgroundTask(
        id="bg_pendcan",
        description="pending task",
        agent_or_team="worker",
        prompt="do something",
        parent_session_id="ses_parent",
        child_session_id="ses_child",
        status="pending",
    )
    _get_task_manager(capability, ctx).register_task(task_model)

    result = await capability._background_cancel(_wrap_in_run_context(ctx), task_id="bg_pendcan")

    assert "cancelled" in result.lower()


# ---------------------------------------------------------------------------
# _format_terminal_task_output: direct unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_format_terminal_task_output_completed_with_result():
    """Test _format_terminal_task_output for completed task with result."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()
    task = _make_terminal_task(
        task_id="bg_test0001",
        status="completed",
        result="The answer is 42",
        completed_at=datetime(2025, 4, 25, 12, 0, 0, tzinfo=UTC),
    )
    output = capability._format_terminal_task_output(ctx, task)
    assert "Task Result" in output
    assert "bg_test0001" in output
    assert "The answer is 42" in output


@pytest.mark.unit
def test_format_terminal_task_output_error():
    """Test _format_terminal_task_output for error task."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()
    task = _make_terminal_task(
        task_id="bg_test0002",
        status="error",
        error="RuntimeError: boom",
        completed_at=datetime(2025, 4, 25, 12, 0, 0, tzinfo=UTC),
    )
    output = capability._format_terminal_task_output(ctx, task)
    assert "Task Error" in output
    assert "RuntimeError: boom" in output


@pytest.mark.unit
def test_format_terminal_task_output_cancelled():
    """Test _format_terminal_task_output for cancelled task."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()
    task = _make_terminal_task(
        task_id="bg_test0003",
        status="cancelled",
        completed_at=datetime(2025, 4, 25, 12, 0, 0, tzinfo=UTC),
    )
    output = capability._format_terminal_task_output(ctx, task)
    assert "Task Cancelled" in output


@pytest.mark.unit
def test_format_terminal_task_output_timed_out():
    """Test _format_terminal_task_output for timed-out task."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()
    task = _make_terminal_task(
        task_id="bg_test0004",
        status="timed_out",
        error="Task timed out after 60s",
        completed_at=datetime(2025, 4, 25, 12, 0, 0, tzinfo=UTC),
    )
    output = capability._format_terminal_task_output(ctx, task)
    assert "Task Error" in output
    assert "Task timed out after 60s" in output


# ---------------------------------------------------------------------------
# _format_terminal_task_output: reads from output_file when result is None
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_format_terminal_task_output_reads_output_file_when_result_none():
    """Test that _format_terminal_task_output reads from internal_fs when result is None but output_file exists."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()

    # Write content to the internal_fs at the output_file path
    output_path = "/tasks/bg_testid1/output.md"
    ctx.internal_fs.mkdirs("/tasks/bg_testid1", exist_ok=True)
    ctx.internal_fs.pipe(output_path, b"Analysis complete: motor bearing worn")

    task = _make_terminal_task(
        task_id="bg_testid1",
        status="completed",
        result=None,
        completed_at=datetime(2025, 4, 25, 12, 0, 0, tzinfo=UTC),
    )
    task.output_file = output_path

    output = capability._format_terminal_task_output(ctx, task)
    assert "Task Result" in output
    assert "Analysis complete: motor bearing worn" in output


@pytest.mark.unit
def test_format_terminal_task_output_no_result_no_output_file():
    """Test that _format_terminal_task_output returns 'No result available' when both result and output_file are None."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()

    task = _make_terminal_task(
        task_id="bg_norsltn1",
        status="completed",
        result=None,
        completed_at=datetime(2025, 4, 25, 12, 0, 0, tzinfo=UTC),
    )
    # output_file is None by default

    output = capability._format_terminal_task_output(ctx, task)
    assert "Task Result" in output
    assert "No result available" in output


@pytest.mark.unit
def test_format_terminal_task_output_output_file_read_fails_gracefully():
    """Test that _format_terminal_task_output falls back gracefully when internal_fs read fails."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()

    task = _make_terminal_task(
        task_id="bg_failsaf1",
        status="completed",
        result=None,
        completed_at=datetime(2025, 4, 25, 12, 0, 0, tzinfo=UTC),
    )
    task.output_file = "/tasks/bg_failsaf1/nonexistent.md"

    output = capability._format_terminal_task_output(ctx, task)
    assert "Task Result" in output
    assert "No result available" in output


# ---------------------------------------------------------------------------
# background_output: timeout_seconds parameter
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_background_output_timeout_seconds_default():
    """Test that background_output passes timeout_seconds to wait_for_task."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()

    # Register a task that completes quickly
    task_model = BackgroundTask(
        id="bg_quick0001",
        description="quick task",
        agent_or_team="worker",
        prompt="do something quick",
        parent_session_id="ses_parent",
        child_session_id="ses_child",
    )
    _get_task_manager(capability, ctx).register_task(task_model)

    async def _fast_coro() -> None:
        pass

    _get_task_manager(capability, ctx).start_task("bg_quick0001", _fast_coro())

    # Use block=True with default timeout_seconds (60.0)
    result = await capability._background_output(
        _wrap_in_run_context(ctx), task_id="bg_quick0001", block=True
    )

    assert "Task Result" in result


@pytest.mark.unit
async def test_background_output_timeout_seconds_custom():
    """Test that background_output accepts a custom timeout_seconds value."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()

    # Register a task that completes quickly
    task_model = BackgroundTask(
        id="bg_quick0002",
        description="quick task",
        agent_or_team="worker",
        prompt="do something quick",
        parent_session_id="ses_parent",
        child_session_id="ses_child",
    )
    _get_task_manager(capability, ctx).register_task(task_model)

    async def _fast_coro() -> None:
        pass

    _get_task_manager(capability, ctx).start_task("bg_quick0002", _fast_coro())

    # Use block=True with custom timeout_seconds
    result = await capability._background_output(
        _wrap_in_run_context(ctx),
        task_id="bg_quick0002",
        block=True,
        timeout_seconds=120.0,
    )

    assert "Task Result" in result


# ---------------------------------------------------------------------------
# background_output: block=True timeout returns status, does NOT cancel task
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_background_output_block_true_timeout_returns_status_not_cancel():
    """Test that background_output(block=True) on timeout returns a timeout/status
    response and does NOT cancel the task.

    Regression test: the task must remain running after a blocking wait times
    out, and the response must not misleadingly format it as cancelled.
    """
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()

    # Register a task that takes longer than our wait timeout
    task_model = BackgroundTask(
        id="bg_slow0001",
        description="slow task",
        agent_or_team="worker",
        prompt="do something slow",
        parent_session_id="ses_parent",
        child_session_id="ses_child",
    )
    _get_task_manager(capability, ctx).register_task(task_model)

    async def _slow_coro() -> None:
        await asyncio.sleep(10)

    _get_task_manager(capability, ctx).start_task("bg_slow0001", _slow_coro())

    # Give the event loop a chance to start the task
    await asyncio.sleep(0.1)

    # Block with a short timeout — should time out
    result = await capability._background_output(
        _wrap_in_run_context(ctx),
        task_id="bg_slow0001",
        block=True,
        timeout_seconds=0.2,
    )

    # The response must indicate the wait timed out, NOT that the task was cancelled
    assert "still running" in result
    assert "timed out" in result or "waiting" in result
    assert "cancelled" not in result
    assert "continues running" in result

    # Verify the task is still running (not cancelled)
    task_after = _get_task_manager(capability, ctx).get_task("bg_slow0001")
    assert task_after is not None
    assert task_after.status not in ("cancelled", "error", "timed_out")

    # Verify the task can still complete normally — cancel the slow sleep
    # and let the task finish
    await _get_task_manager(capability, ctx).cancel_all()

    # After cancellation, the task should now be in a terminal state
    await asyncio.sleep(0.1)
    task_final = _get_task_manager(capability, ctx).get_task("bg_slow0001")
    if task_final is not None:
        assert task_final.status in ("cancelled", "completed", "error", "timed_out")


@pytest.mark.unit
async def test_background_output_block_true_timeout_then_task_completes():
    """Test that after a blocking wait times out, the task can still complete
    and its output can be retrieved with a subsequent background_output call.
    """
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()

    # Register a task that completes after a short delay
    task_model = BackgroundTask(
        id="bg_delay001",
        description="delayed task",
        agent_or_team="worker",
        prompt="do something delayed",
        parent_session_id="ses_parent",
        child_session_id="ses_child",
        output_file="/tasks/bg_delay001/output.md",
    )
    _get_task_manager(capability, ctx).register_task(task_model)

    # Write result to output file so _format_terminal_task_output can read it
    ctx.internal_fs.mkdirs("/tasks/bg_delay001", exist_ok=True)
    ctx.internal_fs.pipe("/tasks/bg_delay001/output.md", b"Delayed result: bearing worn")

    async def _delayed_coro() -> None:
        await asyncio.sleep(0.3)

    _get_task_manager(capability, ctx).start_task("bg_delay001", _delayed_coro())

    # Give the event loop a chance to start the task
    await asyncio.sleep(0.05)

    # Block with a very short timeout — should time out
    result = await capability._background_output(
        _wrap_in_run_context(ctx),
        task_id="bg_delay001",
        block=True,
        timeout_seconds=0.1,
    )

    # Must be a timeout/status response, not terminal
    assert "still running" in result
    assert "cancelled" not in result

    # Now wait longer and the task should complete
    result2 = await capability._background_output(
        _wrap_in_run_context(ctx),
        task_id="bg_delay001",
        block=True,
        timeout_seconds=5.0,
    )

    assert "Task Result" in result2
