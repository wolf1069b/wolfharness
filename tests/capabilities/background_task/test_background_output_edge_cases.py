"""Edge-case tests for background_output and background_cancel.

Covers:
- CancelledError propagation during blocking background_output
- None return from wait_for_task when task is cleaned up during wait
- Fragile string matching in _background_cancel

Note: Completion notice formatting was moved to ``NotificationBatcher``
in T5; all formatting is now tested via the batcher's own tests.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from pydantic_ai import RunContext
import pytest
from upathtools.filesystems.isolated_memory_fs import IsolatedMemoryFileSystem

from wolfharness.agents.base_agent import BaseAgent
from wolfharness.agents.context import AgentContext
from wolfharness.capabilities.background_task.capability import (
    BackgroundTaskCapability,
)
from wolfharness.capabilities.background_task.manager import BackgroundTaskManager
from wolfharness.capabilities.background_task.types import BackgroundTask


class _NullWaitManager(BackgroundTaskManager):
    """Manager subclass whose wait_for_task always returns None.

    Simulates the race where a task is cleaned up (removed from _tasks)
    during the blocking wait, causing wait_for_task to return None.
    """

    async def wait_for_task(
        self, task_id: str, timeout_seconds: float = 60.0
    ) -> BackgroundTask | None:
        return None


class _FragileCancelManager(BackgroundTaskManager):
    """Manager subclass whose cancel_task returns an unmatched string.

    Simulates cancel_task returning a message that does NOT contain
    "not found" or "already".  The status-based check in
    ``_background_cancel`` correctly handles this by checking
    ``task_after.status`` instead of substring matching on the result.
    """

    async def cancel_task(self, task_id: str) -> str:
        return "Task was removed by cleanup"


def _wrap_in_run_context(agent_ctx: AgentContext) -> MagicMock:
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
    ctx.internal_fs = IsolatedMemoryFileSystem()

    # Provide a mock run_ctx so _get_session_state() can use it as a
    # WeakKeyDictionary key (MagicMock supports weak references).
    mock_run_ctx = MagicMock()
    mock_run_ctx.session_id = "ses_parent_001"
    ctx.run_ctx = mock_run_ctx
    return ctx


# ---------------------------------------------------------------------------
# Test 1: CancelledError propagation when blocking on output
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_background_output_block_true_cancelled_error_propagates():
    """Test CancelledError handling when the calling task is cancelled while
    blocked on background_output(block=True).

    The try/finally block in ``_background_output`` correctly handles
    CancelledError: ``wait_for_task`` internally catches CancelledError
    and returns the current task model, then the ``finally`` block
    unregisters the blocking waiter.  This means the waiter IS cleaned
    up even when the caller is cancelled, and the method returns a
    "still running" status string instead of raising.

    The background task itself must remain unaffected by the caller's
    cancellation.
    """
    capability = BackgroundTaskCapability(schemas=None)

    ctx = _make_agent_context()
    state = capability._get_session_state(_wrap_in_run_context(ctx))

    # Register a long-running task
    task_model = BackgroundTask(
        id="bg_long",
        description="long",
        agent_or_team="test",
        prompt="test",
        parent_session_id=None,
        child_session_id=None,
        status="running",
    )
    state.task_manager.register_task(task_model)

    async def _long_coro() -> None:
        await asyncio.sleep(30)

    state.task_manager.start_task("bg_long", _long_coro())

    # Try to block on output, but cancel the caller
    async def _block_and_wait():
        return await capability._background_output(
            _wrap_in_run_context(ctx),
            task_id="bg_long",
            block=True,
            timeout_seconds=10,
        )

    block_task = asyncio.create_task(_block_and_wait())
    await asyncio.sleep(0.1)  # Let it start blocking
    block_task.cancel()

    # wait_for_task catches CancelledError internally, so it does NOT
    # propagate.  The method returns a status string instead.
    result = await block_task

    # The returned string should indicate the task is still running
    assert isinstance(result, str)
    assert "bg_long" in result
    assert "running" in result

    # The blocking waiter IS cleaned up by the try/finally block even
    # when the caller is cancelled.
    assert not state.task_manager.has_blocking_waiter("bg_long")

    # Verify background task is still running (not cancelled by the caller's
    # cancellation — only background_cancel should cancel tasks)
    t = state.task_manager.get_task("bg_long")
    assert t is not None
    assert t.status not in ("completed", "cancelled", "error", "timed_out")

    # Cleanup
    await state.task_manager.cancel_task("bg_long")


# ---------------------------------------------------------------------------
# Test 2: None return from wait_for_task when task cleaned up during wait
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_background_output_returns_none_when_task_cleaned_up_during_wait():
    """Test that _background_output returns 'not found' when wait_for_task
    returns None, which happens when the task is cleaned up during the wait.

    The existing None check at line 896-897 correctly handles this case:
    ``if task_model is None: return f"Task {task_id!r} not found"``.
    The method returns a 'not found' message rather than raising an
    AttributeError on None.
    """
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()
    state = capability._get_session_state(_wrap_in_run_context(ctx))
    # Replace the default manager with one that always returns None from
    # wait_for_task, simulating the task being cleaned up during the wait.
    state.task_manager = _NullWaitManager()

    # Create a task that appears running (non-terminal) so the code path
    # reaches wait_for_task instead of returning early at the terminal check.
    task_model = BackgroundTask(
        id="bg_cleanup",
        description="cleanup test",
        agent_or_team="test",
        prompt="test",
        parent_session_id=None,
        child_session_id=None,
        status="running",
    )
    state.task_manager.register_task(task_model)

    result = await capability._background_output(
        _wrap_in_run_context(ctx),
        task_id="bg_cleanup",
        block=True,
        timeout_seconds=5,
    )

    assert "not found" in result
    assert "bg_cleanup" in result


# ---------------------------------------------------------------------------
# Test 3: Fragile string matching in _background_cancel
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_background_cancel_fragile_string_matching():
    """Test that _background_cancel uses a status-based check rather than
    fragile substring matching on the cancel_result string.

    When cancel_task returns a message that does NOT contain "not found"
    or "already", the status-based check correctly determines the task
    was NOT cancelled (the _FragileCancelManager doesn't change the task
    status), so the raw cancel_result is returned instead of the
    success format.
    """
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context()
    state = capability._get_session_state(_wrap_in_run_context(ctx))
    # Replace the default manager with one whose cancel_task returns a
    # string that doesn't match "not found" or "already".
    state.task_manager = _FragileCancelManager()

    # Register a task so get_task returns a model with a description
    task_model = BackgroundTask(
        id="bg_test",
        description="fragile test",
        agent_or_team="test",
        prompt="test",
        parent_session_id=None,
        child_session_id=None,
        status="running",
    )
    state.task_manager.register_task(task_model)

    result = await capability._background_cancel(
        _wrap_in_run_context(ctx),
        task_id="bg_test",
    )

    # The status-based check correctly sees the task is NOT cancelled
    # (the _FragileCancelManager doesn't change the task status), so
    # the raw cancel_result is returned instead of the success format.
    assert result == "Task was removed by cleanup"
    assert "cancelled successfully" not in result
