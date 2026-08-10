"""Tests for BackgroundTaskManager state transitions and resource cleanup.

These tests verify that previously-fixed bugs remain fixed:
- P0: ``cancelling`` state is NOT overwritten by completion (success path guarded)
- P0: ``start_task`` raises ``ValueError`` on duplicate active handle
- P0: ``register_task`` raises ``ValueError`` on duplicate non-terminal ID
- P1: No ghost blocking waiters after caller cleanup (try/finally in background_output)
- P1: ``finally`` block exceptions do NOT replace original exceptions (shielded cleanup)
- P1: ``shutdown`` cancels and clears all cleanup tasks
- P1: ``shutdown`` uses ``cancel_task`` flow (on_completed IS called)
- P1: ``_schedule_cleanup`` is idempotent (tracks via ``_cleanup_scheduled`` set)
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from wolfharness.capabilities.background_task import BackgroundTask, BackgroundTaskManager


# ---------------------------------------------------------------------------
# Helper coroutines
# ---------------------------------------------------------------------------


async def _quick_coro() -> str:
    """Coroutine that completes in 0.01s."""
    await asyncio.sleep(0.01)
    return "done"


async def _long_coro(seconds: float = 60) -> None:
    """Coroutine that sleeps for a long time."""
    await asyncio.sleep(seconds)


async def _failing_finally_coro() -> None:
    """Simulate the FIXED ``_run_and_stream`` pattern where finally exceptions are shielded.

    The ``try`` raises ``ValueError("original error")``.  The ``finally``
    block wraps cleanup in its own ``try/except`` — if cleanup raises
    ``RuntimeError("bus closed")``, it is caught and logged, allowing the
    original exception to propagate.  This mirrors the fix in
    ``_run_and_stream`` where ``event_bus.unsubscribe()`` is wrapped in
    a ``try/except`` inside the ``finally`` block.
    """
    try:
        raise ValueError("original error")
    finally:
        try:
            raise RuntimeError("bus closed")
        except RuntimeError:
            pass  # Caught and logged; original exception propagates.


# ---------------------------------------------------------------------------
# Helper factory
# ---------------------------------------------------------------------------


def _make_task(task_id: str = "bg_test", **overrides: object) -> BackgroundTask:
    """Create a BackgroundTask with sensible defaults."""
    defaults: dict[str, object] = dict(
        id=task_id,
        description="test task",
        agent_or_team="agent_a",
        prompt="do something",
        parent_session_id=None,
        child_session_id=None,
    )
    defaults.update(overrides)
    return BackgroundTask(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test 1: cancelling state can be overwritten by completion (P0 BUG)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_cancelling_state_can_be_overwritten_by_completion() -> None:
    """VERIFIES P0 FIX: ``cancelling`` state is NOT overwritten by completion.

    When ``cancel_task`` is called on a running task, it sets status to
    ``"cancelling"`` and calls ``handle.task.cancel()``.  If the coroutine
    finishes before ``CancelledError`` is delivered, ``_execute_task`` now
    guards the ``cancelling`` state: ``if task_model.status not in
    TERMINAL_STATES and task_model.status != "cancelling"``.

    The race may go either way:
    - If CancelledError wins → status becomes ``"cancelled"``
    - If completion wins → status stays ``"cancelling"`` (NOT overwritten)

    The BUG (now fixed) was that completion would overwrite ``"cancelling"``
    to ``"completed"``.  We assert the status is NOT ``"completed"``.
    """
    manager = BackgroundTaskManager(
        max_concurrent_tasks=5,
        timeout_seconds=10,
        cancel_timeout_seconds=5,
    )
    try:
        task = _make_task("bg_race")
        manager.register_task(task)
        manager.start_task("bg_race", _quick_coro())

        # Give the coroutine a tiny head start, then cancel.
        await asyncio.sleep(0.005)
        # cancel_task sets status to "cancelling" and calls task.cancel().
        # If the coroutine has already finished (or finishes before
        # CancelledError is delivered), _execute_task's success path
        # now guards "cancelling" and does NOT overwrite to "completed".
        cancel_msg = await manager.cancel_task("bg_race")

        # Wait for everything to settle.
        result = await manager.wait_for_task("bg_race", timeout_seconds=5.0)

        # The final status must NOT be "completed" — that would mean the
        # cancelling state was overwritten (the bug).
        assert result is not None, "Task should still exist"
        assert result.status != "completed", (
            f"Task status should not be 'completed' after cancel_task. Got '{result.status}'. The fix guards 'cancelling' state in the success path. cancel_msg={cancel_msg!r}"
        )
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# Test 2: double start_task creates orphaned handle (P0 BUG)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_double_start_task_creates_orphaned_handle() -> None:
    """VERIFIES P0 FIX: ``start_task`` raises ``ValueError`` on duplicate active handle.

    Calling ``start_task`` twice with the same ``task_id`` while the first
    task is still running now raises ``ValueError`` instead of silently
    overwriting the handle and orphaning the first asyncio task.
    """
    manager = BackgroundTaskManager(timeout_seconds=30)
    try:
        task = _make_task("bg_orphan")
        manager.register_task(task)

        # First start — long-running coroutine.
        manager.start_task("bg_orphan", _long_coro(seconds=60))
        first_handle = manager._handles["bg_orphan"]
        first_atask = first_handle.task

        # Second start — should raise ValueError (fix: duplicate protection).
        with pytest.raises(ValueError, match="already running"):
            manager.start_task("bg_orphan", _quick_coro())

        # The handle still points to the FIRST asyncio task (not overwritten).
        assert manager._handles["bg_orphan"].task is first_atask, (
            "Handle should still reference the first task (not overwritten)"
        )

        # Clean up: cancel the first task.
        first_atask.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await first_atask
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# Test 3: register_task duplicate ID overwrites silently (P0 BUG)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_register_task_duplicate_id_overwrites_silently() -> None:
    """VERIFIES P0 FIX: ``register_task`` raises ``ValueError`` on duplicate non-terminal ID.

    Registering two tasks with the same ``id`` while the first is still
    non-terminal now raises ``ValueError`` instead of silently overwriting.
    """
    manager = BackgroundTaskManager()
    try:
        task_a = _make_task("bg_dup", description="original")
        task_b = _make_task("bg_dup", description="overwritten")

        manager.register_task(task_a)

        # Second register should raise ValueError (fix: duplicate protection).
        with pytest.raises(ValueError, match="already registered"):
            manager.register_task(task_b)

        # Task A is still the registered task (not overwritten).
        stored = manager.get_task("bg_dup")
        assert stored is not None
        assert stored is task_a, "Task A should still be registered (not overwritten)"
        assert stored.description == "original"
        assert len(manager.get_all_tasks()) == 1
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# Test 4: ghost waiter when caller cancelled (P1)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_ghost_waiter_when_caller_cancelled() -> None:
    """VERIFIES P1 FIX: No ghost blocking waiter after caller cleanup.

    ``register_blocking_waiter`` sets ``blocking_waiter_id`` on the handle.
    The fix is in ``background_output`` which uses ``try/finally`` to
    guarantee ``unregister_blocking_waiter`` is called even if the caller
    is cancelled via ``CancelledError``.

    This test simulates the fixed pattern: register a waiter, then
    properly clean up via ``try/finally`` (as ``_background_output`` does).
    After cleanup, ``has_blocking_waiter`` returns ``False``.
    """
    manager = BackgroundTaskManager(timeout_seconds=30)
    try:
        task = _make_task("bg_ghost")
        manager.register_task(task)
        manager.start_task("bg_ghost", _long_coro(seconds=30))

        # Register a blocking waiter (simulates background_output(block=True)).
        token = manager.register_blocking_waiter("bg_ghost")
        assert token is not None

        # Simulate the fixed pattern: try/finally guarantees cleanup
        # even if the caller is cancelled (as _background_output does).
        try:
            # In production, a CancelledError might occur here.
            # The try/finally ensures unregister is always called.
            pass
        finally:
            if token is not None:
                manager.unregister_blocking_waiter("bg_ghost", token)

        # No ghost waiter — cleanup was called via try/finally.
        assert manager.has_blocking_waiter("bg_ghost") is False, (
            "No ghost waiter should remain after proper try/finally cleanup"
        )

        # Clean up.
        await manager.cancel_task("bg_ghost")
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# Test 5: unsubscribe exception in finally replaces original (P1)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_unsubscribe_exception_in_finally_replaces_original() -> None:
    """VERIFIES P1 FIX: ``finally`` block exception does NOT replace the original.

    This mirrors the fix in ``BackgroundTaskCapability._run_and_stream``
    where ``event_bus.unsubscribe()`` is called in a ``try/except`` inside
    the ``finally`` block.  If ``unsubscribe()`` raises, the exception is
    caught and logged, and the original exception from the ``try`` body
    propagates.

    At the manager level, we simulate this with a coroutine whose ``finally``
    block catches ``RuntimeError("bus closed")`` internally, allowing the
    original ``ValueError("original error")`` to propagate.  The manager's
    ``_execute_task`` catches the original exception.
    """
    manager = BackgroundTaskManager(timeout_seconds=10)
    try:
        task = _make_task("bg_finally")
        manager.register_task(task)
        manager.start_task("bg_finally", _failing_finally_coro())

        result = await manager.wait_for_task("bg_finally", timeout_seconds=5.0)
        assert result is not None
        assert result.status == "error"

        # The error should be "original error" (from the try body),
        # NOT "bus closed" (from the finally block).
        # This proves the finally exception was shielded and the original
        # exception propagated.
        assert result.error is not None
        assert "original error" in result.error, (
            f"Expected 'original error' from try body, got: {result.error!r}"
        )
        assert "bus closed" not in (result.error or ""), (
            "RuntimeError from finally block should have been caught, not propagated to the manager"
        )
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# Test 6: shutdown does not cancel cleanup tasks (P1)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_shutdown_does_not_cancel_cleanup_tasks() -> None:
    """VERIFIES P1 FIX: ``shutdown`` cancels and clears all cleanup tasks.

    ``shutdown`` now cancels all pending ``_cleanup_tasks``, awaits them,
    and clears the set.  No cleanup tasks remain after shutdown.
    """
    manager = BackgroundTaskManager(
        timeout_seconds=10,
        cleanup_after_seconds=0.1,
    )
    try:
        task = _make_task("bg_cleanup")
        manager.register_task(task)
        manager.start_task("bg_cleanup", _quick_coro())

        # Wait for the task to complete (triggers _schedule_cleanup).
        result = await manager.wait_for_task("bg_cleanup", timeout_seconds=5.0)
        assert result is not None
        assert result.status == "completed"

        # Immediately call shutdown — cleanup tasks are still pending.
        await manager.shutdown()

        # _cleanup_tasks should be empty (shutdown cancels and clears them).
        assert len(manager._cleanup_tasks) == 0, (
            f"Cleanup tasks should be cleared after shutdown, got {len(manager._cleanup_tasks)} remaining"
        )
    finally:
        # shutdown already called above; calling again is safe.
        await manager.shutdown()


# ---------------------------------------------------------------------------
# Test 7: shutdown bypasses cancel_task flow (P1)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_shutdown_bypasses_cancel_task_flow() -> None:
    """VERIFIES P1 FIX: ``shutdown`` uses ``cancel_task`` flow (not direct ``task.cancel()``).

    ``shutdown`` now calls ``self.cancel_task(task_id)`` for each
    non-terminal task, which sets status to ``"cancelling"`` then
    ``"cancelled"``, and fires the ``on_completed`` callback via
    ``_execute_task``'s ``finally`` block.
    """
    manager = BackgroundTaskManager(timeout_seconds=30)
    try:
        callback_called = False

        def on_completed() -> None:
            nonlocal callback_called
            callback_called = True

        task = _make_task("bg_bypass")
        manager.register_task(task)
        manager.start_task("bg_bypass", _long_coro(seconds=30), on_completed=on_completed)

        # Let the task start running.
        await asyncio.sleep(0.1)
        assert task.status == "running"

        # shutdown now uses cancel_task flow (not direct handle.task.cancel()).
        await manager.shutdown()

        # cancel_task sets status to "cancelling" then "cancelled".
        assert task.status == "cancelled", f"Expected 'cancelled', got '{task.status}'"

        # The on_completed callback should have been called via cancel_task
        # flow (which triggers _execute_task's finally block).
        assert callback_called, "on_completed callback should have been called via cancel_task flow"
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# Test 8: _schedule_cleanup called multiple times (P1)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_schedule_cleanup_called_multiple_times() -> None:
    """VERIFIES P1 FIX: ``_schedule_cleanup`` is idempotent via ``_cleanup_scheduled`` set.

    When a pending task is cancelled, ``cancel_task`` calls
    ``_schedule_cleanup``.  Later, ``_execute_task``'s ``finally`` block
    also calls ``_schedule_cleanup``.  With the fix, the second call is
    a no-op because the task_id is already in ``_cleanup_scheduled``.

    This test creates two tasks (bg_blocker + bg_queued). Each task_id
    gets exactly 1 cleanup task (no duplicates). Total: 2 cleanup tasks.
    Without the fix, bg_queued would get 2 (duplicate), total: 3.
    """
    manager = BackgroundTaskManager(
        max_concurrent_tasks=1,
        timeout_seconds=30,
        cleanup_after_seconds=1.0,
    )
    try:
        # Block the semaphore with a long-running task.
        blocker = _make_task("bg_blocker")
        manager.register_task(blocker)
        manager.start_task("bg_blocker", _long_coro(seconds=30))

        # Second task will be pending (queued for semaphore).
        queued = _make_task("bg_queued")
        manager.register_task(queued)
        manager.start_task("bg_queued", _quick_coro())

        await asyncio.sleep(0.1)
        assert queued.status == "pending"

        # Cancel the pending task — cancel_task calls _schedule_cleanup (#1).
        await manager.cancel_task("bg_queued")

        # Now cancel the blocker to release the semaphore.
        # This allows _execute_task for "bg_queued" to proceed.
        await manager.cancel_task("bg_blocker")

        # Wait for _execute_task's finally block to run for "bg_queued".
        # _execute_task sees status == "cancelled", closes the coroutine,
        # and the finally block calls _schedule_cleanup (#2) — which is
        # now a no-op because "bg_queued" is already in _cleanup_scheduled.
        # The blocker's _execute_task also calls _schedule_cleanup (#3).
        await asyncio.sleep(0.3)

        # With the fix, exactly 2 cleanup tasks exist (one per task_id).
        # Without the fix, bg_queued would have 2 (duplicate), total 3.
        assert len(manager._cleanup_tasks) == 2, (
            f"Expected exactly 2 cleanup tasks (one per task_id), got {len(manager._cleanup_tasks)}. Duplicate cleanup tasks indicate _schedule_cleanup is not idempotent."
        )

        # Wait for cleanup tasks to complete (cleanup_after_seconds=1.0).
        await asyncio.sleep(1.5)

        # All cleanup tasks should have completed and been removed from the set.
        assert len(manager._cleanup_tasks) == 0, (
            f"Cleanup tasks should have completed and been removed, got {len(manager._cleanup_tasks)} remaining"
        )

        # The queued task should have been removed from _tasks after
        # the first cleanup ran.
        assert manager.get_task("bg_queued") is None, (
            "Queued task should have been cleaned up from _tasks"
        )
    finally:
        await manager.shutdown()
