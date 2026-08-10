"""TDD tests for NotificationBatcher with anyio structured concurrency.

Covers debounce timing, dedup, batch formatting, status ordering,
shutdown flush, CancelScope cancellation, orphan prevention,
fail_after timeout protection, and more.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
import inspect
from unittest.mock import AsyncMock
import warnings

import pytest

from wolfharness.capabilities.background_task.notification import NotificationBatcher
from wolfharness.capabilities.background_task.types import BackgroundTask


# Flush delivery runs via anyio TimerHandle callbacks, so tests must run under
# the anyio asyncio backend (which sets the sniffio async-library context).
pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SESSION_ID = "sess-001"


def _make_task(
    task_id: str = "task-1",
    *,
    status: str = "completed",
    description: str = "Test task",
    result: str | None = "done",
    error: str | None = None,
    parent_session_id: str | None = SESSION_ID,
) -> BackgroundTask:
    """Create a BackgroundTask with sensible defaults."""
    return BackgroundTask(
        id=task_id,
        description=description,
        agent_or_team="test_agent",
        prompt="do something",
        parent_session_id=parent_session_id,
        child_session_id="child-1",
        status=status,
        result=result,
        error=error,
        started_at=datetime.now(tz=UTC),
        completed_at=datetime.now(tz=UTC),
    )


# ---------------------------------------------------------------------------
# 1. Debounce timing
# ---------------------------------------------------------------------------


async def test_single_task_one_notification_after_debounce():
    """A single task triggers exactly one notification after the debounce window."""
    delivered: list[tuple[str, list[BackgroundTask], str]] = []

    async def deliver(sid: str, tasks: list[BackgroundTask], notice: str) -> None:
        delivered.append((sid, tasks, notice))

    batcher = NotificationBatcher(deliver, debounce_ms=50)
    await batcher.start()

    task = _make_task()
    batcher.submit(task)

    # Wait beyond debounce
    await asyncio.sleep(0.15)

    assert len(delivered) == 1
    assert delivered[0][0] == SESSION_ID
    assert len(delivered[0][1]) == 1

    await batcher.shutdown()


async def test_two_tasks_within_debounce_one_batch():
    """Two tasks submitted within the debounce window produce one batched notification."""
    delivered: list[tuple[str, list[BackgroundTask], str]] = []

    async def deliver(sid: str, tasks: list[BackgroundTask], notice: str) -> None:
        delivered.append((sid, tasks, notice))

    batcher = NotificationBatcher(deliver, debounce_ms=100)
    await batcher.start()

    batcher.submit(_make_task("task-a"))
    batcher.submit(_make_task("task-b"))

    await asyncio.sleep(0.25)

    assert len(delivered) == 1
    assert len(delivered[0][1]) == 2

    await batcher.shutdown()


async def test_task_outside_debounce_separate_notification():
    """A task submitted after the debounce window fires gets a separate notification."""
    delivered: list[tuple[str, list[BackgroundTask], str]] = []

    async def deliver(sid: str, tasks: list[BackgroundTask], notice: str) -> None:
        delivered.append((sid, tasks, notice))

    batcher = NotificationBatcher(deliver, debounce_ms=50)
    await batcher.start()

    batcher.submit(_make_task("task-a"))
    await asyncio.sleep(0.15)  # wait for first debounce to fire

    batcher.submit(_make_task("task-b"))
    await asyncio.sleep(0.15)  # wait for second debounce to fire

    assert len(delivered) == 2
    assert len(delivered[0][1]) == 1
    assert len(delivered[1][1]) == 1

    await batcher.shutdown()


# ---------------------------------------------------------------------------
# 2. Dedup
# ---------------------------------------------------------------------------


async def test_dedup_same_task_id_silently_dropped():
    """Submitting the same task ID twice results in only one notification."""
    delivered: list[tuple[str, list[BackgroundTask], str]] = []

    async def deliver(sid: str, tasks: list[BackgroundTask], notice: str) -> None:
        delivered.append((sid, tasks, notice))

    batcher = NotificationBatcher(deliver, debounce_ms=50)
    await batcher.start()

    task = _make_task("dup-task")
    batcher.submit(task)
    batcher.submit(task)  # same ID — should be dropped

    await asyncio.sleep(0.15)

    assert len(delivered) == 1
    assert len(delivered[0][1]) == 1
    assert delivered[0][1][0].id == "dup-task"

    await batcher.shutdown()


# ---------------------------------------------------------------------------
# 3. Batch formatting — headers
# ---------------------------------------------------------------------------


async def test_batch_header_single_completed():
    """Single completed task uses [BACKGROUND TASK RESULT READY] header."""
    delivered: list[str] = []

    async def deliver(sid: str, tasks: list[BackgroundTask], notice: str) -> None:
        delivered.append(notice)

    batcher = NotificationBatcher(
        deliver,
        debounce_ms=50,
        pending_count_callback=lambda: 0,
    )
    await batcher.start()

    batcher.submit(_make_task("t1", status="completed"))
    await asyncio.sleep(0.15)

    assert "[BACKGROUND TASK RESULT READY]" in delivered[0]

    await batcher.shutdown()


async def test_batch_header_all_complete():
    """When pending_count_callback returns 0, header includes [ALL BACKGROUND TASKS COMPLETE]."""
    delivered: list[str] = []

    async def deliver(sid: str, tasks: list[BackgroundTask], notice: str) -> None:
        delivered.append(notice)

    batcher = NotificationBatcher(
        deliver,
        debounce_ms=50,
        pending_count_callback=lambda: 0,
    )
    await batcher.start()

    batcher.submit(_make_task("t1", status="completed"))
    batcher.submit(_make_task("t2", status="completed"))
    await asyncio.sleep(0.15)

    assert "[ALL BACKGROUND TASKS COMPLETE]" in delivered[0]

    await batcher.shutdown()


async def test_batch_header_error():
    """Error tasks produce [BACKGROUND TASK ERROR] header."""
    delivered: list[str] = []

    async def deliver(sid: str, tasks: list[BackgroundTask], notice: str) -> None:
        delivered.append(notice)

    batcher = NotificationBatcher(
        deliver,
        debounce_ms=50,
        pending_count_callback=lambda: 0,
    )
    await batcher.start()

    batcher.submit(_make_task("t1", status="error", error="something broke"))
    await asyncio.sleep(0.15)

    assert "[BACKGROUND TASK ERROR]" in delivered[0]

    await batcher.shutdown()


# ---------------------------------------------------------------------------
# 4. Status ordering: error > completed > cancelled/timed_out
# ---------------------------------------------------------------------------


async def test_status_ordering_in_batch():
    """Tasks are ordered: error first, then completed, then cancelled/timed_out."""
    delivered: list[list[BackgroundTask]] = []

    async def deliver(sid: str, tasks: list[BackgroundTask], notice: str) -> None:
        delivered.append(list(tasks))

    batcher = NotificationBatcher(
        deliver,
        debounce_ms=100,
        pending_count_callback=lambda: 0,
    )
    await batcher.start()

    # Submit in mixed order
    batcher.submit(_make_task("completed-1", status="completed"))
    batcher.submit(_make_task("error-1", status="error", error="fail"))
    batcher.submit(_make_task("cancelled-1", status="cancelled"))
    batcher.submit(_make_task("timed_out-1", status="timed_out", error="timeout"))
    batcher.submit(_make_task("completed-2", status="completed"))

    await asyncio.sleep(0.25)

    assert len(delivered) == 1
    tasks = delivered[0]
    statuses = [t.status for t in tasks]

    # Errors first
    error_idx = [i for i, s in enumerate(statuses) if s == "error"]
    completed_idx = [i for i, s in enumerate(statuses) if s == "completed"]
    other_idx = [i for i, s in enumerate(statuses) if s in ("cancelled", "timed_out")]

    assert all(e < c for e in error_idx for c in completed_idx), "errors must precede completed"
    assert all(c < o for c in completed_idx for o in other_idx), (
        "completed must precede cancelled/timed_out"
    )

    await batcher.shutdown()


# ---------------------------------------------------------------------------
# 5. "N tasks still in progress" hint
# ---------------------------------------------------------------------------


async def test_pending_count_hint_when_in_progress():
    """When pending_count_callback > 0, notice includes 'N tasks still in progress'."""
    delivered: list[str] = []

    async def deliver(sid: str, tasks: list[BackgroundTask], notice: str) -> None:
        delivered.append(notice)

    batcher = NotificationBatcher(
        deliver,
        debounce_ms=50,
        pending_count_callback=lambda: 3,
    )
    await batcher.start()

    batcher.submit(_make_task("t1", status="completed"))
    await asyncio.sleep(0.15)

    assert "3 tasks still in progress" in delivered[0]

    await batcher.shutdown()


# ---------------------------------------------------------------------------
# 6. "ALL complete" numbered summary when pending_count == 0
# ---------------------------------------------------------------------------


async def test_all_complete_summary_when_pending_zero():
    """When pending_count_callback returns 0, notice includes numbered summary."""
    delivered: list[str] = []

    async def deliver(sid: str, tasks: list[BackgroundTask], notice: str) -> None:
        delivered.append(notice)

    batcher = NotificationBatcher(
        deliver,
        debounce_ms=50,
        pending_count_callback=lambda: 0,
    )
    await batcher.start()

    batcher.submit(_make_task("t1", status="completed", description="First task"))
    batcher.submit(_make_task("t2", status="completed", description="Second task"))
    await asyncio.sleep(0.15)

    notice = delivered[0]
    assert "[ALL BACKGROUND TASKS COMPLETE]" in notice
    # numbered summary should contain task IDs or descriptions
    assert "t1" in notice
    assert "t2" in notice

    await batcher.shutdown()


# ---------------------------------------------------------------------------
# 7. shutdown() flushes remaining pending + cancels all TimerHandles
# ---------------------------------------------------------------------------


async def test_shutdown_flushes_pending():
    """shutdown() flushes remaining pending tasks before tearing down."""
    delivered: list[tuple[str, list[BackgroundTask], str]] = []

    async def deliver(sid: str, tasks: list[BackgroundTask], notice: str) -> None:
        delivered.append((sid, tasks, notice))

    batcher = NotificationBatcher(
        deliver,
        debounce_ms=500,  # long debounce — won't fire before shutdown
        pending_count_callback=lambda: 0,
    )
    await batcher.start()

    batcher.submit(_make_task("t1", status="completed"))
    # Don't wait for debounce — call shutdown immediately
    await batcher.shutdown()

    # The pending task should have been flushed
    assert len(delivered) == 1
    assert len(delivered[0][1]) == 1


async def test_shutdown_cancels_timer_handles():
    """shutdown() cancels all pending TimerHandles."""
    delivered: list[tuple[str, list[BackgroundTask], str]] = []

    async def deliver(sid: str, tasks: list[BackgroundTask], notice: str) -> None:
        delivered.append((sid, tasks, notice))

    batcher = NotificationBatcher(deliver, debounce_ms=500)
    await batcher.start()

    batcher.submit(_make_task("t1"))
    batcher.submit(_make_task("t2", parent_session_id="sess-002"))

    # Verify timers were created
    assert len(batcher._timers) == 2

    await batcher.shutdown()

    # All timers should be cancelled (no pending timers)
    for handle in batcher._timers.values():
        assert handle.cancelled()


# ---------------------------------------------------------------------------
# 8. CancelScope cancellation propagation
# ---------------------------------------------------------------------------


async def test_cancel_scope_cancellation_stops_timers():
    """Cancelling the CancelScope stops pending delivery.

    After CancelScope.cancel(), the _schedule_flush callback still fires
    (it's a raw asyncio TimerHandle), but since the TaskGroup is under
    a cancelled scope, ``start_soon`` either no-ops or the spawned task
    is immediately cancelled.  No delivery should reach the callback.
    """
    delivered: list[tuple[str, list[BackgroundTask], str]] = []

    async def deliver(sid: str, tasks: list[BackgroundTask], notice: str) -> None:
        delivered.append((sid, tasks, notice))

    batcher = NotificationBatcher(deliver, debounce_ms=50)
    await batcher.start()

    batcher.submit(_make_task("t1"))

    # Cancel the scope immediately (before the 50ms debounce fires)
    batcher._cancel_scope.cancel()

    # After cancel scope, we can't use asyncio.sleep normally because
    # anyio propagates the cancellation.  Use suppress instead.
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.sleep(0.2)

    # No delivery should have occurred
    assert len(delivered) == 0

    # Cleanup — manually tear down since shutdown may fail under cancelled scope
    for handle in batcher._timers.values():
        handle.cancel()
    batcher._timers.clear()
    batcher._pending.clear()
    batcher._started = False


# ---------------------------------------------------------------------------
# 9. Orphan prevention
# ---------------------------------------------------------------------------


async def test_no_orphaned_tasks_after_shutdown():
    """After shutdown, no orphaned flush tasks remain."""
    delivered: list[tuple[str, list[BackgroundTask], str]] = []

    async def deliver(sid: str, tasks: list[BackgroundTask], notice: str) -> None:
        delivered.append((sid, tasks, notice))

    batcher = NotificationBatcher(deliver, debounce_ms=50)
    await batcher.start()

    batcher.submit(_make_task("t1"))
    await asyncio.sleep(0.15)

    await batcher.shutdown()

    assert len(batcher._flush_tasks) == 0
    assert not batcher._started


# ---------------------------------------------------------------------------
# 10. fail_after timeout protection
# ---------------------------------------------------------------------------


async def test_deliver_callback_timeout_caught():
    """If deliver_callback hangs, TimeoutError is caught and logged."""
    call_count = 0

    async def slow_deliver(sid: str, tasks: list[BackgroundTask], notice: str) -> None:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(10)  # hangs

    batcher = NotificationBatcher(
        slow_deliver,
        debounce_ms=50,
        deliver_timeout=0.1,
    )
    await batcher.start()

    batcher.submit(_make_task("t1"))

    # Wait long enough for debounce + timeout
    await asyncio.sleep(0.3)

    # deliver was called but timed out — no crash
    assert call_count >= 1

    await batcher.shutdown()


# ---------------------------------------------------------------------------
# 11. pending_count_callback integration
# ---------------------------------------------------------------------------


async def test_pending_count_callback_called():
    """The pending_count_callback is called during formatting."""
    callback_count = 0

    def count_cb() -> int:
        nonlocal callback_count
        callback_count += 1
        return 0

    delivered: list[str] = []

    async def deliver(sid: str, tasks: list[BackgroundTask], notice: str) -> None:
        delivered.append(notice)

    batcher = NotificationBatcher(
        deliver,
        debounce_ms=50,
        pending_count_callback=count_cb,
    )
    await batcher.start()

    batcher.submit(_make_task("t1"))
    await asyncio.sleep(0.15)

    assert callback_count >= 1

    await batcher.shutdown()


# ---------------------------------------------------------------------------
# 12. None/empty parent_session_id rejection
# ---------------------------------------------------------------------------


async def test_submit_rejects_none_parent_session_id():
    """submit() raises ValueError when parent_session_id is None."""
    batcher = NotificationBatcher(AsyncMock(), debounce_ms=50)
    await batcher.start()

    task = _make_task(parent_session_id=None)
    with pytest.raises(ValueError, match="parent_session_id"):
        batcher.submit(task)

    await batcher.shutdown()


async def test_submit_rejects_empty_parent_session_id():
    """submit() raises ValueError when parent_session_id is empty string."""
    batcher = NotificationBatcher(AsyncMock(), debounce_ms=50)
    await batcher.start()

    task = _make_task(parent_session_id="")
    with pytest.raises(ValueError, match="parent_session_id"):
        batcher.submit(task)

    await batcher.shutdown()


# ---------------------------------------------------------------------------
# 13. Task completing during _flush() — atomic pop
# ---------------------------------------------------------------------------


async def test_atomic_pop_no_lost_or_double_notifications():
    """_pending is popped atomically before _flush runs — no lost or double notifications.

    When two tasks are submitted within the same debounce window, they
    are delivered as a single batch.  The atomic pop in ``_flush``
    ensures no task is lost or delivered twice.

    Note: after delivery, ``_delivered`` is cleared when
    ``pending_count_callback()`` returns 0 (by design, to prevent
    unbounded growth).  So resubmitting a cleared task ID will deliver
    again — this is expected behavior, not a bug.
    """
    delivered: list[list[BackgroundTask]] = []

    async def deliver(sid: str, tasks: list[BackgroundTask], notice: str) -> None:
        delivered.append(list(tasks))

    batcher = NotificationBatcher(
        deliver,
        debounce_ms=50,
        pending_count_callback=lambda: 0,
    )
    await batcher.start()

    batcher.submit(_make_task("t1"))
    batcher.submit(_make_task("t2"))
    await asyncio.sleep(0.15)

    # Only one delivery with both tasks (atomic pop)
    assert len(delivered) == 1
    assert len(delivered[0]) == 2

    # Verify the tasks were the correct ones
    ids = {t.id for t in delivered[0]}
    assert ids == {"t1", "t2"}

    await batcher.shutdown()


# ---------------------------------------------------------------------------
# 14. start() called before first submit — auto-start
# ---------------------------------------------------------------------------


async def test_auto_start_on_first_submit():
    """If start() hasn't been called, submit() auto-starts the batcher."""
    delivered: list[tuple[str, list[BackgroundTask], str]] = []

    async def deliver(sid: str, tasks: list[BackgroundTask], notice: str) -> None:
        delivered.append((sid, tasks, notice))

    batcher = NotificationBatcher(deliver, debounce_ms=50)

    # Don't call start() — submit should auto-start
    batcher.submit(_make_task("t1"))

    # Give auto-start task time to initialize
    await asyncio.sleep(0.05)
    assert batcher._started is True

    await asyncio.sleep(0.15)
    assert len(delivered) == 1

    await batcher.shutdown()


# ---------------------------------------------------------------------------
# 15. _delivered cleared when pending_count_callback returns 0
# ---------------------------------------------------------------------------


async def test_delivered_cleared_when_pending_zero():
    """_delivered set is cleared when pending_count_callback returns 0."""
    delivered: list[str] = []

    async def deliver(sid: str, tasks: list[BackgroundTask], notice: str) -> None:
        delivered.append(notice)

    batcher = NotificationBatcher(
        deliver,
        debounce_ms=50,
        pending_count_callback=lambda: 0,
    )
    await batcher.start()

    batcher.submit(_make_task("t1"))
    await asyncio.sleep(0.15)

    # After delivery with pending==0, _delivered should be cleared
    assert len(batcher._delivered) == 0

    await batcher.shutdown()


async def test_delivered_not_cleared_when_pending_nonzero():
    """_delivered set is NOT cleared when pending_count_callback returns > 0."""
    delivered: list[str] = []

    async def deliver(sid: str, tasks: list[BackgroundTask], notice: str) -> None:
        delivered.append(notice)

    batcher = NotificationBatcher(
        deliver,
        debounce_ms=50,
        pending_count_callback=lambda: 2,
    )
    await batcher.start()

    batcher.submit(_make_task("t1"))
    await asyncio.sleep(0.15)

    # _delivered should still contain the task ID since pending > 0
    assert "t1" in batcher._delivered

    await batcher.shutdown()


# ---------------------------------------------------------------------------
# 16. _schedule_flush is SYNC (not async)
# ---------------------------------------------------------------------------


async def test_schedule_flush_is_sync_no_runtime_warning():
    """_schedule_flush is sync — calling it does not produce a coroutine warning."""
    delivered: list[tuple[str, list[BackgroundTask], str]] = []

    async def deliver(sid: str, tasks: list[BackgroundTask], notice: str) -> None:
        delivered.append((sid, tasks, notice))

    batcher = NotificationBatcher(deliver, debounce_ms=50)
    await batcher.start()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # _schedule_flush should be callable without await
        batcher._schedule_flush(SESSION_ID)

    await asyncio.sleep(0.15)

    await batcher.shutdown()


def test_schedule_flush_is_not_coroutine_function():
    """_schedule_flush must not be an async function."""
    assert not inspect.iscoroutinefunction(NotificationBatcher._schedule_flush)


def test_submit_is_not_async():
    """submit() must be a sync method — no `async` keyword."""
    assert not inspect.iscoroutinefunction(NotificationBatcher.submit)


# ---------------------------------------------------------------------------
# Additional: deliver_callback settable post-construction
# ---------------------------------------------------------------------------


async def test_deliver_callback_settable_post_construction():
    """The deliver_callback attribute can be set after construction."""
    initial = AsyncMock()
    batcher = NotificationBatcher(initial, debounce_ms=50)

    assert batcher.deliver_callback is initial

    replacement = AsyncMock()
    batcher.deliver_callback = replacement
    assert batcher.deliver_callback is replacement

    await batcher.start()
    await batcher.shutdown()


# ---------------------------------------------------------------------------
# Additional: multiple sessions isolated
# ---------------------------------------------------------------------------


async def test_multiple_sessions_isolated():
    """Tasks from different sessions get separate notifications."""
    delivered: list[tuple[str, list[BackgroundTask], str]] = []

    async def deliver(sid: str, tasks: list[BackgroundTask], notice: str) -> None:
        delivered.append((sid, tasks, notice))

    batcher = NotificationBatcher(deliver, debounce_ms=50)
    await batcher.start()

    batcher.submit(_make_task("t1", parent_session_id="sess-A"))
    batcher.submit(_make_task("t2", parent_session_id="sess-B"))

    await asyncio.sleep(0.15)

    session_ids = {d[0] for d in delivered}
    assert session_ids == {"sess-A", "sess-B"}

    await batcher.shutdown()


# ---------------------------------------------------------------------------
# Additional: _format_batch directly
# ---------------------------------------------------------------------------


async def test_format_batch_orders_by_status():
    """_format_batch returns error tasks first, then completed, then cancelled/timed_out."""
    batcher = NotificationBatcher(AsyncMock(), debounce_ms=50)
    await batcher.start()

    tasks = [
        _make_task("completed-1", status="completed"),
        _make_task("error-1", status="error", error="fail"),
        _make_task("cancelled-1", status="cancelled"),
        _make_task("timed_out-1", status="timed_out", error="timeout"),
        _make_task("completed-2", status="completed"),
    ]

    result = batcher._format_batch(tasks)

    # Verify ordering in the output
    err_pos = result.find("error-1")
    comp1_pos = result.find("completed-1")
    comp2_pos = result.find("completed-2")
    cancel_pos = result.find("cancelled-1")
    timeout_pos = result.find("timed_out-1")

    assert err_pos < comp1_pos < cancel_pos
    assert err_pos < comp2_pos < cancel_pos
    assert timeout_pos > comp2_pos

    await batcher.shutdown()


async def test_format_batch_xml_wrapper():
    """_format_batch wraps output in <system-reminder> tags."""
    batcher = NotificationBatcher(AsyncMock(), debounce_ms=50)
    await batcher.start()

    tasks = [_make_task("t1", status="completed")]
    result = batcher._format_batch(tasks)

    assert "<system-reminder>" in result
    assert "</system-reminder>" in result

    await batcher.shutdown()
