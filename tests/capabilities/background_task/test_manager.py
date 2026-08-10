"""Tests for BackgroundTaskManager."""

from __future__ import annotations

import asyncio
import contextlib
import time

from wolfharness.capabilities.background_task import BackgroundTask, BackgroundTaskManager


# ---------------------------------------------------------------------------
# Helper coroutines
# ---------------------------------------------------------------------------


async def simple_coro() -> None:
    """Coroutine that completes immediately."""


async def slow_coro(seconds: float = 10) -> None:
    """Coroutine that sleeps for the specified duration."""
    await asyncio.sleep(seconds)


async def failing_coro() -> None:
    """Coroutine that raises a ValueError."""
    msg = "test error"
    raise ValueError(msg)


async def result_coro() -> str:
    """Coroutine that returns a string result."""
    return "done"


async def os_error_coro() -> None:
    """Coroutine that raises an OSError."""
    raise OSError("connection refused")


# ---------------------------------------------------------------------------
# Helper factory
# ---------------------------------------------------------------------------


def _make_task(task_id: str = "t1", **overrides: object) -> BackgroundTask:
    """Create a BackgroundTask with sensible defaults."""
    defaults = dict(
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
# Tests: register and start
# ---------------------------------------------------------------------------


async def test_register_and_start_task() -> None:
    """Registering and starting a task transitions pending→running→completed."""
    manager = BackgroundTaskManager()
    task = _make_task("t1")
    manager.register_task(task)

    assert task.status == "pending"
    manager.start_task("t1", simple_coro())

    result = await manager.wait_for_task("t1", timeout_seconds=2.0)
    assert result is not None
    assert result.status == "completed"
    assert result.started_at is not None
    assert result.completed_at is not None


async def test_start_task_with_result() -> None:
    """A task that returns a value stores str(result) in result field."""
    manager = BackgroundTaskManager()
    task = _make_task("t1")
    manager.register_task(task)
    manager.start_task("t1", result_coro())

    result = await manager.wait_for_task("t1", timeout_seconds=2.0)
    assert result is not None
    assert result.status == "completed"
    assert result.result == "done"


# ---------------------------------------------------------------------------
# Tests: concurrency limit
# ---------------------------------------------------------------------------


async def test_concurrency_limit() -> None:
    """Tasks beyond max_concurrent_tasks stay pending until semaphore frees."""
    manager = BackgroundTaskManager(max_concurrent_tasks=2, timeout_seconds=10)
    tasks = []
    for i in range(4):
        t = _make_task(f"t{i}")
        manager.register_task(t)
        tasks.append(t)

    # Start all 4 tasks (only 2 can run concurrently).
    for i in range(4):
        manager.start_task(f"t{i}", slow_coro(seconds=5))

    # Give the event loop a moment to let the first two acquire the semaphore.
    await asyncio.sleep(0.1)

    running_count = sum(1 for t in tasks if t.status == "running")
    pending_count = sum(1 for t in tasks if t.status == "pending")
    assert running_count <= 2
    assert pending_count >= 2

    # Cancel all to clean up.
    await manager.cancel_all()


# ---------------------------------------------------------------------------
# Tests: cancel pending
# ---------------------------------------------------------------------------


async def test_cancel_pending_task() -> None:
    """Cancelling a pending task transitions directly to cancelled."""
    manager = BackgroundTaskManager()
    task = _make_task("t1")
    manager.register_task(task)

    msg = await manager.cancel_task("t1")
    assert "cancelled while pending" in msg
    assert task.status == "cancelled"
    assert task.completed_at is not None


# ---------------------------------------------------------------------------
# Tests: cancel running
# ---------------------------------------------------------------------------


async def test_cancel_running_task() -> None:
    """Cancelling a running task transitions cancelling→cancelled."""
    manager = BackgroundTaskManager(cancel_timeout_seconds=5)
    task = _make_task("t1")
    manager.register_task(task)
    manager.start_task("t1", slow_coro(seconds=30))

    # Wait briefly so the task starts running.
    await asyncio.sleep(0.1)
    assert task.status == "running"

    msg = await manager.cancel_task("t1")
    assert "cancelled" in msg
    assert task.status == "cancelled"
    assert task.completed_at is not None


# ---------------------------------------------------------------------------
# Tests: cancel terminal task (no-op)
# ---------------------------------------------------------------------------


async def test_cancel_terminal_task_is_noop() -> None:
    """Cancelling an already-completed task is a no-op."""
    manager = BackgroundTaskManager()
    task = _make_task("t1")
    manager.register_task(task)
    manager.start_task("t1", simple_coro())

    result = await manager.wait_for_task("t1", timeout_seconds=2.0)
    assert result is not None
    assert result.status == "completed"

    msg = await manager.cancel_task("t1")
    assert "already completed" in msg
    assert task.status == "completed"


async def test_cancel_nonexistent_task() -> None:
    """Cancelling a task that does not exist returns not-found message."""
    manager = BackgroundTaskManager()
    msg = await manager.cancel_task("ghost")
    assert "not found" in msg


# ---------------------------------------------------------------------------
# Tests: timeout
# ---------------------------------------------------------------------------


async def test_task_timeout() -> None:
    """A task that exceeds timeout_seconds is marked timed_out."""
    manager = BackgroundTaskManager(timeout_seconds=0.2)
    task = _make_task("t1")
    manager.register_task(task)
    manager.start_task("t1", slow_coro(seconds=10))

    result = await manager.wait_for_task("t1", timeout_seconds=5.0)
    assert result is not None
    assert result.status == "timed_out"
    assert result.error is not None
    assert "timed out" in result.error
    assert result.completed_at is not None


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


async def test_task_error_handling() -> None:
    """A task that raises ValueError is marked error with message."""
    manager = BackgroundTaskManager()
    task = _make_task("t1")
    manager.register_task(task)
    manager.start_task("t1", failing_coro())

    result = await manager.wait_for_task("t1", timeout_seconds=2.0)
    assert result is not None
    assert result.status == "error"
    assert "test error" in (result.error or "")


async def test_task_unexpected_error_handling() -> None:
    """A task that raises an unexpected exception type is marked error."""
    manager = BackgroundTaskManager()
    task = _make_task("t1")
    manager.register_task(task)
    manager.start_task("t1", os_error_coro())

    result = await manager.wait_for_task("t1", timeout_seconds=2.0)
    assert result is not None
    assert result.status == "error"
    assert "OSError" in (result.error or "")
    assert "connection refused" in (result.error or "")


# ---------------------------------------------------------------------------
# Tests: wait_for_task
# ---------------------------------------------------------------------------


async def test_wait_for_task_returns_completed() -> None:
    """wait_for_task returns the task once it completes."""
    manager = BackgroundTaskManager()
    task = _make_task("t1")
    manager.register_task(task)
    manager.start_task("t1", simple_coro())

    result = await manager.wait_for_task("t1", timeout_seconds=2.0)
    assert result is not None
    assert result.status == "completed"


async def test_wait_for_task_timeout_returns_running() -> None:
    """wait_for_task with short timeout returns current state without cancelling."""
    manager = BackgroundTaskManager(timeout_seconds=30)
    task = _make_task("t1")
    manager.register_task(task)
    manager.start_task("t1", slow_coro(seconds=10))

    # Very short wait — should time out and return running task.
    result = await manager.wait_for_task("t1", timeout_seconds=0.1)
    assert result is not None
    assert result.status in {"pending", "running"}

    # Clean up.
    await manager.cancel_task("t1")


async def test_wait_for_task_not_found() -> None:
    """wait_for_task on a missing id returns None."""
    manager = BackgroundTaskManager()
    result = await manager.wait_for_task("ghost", timeout_seconds=0.1)
    assert result is None


# ---------------------------------------------------------------------------
# Tests: cancel_all
# ---------------------------------------------------------------------------


async def test_cancel_all() -> None:
    """cancel_all cancels all non-terminal tasks and returns count."""
    manager = BackgroundTaskManager(timeout_seconds=30)
    for i in range(3):
        t = _make_task(f"t{i}")
        manager.register_task(t)
        manager.start_task(f"t{i}", slow_coro(seconds=30))

    # Let tasks start.
    await asyncio.sleep(0.1)

    count = await manager.cancel_all()
    assert count == 3

    for i in range(3):
        t = manager.get_task(f"t{i}")
        assert t is not None
        assert t.status in {"cancelled", "cancelling"}


# ---------------------------------------------------------------------------
# Tests: get_task / get_all_tasks
# ---------------------------------------------------------------------------


async def test_get_task() -> None:
    """get_task returns the registered task or None."""
    manager = BackgroundTaskManager()
    task = _make_task("t1")
    manager.register_task(task)

    assert manager.get_task("t1") is task
    assert manager.get_task("ghost") is None


async def test_get_all_tasks() -> None:
    """get_all_tasks returns all registered tasks."""
    manager = BackgroundTaskManager()
    t1 = _make_task("t1")
    t2 = _make_task("t2")
    manager.register_task(t1)
    manager.register_task(t2)

    all_tasks = manager.get_all_tasks()
    assert len(all_tasks) == 2
    assert t1 in all_tasks
    assert t2 in all_tasks


# ---------------------------------------------------------------------------
# Tests: cleanup
# ---------------------------------------------------------------------------


async def test_cleanup_removes_task_after_retention() -> None:
    """Tasks are removed from registry after cleanup_after_seconds."""
    manager = BackgroundTaskManager(cleanup_after_seconds=0.2)
    task = _make_task("t1")
    manager.register_task(task)
    manager.start_task("t1", simple_coro())

    result = await manager.wait_for_task("t1", timeout_seconds=2.0)
    assert result is not None
    assert result.status == "completed"

    # Task should still be present right after completion.
    assert manager.get_task("t1") is not None

    # Wait for cleanup to fire.
    await asyncio.sleep(0.4)

    assert manager.get_task("t1") is None


# ---------------------------------------------------------------------------
# Tests: shutdown
# ---------------------------------------------------------------------------


async def test_shutdown() -> None:
    """Shutdown cancels all running tasks and clears registries."""
    manager = BackgroundTaskManager(timeout_seconds=30)
    for i in range(3):
        t = _make_task(f"t{i}")
        manager.register_task(t)
        manager.start_task(f"t{i}", slow_coro(seconds=30))

    await asyncio.sleep(0.1)
    await manager.shutdown()

    assert manager.get_all_tasks() == []
    assert manager.get_task("t0") is None


# ---------------------------------------------------------------------------
# Tests: cancel while pending (queued behind semaphore)
# ---------------------------------------------------------------------------


async def test_cancel_pending_while_queued() -> None:
    """A task queued behind the semaphore can be cancelled directly."""
    manager = BackgroundTaskManager(
        max_concurrent_tasks=1,
        timeout_seconds=30,
        cancel_timeout_seconds=5,
    )
    # Block the semaphore with a long task.
    blocker = _make_task("blocker")
    manager.register_task(blocker)
    manager.start_task("blocker", slow_coro(seconds=30))

    # Second task will be pending (queued for semaphore).
    queued = _make_task("queued")
    manager.register_task(queued)
    manager.start_task("queued", simple_coro())

    await asyncio.sleep(0.1)
    assert queued.status == "pending"

    msg = await manager.cancel_task("queued")
    assert "cancelled while pending" in msg
    assert queued.status == "cancelled"

    # Clean up.
    await manager.cancel_task("blocker")


# ---------------------------------------------------------------------------
# Tests: cancel_timeout (cancel takes too long)
# ---------------------------------------------------------------------------


async def test_cancel_timeout_marks_error() -> None:
    """If cancellation does not complete in time, task is marked error."""
    # Use a very short cancel timeout so the await times out.
    manager = BackgroundTaskManager(
        timeout_seconds=30,
        cancel_timeout_seconds=0.05,
    )
    task = _make_task("t1")
    manager.register_task(task)
    # A coroutine that catches CancelledError and keeps running —
    # this simulates a task that won't honour cancellation promptly.
    manager.start_task("t1", _uncancellable_coro())

    await asyncio.sleep(0.1)
    assert task.status == "running"

    msg = await manager.cancel_task("t1")
    # Should indicate cancellation timed out.
    assert "timed out" in msg or "error" in msg

    # The task should be in a terminal state.
    assert task.status in {"error", "cancelled"}


async def _uncancellable_coro() -> None:
    """Coroutine that swallows CancelledError and sleeps again.

    This simulates poorly-behaved code that does not honour cancellation.
    """
    try:
        await asyncio.sleep(100)
    except asyncio.CancelledError:
        # Swallow and keep going.
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(100)


# ---------------------------------------------------------------------------
# Tests: terminal state protection
# ---------------------------------------------------------------------------


async def test_terminal_state_not_overwritten() -> None:
    """Once a task reaches a terminal state, it must not be overwritten."""
    manager = BackgroundTaskManager(timeout_seconds=0.2)
    task = _make_task("t1")
    manager.register_task(task)
    manager.start_task("t1", slow_coro(seconds=10))

    # Wait for timeout.
    result = await manager.wait_for_task("t1", timeout_seconds=2.0)
    assert result is not None
    assert result.status == "timed_out"

    # Trying to cancel a terminal task should be a no-op.
    msg = await manager.cancel_task("t1")
    assert "already" in msg
    assert task.status == "timed_out"


# ---------------------------------------------------------------------------
# Tests: nested asyncio.wait_for regression
# ---------------------------------------------------------------------------


async def test_wait_for_task_timeout_under_nested_wait_for() -> None:
    """Regression test: wait_for_task timeout must work when nested
    inside another asyncio.wait_for (simulating agent event loop).

    Python 3.12+ has a bug where nested asyncio.wait_for calls can
    convert CancelledError into TimeoutError, breaking inner timeouts.
    Using asyncio.wait instead of asyncio.wait_for avoids this.
    """
    manager = BackgroundTaskManager(timeout_seconds=300)

    task = _make_task("test-nested-timeout")
    manager.register_task(task)

    async def long_running() -> None:
        await asyncio.sleep(300)

    manager.start_task("test-nested-timeout", long_running())
    await asyncio.sleep(0.1)

    # Simulate agent event loop: outer wait_for polling with short timeout
    # while wait_for_task uses its own wait internally.
    start = time.monotonic()

    async def simulate_agent_event_loop() -> None:
        """Mimics how agent run_stream polls event_queue.get() with timeout."""
        while time.monotonic() - start < 10:  # Safety limit
            # This is how agent's run_stream polls events.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    asyncio.sleep(0.05),  # simulated event_queue.get()
                    timeout=0.1,
                )

    # Run wait_for_task and agent event loop concurrently.
    wait_task = asyncio.create_task(
        manager.wait_for_task("test-nested-timeout", timeout_seconds=2.0),
    )

    # Also run the agent event loop simulation.
    agent_loop_task = asyncio.create_task(simulate_agent_event_loop())

    # wait_for_task should complete in ~2 seconds.
    done, _pending = await asyncio.wait(
        {wait_task, agent_loop_task},
        timeout=5.0,
        return_when=asyncio.FIRST_COMPLETED,
    )

    elapsed = time.monotonic() - start
    assert wait_task in done, f"wait_for_task did not complete within {elapsed:.1f}s"
    assert elapsed < 4.0, f"wait_for_task took {elapsed:.1f}s, expected ~2s"

    # Cleanup.
    agent_loop_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await agent_loop_task
    await manager.shutdown()
