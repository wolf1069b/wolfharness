"""Tests for ManagedTaskGroup utility."""

from __future__ import annotations

import warnings

import anyio
import pytest

from wolfharness.utils.task_group import ManagedTaskGroup


pytestmark = pytest.mark.unit


async def test_enter_exit_cleanly() -> None:
    """Given a fresh ManagedTaskGroup, when entered and exited, no warnings."""
    tg = ManagedTaskGroup()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        async with tg:
            pass
    assert tg._closed


async def test_queue_then_flush() -> None:
    """Given tasks queued before enter, when entered, tasks run."""
    tg = ManagedTaskGroup()
    results: list[str] = []

    async def task(label: str) -> None:
        results.append(label)

    tg.start_soon(task, "a")
    tg.start_soon(task, "b")
    assert len(tg._pending) == 2
    assert tg.is_busy()
    async with tg:
        await tg.wait_all()
    assert results == ["a", "b"]
    assert not tg.is_busy()


async def test_start_soon_after_enter() -> None:
    """Given an entered group, when start_soon is called, task runs immediately."""
    tg = ManagedTaskGroup()
    results: list[str] = []

    async def task(label: str) -> None:
        results.append(label)

    async with tg:
        tg.start_soon(task, "x")
        await tg.wait_all()
    assert results == ["x"]


async def test_fire_and_forget_exception_isolation() -> None:
    """Given a fire_and_forget task that raises, when group exits, no ExceptionGroup."""
    tg = ManagedTaskGroup()
    ran_after: list[str] = []

    async def failing() -> None:
        msg = "boom"
        raise ValueError(msg)

    async def healthy() -> None:
        ran_after.append("ran")

    async with tg:
        tg.fire_and_forget(failing)
        tg.start_soon(healthy)
        await tg.wait_all()
    assert ran_after == ["ran"]


async def test_is_busy_idle() -> None:
    """Given a fresh group with no tasks, is_busy returns False."""
    tg = ManagedTaskGroup()
    assert not tg.is_busy()


async def test_is_busy_pending() -> None:
    """Given a group with pending tasks, is_busy returns True."""
    tg = ManagedTaskGroup()

    async def task() -> None:
        pass

    tg.start_soon(task)
    assert tg.is_busy()
    # Clean up: enter and exit to flush the pending task.
    async with tg:
        await tg.wait_all()


async def test_is_busy_running() -> None:
    """Given an entered group with running tasks, is_busy returns True."""
    tg = ManagedTaskGroup()
    started = anyio.Event()
    release = anyio.Event()

    async def blocking_task() -> None:
        started.set()
        await release.wait()

    async with tg:
        tg.start_soon(blocking_task)
        await started.wait()
        assert tg.is_busy()
        release.set()
        await tg.wait_all()
    assert not tg.is_busy()


async def test_wait_all_blocks_until_complete() -> None:
    """Given running tasks, when wait_all is called, it blocks until done."""
    tg = ManagedTaskGroup()
    completed: list[str] = []

    async def slow_task() -> None:
        await anyio.sleep(0.01)
        completed.append("done")

    async with tg:
        tg.start_soon(slow_task)
        await tg.wait_all()
    assert completed == ["done"]


async def test_wait_all_group_stays_open() -> None:
    """Given wait_all completes, the group stays open for new tasks."""
    tg = ManagedTaskGroup()
    results: list[str] = []

    async def task(label: str) -> None:
        results.append(label)

    async with tg:
        tg.start_soon(task, "first")
        await tg.wait_all()
        assert not tg.is_busy()
        tg.start_soon(task, "second")
        await tg.wait_all()
    assert results == ["first", "second"]


async def test_close_unentered_group() -> None:
    """Given an unentered group, when close is called, it is a no-op."""
    tg = ManagedTaskGroup()
    await tg.close()
    assert tg._closed


async def test_close_already_closed_group() -> None:
    """Given an already-closed group, when close is called again, it is a no-op."""
    tg = ManagedTaskGroup()
    async with tg:
        pass
    assert tg._closed
    await tg.close()  # should not raise


async def test_close_open_group() -> None:
    """Given an open group, when close is called, it closes properly."""
    tg = ManagedTaskGroup()
    completed: list[str] = []

    async def task() -> None:
        completed.append("done")

    async with tg:
        tg.start_soon(task)
        await tg.close()
    assert tg._closed
    assert completed == ["done"]


async def test_deprecated_create_task_emits_warning() -> None:
    """Given create_task call, it emits DeprecationWarning and runs the task."""
    tg = ManagedTaskGroup()
    results: list[str] = []

    async def coro() -> str:
        results.append("ran")
        return "result"

    async with tg:
        with pytest.warns(DeprecationWarning, match="create_task is deprecated"):
            tg.create_task(coro())
        await tg.wait_all()
    assert results == ["ran"]


async def test_deprecated_cleanup_tasks_emits_warning() -> None:
    """Given cleanup_tasks call, it emits DeprecationWarning and delegates to wait_all."""
    tg = ManagedTaskGroup()
    completed: list[str] = []

    async def task() -> None:
        completed.append("done")

    async with tg:
        tg.start_soon(task)
        with pytest.warns(DeprecationWarning, match="cleanup_tasks is deprecated"):
            await tg.cleanup_tasks()
    assert completed == ["done"]


async def test_deprecated_complete_tasks_emits_warning() -> None:
    """Given complete_tasks call, it emits DeprecationWarning and delegates to wait_all."""
    tg = ManagedTaskGroup()
    completed: list[str] = []

    async def task() -> None:
        completed.append("done")

    async with tg:
        tg.start_soon(task)
        with pytest.warns(DeprecationWarning, match="complete_tasks is deprecated"):
            await tg.complete_tasks()
    assert completed == ["done"]


async def test_wait_all_with_concurrent_task_start() -> None:
    """wait_all() must complete even if new tasks start while waiting.

    Regression test: _start_tracked() previously invalidated _idle_event,
    causing wait_all() to hang when a new task started during the wait.
    """
    async with ManagedTaskGroup() as mtg:
        started = anyio.Event()
        finished = anyio.Event()

        async def slow_task() -> None:
            started.set()
            await anyio.sleep(0.1)
            finished.set()

        mtg.start_soon(slow_task)
        await started.wait()  # ensure task is running

        # Start a second task (this used to invalidate the idle event)
        mtg.start_soon(anyio.sleep, 0.05)

        # wait_all must complete, not hang
        with anyio.move_on_after(5) as scope:
            await mtg.wait_all()

        assert not scope.cancelled_caught, "wait_all() hung — race condition"
        assert finished.is_set()
