"""Background task lifecycle manager."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
import logging
import traceback
from typing import TYPE_CHECKING, Any
import uuid

import logfire

from wolfharness.capabilities.background_task.types import (
    BackgroundTask,
    TaskHandle,
    TaskStatus,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine


logger = logging.getLogger(__name__)

TERMINAL_STATES: frozenset[TaskStatus] = frozenset(
    {"completed", "error", "cancelled", "timed_out"},
)


class BackgroundTaskManager:
    """Manages background task lifecycle, concurrency, timeout, and cleanup.

    Coordinates task registration, execution with semaphore-based concurrency
    gating, per-task timeout enforcement, cancellation with graceful shutdown,
    and retention-based cleanup of terminal tasks.

    Also tracks active blocking waiters so that completion callbacks can
    decide whether to inject a lead-agent notification or rely on the
    blocking ``background_output`` call to return the result directly.
    """

    def __init__(
        self,
        timeout_seconds: float = 1800,
        max_concurrent_tasks: int = 50,
        cleanup_after_seconds: float = 3600,
        cancel_timeout_seconds: float = 30,
    ) -> None:
        self._tasks: dict[str, BackgroundTask] = {}
        self._handles: dict[str, TaskHandle] = {}
        self._max_concurrent_tasks = max_concurrent_tasks
        if max_concurrent_tasks < 1:
            msg = f"max_concurrent_tasks must be >= 1, got {max_concurrent_tasks}"
            raise ValueError(msg)
        self._concurrency_semaphore: asyncio.Semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._timeout_seconds = timeout_seconds
        self._cleanup_after_seconds = cleanup_after_seconds
        self._cancel_timeout_seconds = cancel_timeout_seconds

        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._cleanup_scheduled: set[str] = set()

        # task_id → waiter_id  (at most one blocking waiter per task)
        self._blocking_waiters: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_task(self, task: BackgroundTask) -> None:
        """Add a task to the registry in ``pending`` state."""
        existing = self._tasks.get(task.id)
        if existing is not None and existing.status not in TERMINAL_STATES:
            msg = f"Task {task.id!r} already registered and non-terminal (status={existing.status})"
            raise ValueError(msg)
        self._tasks[task.id] = task

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def start_task(
        self,
        task_id: str,
        coro: Coroutine[Any, Any, None],
        on_completed: Callable[[], None] | None = None,
    ) -> None:
        """Transition a registered task to running and begin execution.

        Creates an ``asyncio.Task`` wrapped by ``_execute_task``.  The task
        remains ``pending`` until the concurrency semaphore is acquired inside
        the execution wrapper.

        Args:
            task_id: The task identifier
            coro: The coroutine to execute
            on_completed: Optional callback fired after the task reaches a
                terminal state but **before** the completion event is set.
                This allows the callback to inspect ``blocking_waiter_id``
                while the blocking waiter is still registered.
        """
        with logfire.span("background_task.manager.start_task", task_id=task_id):
            existing_handle = self._handles.get(task_id)
            if existing_handle is not None and not existing_handle.task.done():
                msg = f"Task {task_id!r} is already running"
                raise ValueError(msg)
            atask = asyncio.create_task(self._execute_task(task_id, coro))
            handle = TaskHandle(task=atask, on_completed=on_completed)
            self._handles[task_id] = handle

    @logfire.instrument("background_task.manager.execute_task")
    async def _execute_task(
        self,
        task_id: str,
        coro: Coroutine[Any, Any, None],
    ) -> None:
        """Wrap a user coroutine with semaphore, timeout, and state management."""
        task_model = self._tasks[task_id]
        handle = self._handles.get(task_id)

        try:
            await self._concurrency_semaphore.acquire()
        except asyncio.CancelledError:
            if task_model.status not in TERMINAL_STATES:
                task_model.status = "cancelled"
                task_model.completed_at = datetime.now(tz=UTC)
                if handle is not None:
                    if handle.on_completed is not None:
                        handle.on_completed()
                    handle.completion_event.set()
                self._schedule_cleanup(task_id)
            return

        try:
            # Re-check: task may have been cancelled while queued for the semaphore.
            if task_model.status == "cancelled":
                coro.close()  # Prevent "coroutine was never awaited" warning.
                return

            task_model.status = "running"
            task_model.started_at = datetime.now(tz=UTC)

            try:
                result = await self._run_with_timeout(task_model, coro)
                # Status may have been changed to "cancelling" by another task
                # during the await, so we can't let mypy narrow it to "running".
                current_status: TaskStatus = task_model.status
                if current_status not in TERMINAL_STATES and current_status != "cancelling":
                    task_model.status = "completed"
                    task_model.result = str(result) if result is not None else None
                    task_model.completed_at = datetime.now(tz=UTC)
            except asyncio.CancelledError:
                # MUST NOT retry — set cancelled and terminate immediately.
                if task_model.status not in TERMINAL_STATES:
                    task_model.status = "cancelled"
                    task_model.completed_at = datetime.now(tz=UTC)
            except TimeoutError:
                if task_model.status not in TERMINAL_STATES:
                    task_model.status = "timed_out"
                    task_model.error = f"Task timed out after {self._timeout_seconds}s"
                    task_model.completed_at = datetime.now(tz=UTC)
            except (ValueError, RuntimeError, TypeError, KeyError, AttributeError) as exc:
                if task_model.status not in TERMINAL_STATES:
                    task_model.status = "error"
                    task_model.error = str(exc)
                    task_model.completed_at = datetime.now(tz=UTC)
                    logger.error(  # noqa: TRY400
                        "Task exception: %s\n%s",
                        exc,
                        traceback.format_exc(),
                    )
            except Exception as exc:  # noqa: BLE001
                # Catch-all: guarantee task reaches terminal state regardless of
                # exception type.  We must not leave the task stuck in "running"
                # when the coroutine has crashed.
                if task_model.status not in TERMINAL_STATES:
                    task_model.status = "error"
                    task_model.error = f"{type(exc).__name__}: {exc}"
                    task_model.completed_at = datetime.now(tz=UTC)
                    logger.error(  # noqa: TRY400
                        "Task unexpected exception: %s\n%s",
                        exc,
                        traceback.format_exc(),
                    )
        finally:
            self._concurrency_semaphore.release()
            if handle is not None:
                # Fire the completion callback *before* setting the event so
                # the callback can inspect ``blocking_waiter_id`` while the
                # blocking waiter is still registered.
                if handle.on_completed is not None:
                    logger.debug(
                        "Calling on_completed for task_id=%s",
                        task_id,
                    )
                    try:
                        handle.on_completed()
                        logger.debug(
                            "on_completed returned for task_id=%s",
                            task_id,
                        )
                    except Exception:  # noqa: BLE001
                        logger.error(  # noqa: TRY400
                            "on_completed FAILED for task_id=%s\n%s",
                            task_id,
                            traceback.format_exc(),
                        )
                handle.completion_event.set()
            self._schedule_cleanup(task_id)
            logger.debug(
                "_execute_task finished for task_id=%s status=%s",
                task_id,
                task_model.status,
            )

    async def _run_with_timeout(
        self,
        task_model: BackgroundTask,
        coro: Coroutine[Any, Any, None],
    ) -> Any:
        """Run ``coro`` under ``self._timeout_seconds``, distinguishing timeout from cancel.

        ``asyncio.wait_for`` cancels the inner coroutine *before* raising
        ``TimeoutError``, so a cancelled coroutine cannot tell whether the
        cancellation came from a timeout or an explicit ``cancel_task`` call.
        This helper marks the task model ``timed_out`` *before* cancelling the
        inner coroutine, letting the coroutine's ``CancelledError`` handler
        observe the terminal status and write the correct output message.
        """
        inner_task = asyncio.create_task(coro)
        timeout_task = asyncio.create_task(asyncio.sleep(self._timeout_seconds))
        try:
            _done, _pending = await asyncio.wait(
                {inner_task, timeout_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            # Explicit cancellation of this manager task: propagate to the
            # inner coroutine and clean up the timeout sleeper.
            inner_task.cancel()
            timeout_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await inner_task
            with contextlib.suppress(asyncio.CancelledError):
                await timeout_task
            raise
        finally:
            if not timeout_task.done():
                timeout_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await timeout_task

        if inner_task not in _done:
            # Timeout fired first — mark timed_out BEFORE cancelling so the
            # coroutine's CancelledError handler can distinguish the two cases.
            if task_model.status not in TERMINAL_STATES:
                task_model.status = "timed_out"
                task_model.error = f"Task timed out after {self._timeout_seconds}s"
                task_model.completed_at = datetime.now(tz=UTC)
            inner_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await inner_task
            raise TimeoutError

        return inner_task.result()

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    async def cancel_task(self, task_id: str) -> str:
        """Cancel a task by id.

        Returns a descriptive message about what happened.  Terminal-state
        tasks are not modified.
        """
        with logfire.span("background_task.manager.cancel_task", task_id=task_id):
            return await self._cancel_task_impl(task_id)

    async def _cancel_task_impl(self, task_id: str) -> str:
        task_model = self._tasks.get(task_id)
        if task_model is None:
            return f"Task {task_id!r} not found"

        if task_model.status in TERMINAL_STATES:
            return f"Task {task_id!r} is already {task_model.status}"

        # Pending — cancel directly without asyncio.Task.cancel().
        if task_model.status == "pending":
            task_model.status = "cancelled"
            task_model.completed_at = datetime.now(tz=UTC)
            handle = self._handles.get(task_id)
            if handle is not None:
                # Fire on_completed before setting the event, matching the
                # running path (where _execute_task's finally fires it).
                if handle.on_completed is not None:
                    handle.on_completed()
                handle.completion_event.set()
            self._schedule_cleanup(task_id)
            return f"Task {task_id!r} cancelled while pending"

        # Running — transition to cancelling, then await with timeout.
        task_model.status = "cancelling"
        handle = self._handles.get(task_id)
        if handle is None:
            # No handle yet (shouldn't happen, but defensive).
            task_model.status = "cancelled"
            task_model.completed_at = datetime.now(tz=UTC)
            self._schedule_cleanup(task_id)
            return f"Task {task_id!r} cancelled (no handle)"

        handle.task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.shield(handle.task),
                timeout=self._cancel_timeout_seconds,
            )
        except TimeoutError:
            # Cancellation did not complete in time.
            if task_model.status not in TERMINAL_STATES:
                task_model.status = "error"
                task_model.error = (
                    f"Cancellation of task {task_id!r} timed out after "
                    f"{self._cancel_timeout_seconds}s"
                )
                task_model.completed_at = datetime.now(tz=UTC)
                handle.completion_event.set()
                self._schedule_cleanup(task_id)
            return f"Task {task_id!r} cancellation timed out; marked as error"
        except asyncio.CancelledError:
            # We ourselves were cancelled while waiting.
            if task_model.status not in TERMINAL_STATES:
                task_model.status = "cancelled"
                task_model.completed_at = datetime.now(tz=UTC)
                handle.completion_event.set()
                self._schedule_cleanup(task_id)
            return f"Task {task_id!r} cancelled"

        # Task completed (either via CancelledError in _execute_task or
        # finished normally before cancellation took effect).
        # Re-read status into a local to avoid mypy narrowing from "cancelling".
        post_cancel_status: TaskStatus = task_model.status
        if post_cancel_status == "cancelled":
            return f"Task {task_id!r} cancelled successfully"
        if post_cancel_status in TERMINAL_STATES:
            return (
                f"Task {task_id!r} completed as {task_model.status} before cancellation took effect"
            )
        # Should not reach here, but be safe.
        return f"Task {task_id!r} cancel processed (status={task_model.status})"

    async def cancel_all(self) -> int:
        """Cancel all non-terminal tasks.

        Returns the count of tasks that were cancelled.
        """
        cancelled_count = 0
        for task_id, task_model in list(self._tasks.items()):
            if task_model.status not in TERMINAL_STATES:
                await self.cancel_task(task_id)
                cancelled_count += 1
        return cancelled_count

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_task(self, task_id: str) -> BackgroundTask | None:
        """Return the task model, or ``None`` if not found."""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[BackgroundTask]:
        """Return all tasks in the registry."""
        return list(self._tasks.values())

    # ------------------------------------------------------------------
    # Blocking-waiter tracking
    # ------------------------------------------------------------------

    def register_blocking_waiter(self, task_id: str) -> str | None:
        """Register that a ``background_output(block=True)`` is actively waiting.

        Returns a waiter token that must be passed to
        ``unregister_blocking_waiter``, or ``None`` if the task has no
        handle (e.g., already cleaned up).  Safe to call from the asyncio
        event loop — no external locking needed.
        """
        handle = self._handles.get(task_id)
        if handle is None:
            return None
        waiter_id = uuid.uuid4().hex[:12]
        handle.blocking_waiter_id = waiter_id
        return waiter_id

    def unregister_blocking_waiter(self, task_id: str, waiter_id: str) -> None:
        """Unregister a blocking waiter, but only if the token still matches.

        This prevents a stale unregister from clearing a newer waiter.
        """
        handle = self._handles.get(task_id)
        if handle is not None and handle.blocking_waiter_id == waiter_id:
            handle.blocking_waiter_id = None

    def has_blocking_waiter(self, task_id: str) -> bool:
        """Return ``True`` if a blocking ``background_output`` is waiting on this task."""
        handle = self._handles.get(task_id)
        return handle is not None and handle.blocking_waiter_id is not None

    async def wait_for_task(
        self,
        task_id: str,
        timeout_seconds: float = 60.0,
    ) -> BackgroundTask | None:
        """Wait for a task to reach a terminal state.

        Returns the updated task model, or the current task if the wait
        timeout expires (without cancelling the task).

        Uses ``asyncio.wait`` with a sleep task instead of
        ``asyncio.wait_for`` + ``Event.wait()``.  ``asyncio.wait_for``
        is broken in Python 3.12+ when nested inside another
        ``asyncio.wait_for`` (e.g., agent event loop polling with
        ``wait_for(event_queue.get(), timeout=0.1)``).  The outer
        ``wait_for`` can convert external ``CancelledError`` into
        ``TimeoutError``, preventing the inner timeout from ever firing.
        ``asyncio.wait`` does NOT use cancellation internally, so it is
        safe.
        """
        with logfire.span(
            "background_task.manager.wait_for_task",
            task_id=task_id,
            timeout_seconds=timeout_seconds,
        ):
            return await self._wait_for_task_impl(task_id, timeout_seconds)

    async def _wait_for_task_impl(
        self,
        task_id: str,
        timeout_seconds: float,
    ) -> BackgroundTask | None:
        task_model = self._tasks.get(task_id)
        if task_model is None:
            return None

        handle = self._handles.get(task_id)
        if handle is None:
            return task_model

        wait_task = asyncio.ensure_future(handle.completion_event.wait())
        timeout_task = asyncio.ensure_future(asyncio.sleep(timeout_seconds))

        try:
            _done, pending = await asyncio.wait(
                {wait_task, timeout_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            # If we ourselves are cancelled, clean up and return current state.
            wait_task.cancel()
            timeout_task.cancel()
            for t in (wait_task, timeout_task):
                with contextlib.suppress(asyncio.CancelledError):
                    await t
            return self._tasks.get(task_id)

        # Cancel any pending tasks.
        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        return self._tasks.get(task_id)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _schedule_cleanup(self, task_id: str) -> None:
        """Schedule removal of a terminal task after the retention period."""
        if task_id in self._cleanup_scheduled:
            return
        self._cleanup_scheduled.add(task_id)

        async def _cleanup() -> None:
            await asyncio.sleep(self._cleanup_after_seconds)
            self._tasks.pop(task_id, None)
            self._handles.pop(task_id, None)
            self._cleanup_scheduled.discard(task_id)

        with logfire.span(
            "background_task.manager.schedule_cleanup",
            task_id=task_id,
        ):
            cleanup_task = asyncio.create_task(_cleanup())
            self._cleanup_tasks.add(cleanup_task)
            cleanup_task.add_done_callback(self._cleanup_tasks.discard)

    @logfire.instrument("background_task.manager.shutdown")
    async def shutdown(self) -> None:
        """Cancel all running tasks, await completion, and clear registries."""
        for task_id in list(self._tasks.keys()):
            task = self._tasks[task_id]
            if task.status not in TERMINAL_STATES:
                await self.cancel_task(task_id)

        handles = list(self._handles.values())
        for handle in handles:
            if not handle.task.done():
                with contextlib.suppress(asyncio.CancelledError):
                    await handle.task

        for cleanup_task in list(self._cleanup_tasks):
            cleanup_task.cancel()
        for cleanup_task in list(self._cleanup_tasks):
            with contextlib.suppress(asyncio.CancelledError):
                await cleanup_task

        self._tasks.clear()
        self._handles.clear()
        self._cleanup_tasks.clear()
