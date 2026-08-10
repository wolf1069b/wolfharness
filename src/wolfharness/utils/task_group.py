"""Managed task group wrapping anyio.TaskGroup with queue-then-flush semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self
import warnings

import anyio

from wolfharness.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from datetime import timedelta
    from types import TracebackType


logger = get_logger(__name__)

type _TaskFn = Callable[..., Coroutine[Any, Any, Any]]


class ManagedTaskGroup:
    """Long-lived async task manager wrapping ``anyio.TaskGroup``.

    Supports queue-then-flush semantics (submit tasks before the group is
    entered), fire-and-forget exception isolation, non-closing wait, and
    idempotent close. Deprecated-compat shims mirror the ``TaskManager`` API
    surface to give callers a mechanical migration path.

    Example:
        tg = ManagedTaskGroup()
        tg.start_soon(my_coro_fn, arg1)  # queued before enter
        async with tg:
            tg.start_soon(another_coro_fn)  # runs immediately
            await tg.wait_all()
        # group is closed
    """

    def __init__(self) -> None:
        """Initialize the managed task group with no underlying TaskGroup yet."""
        self._tg: anyio.abc.TaskGroup | None = None  # ty: ignore[unresolved-attribute]
        self._pending: list[tuple[_TaskFn, tuple[Any, ...]]] = []
        self._closed: bool = False
        self._active_count: int = 0
        self._idle_event: anyio.Event | None = None

    async def __aenter__(self) -> Self:
        """Enter the task group, flushing any pending tasks.

        Returns:
            self (the ManagedTaskGroup instance).
        """
        if self._closed:
            msg = "Cannot enter a closed ManagedTaskGroup"
            raise RuntimeError(msg)
        self._tg = anyio.create_task_group()
        await self._tg.__aenter__()
        # Flush pending tasks that were queued before __aenter__.
        pending = self._pending
        self._pending = []
        for fn, args in pending:
            self._start_tracked(fn, *args)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit and close the underlying task group.

        Guards against double-close by checking the ``_closed`` flag.
        Sets ``_closed`` and clears ``_tg`` before awaiting the underlying
        TG exit so the state is consistent even if ``CancelledError``
        interrupts the await.
        """
        if self._closed:
            return
        self._closed = True
        tg = self._tg
        self._tg = None
        if tg is not None:
            await tg.__aexit__(exc_type, exc_val, exc_tb)

    def start_soon(self, fn: _TaskFn, *args: Any) -> None:
        """Submit a task for execution.

        If the task group has not been entered yet, the task is queued and
        flushed on ``__aenter__``. If entered, the task is delegated to the
        underlying ``anyio.TaskGroup.start_soon``.

        Args:
            fn: Coroutine function to run.
            *args: Positional arguments passed to ``fn``.
        """
        if self._closed:
            msg = "Cannot start_soon on a closed ManagedTaskGroup"
            raise RuntimeError(msg)
        if self._tg is None:
            self._pending.append((fn, args))
            return
        self._start_tracked(fn, *args)

    def fire_and_forget(self, coro_fn: _TaskFn, *args: Any) -> None:
        """Submit a task whose exceptions are logged and swallowed.

        Wraps the coroutine in a try/except that logs and swallows exceptions,
        preventing one non-critical task failure from cancelling sibling tasks
        via ExceptionGroup propagation.

        Args:
            coro_fn: Coroutine function to run.
            *args: Positional arguments passed to ``coro_fn``.
        """

        async def _safe_run() -> None:
            try:
                await coro_fn(*args)
            except Exception:
                logger.exception("fire_and_forget task failed")

        self.start_soon(_safe_run)

    def is_busy(self) -> bool:
        """Check if there are active or pending tasks.

        Returns:
            True if there are active tasks or pending tasks, False otherwise.
        """
        return self._active_count > 0 or len(self._pending) > 0

    async def wait_all(self) -> None:
        """Wait for all active tasks to complete without closing the group.

        The group remains open and accepts new tasks after this returns.
        If there are no active tasks, returns immediately.
        """
        if self._active_count == 0:
            return
        if self._idle_event is None:
            self._idle_event = anyio.Event()
        await self._idle_event.wait()
        # Reset the event so future wait_all calls can block again.
        self._idle_event = None

    async def close(self) -> None:
        """Close the group idempotently.

        Safe to call on an unentered, already-closed, or currently-open group.
        """
        await self.__aexit__(None, None, None)

    def _start_tracked(self, fn: _TaskFn, *args: Any) -> None:
        """Start a task on the underlying TG with active-count tracking."""
        assert self._tg is not None
        self._active_count += 1

        async def _tracked() -> None:
            try:
                await fn(*args)
            finally:
                self._active_count -= 1
                if self._active_count == 0 and self._idle_event is not None:
                    self._idle_event.set()

        self._tg.start_soon(_tracked)

    # ------------------------------------------------------------------
    # Deprecated-compat shims (TaskManager API surface)
    # ------------------------------------------------------------------

    def create_task(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        name: str | None = None,
        priority: int = 0,
        delay: timedelta | None = None,
    ) -> None:
        """Deprecated: create and track a task.

        Use ``start_soon`` instead. This shim wraps the coroutine in a
        callable and delegates to ``start_soon``.

        Args:
            coro: Coroutine to run.
            name: Optional name (ignored).
            priority: Priority (ignored).
            delay: Optional delay (ignored).
        """
        warnings.warn(
            "create_task is deprecated. Use start_soon instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        del name, priority, delay  # unused, accepted for API compat

        async def _runner() -> None:
            await coro

        self.start_soon(_runner)

    async def cleanup_tasks(self) -> None:
        """Deprecated: wait for all pending tasks to complete.

        Use ``wait_all`` instead.

        Args:
            None.
        """
        warnings.warn(
            "cleanup_tasks is deprecated. Use wait_all instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        await self.wait_all()

    async def complete_tasks(self, cancel: bool = False) -> None:
        """Deprecated: wait for all tasks to complete.

        Use ``wait_all`` instead.

        Args:
            cancel: If True, ignored (cancellation is handled by ``close``).
        """
        warnings.warn(
            "complete_tasks is deprecated. Use wait_all instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        del cancel  # unused, accepted for API compat
        await self.wait_all()
