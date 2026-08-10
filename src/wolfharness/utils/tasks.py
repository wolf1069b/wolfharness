"""TaskManager — removed, use ManagedTaskGroup or asyncio directly.

This module previously provided ``TaskManager`` and ``PrioritizedTask``.
All call sites have been migrated to either ``ManagedTaskGroup``
(see :mod:`wolfharness.utils.task_group`) or direct ``asyncio`` task
tracking via ``MessageNode._pending_tasks`` / ``MessageNode.spawn_task()``.
"""

from __future__ import annotations

from typing import Any


class TaskManager:
    """Stub — TaskManager has been removed.

    Use :class:`wolfharness.utils.task_group.ManagedTaskGroup` for structured
    concurrency, or ``MessageNode.spawn_task()`` / ``MessageNode._pending_tasks``
    for direct asyncio task tracking.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        msg = (
            "TaskManager is removed. Use ManagedTaskGroup "
            "(wolfharness.utils.task_group) or MessageNode.spawn_task() instead."
        )
        raise RuntimeError(msg)
