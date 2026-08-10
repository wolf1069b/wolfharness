"""Tests for MessageNode.__aexit__ task cleanup."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from wolfharness.messaging.messagenode import MessageNode
from wolfharness.talk.stats import AggregatedMessageStats, MessageStats


pytestmark = pytest.mark.unit


class _CleanupNode(MessageNode[Any, Any]):
    """Minimal concrete MessageNode for testing __aexit__ cleanup."""

    async def get_stats(self) -> MessageStats | AggregatedMessageStats:
        return MessageStats()

    def run_iter(self, *prompts: Any, **kwargs: Any) -> Any:
        pass


async def test_aexit_cancels_pending_tasks() -> None:
    """__aexit__ must cancel pending tasks, not just gather them.

    Without cancel, a long-running task would cause __aexit__ to hang
    because ``asyncio.gather`` waits for completion.
    """
    task_was_cancelled = False

    async def long_running() -> None:
        nonlocal task_was_cancelled
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            task_was_cancelled = True
            raise

    async with _CleanupNode(name="test") as node:
        node.spawn_task(long_running)
        await asyncio.sleep(0.01)  # let the task start

    assert task_was_cancelled, "Pending task was not cancelled on __aexit__"
    assert len(node._pending_tasks) == 0
