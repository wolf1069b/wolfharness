"""Tests for BaseTeam.stop() exception isolation."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from wolfharness.delegation.base_team import BaseTeam


pytestmark = pytest.mark.unit


async def test_stop_continues_cleanup_on_main_task_error() -> None:
    """stop() must continue cleanup even if _main_task raises a non-CancelledError.

    Regression test: without the ``except Exception`` handler in stop(),
    a task that raised a non-CancelledError during shutdown would propagate
    the exception and skip ``_cleanup_pending_tasks``.
    """
    team = BaseTeam[Any, Any]([], mode="parallel", name="test_team")

    async def stubborn_task() -> Any:
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            msg = "refused to cancel"
            raise ValueError(msg) from None

    team._main_task = asyncio.create_task(stubborn_task())
    await asyncio.sleep(0.01)  # let task start

    # stop() should not raise the ValueError
    await team.stop()

    # Cleanup must have run: _main_task is None, _pending_tasks is cleared
    assert team._main_task is None
    assert len(team._pending_tasks) == 0
