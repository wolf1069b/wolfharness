"""L2 integration tests for EventManager lifecycle with ManagedTaskGroup.

Tests that the EventManager properly opens and closes its ManagedTaskGroup
during __aenter__/__aexit__, and that event processing tasks are tracked
and cleaned up.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wolfharness.messaging.event_manager import EventManager


pytestmark = pytest.mark.integration


async def test_event_manager_lifecycle_closes_task_group() -> None:
    """Given an EventManager, when entered and exited, event tasks are cleaned up and not busy."""
    manager = EventManager(configs=[], enable_events=True)
    assert len(manager._event_tasks) == 0
    async with manager:
        # No sources added, no tasks running
        assert len(manager._event_tasks) == 0
    # After exit, tasks are cleared
    assert len(manager._event_tasks) == 0


async def test_event_manager_add_source_starts_task_and_cleans_up() -> None:
    """Given an EventManager with a mocked source, add_source starts a task cleaned up on exit."""
    manager = EventManager(configs=[], enable_events=True)

    # Create a mock source that yields no events then ends
    mock_source = MagicMock()

    async def mock_events() -> Any:
        # Empty async generator
        return
        yield  # makes this an async generator

    mock_source.events = mock_events
    mock_source.__aenter__ = AsyncMock(return_value=mock_source)
    mock_source.__aexit__ = AsyncMock(return_value=None)

    # Mock EventSource.from_config to return our mock
    mock_config = MagicMock()
    mock_config.name = "test_source"

    with patch("evented.base.EventSource.from_config", return_value=mock_source):
        async with manager:
            await manager.add_source(mock_config)
            # Task should be started
            assert len(manager._event_tasks) > 0
            # Wait for the task to complete (it will end quickly since mock_events yields nothing)
            await asyncio.gather(*manager._event_tasks, return_exceptions=True)
        # After exit, tasks are cleared
    assert len(manager._event_tasks) == 0


async def test_event_manager_disabled_still_cleans_up() -> None:
    """Given a disabled EventManager, when entered and exited, tasks are properly cleaned up."""
    manager = EventManager(configs=[], enable_events=False)
    async with manager:
        pass
    assert len(manager._event_tasks) == 0


async def test_event_manager_cancels_long_running_tasks() -> None:
    """__aexit__ must cancel long-running event processing tasks.

    Without cancellation, ``asyncio.gather`` in ``__aexit__`` would hang
    waiting for a task that never completes.
    """
    manager = EventManager(configs=[], enable_events=True)
    task_was_cancelled = False

    # Create a mock source with an infinite event stream
    mock_source = MagicMock()

    async def mock_events() -> Any:
        nonlocal task_was_cancelled
        try:
            while True:
                yield MagicMock()  # infinite stream of events
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            task_was_cancelled = True
            raise

    mock_source.events = mock_events
    mock_source.__aenter__ = AsyncMock(return_value=mock_source)
    mock_source.__aexit__ = AsyncMock(return_value=None)

    mock_config = MagicMock()
    mock_config.name = "test_long_running"

    with patch("evented.base.EventSource.from_config", return_value=mock_source):
        async with manager:
            await manager.add_source(mock_config)
            await asyncio.sleep(0.05)  # let the processing task start
            assert len(manager._event_tasks) > 0

    # After __aexit__, the long-running task should have been cancelled
    assert task_was_cancelled, "Long-running event task was not cancelled on __aexit__"
    assert len(manager._event_tasks) == 0
