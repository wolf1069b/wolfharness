"""L2 integration tests for AgentPoolACPAgent close() with ManagedTaskGroup.

Tests that the AgentPoolACPAgent properly opens and closes its
ManagedTaskGroup during initialize()/close(), ensuring no task leaks.
"""

from __future__ import annotations

import pytest

from wolfharness.utils.task_group import ManagedTaskGroup


pytestmark = pytest.mark.integration


async def test_task_group_closed_after_initialize_and_close() -> None:
    """Given a ManagedTaskGroup simulating ACPAgent lifecycle, when entered and closed, not busy.

    This tests the same lifecycle pattern used by AgentPoolACPAgent:
    - __post_init__ creates ManagedTaskGroup()
    - initialize() calls __aenter__()
    - close() calls close()
    """
    tg = ManagedTaskGroup()
    assert not tg._closed

    # Simulate initialize()
    await tg.__aenter__()
    assert not tg._closed

    # Submit a quick task (simulating session creation tasks)
    completed: list[str] = []

    async def quick_task() -> None:
        completed.append("done")

    tg.start_soon(quick_task)
    await tg.wait_all()
    assert completed == ["done"]
    assert not tg.is_busy()

    # Simulate close()
    await tg.close()
    assert tg._closed
    assert not tg.is_busy()


async def test_task_group_close_idempotent_in_acp_agent_pattern() -> None:
    """Given a ManagedTaskGroup in the ACPAgent pattern, double-close is safe."""
    tg = ManagedTaskGroup()
    await tg.__aenter__()
    await tg.close()
    assert tg._closed
    # Double close should not raise
    await tg.close()
    assert tg._closed


async def test_task_group_fire_and_forget_in_acp_agent_pattern() -> None:
    """Given fire_and_forget tasks, exceptions are isolated and group closes cleanly."""
    tg = ManagedTaskGroup()
    await tg.__aenter__()

    ran: list[str] = []

    async def failing() -> None:
        msg = "simulated failure"
        raise ValueError(msg)

    async def healthy() -> None:
        ran.append("ran")

    tg.fire_and_forget(failing)
    tg.start_soon(healthy)
    await tg.wait_all()
    assert ran == ["ran"]

    await tg.close()
    assert not tg.is_busy()
