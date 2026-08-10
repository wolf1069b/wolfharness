"""Tests for uncaught exception propagation and input validation gaps in BackgroundTaskCapability.

Each test documents a specific missing error-handling path in the source:

- ``_format_skills_instructions`` (line 1063): calls ``load_skill()`` with no try/except.
- ``_task`` (lines 422-429): calls ``create_child_session()`` with no try/except.
- ``_task_async`` (line 591): calls ``fs.mkdirs()`` with no try/except.
- ``BackgroundTaskManager.__init__`` (line 40): creates ``asyncio.Semaphore(max_concurrent_tasks)``
  with no guard against 0 (blocks forever) or negative values (raises ``ValueError``).
"""

# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
# Mock-heavy test code: spec'd MagicMock attribute assignment and union member access are expected.

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wolfharness.agents.base_agent import BaseAgent
from wolfharness.agents.context import AgentContext
from wolfharness.capabilities.background_task.capability import (
    BackgroundTaskCapability,
)
from wolfharness.capabilities.background_task.manager import BackgroundTaskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrap_in_run_context(agent_ctx: MagicMock) -> MagicMock:
    """Wrap an AgentContext in a mock RunContext for capability tool methods."""
    run_ctx = MagicMock()
    run_ctx.deps = agent_ctx
    run_ctx.tool_call_id = agent_ctx.tool_call_id
    return run_ctx


def _make_mock_pool() -> MagicMock:
    """Create a mock AgentPool with a single ``test_agent`` node."""
    mock_node = MagicMock(spec=BaseAgent)
    mock_node.name = "test_agent"
    mock_node.description = "A test agent"
    mock_node.model_name = "test:model"
    mock_node.session_id = "ses_parent_123"
    mock_node.type = "agent"

    pool = MagicMock()
    pool.manifest = MagicMock()
    pool.manifest.agents = {"test_agent": mock_node}
    pool.nodes = {"test_agent": mock_node}
    pool.agent_configs = {"test_agent": mock_node}
    pool.all_agents = [("test_agent", mock_node)]
    pool.teams = {}
    pool.sessions = None

    mock_session_pool = MagicMock()
    mock_session_pool.event_bus = MagicMock()
    mock_session_pool.event_bus.subscribe = AsyncMock(return_value=asyncio.Queue())
    mock_session_pool.event_bus.unsubscribe = AsyncMock()
    mock_session_pool.send_message = AsyncMock(return_value=MagicMock())
    mock_session_pool.sessions = MagicMock()
    mock_session_pool.sessions.get_or_create_session_agent = AsyncMock(return_value=mock_node)
    pool.session_pool = mock_session_pool
    return pool


def _make_agent_context(pool: MagicMock) -> MagicMock:
    """Create a minimal mock AgentContext for capability tests."""
    agent = MagicMock(spec=BaseAgent)
    agent.name = "coordinator"
    agent.session_id = "ses_parent_123"
    agent.agent_pool = pool

    ctx = MagicMock(spec=AgentContext)
    ctx.node = agent
    ctx.pool = pool
    ctx.data = {}
    ctx.tool_call_id = "tc_001"
    ctx.run_ctx = None
    ctx.events = MagicMock()
    ctx.events.emit_event = AsyncMock()
    ctx.create_child_session = AsyncMock(return_value="ses_child_456")
    ctx.internal_fs = MagicMock()
    ctx.internal_fs.mkdirs = MagicMock()
    ctx.internal_fs.pipe = MagicMock()
    ctx.input_provider = None
    return ctx


# ---------------------------------------------------------------------------
# Test 1: load_skill exception propagates uncaught (P3)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@patch(
    "wolfharness.capabilities.background_task.capability.load_skill_for_node",
    new_callable=AsyncMock,
)
async def test_load_skill_exception_propagates_uncaught(mock_load_skill: AsyncMock) -> None:
    """``load_skill_for_node`` RuntimeError is caught by ``_format_skills_instructions``.

    ``_format_skills_instructions`` wraps ``load_skill_for_node()`` in try/except.
    A ``RuntimeError`` from ``load_skill_for_node`` is caught and an error string is
    returned in place of instructions — no exception propagates out of
    ``_task()``.
    """
    mock_load_skill.side_effect = RuntimeError("skill not found")

    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool)

    result = await capability._task(
        _wrap_in_run_context(ctx),
        agent="test_agent",
        message="test task",
        load_skills=["nonexistent_skill"],
    )

    # No exception propagates; the error is embedded in the prompt as a string.
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Test 2: create_child_session exception propagates uncaught (P3)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_create_child_session_exception_propagates_uncaught() -> None:
    """``create_child_session`` RuntimeError is caught in ``_task``.

    Lines 422-432 wrap ``await agent_ctx.create_child_session(...)`` in
    try/except.  A ``RuntimeError`` from the session-creation layer is
    caught and an error string is returned — no exception propagates.
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool)
    ctx.create_child_session = AsyncMock(side_effect=RuntimeError("DB connection lost"))

    result = await capability._task(
        _wrap_in_run_context(ctx),
        agent="test_agent",
        message="test task",
    )

    assert isinstance(result, str)
    assert "Failed to create child session" in result
    assert "DB connection lost" in result


# ---------------------------------------------------------------------------
# Test 3: mkdirs exception propagates from _task_async (P3)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_mkdirs_exception_propagates_from_task_async() -> None:
    """``internal_fs.mkdirs`` OSError is caught in ``_task_async``.

    Line 594-597 wraps ``fs.mkdirs(f"/tasks/{task_id}", exist_ok=True)`` in
    try/except.  An ``OSError`` from the filesystem layer is caught and an
    error string is returned — no exception propagates.
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool)
    ctx.internal_fs.mkdirs = MagicMock(side_effect=OSError("disk full"))

    with patch(
        "wolfharness.capabilities.background_task.capability._generate_task_id",
        return_value="bg_mkdir01",
    ):
        result = await capability._task(
            _wrap_in_run_context(ctx),
            agent="test_agent",
            message="test task",
            async_mode=True,
        )

    assert isinstance(result, str)
    assert "Failed to create task directory" in result
    assert "disk full" in result


# ---------------------------------------------------------------------------
# Test 4: max_concurrent_tasks=0 blocks forever (P3)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_max_concurrent_tasks_zero_blocks_forever() -> None:
    """``BackgroundTaskManager(max_concurrent_tasks=0)`` raises ``ValueError``.

    ``__init__`` (line 45) validates ``max_concurrent_tasks < 1`` and raises
    ``ValueError`` immediately, preventing the creation of a ``Semaphore(0)``
    that would block forever.
    """
    with pytest.raises(ValueError, match="max_concurrent_tasks"):
        BackgroundTaskManager(max_concurrent_tasks=0)


# ---------------------------------------------------------------------------
# Test 5: max_concurrent_tasks=-1 raises ValueError (P3)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_max_concurrent_tasks_negative_raises_value_error() -> None:
    """``BackgroundTaskManager(max_concurrent_tasks=-1)`` raises ``ValueError``.

    ``__init__`` (line 45) validates ``max_concurrent_tasks < 1`` and raises
    ``ValueError`` with a message mentioning ``max_concurrent_tasks``, before
    reaching ``asyncio.Semaphore``.
    """
    with pytest.raises(ValueError, match="max_concurrent_tasks"):
        BackgroundTaskManager(max_concurrent_tasks=-1)
