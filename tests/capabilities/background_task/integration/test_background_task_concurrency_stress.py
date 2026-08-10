"""Concurrent operation stress tests for background task management.

Tests real concurrency behavior of ``BackgroundTaskManager`` and
``BackgroundTaskCapability`` under parallel pressure scenarios:

1. Serial cancellation of multiple slow tasks via ``cancel_all``
2. Concurrent ``wait_for_task`` and ``cancel_task`` on the same task
3. Race between ``wait_for_task`` and automatic cleanup
4. Multi-pending ``force_retrieval`` flow at the capability level
5. Team delegation async path with ``source_type="team_parallel"``
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic_ai._agent_graph import ModelRequestNode
from pydantic_graph import End
import pytest


if TYPE_CHECKING:
    from collections.abc import Callable

from wolfharness import ChatMessage
from wolfharness.agents.base_agent import BaseAgent
from wolfharness.agents.context import AgentContext
from wolfharness.agents.events import StreamCompleteEvent
from wolfharness.capabilities.background_task.capability import (
    BackgroundTaskCapability,
)
from wolfharness.capabilities.background_task.manager import TERMINAL_STATES, BackgroundTaskManager
from wolfharness.capabilities.background_task.types import BackgroundTask
from wolfharness.delegation import AgentPool


# ---------------------------------------------------------------------------
# Shared helpers (adapted from test_background_task_cancellation_regression.py)
# ---------------------------------------------------------------------------


def _make_mock_node(*, name: str = "worker", description: str = "A worker agent") -> MagicMock:
    """Create a mock node that looks like a BaseAgent with run_stream support."""
    node = MagicMock(spec=BaseAgent)
    node.name = name
    node.description = description
    node.model_name = "test:model"
    node.session_id = "ses_parent_123"
    return node


def _make_mock_pool(
    nodes: dict[str, MagicMock] | None = None,
    agent_configs: dict[str, Any] | None = None,
) -> AgentPool:
    """Create a mock AgentPool with the given nodes.

    Args:
        nodes: Node objects for ``pool.nodes``.  Defaults to a single worker.
        agent_configs: Manifest agent configs for ``pool.manifest.agents``
            (e.g. team configs).  When ``None``, defaults to ``nodes``.
    """
    pool = MagicMock(spec=AgentPool)

    if nodes is None:
        mock_node = _make_mock_node()
        nodes = {"worker": mock_node}

    configs = agent_configs if agent_configs is not None else nodes
    pool.manifest = MagicMock()
    pool.manifest.agents = configs
    pool.nodes = nodes
    pool.configure_mock(agent_configs=configs)
    pool.all_agents = list(configs.items())
    pool.teams = {}
    pool.sessions = None
    mock_session_pool = MagicMock()

    async def _get_or_create_session_agent(agent_name: str, agent_type: str):
        return nodes.get(agent_name, next(iter(nodes.values())))

    mock_session_pool.sessions = MagicMock()
    mock_session_pool.sessions.get_or_create_session_agent = AsyncMock(
        side_effect=_get_or_create_session_agent,
    )

    mock_session_pool.inject_prompt = AsyncMock()
    mock_session_pool.steer = AsyncMock()

    async def _send_message(session_id: str, prompt: str, input_provider=None, **kwargs: Any):
        await asyncio.sleep(0.1)
        return MagicMock()

    mock_session_pool.send_message = AsyncMock(side_effect=_send_message)

    async def _subscribe(session_id: str, scope: str = "session"):
        queue = asyncio.Queue()
        await queue.put(
            StreamCompleteEvent(message=ChatMessage(content="Team result", role="assistant"))
        )
        return queue

    mock_session_pool.event_bus = MagicMock()
    mock_session_pool.event_bus.subscribe = AsyncMock(side_effect=_subscribe)
    mock_session_pool.event_bus.unsubscribe = AsyncMock()
    pool.session_pool = mock_session_pool
    return pool


def _make_agent_context(
    pool: AgentPool | None = None,
    data: dict[str, Any] | None = None,
) -> AgentContext:
    """Create a minimal AgentContext for testing."""
    agent = MagicMock(spec=BaseAgent)
    agent.name = "coordinator"
    agent.session_id = "ses_parent_123"
    agent.agent_pool = pool
    agent.inject_prompt = MagicMock()

    ctx = MagicMock(spec=AgentContext)
    ctx.node = agent
    ctx.agent = agent
    ctx.pool = pool
    ctx.data = data if data is not None else {}
    ctx.tool_call_id = "tc_001"
    ctx.run_ctx = None
    ctx.events = MagicMock()
    ctx.events.emit_event = AsyncMock()
    ctx.create_child_session = AsyncMock(return_value="ses_child_456")
    ctx.internal_fs = MagicMock()
    ctx.internal_fs.mkdirs = MagicMock()
    ctx.internal_fs.pipe = MagicMock()

    return ctx


def _wrap_in_run_context(agent_ctx: AgentContext) -> MagicMock:
    """Wrap an AgentContext in a mock RunContext for capability tool methods."""
    run_ctx = MagicMock()
    run_ctx.deps = agent_ctx
    run_ctx.tool_call_id = agent_ctx.tool_call_id
    return run_ctx


async def _collect_stream_events(events: list[Any]):
    """Create an async generator that yields the given events."""
    for event in events:
        yield event


# ===========================================================================
# Test 1: Serial cancellation with slow tasks
# ===========================================================================


@pytest.mark.integration
async def test_cancel_all_serial_cancellation_with_slow_tasks():
    """``cancel_all`` cancels 5 slow tasks serially without hanging.

    Since ``cancel_all`` iterates tasks and calls ``cancel_task``
    sequentially, 5 tasks x 0.5s cancel timeout = 2.5s max.  This is a
    performance concern, not a correctness issue — the test verifies
    it completes within a reasonable bound.
    """
    manager = BackgroundTaskManager(
        max_concurrent_tasks=5,
        cancel_timeout_seconds=0.5,
        timeout_seconds=30,
        cleanup_after_seconds=60,
    )
    try:
        for i in range(5):
            task = BackgroundTask(
                id=f"bg_stress_{i}",
                description=f"stress {i}",
                agent_or_team="test",
                prompt="test",
                parent_session_id=None,
                child_session_id=None,
            )
            manager.register_task(task)

            async def _slow_coro() -> None:
                await asyncio.sleep(0.3)

            manager.start_task(f"bg_stress_{i}", _slow_coro())

        start = time.monotonic()
        result = await manager.cancel_all()
        elapsed = time.monotonic() - start

        # cancel_all returns the count of cancelled tasks (int)
        assert isinstance(result, int)
        assert result >= 1, f"Expected at least 1 cancellation, got {result}"
        assert elapsed < 5.0, f"cancel_all took {elapsed:.2f}s, expected < 5.0s"

        # All tasks should be in terminal states
        for i in range(5):
            t = manager.get_task(f"bg_stress_{i}")
            if t is not None:
                assert t.status in TERMINAL_STATES, (
                    f"bg_stress_{i} has non-terminal status: {t.status}"
                )
    finally:
        await manager.shutdown()


# ===========================================================================
# Test 2: Concurrent block output and cancel on the same task
# ===========================================================================


@pytest.mark.integration
async def test_concurrent_block_output_and_cancel_same_task():
    """Concurrent ``wait_for_task`` and ``cancel_task`` on the same task.

    A blocking wait and a cancellation targeting the same task must
    both complete without hanging.  The task must end in ``cancelled``
    state.
    """
    manager = BackgroundTaskManager(
        max_concurrent_tasks=3,
        cancel_timeout_seconds=5,
        timeout_seconds=30,
        cleanup_after_seconds=60,
    )
    try:
        task = BackgroundTask(
            id="bg_concurrent_cancel",
            description="concurrent cancel target",
            agent_or_team="test",
            prompt="test",
            parent_session_id=None,
            child_session_id=None,
        )
        manager.register_task(task)

        async def _long_coro() -> None:
            await asyncio.sleep(2)

        manager.start_task("bg_concurrent_cancel", _long_coro())

        # Give the task a moment to start running
        await asyncio.sleep(0.05)

        async def _wait_for_task() -> Any:
            return await manager.wait_for_task(
                "bg_concurrent_cancel",
                timeout_seconds=5,
            )

        async def _cancel_after_delay() -> str:
            await asyncio.sleep(0.1)
            return await manager.cancel_task("bg_concurrent_cancel")

        wait_result, cancel_result = await asyncio.gather(
            _wait_for_task(),
            _cancel_after_delay(),
        )

        # wait_for_task should return (not hang)
        assert wait_result is not None, "wait_for_task returned None"
        assert wait_result.status == "cancelled", f"Expected cancelled, got {wait_result.status}"

        # cancel_task should mention cancellation
        assert "cancel" in cancel_result.lower(), f"Unexpected cancel result: {cancel_result}"

        # No exception from either operation — if we got here, both completed
    finally:
        await manager.shutdown()


# ===========================================================================
# Test 3: Race between wait_for_task and cleanup
# ===========================================================================


@pytest.mark.integration
async def test_concurrent_block_output_and_cleanup():
    """``wait_for_task`` wins the race against automatic cleanup.

    A task completes at ~0.02s; cleanup fires at ~0.07s (0.02 + 0.05).
    ``wait_for_task`` started before completion should return the task
    model before cleanup removes it.  After cleanup runs, ``get_task``
    returns None.
    """
    manager = BackgroundTaskManager(
        max_concurrent_tasks=3,
        cancel_timeout_seconds=5,
        timeout_seconds=30,
        cleanup_after_seconds=0.05,
    )
    try:
        task = BackgroundTask(
            id="bg_cleanup_race",
            description="cleanup race target",
            agent_or_team="test",
            prompt="test",
            parent_session_id=None,
            child_session_id=None,
        )
        manager.register_task(task)

        async def _quick_coro() -> None:
            await asyncio.sleep(0.02)

        manager.start_task("bg_cleanup_race", _quick_coro())

        # Wait 0.01s — task has not completed yet
        await asyncio.sleep(0.01)

        # wait_for_task should block until completion (~0.02s) and return
        # the task model before cleanup fires (~0.07s)
        result = await manager.wait_for_task(
            "bg_cleanup_race",
            timeout_seconds=5,
        )

        assert result is not None, "wait_for_task returned None before cleanup"
        assert result.status in TERMINAL_STATES, f"Expected terminal status, got {result.status}"

        # Wait past the cleanup time (0.05s after completion)
        await asyncio.sleep(0.1)

        # get_task should return None — cleanup has run
        assert manager.get_task("bg_cleanup_race") is None, (
            "get_task should return None after cleanup"
        )
    finally:
        await manager.shutdown()


# ===========================================================================
# Test 4: Multi-pending force_retrieval flow
# ===========================================================================


@pytest.mark.integration
async def test_force_retrieval_with_multiple_pending_tasks():
    """``force_retrieval`` intercepts ``End`` and forces ``background_output``.

    With 3 pending retrievals, ``after_node_run`` redirects ``End`` to a
    ``ModelRequestNode`` whose prompt mentions all 3 task IDs.
    ``get_model_settings`` forces ``tool_choice`` to ``background_output``
    while any pending retrievals remain.
    """
    capability = BackgroundTaskCapability(schemas=None, force_retrieval="tool_choice")

    mock_ctx = MagicMock()
    mock_node = MagicMock()
    end_result = End(data=None)

    # Simulate 3 unretrieved task IDs on the session state
    state = capability._get_session_state(mock_ctx)
    state.pending_retrievals = {"bg_aaa", "bg_bbb", "bg_ccc"}

    # after_node_run should intercept End and return a ModelRequestNode
    result = await capability.after_node_run(
        mock_ctx,
        node=mock_node,
        result=end_result,
    )

    assert isinstance(result, ModelRequestNode), (
        f"Expected ModelRequestNode, got {type(result).__name__}"
    )

    content = result.request.parts[0].content
    assert "3 background task" in content, f"Expected '3 background task' in prompt, got: {content}"
    assert "bg_aaa" in content, "bg_aaa missing from prompt"
    assert "bg_bbb" in content, "bg_bbb missing from prompt"
    assert "bg_ccc" in content, "bg_ccc missing from prompt"

    # get_model_settings returns a callable; invoke it with the same mock_ctx
    raw_fn = capability.get_model_settings()
    assert raw_fn is not None, "get_model_settings returned None"
    assert callable(raw_fn), "get_model_settings did not return a callable"
    settings_fn: Callable[[Any], Any] = cast(Any, raw_fn)

    settings = settings_fn(mock_ctx)
    assert settings.get("tool_choice") == ["background_output"], (
        f"Expected tool_choice=['background_output'], got: {settings}"
    )

    # Discard 2 of 3 — should still force tool_choice (1 pending remains)
    state.pending_retrievals.discard("bg_aaa")
    state.pending_retrievals.discard("bg_bbb")

    settings = settings_fn(mock_ctx)
    assert settings.get("tool_choice") == ["background_output"], (
        "Should still force tool_choice with 1 pending retrieval"
    )

    # Discard the last one — should return empty ModelSettings
    state.pending_retrievals.discard("bg_ccc")

    settings = settings_fn(mock_ctx)
    assert len(settings) == 0, f"Expected empty ModelSettings, got: {settings}"


# ===========================================================================
# Test 5: Team delegation async path
# ===========================================================================


@pytest.mark.integration
async def test_team_delegation_async_path():
    """Team config with ``type="team"`` sets ``source_type="team_parallel"``.

    When ``_task(async_mode=True)`` is called with a team-type agent
    config, the internal ``source_type`` is set to ``"team_parallel"``
    (not ``"agent"``).  The task is registered in the manager and
    completes without error.
    """
    capability = BackgroundTaskCapability(schemas=None)

    # Create a mock pool with a team-type agent config
    mock_node = _make_mock_node(name="team_worker")
    team_config = MagicMock()
    team_config.type = "team"
    team_config.description = "A team agent"
    pool = _make_mock_pool(
        nodes={"team_worker": mock_node},
        agent_configs={"team_worker": team_config},
    )

    ctx = _make_agent_context(pool=pool)
    run_ctx = _wrap_in_run_context(ctx)

    # Patch _task_async to capture the source_type argument
    original_task_async = capability._task_async.__func__  # type: ignore[attr-defined]
    captured_source_type: list[str] = []

    async def _capturing_task_async(self_inner, *args, **kwargs):
        captured_source_type.append(
            kwargs.get("source_type", args[3] if len(args) > 3 else "unknown")
        )
        return await original_task_async(capability, *args, **kwargs)

    with (
        patch(
            "wolfharness.capabilities.background_task.capability._generate_task_id",
            return_value="bg_team_001",
        ),
        patch.object(
            type(capability),
            "_task_async",
            _capturing_task_async,
        ),
    ):
        result = await capability._task(
            run_ctx,
            agent="team_worker",
            message="team async task",
            async_mode=True,
        )

    # Assert source_type was "team_parallel" (not "agent")
    assert len(captured_source_type) > 0, "source_type was not captured"
    assert captured_source_type[0] == "team_parallel", (
        f"Expected source_type='team_parallel', got '{captured_source_type[0]}'"
    )

    # Assert task is registered in the manager
    state = capability._get_session_state(ctx)
    task_model = state.task_manager.get_task("bg_team_001")
    assert task_model is not None, "Task not found in manager registry"
    assert task_model.agent_or_team == "team_worker"

    # Assert result contains task ID
    assert "bg_team_001" in result, f"Task ID missing from result: {result}"

    # Wait for the background task to complete
    await asyncio.sleep(0.3)

    # Assert task completed without error
    task_model = state.task_manager.get_task("bg_team_001")
    if task_model is not None:
        assert task_model.status in TERMINAL_STATES, (
            f"Expected terminal status, got {task_model.status}"
        )
        assert task_model.status != "error", (
            f"Task should not be in error state: {task_model.error}"
        )
