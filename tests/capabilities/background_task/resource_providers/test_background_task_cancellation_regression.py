"""Red-flag regression tests for background-task cancellation failures.

Reproduces two reported symptom patterns:
1) One async task unexpectedly becomes ``cancelled`` when multiple are running
   (concurrency-limit / scheduling hypothesis).
2) ``background_output(block=True)`` may interact with task completion in a way
   that looks like cancellation (blocking-waiter hypothesis).

Each test is designed to be HIGH-SIGNAL: if it fails, the failure directly
maps to one of the two hypotheses.  Tests that pass confirm the hypothesis
is not currently reproducible under those conditions.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic_ai import RunContext
import pytest

from wolfharness import ChatMessage
from wolfharness.agents.base_agent import BaseAgent
from wolfharness.agents.context import AgentContext
from wolfharness.agents.events import StreamCompleteEvent
from wolfharness.capabilities.background_task.capability import (
    BackgroundTaskCapability,
)
from wolfharness.capabilities.background_task.manager import BackgroundTaskManager
from wolfharness.capabilities.background_task.types import BackgroundTask
from wolfharness.delegation import AgentPool


pytestmark = pytest.mark.anyio


async def _wait_until_called(mock_obj: Any, timeout: float = 3.0, interval: float = 0.05) -> None:
    elapsed = 0.0
    while mock_obj.call_count == 0 and elapsed < timeout:
        await asyncio.sleep(interval)
        elapsed += interval
    assert mock_obj.call_count > 0, f"Mock was not called within {timeout}s"


def _wrap_in_run_context(agent_ctx):
    """Wrap an AgentContext in a mock RunContext for capability tool methods."""
    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = agent_ctx
    run_ctx.tool_call_id = agent_ctx.tool_call_id
    return run_ctx


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_mock_node(*, name: str = "worker", description: str = "A worker agent") -> MagicMock:
    """Create a mock node that looks like a BaseAgent with run_stream support."""
    node = MagicMock(spec=BaseAgent)
    node.type = "agent"
    node.name = name
    node.description = description
    node.model_name = "test:model"
    node.session_id = "ses_parent_123"
    return node


def _make_mock_pool(nodes: dict[str, MagicMock] | None = None) -> AgentPool:
    """Create a mock AgentPool with the given nodes."""
    pool = MagicMock(spec=AgentPool)

    if nodes is None:
        mock_node = _make_mock_node()
        nodes = {"worker": mock_node}

    pool.manifest = MagicMock()
    pool.manifest.agents = nodes
    pool.nodes = nodes
    pool.agent_configs = nodes
    pool.all_agents = list(nodes.items())
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
    mock_session_pool.followup = AsyncMock(return_value=True)

    async def _send_message(session_id: str, prompt: str, input_provider=None, **kwargs: Any):
        await asyncio.sleep(0.1)
        return MagicMock()

    mock_session_pool.send_message = AsyncMock(side_effect=_send_message)

    async def _subscribe(session_id: str, scope: str = "session"):
        queue = MagicMock()
        queue.get = AsyncMock(side_effect=asyncio.QueueShutDown())
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
    agent.type = "agent"
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

    # Mock AgentRunContext for session state resolution
    mock_run_ctx = MagicMock()
    mock_run_ctx.session_id = "ses_parent_123"
    mock_run_ctx._run_handle = None
    mock_run_ctx.child_done_events = {}
    ctx.run_ctx = mock_run_ctx

    ctx.events = MagicMock()
    ctx.events.emit_event = AsyncMock()
    ctx.create_child_session = AsyncMock(return_value="ses_child_456")
    ctx.internal_fs = MagicMock()
    ctx.internal_fs.mkdirs = MagicMock()
    ctx.internal_fs.pipe = MagicMock()

    return ctx


def _get_tm(capability: BackgroundTaskCapability, ctx: AgentContext):
    """Get the task manager from the capability's session state."""
    return capability._get_session_state(_wrap_in_run_context(ctx)).task_manager


async def _collect_stream_events(events: list[Any]):
    """Create an async generator that yields the given events."""
    for event in events:
        yield event


# ===========================================================================
# HYPOTHESIS 1: Concurrency-limit / scheduling issue
#
# When multiple background tasks are running simultaneously, the
# concurrency semaphore should queue excess tasks — not cancel them.
# ===========================================================================


class TestConcurrencyLimitDoesNotCancelUnrelatedTasks:
    """Regression tests for the concurrency-limit cancellation hypothesis.

    If a task is waiting on the semaphore while other tasks run, it should
    transition to ``running`` once a slot opens — not be cancelled.
    """

    @pytest.mark.unit
    async def test_three_tasks_within_default_concurrency_all_complete(self):
        """Three tasks (within the default limit of 5) must all complete."""
        manager = BackgroundTaskManager(max_concurrent_tasks=5, timeout_seconds=10)

        for i in range(3):
            task_model = BackgroundTask(
                id=f"task-{i}",
                description=f"task {i}",
                agent_or_team="worker",
                prompt="do it",
                parent_session_id=None,
                child_session_id=None,
            )
            manager.register_task(task_model)

            async def _quick_coro() -> None:
                await asyncio.sleep(0.05)

            manager.start_task(f"task-{i}", _quick_coro())

        # Wait for all tasks to complete
        await asyncio.sleep(0.3)

        for i in range(3):
            task_model = manager.get_task(f"task-{i}")
            assert task_model is not None, f"task-{i} missing from registry"
            assert task_model.status == "completed", (
                f"task-{i} expected completed, got {task_model.status}"
            )

    @pytest.mark.unit
    async def test_tasks_exceeding_concurrency_limit_queue_not_cancel(self):
        """When tasks exceed max_concurrent_tasks, the excess must queue,
        not be cancelled.
        """
        # Only 2 concurrent slots — launch 4 tasks
        manager = BackgroundTaskManager(max_concurrent_tasks=2, timeout_seconds=10)

        for i in range(4):
            task_model = BackgroundTask(
                id=f"queued-task-{i}",
                description=f"queued task {i}",
                agent_or_team="worker",
                prompt="do it",
                parent_session_id=None,
                child_session_id=None,
            )
            manager.register_task(task_model)

            async def _slow_coro() -> None:
                await asyncio.sleep(0.1)

            manager.start_task(f"queued-task-{i}", _slow_coro())

        # Wait for all tasks to complete (2 concurrent × 0.1s each = ~0.2s)
        await asyncio.sleep(0.6)

        for i in range(4):
            task_model = manager.get_task(f"queued-task-{i}")
            assert task_model is not None, f"queued-task-{i} missing from registry"
            assert task_model.status == "completed", (
                f"queued-task-{i} expected completed, got {task_model.status}"
            )

    @pytest.mark.unit
    async def test_single_concurrency_slot_three_tasks_all_complete(self):
        """With max_concurrent_tasks=1, three sequential tasks must all complete
        without any being cancelled.
        """
        manager = BackgroundTaskManager(max_concurrent_tasks=1, timeout_seconds=10)

        for i in range(3):
            task_model = BackgroundTask(
                id=f"seq-task-{i}",
                description=f"sequential task {i}",
                agent_or_team="worker",
                prompt="do it",
                parent_session_id=None,
                child_session_id=None,
            )
            manager.register_task(task_model)

            async def _coro() -> None:
                await asyncio.sleep(0.05)

            manager.start_task(f"seq-task-{i}", _coro())

        # Wait for all tasks to complete
        await asyncio.sleep(0.5)

        for i in range(3):
            task_model = manager.get_task(f"seq-task-{i}")
            assert task_model is not None, f"seq-task-{i} missing from registry"
            assert task_model.status == "completed", (
                f"seq-task-{i} expected completed, got {task_model.status}"
            )

    @pytest.mark.unit
    async def test_parallel_background_tasks_via_provider_all_complete(self):
        """Launch 3 background tasks through BackgroundTaskProvider and verify
        none are cancelled.
        """
        capability = BackgroundTaskCapability(schemas=None)
        pool = _make_mock_pool()
        ctx = _make_agent_context(pool=pool)

        mock_node = pool.nodes["worker"]

        task_ids = []

        for i in range(3):
            complete_event = StreamCompleteEvent(
                message=ChatMessage(content=f"Result {i}", role="assistant"),
            )
            mock_node.run_stream = MagicMock(
                return_value=_collect_stream_events([complete_event]),
            )

            with patch(
                "wolfharness.capabilities.background_task.capability._generate_task_id",
                return_value=f"bg_parl{i:02d}",
            ):
                result = await capability._task(
                    _wrap_in_run_context(ctx),
                    agent="worker",
                    message=f"parallel task {i}",
                    async_mode=True,
                )
                # Parse task_id from formatted text output
                match = re.search(r"Task ID: (bg_\w+)", result)
                assert match is not None
                task_ids.append(match.group(1))

        # Wait for all background tasks to complete
        await asyncio.sleep(0.3)

        for _i, task_id in enumerate(task_ids):
            task_model = capability._get_session_state(
                _wrap_in_run_context(ctx)
            ).task_manager.get_task(task_id)
            assert task_model is not None, f"{task_id} missing from registry"
            assert task_model.status == "completed", (
                f"{task_id} expected completed, got {task_model.status}"
            )


# ===========================================================================
# HYPOTHESIS 2: background_output(block=True) interaction
#
# Calling background_output(block=True) for one task should not affect
# the lifecycle of other running tasks.
# ===========================================================================


class TestBlockingOutputDoesNotCancelOtherTasks:
    """Regression tests for the background_output(block=True) hypothesis.

    Blocking on one task's output must not cancel or interfere with
    other concurrently running tasks.
    """

    @pytest.mark.unit
    async def test_blocking_on_task_a_does_not_cancel_task_b(self):
        """When background_output(block=True) is waiting on task A,
        task B must still complete normally.
        """
        capability = BackgroundTaskCapability(schemas=None)
        pool = _make_mock_pool()
        ctx = _make_agent_context(pool=pool)

        mock_node = pool.nodes["worker"]

        # Task A: completes quickly
        complete_a = StreamCompleteEvent(
            message=ChatMessage(content="Result A", role="assistant"),
        )
        mock_node.run_stream = MagicMock(
            return_value=_collect_stream_events([complete_a]),
        )

        with patch(
            "wolfharness.capabilities.background_task.capability._generate_task_id",
            return_value="bg_task_a01",
        ):
            await capability._task(
                _wrap_in_run_context(ctx), agent="worker", message="task A", async_mode=True
            )

        # Task B: takes a bit longer
        async def _slow_stream_b(**kwargs):
            await asyncio.sleep(0.1)
            yield StreamCompleteEvent(
                message=ChatMessage(content="Result B", role="assistant"),
            )

        mock_node.run_stream = MagicMock(return_value=_slow_stream_b())

        with patch(
            "wolfharness.capabilities.background_task.capability._generate_task_id",
            return_value="bg_task_b01",
        ):
            await capability._task(
                _wrap_in_run_context(ctx), agent="worker", message="task B", async_mode=True
            )

        # Block on task A — task B is still running
        result_a = await capability._background_output(
            _wrap_in_run_context(ctx),
            task_id="bg_task_a01",
            block=True,
            timeout_seconds=5.0,
        )
        assert "Task Result" in result_a

        # Wait for task B to finish
        await asyncio.sleep(0.3)

        # Task B must NOT be cancelled
        task_b = _get_tm(capability, ctx).get_task("bg_task_b01")
        assert task_b is not None
        assert task_b.status == "completed", f"task-b expected completed, got {task_b.status}"

    @pytest.mark.unit
    async def test_concurrent_blocking_output_on_two_tasks(self):
        """When background_output(block=True) is called concurrently for two
        tasks, both must complete without cancellation.
        """
        capability = BackgroundTaskCapability(schemas=None)
        pool = _make_mock_pool()
        ctx = _make_agent_context(pool=pool)

        mock_node = pool.nodes["worker"]

        # Both tasks complete after a short delay
        async def _delayed_stream(**kwargs):
            await asyncio.sleep(0.1)
            yield StreamCompleteEvent(
                message=ChatMessage(content="Delayed result", role="assistant"),
            )

        for task_label in ("x", "y"):
            mock_node.run_stream = MagicMock(return_value=_delayed_stream())

            with patch(
                "wolfharness.capabilities.background_task.capability._generate_task_id",
                return_value=f"bg_concur_{task_label}",
            ):
                await capability._task(
                    _wrap_in_run_context(ctx),
                    agent="worker",
                    message=f"concurrent task {task_label}",
                    async_mode=True,
                )

        # Block on both tasks concurrently
        results = await asyncio.gather(
            capability._background_output(
                _wrap_in_run_context(ctx),
                task_id="bg_concur_x",
                block=True,
                timeout_seconds=5.0,
            ),
            capability._background_output(
                _wrap_in_run_context(ctx),
                task_id="bg_concur_y",
                block=True,
                timeout_seconds=5.0,
            ),
        )

        for result in results:
            assert "Task Result" in result

        # Both tasks must be completed (not cancelled)
        for label in ("x", "y"):
            task_model = _get_tm(capability, ctx).get_task(
                f"bg_concur_{label}",
            )
            assert task_model is not None
            assert task_model.status == "completed", (
                f"concurrent-{label} expected completed, got {task_model.status}"
            )

    @pytest.mark.unit
    async def test_blocking_output_while_third_task_running(self):
        """Three-task scenario: block on task A while tasks B and C run
        independently.  This directly reproduces the reported symptom pattern
        where the third task is cancelled.
        """
        capability = BackgroundTaskCapability(schemas=None)
        pool = _make_mock_pool()
        ctx = _make_agent_context(pool=pool)

        mock_node = pool.nodes["worker"]

        # Task A: completes quickly
        complete_a = StreamCompleteEvent(
            message=ChatMessage(content="Result A", role="assistant"),
        )
        mock_node.run_stream = MagicMock(
            return_value=_collect_stream_events([complete_a]),
        )

        with patch(
            "wolfharness.capabilities.background_task.capability._generate_task_id",
            return_value="bg_three_a",
        ):
            await capability._task(
                _wrap_in_run_context(ctx), agent="worker", message="task A", async_mode=True
            )

        # Task B: short delay
        async def _stream_b(**kwargs):
            await asyncio.sleep(0.1)
            yield StreamCompleteEvent(
                message=ChatMessage(content="Result B", role="assistant"),
            )

        mock_node.run_stream = MagicMock(return_value=_stream_b())

        with patch(
            "wolfharness.capabilities.background_task.capability._generate_task_id",
            return_value="bg_three_b",
        ):
            await capability._task(
                _wrap_in_run_context(ctx), agent="worker", message="task B", async_mode=True
            )

        # Task C: slightly longer delay — the "third task" in the reported pattern
        async def _stream_c(**kwargs):
            await asyncio.sleep(0.15)
            yield StreamCompleteEvent(
                message=ChatMessage(content="Result C", role="assistant"),
            )

        mock_node.run_stream = MagicMock(return_value=_stream_c())

        with patch(
            "wolfharness.capabilities.background_task.capability._generate_task_id",
            return_value="bg_three_c",
        ):
            await capability._task(
                _wrap_in_run_context(ctx), agent="worker", message="task C", async_mode=True
            )

        # Block on task A while B and C are still running
        result_a = await capability._background_output(
            _wrap_in_run_context(ctx),
            task_id="bg_three_a",
            block=True,
            timeout_seconds=5.0,
        )
        assert "Task Result" in result_a

        # Wait for B and C
        await asyncio.sleep(0.5)

        # All three must be completed — especially task C (the third task)
        for label in ("a", "b", "c"):
            task_model = _get_tm(capability, ctx).get_task(
                f"bg_three_{label}",
            )
            assert task_model is not None, f"three-{label} missing from registry"
            assert task_model.status == "completed", (
                f"three-{label} expected completed, got {task_model.status}"
            )

    @pytest.mark.unit
    async def test_blocking_waiter_unregister_does_not_affect_other_tasks(self):
        """After background_output(block=True) unregisters its waiter,
        other tasks' completion callbacks must still function correctly.
        """
        capability = BackgroundTaskCapability(schemas=None)
        pool = _make_mock_pool()
        ctx = _make_agent_context(pool=pool)

        mock_node = pool.nodes["worker"]

        # Task 1: complete immediately
        complete_1 = StreamCompleteEvent(
            message=ChatMessage(content="Done 1", role="assistant"),
        )
        mock_node.run_stream = MagicMock(
            return_value=_collect_stream_events([complete_1]),
        )

        with patch(
            "wolfharness.capabilities.background_task.capability._generate_task_id",
            return_value="bg_waiter01",
        ):
            await capability._task(
                _wrap_in_run_context(ctx), agent="worker", message="task 1", async_mode=True
            )

        # Task 2: short delay
        async def _stream_2(**kwargs):
            await asyncio.sleep(0.1)
            yield StreamCompleteEvent(
                message=ChatMessage(content="Done 2", role="assistant"),
            )

        mock_node.run_stream = MagicMock(return_value=_stream_2())

        with patch(
            "wolfharness.capabilities.background_task.capability._generate_task_id",
            return_value="bg_waiter02",
        ):
            await capability._task(
                _wrap_in_run_context(ctx), agent="worker", message="task 2", async_mode=True
            )

        # Block on task 1 — this registers and then unregisters a waiter
        await capability._background_output(
            _wrap_in_run_context(ctx),
            task_id="bg_waiter01",
            block=True,
            timeout_seconds=5.0,
        )

        # Wait for task 2 + 500ms debounce
        await _wait_until_called(pool.session_pool.followup)

        # Task 2 must have had its completion callback fired
        # (followup called because no blocking waiter was present)
        task_2 = _get_tm(capability, ctx).get_task("bg_waiter02")
        assert task_2 is not None
        assert task_2.status == "completed", f"waiter-2 expected completed, got {task_2.status}"

        # followup should have been called for task 2 (no blocking waiter)
        followup_calls = pool.session_pool.followup.call_args_list
        # At least one followup call should mention task 2
        task_2_notified = any("bg_waiter02" in str(call) for call in followup_calls)
        assert task_2_notified, "followup should have been called for waiter-2 completion"


# ===========================================================================
# CROSS-CUTTING: Manager-level stress tests
#
# Directly test the BackgroundTaskManager (without provider) under
# concurrent pressure to isolate scheduling vs. cancellation issues.
# ===========================================================================


class TestManagerConcurrentStress:
    """Stress tests on BackgroundTaskManager to expose scheduling bugs."""

    @pytest.mark.unit
    async def test_rapid_task_launch_all_complete(self):
        """Rapidly launch many tasks; all must complete without cancellation."""
        manager = BackgroundTaskManager(max_concurrent_tasks=3, timeout_seconds=10)
        n_tasks = 10

        for i in range(n_tasks):
            task_model = BackgroundTask(
                id=f"stress-{i}",
                description=f"stress task {i}",
                agent_or_team="worker",
                prompt="do it",
                parent_session_id=None,
                child_session_id=None,
            )
            manager.register_task(task_model)

            async def _coro() -> None:
                await asyncio.sleep(0.02)

            manager.start_task(f"stress-{i}", _coro())

        # Wait for all to complete
        await asyncio.sleep(1.0)

        cancelled_tasks = []
        for i in range(n_tasks):
            task_model = manager.get_task(f"stress-{i}")
            if task_model is not None and task_model.status == "cancelled":
                cancelled_tasks.append(f"stress-{i}")

        assert cancelled_tasks == [], f"Unexpectedly cancelled tasks: {cancelled_tasks}"

    @pytest.mark.unit
    async def test_wait_for_one_task_does_not_cancel_others(self):
        """Calling wait_for_task() on one task must not cancel other running tasks."""
        manager = BackgroundTaskManager(max_concurrent_tasks=5, timeout_seconds=10)

        # Task A: completes quickly
        task_a = BackgroundTask(
            id="wait-a",
            description="wait target",
            agent_or_team="worker",
            prompt="do it",
            parent_session_id=None,
            child_session_id=None,
        )
        manager.register_task(task_a)

        async def _quick() -> None:
            await asyncio.sleep(0.02)

        manager.start_task("wait-a", _quick())

        # Task B: takes longer
        task_b = BackgroundTask(
            id="wait-b",
            description="background task",
            agent_or_team="worker",
            prompt="do it",
            parent_session_id=None,
            child_session_id=None,
        )
        manager.register_task(task_b)

        async def _slow() -> None:
            await asyncio.sleep(0.2)

        manager.start_task("wait-b", _slow())

        # Wait for task A (which completes quickly)
        result = await manager.wait_for_task("wait-a", timeout_seconds=5.0)
        assert result is not None
        assert result.status == "completed"

        # Task B should still be running (not cancelled by the wait)
        task_b_model = manager.get_task("wait-b")
        assert task_b_model is not None
        # It might still be running or completed by now, but NOT cancelled
        assert task_b_model.status != "cancelled", (
            f"wait-b should not be cancelled, got {task_b_model.status}"
        )

        # Wait for task B to finish
        await asyncio.sleep(0.3)
        task_b_model = manager.get_task("wait-b")
        assert task_b_model is not None
        assert task_b_model.status == "completed", (
            f"wait-b expected completed, got {task_b_model.status}"
        )

    @pytest.mark.unit
    async def test_semaphore_acquisition_not_cancelled_by_other_task_completion(self):
        """When task A completes and releases its semaphore slot, task B
        (which was queued) must acquire the slot and run — not be cancelled.
        """
        manager = BackgroundTaskManager(max_concurrent_tasks=1, timeout_seconds=10)

        # Task A: holds the single slot for a while
        task_a = BackgroundTask(
            id="sem-a",
            description="first task",
            agent_or_team="worker",
            prompt="do it",
            parent_session_id=None,
            child_session_id=None,
        )
        manager.register_task(task_a)

        async def _coro_a() -> None:
            await asyncio.sleep(0.1)

        manager.start_task("sem-a", _coro_a())

        # Give task A a moment to acquire the semaphore
        await asyncio.sleep(0.02)

        # Task B: queued behind task A
        task_b = BackgroundTask(
            id="sem-b",
            description="second task",
            agent_or_team="worker",
            prompt="do it",
            parent_session_id=None,
            child_session_id=None,
        )
        manager.register_task(task_b)

        async def _coro_b() -> None:
            await asyncio.sleep(0.05)

        manager.start_task("sem-b", _coro_b())

        # Wait for both to complete
        await asyncio.sleep(0.5)

        for label in ("a", "b"):
            task_model = manager.get_task(f"sem-{label}")
            assert task_model is not None, f"sem-{label} missing from registry"
            assert task_model.status == "completed", (
                f"sem-{label} expected completed, got {task_model.status}"
            )

    @pytest.mark.unit
    async def test_three_concurrent_wait_for_task_calls_do_not_interfere(self):
        """Three concurrent wait_for_task() calls must not interfere with
        each other or cause any task to be cancelled.
        """
        manager = BackgroundTaskManager(max_concurrent_tasks=5, timeout_seconds=10)

        for i in range(3):
            task_model = BackgroundTask(
                id=f"wait-stress-{i}",
                description=f"wait stress {i}",
                agent_or_team="worker",
                prompt="do it",
                parent_session_id=None,
                child_session_id=None,
            )
            manager.register_task(task_model)

            async def _coro() -> None:
                await asyncio.sleep(0.05)

            manager.start_task(f"wait-stress-{i}", _coro())

        # Concurrently wait for all three
        results = await asyncio.gather(
            manager.wait_for_task("wait-stress-0", timeout_seconds=5.0),
            manager.wait_for_task("wait-stress-1", timeout_seconds=5.0),
            manager.wait_for_task("wait-stress-2", timeout_seconds=5.0),
        )

        for i, result in enumerate(results):
            assert result is not None, f"wait-stress-{i} returned None"
            assert result.status == "completed", (
                f"wait-stress-{i} expected completed, got {result.status}"
            )


# ===========================================================================
# EDGE CASE: wait_for inner interaction with cancel_task
#
# Test the interaction between wait_for_task() and cancel_task() to
# ensure that cancelling one task via the API does not accidentally
# cancel a different task that is being waited on.
# ===========================================================================


class TestCancelAndWaitInteraction:
    """Tests for the interaction between cancel_task and wait_for_task."""

    @pytest.mark.unit
    async def test_cancel_task_a_does_not_cancel_task_b_via_wait(self):
        """Cancelling task A while wait_for_task() is blocking on task B
        must not cancel task B.
        """
        manager = BackgroundTaskManager(max_concurrent_tasks=5, timeout_seconds=10)

        # Task A: long-running (will be cancelled)
        task_a = BackgroundTask(
            id="cancel-target",
            description="will be cancelled",
            agent_or_team="worker",
            prompt="do it",
            parent_session_id=None,
            child_session_id=None,
        )
        manager.register_task(task_a)

        async def _long_coro() -> None:
            await asyncio.sleep(100)

        manager.start_task("cancel-target", _long_coro())

        # Task B: short-running (should complete normally)
        task_b = BackgroundTask(
            id="wait-target",
            description="should complete",
            agent_or_team="worker",
            prompt="do it",
            parent_session_id=None,
            child_session_id=None,
        )
        manager.register_task(task_b)

        async def _short_coro() -> None:
            await asyncio.sleep(0.1)

        manager.start_task("wait-target", _short_coro())

        # Let tasks start
        await asyncio.sleep(0.05)

        # Cancel task A while concurrently waiting on task B
        cancel_result, wait_result = await asyncio.gather(
            manager.cancel_task("cancel-target"),
            manager.wait_for_task("wait-target", timeout_seconds=5.0),
        )

        assert "cancel" in cancel_result.lower()
        assert wait_result is not None
        assert wait_result.status == "completed", (
            f"wait-target expected completed, got {wait_result.status}"
        )

        # Task A must be cancelled, task B must be completed
        task_a_model = manager.get_task("cancel-target")
        assert task_a_model is not None
        assert task_a_model.status == "cancelled"

        task_b_model = manager.get_task("wait-target")
        assert task_b_model is not None
        assert task_b_model.status == "completed"
