"""Tests for RunHandle lifecycle, metrics collection, and ContextVar streaming.

Consolidated from:
- test_run_handle.py (RunHandle lifecycle: creation, start, complete, fail, cancel)
- test_metrics.py (MetricsCollector active runs and agent type breakdown)
- test_contextvar_stream.py (ContextVar compliance during stream execution)
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic_ai.models.test import TestModel
import pytest

from wolfharness import Agent, AgentPool
from wolfharness.agents.base_agent import _current_run_ctx_var
from wolfharness.agents.context import AgentRunContext
from wolfharness.lifecycle import RunOutcome
from wolfharness.orchestrator.core import SessionPool
from wolfharness.orchestrator.metrics import MetricsCollector
from wolfharness.orchestrator.run import RunHandle


pytestmark = [pytest.mark.unit, pytest.mark.anyio]


# ============================================================================
# RunHandle Lifecycle
# ============================================================================


def test_run_handle_defaults() -> None:
    """RunHandle starts with fresh context and complete_event unset."""
    handle = RunHandle(run_id="r1", session_id="s1", agent_type="native")
    assert handle.run_id == "r1"
    assert handle.session_id == "s1"
    assert handle.agent_type == "native"
    assert handle.is_running  # complete_event not set → is_running
    assert handle.run_ctx.current_task is None
    assert not handle.complete_event.is_set()


@pytest.mark.anyio
async def test_start_transitions_to_running() -> None:
    """start() is_running is True for a fresh RunHandle and stores the task."""
    handle = RunHandle(run_id="r1", session_id="s1", agent_type="native")
    task: asyncio.Task[Any] = asyncio.create_task(asyncio.sleep(0))
    handle.run_ctx.current_task = task
    assert handle.is_running
    assert handle.run_ctx.current_task is task
    await task


def test_start_without_task() -> None:
    """RunHandle is_running when no task is provided."""
    handle = RunHandle(run_id="r1", session_id="s1", agent_type="native")
    assert handle.is_running
    assert handle.run_ctx.current_task is None


def test_complete_transitions_and_sets_event() -> None:
    """complete() sets outcome=COMPLETED and sets complete_event."""
    handle = RunHandle(run_id="r1", session_id="s1", agent_type="native")
    handle.complete()
    assert handle.outcome == RunOutcome.COMPLETED
    assert handle.complete_event.is_set()
    assert not handle.is_running


def test_complete_invokes_cleanup_callback() -> None:
    """complete() calls _cleanup_callback before setting complete_event."""
    cleanup_calls: list[str] = []

    def cleanup(run_id: str) -> None:
        cleanup_calls.append(run_id)
        # Event should NOT be set yet during callback
        assert not handle.complete_event.is_set()

    handle = RunHandle(
        run_id="r1",
        session_id="s1",
        agent_type="native",
        _cleanup_callback=cleanup,
    )
    handle.complete()
    assert cleanup_calls == ["r1"]
    assert handle.complete_event.is_set()


def test_fail_transitions_and_sets_event() -> None:
    """fail() sets outcome=FAILED and sets complete_event."""
    handle = RunHandle(run_id="r1", session_id="s1", agent_type="native")
    handle.fail()
    assert handle.outcome == RunOutcome.FAILED
    assert handle.complete_event.is_set()


def test_fail_with_exception_sets_cancelled() -> None:
    """fail(exception) sets the cancelled flag on run_ctx."""
    handle = RunHandle(run_id="r1", session_id="s1", agent_type="native")
    exc = RuntimeError("boom")
    handle.fail(exc)
    assert handle.outcome == RunOutcome.FAILED
    assert handle.run_ctx.cancelled is True


def test_fail_invokes_cleanup_callback() -> None:
    """fail() calls _cleanup_callback before setting complete_event."""
    cleanup_calls: list[str] = []

    def cleanup(run_id: str) -> None:
        cleanup_calls.append(run_id)
        assert not handle.complete_event.is_set()

    handle = RunHandle(
        run_id="r1",
        session_id="s1",
        agent_type="native",
        _cleanup_callback=cleanup,
    )
    handle.fail(ValueError("oops"))
    assert cleanup_calls == ["r1"]
    assert handle.complete_event.is_set()


@pytest.mark.anyio
async def test_cancel_sets_cancelled_flag() -> None:
    """cancel() sets run_ctx.cancelled without calling cleanup."""
    handle = RunHandle(run_id="r1", session_id="s1", agent_type="native")
    task = asyncio.create_task(asyncio.sleep(10))
    handle.run_ctx.current_task = task

    handle.cancel()
    assert handle.run_ctx.cancelled is True
    # complete_event not set by cancel — only by close/complete/fail
    assert not handle.complete_event.is_set()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.anyio
async def test_cancel_does_not_call_cleanup_callback() -> None:
    """cancel() must NOT invoke _cleanup_callback synchronously."""
    cleanup_calls: list[str] = []

    def cleanup(run_id: str) -> None:
        cleanup_calls.append(run_id)

    handle = RunHandle(
        run_id="r1",
        session_id="s1",
        agent_type="native",
        _cleanup_callback=cleanup,
    )
    task = asyncio.create_task(asyncio.sleep(10))
    handle.run_ctx.current_task = task

    handle.cancel()
    assert cleanup_calls == []
    assert not handle.complete_event.is_set()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.anyio
async def test_cancel_no_task_is_safe() -> None:
    """cancel() is safe when no task is stored."""
    handle = RunHandle(run_id="r1", session_id="s1", agent_type="native")
    handle.cancel()
    assert handle.run_ctx.cancelled is True
    assert not handle.complete_event.is_set()


@pytest.mark.anyio
async def test_cancel_done_task_is_safe() -> None:
    """cancel() is safe when the task is already done."""
    handle = RunHandle(run_id="r1", session_id="s1", agent_type="native")
    task = asyncio.create_task(asyncio.sleep(0))
    await task
    handle.run_ctx.current_task = task

    handle.cancel()
    assert handle.run_ctx.cancelled is True
    assert not handle.complete_event.is_set()


# ============================================================================
# Metrics Collection
# ============================================================================


@pytest.fixture
def mock_pool_for_metrics(minimal_pool: AgentPool) -> AgentPool:
    """Return the real AgentPool for metrics testing."""
    return minimal_pool


class TestMetricsCollectorActiveRuns:
    """Tests for MetricsCollector.active_turns and active_runs_by_agent_type."""

    @pytest.mark.anyio
    async def test_get_metrics_returns_zero_initially(
        self, mock_pool_for_metrics: AgentPool
    ) -> None:
        """MetricsCollector should use SessionPool.active_runs for active_turns."""
        session_pool = SessionPool(mock_pool_for_metrics)
        collector = MetricsCollector(session_pool)

        # No active runs initially
        metrics = await collector.get_metrics()
        assert metrics.active_turns == 0
        assert metrics.active_runs_by_agent_type == {}

    @pytest.mark.anyio
    async def test_get_metrics_counts_native_vs_non_native(
        self, mock_pool_for_metrics: AgentPool
    ) -> None:
        """active_runs_by_agent_type should count native and non-native runs."""
        session_pool = SessionPool(mock_pool_for_metrics)
        collector = MetricsCollector(session_pool)

        # Create two sessions: one native (per-session), one non-native
        state_native, _ = await session_pool.sessions.get_or_create_session("sess-native")
        state_native.metadata["agent_type"] = "native"
        handle_native = RunHandle(
            run_id="run-1",
            session_id="sess-native",
            agent_type="native",
        )
        session_pool.sessions._runs["run-1"] = handle_native

        state_non_native, _ = await session_pool.sessions.get_or_create_session("sess-non-native")
        state_non_native.metadata["agent_type"] = "non-native"
        handle_non_native = RunHandle(
            run_id="run-2",
            session_id="sess-non-native",
            agent_type="non-native",
        )
        session_pool.sessions._runs["run-2"] = handle_non_native

        metrics = await collector.get_metrics()
        assert metrics.active_turns == 2
        assert metrics.active_runs_by_agent_type.get("native") == 1
        assert metrics.active_runs_by_agent_type.get("non-native") == 1

        # Cleanup
        session_pool.sessions._runs.clear()
        await session_pool.close_session("sess-native")
        await session_pool.close_session("sess-non-native")


# ============================================================================
# ContextVar Streaming
# ============================================================================


@pytest.fixture
def ctxvar_agent() -> Agent[None]:
    """Agent with instant TestModel for ContextVar testing."""
    model = TestModel(custom_output_text="Hello")
    return Agent(name="ctxvar-test-agent", model=model)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_contextvar_set_during_stream_events(ctxvar_agent: Agent[None]) -> None:
    """_current_run_ctx_var must be non-None during _stream_events and None after."""
    # Before stream starts
    assert _current_run_ctx_var.get() is None

    captured_ctx: AgentRunContext | None = None

    # Fully consume the stream so the generator's finally block runs naturally
    async for _event in ctxvar_agent.run_stream("Test prompt"):
        # During the stream _stream_events is active
        if captured_ctx is None:
            captured_ctx = _current_run_ctx_var.get()
            assert captured_ctx is not None, (
                "_current_run_ctx_var must be set during _stream_events"
            )
            assert isinstance(captured_ctx, AgentRunContext)

    # After stream completes the finally block in run_stream should have reset it
    assert _current_run_ctx_var.get() is None, (
        "_current_run_ctx_var must be reset to None after run_stream completes"
    )
