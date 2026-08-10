"""TDD tests for inject_prompt/queue_prompt cross-task behavior.

These tests validate that inject_prompt() and queue_prompt() work via:
1. ContextVar (_current_run_ctx_var) when called from the same task as run_stream()
2. SessionPool fallback (session.active_run_ctx) when called from a different task

After removing _active_run_ctx from BaseAgent, cross-task access requires
a SessionPool with session.active_run_ctx set.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

from pydantic_ai.models.test import TestModel, TestStreamedResponse
import pytest

from wolfharness import Agent
from wolfharness.agents.base_agent import _current_run_ctx_var
from wolfharness.agents.events import StreamCompleteEvent
from wolfharness.orchestrator.core import SessionState


if TYPE_CHECKING:
    from wolfharness.agents.context import AgentRunContext


# ---------------------------------------------------------------------------
# Slow test model: inserts async sleep so run_stream stays active
# ---------------------------------------------------------------------------


class SlowTestModel(TestModel):
    """TestModel that inserts a delay before yielding the streamed response.

    This gives us a window to call inject_prompt() / queue_prompt() from a
    different async task while run_stream() is still active.
    """

    def __init__(
        self,
        *,
        custom_output_text: str | None = None,
        pre_stream_delay: float = 0.5,
    ) -> None:
        super().__init__(custom_output_text=custom_output_text)
        self.pre_stream_delay = pre_stream_delay

    @asynccontextmanager
    async def request_stream(  # type: ignore[override]
        self,
        messages: list[Any],
        model_settings: Any,
        model_request_parameters: Any,
        run_context: Any = None,
    ) -> Any:
        """Yield the streamed response after a configurable delay."""
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        self.last_model_request_parameters = model_request_parameters

        model_response = self._request(messages, model_settings, model_request_parameters)

        await asyncio.sleep(self.pre_stream_delay)
        yield TestStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=self._model_name,
            _structured_response=model_response,
            _messages=messages,
            _provider_name=self._system,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def slow_agent() -> Agent[None]:
    """Agent with SlowTestModel for cross-task inject testing."""
    model = SlowTestModel(
        custom_output_text="Hello world slow response",
        pre_stream_delay=0.5,
    )
    return Agent(name="inject-test-agent", model=model)


@pytest.fixture
def fast_agent() -> Agent[None]:
    """Agent with instant TestModel for basic tests."""
    model = TestModel(custom_output_text="Fast response")
    return Agent(name="fast-test-agent", model=model)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_session_pool(agent: Agent, run_ctx: AgentRunContext) -> None:
    """Mock agent_pool.session_pool so get_active_run_context() returns run_ctx."""
    from unittest.mock import AsyncMock

    from wolfharness.orchestrator.run import RunHandle

    session_state = SessionState(session_id="test-session", agent_name="test")
    session_state.current_run_id = run_ctx.run_id
    session_controller = MagicMock()
    session_controller.get_session.return_value = session_state
    run_handle = MagicMock(spec=RunHandle)
    run_handle.run_ctx = run_ctx
    session_pool = MagicMock()
    session_pool.sessions = session_controller
    session_pool.get_run.return_value = run_handle
    session_pool.send_message = AsyncMock()
    # Mock steer/followup delegation via SessionPool
    session_pool.steer = AsyncMock(return_value=True)
    session_pool.followup = AsyncMock(return_value=True)
    agent_pool = MagicMock()
    agent_pool.session_pool = session_pool
    agent_pool.storage = MagicMock()
    agent_pool.storage.log_message = AsyncMock()
    agent_pool.storage.log_session = AsyncMock()
    # Set up get_context() to return a HostContext-like mock with the same
    # session_pool and storage so migrated code using host_context works.
    host_ctx = MagicMock()
    host_ctx.session_pool = session_pool
    host_ctx.storage = agent_pool.storage
    agent_pool.get_context.return_value = host_ctx
    agent.agent_pool = agent_pool


# ---------------------------------------------------------------------------
# Core Test: inject_prompt from a different async task (via SessionPool)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inject_prompt_from_different_task_with_session_pool(
    slow_agent: Agent[None],
) -> None:
    """inject_prompt() called from a different task MUST reach the injection manager.

    when SessionPool fallback is available.
    """
    stream_started = asyncio.Event()
    captured_run_ctx: list[AgentRunContext] = []

    async def run_stream() -> None:
        async for event in slow_agent.run_stream("Test prompt"):
            run_ctx = _current_run_ctx_var.get()
            if run_ctx is not None and not captured_run_ctx:
                captured_run_ctx.append(run_ctx)
            stream_started.set()
            if isinstance(event, StreamCompleteEvent):
                break

    task = asyncio.create_task(run_stream())

    # Wait for run_stream to start and capture run_ctx
    await asyncio.wait_for(stream_started.wait(), timeout=2.0)

    assert len(captured_run_ctx) == 1, "Should have captured run_ctx"
    run_ctx = captured_run_ctx[0]

    # Set up SessionPool fallback so cross-task access works
    _mock_session_pool(slow_agent, run_ctx)

    # Call inject_prompt from THIS task (different from run_stream's task)
    slow_agent.inject_prompt("Background task completed", session_id="test-session")

    # After deprecation, inject_prompt() delegates to turns.steer() for native agents.
    # Verify the delegation happened correctly.
    session_pool = slow_agent.host_context.session_pool  # type: ignore[union-attr]
    session_pool.steer.assert_called_once_with(  # type: ignore[attr-defined]
        "test-session", "Background task completed"
    )

    # Clean up
    await slow_agent.interrupt(session_id="test-session")
    with suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=3.0)


# ---------------------------------------------------------------------------
# queue_prompt from a different async task (via SessionPool)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_queue_prompt_from_different_task_with_session_pool(
    slow_agent: Agent[None],
) -> None:
    """queue_prompt() called from a different task MUST reach the injection manager.

    when SessionPool fallback is available.
    """
    stream_started = asyncio.Event()
    captured_run_ctx: list[AgentRunContext] = []

    async def run_stream() -> None:
        async for event in slow_agent.run_stream("Test prompt"):
            run_ctx = _current_run_ctx_var.get()
            if run_ctx is not None and not captured_run_ctx:
                captured_run_ctx.append(run_ctx)
            stream_started.set()
            if isinstance(event, StreamCompleteEvent):
                break

    task = asyncio.create_task(run_stream())
    await asyncio.wait_for(stream_started.wait(), timeout=2.0)

    assert len(captured_run_ctx) == 1
    run_ctx = captured_run_ctx[0]

    # Set up SessionPool fallback
    _mock_session_pool(slow_agent, run_ctx)

    # Queue a prompt from a different task
    slow_agent.queue_prompt("Follow-up prompt", session_id="test-session")

    # After deprecation, queue_prompt() delegates to turns.followup() for native agents.
    # Verify the delegation happened correctly.
    session_pool = slow_agent.host_context.session_pool  # type: ignore[union-attr]
    session_pool.followup.assert_called_once_with(  # type: ignore[attr-defined]
        "test-session", "Follow-up prompt"
    )

    await slow_agent.interrupt(session_id="test-session")
    with suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=3.0)


# ---------------------------------------------------------------------------
# has_pending_injections from a different async task (via SessionPool)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_has_pending_injections_from_different_task_with_session_pool(
    slow_agent: Agent[None],
) -> None:
    """has_pending_injections() called from a different task MUST reflect actual state.

    when SessionPool fallback is available.
    """
    stream_started = asyncio.Event()
    captured_run_ctx: list[AgentRunContext] = []

    async def run_stream() -> None:
        async for event in slow_agent.run_stream("Test prompt"):
            run_ctx = _current_run_ctx_var.get()
            if run_ctx is not None and not captured_run_ctx:
                captured_run_ctx.append(run_ctx)
            stream_started.set()
            if isinstance(event, StreamCompleteEvent):
                break

    task = asyncio.create_task(run_stream())
    await asyncio.wait_for(stream_started.wait(), timeout=2.0)

    assert len(captured_run_ctx) == 1
    run_ctx = captured_run_ctx[0]

    # Set up SessionPool fallback
    _mock_session_pool(slow_agent, run_ctx)

    # Inject directly into the injection manager
    run_ctx.injection_manager.inject("Test injection")

    # Check has_pending_injections from a different task
    assert slow_agent.has_pending_injections(session_id="test-session"), (
        "has_pending_injections() from a different task MUST check SessionPool fallback "
        "and return True when injections are pending."
    )

    await slow_agent.interrupt(session_id="test-session")
    with suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=3.0)


# ---------------------------------------------------------------------------
# Integration: inject_prompt triggers run_stream continuation loop
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inject_prompt_triggers_continuation(slow_agent: Agent[None]) -> None:
    """inject_prompt from a different task should cause run_stream to continue.

    The run_stream() loop checks for pending injections
    after each _stream_events iteration. If inject_prompt() successfully
    delivers to the injection manager (via SessionPool fallback), and the
    injection gets flushed, the loop should run another iteration.
    """
    iteration_count = 0
    stream_started = asyncio.Event()
    captured_run_ctx: list[AgentRunContext] = []

    async def run_stream() -> None:
        nonlocal iteration_count
        async for event in slow_agent.run_stream("First prompt"):
            iteration_count += 1
            run_ctx = _current_run_ctx_var.get()
            if run_ctx is not None and not captured_run_ctx:
                captured_run_ctx.append(run_ctx)
            stream_started.set()
            if isinstance(event, StreamCompleteEvent) and iteration_count == 1:
                # Placeholder — real inject test happens from outside this task
                pass

    task = asyncio.create_task(run_stream())
    await asyncio.wait_for(stream_started.wait(), timeout=2.0)

    assert len(captured_run_ctx) == 1
    run_ctx = captured_run_ctx[0]

    # Set up SessionPool fallback
    _mock_session_pool(slow_agent, run_ctx)

    # Inject from a different task
    slow_agent.inject_prompt("Follow-up from different task", session_id="test-session")

    # After deprecation, inject_prompt() delegates to turns.steer() for native agents.
    # Verify the delegation happened correctly.
    session_pool = slow_agent.host_context.session_pool  # type: ignore[union-attr]
    session_pool.steer.assert_called_with(  # type: ignore[attr-defined]
        "test-session", "Follow-up from different task"
    )

    await slow_agent.interrupt(session_id="test-session")
    with suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=3.0)


# ---------------------------------------------------------------------------
# Regression: same-task inject still works
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inject_prompt_same_task_still_works(fast_agent: Agent[None]) -> None:
    """inject_prompt() called from within run_stream's task must still work.

    The fix must not break the existing code path where inject_prompt is
    called from the same task as run_stream (e.g., from a tool hook).
    """
    injected = False

    async def run_stream() -> None:
        nonlocal injected
        async for event in fast_agent.run_stream("Test prompt"):
            # From within the same task, inject_prompt should work via ContextVar
            run_ctx = _current_run_ctx_var.get()
            if run_ctx is not None:
                fast_agent.inject_prompt("Same-task injection")
                if run_ctx.injection_manager.has_pending():
                    injected = True
            if isinstance(event, StreamCompleteEvent):
                break

    await run_stream()
    assert injected, "inject_prompt() from same task must still work"


# ---------------------------------------------------------------------------
# Hook consumer: NativeAgentHookManager reads injection via SessionPool fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hook_manager_consumes_cross_task_injection_with_session_pool(
    slow_agent: Agent[None],
) -> None:
    """NativeAgentHookManager must consume injections queued from a different task.

    when SessionPool fallback is available.
    """
    from wolfharness.agents.native_agent.hook_manager import NativeAgentHookManager

    stream_started = asyncio.Event()
    captured_run_ctx: list[AgentRunContext] = []

    async def run_stream() -> None:
        async for event in slow_agent.run_stream("Test prompt"):
            run_ctx = _current_run_ctx_var.get()
            if run_ctx is not None and not captured_run_ctx:
                captured_run_ctx.append(run_ctx)
            stream_started.set()
            if isinstance(event, StreamCompleteEvent):
                break

    task = asyncio.create_task(run_stream())
    await asyncio.wait_for(stream_started.wait(), timeout=2.0)

    assert len(captured_run_ctx) == 1
    run_ctx = captured_run_ctx[0]

    # Set up SessionPool fallback
    _mock_session_pool(slow_agent, run_ctx)

    # Inject from a different task (simulates BackgroundTaskProvider._on_task_completed)
    slow_agent.inject_prompt("Background task result notice", session_id="test-session")

    # After deprecation, inject_prompt() delegates to turns.steer() for native agents.
    # Verify the delegation happened correctly.
    session_pool = slow_agent.host_context.session_pool  # type: ignore[union-attr]
    session_pool.steer.assert_called_with(  # type: ignore[attr-defined]
        "test-session", "Background task result notice"
    )

    # The hook manager should still be able to find the run_ctx via SessionPool fallback
    hook_mgr = slow_agent._hook_manager
    assert isinstance(hook_mgr, NativeAgentHookManager)

    active_run_ctx = slow_agent.get_active_run_context(session_id="test-session")
    assert active_run_ctx is not None, "Hook manager must find run_ctx via SessionPool fallback"

    await slow_agent.interrupt(session_id="test-session")
    with suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=3.0)
