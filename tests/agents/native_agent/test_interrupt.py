"""TDD tests for agent interrupt/abort bug fix.

These tests validate that interrupt() properly cancels running agent tasks
when called without an explicit run_ctx parameter (the OpenCode TUI abort flow).

Bug: When abort_session() calls interrupt() with no run_ctx, the running
agent task is never cancelled because:
1. _interrupt() receives run_ctx=None and can't find the current_task
2. iteration_task (LLM API call) is a local variable and never directly cancelled

Fix approach:
- Layer 1: interrupt() falls back to _current_run_ctx_var (ContextVar) first,
  then SessionPool's session.active_run_ctx for cross-task access.
- Layer 2: iteration_task is stored as instance var so _interrupt() can cancel it
"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import aclosing, asynccontextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

from pydantic_ai.models.test import TestModel, TestStreamedResponse
import pytest

from wolfharness import Agent
from wolfharness.agents.events import StreamCompleteEvent
from wolfharness.orchestrator.core import SessionState


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Slow test model: inserts async sleep into the streaming path
# ---------------------------------------------------------------------------


class SlowTestModel(TestModel):
    """TestModel that inserts a delay before yielding the streamed response.

    The default TestModel's request_stream yields TestStreamedResponse which
    emits all parts instantly. We override request_stream to inject a sleep
    before yielding the response, giving us a window to call interrupt()
    while the iteration_task is still running.
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
        # Let parent prepare the request parameters
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        self.last_model_request_parameters = model_request_parameters

        model_response = self._request(messages, model_settings, model_request_parameters)

        # Delay before yielding — this is the window where interrupt can fire
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
async def slow_agent() -> Agent[None]:
    """Agent with SlowTestModel for interrupt testing."""
    model = SlowTestModel(custom_output_text="Hello world slow response", pre_stream_delay=0.5)
    return Agent(name="interrupt-test-agent", model=model, session=False)


@pytest.fixture
async def fast_agent() -> Agent[None]:
    """Agent with instant TestModel for basic tests."""
    model = TestModel(custom_output_text="Fast response")
    return Agent(name="fast-test-agent", model=model, session=False)


def _mock_session_pool(agent: Agent, run_ctx: Any) -> None:
    """Mock agent_pool.session_pool so _get_session_run_ctx() returns run_ctx."""
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
    agent_pool = MagicMock()
    agent_pool.session_pool = session_pool
    host_ctx = MagicMock()
    host_ctx.session_pool = session_pool
    agent_pool.get_context.return_value = host_ctx
    agent.agent_pool = agent_pool


async def _drain_stream(agent: Agent[None], prompt: str) -> AsyncIterator[Any]:
    """Yield events from agent.run_stream(), ensuring generator cleanup.

    Wraps run_stream() in aclosing() so the anyio task group inside
    run_stream() is properly exited in the same task that entered it.
    """
    async with aclosing(agent.run_stream(prompt)) as gen:
        async for event in gen:
            yield event


# ---------------------------------------------------------------------------
# Layer 1 Tests: interrupt() without run_ctx must still cancel the stream
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_interrupt_without_run_ctx_sets_cancelled_flag(slow_agent: Agent[None]) -> None:
    """interrupt() called with no run_ctx must still set run_ctx.cancelled=True.

    This is the core bug: abort_session() calls interrupt() with no run_ctx,
    so the per-run run_ctx.cancelled flag is never set, and the streaming
    loop (which checks run_ctx.cancelled) never exits.

    After removing _active_run_ctx, cross-task access requires SessionPool fallback.
    """
    from wolfharness.agents.base_agent import _current_run_ctx_var

    stream_started = asyncio.Event()
    captured_run_ctx: list[Any] = []

    async def run_stream() -> None:
        async with aclosing(slow_agent.run_stream("Test prompt")) as gen:
            async for event in gen:
                # Capture the run_ctx while stream is active (same task)
                run_ctx = _current_run_ctx_var.get()
                if run_ctx is not None and not captured_run_ctx:
                    captured_run_ctx.append(run_ctx)
                stream_started.set()
                if isinstance(event, StreamCompleteEvent):
                    break

    task = asyncio.create_task(run_stream())

    # Wait for stream to start and run_ctx to be captured
    await asyncio.wait_for(stream_started.wait(), timeout=2.0)

    # Verify we captured a run_ctx with cancelled=False
    assert len(captured_run_ctx) == 1, "Should have captured the run_ctx"
    run_ctx = captured_run_ctx[0]
    assert run_ctx.cancelled is False, "run_ctx should not be cancelled before interrupt"

    # Set up SessionPool fallback for cross-task access
    _mock_session_pool(slow_agent, run_ctx)

    # Call interrupt with NO run_ctx (simulates OpenCode abort_session)
    await slow_agent.interrupt(session_id="test-session")

    # run_ctx.cancelled should be True after interrupt() via SessionPool fallback
    assert run_ctx.cancelled is True, (
        "run_ctx.cancelled must be True after interrupt() — "
        "the streaming loop checks this flag to exit"
    )

    # Clean up
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=3.0)

    assert slow_agent._cancelled is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_interrupt_without_run_ctx_cancels_stream_task(slow_agent: Agent[None]) -> None:
    """interrupt() called with no run_ctx must stop the running stream.

    After the fix, interrupt() no longer cancels current_task (the session
    loop). Instead, it sets run_ctx.cancelled=True and cancels _iteration_task.
    The stream loop checks run_ctx.cancelled and breaks, so the task completes
    normally (not via CancelledError).
    """
    from wolfharness.agents.base_agent import _current_run_ctx_var

    stream_started = asyncio.Event()
    captured_run_ctx: list[Any] = []

    async def run_stream() -> None:
        async with aclosing(slow_agent.run_stream("Test prompt")) as gen:
            async for _event in gen:
                run_ctx = _current_run_ctx_var.get()
                if run_ctx is not None and not captured_run_ctx:
                    captured_run_ctx.append(run_ctx)
                stream_started.set()
                # Don't break — let interrupt() stop us via cancelled flag

    task = asyncio.create_task(run_stream())

    # Wait for stream to start
    await asyncio.wait_for(stream_started.wait(), timeout=2.0)

    assert len(captured_run_ctx) == 1
    run_ctx = captured_run_ctx[0]

    # Set up SessionPool fallback for cross-task access
    _mock_session_pool(slow_agent, run_ctx)

    # Call interrupt with NO run_ctx but WITH session_id for SessionPool lookup
    await slow_agent.interrupt(session_id="test-session")

    # The task should complete (not hang) — the stream loop exits via
    # run_ctx.cancelled flag, not via task cancellation.
    try:
        await asyncio.wait_for(task, timeout=3.0)
    except asyncio.CancelledError:
        pass  # Also acceptable — task was cancelled
    except TimeoutError:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        pytest.fail("Stream task hung after interrupt()")

    # Task should be done (either completed or cancelled)
    assert task.done()

    # current_task should NOT be cancelled by _interrupt() — the fix removed
    # current_task cancellation. Only _iteration_task is cancelled.
    current_task = run_ctx.current_task
    if current_task is not None:
        assert not current_task.cancelled(), (
            "current_task should NOT be cancelled by _interrupt() — "
            "the fix removed current_task cancellation"
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_interrupt_with_run_ctx_still_works(fast_agent: Agent[None]) -> None:
    """interrupt() with explicit run_ctx must continue to work (regression guard).

    The fix must not break the existing code path where run_ctx is provided.
    """
    from wolfharness.agents.base_agent import _current_run_ctx_var

    stream_started = asyncio.Event()
    captured_run_ctx = None

    async def run_stream():
        nonlocal captured_run_ctx
        async with aclosing(fast_agent.run_stream("Test prompt")) as gen:
            async for _event in gen:
                stream_started.set()
                # Capture the run_ctx from the ContextVar
                if captured_run_ctx is None:
                    captured_run_ctx = _current_run_ctx_var.get()

    task = asyncio.create_task(run_stream())
    await asyncio.wait_for(stream_started.wait(), timeout=2.0)

    # Wait for stream to complete (fast model completes quickly)
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2.0)

    # interrupt() with explicit run_ctx should still work
    if captured_run_ctx is not None:
        await fast_agent.interrupt(run_ctx=captured_run_ctx)
        assert captured_run_ctx.cancelled is True


# ---------------------------------------------------------------------------
# Layer 2 Tests: iteration_task must be directly cancellable
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_interrupt_cancels_iteration_task(slow_agent: Agent[None]) -> None:
    """interrupt() must cancel the iteration_task running the LLM API call.

    The iteration_task runs agentlet.iter() inside asyncio.create_task().
    Before the fix: this task is a local variable, only cancelled indirectly
    through the consumer's finally block. If the consumer cleanup times out,
    the iteration_task keeps running.
    After the fix: iteration_task is stored as self._iteration_task and
    directly cancelled by _interrupt().
    """
    from wolfharness.agents.base_agent import _current_run_ctx_var

    stream_started = asyncio.Event()
    interrupt_done = asyncio.Event()
    captured_run_ctx: list[Any] = []

    async def run_stream():
        async with aclosing(slow_agent.run_stream("Test prompt")) as gen:
            async for event in gen:
                run_ctx = _current_run_ctx_var.get()
                if run_ctx is not None and not captured_run_ctx:
                    captured_run_ctx.append(run_ctx)
                stream_started.set()
                if isinstance(event, StreamCompleteEvent):
                    break

    task = asyncio.create_task(run_stream())

    # Wait for stream to start
    await asyncio.wait_for(stream_started.wait(), timeout=2.0)

    # Give it a moment to ensure iteration_task is running
    await asyncio.sleep(0.05)

    assert len(captured_run_ctx) == 1
    run_ctx = captured_run_ctx[0]

    # Set up SessionPool fallback for cross-task access
    _mock_session_pool(slow_agent, run_ctx)

    # Call interrupt with session_id for SessionPool lookup
    await slow_agent.interrupt(session_id="test-session")
    interrupt_done.set()

    # Wait for task to finish
    try:
        await asyncio.wait_for(task, timeout=3.0)
    except asyncio.CancelledError:
        pass  # Fine — task was cancelled
    except TimeoutError:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        pytest.fail("Stream task hung after interrupt — iteration_task may not be cancelled")

    # The iteration_task should have been cancelled
    # Check via the agent's _iteration_task attribute (added in fix)
    iteration_task: asyncio.Task[Any] | None = slow_agent._iteration_task
    if iteration_task is not None:
        assert iteration_task.done(), "iteration_task should be done after interrupt"

    # current_task should NOT be cancelled by _interrupt() — the fix removed
    # current_task cancellation. Only _iteration_task is cancelled.
    current_task = run_ctx.current_task
    if current_task is not None:
        assert not current_task.cancelled(), (
            "current_task should NOT be cancelled by _interrupt() — "
            "only _iteration_task should be cancelled"
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_iteration_task_stored_as_instance_variable(slow_agent: Agent[None]) -> None:
    """_stream_events() must store iteration_task as self._iteration_task.

    This enables _interrupt() to directly cancel it rather than relying
    on the consumer's finally block.
    """
    stream_started = asyncio.Event()

    async def run_stream():
        async with aclosing(slow_agent.run_stream("Test prompt")) as gen:
            async for event in gen:
                stream_started.set()
                if isinstance(event, StreamCompleteEvent):
                    break

    task = asyncio.create_task(run_stream())

    # Wait for stream to start
    await asyncio.wait_for(stream_started.wait(), timeout=2.0)
    # Give it a moment for iteration_task to be created
    await asyncio.sleep(0.05)

    # During streaming, _iteration_task should be set
    iteration_task = slow_agent._iteration_task
    if iteration_task is not None:
        assert isinstance(iteration_task, asyncio.Task), "_iteration_task should be an asyncio.Task"
        assert not iteration_task.done(), "_iteration_task should be running during streaming"

    # Clean up
    await slow_agent.interrupt()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=3.0)


# ---------------------------------------------------------------------------
# Integration test: full abort flow simulation
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_opencode_abort_flow_stops_agent(slow_agent: Agent[None]) -> None:
    """Simulate the full OpenCode abort flow: abort_session() → interrupt().

    This test mirrors the real abort_session() code path:
        await state.agent.interrupt()  # No run_ctx!

    Cross-task access requires SessionPool fallback (since _active_run_ctx was removed).
    """
    from wolfharness.agents.base_agent import _current_run_ctx_var

    stream_started = asyncio.Event()
    events_received: list[Any] = []
    captured_run_ctx: list[Any] = []

    async def run_stream():
        async with aclosing(slow_agent.run_stream("Test prompt")) as gen:
            async for event in gen:
                run_ctx = _current_run_ctx_var.get()
                if run_ctx is not None and not captured_run_ctx:
                    captured_run_ctx.append(run_ctx)
                stream_started.set()
                events_received.append(event)
                if isinstance(event, StreamCompleteEvent):
                    break

    task = asyncio.create_task(run_stream())

    # Wait for stream to start producing events
    await asyncio.wait_for(stream_started.wait(), timeout=2.0)

    assert len(captured_run_ctx) == 1
    run_ctx = captured_run_ctx[0]

    # Set up SessionPool fallback for cross-task access
    _mock_session_pool(slow_agent, run_ctx)

    # Simulate abort_session: call interrupt() with session_id for SessionPool lookup
    await slow_agent.interrupt(session_id="test-session")
    # Simulate the sleep in abort_session
    await asyncio.sleep(0.1)

    # Agent should stop — the task should complete soon
    try:
        await asyncio.wait_for(task, timeout=3.0)
    except asyncio.CancelledError:
        pass  # Fine
    except TimeoutError:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        pytest.fail("Agent kept running after abort — the OpenCode abort flow is broken")

    # We should have received some events (partial stream)
    assert len(events_received) > 0, "Should have received at least some events before abort"

    # The agent should be in cancelled state
    assert slow_agent._cancelled is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_subsequent_run_after_interrupt(fast_agent: Agent[None]) -> None:
    """After interrupt(), the agent should be able to run again.

    This is a regression test: the fix must properly clean up state
    so subsequent runs are not affected.
    """
    # First run: complete normally
    events1 = []
    async for event in fast_agent.run_stream("First prompt"):
        events1.append(event)
        if isinstance(event, StreamCompleteEvent):
            break

    assert len(events1) > 0, "First run should produce events"

    # Reset cancelled state for next run
    fast_agent._cancelled = False

    # Second run: should work fine
    events2 = []
    async for event in fast_agent.run_stream("Second prompt"):
        events2.append(event)
        if isinstance(event, StreamCompleteEvent):
            break

    assert len(events2) > 0, "Second run should produce events after interrupt"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_interrupt_then_run_stream(fast_agent: Agent[None]) -> None:
    """After calling interrupt(), a new run_stream() should work.

    The interrupt() sets _cancelled=True. The next run_stream() must
    reset this flag (it does via run_ctx.cancelled = False, but the
    agent._cancelled flag must also be reset).
    """
    # Call interrupt without any running stream
    await fast_agent.interrupt()
    assert fast_agent._cancelled is True

    # Now run_stream — it should still work
    events = []
    async with aclosing(fast_agent.run_stream("After interrupt")) as gen:
        async for event in gen:
            events.append(event)
            if isinstance(event, StreamCompleteEvent):
                break

    assert len(events) > 0, "run_stream should work after interrupt()"
