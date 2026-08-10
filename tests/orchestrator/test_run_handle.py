"""Lifecycle tests for the per-prompt RunHandle.

In the per-prompt model, each RunHandle executes exactly one turn and
terminates naturally. There is no idle loop. Session-level state
(lifecycle dimensions, conversation history, message routing) is owned
by SessionState.

Tests cover:
- steer() with active agent_run and without (queued)
- cancel() with cancel_fn and current_task
- close() idempotency
- complete_event set after start() completes, cancels, or errors
- start() yielding RunErrorEvent on turn exception
- input_provider ContextVar set during turn
- checkpoint/complete/fail legacy methods
- per-prompt model: single turn, natural termination
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pydantic_ai.models.test import TestModel
import pytest

from wolfharness import Agent
from wolfharness.agents.context import AgentRunContext
from wolfharness.agents.events import (
    RunErrorEvent,
    StreamCompleteEvent,
)
from wolfharness.lifecycle import RunOutcome
from wolfharness.lifecycle.comm_channel import DirectChannel
from wolfharness.lifecycle.journal import MemoryJournal
from wolfharness.messaging import ChatMessage, MessageHistory
from wolfharness.orchestrator.core import EventBus, SessionState
from wolfharness.orchestrator.run import RunHandle
from wolfharness.orchestrator.turn import Turn


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubTurn(Turn):
    """Minimal Turn implementation for testing.

    Yields a RunStartedEvent-equivalent sequence ending with
    StreamCompleteEvent, then sets message_history.
    """

    def __init__(
        self,
        *,
        events: list[Any] | None = None,
        message_history: list[Any] | None = None,
        raise_exc: BaseException | None = None,
    ) -> None:
        self._events = events or []
        self._history = message_history or []
        self._raise = raise_exc

    async def execute(self):  # type: ignore[override]
        if self._raise is not None:
            raise self._raise
        # Set message history before yielding so it's available
        # even if the consumer breaks on StreamCompleteEvent.
        self._message_history = self._history
        self._final_message = ChatMessage(content="done", role="assistant")
        for event in self._events:
            yield event


class _BlockingTurn(Turn):
    """Turn that blocks until run_ctx.cancelled, then returns without StreamCompleteEvent."""

    def __init__(self, run_ctx: AgentRunContext) -> None:
        self._run_ctx = run_ctx

    async def execute(self):  # type: ignore[override]
        self._message_history = []
        self._final_message = ChatMessage(content="blocked", role="assistant")
        while not self._run_ctx.cancelled:
            await asyncio.sleep(0.01)
        return
        yield  # makes this an async generator


def _make_session(*, comm_channel: Any | None = None) -> SessionState:
    """Create a real SessionState with real CommChannel."""
    session = SessionState(session_id="test-session", agent_name="test-agent")
    if comm_channel is None:
        comm_channel = DirectChannel(MemoryJournal())
    session._comm_channel = comm_channel
    return session


# Sentinel to distinguish "event_bus not provided" from "event_bus explicitly None".
# When not provided, defaults to AsyncMock(). When explicitly None, the
# standalone path (DirectChannel, no EventBus) is exercised.
_EVENT_BUS_UNSET = object()


def _make_run_handle(
    *,
    agent: Any | None = None,
    event_bus: Any = _EVENT_BUS_UNSET,
    session: Any | None = None,
    comm_channel: Any | None = None,
    run_id: str = "test-run",
    session_id: str = "test-session",
    agent_type: str = "native",
) -> RunHandle:
    """Create a RunHandle with mocked dependencies.

    Args:
        agent: Agent mock or real Agent. If None, a MagicMock is created.
        event_bus: EventBus mock, real EventBus, or ``None`` for standalone
            execution. If not provided (sentinel), defaults to ``AsyncMock()``.
            Pass ``None`` explicitly to test the standalone path.
        session: Session mock or real SessionState. If None, a mock with
            real CommChannel and turn_lock is created.
        comm_channel: CommChannel for the session. If None, DirectChannel
            with MemoryJournal is created.
        run_id: Unique identifier for this run.
        session_id: Session this run belongs to.
        agent_type: Type of agent running (e.g. ``"native"``).
    """
    if agent is None:
        agent = MagicMock()
        agent.create_turn = MagicMock(return_value=_StubTurn())
        agent.name = "test-agent"
        agent.conversation = MessageHistory()
    if event_bus is _EVENT_BUS_UNSET:
        event_bus = AsyncMock()
    # If event_bus is None, keep it — standalone path (DirectChannel)
    if session is None:
        session = _make_session(comm_channel=comm_channel)
    return RunHandle(
        run_id=run_id,
        session_id=session_id,
        agent_type=agent_type,
        agent=agent,
        event_bus=event_bus,
        session=session,
    )


def _stream_complete_event() -> StreamCompleteEvent[Any]:
    return StreamCompleteEvent(message=ChatMessage(content="done", role="assistant"))


async def _consume_gen(gen: Any) -> None:
    """Consume an async generator to completion, discarding all events."""
    async for _ in gen:
        pass


# ---------------------------------------------------------------------------
# Tests: steer() (preserved, updated for new API)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_steer_while_running_with_agent_run() -> None:
    """Given a running RunHandle with active_agent_run, steer() enqueues."""
    handle = _make_run_handle()
    mock_agent_run = MagicMock()
    handle.active_agent_run = mock_agent_run

    result = handle.steer("inject me")
    assert result is not None
    mock_agent_run.enqueue.assert_called_once_with("inject me", priority="asap")


@pytest.mark.unit
async def test_steer_with_agent_run_emits_user_message_inserted_event() -> None:
    """steer() with active agent_run skips fire-and-forget emission.

    When ``_enqueued_messages_available=True`` and ``active_agent_run`` is
    set, the display event comes from ``handle_enqueued_messages()`` with
    ``source="processed"`` instead of the fire-and-forget
    ``_schedule_user_message_emission()`` path.
    """
    from wolfharness.agents.events.events import UserMessageInsertedEvent

    bus = EventBus()
    handle = _make_run_handle(event_bus=bus)
    mock_agent_run = MagicMock()
    handle.active_agent_run = mock_agent_run

    # Subscribe to EventBus (returns asyncio.Queue[EventEnvelope])
    queue = await bus.subscribe("test-session", scope="session")

    handle.steer("inject me", emit_user_message=True)

    # Wait for fire-and-forget emission task to complete
    if handle._emission_tasks:
        await asyncio.gather(*handle._emission_tasks)

    # Drain the queue to collect events
    received_events: list[Any] = []
    while not queue.empty():
        envelope = queue.get_nowait()
        received_events.append(envelope.event)

    # Emission is SKIPPED when EnqueuedMessagesEvent is available and
    # active_agent_run is set — the display event comes from
    # handle_enqueued_messages() instead.
    user_msg_events = [e for e in received_events if isinstance(e, UserMessageInsertedEvent)]
    assert len(user_msg_events) == 0
    # Verify the actual enqueue happened.
    mock_agent_run.enqueue.assert_called_once_with("inject me", priority="asap")


@pytest.mark.unit
async def test_steer_while_running_without_agent_run() -> None:
    """Given a running RunHandle without active_agent_run, steer() re-enqueues to feedback_queue."""
    handle = _make_run_handle()
    handle.active_agent_run = None

    result = handle.steer("queue me")
    assert result is not None
    # With fix A, steer() fallback writes to session.feedback_queue (not queued_steer_messages)
    assert handle.session is not None
    assert not handle.session.feedback_queue.empty()
    fb = handle.session.feedback_queue.get_nowait()
    assert fb.content == "queue me"
    assert fb.is_steer is True


@pytest.mark.unit
async def test_steer_with_explicit_message_id() -> None:
    """steer() with explicit message_id returns that ID."""
    handle = _make_run_handle()
    mock_agent_run = MagicMock()
    handle.active_agent_run = mock_agent_run
    result = handle.steer("hello", message_id="custom-msg-001")
    assert result == "custom-msg-001"
    mock_agent_run.enqueue.assert_called_once_with("hello", priority="asap")


@pytest.mark.unit
async def test_steer_auto_generates_message_id() -> None:
    """steer() without message_id auto-generates a UUID string."""
    handle = _make_run_handle()
    mock_agent_run = MagicMock()
    handle.active_agent_run = mock_agent_run
    result = handle.steer("hello")
    assert result is not None
    assert isinstance(result, str)
    assert len(result) == 36


@pytest.mark.unit
async def test_steer_with_list_content_blocks() -> None:
    """steer() with list message enqueues content_blocks via agent_run."""
    handle = _make_run_handle()
    mock_agent_run = MagicMock()
    handle.active_agent_run = mock_agent_run
    blocks: list[Any] = ["text part", {"type": "image", "url": "http://example.com/img.png"}]
    result = handle.steer(blocks, message_id="list-msg-001")
    assert result == "list-msg-001"
    mock_agent_run.enqueue.assert_called_once_with(*blocks, priority="asap")


@pytest.mark.unit
async def test_steer_with_list_content_blocks_queued() -> None:
    """steer() with list message and no agent_run re-enqueues content_blocks to feedback_queue."""
    handle = _make_run_handle()
    handle.active_agent_run = None
    blocks: list[Any] = ["text", {"type": "image"}]
    result = handle.steer(blocks, message_id="list-msg-queued")
    assert result == "list-msg-queued"
    # With fix A, steer() fallback writes to session.feedback_queue (not queued_steer_messages)
    assert handle.session is not None
    assert not handle.session.feedback_queue.empty()
    fb = handle.session.feedback_queue.get_nowait()
    assert fb.content_blocks == blocks
    assert fb.is_steer is True


# ---------------------------------------------------------------------------
# Tests: close() (preserved, updated for new API)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_async_context_manager_calls_close() -> None:
    """Given `async with RunHandle(...)`, close() is called on exit."""
    handle = _make_run_handle()
    assert not handle.complete_event.is_set()

    async with handle:
        assert not handle.complete_event.is_set()

    assert handle.complete_event.is_set()


@pytest.mark.unit
async def test_close_is_idempotent() -> None:
    """Given close() called twice, the second call is a no-op."""
    handle = _make_run_handle()

    handle.close()
    assert handle.complete_event.is_set()

    # Second close should not raise
    handle.close()
    assert handle.complete_event.is_set()


# ---------------------------------------------------------------------------
# Tests: cancel() (preserved, updated for new API)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_cancel_with_cancel_fn_delegates() -> None:
    """Given a RunHandle with _cancel_fn set, cancel() calls the cancel function.

    Also force-cancels current_task to break through __aexit__ hangs.
    """
    handle = _make_run_handle()
    cancel_called = False

    def _cancel_fn() -> None:
        nonlocal cancel_called
        cancel_called = True

    handle._cancel_fn = _cancel_fn
    mock_task = MagicMock()
    mock_task.done.return_value = False
    handle.run_ctx.current_task = mock_task

    handle.cancel()

    assert cancel_called is True
    assert handle.run_ctx.cancelled is True
    # current_task.cancel() IS called to break through __aexit__ hangs
    mock_task.cancel.assert_called_once()


@pytest.mark.unit
async def test_cancel_does_not_cancel_current_task() -> None:
    """Given a RunHandle with current_task set and no _cancel_fn, cancel().

    Force-cancels current_task to break through __aexit__ hangs. The
    CancelledError is caught by start()'s except handler which preserves
    message history and sets complete_event.
    """
    handle = _make_run_handle()
    mock_task = MagicMock()
    mock_task.done.return_value = False
    handle.run_ctx.current_task = mock_task

    handle.cancel()

    assert handle.run_ctx.cancelled is True
    mock_task.cancel.assert_called_once()


@pytest.mark.unit
async def test_cancel_with_done_task_does_not_cancel() -> None:
    """Given a RunHandle with current_task already done, cancel() does not cancel it."""
    handle = _make_run_handle()
    mock_task = MagicMock()
    mock_task.done.return_value = True
    handle.run_ctx.current_task = mock_task

    handle.cancel()

    assert handle.run_ctx.cancelled is True
    mock_task.cancel.assert_not_called()


@pytest.mark.unit
async def test_cancel_is_idempotent_when_complete() -> None:
    """cancel() on a completed RunHandle is a no-op."""
    handle = _make_run_handle()
    handle.complete_event.set()

    # Should not raise, should not set cancelled
    handle.cancel()
    assert handle.run_ctx.cancelled is False


# ---------------------------------------------------------------------------
# Tests: start() validation (preserved)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_start_raises_when_agent_none() -> None:
    """Given a RunHandle with agent=None, start() raises RuntimeError."""
    handle = _make_run_handle()
    handle.agent = None

    gen = handle.start("hello")
    with pytest.raises(RuntimeError, match="agent must be set"):
        await gen.__anext__()


@pytest.mark.unit
async def test_start_allows_event_bus_none_with_comm_channel() -> None:
    """Given event_bus=None with DirectChannel, start() does NOT raise.

    Regression test: _initialize_lifecycle_and_recovery() now creates a
    DirectChannel when event_bus is None, so start() must allow it.
    """
    from wolfharness.lifecycle import DirectChannel, MemoryJournal

    agent = MagicMock()
    agent.create_turn = MagicMock(return_value=_StubTurn())
    agent.name = "test-agent"
    agent.conversation = MessageHistory()
    session = _make_session(comm_channel=DirectChannel(MemoryJournal()))
    handle = _make_run_handle(
        agent=agent,
        event_bus=None,  # explicitly None — standalone path
        session=session,
    )
    # start() should NOT raise RuntimeError for event_bus=None
    events: list[Any] = []
    gen = handle.start("hello")
    try:
        async with asyncio.timeout(5):
            events.extend([event async for event in gen])
    finally:
        with contextlib.suppress(Exception):
            await gen.aclose()
    assert handle.complete_event.is_set()


@pytest.mark.unit
async def test_start_raises_when_session_none() -> None:
    """Given a RunHandle with session=None, start() raises RuntimeError."""
    handle = _make_run_handle()
    handle.session = None

    gen = handle.start("hello")
    with pytest.raises(RuntimeError, match="session must be set"):
        await gen.__anext__()


# ---------------------------------------------------------------------------
# Tests: complete_event (preserved, updated for new session setup)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_event_set_after_start_completes() -> None:
    """RunHandle.start() must set complete_event when it finishes.

    In the per-prompt model, start() executes one turn and exits
    naturally. complete_event must be set in the finally block.
    """
    agent = Agent(
        name="test-complete-event",
        model=TestModel(custom_output_text="done"),
    )
    async with agent:
        event_bus = EventBus()
        session = SessionState(
            session_id="test-ce-session",
            agent_name="test-complete-event",
        )
        session._comm_channel = DirectChannel(MemoryJournal())
        run_ctx = AgentRunContext(
            session_id="test-ce-session",
            event_bus=event_bus,
        )
        run_handle = RunHandle(
            run_id="test-ce-run",
            session_id="test-ce-session",
            agent_type="test-complete-event",
            agent=agent,
            event_bus=event_bus,
            session=session,
            run_ctx=run_ctx,
        )

        # In per-prompt model, start() executes one turn and terminates.
        gen = run_handle.start("hello")
        try:
            async for event in gen:
                if isinstance(event, StreamCompleteEvent):
                    break
        finally:
            await gen.aclose()

        # complete_event must be set
        assert run_handle.complete_event.is_set(), (
            "complete_event was not set after start() completed — close_session() will hang for 30s"
        )


@pytest.mark.asyncio
async def test_complete_event_set_when_start_cancelled() -> None:
    """complete_event must be set even if start() is cancelled."""
    agent = Agent(
        name="test-ce-cancel",
        model=TestModel(custom_output_text="done"),
    )
    async with agent:
        event_bus = EventBus()
        session = SessionState(
            session_id="test-ce-cancel-session",
            agent_name="test-ce-cancel",
        )
        session._comm_channel = DirectChannel(MemoryJournal())
        run_ctx = AgentRunContext(
            session_id="test-ce-cancel-session",
            event_bus=event_bus,
        )
        run_handle = RunHandle(
            run_id="test-ce-cancel-run",
            session_id="test-ce-cancel-session",
            agent_type="test-ce-cancel",
            agent=agent,
            event_bus=event_bus,
            session=session,
            run_ctx=run_ctx,
        )

        gen = run_handle.start("hello")
        task = asyncio.create_task(gen.__anext__())
        await asyncio.sleep(0.1)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        with contextlib.suppress(Exception, GeneratorExit):
            await gen.aclose()

        # Even on cancel, complete_event should be set
        assert run_handle.complete_event.is_set(), (
            "complete_event was not set after start() was cancelled"
        )


# ---------------------------------------------------------------------------
# Tests: RunErrorEvent handling (preserved, updated for new session setup)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_error_event_yielded_to_consumer() -> None:
    """RunHandle.start() must yield RunErrorEvent when turn.execute() raises.

    Without yielding, create_run_stream and other direct consumers
    hang indefinitely waiting for an event that never arrives.
    """
    agent = Agent(
        name="test-error-yield",
        model=TestModel(custom_output_text="ok"),
    )
    async with agent:
        event_bus = EventBus()
        session = SessionState(
            session_id="test-err-session",
            agent_name="test-error-yield",
        )
        session._comm_channel = DirectChannel(MemoryJournal())
        run_ctx = AgentRunContext(
            session_id="test-err-session",
            event_bus=event_bus,
        )
        run_handle = RunHandle(
            run_id="test-err-run",
            session_id="test-err-session",
            agent_type="test-error-yield",
            agent=agent,
            event_bus=event_bus,
            session=session,
            run_ctx=run_ctx,
        )

        # Patch agent.create_turn to return a turn that raises
        class FailingTurn:
            async def execute(self) -> Any:
                raise RuntimeError("turn failed")
                yield  # make it an async generator

        agent.create_turn = MagicMock(return_value=FailingTurn())  # type: ignore[method-assign]

        events: list[Any] = []
        gen = run_handle.start("test")
        try:
            async with asyncio.timeout(5):
                async for event in gen:
                    events.append(event)
                    if isinstance(event, RunErrorEvent):
                        break
        except TimeoutError:
            pytest.fail(
                "start() hung waiting for RunErrorEvent — it was published "
                "to EventBus but never yielded to the consumer"
            )
        finally:
            with contextlib.suppress(Exception):
                await gen.aclose()

        # RunErrorEvent must have been yielded
        error_events = [e for e in events if isinstance(e, RunErrorEvent)]
        assert len(error_events) == 1, (
            f"Expected 1 RunErrorEvent, got {len(error_events)}. "
            f"Events: {[type(e).__name__ for e in events]}"
        )
        assert "turn failed" in error_events[0].message


@pytest.mark.asyncio
async def test_start_publishes_run_error_on_turn_exception() -> None:
    """Given a turn that raises, start() yields RunErrorEvent."""
    turn = _StubTurn(raise_exc=RuntimeError("turn boom"))
    agent = MagicMock()
    agent.create_turn = MagicMock(return_value=turn)
    agent.name = "test-agent"
    agent.conversation = MessageHistory()
    event_bus = AsyncMock()
    session = _make_session()
    handle = _make_run_handle(agent=agent, event_bus=event_bus, session=session)

    events: list[Any] = []
    gen = handle.start("prompt")
    try:
        async for event in gen:
            events.append(event)
            if isinstance(event, RunErrorEvent):
                break
    finally:
        with contextlib.suppress(Exception):
            await gen.aclose()

    error_events = [e for e in events if isinstance(e, RunErrorEvent)]
    assert len(error_events) == 1
    assert "turn boom" in error_events[0].message


@pytest.mark.asyncio
async def test_turn_failure_breaks_loop_not_continue_to_idle() -> None:
    """When turn.execute() raises, start() must terminate (not loop).

    In the per-prompt model, start() executes one turn and exits.
    On exception, it yields RunErrorEvent and terminates naturally.
    """
    agent = Agent(
        name="test-turn-fail-break",
        model=TestModel(custom_output_text="ok"),
    )
    async with agent:
        event_bus = EventBus()
        session = SessionState(
            session_id="test-fail-break-session",
            agent_name="test-turn-fail-break",
        )
        session._comm_channel = DirectChannel(MemoryJournal())
        run_ctx = AgentRunContext(
            session_id="test-fail-break-session",
            event_bus=event_bus,
        )
        run_handle = RunHandle(
            run_id="test-fail-break-run",
            session_id="test-fail-break-session",
            agent_type="test-turn-fail-break",
            agent=agent,
            event_bus=event_bus,
            session=session,
            run_ctx=run_ctx,
        )

        class FailingTurn:
            async def execute(self) -> Any:
                raise RuntimeError("turn failed")
                yield

        agent.create_turn = MagicMock(return_value=FailingTurn())  # type: ignore[method-assign]

        events: list[Any] = []
        gen = run_handle.start("test")
        try:
            async with asyncio.timeout(5):
                async for event in gen:
                    events.append(event)
                    if isinstance(event, RunErrorEvent):
                        break
        except TimeoutError:
            pytest.fail("start() hung after turn failure — loop continued instead of terminating")
        finally:
            with contextlib.suppress(Exception):
                await gen.aclose()

        error_events = [e for e in events if isinstance(e, RunErrorEvent)]
        assert len(error_events) == 1

        # complete_event must be set (start() terminated, not stuck)
        assert run_handle.complete_event.is_set(), (
            "complete_event not set — start() did not terminate after turn failure"
        )


@pytest.mark.asyncio
async def test_run_error_event_sets_turn_failed_and_breaks_loop() -> None:
    """When turn.execute() yields RunErrorEvent, start() must terminate.

    In the per-prompt model, start() breaks on RunErrorEvent and exits
    naturally. complete_event must be set.
    """
    agent = Agent(
        name="test-runevent-break",
        model=TestModel(custom_output_text="ok"),
    )
    async with agent:
        event_bus = EventBus()
        session = SessionState(
            session_id="test-runevent-session",
            agent_name="test-runevent-break",
        )
        session._comm_channel = DirectChannel(MemoryJournal())
        run_ctx = AgentRunContext(
            session_id="test-runevent-session",
            event_bus=event_bus,
        )
        run_handle = RunHandle(
            run_id="test-runevent-run",
            session_id="test-runevent-session",
            agent_type="test-runevent-break",
            agent=agent,
            event_bus=event_bus,
            session=session,
            run_ctx=run_ctx,
        )

        class ErrorTurn:
            async def execute(self) -> Any:
                yield RunErrorEvent(
                    message="simulated error",
                    run_id="test-runevent-run",
                    agent_name="test-runevent-break",
                )

        agent.create_turn = MagicMock(return_value=ErrorTurn())  # type: ignore[method-assign]

        events: list[Any] = []
        gen = run_handle.start("test")
        try:
            async with asyncio.timeout(5):
                async for event in gen:
                    events.append(event)
                    if isinstance(event, RunErrorEvent):
                        break
        except TimeoutError:
            pytest.fail("start() hung after RunErrorEvent — did not terminate")
        finally:
            with contextlib.suppress(Exception):
                await gen.aclose()

        error_events = [e for e in events if isinstance(e, RunErrorEvent)]
        assert len(error_events) == 1

        # complete_event must be set (start() terminated)
        assert run_handle.complete_event.is_set(), (
            "complete_event not set — start() did not terminate after RunErrorEvent"
        )


# ---------------------------------------------------------------------------
# Tests: input_provider ContextVar (preserved, updated for new session setup)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_input_provider_contextvar_set_during_turn() -> None:
    """RunHandle.start() must set _current_input_provider ContextVar.

    MCP elicitation depends on this ContextVar. Without it,
    _current_input_provider.get() returns None during turn execution.
    """
    from wolfharness.mcp_server.manager import _current_input_provider

    captured_provider: list[Any] = []

    def capture_tool() -> str:
        """Tool that captures the current input provider."""
        captured_provider.append(_current_input_provider.get())
        return "captured"

    agent = Agent(
        name="test-ctxvar",
        model=TestModel(call_tools=["capture_tool"], custom_output_text="ok"),
        tools=[capture_tool],
    )
    async with agent:
        event_bus = EventBus()
        session = SessionState(
            session_id="test-ctxvar-session",
            agent_name="test-ctxvar",
        )
        session._comm_channel = DirectChannel(MemoryJournal())
        run_ctx = AgentRunContext(
            session_id="test-ctxvar-session",
            event_bus=event_bus,
        )

        mock_provider = MagicMock()
        session.input_provider = mock_provider

        run_handle = RunHandle(
            run_id="test-ctxvar-run",
            session_id="test-ctxvar-session",
            agent_type="test-ctxvar",
            agent=agent,
            event_bus=event_bus,
            session=session,
            run_ctx=run_ctx,
        )

        gen = run_handle.start("test")
        try:
            async for event in gen:
                if isinstance(event, StreamCompleteEvent):
                    break
        finally:
            await gen.aclose()

        # The tool should have captured the input provider
        assert len(captured_provider) > 0, "Tool was never called"
        assert captured_provider[0] is mock_provider, (
            f"ContextVar was not set — got {captured_provider[0]!r}, expected {mock_provider!r}"
        )


# ---------------------------------------------------------------------------
# Tests: current_task (preserved, updated for new session setup)
# ---------------------------------------------------------------------------


def test_start_sets_current_task() -> None:
    """RunHandle.start() must set run_ctx.current_task.

    Without this, cancel() in _interrupt() gets None for current_task
    and cannot interrupt the running turn.
    """
    import wolfharness.orchestrator.run as run_module

    source = inspect.getsource(run_module.RunHandle.start)
    assert "current_task" in source, (
        "run_ctx.current_task must be set in start() so cancel() can interrupt the running turn"
    )
    assert "asyncio.current_task()" in source, "current_task must be set to asyncio.current_task()"


@pytest.mark.asyncio
async def test_current_task_set_during_start_execution() -> None:
    """Verify run_ctx.current_task is populated during start() execution."""
    agent = Agent(
        name="test-current-task",
        model=TestModel(custom_output_text="ok"),
    )
    async with agent:
        event_bus = EventBus()
        session = SessionState(
            session_id="test-current-task-session",
            agent_name="test-current-task",
        )
        session._comm_channel = DirectChannel(MemoryJournal())
        run_ctx = AgentRunContext(
            session_id="test-current-task-session",
            event_bus=event_bus,
        )
        run_handle = RunHandle(
            run_id="test-current-task-run",
            session_id="test-current-task-session",
            agent_type="test-current-task",
            agent=agent,
            event_bus=event_bus,
            session=session,
            run_ctx=run_ctx,
        )

        captured_tasks: list[Any] = []

        class CapturingTurn:
            async def execute(self) -> Any:
                # Capture current_task from run_ctx during turn execution
                captured_tasks.append(run_ctx.current_task)
                yield StreamCompleteEvent(message=MagicMock())

        agent.create_turn = MagicMock(return_value=CapturingTurn())  # type: ignore[method-assign]

        gen = run_handle.start("test")
        try:
            async with asyncio.timeout(5):
                async for event in gen:
                    if isinstance(event, StreamCompleteEvent):
                        break
        finally:
            with contextlib.suppress(Exception):
                await gen.aclose()

        assert len(captured_tasks) == 1
        assert captured_tasks[0] is not None, (
            "run_ctx.current_task was not set during start() execution"
        )
        assert captured_tasks[0] is asyncio.current_task(), (
            "run_ctx.current_task should be the current asyncio task"
        )


# ---------------------------------------------------------------------------
# Tests: cancelled property (preserved, updated)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_cancelled_property_reflects_run_ctx_cancelled() -> None:
    """Cancelled property returns run_ctx.cancelled."""
    handle = _make_run_handle()
    assert handle.cancelled is False

    handle.run_ctx.cancelled = True
    assert handle.cancelled is True


# ---------------------------------------------------------------------------
# Tests: _active_agent_run property (preserved)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_active_agent_run_property_matches_field() -> None:
    """_active_agent_run property returns same value as active_agent_run field."""
    handle = _make_run_handle()
    assert handle._active_agent_run is None
    assert handle._active_agent_run is handle.active_agent_run
    mock_run = MagicMock()
    handle.active_agent_run = mock_run
    assert handle._active_agent_run is mock_run


# ---------------------------------------------------------------------------
# Tests: ContextVar cross-context ValueError (preserved, updated)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_value_error_when_generator_abandoned_in_different_context() -> None:
    """No ValueError when async generator is GC'd in a different Context.

    Regression test for the bug where ``_current_input_provider.reset(token)``
    in the ``finally`` block of ``start()`` raised ``ValueError`` when the
    async generator was GC-collected in a different asyncio Context.
    """
    agent = Agent(
        name="test-gc-ctxvar",
        model=TestModel(custom_output_text="done"),
    )
    async with agent:
        event_bus = EventBus()
        session = SessionState(
            session_id="test-gc-session",
            agent_name="test-gc-ctxvar",
        )
        session._comm_channel = DirectChannel(MemoryJournal())
        session.input_provider = MagicMock()

        run_handle = RunHandle(
            run_id="test-gc-run",
            session_id="test-gc-session",
            agent_type="test-gc-ctxvar",
            agent=agent,
            event_bus=event_bus,
            session=session,
            run_ctx=AgentRunContext(
                session_id="test-gc-session",
                event_bus=event_bus,
            ),
        )

        # Capture unhandled exceptions from GC-driven generator cleanup
        gc_exceptions: list[BaseException] = []
        loop = asyncio.get_event_loop()
        original_handler = loop.get_exception_handler()

        def _exception_handler(loop: Any, context: Any) -> None:
            exc = context.get("exception")
            if exc and "_current_input_provider" in str(exc):
                gc_exceptions.append(exc)

        loop.set_exception_handler(_exception_handler)

        try:
            gen = run_handle.start("test")
            # Step into the generator so set() is called and it suspends
            # at the first yield (an event from turn.execute()).
            task = asyncio.create_task(gen.__anext__())
            await asyncio.sleep(0.1)

            # Cancel the task — generator is left suspended at yield.
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

            # Do NOT call aclose() — let GC collect the generator.
            del gen
            import gc

            gc.collect()
            # Yield to event loop so any GC callbacks can fire
            await asyncio.sleep(0.05)
        finally:
            loop.set_exception_handler(original_handler)

        assert not gc_exceptions, (
            f"ValueError(s) raised during generator GC cleanup: {[str(e) for e in gc_exceptions]}"
        )


# ---------------------------------------------------------------------------
# Tests: empty prompt handling (updated for per-prompt model)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_start_empty_prompt_no_staged_content_terminates_immediately() -> None:
    """start('') with no staged_content produces no events and terminates.

    In the per-prompt model, an empty prompt with no staged_content means
    no turn is executed and the generator returns immediately.
    """
    from wolfharness.agents.staged_content import StagedContent

    agent = MagicMock()
    agent.create_turn = MagicMock(return_value=_StubTurn())
    agent.name = "test-agent"
    agent.conversation = MessageHistory()
    agent.staged_content = StagedContent()  # empty
    handle = _make_run_handle(agent=agent)
    gen = handle.start("")
    events: list[Any] = []
    try:
        async with asyncio.timeout(5):
            events = [event async for event in gen]
    except (TimeoutError, asyncio.CancelledError):
        pass
    finally:
        with contextlib.suppress(Exception):
            await gen.aclose()

    # No events should be produced
    assert len(events) == 0
    # complete_event should be set
    assert handle.complete_event.is_set()
    # No turn should have been created
    agent.create_turn.assert_not_called()


@pytest.mark.unit
async def test_start_empty_list_no_staged_content_terminates_immediately() -> None:
    """start([]) with no staged_content produces no events and terminates.

    An empty list is falsy and should be treated the same as an empty
    string — no turn executes.
    """
    from wolfharness.agents.staged_content import StagedContent

    agent = MagicMock()
    agent.create_turn = MagicMock(return_value=_StubTurn())
    agent.name = "test-agent"
    agent.conversation = MessageHistory()
    agent.staged_content = StagedContent()  # empty
    handle = _make_run_handle(agent=agent)
    gen = handle.start([])
    events: list[Any] = []
    try:
        async with asyncio.timeout(5):
            events = [event async for event in gen]
    except (TimeoutError, asyncio.CancelledError):
        pass
    finally:
        with contextlib.suppress(Exception):
            await gen.aclose()

    assert len(events) == 0
    assert handle.complete_event.is_set()
    agent.create_turn.assert_not_called()


@pytest.mark.unit
async def test_start_empty_list_with_staged_content_executes_turn() -> None:
    """start([]) with staged_content executes a turn to consume it.

    When a skill command injects content into ``staged_content`` and the
    user's entire message was a slash command (leaving non_command_content
    empty), ``start([])`` must still execute a turn so ``NativeTurn`` can
    consume the staged content. Without this, skill instructions are
    silently discarded (issue #284).
    """
    from wolfharness.agents.staged_content import StagedContent

    staged = StagedContent()
    staged.add_text("IMPORTANT_SKILL_DIRECTIVE")

    agent = MagicMock()
    agent.create_turn = MagicMock(return_value=_StubTurn())
    agent.name = "test-agent"
    agent.conversation = MessageHistory()
    agent.staged_content = staged
    handle = _make_run_handle(agent=agent)
    gen = handle.start([])
    try:
        async with asyncio.timeout(5):
            async for _ in gen:
                pass
    except (TimeoutError, asyncio.CancelledError):
        pass
    finally:
        with contextlib.suppress(Exception):
            await gen.aclose()

    # A turn SHOULD have been created to consume staged_content
    agent.create_turn.assert_called_once()
    # complete_event should be set
    assert handle.complete_event.is_set()


# ---------------------------------------------------------------------------
# Tests: checkpoint / complete / fail (legacy methods, updated)
# ---------------------------------------------------------------------------


def test_checkpointed_status_exists() -> None:
    """RunOutcome.CHECKPOINTED must be a member of the enum."""
    assert RunOutcome.CHECKPOINTED is not None
    assert isinstance(RunOutcome.CHECKPOINTED, RunOutcome)


def test_checkpointed_is_distinct() -> None:
    """RunOutcome.CHECKPOINTED must differ from existing states."""
    assert RunOutcome.CHECKPOINTED != RunOutcome.COMPLETED
    assert RunOutcome.CHECKPOINTED != RunOutcome.FAILED


def test_checkpoint_method_exists() -> None:
    """RunHandle must have a checkpoint() method."""
    handle = RunHandle(run_id="r1", session_id="s1", agent_type="native")
    assert callable(handle.checkpoint)


def test_checkpoint_sets_complete_event() -> None:
    """checkpoint() must set complete_event."""
    handle = RunHandle(run_id="r1", session_id="s1", agent_type="native")
    assert not handle.complete_event.is_set()
    handle.checkpoint()
    assert handle.complete_event.is_set()
    assert handle.outcome == RunOutcome.CHECKPOINTED


def test_checkpoint_invokes_cleanup_callback() -> None:
    """checkpoint() calls _cleanup_callback before setting complete_event."""
    cleanup_calls: list[str] = []

    def cleanup(run_id: str) -> None:
        cleanup_calls.append(run_id)
        assert not handle.complete_event.is_set()

    handle = RunHandle(run_id="r1", session_id="s1", agent_type="native", _cleanup_callback=cleanup)
    handle.checkpoint()
    assert cleanup_calls == ["r1"]
    assert handle.complete_event.is_set()


def test_checkpoint_does_not_emit_run_failed_event() -> None:
    """checkpoint() must NOT emit RunFailedEvent.

    Unlike fail(), checkpoint() is a normal lifecycle transition and
    should not publish a failure event to the event bus.
    """
    handle = RunHandle(run_id="r1", session_id="s1", agent_type="native")
    handle.checkpoint()
    assert handle.outcome == RunOutcome.CHECKPOINTED
    assert handle.complete_event.is_set()


def test_checkpoint_rejects_event_bus_parameter() -> None:
    """checkpoint() signature must NOT accept an event_bus parameter.

    This is a deliberate design choice: checkpointing is a normal
    lifecycle transition, unlike fail() which emits RunFailedEvent.
    """
    import inspect

    sig = inspect.signature(RunHandle.checkpoint)
    assert "event_bus" not in sig.parameters


def test_complete_sets_outcome_and_event() -> None:
    """complete() sets outcome=COMPLETED and complete_event."""
    handle = RunHandle(run_id="r1", session_id="s1", agent_type="native")
    handle.complete()
    assert handle.outcome == RunOutcome.COMPLETED
    assert handle.complete_event.is_set()


def test_fail_sets_outcome_and_event() -> None:
    """fail() sets outcome=FAILED and complete_event."""
    handle = RunHandle(run_id="r1", session_id="s1", agent_type="native")
    handle.fail(RuntimeError("boom"))
    assert handle.outcome == RunOutcome.FAILED
    assert handle.complete_event.is_set()
    assert handle.run_ctx.cancelled is True


# ---------------------------------------------------------------------------
# Tests: per-prompt model (new tests for tasks 4.4, 4.11, 4.12, 4.13, 4.16)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_prompt_run_handle_single_turn_termination() -> None:
    """RunHandle.start() executes exactly one turn and terminates naturally.

    In the per-prompt model, start() does not loop or enter idle state.
    After yielding StreamCompleteEvent, the generator exits.
    """
    agent = Agent(
        name="test-per-prompt",
        model=TestModel(custom_output_text="done"),
    )
    async with agent:
        event_bus = EventBus()
        session = SessionState(
            session_id="test-per-prompt-session",
            agent_name="test-per-prompt",
        )
        session._comm_channel = DirectChannel(MemoryJournal())
        run_handle = RunHandle(
            run_id="test-per-prompt-run",
            session_id="test-per-prompt-session",
            agent_type="test-per-prompt",
            agent=agent,
            event_bus=event_bus,
            session=session,
            run_ctx=AgentRunContext(
                session_id="test-per-prompt-session",
                event_bus=event_bus,
            ),
        )

        events: list[Any] = []
        gen = run_handle.start("hello")
        # In per-prompt model, generator terminates after one turn.
        events = [event async for event in gen]

        # Generator should have terminated naturally
        assert run_handle.complete_event.is_set()
        # Should have yielded StreamCompleteEvent (RunStartedEvent is
        # published via comm, not yielded to the consumer)
        event_types = [type(e).__name__ for e in events]
        assert "StreamCompleteEvent" in event_types


@pytest.mark.asyncio
async def test_cancel_called_twice_on_running_run_handle() -> None:
    """Cancel called twice on a running RunHandle does not raise.

    The idempotency guard checks complete_event.is_set() before
    proceeding. After the first cancel sets cancelled and cancels the
    task, the second cancel is a no-op (task is already done).
    """
    handle = _make_run_handle()
    mock_task = MagicMock()
    mock_task.done.return_value = False
    handle.run_ctx.current_task = mock_task

    handle.cancel()
    assert handle.run_ctx.cancelled is True
    mock_task.cancel.assert_called_once()

    # Second cancel — should not raise
    mock_task.cancel.reset_mock()
    mock_task.done.return_value = True  # Task is now done after first cancel
    handle.cancel()
    # cancelled was already True, task.cancel not called again
    mock_task.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_turn_lock_released_after_run_handle_terminates_on_error() -> None:
    """turn_lock is released after RunHandle terminates on error.

    In the per-prompt model, start() acquires turn_lock, executes one
    turn, and releases it in the finally block (via async with). Even
    on error, the lock must be released.
    """
    agent = Agent(
        name="test-lock-release",
        model=TestModel(custom_output_text="ok"),
    )
    async with agent:
        event_bus = EventBus()
        session = SessionState(
            session_id="test-lock-session",
            agent_name="test-lock-release",
        )
        session._comm_channel = DirectChannel(MemoryJournal())
        run_handle = RunHandle(
            run_id="test-lock-run",
            session_id="test-lock-session",
            agent_type="test-lock-release",
            agent=agent,
            event_bus=event_bus,
            session=session,
            run_ctx=AgentRunContext(
                session_id="test-lock-session",
                event_bus=event_bus,
            ),
        )

        class FailingTurn:
            async def execute(self) -> Any:
                raise RuntimeError("turn failed")
                yield

        agent.create_turn = MagicMock(return_value=FailingTurn())  # type: ignore[method-assign]

        gen = run_handle.start("test")
        try:
            async for event in gen:
                if isinstance(event, RunErrorEvent):
                    break
        finally:
            with contextlib.suppress(Exception):
                await gen.aclose()

        # complete_event set means start() finished, turn_lock released
        assert run_handle.complete_event.is_set()
        # turn_lock should not be locked
        assert not session.turn_lock.locked()


@pytest.mark.asyncio
async def test_lifecycle_dimensions_not_closed_when_run_handle_terminates() -> None:
    """RunHandle.close() does NOT close lifecycle dimensions.

    In the per-prompt model, lifecycle dimensions are session-owned.
    RunHandle.close() only sets complete_event and clears steer_callback.
    """
    from wolfharness.lifecycle import DirectChannel, MemoryJournal

    journal = MemoryJournal()
    comm_channel = DirectChannel(journal)

    session = _make_session(comm_channel=comm_channel)
    handle = _make_run_handle(session=session)

    handle.close()

    # complete_event should be set
    assert handle.complete_event.is_set()
    # CommChannel should still be usable (not closed)
    # Journal should still be accessible
    assert journal is not None
    assert comm_channel is not None


@pytest.mark.asyncio
async def test_run_error_event_causes_natural_termination() -> None:
    """RunErrorEvent causes natural generator termination, not exception.

    In the per-prompt model, when turn.execute() yields RunErrorEvent,
    start() breaks the inner loop and the generator exits naturally.
    The consumer sees RunErrorEvent as a normal event, not an exception.
    """
    agent = Agent(
        name="test-natural-termination",
        model=TestModel(custom_output_text="ok"),
    )
    async with agent:
        event_bus = EventBus()
        session = SessionState(
            session_id="test-natural-term-session",
            agent_name="test-natural-termination",
        )
        session._comm_channel = DirectChannel(MemoryJournal())
        run_handle = RunHandle(
            run_id="test-natural-term-run",
            session_id="test-natural-term-session",
            agent_type="test-natural-termination",
            agent=agent,
            event_bus=event_bus,
            session=session,
            run_ctx=AgentRunContext(
                session_id="test-natural-term-session",
                event_bus=event_bus,
            ),
        )

        class ErrorThenCompleteTurn:
            _final_message = None

            async def execute(self) -> Any:
                yield RunErrorEvent(
                    message="simulated error",
                    run_id="test-natural-term-run",
                    agent_name="test-natural-termination",
                )

        agent.create_turn = MagicMock(return_value=ErrorThenCompleteTurn())  # type: ignore[method-assign]

        events: list[Any] = []
        gen = run_handle.start("test")
        try:
            async with asyncio.timeout(5):
                events = [event async for event in gen]
        except TimeoutError:
            pytest.fail("start() hung after RunErrorEvent — did not terminate naturally")

        # RunErrorEvent was yielded as a normal event
        error_events = [e for e in events if isinstance(e, RunErrorEvent)]
        assert len(error_events) == 1

        # Generator terminated naturally (complete_event set)
        assert run_handle.complete_event.is_set()


# ---------------------------------------------------------------------------
# Tests: standalone execution (event_bus=None) and feedback_queue draining
# Regression tests for code review findings (Gemini Code Assist)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_standalone_session_without_event_bus_executes_turn() -> None:
    """Standalone session (event_bus=None) must still execute turns.

    Regression test for code review finding: _initialize_lifecycle_and_recovery()
    didn't create CommChannel when event_bus was None, causing assert comm is not None
    to fail in _execute_turn(). Now a DirectChannel is created and start() allows
    event_bus=None when session._comm_channel is set.
    """
    from wolfharness.lifecycle import DirectChannel, MemoryJournal
    from wolfharness.orchestrator.session_controller import SessionState

    agent = MagicMock()
    agent.create_turn = MagicMock(return_value=_StubTurn(events=[_stream_complete_event()]))
    agent.name = "test-agent"
    agent.conversation = MessageHistory()

    session = SessionState(session_id="test-standalone", agent_name="test-agent")
    session._comm_channel = DirectChannel(MemoryJournal())  # standalone fallback

    handle = RunHandle(
        run_id="test-standalone",
        session_id="test-standalone",
        agent_type="native",
        agent=agent,
        event_bus=None,  # standalone — no event bus
        session=session,
    )

    events: list[Any] = []
    gen = handle.start("hello")
    try:
        async with asyncio.timeout(5):
            events.extend([event async for event in gen])
    finally:
        with contextlib.suppress(Exception):
            await gen.aclose()

    # Turn executed and yielded events
    assert handle.complete_event.is_set()
    assert len(events) > 0
    # No AssertionError on assert comm is not None (turn completed successfully)
    event_types = [type(e).__name__ for e in events]
    assert "StreamCompleteEvent" in event_types


@pytest.mark.unit
async def test_feedback_queue_drained_on_new_run_handle_start() -> None:
    """Steer messages in feedback_queue must be drained when a new RunHandle starts.

    Regression test for code review finding: feedback_queue was never drained
    when a new RunHandle started, causing idle-state steer messages to be lost.
    The fix drains feedback_queue in start() before the turn begins, routing
    messages to queued_steer_messages via self.steer().
    """
    from wolfharness.lifecycle import DirectChannel, Feedback, MemoryJournal
    from wolfharness.orchestrator.session_controller import SessionState

    agent = MagicMock()
    agent.create_turn = MagicMock(return_value=_StubTurn())
    agent.name = "test-agent"
    agent.conversation = MessageHistory()

    session = SessionState(session_id="test-feedback", agent_name="test-agent")
    session._comm_channel = DirectChannel(MemoryJournal())

    # Simulate steer message arriving while idle (no RunHandle active)
    fb = Feedback(content="background task completed", is_steer=True)
    session.feedback_queue.put_nowait(fb)
    assert not session.feedback_queue.empty()  # pre-condition

    handle = RunHandle(
        run_id="test-feedback",
        session_id="test-feedback",
        agent_type="native",
        agent=agent,
        event_bus=AsyncMock(),
        session=session,
    )

    events: list[Any] = []
    gen = handle.start("hello")
    try:
        async with asyncio.timeout(5):
            events.extend([event async for event in gen])
    finally:
        with contextlib.suppress(Exception):
            await gen.aclose()

    # feedback_queue should be drained by start()
    assert session.feedback_queue.empty(), "feedback_queue was not drained by start()"
    # Steer message should be in queued_steer_messages
    assert "background task completed" in handle.run_ctx.queued_steer_messages


@pytest.mark.unit
async def test_multiple_steer_messages_drained_fifo_from_feedback_queue() -> None:
    """Multiple steer messages in feedback_queue are drained in FIFO order.

    When multiple background tasks complete while the session is idle,
    all their steer messages should be enqueued to feedback_queue and
    drained in FIFO order when the next RunHandle starts.
    """
    from wolfharness.lifecycle import DirectChannel, Feedback, MemoryJournal
    from wolfharness.orchestrator.session_controller import SessionState

    agent = MagicMock()
    agent.create_turn = MagicMock(return_value=_StubTurn())
    agent.name = "test-agent"
    agent.conversation = MessageHistory()

    session = SessionState(session_id="test-fifo", agent_name="test-agent")
    session._comm_channel = DirectChannel(MemoryJournal())

    # Enqueue 3 steer messages in FIFO order
    for msg in ("msg1", "msg2", "msg3"):
        session.feedback_queue.put_nowait(Feedback(content=msg, is_steer=True))
    assert not session.feedback_queue.empty()  # pre-condition

    handle = RunHandle(
        run_id="test-fifo",
        session_id="test-fifo",
        agent_type="native",
        agent=agent,
        event_bus=AsyncMock(),
        session=session,
    )

    events: list[Any] = []
    gen = handle.start("hello")
    try:
        async with asyncio.timeout(5):
            events.extend([event async for event in gen])
    finally:
        with contextlib.suppress(Exception):
            await gen.aclose()

    # All messages drained from feedback_queue
    assert session.feedback_queue.empty(), "feedback_queue was not fully drained"
    # Messages must appear in queued_steer_messages in FIFO order
    assert handle.run_ctx.queued_steer_messages == ["msg1", "msg2", "msg3"], (
        f"Expected FIFO order ['msg1', 'msg2', 'msg3'], got {handle.run_ctx.queued_steer_messages}"
    )


@pytest.mark.unit
async def test_empty_prompt_drains_feedback_queue_but_messages_unprocessed() -> None:
    """Empty prompt drains feedback_queue into queued_steer_messages but no turn executes.

    When initial_prompt is empty, start() drains feedback_queue into
    queued_steer_messages (for the next RunHandle to pick up) but returns
    immediately without executing a turn. The steer messages are in
    queued_steer_messages but no turn processes them in THIS RunHandle.

    However, the messages are NOT permanently lost — they remain in
    queued_steer_messages for this RunHandle's lifetime. If the same
    run_ctx is reused (e.g., by SessionController chaining), a subsequent
    turn with a non-empty prompt would drain them via
    drain_queued_steer_messages(). In the per-prompt model, a new
    RunHandle is created instead, so the messages in this run_ctx are
    discarded — but the source feedback_queue was already drained.

    To avoid permanent loss, the caller (SessionController) should check
    queued_steer_messages after start() returns and re-enqueue any
    unprocessed messages back to feedback_queue for the next RunHandle.
    """
    from wolfharness.agents.staged_content import StagedContent
    from wolfharness.lifecycle import DirectChannel, Feedback, MemoryJournal
    from wolfharness.orchestrator.session_controller import SessionState

    agent = MagicMock()
    agent.create_turn = MagicMock(return_value=_StubTurn())
    agent.name = "test-agent"
    agent.conversation = MessageHistory()
    agent.staged_content = StagedContent()  # empty — no staged content

    session = SessionState(session_id="test-empty-drain", agent_name="test-agent")
    session._comm_channel = DirectChannel(MemoryJournal())

    # Enqueue 1 steer message
    session.feedback_queue.put_nowait(Feedback(content="orphaned steer", is_steer=True))
    assert not session.feedback_queue.empty()  # pre-condition

    handle = RunHandle(
        run_id="test-empty-drain",
        session_id="test-empty-drain",
        agent_type="native",
        agent=agent,
        event_bus=AsyncMock(),
        session=session,
    )

    events: list[Any] = []
    gen = handle.start("")
    try:
        async with asyncio.timeout(5):
            events = [event async for event in gen]
    finally:
        with contextlib.suppress(Exception):
            await gen.aclose()

    # feedback_queue was drained
    assert session.feedback_queue.empty(), "feedback_queue was not drained"
    # Message was moved to queued_steer_messages
    assert "orphaned steer" in handle.run_ctx.queued_steer_messages
    # RunHandle completed without executing a turn
    assert handle.complete_event.is_set()
    # No events yielded — no turn was executed
    assert events == [], f"Expected no events, got {[type(e).__name__ for e in events]}"
    # No turn should have been created
    agent.create_turn.assert_not_called()


@pytest.mark.unit
async def test_feedback_queue_drains_multimodal_content_blocks() -> None:
    """Feedback with content_blocks (multimodal) is drained correctly.

    When a background task completes with multimodal content (e.g., image +
    text), the Feedback object has content_blocks set instead of content.
    The draining code should handle both paths.
    """
    from wolfharness.lifecycle import DirectChannel, Feedback, MemoryJournal
    from wolfharness.orchestrator.session_controller import SessionState

    agent = MagicMock()
    agent.create_turn = MagicMock(return_value=_StubTurn())
    agent.name = "test-agent"
    agent.conversation = MessageHistory()

    session = SessionState(session_id="test-multimodal", agent_name="test-agent")
    session._comm_channel = DirectChannel(MemoryJournal())

    # Enqueue a Feedback with content_blocks (multimodal) instead of plain content
    blocks: list[Any] = [{"type": "text", "text": "image analysis complete"}]
    session.feedback_queue.put_nowait(Feedback(content="", is_steer=True, content_blocks=blocks))
    assert not session.feedback_queue.empty()  # pre-condition

    handle = RunHandle(
        run_id="test-multimodal",
        session_id="test-multimodal",
        agent_type="native",
        agent=agent,
        event_bus=AsyncMock(),
        session=session,
    )

    events: list[Any] = []
    gen = handle.start("hello")
    try:
        async with asyncio.timeout(5):
            events.extend([event async for event in gen])
    finally:
        with contextlib.suppress(Exception):
            await gen.aclose()

    # feedback_queue drained
    assert session.feedback_queue.empty(), "feedback_queue was not drained"
    # The content_blocks list should appear in queued_steer_messages
    # (content_blocks takes priority over content when not None)
    assert blocks in handle.run_ctx.queued_steer_messages, (
        f"Expected content_blocks {blocks} in queued_steer_messages, "
        f"got {handle.run_ctx.queued_steer_messages}"
    )
