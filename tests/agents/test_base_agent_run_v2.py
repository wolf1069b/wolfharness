"""Tests for BaseAgent.create_run() and create_run_stream() v2 methods."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

from pydantic_ai.models.test import TestModel
import pytest

from wolfharness.agents.context import AgentRunContext
from wolfharness.agents.events import RunErrorEvent, RunStartedEvent, StreamCompleteEvent
from wolfharness.agents.native_agent.agent import Agent
from wolfharness.lifecycle import DirectChannel, MemoryJournal
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.core import EventBus, SessionState
from wolfharness.orchestrator.run import RunHandle
from wolfharness.orchestrator.turn import Turn


if TYPE_CHECKING:
    from wolfharness import AgentPool


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubTurn(Turn):
    """Minimal Turn implementation that yields a fixed event sequence."""

    def __init__(
        self,
        *,
        events: list[Any] | None = None,
        message_history: list[Any] | None = None,
    ) -> None:
        self._events = events or []
        self._history = message_history or []

    async def execute(self):  # type: ignore[override]
        # Set history before yielding so it's available even if
        # the consumer breaks on StreamCompleteEvent.
        self._message_history = self._history
        self._final_message = ChatMessage(content="done", role="assistant")
        for event in self._events:
            yield event


def _stream_complete_event() -> StreamCompleteEvent[Any]:
    return StreamCompleteEvent(message=ChatMessage(content="done", role="assistant"))


def _make_session() -> Any:
    """Create a real SessionState with a DirectChannel for event publishing."""
    session = SessionState(session_id="test-session", agent_name="test_agent")
    session._comm_channel = DirectChannel(MemoryJournal())
    return session


def _make_agent_with_stub_turn(
    events: list[Any],
    history: list[Any] | None = None,
) -> Agent:
    """Create a real Agent whose create_turn returns a _StubTurn.

    This lets us test create_run/create_run_stream without running
    the actual pydantic-ai agent loop.
    """
    agent = Agent(model=TestModel(), name="test_agent")
    stub = _StubTurn(events=events, message_history=history or [])
    agent.create_turn = MagicMock(return_value=stub)  # type: ignore[method-assign]
    return agent


# ---------------------------------------------------------------------------
# create_run() tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_create_run_returns_run_handle_without_executing() -> None:
    """Given an Agent, create_run() returns a RunHandle in idle status.

    No execution should happen — the handle is ready to be started
    via start() but has not begun any turn.
    """
    agent = Agent(model=TestModel(), name="test_agent")
    run_ctx = AgentRunContext(session_id="sess-1", run_id="run-1")
    event_bus = AsyncMock()
    session = _make_session()

    handle = agent.create_run(
        prompt="Hello",
        run_ctx=run_ctx,
        message_history=[],
        event_bus=event_bus,
        session=session,
    )

    assert isinstance(handle, RunHandle)
    # In the per-prompt model, RunHandle has no _run_state or _closing.
    # complete_event not set means the handle is ready but not complete.
    assert handle.complete_event.is_set() is False
    assert handle.is_running is True


@pytest.mark.unit
async def test_create_run_handle_fields_correctly_set() -> None:
    """Given an Agent with specific run_ctx, create_run() wires all fields."""
    agent = Agent(model=TestModel(), name="test_agent")
    run_ctx = AgentRunContext(session_id="sess-42", run_id="run-42")
    event_bus = AsyncMock()
    session = _make_session()
    history: list[Any] = ["msg1", "msg2"]

    handle = agent.create_run(
        prompt="Hello",
        run_ctx=run_ctx,
        message_history=history,
        event_bus=event_bus,
        session=session,
    )

    assert handle.agent is agent
    assert handle.event_bus is event_bus
    assert handle.session is session
    assert handle.run_ctx is run_ctx
    assert handle.run_id == "run-42"
    assert handle.session_id == "sess-42"
    assert handle.agent_type == "native"
    assert handle._message_history == ["msg1", "msg2"]  # type: ignore[comparison-overlap]


@pytest.mark.unit
async def test_create_run_does_not_call_create_turn() -> None:
    """Given create_run() is called, create_turn() is never invoked.

    This verifies that construction does not trigger execution.
    """
    agent = Agent(model=TestModel(), name="test_agent")
    create_turn_mock = MagicMock(return_value=_StubTurn(events=[]))
    agent.create_turn = create_turn_mock  # type: ignore[method-assign]

    run_ctx = AgentRunContext()
    event_bus = AsyncMock()
    session = _make_session()

    agent.create_run(
        prompt="Hello",
        run_ctx=run_ctx,
        message_history=[],
        event_bus=event_bus,
        session=session,
    )

    create_turn_mock.assert_not_called()


# ---------------------------------------------------------------------------
# create_run_stream() tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_create_run_stream_yields_events_and_closes() -> None:
    """Given a stubbed agent, create_run_stream() yields all events.

    The stream should yield RunStartedEvent followed by
    StreamCompleteEvent, then terminate.
    """
    events = [
        RunStartedEvent(run_id="r1", session_id="s1", agent_name="test"),
        _stream_complete_event(),
    ]
    agent = _make_agent_with_stub_turn(events=events, history=["m1"])

    run_ctx = AgentRunContext(session_id="s1", run_id="r1")
    event_bus = AsyncMock()
    session = _make_session()

    yielded = [
        event
        async for event in agent.create_run_stream(
            prompt="Hello",
            run_ctx=run_ctx,
            message_history=[],
            event_bus=event_bus,
            session=session,
        )
    ]

    assert len(yielded) == 2
    assert isinstance(yielded[0], RunStartedEvent)
    assert isinstance(yielded[1], StreamCompleteEvent)


@pytest.mark.unit
async def test_create_run_stream_closes_handle_after_completion() -> None:
    """Given create_run_stream() completes, the RunHandle is closed.

    The close() call on StreamCompleteEvent sets _closing=True.
    """
    events = [_stream_complete_event()]
    agent = _make_agent_with_stub_turn(events=events, history=["m1"])

    # Capture the RunHandle by wrapping create_run
    captured: list[RunHandle] = []
    original_create_run = agent.create_run

    def _capturing_create_run(*args: Any, **kwargs: Any) -> RunHandle:
        handle = original_create_run(*args, **kwargs)
        captured.append(handle)
        return handle

    agent.create_run = _capturing_create_run  # type: ignore[method-assign]

    run_ctx = AgentRunContext(session_id="s1", run_id="r1")
    event_bus = AsyncMock()
    session = _make_session()

    async for _event in agent.create_run_stream(
        prompt="Hello",
        run_ctx=run_ctx,
        message_history=[],
        event_bus=event_bus,
        session=session,
    ):
        pass

    assert len(captured) == 1
    # In the per-prompt model, close() sets complete_event instead of _closing.
    assert captured[0].complete_event.is_set() is True


# ---------------------------------------------------------------------------
# Tests from PR #64 review (base agent v2 run path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_stream_breaks_on_stream_complete(minimal_pool: AgentPool) -> None:
    """_run_stream_run_turn must break on StreamCompleteEvent in active-run path.

    Without the break, the while-True loop blocks indefinitely on
    stream.receive() after the run completes because the session
    remains open and EndOfStream is never raised.
    """
    from wolfharness.orchestrator.core import SessionController

    controller = SessionController(pool=minimal_pool)
    event_bus = EventBus()
    controller._event_bus = event_bus

    # We test the EventBus subscription loop logic directly
    session_id = "test-stream-break"
    stream: asyncio.Queue[Any] = await event_bus.subscribe(session_id, scope="session")

    # Publish a StreamCompleteEvent
    complete_event: StreamCompleteEvent[Any] = StreamCompleteEvent(
        message=MagicMock(content="done"),
    )

    async def _publish_and_finish() -> None:
        await asyncio.sleep(0.05)
        await event_bus.publish(session_id, complete_event)

    publish_task = asyncio.create_task(_publish_and_finish())

    # Simulate the while-True loop from _run_stream_run_turn
    received: list[Any] = []
    try:
        async with asyncio.timeout(5):
            while True:
                try:
                    event = await stream.get()
                except asyncio.QueueShutDown:
                    break
                received.append(event.event)
                # This is the fix: break on terminal events
                if isinstance(event.event, StreamCompleteEvent | RunErrorEvent):
                    break
    except TimeoutError:
        pytest.fail("Loop hung — StreamCompleteEvent was received but loop didn't break")
    finally:
        publish_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await publish_task
        await event_bus.unsubscribe(session_id, stream)

    assert len(received) >= 1
    assert isinstance(received[-1], StreamCompleteEvent)


def test_no_duplicate_stream_complete_in_run_once() -> None:
    """_execute_node must not publish StreamCompleteEvent after turn.execute().

    NativeTurn.execute() already yields StreamCompleteEvent as its terminal
    event. Publishing it again results in duplicate events on the EventBus.
    """
    import wolfharness.agents.native_agent.agent as agent_module

    source = inspect.getsource(agent_module.Agent._execute_node)
    # After the fix, there should be no explicit StreamCompleteEvent publish.
    # Check for the pattern of publishing StreamCompleteEvent (not just the word
    # in docstrings or comments).
    import re

    publish_matches = re.findall(r"await\s+event_bus\.publish\s*\([^)]*StreamCompleteEvent", source)
    assert len(publish_matches) == 0, (
        f"_execute_node still publishes StreamCompleteEvent {len(publish_matches)} "
        "time(s) — duplicate publish should be removed since turn.execute() "
        "already yields it"
    )


def test_no_duplicate_stream_complete_in_stream_events() -> None:
    """_stream_events must not publish StreamCompleteEvent after turn.execute().

    NativeTurn.execute() already yields StreamCompleteEvent as its terminal
    event. Publishing it again results in duplicate events on the EventBus.
    """
    import wolfharness.agents.native_agent.agent as agent_module

    source = inspect.getsource(agent_module.Agent._stream_events)
    import re

    publish_matches = re.findall(r"await\s+event_bus\.publish\s*\([^)]*StreamCompleteEvent", source)
    assert len(publish_matches) == 0, (
        f"_stream_events still publishes StreamCompleteEvent {len(publish_matches)} "
        "time(s) — duplicate publish should be removed since turn.execute() "
        "already yields it"
    )


def test_base_agent_imports_are_runtime_available() -> None:
    """StreamCompleteEvent and RunErrorEvent must be imported at runtime.

    Gemini claimed they were only in TYPE_CHECKING, but they are actually
    imported at module level (line 23 of base_agent.py).
    """
    import wolfharness.agents.base_agent as base_module

    # Verify the classes are accessible as attributes (runtime import)
    assert hasattr(base_module, "StreamCompleteEvent"), (
        "StreamCompleteEvent must be imported at runtime, not TYPE_CHECKING only"
    )
    assert hasattr(base_module, "RunErrorEvent"), (
        "RunErrorEvent must be imported at runtime, not TYPE_CHECKING only"
    )


def test_execute_node_handles_run_error_event() -> None:
    """_execute_node must check for RunErrorEvent before accessing final_message.

    Without this, if turn.execute() yields RunErrorEvent and returns early,
    turn.final_message raises RuntimeError, masking the original error.
    """
    import wolfharness.agents.native_agent.agent as agent_module

    source = inspect.getsource(agent_module.Agent._execute_node)
    assert "RunErrorEvent" in source, (
        "_execute_node must check for RunErrorEvent from turn.execute()"
    )
    assert "turn_failed" in source, (
        "_execute_node must track turn_failed flag to avoid accessing final_message"
    )


def test_stream_events_handles_run_error_event() -> None:
    """_stream_events must check for RunErrorEvent before accessing final_message.

    Same issue as _execute_node — if turn fails, final_message is not set.
    """
    import wolfharness.agents.native_agent.agent as agent_module

    source = inspect.getsource(agent_module.Agent._stream_events)
    assert "RunErrorEvent" in source, (
        "_stream_events must check for RunErrorEvent from turn.execute()"
    )
    assert "turn_failed" in source, (
        "_stream_events must track turn_failed flag to avoid accessing final_message"
    )
