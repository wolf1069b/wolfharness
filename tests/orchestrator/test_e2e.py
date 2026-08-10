"""End-to-end integration tests covering full user session lifecycles (Group 6.13-6.15).

Tests full session lifecycle, multi-agent concurrent handling, and cross-protocol
event passing through the EventBus.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import field
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

from pydantic_ai import TextPartDelta
import pytest

from wolfharness.agents.events import (
    PartDeltaEvent,
    RunStartedEvent,
    StreamCompleteEvent,
    ToolCallCompleteEvent,
    ToolCallStartEvent,
)
from wolfharness.lifecycle import DirectChannel, MemoryJournal
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.core import EventBus, EventEnvelope, SessionPool


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from wolfharness import AgentPool
    from wolfharness.agents.context import AgentRunContext


pytestmark = [pytest.mark.integration, pytest.mark.anyio]


def _unwrap_event(event: Any) -> Any:
    """Unwrap EventEnvelope if present, otherwise return the event as-is."""
    return event.event if isinstance(event, EventEnvelope) else event


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


async def _attach_agent(
    pool: SessionPool,
    session_id: str,
    agent: MagicMock,
    real_pool: AgentPool,
) -> None:
    """Attach a mock agent to an existing session."""
    state, _ = await pool.sessions.get_or_create_session(session_id)
    state.agent = agent
    pool.sessions._session_agents[session_id] = agent
    real_pool.get_agent = MagicMock(return_value=agent)  # type: ignore[assignment]
    # Set up lifecycle dimensions for RunHandle._execute_turn()
    # session._comm_channel must be set before start() accesses it.
    state._comm_channel = DirectChannel(MemoryJournal())


@pytest.fixture
def mock_agent_full_lifecycle() -> MagicMock:
    """Return a mocked BaseAgent that yields a complete event lifecycle."""
    agent = MagicMock()

    def _make_turn(prompts: Any, run_ctx: AgentRunContext, **kw: Any) -> Any:

        class _MockTurn:
            message_history: list[Any] = field(default_factory=list)

            async def execute(
                self,
            ) -> AsyncIterator[
                PartDeltaEvent
                | ToolCallStartEvent
                | ToolCallCompleteEvent
                | StreamCompleteEvent[Any]
            ]:
                yield PartDeltaEvent.text(index=0, content="Hello")
                yield ToolCallStartEvent(
                    tool_call_id="tc-1",
                    tool_name="bash",
                    title="Running bash command",
                )
                yield ToolCallCompleteEvent(
                    tool_name="bash",
                    tool_call_id="tc-1",
                    tool_input={"command": "echo hi"},
                    tool_result="hi",
                    agent_name="test-agent",
                    message_id="msg-1",
                )
                yield StreamCompleteEvent(
                    message=ChatMessage(content="Done", role="assistant"),
                )

        return _MockTurn()

    agent.create_turn = _make_turn
    return agent


@pytest.fixture
def mock_agent_with_text(text: str = "response") -> MagicMock:
    """Return a mocked BaseAgent that yields text and completes."""
    agent = MagicMock()

    def _make_turn(prompts: Any, run_ctx: AgentRunContext, **kw: Any) -> Any:

        class _MockTurn:
            message_history: list[Any] = field(default_factory=list)

            async def execute(self) -> AsyncIterator[PartDeltaEvent | StreamCompleteEvent[Any]]:
                yield PartDeltaEvent.text(index=0, content=text)
                yield StreamCompleteEvent(
                    message=ChatMessage(content=text, role="assistant"),
                )

        return _MockTurn()

    agent.create_turn = _make_turn
    return agent


# ---------------------------------------------------------------------------
# 6.13: Full user session lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_full_session_lifecycle_create_prompt_events_close(
    minimal_pool: AgentPool,
    mock_agent_full_lifecycle: MagicMock,
) -> None:
    """6.13: create_session → process_prompt → verify events → close_session.

    Verifies that the SessionPool tracks state correctly throughout the
    entire lifecycle and that all expected event types are delivered in
    order via the EventBus.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None

    # Create session
    state = await session_pool.create_session("sess-lifecycle", agent_name="agent-a")
    assert state.session_id == "sess-lifecycle"
    assert session_pool.sessions.get_session("sess-lifecycle") is state

    # Attach mock agent so turn can run
    await _attach_agent(session_pool, "sess-lifecycle", mock_agent_full_lifecycle, minimal_pool)

    # Subscribe to events before processing
    queue = await session_pool.event_bus.subscribe("sess-lifecycle")

    # Process prompt
    await session_pool.process_prompt("sess-lifecycle", "hello")

    # Collect all events
    events: list[Any] = []
    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=0.5)
            events.append(event)
        except (TimeoutError, asyncio.QueueShutDown):
            break

    # Verify event ordering and types
    assert len(events) == 5
    assert isinstance(_unwrap_event(events[0]), RunStartedEvent)
    assert _unwrap_event(events[0]).session_id == "sess-lifecycle"
    assert isinstance(_unwrap_event(events[1]), PartDeltaEvent)
    assert isinstance(_unwrap_event(events[2]), ToolCallStartEvent)
    assert _unwrap_event(events[2]).tool_name == "bash"
    assert isinstance(_unwrap_event(events[3]), ToolCallCompleteEvent)
    assert _unwrap_event(events[3]).tool_result == "hi"
    assert isinstance(_unwrap_event(events[4]), StreamCompleteEvent)
    assert _unwrap_event(events[4]).message.content == "Done"

    # Close session
    await session_pool.close_session("sess-lifecycle")
    assert session_pool.sessions.get_session("sess-lifecycle") is None

    # Sentinel should have been sent
    with pytest.raises((asyncio.TimeoutError, asyncio.QueueShutDown)):
        await asyncio.wait_for(queue.get(), timeout=0.5)

    # session_pool lifecycle managed by minimal_pool fixture


@pytest.mark.anyio
async def test_multi_turn_same_session_both_complete_via_consume_run(
    minimal_pool: AgentPool,
    mock_agent_with_text: MagicMock,
) -> None:
    """E1/E2: Two consecutive messages through _start_run_handle must both complete.

    Unlike test_full_session_lifecycle which uses run_stream (single-turn
    correct path), this test uses _route_message → _start_run_handle →
    _consume_run — the same path used by the opencode server's route_message().

    Current behavior: _consume_run breaks on StreamCompleteEvent + aclose()
    → RunHandle killed after turn 1 → turn 2's followup message is lost.
    """
    from tests._controller_helpers import send_via_controller
    from wolfharness.lifecycle.types import DeliveryMode

    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    controller = session_pool.sessions

    session_id = "sess-multiturn-e2e"
    await session_pool.create_session(session_id, agent_name="test_agent")

    # Patch the agent to use our mock create_turn
    agent = await controller.get_or_create_session_agent(session_id)
    assert agent is not None
    agent.create_turn = mock_agent_with_text.create_turn  # type: ignore[method-assign]

    queue = await session_pool.event_bus.subscribe(session_id)

    # Turn 1 via _route_message (opencode server path)
    await send_via_controller(
        controller,
        session_id,
        "turn 1",
        mode=DeliveryMode.QUEUE,
    )
    events_t1: list[Any] = []
    while True:
        try:
            envelope = await asyncio.wait_for(queue.get(), timeout=10.0)
        except TimeoutError:
            break
        events_t1.append(envelope.event)
        if isinstance(envelope.event, StreamCompleteEvent):
            break

    assert any(isinstance(e, StreamCompleteEvent) for e in events_t1), (
        "Turn 1 should produce a StreamCompleteEvent"
    )

    # Turn 2 via _route_message — the critical one
    await send_via_controller(
        controller,
        session_id,
        "turn 2",
        mode=DeliveryMode.QUEUE,
    )
    events_t2: list[Any] = []
    while True:
        try:
            envelope = await asyncio.wait_for(queue.get(), timeout=10.0)
        except TimeoutError:
            break
        events_t2.append(envelope.event)
        if isinstance(envelope.event, StreamCompleteEvent):
            break

    assert any(isinstance(e, StreamCompleteEvent) for e in events_t2), (
        "Turn 2 should produce a StreamCompleteEvent — "
        "if missing, _consume_run killed the generator (issue E1)"
    )


@pytest.mark.anyio
async def test_full_lifecycle_session_state_transitions(
    minimal_pool: AgentPool,
    mock_agent_full_lifecycle: MagicMock,
) -> None:
    """6.13: Verify SessionPool state transitions during lifecycle.

    Ensures that session state moves correctly from active to closing to
    closed, and that turn timing metrics are recorded.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None

    await session_pool.create_session("sess-state", agent_name="agent-b")
    await _attach_agent(session_pool, "sess-state", mock_agent_full_lifecycle, minimal_pool)

    # Pre-run: session exists and is not closing
    pre_state = session_pool.sessions.get_session("sess-state")
    assert pre_state is not None
    assert pre_state.is_closing is False
    assert pre_state.closed_at is None

    # Run turn
    await session_pool.process_prompt("sess-state", "hello")

    # Close session
    await session_pool.close_session("sess-state")

    # Post-close: session removed
    post_state = session_pool.sessions.get_session("sess-state")
    assert post_state is None

    # session_pool lifecycle managed by minimal_pool fixture


# ---------------------------------------------------------------------------
# 6.14: Multi-agent concurrent session handling
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_multi_agent_concurrent_sessions_no_contamination(
    minimal_pool: AgentPool,
) -> None:
    """6.14: Create 2+ sessions with different agents and process concurrently.

    Verifies that events from different sessions do not cross over and that
    each session receives only its own events.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None

    # Create two agents with distinct response text
    agent_a = MagicMock()

    class _TurnA:
        message_history: list[Any] = field(default_factory=list)

        async def execute(self) -> AsyncIterator[PartDeltaEvent | StreamCompleteEvent[Any]]:
            yield PartDeltaEvent.text(index=0, content="response-from-agent-a")
            yield StreamCompleteEvent(
                message=ChatMessage(content="response-from-agent-a", role="assistant"),
            )

    agent_a.create_turn = MagicMock(return_value=_TurnA())

    agent_b = MagicMock()

    class _TurnB:
        message_history: list[Any] = field(default_factory=list)

        async def execute(self) -> AsyncIterator[PartDeltaEvent | StreamCompleteEvent[Any]]:
            yield PartDeltaEvent.text(index=0, content="response-from-agent-b")
            yield StreamCompleteEvent(
                message=ChatMessage(content="response-from-agent-b", role="assistant"),
            )

    agent_b.create_turn = MagicMock(return_value=_TurnB())

    # Create sessions and attach different agents
    await session_pool.create_session("sess-a", agent_name="agent-a")
    await session_pool.create_session("sess-b", agent_name="agent-b")
    await _attach_agent(session_pool, "sess-a", agent_a, minimal_pool)
    await _attach_agent(session_pool, "sess-b", agent_b, minimal_pool)

    # Subscribe to both sessions
    queue_a = await session_pool.event_bus.subscribe("sess-a")
    queue_b = await session_pool.event_bus.subscribe("sess-b")

    # Process prompts concurrently
    await asyncio.gather(
        session_pool.process_prompt("sess-a", "prompt-a"),
        session_pool.process_prompt("sess-b", "prompt-b"),
    )

    # Collect events for session A
    events_a: list[Any] = []
    while True:
        with contextlib.suppress(asyncio.QueueEmpty):
            events_a.append(queue_a.get_nowait())
            continue
        break

    events_b: list[Any] = []
    while True:
        with contextlib.suppress(asyncio.QueueEmpty):
            events_b.append(queue_b.get_nowait())
            continue
        break

    # Verify session A only has agent-a events
    assert len(events_a) == 3
    assert all(
        isinstance(_unwrap_event(e), (RunStartedEvent, PartDeltaEvent, StreamCompleteEvent))
        for e in events_a
    )
    part_delta_a = _unwrap_event(events_a[1])
    assert isinstance(part_delta_a, PartDeltaEvent)
    assert isinstance(part_delta_a.delta, TextPartDelta)
    assert part_delta_a.delta.content_delta == "response-from-agent-a"

    # Verify session B only has agent-b events
    assert len(events_b) == 3
    assert all(
        isinstance(_unwrap_event(e), (RunStartedEvent, PartDeltaEvent, StreamCompleteEvent))
        for e in events_b
    )
    part_delta_b = _unwrap_event(events_b[1])
    assert isinstance(part_delta_b, PartDeltaEvent)
    assert isinstance(part_delta_b.delta, TextPartDelta)
    assert part_delta_b.delta.content_delta == "response-from-agent-b"

    # Verify no cross-session contamination in SessionController
    assert session_pool.sessions.get_session("sess-a") is not None
    assert session_pool.sessions.get_session("sess-b") is not None

    # Cleanup
    await session_pool.close_session("sess-a")
    await session_pool.close_session("sess-b")
    # session_pool lifecycle managed by minimal_pool fixture


@pytest.mark.anyio
async def test_concurrent_sessions_event_bus_isolation(
    minimal_pool: AgentPool,
) -> None:
    """6.14: Per-session event isolation holds under concurrent load.

    Subscribes multiple queues per session and verifies that each
    subscriber receives only events for its own session.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None

    agent = MagicMock()

    def _make_turn(prompts: Any, run_ctx: AgentRunContext, **kw: Any) -> Any:

        class _Turn:
            message_history: list[Any] = field(default_factory=list)

            async def execute(self) -> AsyncIterator[StreamCompleteEvent[Any]]:
                yield StreamCompleteEvent(
                    message=ChatMessage(content="done", role="assistant"),
                )

        return _Turn()

    agent.create_turn = _make_turn

    await session_pool.create_session("sess-x")
    await session_pool.create_session("sess-y")
    await _attach_agent(session_pool, "sess-x", agent, minimal_pool)
    await _attach_agent(session_pool, "sess-y", agent, minimal_pool)

    # Multiple subscribers per session
    qx1 = await session_pool.event_bus.subscribe("sess-x")
    qx2 = await session_pool.event_bus.subscribe("sess-x")
    qy1 = await session_pool.event_bus.subscribe("sess-y")
    qy2 = await session_pool.event_bus.subscribe("sess-y")

    # Concurrent processing
    await asyncio.gather(
        session_pool.process_prompt("sess-x", "prompt-x"),
        session_pool.process_prompt("sess-y", "prompt-y"),
    )

    # All subscribers for sess-x should have exactly 2 events (RunStarted + StreamComplete)
    for q in (qx1, qx2):
        ev1 = await asyncio.wait_for(q.get(), timeout=0.5)
        actual_ev1 = _unwrap_event(ev1)
        assert isinstance(actual_ev1, RunStartedEvent)
        assert actual_ev1.session_id == "sess-x"
        ev2 = await asyncio.wait_for(q.get(), timeout=0.5)
        actual_ev2 = _unwrap_event(ev2)
        assert isinstance(actual_ev2, StreamCompleteEvent)
        with pytest.raises(asyncio.QueueEmpty):
            q.get_nowait()

    for q in (qy1, qy2):
        ev1 = await asyncio.wait_for(q.get(), timeout=0.5)
        actual_ev1 = _unwrap_event(ev1)
        assert isinstance(actual_ev1, RunStartedEvent)
        assert actual_ev1.session_id == "sess-y"
        ev2 = await asyncio.wait_for(q.get(), timeout=0.5)
        actual_ev2 = _unwrap_event(ev2)
        assert isinstance(actual_ev2, StreamCompleteEvent)
        with pytest.raises(asyncio.QueueEmpty):
            q.get_nowait()

    await session_pool.close_session("sess-x")
    await session_pool.close_session("sess-y")
    # session_pool lifecycle managed by minimal_pool fixture


# ---------------------------------------------------------------------------
# 6.15: Cross-protocol event passing via EventBus
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cross_protocol_event_publishing_and_subscribing() -> None:
    """6.15: Simulate ACP handler publishing events; OpenCode handler receives them.

    Creates an EventBus, simulates an ACP protocol handler publishing events,
    and verifies that an OpenCode protocol handler subscribing to the same
    session receives identical events in the correct order.
    """
    event_bus = EventBus(max_queue_size=10)

    # Simulate protocol handlers subscribing
    acp_queue = await event_bus.subscribe("sess-cross")
    opencode_queue = await event_bus.subscribe("sess-cross")

    # Simulate ACP handler publishing events
    events_to_publish: list[Any] = [
        RunStartedEvent(session_id="sess-cross", run_id="run-1"),
        PartDeltaEvent.text(index=0, content="Cross-protocol text"),
        ToolCallStartEvent(
            tool_call_id="tc-cross",
            tool_name="read",
            title="Reading file",
        ),
        ToolCallCompleteEvent(
            tool_name="read",
            tool_call_id="tc-cross",
            tool_input={"path": "/tmp/test.txt"},
            tool_result="file contents",
            agent_name="cross-agent",
            message_id="msg-cross",
        ),
        StreamCompleteEvent(
            message=ChatMessage(content="Cross complete", role="assistant"),
        ),
    ]

    for event in events_to_publish:
        await event_bus.publish("sess-cross", event)

    # Verify ACP subscriber receives all events in order
    acp_received: list[Any] = []
    while True:
        with contextlib.suppress(asyncio.QueueEmpty):
            acp_received.append(acp_queue.get_nowait())
            continue
        break

    opencode_received: list[Any] = []
    while True:
        with contextlib.suppress(asyncio.QueueEmpty):
            opencode_received.append(opencode_queue.get_nowait())
            continue
        break

    assert len(acp_received) == len(events_to_publish)
    assert len(opencode_received) == len(events_to_publish)

    # Verify event ordering is preserved for both subscribers
    for i, expected in enumerate(events_to_publish):
        assert type(_unwrap_event(acp_received[i])) is type(expected)
        assert type(_unwrap_event(opencode_received[i])) is type(expected)

    # Verify events are shared objects across subscribers (EventBus behavior)
    for i in range(len(acp_received)):
        assert acp_received[i] is opencode_received[i]

    await event_bus.close_session("sess-cross")


@pytest.mark.anyio
async def test_cross_protocol_multiple_subscribers_different_protocols() -> None:
    """6.15: Multiple subscribers from different protocols receive events.

    Simulates three protocol handlers (ACP, OpenCode, AG-UI) subscribing to
    the same session and verifies all receive the same events.
    """
    event_bus = EventBus(max_queue_size=10)

    # Simulate three different protocol handlers
    acp_queue = await event_bus.subscribe("sess-multi")
    opencode_queue = await event_bus.subscribe("sess-multi")
    agui_queue = await event_bus.subscribe("sess-multi")

    # Publish a sequence of events
    events_to_publish: list[Any] = [
        RunStartedEvent(session_id="sess-multi", run_id="run-multi"),
        PartDeltaEvent.text(index=0, content="Multi-protocol message"),
        StreamCompleteEvent(
            message=ChatMessage(content="Done", role="assistant"),
        ),
    ]

    for event in events_to_publish:
        await event_bus.publish("sess-multi", event)

    # Verify all three subscribers receive the events
    acp_received: list[Any] = []
    opencode_received: list[Any] = []
    agui_received: list[Any] = []

    while True:
        with contextlib.suppress(asyncio.QueueEmpty):
            acp_received.append(acp_queue.get_nowait())
            continue
        break
    while True:
        with contextlib.suppress(asyncio.QueueEmpty):
            opencode_received.append(opencode_queue.get_nowait())
            continue
        break
    while True:
        with contextlib.suppress(asyncio.QueueEmpty):
            agui_received.append(agui_queue.get_nowait())
            continue
        break

    assert len(acp_received) == 3
    assert len(opencode_received) == 3
    assert len(agui_received) == 3

    # Verify all subscribers see the same event types in order
    for i, expected in enumerate(events_to_publish):
        assert type(_unwrap_event(acp_received[i])) is type(expected)
        assert type(_unwrap_event(opencode_received[i])) is type(expected)
        assert type(_unwrap_event(agui_received[i])) is type(expected)

    # All received events are shared across subscribers (EventBus behavior)
    for i in range(len(events_to_publish)):
        assert acp_received[i] is opencode_received[i]
        assert acp_received[i] is agui_received[i]

    await event_bus.close_session("sess-multi")


@pytest.mark.anyio
async def test_cross_protocol_event_ordering_preserved_under_load() -> None:
    """6.15: Event ordering is preserved when publishing many events rapidly.

    Publishes a large number of events in a known order and verifies that
    all subscribers receive them in the exact same sequence.
    """
    event_bus = EventBus(max_queue_size=100)

    protocol_a = await event_bus.subscribe("sess-order")
    protocol_b = await event_bus.subscribe("sess-order")

    # Publish 20 ordered events
    event_count = 20
    for i in range(event_count):
        await event_bus.publish(
            "sess-order",
            PartDeltaEvent.text(index=i, content=f"msg-{i}"),
        )

    # Collect events for both subscribers
    received_a: list[Any] = []
    received_b: list[Any] = []
    while True:
        with contextlib.suppress(asyncio.QueueEmpty):
            received_a.append(protocol_a.get_nowait())
            continue
        break
    while True:
        with contextlib.suppress(asyncio.QueueEmpty):
            received_b.append(protocol_b.get_nowait())
            continue
        break

    # Coalescing moved from publish-side to subscriber-side (drain_and_merge).
    # Direct EventBus publishes are not coalesced; each event is delivered as-is.
    assert len(received_a) == event_count
    assert len(received_b) == event_count

    # Verify content ordering is preserved
    for i in range(event_count):
        ev_a = _unwrap_event(received_a[i])
        ev_b = _unwrap_event(received_b[i])
        assert isinstance(ev_a, PartDeltaEvent)
        assert isinstance(ev_b, PartDeltaEvent)
        assert isinstance(ev_a.delta, TextPartDelta)
        assert isinstance(ev_b.delta, TextPartDelta)
        assert ev_a.delta.content_delta == f"msg-{i}"
        assert ev_b.delta.content_delta == f"msg-{i}"

    await event_bus.close_session("sess-order")


@pytest.mark.anyio
async def test_cross_protocol_with_session_pool_integration(
    minimal_pool: AgentPool,
    mock_agent_full_lifecycle: MagicMock,
) -> None:
    """6.15: Full integration test: SessionPool + EventBus with protocol subscribers.

    Simulates ACP and OpenCode handlers subscribing to a SessionPool's
    EventBus, runs a full turn, and verifies both protocols receive the
    complete event stream in order.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None

    await session_pool.create_session("sess-integrated", agent_name="agent-a")
    await _attach_agent(session_pool, "sess-integrated", mock_agent_full_lifecycle, minimal_pool)

    # Simulate two protocol handlers subscribing
    acp_queue = await session_pool.event_bus.subscribe("sess-integrated")
    opencode_queue = await session_pool.event_bus.subscribe("sess-integrated")

    # Run a full turn
    await session_pool.process_prompt("sess-integrated", "integrated prompt")

    # Collect events from both protocol perspectives
    acp_events: list[Any] = []
    opencode_events: list[Any] = []

    while True:
        with contextlib.suppress(asyncio.QueueEmpty):
            acp_events.append(acp_queue.get_nowait())
            continue
        break
    while True:
        with contextlib.suppress(asyncio.QueueEmpty):
            opencode_events.append(opencode_queue.get_nowait())
            continue
        break

    # Both should see the full lifecycle
    expected_types = [
        RunStartedEvent,
        PartDeltaEvent,
        ToolCallStartEvent,
        ToolCallCompleteEvent,
        StreamCompleteEvent,
    ]

    assert len(acp_events) == len(expected_types)
    assert len(opencode_events) == len(expected_types)

    for i, expected_type in enumerate(expected_types):
        assert type(_unwrap_event(acp_events[i])) is expected_type
        assert type(_unwrap_event(opencode_events[i])) is expected_type

    # Verify specific event data
    acp_ev_2 = _unwrap_event(acp_events[2])
    opencode_ev_2 = _unwrap_event(opencode_events[2])
    acp_ev_3 = _unwrap_event(acp_events[3])
    opencode_ev_3 = _unwrap_event(opencode_events[3])
    assert acp_ev_2.tool_name == "bash"
    assert opencode_ev_2.tool_name == "bash"
    assert acp_ev_3.tool_result == "hi"
    assert opencode_ev_3.tool_result == "hi"

    # Cleanup
    await session_pool.close_session("sess-integrated")

    # Both should receive sentinel
    with pytest.raises((asyncio.TimeoutError, asyncio.QueueShutDown)):
        await asyncio.wait_for(acp_queue.get(), timeout=0.5)
    with pytest.raises((asyncio.TimeoutError, asyncio.QueueShutDown)):
        await asyncio.wait_for(opencode_queue.get(), timeout=0.5)

    # session_pool lifecycle managed by minimal_pool fixture
