"""Tests for OpenCodeEventBridge behavior parity.

Validates that the event bridge correctly dual-publishes events to both
legacy SSE subscribers and the SessionPool EventBus, while preserving
backward compatibility for the legacy path.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, Mock

import pytest

from wolfharness.agents.events.events import (
    CustomEvent,
    RunFailedEvent,
    RunStartedEvent,
    StreamCompleteEvent,
)
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.core import EventBus, EventEnvelope
from wolfharness_server.opencode_server.event_processor_context import (
    EventProcessorContext,
)
from wolfharness_server.opencode_server.models import (
    AssistantMessage,
    MessagePath,
    MessageTime,
    MessageUpdatedEvent,
    MessageWithParts,
    SessionIdleEvent,
    SessionStatus,
    SessionStatusEvent,
)
from wolfharness_server.opencode_server.models.events import ServerConnectedEvent
from wolfharness_server.opencode_server.opencode_event_bridge import (
    OpenCodeEventBridgeMixin,
)
from wolfharness_server.opencode_server.state import ServerState


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from wolfharness_server.opencode_server.models.events import Event


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def bridged_state(tmp_project_dir: Path, mock_agent: Mock) -> ServerState:
    """Create a ServerState with an active OpenCodeEventBridge."""
    from wolfharness.orchestrator.core import EventBus

    # Wire a real EventBus into the mock pool so __post_init__ can discover it
    mock_agent.host_context.session_pool.event_bus = EventBus()

    return ServerState(
        working_dir=str(tmp_project_dir),
        agent=mock_agent,
        session_controller=Mock(),  # non-None triggers bridge instantiation
    )


@pytest.fixture
def event_bus(bridged_state: ServerState) -> EventBus:
    """Return the EventBus attached to the bridged state."""
    assert bridged_state.event_bridge is not None
    return bridged_state.event_bridge._event_bus


# =============================================================================
# Legacy path tests (no session_controller)
# =============================================================================


@pytest.mark.anyio
async def test_legacy_path_no_event_push(
    tmp_project_dir: Path,
    mock_agent: Mock,
) -> None:
    """Without a session_controller, broadcast_event is a no-op (no queue push).

    The legacy path that pushed events directly to event_subscribers queues
    has been removed. Event delivery now flows exclusively through the EventBus
    via the event bridge. When no bridge is available, broadcast_event logs
    a debug message and returns without pushing to any queues.
    """
    state = ServerState(
        working_dir=str(tmp_project_dir),
        agent=mock_agent,
        session_controller=None,
    )
    queue: asyncio.Queue[Any] = asyncio.Queue()
    state.event_subscribers.append(queue)

    event = SessionStatusEvent.create("sess-legacy", SessionStatus(type="busy"))
    await state.broadcast_event(event)

    # No events should be pushed to subscriber queues
    assert queue.qsize() == 0


@pytest.mark.anyio
async def test_legacy_path_no_bridge_created(
    tmp_project_dir: Path,
    mock_agent: Mock,
) -> None:
    """ServerState without session_controller has no event_bridge."""
    state = ServerState(
        working_dir=str(tmp_project_dir),
        agent=mock_agent,
        session_controller=None,
    )
    assert state.event_bridge is None


# =============================================================================
# SessionPool path tests (bridge active)
# =============================================================================


@pytest.mark.anyio
async def test_session_pool_path_does_not_push_to_sse_queues(
    bridged_state: ServerState,
) -> None:
    """With the bridge active, events flow through EventBus only (not SSE queues).

    The legacy event_subscribers queue push has been removed from
    event_bridge.publish(). Events are delivered exclusively through
    the EventBus subscription mechanism.
    """
    queue: asyncio.Queue[Any] = asyncio.Queue()
    bridged_state.event_subscribers.append(queue)

    event = SessionStatusEvent.create("sess-pool", SessionStatus(type="busy"))
    await bridged_state.broadcast_event(event)

    # SSE subscriber queues should NOT receive events (dead code removed)
    assert queue.qsize() == 0


@pytest.mark.anyio
async def test_bridge_republishes_to_event_bus(
    bridged_state: ServerState,
    event_bus: EventBus,
) -> None:
    """Events are republished to the EventBus as CustomEvent wrappers."""
    subscriber = await event_bus.subscribe("sess-pool")

    event = SessionStatusEvent.create("sess-pool", SessionStatus(type="busy"))
    await bridged_state.broadcast_event(event)

    # Allow the async publish to propagate
    await asyncio.sleep(0.05)

    envelope = subscriber.get_nowait()
    assert isinstance(envelope, EventEnvelope)
    wrapped = envelope.event
    assert isinstance(wrapped, CustomEvent)
    assert wrapped.event_data is event
    assert wrapped.event_type == "opencode:session.status"


@pytest.mark.anyio
async def test_bridge_wraps_different_event_types(
    bridged_state: ServerState,
    event_bus: EventBus,
) -> None:
    """Various OpenCode event types are correctly wrapped."""
    subscriber = await event_bus.subscribe("sess-mixed")

    events: list[Event] = [
        SessionStatusEvent.create("sess-mixed", SessionStatus(type="busy")),
        SessionIdleEvent.create("sess-mixed"),
    ]

    for evt in events:
        await bridged_state.broadcast_event(evt)

    await asyncio.sleep(0.05)

    for _i, evt in enumerate(events):
        envelope = subscriber.get_nowait()
        assert isinstance(envelope, EventEnvelope)
        wrapped = envelope.event
        assert isinstance(wrapped, CustomEvent)
        assert wrapped.event_data is evt
        expected_type = f"opencode:{evt.type}"
        assert wrapped.event_type == expected_type


@pytest.mark.anyio
async def test_global_event_not_republished_to_event_bus(
    bridged_state: ServerState,
    event_bus: EventBus,
) -> None:
    """Global events without session_id are NOT republished to EventBus."""
    # Use a dummy session just to have a subscriber queue; the event itself
    # has no session_id so it should not be published there.
    subscriber = await event_bus.subscribe("global-session")

    event = ServerConnectedEvent()
    await bridged_state.broadcast_event(event)

    await asyncio.sleep(0.05)

    # EventBus should receive nothing because the event has no session_id
    with pytest.raises(asyncio.QueueEmpty):
        subscriber.get_nowait()

    # SSE subscriber queues also do NOT receive events (dead code removed)
    queue: asyncio.Queue[Any] = asyncio.Queue()
    bridged_state.event_subscribers.append(queue)
    await bridged_state.broadcast_event(event)
    assert queue.qsize() == 0


# =============================================================================
# Bridge unit tests
# =============================================================================


@pytest.mark.anyio
async def test_bridge_publish_does_not_push_to_sse_queues(
    bridged_state: ServerState,
) -> None:
    """Bridge.publish does NOT push to SSE subscriber queues (dead code removed).

    Event delivery flows exclusively through the EventBus. The legacy
    event_subscribers queue push has been removed.
    """
    queue: asyncio.Queue[Any] = asyncio.Queue()
    bridged_state.event_subscribers.append(queue)

    event = SessionStatusEvent.create("sess-unit", SessionStatus(type="idle"))
    assert bridged_state.event_bridge is not None
    await bridged_state.event_bridge.publish(event)

    # SSE subscriber queues should NOT receive events
    assert queue.qsize() == 0


@pytest.mark.anyio
async def test_bridge_extract_session_id_variations(
    bridged_state: ServerState,
) -> None:
    """_extract_session_id handles events with and without session_id."""
    bridge = bridged_state.event_bridge
    assert bridge is not None

    # Event with session_id
    status_event = SessionStatusEvent.create("sess-1", SessionStatus(type="busy"))
    assert bridge._extract_session_id(status_event) == "sess-1"

    # Event without session_id
    connected_event = ServerConnectedEvent()
    assert bridge._extract_session_id(connected_event) is None

    # Edge case: object with no properties attribute
    class NoProperties:
        pass

    assert bridge._extract_session_id(NoProperties()) is None  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_bridge_wrap_event_format(
    bridged_state: ServerState,
) -> None:
    """_wrap_event produces a correctly formatted CustomEvent."""
    bridge = bridged_state.event_bridge
    assert bridge is not None

    event = SessionIdleEvent.create("sess-wrap")
    wrapped = bridge._wrap_event(event)

    assert isinstance(wrapped, CustomEvent)
    assert wrapped.event_data is event
    assert wrapped.event_type == "opencode:session.idle"
    assert wrapped.source == "opencode_event_bridge"


@pytest.mark.anyio
async def test_bridge_isolation_between_sessions(
    bridged_state: ServerState,
    event_bus: EventBus,
) -> None:
    """Events for session A do not leak into session B's EventBus subscription."""
    sub_a = await event_bus.subscribe("sess-a")
    sub_b = await event_bus.subscribe("sess-b")

    await bridged_state.broadcast_event(
        SessionStatusEvent.create("sess-a", SessionStatus(type="busy"))
    )
    await asyncio.sleep(0.05)

    envelope = sub_a.get_nowait()
    assert isinstance(envelope, EventEnvelope)
    wrapped = envelope.event
    assert wrapped.event_data.properties.session_id == "sess-a"

    with pytest.raises(asyncio.QueueEmpty):
        sub_b.get_nowait()


# =============================================================================
# --- Merged from test_event_bridge_review.py ---
# =============================================================================

"""Tests for stop_event_consumer exception handling (2nd round review).

Verifies that when one child's stop_event_consumer raises an exception,
the remaining children are still stopped (the loop doesn't break).
"""

pytestmark = pytest.mark.unit


class _FakeBridge(OpenCodeEventBridgeMixin):
    """Minimal concrete subclass for testing the mixin."""

    def __init__(self) -> None:
        self.session_pool = MagicMock()
        self.server_state = MagicMock()
        self._contexts: dict[str, Any] = {}
        self._adapters: dict[str, Any] = {}
        self._message_registered: dict[str, bool] = {}
        self._child_to_parent: dict[str, str] = {}
        self._child_spawns: dict[str, Any] = {}
        self._children_of: dict[str, set[str]] = {}
        self._resume_contexts: dict[str, dict[str, Any]] = {}
        self._pending_message_ids: dict[str, str] = {}
        self._pending_message_metadata: dict[str, dict[str, str | None]] = {}
        self._steer_split_ids: dict[str, set[str]] = {}

    def set_session_context_data(self, session_id: str, data: dict[str, Any]) -> None:
        self._resume_contexts[session_id] = data

    def get_session_context_data(self, session_id: str) -> dict[str, Any] | None:
        return self._resume_contexts.pop(session_id, None)


@pytest.mark.anyio
@pytest.mark.unit
async def test_stop_event_consumer_exception_does_not_break_loop() -> None:
    """Stop one child's consumer must not break the loop.

    When stop_event_consumer raises for one child, remaining children
    must still be stopped.
    """
    bridge = _FakeBridge()

    child1 = "child-1"
    child2 = "child-2"
    parent = "parent-session"
    bridge._children_of[parent] = {child1, child2}

    attempted: list[str] = []

    async def fake_stop(child_id: str) -> None:
        attempted.append(child_id)
        if child_id == child1:
            raise RuntimeError("simulated failure for child-1")

    bridge.stop_event_consumer = fake_stop  # type: ignore[method-assign]

    await bridge._after_consumer_loop(parent)

    assert len(attempted) == 2, f"Expected both children to be attempted, but only got {attempted}"
    assert child1 in attempted, "child-1 was not attempted"
    assert child2 in attempted, "child-2 was not attempted"
    assert parent not in bridge._children_of


@pytest.mark.anyio
@pytest.mark.unit
async def test_stop_event_consumer_all_succeed() -> None:
    """Normal case: all children stopped successfully."""
    bridge = _FakeBridge()

    child1 = "child-1"
    child2 = "child-2"
    parent = "parent-session"
    bridge._children_of[parent] = {child1, child2}

    attempted: list[str] = []

    async def fake_stop(child_id: str) -> None:
        attempted.append(child_id)

    bridge.stop_event_consumer = fake_stop  # type: ignore[method-assign]

    await bridge._after_consumer_loop(parent)

    assert len(attempted) == 2
    assert parent not in bridge._children_of


@pytest.mark.anyio
@pytest.mark.unit
async def test_stop_event_consumer_no_children() -> None:
    """When there are no children, _after_consumer_loop runs cleanly."""
    bridge = _FakeBridge()
    parent = "parent-session"

    attempted: list[str] = []

    async def fake_stop(child_id: str) -> None:
        attempted.append(child_id)

    bridge.stop_event_consumer = fake_stop  # type: ignore[method-assign]

    await bridge._after_consumer_loop(parent)

    assert attempted == []


# =============================================================================
# D3/D2: time.completed finalization tests
# =============================================================================


async def _async_iter(items: list[Any]) -> AsyncIterator[Any]:
    """Yield items from a list as an async iterator."""
    for item in items:
        yield item


def _make_test_ctx(
    session_id: str = "sess-d3",
    *,
    completed: int | None = None,
    msg_id: str = "msg-assistant-1",
) -> EventProcessorContext:
    """Create an EventProcessorContext with an AssistantMessage for D3 tests.

    Args:
        session_id: The session ID to use.
        completed: The value for time.completed (None = not finalized).
        msg_id: The assistant message ID.

    Returns:
        An EventProcessorContext with a properly constructed AssistantMessage.
    """
    assistant_msg = MessageWithParts.assistant(
        message_id=msg_id,
        session_id=session_id,
        time=MessageTime(created=1000, completed=completed),
        agent_name="test-agent",
        model_id="test-model",
        parent_id=session_id,
        provider_id="test-provider",
        path=MessagePath(cwd="/tmp", root="/tmp"),
        mode="test-agent",
    )
    return EventProcessorContext(
        session_id=session_id,
        assistant_msg_id=msg_id,
        assistant_msg=assistant_msg,
        state=MagicMock(),
        working_dir="/tmp",
    )


def _setup_bridge_for_handle(
    session_id: str = "sess-d3",
    *,
    completed: int | None = None,
    message_registered: bool = True,
) -> tuple[_FakeBridge, EventProcessorContext, list[Any]]:
    """Set up a _FakeBridge ready for _handle_event calls.

    Returns:
        A tuple of (bridge, ctx, broadcast_calls) where broadcast_calls is
        a list that accumulates all events passed to broadcast_event.
    """
    bridge = _FakeBridge()
    ctx = _make_test_ctx(session_id, completed=completed)

    bridge._contexts[session_id] = ctx
    bridge._message_registered[session_id] = message_registered

    # Adapter mock: convert_event returns empty async iterator
    adapter_mock = MagicMock()
    adapter_mock.convert_event = lambda _e: _async_iter([])
    bridge._adapters[session_id] = adapter_mock

    # Track broadcast_event calls
    broadcast_calls: list[Any] = []

    async def fake_broadcast(event: Any) -> None:
        broadcast_calls.append(event)

    bridge.server_state.broadcast_event = fake_broadcast  # type: ignore[method-assign]
    bridge.server_state.working_dir = "/tmp"
    bridge.server_state.resolve_default_model_info = Mock(
        return_value=("test-model", "test-provider")
    )
    bridge.session_pool.sessions.get_session = Mock(return_value=None)

    return bridge, ctx, broadcast_calls


@pytest.mark.anyio
@pytest.mark.unit
async def test_stream_complete_sets_time_completed() -> None:
    """D3: StreamCompleteEvent must set time.completed on assistant message.

    The prompt_async path returns 204 immediately and never finalizes the
    assistant message. The event bridge must set time.completed when it
    receives StreamCompleteEvent.
    """
    from unittest.mock import AsyncMock, patch

    session_id = "sess-d3"
    bridge, ctx, broadcast_calls = _setup_bridge_for_handle(
        session_id, completed=None, message_registered=True
    )

    event = StreamCompleteEvent(
        message=ChatMessage(content="done", role="assistant"),
        session_id=session_id,
    )
    envelope = EventEnvelope(source_session_id=session_id, event=event)

    with (
        patch(
            "wolfharness_server.opencode_server.opencode_event_bridge.set_session_status",
            new_callable=AsyncMock,
        ),
        patch(
            "wolfharness_server.opencode_server.opencode_event_bridge.append_message_to_session",
            new_callable=AsyncMock,
        ) as mock_append,
    ):
        await bridge._handle_event(session_id, envelope)

    # Assert time.completed was set
    info = ctx.assistant_msg.info
    assert isinstance(info, AssistantMessage)
    assert info.time.completed is not None, "time.completed should be set after StreamCompleteEvent"
    assert info.time.completed > 1000, "time.completed should be > created time"

    # Assert MessageUpdatedEvent was broadcast
    updated_events = [e for e in broadcast_calls if isinstance(e, MessageUpdatedEvent)]
    assert len(updated_events) >= 1, "MessageUpdatedEvent should be broadcast"

    # Assert append_message_to_session was called (persistence)
    assert mock_append.called, "append_message_to_session should be called"


@pytest.mark.anyio
@pytest.mark.unit
async def test_stream_complete_skips_if_already_completed() -> None:
    """D3: StreamCompleteEvent should not overwrite an existing time.completed."""
    from unittest.mock import AsyncMock, patch

    session_id = "sess-d3"
    bridge, ctx, broadcast_calls = _setup_bridge_for_handle(
        session_id, completed=5000, message_registered=True
    )

    event = StreamCompleteEvent(
        message=ChatMessage(content="done", role="assistant"),
        session_id=session_id,
    )
    envelope = EventEnvelope(source_session_id=session_id, event=event)

    with (
        patch(
            "wolfharness_server.opencode_server.opencode_event_bridge.set_session_status",
            new_callable=AsyncMock,
        ),
        patch(
            "wolfharness_server.opencode_server.opencode_event_bridge.append_message_to_session",
            new_callable=AsyncMock,
        ) as mock_append,
    ):
        await bridge._handle_event(session_id, envelope)

    # Assert time.completed was NOT overwritten
    info = ctx.assistant_msg.info
    assert isinstance(info, AssistantMessage)
    assert info.time.completed == 5000, "time.completed should not be overwritten"

    # Assert no MessageUpdatedEvent was broadcast for finalization
    updated_events = [e for e in broadcast_calls if isinstance(e, MessageUpdatedEvent)]
    assert len(updated_events) == 0, (
        "No MessageUpdatedEvent should be broadcast if already completed"
    )

    # Assert append_message_to_session was NOT called for finalization
    assert not mock_append.called, (
        "append_message_to_session should not be called if already completed"
    )


@pytest.mark.anyio
@pytest.mark.unit
async def test_d1_reset_finalizes_previous_turn() -> None:
    """D3/D2: D1 reset must finalize previous turn's assistant message.

    When RunStartedEvent arrives for a subsequent turn (consumer already
    running), the D1 reset creates a new assistant message. The previous
    turn's message must have time.completed set before it's replaced.
    """
    from unittest.mock import AsyncMock, patch

    session_id = "sess-d1"
    bridge, ctx, broadcast_calls = _setup_bridge_for_handle(
        session_id, completed=None, message_registered=True
    )

    original_msg_id = ctx.assistant_msg_id

    event = RunStartedEvent(
        run_id="run-2",
        agent_name="test-agent",
        session_id=session_id,
    )
    envelope = EventEnvelope(source_session_id=session_id, event=event)

    with (
        patch(
            "wolfharness_server.opencode_server.opencode_event_bridge.set_session_status",
            new_callable=AsyncMock,
        ),
        patch(
            "wolfharness_server.opencode_server.opencode_event_bridge.append_message_to_session",
            new_callable=AsyncMock,
        ) as mock_append,
    ):
        await bridge._handle_event(session_id, envelope)

    # Find the MessageUpdatedEvent for the finalization (first one broadcast)
    updated_events = [e for e in broadcast_calls if isinstance(e, MessageUpdatedEvent)]
    assert len(updated_events) >= 2, (
        "Should broadcast at least 2 MessageUpdatedEvents: "
        "1 for finalization, 1 for new message registration"
    )

    # First MessageUpdatedEvent should be for the previous message (finalization)
    first_update = updated_events[0]
    assert first_update.properties.info.id == original_msg_id, (
        "First MessageUpdatedEvent should be for the previous assistant message"
    )
    # The finalization should have set time.completed
    first_info = first_update.properties.info
    assert isinstance(first_info, AssistantMessage)
    assert first_info.time.completed is not None, (
        "Previous turn's time.completed should be set during D1 reset"
    )

    # Second MessageUpdatedEvent should be for the new message
    second_update = updated_events[1]
    assert second_update.properties.info.id != original_msg_id, (
        "Second MessageUpdatedEvent should be for the new assistant message"
    )

    # append_message_to_session should have been called at least twice:
    # 1 for finalization, 1 for new message registration
    assert mock_append.call_count >= 2, (
        "append_message_to_session should be called for both finalization and registration"
    )

    # The new assistant message should have completed=None (fresh turn)
    new_info = ctx.assistant_msg.info
    assert isinstance(new_info, AssistantMessage)
    assert new_info.time.completed is None, "New turn's time.completed should be None"


@pytest.mark.anyio
@pytest.mark.unit
async def test_d1_reset_skips_finalize_if_already_completed() -> None:
    """D1 reset should skip finalization if time.completed is already set."""
    from unittest.mock import AsyncMock, patch

    session_id = "sess-d1"
    bridge, _ctx, broadcast_calls = _setup_bridge_for_handle(
        session_id, completed=5000, message_registered=True
    )

    event = RunStartedEvent(
        run_id="run-2",
        agent_name="test-agent",
        session_id=session_id,
    )
    envelope = EventEnvelope(source_session_id=session_id, event=event)

    with (
        patch(
            "wolfharness_server.opencode_server.opencode_event_bridge.set_session_status",
            new_callable=AsyncMock,
        ),
        patch(
            "wolfharness_server.opencode_server.opencode_event_bridge.append_message_to_session",
            new_callable=AsyncMock,
        ) as mock_append,
    ):
        await bridge._handle_event(session_id, envelope)

    # Only 1 MessageUpdatedEvent (for the new message registration, no finalization)
    updated_events = [e for e in broadcast_calls if isinstance(e, MessageUpdatedEvent)]
    assert len(updated_events) == 1, (
        "Only 1 MessageUpdatedEvent for new message; no finalization needed"
    )

    # append_message_to_session called once (for new message only, no finalization)
    assert mock_append.call_count == 1, (
        "append_message_to_session should be called once (new message only)"
    )


@pytest.mark.anyio
@pytest.mark.unit
async def test_d2_warning_logged_on_incomplete_turn() -> None:
    """D2: Warning should be logged when finalizing an incomplete turn.

    If the D1 reset finds time.completed is None, it means StreamCompleteEvent
    was missed or not yet processed. A warning should be logged so the D2
    red flag (running turn killed by new turn) is visible.
    """
    from unittest.mock import AsyncMock, patch

    import wolfharness_server.opencode_server.opencode_event_bridge as bridge_module

    session_id = "sess-d2"
    bridge, _ctx, _broadcast_calls = _setup_bridge_for_handle(
        session_id, completed=None, message_registered=True
    )

    event = RunStartedEvent(
        run_id="run-2",
        agent_name="test-agent",
        session_id=session_id,
    )
    envelope = EventEnvelope(source_session_id=session_id, event=event)

    with (
        patch(
            "wolfharness_server.opencode_server.opencode_event_bridge.set_session_status",
            new_callable=AsyncMock,
        ),
        patch(
            "wolfharness_server.opencode_server.opencode_event_bridge.append_message_to_session",
            new_callable=AsyncMock,
        ),
        patch.object(bridge_module.logger, "warning") as mock_warning,
    ):
        await bridge._handle_event(session_id, envelope)

    # Assert a warning was logged about finalizing an incomplete turn
    mock_warning.assert_called_once()
    call_args = mock_warning.call_args
    assert (
        "incomplete turn" in call_args.args[0].lower() or "StreamCompleteEvent" in call_args.args[0]
    )


@pytest.mark.anyio
@pytest.mark.unit
async def test_run_failed_sets_time_completed() -> None:
    """D3: RunFailedEvent must also finalize time.completed.

    If a run fails, StreamCompleteEvent is not emitted. Without this
    finalization, the next turn's D1 reset would log a false-positive
    warning about a missed StreamCompleteEvent.
    """
    from unittest.mock import AsyncMock, patch

    session_id = "sess-rf"
    bridge, ctx, broadcast_calls = _setup_bridge_for_handle(
        session_id, completed=None, message_registered=True
    )

    event = RunFailedEvent(
        run_id="run-fail-1",
        exception=RuntimeError("test failure"),
        session_id=session_id,
    )
    envelope = EventEnvelope(source_session_id=session_id, event=event)

    with (
        patch(
            "wolfharness_server.opencode_server.opencode_event_bridge.set_session_status",
            new_callable=AsyncMock,
        ),
        patch(
            "wolfharness_server.opencode_server.opencode_event_bridge.append_message_to_session",
            new_callable=AsyncMock,
        ) as mock_append,
    ):
        await bridge._handle_event(session_id, envelope)

    # Assert time.completed was set despite run failure
    info = ctx.assistant_msg.info
    assert isinstance(info, AssistantMessage)
    assert info.time.completed is not None, "time.completed should be set after RunFailedEvent"

    # Assert MessageUpdatedEvent was broadcast
    updated_events = [e for e in broadcast_calls if isinstance(e, MessageUpdatedEvent)]
    assert len(updated_events) >= 1, "MessageUpdatedEvent should be broadcast"

    # Assert append_message_to_session was called
    assert mock_append.called, "append_message_to_session should be called"


@pytest.mark.anyio
@pytest.mark.unit
async def test_stream_complete_cancelled_sets_time_completed() -> None:
    """D3: StreamCompleteEvent with cancelled=True should still set time.completed.

    A cancelled run still completed (just was cancelled). The UI needs
    time.completed to show the message as finished.
    """
    from unittest.mock import AsyncMock, patch

    session_id = "sess-cancel"
    bridge, ctx, broadcast_calls = _setup_bridge_for_handle(
        session_id, completed=None, message_registered=True
    )

    event = StreamCompleteEvent(
        message=ChatMessage(content="partial", role="assistant"),
        cancelled=True,
        session_id=session_id,
    )
    envelope = EventEnvelope(source_session_id=session_id, event=event)

    with (
        patch(
            "wolfharness_server.opencode_server.opencode_event_bridge.set_session_status",
            new_callable=AsyncMock,
        ),
        patch(
            "wolfharness_server.opencode_server.opencode_event_bridge.append_message_to_session",
            new_callable=AsyncMock,
        ),
    ):
        await bridge._handle_event(session_id, envelope)

    # Assert time.completed was set even for cancelled runs
    info = ctx.assistant_msg.info
    assert isinstance(info, AssistantMessage)
    assert info.time.completed is not None, "time.completed should be set for cancelled runs"

    updated_events = [e for e in broadcast_calls if isinstance(e, MessageUpdatedEvent)]
    assert len(updated_events) >= 1


@pytest.mark.anyio
@pytest.mark.unit
async def test_finalize_skips_when_no_context() -> None:
    """_finalize_assistant_time should be a no-op when ctx is None."""
    from unittest.mock import AsyncMock, patch

    session_id = "sess-no-ctx"
    bridge = _FakeBridge()
    # Deliberately do NOT set bridge._contexts[session_id]
    bridge._message_registered[session_id] = True

    event = StreamCompleteEvent(
        message=ChatMessage(content="done", role="assistant"),
        session_id=session_id,
    )
    envelope = EventEnvelope(source_session_id=session_id, event=event)

    with (
        patch(
            "wolfharness_server.opencode_server.opencode_event_bridge.set_session_status",
            new_callable=AsyncMock,
        ),
        patch(
            "wolfharness_server.opencode_server.opencode_event_bridge.append_message_to_session",
            new_callable=AsyncMock,
        ) as mock_append,
    ):
        # Should not raise even though ctx is None
        await bridge._handle_event(session_id, envelope)

    # append_message_to_session should NOT be called (no ctx to finalize)
    # But it might be called for message registration if _message_registered
    # is True and the code reaches that block. Since ctx is None, the code
    # returns early at "if ctx is None: return" (line ~465).
    assert not mock_append.called, "append_message_to_session should not be called when ctx is None"
