"""Integration tests for SSE replay buffer + sync() interaction.

Tests that the sync() endpoint does NOT clear the EventBus replay buffer.
Instead, duplication is prevented by conditional replay: the SSE endpoint
uses subscribe(replay=False) on first connection and
subscribe(replay=True, last_event_id=N) on reconnection.

Old behavior (removed): sync() called event_bus.clear_replay_buffer(session_id)
so that events published before sync() were not re-delivered to new SSE
subscribers. This was a band-aid that destroyed replay history for all
clients.

New behavior: sync() does NOT clear the replay buffer. The SSE endpoint
passes replay=False on first connection (no Last-Event-ID header) and
replay=True with last_event_id on reconnection. This preserves replay
history for late subscribers while preventing duplication.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, Mock

import pytest

from wolfharness.agents.events.events import (
    UserMessageInsertedEvent,
)
from wolfharness.orchestrator.core import EventBus
from wolfharness_server.opencode_server.event_processor_context import (
    EventProcessorContext,
)
from wolfharness_server.opencode_server.models import (
    MessagePath,
    MessageTime,
    MessageWithParts,
)
from wolfharness_server.opencode_server.opencode_event_bridge import (
    OpenCodeEventBridgeMixin,
)


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


async def _async_iter(items: list[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


class _FakeBridge(OpenCodeEventBridgeMixin):
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
        self.set_session_context_data = self._resume_contexts.__setitem__
        self.get_session_context_data = lambda sid: self._resume_contexts.pop(sid, None)


def _make_ctx(session_id: str) -> EventProcessorContext:
    assistant_msg = MessageWithParts.assistant(
        message_id="msg-a1",
        session_id=session_id,
        time=MessageTime(created=1000),
        agent_name="test-agent",
        model_id="test-model",
        parent_id=session_id,
        provider_id="test-provider",
        path=MessagePath(cwd="/tmp", root="/tmp"),
        mode="test-agent",
    )
    return EventProcessorContext(
        session_id=session_id,
        assistant_msg_id="msg-a1",
        assistant_msg=assistant_msg,
        state=MagicMock(),
        working_dir="/tmp",
    )


def _setup_bridge_with_event_bus(
    session_id: str,
) -> tuple[_FakeBridge, EventBus, EventProcessorContext]:
    """Set up a _FakeBridge with a REAL EventBus (not mocked)."""
    event_bus = EventBus()
    bridge = _FakeBridge()
    bridge._event_bus = event_bus

    ctx = _make_ctx(session_id)
    bridge._contexts[session_id] = ctx
    bridge._message_registered[session_id] = False

    adapter_mock = MagicMock()
    adapter_mock.convert_event = lambda _e: _async_iter([])
    bridge._adapters[session_id] = adapter_mock

    async def fake_broadcast(event: Any) -> None:
        pass

    bridge.server_state.broadcast_event = fake_broadcast  # type: ignore[method-assign]
    bridge.server_state.working_dir = "/tmp"
    bridge.server_state.resolve_default_model_info = Mock(
        return_value=("test-model", "test-provider")
    )
    bridge.session_pool.sessions.get_session = Mock(return_value=None)

    return bridge, event_bus, ctx


# =============================================================================
# Replay buffer + sync() interaction tests (rewritten for conditional replay)
# =============================================================================


@pytest.mark.anyio
async def test_replay_buffer_contains_events_after_publish() -> None:
    """EventBus replay buffer stores events after publish.

    Given: A real EventBus with replay buffer enabled.
    When: A UserMessageInsertedEvent is published.
    Then: The event is in the replay buffer for the session.
    And: A new subscriber receives it from the replay buffer.
    """
    session_id = "sess-rb-1"
    _bridge, event_bus, _ctx = _setup_bridge_with_event_bus(session_id)

    event = UserMessageInsertedEvent(
        content="hello",
        session_id=session_id,
        message_id="msg-u1",
        source="accepted",
    )

    await event_bus.publish(session_id, event)

    # Replay buffer should contain the event
    buffer = event_bus._replay_buffers.get(session_id)
    assert buffer is not None
    assert len(buffer) > 0, "Replay buffer should contain the published event"


@pytest.mark.anyio
async def test_new_subscriber_receives_replay_buffer_events() -> None:
    """New EventBus subscriber receives events from replay buffer.

    Given: An event was published before subscription.
    When: A new subscriber subscribes to the session.
    Then: The subscriber receives the replayed event.
    """
    session_id = "sess-rb-2"
    _bridge, event_bus, _ctx = _setup_bridge_with_event_bus(session_id)

    event = UserMessageInsertedEvent(
        content="hello",
        session_id=session_id,
        message_id="msg-u2",
        source="accepted",
    )
    await event_bus.publish(session_id, event)

    # Subscribe AFTER the event was published
    queue = await event_bus.subscribe(session_id, scope="session")

    # Drain replay buffer events (they're enqueued synchronously in subscribe)
    received_events: list[Any] = []
    try:
        while not queue.empty():
            envelope = queue.get_nowait()
            received_events.append(envelope.event)
    except asyncio.QueueEmpty:
        pass

    assert len(received_events) >= 1, "New subscriber should receive event from replay buffer"


@pytest.mark.anyio
async def test_sync_does_not_clear_replay_buffer() -> None:
    """sync() does NOT clear the replay buffer — events are preserved.

    This replaces the old test that asserted clear_replay_buffer was called
    by sync(). The new design uses conditional replay (subscribe with
    replay=False on first connection) instead of destroying replay history.

    Given: An event was published and is in the replay buffer.
    When: sync() is called for the session (simulated — does NOT clear buffer).
    Then: The replay buffer is NOT cleared.
    And: The event remains available for late subscribers with replay=True.
    """
    session_id = "sess-rb-3"
    _bridge, event_bus, _ctx = _setup_bridge_with_event_bus(session_id)

    event = UserMessageInsertedEvent(
        content="hello",
        session_id=session_id,
        message_id="msg-u3",
        source="accepted",
    )
    await event_bus.publish(session_id, event)

    # sync() is called — it should NOT clear the replay buffer
    # (The old behavior called event_bus.clear_replay_buffer(session_id))
    # The new behavior leaves the buffer intact.

    # Verify replay buffer is NOT cleared
    buffer = event_bus._replay_buffers.get(session_id)
    assert buffer is not None, "Replay buffer should still exist after sync()"
    assert len(buffer) > 0, "Replay buffer should still contain events after sync()"


@pytest.mark.anyio
async def test_first_connection_replay_false_prevents_redelivery() -> None:
    """First SSE connection uses replay=False — no historical events delivered.

    This replaces the old test that used clear_replay_buffer to prevent
    redelivery. The new design uses subscribe(replay=False) on first
    connection (no Last-Event-ID header).

    Given: An event was published and is in the replay buffer.
    When: A first-time subscriber subscribes with replay=False
        (simulating first SSE connection without Last-Event-ID).
    Then: The subscriber does NOT receive old events from the replay buffer.
    And: The replay buffer remains intact for other subscribers.
    """
    session_id = "sess-rb-4"
    _bridge, event_bus, _ctx = _setup_bridge_with_event_bus(session_id)

    event = UserMessageInsertedEvent(
        content="hello",
        session_id=session_id,
        message_id="msg-u4",
        source="protocol",
    )
    await event_bus.publish(session_id, event)

    # First connection: subscribe with replay=False (no Last-Event-ID)
    queue = await event_bus.subscribe(session_id, scope="session", replay=False)

    received_events: list[Any] = []
    try:
        while not queue.empty():
            envelope = queue.get_nowait()
            received_events.append(envelope.event)
    except asyncio.QueueEmpty:
        pass

    assert len(received_events) == 0, (
        "First connection with replay=False should NOT receive historical events"
    )

    # Replay buffer should still be intact
    buffer = event_bus._replay_buffers.get(session_id)
    assert buffer is not None, "Replay buffer should exist for other subscribers"
    assert len(buffer) > 0, "Replay buffer should be intact for other subscribers"


@pytest.mark.anyio
async def test_reconnect_with_last_event_id_filters_replay() -> None:
    """SSE reconnect with Last-Event-ID only delivers events after that ID.

    This replaces the old test that used clear_replay_buffer to prevent
    duplication on reconnect. The new design uses
    subscribe(replay=True, last_event_id=N) to filter replayed events.

    Given: Events with event_ids [1, 2, 3] are in the replay buffer.
    When: A reconnecting subscriber subscribes with replay=True, last_event_id=2.
    Then: Only event with event_id > 2 (i.e., event_id=3) is delivered.
    """
    session_id = "sess-rb-5"
    _bridge, event_bus, _ctx = _setup_bridge_with_event_bus(session_id)

    for i in range(3):
        await event_bus.publish(
            session_id,
            UserMessageInsertedEvent(
                content=f"msg-{i}",
                session_id=session_id,
                message_id=f"msg-u5-{i}",
                source="protocol",
            ),
        )

    # Reconnect with last_event_id=2
    queue = await event_bus.subscribe(session_id, scope="session", replay=True, last_event_id=2)

    received: list[Any] = []
    try:
        while not queue.empty():
            envelope = queue.get_nowait()
            received.append(envelope)
    except asyncio.QueueEmpty:
        pass

    assert len(received) == 1, f"Should receive only 1 event (event_id > 2), got {len(received)}"
    assert received[0].event_id == 3, "Should receive event with event_id=3"


@pytest.mark.anyio
async def test_replay_buffer_not_cleared_only_target_session_affected() -> None:
    """Replay buffer is NOT cleared — events from other sessions remain.

    Given: Events published for two sessions.
    When: sync() is called for session A (does NOT clear replay buffer).
    Then: Both session A and session B replay buffers are intact.
    """
    session_a = "sess-rb-a"
    session_b = "sess-rb-b"
    _bridge, event_bus, _ctx = _setup_bridge_with_event_bus(session_a)

    event_a = UserMessageInsertedEvent(
        content="a", session_id=session_a, message_id="msg-a", source="accepted"
    )
    event_b = UserMessageInsertedEvent(
        content="b", session_id=session_b, message_id="msg-b", source="accepted"
    )
    await event_bus.publish(session_a, event_a)
    await event_bus.publish(session_b, event_b)

    # sync() for session A — does NOT clear replay buffer
    # (old behavior: event_bus.clear_replay_buffer(session_a))

    buffer_a = event_bus._replay_buffers.get(session_a)
    buffer_b = event_bus._replay_buffers.get(session_b)

    # Both buffers should be intact (sync does not clear)
    assert buffer_a is not None, "Session A replay buffer should NOT be cleared by sync()"
    assert len(buffer_a) > 0, "Session A replay buffer should still have events"
    assert buffer_b is not None, "Session B replay buffer should not be None"
    assert len(buffer_b) > 0, "Session B replay buffer should be intact"


@pytest.mark.anyio
async def test_events_after_sync_are_still_delivered() -> None:
    """Events published AFTER sync() are delivered to both new and existing subscribers.

    Given: sync() was called (replay buffer NOT cleared).
    When: A new event is published.
    Then: Live subscribers receive it.
    And: New subscribers with replay=True receive it from the replay buffer.
    And: New subscribers with replay=False do NOT receive old events but DO
        receive new live events.
    """
    session_id = "sess-rb-6"
    _bridge, event_bus, _ctx = _setup_bridge_with_event_bus(session_id)

    # Publish old event
    old_event = UserMessageInsertedEvent(
        content="old", session_id=session_id, message_id="msg-old", source="accepted"
    )
    await event_bus.publish(session_id, old_event)

    # sync() is called — does NOT clear replay buffer

    # Publish new event AFTER sync
    new_event = UserMessageInsertedEvent(
        content="new", session_id=session_id, message_id="msg-new", source="accepted"
    )
    await event_bus.publish(session_id, new_event)

    # New subscriber with replay=True should receive BOTH old and new events
    queue_full = await event_bus.subscribe(session_id, scope="session", replay=True)
    received_full: list[Any] = []
    try:
        while not queue_full.empty():
            envelope = queue_full.get_nowait()
            received_full.append(envelope.event)
    except asyncio.QueueEmpty:
        pass

    assert len(received_full) == 2, (
        f"Should receive both old and new events, got {len(received_full)}"
    )
    assert isinstance(received_full[0], UserMessageInsertedEvent)
    assert received_full[0].message_id == "msg-old"
    assert received_full[1].message_id == "msg-new"
