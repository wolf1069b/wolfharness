"""Integration test for the first-user-message duplication bug.

Reproduction scenario:
1. User sends message -> EventProcessor broadcasts MessageUpdatedEvent +
   PartUpdatedEvent to SSE subscribers AND EventBus replay buffer
2. TUI's SSE connection is established AFTER events were published
   (race: events published during ~880ms consumer startup)
3. EventBus replay buffer re-delivers the same events to the late subscriber
4. TUI also calls sync() -> loads the same message + parts from DB
5. TUI has parts from BOTH replay buffer (SSE) and sync() (DB) -> DUPLICATE

Fix (new design): sync() does NOT clear the replay buffer. Instead, the SSE
endpoint uses subscribe(replay=False) on first connection (no Last-Event-ID
header) and subscribe(replay=True, last_event_id=N) on reconnection. This
prevents duplication without destroying replay history for other clients.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from wolfharness.orchestrator.core import EventBus, EventEnvelope
from wolfharness_server.opencode_server.models import (
    MessageUpdatedEvent,
    MessageWithParts,
    PartUpdatedEvent,
    TextPart,
    UserMessage,
)
from wolfharness_server.opencode_server.models.common import TimeCreated


pytestmark = pytest.mark.integration


def _make_user_message(session_id: str = "sess-dup") -> MessageWithParts:
    """Create a user message with a text part (simulating DB content)."""
    text_part = TextPart(
        id="part-text-1",
        message_id="msg-user-1",
        session_id=session_id,
        text="hello",
        synthetic=False,
    )
    return MessageWithParts(
        info=UserMessage(
            id="msg-user-1",
            session_id=session_id,
            time=TimeCreated(created=1784726591675),
        ),
        parts=[text_part],
    )


def _wrap_sse_event(data: Any, session_id: str) -> EventEnvelope:
    """Wrap an SSE event as the event bridge does (CustomEvent wrapper)."""
    from wolfharness.agents.events.events import CustomEvent

    return EventEnvelope(
        source_session_id=session_id,
        event=CustomEvent(source="opencode_event_bridge", event_data=data),
    )


async def _drain_queue(queue: asyncio.Queue[Any]) -> list[Any]:
    """Drain all currently-available items from an async queue."""
    items: list[Any] = []
    while not queue.empty():
        try:
            items.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return items


@pytest.mark.anyio
async def test_first_message_duplicated_with_full_replay() -> None:
    """Reproduce: TUI sees duplicate parts when replay buffer + sync() both deliver.

    This test demonstrates the duplication problem when the SSE endpoint
    replays ALL events from the buffer on first connection (no conditional
    replay). The old design had no replay=False option, so all historical
    events were delivered, causing duplication with sync().

    1. Events published to EventBus (user sends message before TUI SSE connects)
    2. Late SSE subscriber receives ALL events from replay buffer (old behavior)
    3. sync() loads same parts from DB
    4. Parts exist in BOTH sources -> duplication
    """
    session_id = "sess-dup-no-clear"
    event_bus = EventBus()

    user_msg = _make_user_message(session_id)

    # Step 1: User sends message -> EventProcessor broadcasts events
    await event_bus.publish(
        session_id,
        _wrap_sse_event(MessageUpdatedEvent.create(user_msg.info), session_id).event,
    )
    await event_bus.publish(
        session_id,
        _wrap_sse_event(PartUpdatedEvent.create(user_msg.parts[0]), session_id).event,
    )

    # Step 2: TUI's SSE connects AFTER events were published
    # OLD behavior: subscribe() replays ALL events from buffer (no replay=False)
    sse_queue = await event_bus.subscribe(session_id)
    replayed_events = await _drain_queue(sse_queue)

    # Extract PartUpdatedEvent from replayed events
    replayed_parts = [
        e.event.event_data
        for e in replayed_events
        if isinstance(e.event.event_data, PartUpdatedEvent)
    ]

    # Step 3: TUI also calls sync() -> loads from DB -> gets same parts
    db_parts = user_msg.parts  # Simulate sync() loading from DB

    # Step 4: DUPLICATION -- parts exist in BOTH replay buffer and DB
    assert len(replayed_parts) > 0, "Replay buffer should have PartUpdatedEvent"
    assert len(db_parts) > 0, "DB should have parts"

    # The part IDs match -> TUI cannot deduplicate
    replayed_part_ids = {p.properties.part.id for p in replayed_parts}
    db_part_ids = {p.id for p in db_parts}
    assert replayed_part_ids & db_part_ids, (
        "Part IDs overlap between replay buffer and DB -> TUI renders duplicate"
    )


@pytest.mark.anyio
async def test_conditional_replay_prevents_duplication() -> None:
    """Fix: SSE uses replay=False on first connection -> no duplicate parts.

    New design: The SSE endpoint calls subscribe(replay=False) when there
    is no Last-Event-ID header (first connection). This means the subscriber
    gets zero historical events from the replay buffer, preventing
    duplication with sync().

    1. Events published to EventBus (user sends message before TUI connects)
    2. sync() called -> loads from DB (replay buffer is NOT cleared)
    3. SSE subscribes with replay=False (first connection, no Last-Event-ID)
    4. SSE subscriber does NOT receive old events from replay buffer
    5. Parts only come from DB (via sync()) -> no duplication
    """
    session_id = "sess-dup-clear"
    event_bus = EventBus()

    user_msg = _make_user_message(session_id)

    # Step 1: Events published before TUI connects
    await event_bus.publish(
        session_id,
        _wrap_sse_event(MessageUpdatedEvent.create(user_msg.info), session_id).event,
    )
    await event_bus.publish(
        session_id,
        _wrap_sse_event(PartUpdatedEvent.create(user_msg.parts[0]), session_id).event,
    )

    # Step 2: TUI calls sync() -> loads from DB (replay buffer NOT cleared)
    db_parts = user_msg.parts  # sync() loads from DB

    # Step 3: SSE subscribes with replay=False (first connection, no Last-Event-ID)
    sse_queue = await event_bus.subscribe(session_id, replay=False)
    replayed_events = await _drain_queue(sse_queue)

    # Step 4: No old events from replay buffer
    replayed_parts = [
        e.event.event_data
        for e in replayed_events
        if isinstance(e.event.event_data, PartUpdatedEvent)
    ]
    assert len(replayed_parts) == 0, (
        "replay=False should deliver zero historical events -> no duplicate parts via SSE"
    )

    # Step 5: Parts only from DB -> no duplication
    assert len(db_parts) > 0, "DB should have parts"
    # Only one source of parts -> no duplication


@pytest.mark.anyio
async def test_live_events_still_delivered_with_replay_false() -> None:
    """Regression: replay=False must not block live events.

    After subscribing with replay=False, NEW events published
    afterwards must still be delivered to SSE subscribers.
    """
    session_id = "sess-dup-live"
    event_bus = EventBus()

    user_msg = _make_user_message(session_id)

    # Subscribe to SSE with replay=False (first connection)
    sse_queue = await event_bus.subscribe(session_id, replay=False)

    # No historical events should be in the queue
    initial_events = await _drain_queue(sse_queue)
    assert len(initial_events) == 0, "replay=False should deliver zero historical events"

    # Publish a NEW event after subscription (e.g., second user message)
    await event_bus.publish(
        session_id,
        _wrap_sse_event(PartUpdatedEvent.create(user_msg.parts[0]), session_id).event,
    )

    # Live event should be delivered
    try:
        event = await asyncio.wait_for(sse_queue.get(), timeout=1.0)
        assert isinstance(event.event.event_data, PartUpdatedEvent), (
            "Should receive PartUpdatedEvent from live publish"
        )
    except TimeoutError:
        pytest.fail("Live event not delivered with replay=False")


@pytest.mark.anyio
async def test_reconnect_with_last_event_id_delivers_only_new() -> None:
    """SSE reconnect with Last-Event-ID only delivers events after that ID.

    New design: On reconnect, the SSE endpoint calls
    subscribe(replay=True, last_event_id=N) where N is the Last-Event-ID
    header value. Only events with event_id > N are replayed.

    1. Events published to EventBus before reconnect
    2. Client reconnects with Last-Event-ID pointing to the last event seen
    3. Only events published AFTER that ID are delivered from replay buffer
    """
    session_id = "sess-dup-reconnect"
    event_bus = EventBus()

    user_msg = _make_user_message(session_id)

    # Step 1: Events published before reconnect
    await event_bus.publish(
        session_id,
        _wrap_sse_event(MessageUpdatedEvent.create(user_msg.info), session_id).event,
    )
    first_event_id = event_bus._replay_buffers[session_id][-1].event_id

    await event_bus.publish(
        session_id,
        _wrap_sse_event(PartUpdatedEvent.create(user_msg.parts[0]), session_id).event,
    )
    second_event_id = event_bus._replay_buffers[session_id][-1].event_id

    assert first_event_id < second_event_id

    # Step 2: Client reconnects with Last-Event-ID = first_event_id
    # Should only get the second event (event_id > first_event_id)
    sse_queue = await event_bus.subscribe(session_id, replay=True, last_event_id=first_event_id)
    events = await _drain_queue(sse_queue)

    assert len(events) == 1, (
        f"Should receive only 1 event (event_id > {first_event_id}), got {len(events)}"
    )
    assert events[0].event_id == second_event_id, (
        f"Should receive event with event_id={second_event_id}, got {events[0].event_id}"
    )
