"""Unit tests for EventBus conditional replay feature (sse-conditional-replay change).

Tests event_id assignment at publish time, conditional replay via
subscribe(replay, last_event_id), gap detection on reconnect, non-destructive
replay, scope="all" filtering with last_event_id, and _rebind event_id preservation.

All tests are self-contained — they create EventBus instances directly.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic_ai import TextPartDelta
import pytest

from wolfharness.agents.events import (
    PartDeltaEvent,
    RunStartedEvent,
)
from wolfharness.orchestrator.core import (
    EventBus,
    drain_and_merge,
)


pytestmark = [pytest.mark.unit, pytest.mark.anyio]


async def _drain_stream(queue: asyncio.Queue[Any]) -> list[Any]:
    """Drain all currently-available items from an async queue without blocking."""
    items: list[Any] = []
    while True:
        try:
            items.append(queue.get_nowait())
        except (asyncio.QueueEmpty, asyncio.QueueShutDown):
            break
    return items


# ---------------------------------------------------------------------------
# 5.1: event_id assigned at publish time
# ---------------------------------------------------------------------------


async def test_event_id_assigned_at_publish_time() -> None:
    """EventBus.publish() assigns monotonically increasing event_id to each envelope.

    Given: A fresh EventBus.
    When: Three events are published in sequence.
    Then: Each envelope in the replay buffer has a unique, monotonically
        increasing event_id.
    """
    bus = EventBus(max_queue_size=10)
    for i in range(3):
        await bus.publish("s1", RunStartedEvent(session_id="s1", run_id=f"run-{i}"))

    buffer = bus._replay_buffers["s1"]
    assert len(buffer) == 3
    event_ids = [env.event_id for env in buffer]
    assert event_ids == [1, 2, 3], f"Expected [1, 2, 3], got {event_ids}"
    # Verify strict monotonic increase
    for i in range(1, len(event_ids)):
        assert event_ids[i] > event_ids[i - 1]


# ---------------------------------------------------------------------------
# 5.2: replayed events preserve original event_id
# ---------------------------------------------------------------------------


async def test_replayed_events_preserve_original_event_id() -> None:
    """Replayed events from the buffer preserve their original event_id.

    Given: Events published with event_ids [1, 2, 3].
    When: A new subscriber triggers replay from the buffer.
    Then: The replayed envelopes have the same event_ids as the originals.
    And: The event_ids are NOT reassigned during replay.
    """
    bus = EventBus(max_queue_size=10)
    for i in range(3):
        await bus.publish("s1", RunStartedEvent(session_id="s1", run_id=f"run-{i}"))

    # Subscribe after publishing — triggers replay
    queue = await bus.subscribe("s1")
    received = await _drain_stream(queue)

    assert len(received) == 3
    replayed_ids = [env.event_id for env in received]
    assert replayed_ids == [1, 2, 3], (
        f"Replayed event_ids should match originals, got {replayed_ids}"
    )


# ---------------------------------------------------------------------------
# 5.3: subscribe(replay=False) delivers zero historical events
# ---------------------------------------------------------------------------


async def test_subscribe_replay_false_delivers_zero_historical() -> None:
    """subscribe(replay=False) skips the replay buffer entirely.

    Given: Events published to the replay buffer.
    When: A subscriber subscribes with replay=False.
    Then: No historical events are delivered to the subscriber queue.
    And: Events published AFTER the subscription are delivered normally.
    """
    bus = EventBus(max_queue_size=10)
    for i in range(5):
        await bus.publish("s1", RunStartedEvent(session_id="s1", run_id=f"hist-{i}"))

    # Subscribe with replay=False — should get zero historical events
    queue = await bus.subscribe("s1", replay=False)
    received = await _drain_stream(queue)
    assert len(received) == 0, (
        f"Expected 0 historical events with replay=False, got {len(received)}"
    )

    # Events published AFTER subscription should be delivered
    await bus.publish("s1", RunStartedEvent(session_id="s1", run_id="live-0"))
    received_live = await _drain_stream(queue)
    assert len(received_live) == 1
    assert isinstance(received_live[0].event, RunStartedEvent)
    assert received_live[0].event.run_id == "live-0"


# ---------------------------------------------------------------------------
# 5.4: subscribe(replay=True, last_event_id=N) delivers only events with event_id > N
# ---------------------------------------------------------------------------


async def test_subscribe_with_last_event_id_filters_replay() -> None:
    """subscribe(replay=True, last_event_id=N) delivers only events with event_id > N.

    Given: Replay buffer contains events with event_ids [1, 2, 3, 4, 5].
    When: A subscriber subscribes with replay=True, last_event_id=3.
    Then: Only events with event_id > 3 (i.e., 4, 5) are delivered from the buffer.
    """
    bus = EventBus(max_queue_size=10)
    for i in range(5):
        await bus.publish("s1", RunStartedEvent(session_id="s1", run_id=f"ev-{i}"))

    # Verify event_ids in buffer
    buffer = bus._replay_buffers["s1"]
    assert [env.event_id for env in buffer] == [1, 2, 3, 4, 5]

    # Subscribe with last_event_id=3 — should only get events 4 and 5
    queue = await bus.subscribe("s1", replay=True, last_event_id=3)
    received = await _drain_stream(queue)

    assert len(received) == 2
    assert received[0].event_id == 4
    assert received[1].event_id == 5
    assert received[0].event.run_id == "ev-3"
    assert received[1].event.run_id == "ev-4"


# ---------------------------------------------------------------------------
# 5.5: subscribe(replay=True, last_event_id=None) delivers all buffered events
# ---------------------------------------------------------------------------


async def test_subscribe_with_none_last_event_id_replays_all() -> None:
    """subscribe(replay=True, last_event_id=None) delivers all buffered events.

    This is the backward-compatible default behavior — all events in the
    replay buffer are delivered when no last_event_id is specified.

    Given: Replay buffer contains 5 events.
    When: A subscriber subscribes with replay=True, last_event_id=None.
    Then: All 5 events are delivered from the replay buffer.
    """
    bus = EventBus(max_queue_size=10)
    for i in range(5):
        await bus.publish("s1", RunStartedEvent(session_id="s1", run_id=f"ev-{i}"))

    queue = await bus.subscribe("s1", replay=True, last_event_id=None)
    received = await _drain_stream(queue)

    assert len(received) == 5
    event_ids = [env.event_id for env in received]
    assert event_ids == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# 5.6: gap detection falls back to full replay
# ---------------------------------------------------------------------------


async def test_gap_detection_falls_back_to_full_replay() -> None:
    """Gap detection replays all buffered events when last_event_id < buffer[0].event_id.

    Given: Replay buffer (size=3) contains events with event_ids [3, 4, 5]
        (events 1-2 were evicted by the bounded buffer).
    When: A subscriber subscribes with replay=True, last_event_id=1.
    Then: All events [3, 4, 5] are delivered (gap detected, full replay fallback)
        rather than filtering by event_id > 1 (which would also give all 3,
        but the gap detection logic ensures no silent data loss).
    """
    bus = EventBus(max_queue_size=10, replay_buffer_size=3)

    # Publish 5 events; buffer evicts oldest, keeping event_ids [3, 4, 5]
    for i in range(5):
        await bus.publish("s1", RunStartedEvent(session_id="s1", run_id=f"ev-{i}"))

    buffer = bus._replay_buffers["s1"]
    assert len(buffer) == 3
    buffer_ids = [env.event_id for env in buffer]
    assert buffer_ids == [3, 4, 5], f"Expected [3, 4, 5], got {buffer_ids}"

    # Subscribe with last_event_id=1 — there's a gap (1 < 3, the oldest in buffer)
    # Gap detection should fall back to full replay, delivering all 3 events
    queue = await bus.subscribe("s1", replay=True, last_event_id=1)
    received = await _drain_stream(queue)

    assert len(received) == 3, (
        f"Gap detection should deliver all 3 buffered events, got {len(received)}"
    )
    received_ids = [env.event_id for env in received]
    assert received_ids == [3, 4, 5]


# ---------------------------------------------------------------------------
# 5.7: conditional replay is non-destructive
# ---------------------------------------------------------------------------


async def test_conditional_replay_is_non_destructive() -> None:
    """subscribe(replay=False) does not modify the replay buffer.

    Given: Events published to the replay buffer.
    When: Subscriber A subscribes with replay=False.
    Then: The replay buffer for s1 remains intact.
    And: Subscriber B (subscribed with replay=True) still receives all
        buffered events.
    """
    bus = EventBus(max_queue_size=10)
    for i in range(5):
        await bus.publish("s1", RunStartedEvent(session_id="s1", run_id=f"ev-{i}"))

    # Subscriber A: replay=False — no historical events
    queue_a = await bus.subscribe("s1", replay=False)
    received_a = await _drain_stream(queue_a)
    assert len(received_a) == 0

    # Buffer should still be intact
    buffer = bus._replay_buffers["s1"]
    assert len(buffer) == 5, "Replay buffer should be intact after replay=False"

    # Subscriber B: replay=True (default) — should get all 5 events
    queue_b = await bus.subscribe("s1", replay=True)
    received_b = await _drain_stream(queue_b)
    assert len(received_b) == 5, f"Subscriber B should get all 5 events, got {len(received_b)}"
    received_ids = [env.event_id for env in received_b]
    assert received_ids == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# 5.8: scope="all" with last_event_id filters across multiple session buffers
# ---------------------------------------------------------------------------


async def test_scope_all_with_last_event_id_filters_across_sessions() -> None:
    """scope="all" with last_event_id filters events across multiple session buffers.

    Given: Session s1 buffer has events with event_ids [1, 2] and
        session s2 buffer has events with event_ids [3, 4].
    When: A subscriber subscribes with scope="all", replay=True, last_event_id=2.
    Then: Only events with event_id > 2 (i.e., 3, 4 from s2) are delivered.
    """
    bus = EventBus(max_queue_size=10)

    # Publish to s1 (event_ids 1, 2)
    await bus.publish("s1", RunStartedEvent(session_id="s1", run_id="s1-ev0"))
    await bus.publish("s1", RunStartedEvent(session_id="s1", run_id="s1-ev1"))

    # Publish to s2 (event_ids 3, 4)
    await bus.publish("s2", RunStartedEvent(session_id="s2", run_id="s2-ev0"))
    await bus.publish("s2", RunStartedEvent(session_id="s2", run_id="s2-ev1"))

    # Subscribe with scope="all" and last_event_id=2
    queue = await bus.subscribe("__global_sse__", scope="all", replay=True, last_event_id=2)
    received = await _drain_stream(queue)

    # Should only get events with event_id > 2 (i.e., 3, 4 from s2)
    assert len(received) == 2, f"Expected 2 events with event_id > 2, got {len(received)}"
    received_ids = [env.event_id for env in received]
    assert received_ids == [3, 4]
    assert all(env.source_session_id == "s2" for env in received)


# ---------------------------------------------------------------------------
# 5.9: merged envelopes via _rebind() preserve original event_id
# ---------------------------------------------------------------------------


async def test_rebind_preserves_original_event_id() -> None:
    """_rebind() preserves event_id from the template envelope in merged envelopes.

    Given: Multiple PartDeltaEvent envelopes published with sequential event_ids.
    When: The envelopes are coalesced via drain_and_merge() (which uses _rebind).
    Then: The merged envelope preserves the event_id of the first envelope
        in the merge group.
    """
    bus = EventBus(max_queue_size=100)

    # Publish text deltas that will be coalesced by drain_and_merge
    await bus.publish("s1", PartDeltaEvent.text(0, "hello "))
    await bus.publish("s1", PartDeltaEvent.text(0, "world"))

    # Get the event_ids from the buffer
    buffer = bus._replay_buffers["s1"]
    first_event_id = buffer[0].event_id
    second_event_id = buffer[1].event_id
    assert first_event_id != second_event_id

    # Subscribe and drain with coalescing
    queue = await bus.subscribe("s1")
    # Shutdown the queue so drain_and_merge terminates after draining
    queue.shutdown()
    results = [env async for env in drain_and_merge(queue)]

    # Should be merged into a single envelope
    assert len(results) == 1
    merged = results[0]
    assert isinstance(merged.event, PartDeltaEvent)
    assert isinstance(merged.event.delta, TextPartDelta)
    assert merged.event.delta.content_delta == "hello world"
    # The merged envelope should preserve the FIRST event's event_id
    assert merged.event_id == first_event_id, (
        f"Merged envelope should preserve first event_id={first_event_id}, got {merged.event_id}"
    )
    assert merged.event_id != second_event_id
