"""Unit tests for EventBus mechanism coverage (Group 5).

Tests overflow policies, cross-session isolation, event ordering, observer
defect isolation, scoped subscriptions, backpressure, and StateUpdate
filtering from ProtocolChannel.

These tests complement tests/orchestrator/test_event_bus.py (which covers
coalescing with 109 tests) by exercising mechanisms that have zero coverage:
overflow policies, cross-session isolation, scoped subscriptions,
backpressure, and StateUpdate filtering.
"""

from __future__ import annotations

from asyncio import Queue as AsyncQueue, QueueEmpty, QueueShutDown
from typing import cast
from unittest.mock import MagicMock

import pytest

from wolfharness.agents.events import RunStartedEvent, StateUpdate
from wolfharness.lifecycle.comm_channel import ProtocolChannel
from wolfharness.lifecycle.journal import MemoryJournal
from wolfharness.lifecycle.types import RunState
from wolfharness.orchestrator.event_bus import EventBus, EventEnvelope, OverflowPolicy


pytestmark = [pytest.mark.unit, pytest.mark.anyio]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drain_queue(queue: AsyncQueue[EventEnvelope]) -> list[EventEnvelope]:
    """Drain all immediately-available items from a subscriber queue.

    Returns items in FIFO order. Does not block.
    """
    items: list[EventEnvelope] = []
    while True:
        try:
            items.append(queue.get_nowait())
        except (QueueEmpty, QueueShutDown):
            break
    return items


def _make_event(run_id: str) -> RunStartedEvent:
    """Create a RunStartedEvent with the given run_id for testing."""
    return RunStartedEvent(session_id="test-sess", run_id=run_id)


def _run_ids(envelopes: list[EventEnvelope]) -> list[str]:
    """Extract run_id values from a list of EventEnvelopes wrapping RunStartedEvent."""
    return [cast(RunStartedEvent, env.event).run_id for env in envelopes]


# ---------------------------------------------------------------------------
# Test 1: Overflow policy — drop_oldest
# ---------------------------------------------------------------------------


async def test_overflow_policy_drop_oldest() -> None:
    """drop_oldest evicts the oldest queued event when the queue is full.

    Given: EventBus with max_queue_size=3 and drop_oldest policy.
    When: 5 events are published without draining.
    Then: Subscriber queue holds the 3 newest events (oldest 2 dropped).
    """
    bus = EventBus(max_queue_size=3, overflow_policy="drop_oldest")
    queue = await bus.subscribe("test-sess")

    for i in range(5):
        await bus.publish("test-sess", _make_event(f"run-{i}"))

    events = _drain_queue(queue)
    assert len(events) == 3
    assert _run_ids(events) == ["run-2", "run-3", "run-4"]


# ---------------------------------------------------------------------------
# Test 2: Overflow policy — drop_newest
# ---------------------------------------------------------------------------


async def test_overflow_policy_drop_newest() -> None:
    """drop_newest discards incoming events when the queue is full.

    Given: EventBus with max_queue_size=3 and drop_newest policy.
    When: 5 events are published without draining.
    Then: Subscriber queue holds the first 3 events (newest 2 dropped).
    """
    bus = EventBus(max_queue_size=3, overflow_policy="drop_newest")
    queue = await bus.subscribe("test-sess")

    for i in range(5):
        await bus.publish("test-sess", _make_event(f"run-{i}"))

    events = _drain_queue(queue)
    assert len(events) == 3
    assert _run_ids(events) == ["run-0", "run-1", "run-2"]


# ---------------------------------------------------------------------------
# Test 3: Overflow policy — block raises ValueError
# ---------------------------------------------------------------------------


async def test_overflow_policy_block_raises() -> None:
    """overflow_policy='block' is rejected at construction time.

    Given: An attempt to create EventBus with overflow_policy="block".
    When: The constructor is called.
    Then: ValueError is raised explaining block would deadlock the run loop.
    """
    with pytest.raises(ValueError, match="block"):
        EventBus(overflow_policy=cast(OverflowPolicy, "block"))


# ---------------------------------------------------------------------------
# Test 4: Cross-session isolation
# ---------------------------------------------------------------------------


async def test_cross_session_isolation() -> None:
    """Events published to session A do not reach subscribers of session B.

    Given: EventBus with subscribers on sessions "A" and "B".
    When: 3 events are published to session "A".
    Then: Subscriber of session "B" receives zero events from A.
    """
    bus = EventBus(max_queue_size=10)
    queue_a = await bus.subscribe("A")
    queue_b = await bus.subscribe("B")

    for i in range(3):
        await bus.publish("A", _make_event(f"run-{i}"))

    events_a = _drain_queue(queue_a)
    events_b = _drain_queue(queue_b)

    assert len(events_a) == 3
    assert len(events_b) == 0


# ---------------------------------------------------------------------------
# Test 5: Event ordering by publish order
# ---------------------------------------------------------------------------


async def test_event_ordering_by_publish_order() -> None:
    """Events arrive at the subscriber in the same order they were published.

    Given: A single subscriber on session "test-sess".
    When: 10 events are published sequentially.
    Then: Draining the subscriber queue yields events in publish order.
    """
    bus = EventBus(max_queue_size=100)
    queue = await bus.subscribe("test-sess")

    for i in range(10):
        await bus.publish("test-sess", _make_event(f"run-{i:02d}"))

    events = _drain_queue(queue)
    assert len(events) == 10
    assert _run_ids(events) == [f"run-{i:02d}" for i in range(10)]


# ---------------------------------------------------------------------------
# Test 6: Observer defect isolation
# ---------------------------------------------------------------------------


async def test_observer_defect_isolation() -> None:
    """A raising consumer does not prevent delivery to other subscribers.

    Given: Two subscribers on the same session.
    When: 3 events are published, and the first consumer raises while
        draining its queue.
    Then: The second consumer's queue still contains all 3 events.
    """
    bus = EventBus(max_queue_size=10)
    queue_a = await bus.subscribe("test-sess")
    queue_b = await bus.subscribe("test-sess")

    for i in range(3):
        await bus.publish("test-sess", _make_event(f"run-{i}"))

    # Consumer A raises on the first event — simulate a buggy consumer.
    try:
        queue_a.get_nowait()
        raise RuntimeError("consumer A defect")  # noqa: TRY301
    except RuntimeError:
        pass  # Consumer A crashed; its queue may still have remaining events.

    # Consumer B should still have all 3 events regardless of A's failure.
    events_b = _drain_queue(queue_b)
    assert len(events_b) == 3
    assert _run_ids(events_b) == ["run-0", "run-1", "run-2"]


# ---------------------------------------------------------------------------
# Test 7: Scoped subscription — descendants
# ---------------------------------------------------------------------------


async def test_scoped_subscription_descendants() -> None:
    """Descendants scope receives events from child sessions.

    Given: EventBus with a parent-child session hierarchy established
        via the internal session tree.
    When: An event is published to the child session.
    Then: A subscriber on the parent with scope="descendants" receives it.
    """
    bus = EventBus(max_queue_size=10)
    # Establish parent → child relationship in the internal session tree.
    # This is the fallback path used when no SessionController is configured.
    bus._session_tree = {"parent": ["child"]}

    parent_queue = await bus.subscribe("parent", scope="descendants")

    await bus.publish("child", _make_event("run-child"))

    events = _drain_queue(parent_queue)
    assert len(events) == 1
    assert cast(RunStartedEvent, events[0].event).run_id == "run-child"


# ---------------------------------------------------------------------------
# Test 8: Backpressure through consumer loop
# ---------------------------------------------------------------------------


async def test_backpressure_through_consumer_loop() -> None:
    """EventBus does not block under backpressure; overflow policy applies.

    Given: EventBus with max_queue_size=2 and drop_oldest policy.
    When: 5 events are published faster than the consumer drains (no draining
        during publishing).
    Then: All publish() calls complete without blocking, and the subscriber
        queue holds the 2 newest events.
    """
    bus = EventBus(max_queue_size=2, overflow_policy="drop_oldest")
    queue = await bus.subscribe("test-sess")

    # Publish all 5 events without draining — simulates a slow consumer.
    for i in range(5):
        await bus.publish("test-sess", _make_event(f"run-{i}"))

    # If publish() blocked, we would never reach this assertion.
    events = _drain_queue(queue)
    assert len(events) == 2
    assert _run_ids(events) == ["run-3", "run-4"]


# ---------------------------------------------------------------------------
# Test 9: StateUpdate filtered from EventBus
# ---------------------------------------------------------------------------


async def test_stateupdate_filtered_from_eventbus() -> None:
    """ProtocolChannel journals StateUpdate but does not publish it to EventBus.

    Given: A ProtocolChannel with a MemoryJournal and EventBus, and a
        subscriber on the EventBus.
    When: channel.publish(StateUpdate(...)) is called.
    Then:
        (a) The StateUpdate IS recorded in the journal (upserted).
        (b) The EventBus subscriber does NOT receive the StateUpdate.
    """
    journal = MemoryJournal()
    bus = EventBus(max_queue_size=10)
    channel = ProtocolChannel(journal=journal, event_bus=bus, session_id="test-sess")

    # Subscribe BEFORE publishing to capture live events.
    queue = await bus.subscribe("test-sess")

    state_update = StateUpdate(session_id="test-sess", state=RunState.IDLE)
    await channel.publish(state_update)

    # (a) Verify the StateUpdate was journaled.
    journal_events: list[object] = []
    async for evt in journal.replay():
        journal_events.append(evt)  # noqa: PERF401

    state_updates_in_journal: list[StateUpdate] = [
        evt for evt in journal_events if isinstance(evt, StateUpdate)
    ]
    assert len(state_updates_in_journal) == 1
    assert state_updates_in_journal[0].session_id == "test-sess"
    assert state_updates_in_journal[0].state == RunState.IDLE

    # (b) Verify the EventBus subscriber did NOT receive the StateUpdate.
    bus_events = _drain_queue(queue)
    assert len(bus_events) == 0


# ---------------------------------------------------------------------------
# Test 10: ProtocolChannel publishes non-StateUpdate to EventBus
# ---------------------------------------------------------------------------


async def test_protocol_channel_publishes_non_stateupdate_to_eventbus() -> None:
    """ProtocolChannel publishes non-StateUpdate events to EventBus and journals them.

    Given: A ProtocolChannel with a MemoryJournal and EventBus, and a
        subscriber on the EventBus.
    When: channel.publish(RunStartedEvent(...)) is called.
    Then:
        (a) The event IS received by the EventBus subscriber.
        (b) The event IS recorded in the journal (appended).
    """
    journal = MemoryJournal()
    bus = EventBus(max_queue_size=10)
    channel = ProtocolChannel(journal=journal, event_bus=bus, session_id="test-sess")

    # Subscribe BEFORE publishing to capture live events.
    queue = await bus.subscribe("test-sess")

    event = RunStartedEvent(session_id="test-sess", run_id="run-positive")
    await channel.publish(event)

    # (a) Verify the EventBus subscriber received the event.
    bus_events = _drain_queue(queue)
    assert len(bus_events) == 1
    assert cast(RunStartedEvent, bus_events[0].event).run_id == "run-positive"

    # (b) Verify the event was journaled.
    journal_events: list[object] = []
    async for evt in journal.replay():
        journal_events.append(evt)  # noqa: PERF401

    run_started_in_journal: list[RunStartedEvent] = [
        evt for evt in journal_events if isinstance(evt, RunStartedEvent)
    ]
    assert len(run_started_in_journal) == 1
    assert run_started_in_journal[0].run_id == "run-positive"


# ---------------------------------------------------------------------------
# Test 11: Scoped subscription — subtree
# ---------------------------------------------------------------------------


async def test_scoped_subscription_subtree() -> None:
    """Subtree scope receives events from self, parent, and siblings.

    Given: EventBus with a session tree: root → [child_a, child_b].
    When: A subscriber on "child_a" with scope="subtree" is created, and
        events are published to "child_a", "root", and "child_b".
    Then: All 3 events are received (self, parent, sibling).
    """
    bus = EventBus(max_queue_size=10)
    bus._session_tree = {"root": ["child_a", "child_b"]}

    queue = await bus.subscribe("child_a", scope="subtree")

    await bus.publish("child_a", _make_event("run-self"))
    await bus.publish("root", _make_event("run-parent"))
    await bus.publish("child_b", _make_event("run-sibling"))

    events = _drain_queue(queue)
    assert len(events) == 3
    assert _run_ids(events) == ["run-self", "run-parent", "run-sibling"]


# ---------------------------------------------------------------------------
# Test 12: Descendants scope with SessionController
# ---------------------------------------------------------------------------


async def test_descendants_scope_with_session_controller() -> None:
    """Descendants scope uses SessionController.get_children for hierarchy queries.

    Given: EventBus with a mock SessionController where
        get_children("parent") -> ["child"] and
        get_parent("child") -> SessionState(session_id="parent").
    When: A subscriber on "parent" with scope="descendants" is created,
        and an event is published to "child".
    Then: The event is received by the subscriber, and
        controller.get_children was called.
    """
    controller = MagicMock()
    parent_state = MagicMock()
    parent_state.session_id = "parent"
    controller.get_parent.return_value = parent_state
    controller.get_children.return_value = ["child"]

    bus = EventBus(max_queue_size=10, session_controller=controller)

    parent_queue = await bus.subscribe("parent", scope="descendants")

    await bus.publish("child", _make_event("run-child"))

    events = _drain_queue(parent_queue)
    assert len(events) == 1
    assert cast(RunStartedEvent, events[0].event).run_id == "run-child"

    # Verify the controller was consulted for hierarchy.
    controller.get_children.assert_called()


# ---------------------------------------------------------------------------
# Test 13: Close session removes replay buffer (no orphaned replay)
# ---------------------------------------------------------------------------


async def test_close_session_no_orphaned_replay_buffer() -> None:
    """close_session removes the replay buffer so new subscribers see no stale events.

    Given: A subscriber on "sess-1" with a published and drained event.
    When: The session is closed via close_session().
    Then:
        (a) "sess-1" is no longer in bus._replay_buffers.
        (b) A new subscriber to "sess-1" receives zero historical events.
    """
    bus = EventBus(max_queue_size=10)
    queue = await bus.subscribe("sess-1")

    await bus.publish("sess-1", _make_event("run-1"))
    _drain_queue(queue)

    await bus.close_session("sess-1")

    # (a) Replay buffer for "sess-1" should be gone.
    assert "sess-1" not in bus._replay_buffers

    # (b) New subscriber should receive zero historical events.
    new_queue = await bus.subscribe("sess-1")
    events = _drain_queue(new_queue)
    assert len(events) == 0


# ---------------------------------------------------------------------------
# Test 14: Drop subscriber overflow policy
# ---------------------------------------------------------------------------


async def test_drop_subscriber_verifies_queue_shutdown() -> None:
    """drop_subscriber policy removes the subscriber and shuts down its queue on overflow.

    Given: EventBus with max_queue_size=3 and overflow_policy="drop_subscriber".
    When: 4 events are published without draining.
    Then:
        (a) The subscriber is removed from get_subscriber_counts().
        (b) The dead queue raises QueueShutDown on get_nowait() after draining
            remaining items.
    """
    bus = EventBus(max_queue_size=3, overflow_policy="drop_subscriber")
    queue = await bus.subscribe("sess-1")

    for i in range(4):
        await bus.publish("sess-1", _make_event(f"run-{i}"))

    # (a) Subscriber should be removed from subscriber counts.
    counts = await bus.get_subscriber_counts()
    assert "sess-1" not in counts

    # (b) The queue was shut down. Drain remaining items (3 from before
    # overflow), then verify QueueShutDown is raised on the empty shut-down queue.
    remaining = _drain_queue(queue)
    assert len(remaining) == 3

    with pytest.raises(QueueShutDown):
        queue.get_nowait()


# ---------------------------------------------------------------------------
# Test 15: Clear replay buffer prevents stale replay
# ---------------------------------------------------------------------------


async def test_clear_replay_buffer_prevents_stale_replay() -> None:
    """clear_replay_buffer removes historical events so new subscribers see none.

    Given: A subscriber on "sess-1" with a published and drained event.
    When: bus.clear_replay_buffer("sess-1") is called.
    Then: A new subscriber to "sess-1" receives zero historical events.
    """
    bus = EventBus(max_queue_size=10)
    queue = await bus.subscribe("sess-1")

    await bus.publish("sess-1", _make_event("run-1"))
    _drain_queue(queue)

    bus.clear_replay_buffer("sess-1")

    new_queue = await bus.subscribe("sess-1")
    events = _drain_queue(new_queue)
    assert len(events) == 0


# ---------------------------------------------------------------------------
# Test 16: Scope "all" replays from multiple sessions
# ---------------------------------------------------------------------------


async def test_scope_all_replay_from_multiple_sessions() -> None:
    """Scope "all" replays historical events from all session replay buffers.

    Given: Events published to sessions "A" and "B" (populating both
        replay buffers).
    When: A subscriber with scope="all" is created.
    Then: The subscriber receives replayed events from BOTH sessions.
    """
    bus = EventBus(max_queue_size=10)

    await bus.publish("A", _make_event("run-a"))
    await bus.publish("B", _make_event("run-b"))

    queue = await bus.subscribe("global", scope="all")

    events = _drain_queue(queue)
    assert len(events) == 2

    run_ids = _run_ids(events)
    assert "run-a" in run_ids
    assert "run-b" in run_ids
