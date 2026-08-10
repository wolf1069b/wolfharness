"""Integration tests for crash recovery event delivery through EventBus.

These tests verify that events are correctly replayed through the
ProtocolChannel to EventBus subscribers during crash recovery. They
focus on the EVENT DELIVERY path (journal → channel → EventBus → subscriber),
not on journal/snapshot mechanics which are covered by
``test_crash_recovery.py``.

Test flow:
    1. Create a MemoryJournal and MemorySnapshotStore.
    2. Publish events through a ProtocolChannel (journaling them).
    3. Simulate a crash by creating a new channel + subscriber.
    4. Call ``journal.resume(snapshot_store)`` to get events since snapshot.
    5. Replay events through the channel with ``set_replaying(True)``.
    6. Verify the EventBus subscriber received the replayed events.

!!! note "Snapshot seq vs journal seq"
    ``MemorySnapshotStore.save()`` uses its own internal counter, not the
    journal's seq. To control which journal events are replayed, tests set
    ``_snapshot`` directly with the desired ``last_journal_seq`` value.
    This follows the same pattern used in ``test_crash_recovery.py``.
"""

from __future__ import annotations

from typing import Any

import anyio
import pytest

from wolfharness.agents.events import (
    PartDeltaEvent,
    RunStartedEvent,
    StateUpdate,
    StreamCompleteEvent,
)
from wolfharness.lifecycle import (
    DurableJournal,
    DurableSnapshotStore,
    MemoryJournal,
    MemorySnapshotStore,
    RunState,
    ToolExecutionRecord,
)
from wolfharness.lifecycle.comm_channel import DirectChannel, ProtocolChannel
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.event_bus import EventBus


pytestmark = [pytest.mark.integration, pytest.mark.anyio]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_started(run_id: str = "run-1", session_id: str = "test") -> RunStartedEvent:
    """Create a RunStartedEvent for testing."""
    return RunStartedEvent(session_id=session_id, run_id=run_id)


def _part_delta(content: str = "chunk") -> PartDeltaEvent:
    """Create a PartDeltaEvent for testing."""
    return PartDeltaEvent.text(index=0, content=content)


def _stream_complete() -> StreamCompleteEvent[Any]:
    """Create a StreamCompleteEvent for testing."""
    return StreamCompleteEvent(message=ChatMessage(content="done", role="assistant"))


def _set_snapshot(snapshot_store: MemorySnapshotStore, last_journal_seq: int) -> None:
    """Set the snapshot directly with a specific journal seq cutoff.

    ``MemorySnapshotStore.save()`` uses its own internal counter, not the
    journal's seq. For crash recovery tests, we need to control which
    journal entries are considered "since snapshot". Setting ``_snapshot``
    directly with the desired ``last_journal_seq`` is the same pattern
    used in ``test_crash_recovery.py``.
    """
    snapshot_store._snapshot = ({"state": RunState.IDLE.value}, last_journal_seq)


async def _drain_queue(
    queue: Any,
    expected_count: int,
    timeout: float = 2.0,
) -> list[Any]:
    """Drain ``expected_count`` items from an asyncio Queue with a timeout.

    Uses anyio for cancellation semantics; never imports asyncio directly.
    """
    items: list[Any] = []
    with anyio.move_on_after(timeout):
        for _ in range(expected_count):
            items.append(await queue.get())  # noqa: PERF401
    return items


# ---------------------------------------------------------------------------
# Test 1: Replayed events reach subscriber
# ---------------------------------------------------------------------------


async def test_replayed_events_reach_subscriber() -> None:
    """Wire journal.resume() → _replaying=True → channel.publish() → subscriber.

    Given:
        A MemoryJournal with pre-crash events and a MemorySnapshotStore
        with a snapshot at journal seq=0 (all events are "since snapshot").
    When:
        journal.resume() is called, then events are replayed through
        ProtocolChannel with set_replaying(True).
    Then:
        The EventBus subscriber receives all replayed events.
    """
    journal = MemoryJournal()
    snapshot_store = MemorySnapshotStore()
    bus = EventBus()

    # Simulate pre-crash: journal events, snapshot at seq=0 (all events new).
    journal.append(_run_started())
    journal.append(_part_delta("hello"))
    _set_snapshot(snapshot_store, last_journal_seq=0)

    # New channel for recovery — subscriber must exist BEFORE replay.
    channel = ProtocolChannel(journal=journal, event_bus=bus, session_id="test")
    queue = await bus.subscribe("test")

    # Crash recovery: resume and replay.
    resume_result = journal.resume(snapshot_store)
    assert resume_result is not None
    assert len(resume_result.events) == 2

    channel.set_replaying(True)
    for event in resume_result.events:
        await channel.publish(event)
    channel.set_replaying(False)

    # Subscriber should receive both replayed events.
    received = await _drain_queue(queue, expected_count=2)
    assert len(received) == 2
    assert isinstance(received[0].event, RunStartedEvent)
    assert isinstance(received[1].event, PartDeltaEvent)


# ---------------------------------------------------------------------------
# Test 2: Replayed events arrive before new events
# ---------------------------------------------------------------------------


async def test_replayed_events_before_new_events() -> None:
    """Replayed events arrive at subscriber before any new events.

    Given:
        A journal with pre-crash events and a snapshot at seq=0.
    When:
        Events are replayed (set_replaying=True), then new events are
        published normally (set_replaying=False).
    Then:
        All replayed events arrive before any new events at the subscriber.
    """
    journal = MemoryJournal()
    snapshot_store = MemorySnapshotStore()
    bus = EventBus()

    journal.append(_run_started(run_id="old-run"))
    _set_snapshot(snapshot_store, last_journal_seq=0)

    channel = ProtocolChannel(journal=journal, event_bus=bus, session_id="test")
    queue = await bus.subscribe("test")

    # Phase 1: Replay events from journal.
    resume_result = journal.resume(snapshot_store)
    assert resume_result is not None
    channel.set_replaying(True)
    for event in resume_result.events:
        await channel.publish(event)
    channel.set_replaying(False)

    # Phase 2: Publish new events after recovery.
    new_event = _part_delta("new-chunk")
    await channel.publish(new_event)

    received = await _drain_queue(queue, expected_count=2)
    assert len(received) == 2
    # Replayed event comes first.
    assert isinstance(received[0].event, RunStartedEvent)
    assert received[0].event.run_id == "old-run"
    # New event comes second.
    assert isinstance(received[1].event, PartDeltaEvent)


# ---------------------------------------------------------------------------
# Test 3: Partial replay from snapshot
# ---------------------------------------------------------------------------


async def test_partial_replay_from_snapshot() -> None:
    """Only events after the snapshot seq are replayed.

    Given:
        Events 1..N are journaled, then a snapshot is taken at seq=N.
        Events N+1..M are journaled after the snapshot.
    When:
        journal.resume(snapshot_store) is called.
    Then:
        Only events N+1..M are replayed, not 1..N.
    """
    journal = MemoryJournal()
    snapshot_store = MemorySnapshotStore()
    bus = EventBus()

    # Events before snapshot (seq 1, 2) — should NOT be replayed.
    journal.append(_run_started(run_id="pre-snapshot-1"))
    journal.append(_part_delta("pre-snapshot-2"))

    # Snapshot at journal seq=2 — events with seq > 2 will be replayed.
    _set_snapshot(snapshot_store, last_journal_seq=2)

    # Events after snapshot (seq 3, 4) — SHOULD be replayed.
    journal.append(_part_delta("post-snapshot-1"))
    journal.append(_part_delta("post-snapshot-2"))

    channel = ProtocolChannel(journal=journal, event_bus=bus, session_id="test")
    queue = await bus.subscribe("test")

    resume_result = journal.resume(snapshot_store)
    assert resume_result is not None
    # Only 2 events (post-snapshot) should be in the replay set.
    assert len(resume_result.events) == 2

    channel.set_replaying(True)
    for event in resume_result.events:
        await channel.publish(event)
    channel.set_replaying(False)

    received = await _drain_queue(queue, expected_count=2)
    assert len(received) == 2
    # Both should be PartDeltaEvent (post-snapshot events).
    assert isinstance(received[0].event, PartDeltaEvent)
    assert isinstance(received[1].event, PartDeltaEvent)
    # Verify they are the post-snapshot events, not pre-snapshot.
    assert received[0].event.delta.content_delta == "post-snapshot-1"
    assert received[1].event.delta.content_delta == "post-snapshot-2"


# ---------------------------------------------------------------------------
# Test 4: Tool execution skip on retry
# ---------------------------------------------------------------------------


async def test_tool_execution_skip_on_retry() -> None:
    """recover_strategy="retry" allows skipping already-completed tools.

    Given:
        A journal with a logged tool execution for turn_id T before crash.
    When:
        journal.get_tool_executions(turn_id) is called after recovery.
    Then:
        The completed tool record is returned, so re-execution can skip it.
    """
    journal = MemoryJournal()

    # Simulate a tool that completed before the crash.
    turn_id = "turn-abc"
    record = ToolExecutionRecord(
        turn_id=turn_id,
        tool_name="bash",
        args={"command": "echo hello"},
        result="hello\n",
        status="completed",
    )
    journal.log_tool_execution(record)

    # After crash, recovery code checks which tools already ran.
    completed_tools = journal.get_tool_executions(turn_id)
    assert len(completed_tools) == 1
    assert completed_tools[0].tool_name == "bash"
    assert completed_tools[0].status == "completed"
    assert completed_tools[0].result == "hello\n"


# ---------------------------------------------------------------------------
# Test 5: _replaying flag prevents duplicate journal entries
# ---------------------------------------------------------------------------


async def test_replaying_flag_prevents_duplicate_journal() -> None:
    """set_replaying(True) prevents journaling but still delivers to EventBus.

    Given:
        A ProtocolChannel with a journal and EventBus subscriber.
    When:
        channel.set_replaying(True) is set, then channel.publish(event).
    Then:
        The event is NOT written to the journal (no duplicate), but IS
        delivered to the EventBus subscriber.
    """
    journal = MemoryJournal()
    bus = EventBus()
    channel = ProtocolChannel(journal=journal, event_bus=bus, session_id="test")
    queue = await bus.subscribe("test")

    # Pre-populate journal with one event.
    pre_event = _run_started(run_id="pre")
    journal.append(pre_event)
    initial_journal_count = len(journal._entries)

    # Publish while replaying — should NOT journal.
    channel.set_replaying(True)
    replay_event = _part_delta("replay-chunk")
    await channel.publish(replay_event)
    channel.set_replaying(False)

    # Journal should NOT have grown.
    assert len(journal._entries) == initial_journal_count

    # EventBus subscriber SHOULD have received the event.
    received = await _drain_queue(queue, expected_count=1)
    assert len(received) == 1
    assert isinstance(received[0].event, PartDeltaEvent)


# ---------------------------------------------------------------------------
# Test 6: Durable journal + snapshot integration
# ---------------------------------------------------------------------------


@pytest.mark.slow
async def test_durable_journal_snapshot_integration(tmp_path: Any) -> None:
    """DurableJournal + DurableSnapshotStore full recovery cycle.

    Given:
        A DurableJournal and DurableSnapshotStore backed by temp files.
    When:
        Events are journaled, a snapshot is saved, more events are
        journaled, then journal.resume() is called.
    Then:
        The recovery returns events since the snapshot and they reach
        the EventBus subscriber.

    !!! note
        ``DurableSnapshotStore.save()`` returns the snapshot row ID (not the
        journal seq) as ``last_journal_seq``. ``DurableJournal.resume()``
        uses this to filter journal entries with ``seq > last_journal_seq``.
        Since both use auto-increment in separate tables, we verify the
        recovery flow works end-to-end without asserting exact event counts.
    """
    journal_db = str(tmp_path / "journal.db")
    snapshot_db = str(tmp_path / "snapshot.db")

    journal = DurableJournal(f"sqlite:///{journal_db}", session_id="test")
    snapshot_store = DurableSnapshotStore(snapshot_db, session_id="test")

    try:
        # Pre-snapshot event.
        journal.append(_run_started(run_id="pre"))
        snapshot_store.save({"state": RunState.IDLE.value})

        # Post-snapshot events.
        journal.append(_part_delta("post-1"))
        journal.append(_part_delta("post-2"))

        # Recovery.
        resume_result = journal.resume(snapshot_store)
        assert resume_result is not None
        # At least the post-snapshot events should be present.
        assert len(resume_result.events) >= 1

        # Replay through a ProtocolChannel + EventBus.
        bus = EventBus()
        channel = ProtocolChannel(journal=journal, event_bus=bus, session_id="test")
        queue = await bus.subscribe("test")

        channel.set_replaying(True)
        for event in resume_result.events:
            await channel.publish(event)
        channel.set_replaying(False)

        received = await _drain_queue(queue, expected_count=len(resume_result.events))
        assert len(received) == len(resume_result.events)
    finally:
        journal.close()
        snapshot_store.close()


# ---------------------------------------------------------------------------
# Test 7: Replayed events maintain publish order
# ---------------------------------------------------------------------------


async def test_replayed_events_maintain_publish_order() -> None:
    """Replayed events arrive at subscriber in original publish order.

    Given:
        Events published in a specific order: RunStarted, PartDelta,
        PartDelta, StreamComplete.
    When:
        A crash occurs, then recovery replays events through EventBus.
    Then:
        The subscriber receives events in the same order they were
        originally published.
    """
    journal = MemoryJournal()
    snapshot_store = MemorySnapshotStore()
    bus = EventBus()

    # Publish events in a specific order.
    events_in_order: list[Any] = [
        _run_started(run_id="ordered-run"),
        _part_delta("first"),
        _part_delta("second"),
        _stream_complete(),
    ]

    for event in events_in_order:
        journal.append(event)
    _set_snapshot(snapshot_store, last_journal_seq=0)

    channel = ProtocolChannel(journal=journal, event_bus=bus, session_id="test")
    queue = await bus.subscribe("test")

    resume_result = journal.resume(snapshot_store)
    assert resume_result is not None
    assert len(resume_result.events) == 4

    channel.set_replaying(True)
    for event in resume_result.events:
        await channel.publish(event)
    channel.set_replaying(False)

    received = await _drain_queue(queue, expected_count=4)
    assert len(received) == 4

    # Verify order matches original publish order.
    assert isinstance(received[0].event, RunStartedEvent)
    assert received[0].event.run_id == "ordered-run"

    assert isinstance(received[1].event, PartDeltaEvent)
    assert received[1].event.delta.content_delta == "first"

    assert isinstance(received[2].event, PartDeltaEvent)
    assert received[2].event.delta.content_delta == "second"

    assert isinstance(received[3].event, StreamCompleteEvent)


# ---------------------------------------------------------------------------
# Test 8: ProtocolChannel replay delivers events to EventBus
# ---------------------------------------------------------------------------


async def test_protocol_channel_replay_delivers_events_to_event_bus() -> None:
    """Replayed events through ProtocolChannel reach the EventBus subscriber.

    Given:
        A MemoryJournal with pre-crash events and a snapshot at seq=0.
    When:
        journal.resume() returns events, then they are replayed through
        ProtocolChannel with set_replaying(True).
    Then:
        The EventBus subscriber receives all replayed events.
    """
    journal = MemoryJournal()
    snapshot_store = MemorySnapshotStore()
    bus = EventBus()

    journal.append(_run_started(run_id="replay-1"))
    journal.append(_part_delta("replay-a"))
    journal.append(_part_delta("replay-b"))
    _set_snapshot(snapshot_store, last_journal_seq=0)

    channel = ProtocolChannel(journal=journal, event_bus=bus, session_id="test")
    queue = await bus.subscribe("test")

    resume_result = journal.resume(snapshot_store)
    assert resume_result is not None
    assert len(resume_result.events) == 3

    channel.set_replaying(True)
    for event in resume_result.events:
        await channel.publish(event)
    channel.set_replaying(False)

    received = await _drain_queue(queue, expected_count=3)
    assert len(received) == 3
    assert isinstance(received[0].event, RunStartedEvent)
    assert isinstance(received[1].event, PartDeltaEvent)
    assert isinstance(received[2].event, PartDeltaEvent)


# ---------------------------------------------------------------------------
# Test 9: ProtocolChannel replay filters StateUpdate from EventBus
# ---------------------------------------------------------------------------


async def test_protocol_channel_replay_filters_state_update_from_event_bus() -> None:
    """StateUpdate events are journaled but NOT published to EventBus during replay.

    Given:
        A journal with a RunStartedEvent and a StateUpdate event, snapshot at seq=0.
    When:
        Events are replayed through ProtocolChannel with set_replaying(True).
    Then:
        The EventBus subscriber receives the RunStartedEvent but NOT the
        StateUpdate (filtered via isinstance check at comm_channel.py line 370).
    """
    journal = MemoryJournal()
    snapshot_store = MemorySnapshotStore()
    bus = EventBus()

    run_event = _run_started(run_id="su-run")
    state_event = StateUpdate(session_id="test", state=RunState.RUNNING)
    journal.append(run_event)
    journal.append(state_event)
    _set_snapshot(snapshot_store, last_journal_seq=0)

    channel = ProtocolChannel(journal=journal, event_bus=bus, session_id="test")
    queue = await bus.subscribe("test")

    resume_result = journal.resume(snapshot_store)
    assert resume_result is not None
    assert len(resume_result.events) == 2

    channel.set_replaying(True)
    for event in resume_result.events:
        await channel.publish(event)
    channel.set_replaying(False)

    # Only the RunStartedEvent should arrive; StateUpdate is filtered.
    received = await _drain_queue(queue, expected_count=1)
    assert len(received) == 1
    assert isinstance(received[0].event, RunStartedEvent)
    assert not isinstance(received[0].event, StateUpdate)


# ---------------------------------------------------------------------------
# Test 10: Replay does not duplicate journal entries
# ---------------------------------------------------------------------------


async def test_replay_does_not_duplicate_journal_entries() -> None:
    """set_replaying(True) prevents duplicate journal entries on replay.

    Given:
        A journal with one pre-existing event.
    When:
        The same event is published through ProtocolChannel with
        set_replaying(True).
    Then:
        The journal entry count is unchanged (no duplicate written).
    """
    journal = MemoryJournal()
    bus = EventBus()
    channel = ProtocolChannel(journal=journal, event_bus=bus, session_id="test")

    pre_event = _run_started(run_id="dup-check")
    journal.append(pre_event)
    initial_count = len(journal._entries)

    channel.set_replaying(True)
    await channel.publish(pre_event)
    channel.set_replaying(False)

    assert len(journal._entries) == initial_count


# ---------------------------------------------------------------------------
# Test 11: DirectChannel replay enqueues StateUpdate
# ---------------------------------------------------------------------------


async def test_direct_channel_replay_enqueues_state_update() -> None:
    """DirectChannel does NOT filter StateUpdate (unlike ProtocolChannel).

    Given:
        A DirectChannel with replaying mode enabled.
    When:
        A StateUpdate event is published during replay.
    Then:
        The StateUpdate IS enqueued in the channel's internal queue
        (DirectChannel has no EventBus filtering).
    """
    journal = MemoryJournal()
    channel = DirectChannel(journal=journal)

    state_event = StateUpdate(session_id="test", state=RunState.IDLE)

    channel.set_replaying(True)
    await channel.publish(state_event)
    channel.set_replaying(False)

    # DirectChannel does not filter StateUpdate — it should be in the queue.
    assert not channel.queue.empty()
    queued = channel.queue.get_nowait()
    assert isinstance(queued, StateUpdate)


# ---------------------------------------------------------------------------
# Test 12: Replay preserves ordering with mixed append/upsert
# ---------------------------------------------------------------------------


async def test_replay_preserves_ordering_mixed_append_upsert() -> None:
    """Journal resume returns events in seq order with upsert replacement.

    Given:
        A journal with: append A (seq=1), upsert "tool:1" B (seq=2),
        append C (seq=3), upsert "tool:1" D (seq=4, replaces B).
    When:
        journal.resume(snapshot_store) is called with snapshot at seq=0.
    Then:
        resume_result.events is [A, C, D] in order (B replaced by D).
        Replay through ProtocolChannel delivers the same order to subscriber.
    """
    journal = MemoryJournal()
    snapshot_store = MemorySnapshotStore()
    bus = EventBus()

    event_a = _run_started(run_id="A")
    event_b = _part_delta("B")
    event_c = _part_delta("C")
    event_d = _part_delta("D")

    journal.append(event_a)  # seq=1
    journal.upsert("tool:1", event_b)  # seq=2
    journal.append(event_c)  # seq=3
    journal.upsert("tool:1", event_d)  # seq=4, replaces B

    _set_snapshot(snapshot_store, last_journal_seq=0)

    channel = ProtocolChannel(journal=journal, event_bus=bus, session_id="test")
    queue = await bus.subscribe("test")

    resume_result = journal.resume(snapshot_store)
    assert resume_result is not None
    # B (seq=2) is replaced by D (seq=4) in the upserts dict, so events are [A, C, D].
    assert len(resume_result.events) == 3
    assert resume_result.events[0] is event_a
    assert resume_result.events[1] is event_c
    assert resume_result.events[2] is event_d

    channel.set_replaying(True)
    for event in resume_result.events:
        await channel.publish(event)
    channel.set_replaying(False)

    received = await _drain_queue(queue, expected_count=3)
    assert len(received) == 3
    assert isinstance(received[0].event, RunStartedEvent)
    assert received[0].event.run_id == "A"
    assert isinstance(received[1].event, PartDeltaEvent)
    assert received[1].event.delta.content_delta == "C"
    assert isinstance(received[2].event, PartDeltaEvent)
    assert received[2].event.delta.content_delta == "D"
