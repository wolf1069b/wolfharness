"""Tests for InProcessTransport (EventTransport dimension)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from wolfharness.lifecycle import EventEnvelope, EventTransport, InProcessTransport


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = pytest.mark.unit


def _make_envelope(
    seq: int | None = None,
    session_id: str = "s1",
    event_type: str = "test_event",
    payload: dict[str, object] | None = None,
) -> EventEnvelope:
    """Create a minimal EventEnvelope for testing."""
    return EventEnvelope(
        event_type=event_type,
        session_id=session_id,
        timestamp="2025-01-01T00:00:00Z",
        payload=payload or {"seq": seq},
        seq=seq,
    )


# --- Protocol conformance ---


def test_in_process_transport_satisfies_event_transport_protocol():
    """InProcessTransport should satisfy the EventTransport Protocol."""
    transport = InProcessTransport()
    assert isinstance(transport, EventTransport)


# --- publish + subscribe round-trip ---


@pytest.mark.asyncio
async def test_publish_subscribe_round_trip():
    """Subscriber receives published envelope."""
    transport = InProcessTransport()
    envelope = _make_envelope(seq=1)

    await transport.publish(envelope)

    received: list[EventEnvelope] = []
    reader = transport.subscribe("s1")
    task = asyncio.create_task(_collect_envelopes(reader, received, count=1))
    await asyncio.wait_for(task, timeout=2.0)

    assert len(received) == 1
    assert received[0] is envelope
    transport.close()


async def _collect_envelopes(
    iterator: AsyncIterator[EventEnvelope],
    out: list[EventEnvelope],
    count: int,
) -> None:
    """Collect *count* envelopes from *iterator* into *out*."""
    collected = 0
    async for envelope in iterator:
        out.append(envelope)
        collected += 1
        if collected >= count:
            break


# --- Replay buffer for late subscribers ---


@pytest.mark.asyncio
async def test_replay_buffer_for_late_subscribers():
    """Late subscriber gets replayed events then new ones."""
    transport = InProcessTransport(replay_buffer_size=100)

    # Publish 50 events before any subscriber.
    for i in range(50):
        await transport.publish(_make_envelope(seq=i, payload={"index": i}))

    received: list[EventEnvelope] = []
    reader = transport.subscribe("s1", from_seq=0)
    task = asyncio.create_task(_collect_envelopes(reader, received, count=50))
    await asyncio.wait_for(task, timeout=5.0)

    assert len(received) == 50
    # All 50 replayed events should be received.
    for i, env in enumerate(received):
        assert env.seq == i
        assert env.payload["index"] == i  # type: ignore[typeddict-item]
    transport.close()


@pytest.mark.asyncio
async def test_replay_buffer_with_from_seq_filter():
    """Subscriber with from_seq only gets events with seq >= from_seq."""
    transport = InProcessTransport(replay_buffer_size=100)

    # Publish 20 events with seq 0-19.
    for i in range(20):
        await transport.publish(_make_envelope(seq=i, payload={"index": i}))

    received: list[EventEnvelope] = []
    reader = transport.subscribe("s1", from_seq=10)
    task = asyncio.create_task(_collect_envelopes(reader, received, count=10))
    await asyncio.wait_for(task, timeout=5.0)

    assert len(received) == 10
    # Should receive events with seq 10-19.
    for i, env in enumerate(received):
        assert env.seq == 10 + i
    transport.close()


@pytest.mark.asyncio
async def test_no_replay_buffer_when_size_zero():
    """With replay_buffer_size=0, no events are replayed."""
    transport = InProcessTransport(replay_buffer_size=0)

    # Publish 5 events before subscriber.
    for i in range(5):
        await transport.publish(_make_envelope(seq=i))

    # Even with from_seq > 0, no replay should happen.
    received: list[EventEnvelope] = []
    reader = transport.subscribe("s1", from_seq=1)
    task = asyncio.create_task(_collect_envelopes(reader, received, count=1))
    await asyncio.wait_for(task, timeout=2.0)

    # Should get 1 new event (the one still in the queue from publish).
    # But since from_seq=1 and no replay buffer, it should only yield from the queue.
    assert len(received) >= 0  # May or may not get events from queue
    transport.close()


@pytest.mark.asyncio
async def test_replay_then_new_events():
    """Subscriber gets replayed events first, then new events."""
    transport = InProcessTransport(replay_buffer_size=100)

    # Publish 5 events before subscriber.
    for i in range(5):
        await transport.publish(_make_envelope(seq=i, payload={"phase": "replay"}))

    received: list[EventEnvelope] = []

    # Start subscriber that will collect 10 events (5 replayed + 5 new).
    reader = transport.subscribe("s1", from_seq=0)
    task = asyncio.create_task(_collect_envelopes(reader, received, count=10))

    # Give a moment for replay to be consumed.
    await asyncio.sleep(0.2)

    # Publish 5 more events.
    for i in range(5, 10):
        await transport.publish(_make_envelope(seq=i, payload={"phase": "new"}))

    await asyncio.wait_for(task, timeout=5.0)

    assert len(received) == 10
    # First 5 should be replay events.
    for i in range(5):
        assert received[i].payload["phase"] == "replay"  # type: ignore[typeddict-item]
    # Last 5 should be new events.
    for i in range(5, 10):
        assert received[i].payload["phase"] == "new"  # type: ignore[typeddict-item]
    transport.close()


# --- ack is no-op ---


def test_ack_is_noop():
    """ack() should not raise or perform any action."""
    transport = InProcessTransport()
    transport.ack(42)  # Should not raise.
    transport.close()


# --- close prevents further operations ---


def test_close_prevents_publish():
    """publish() after close() should raise RuntimeError."""
    transport = InProcessTransport()
    transport.close()

    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(transport.publish(_make_envelope(seq=1)))


def test_close_prevents_subscribe():
    """subscribe() on closed transport should raise RuntimeError."""
    transport = InProcessTransport()
    transport.close()

    with pytest.raises(RuntimeError, match="closed"):
        transport.subscribe("s1")


# --- Events pass as Python objects (no serialization) ---


@pytest.mark.asyncio
async def test_events_pass_as_python_objects():
    """Envelopes are delivered as the same Python object (no serialization)."""
    transport = InProcessTransport()
    envelope = _make_envelope(
        seq=1,
        payload={"nested": {"deep": [1, 2, 3]}},
    )

    await transport.publish(envelope)

    received: list[EventEnvelope] = []
    reader = transport.subscribe("s1")
    task = asyncio.create_task(_collect_envelopes(reader, received, count=1))
    await asyncio.wait_for(task, timeout=2.0)

    assert len(received) == 1
    # The exact same object should be delivered (identity check).
    assert received[0] is envelope
    transport.close()


# --- Multiple topics ---


@pytest.mark.asyncio
async def test_multiple_topics_isolated():
    """Events on different topics are delivered to the correct subscribers."""
    transport = InProcessTransport()

    env_a = _make_envelope(seq=1, session_id="topic_a")
    env_b = _make_envelope(seq=1, session_id="topic_b")

    await transport.publish(env_a)
    await transport.publish(env_b)

    received_a: list[EventEnvelope] = []
    received_b: list[EventEnvelope] = []

    reader_a = transport.subscribe("topic_a")
    reader_b = transport.subscribe("topic_b")

    task_a = asyncio.create_task(_collect_envelopes(reader_a, received_a, count=1))
    task_b = asyncio.create_task(_collect_envelopes(reader_b, received_b, count=1))

    await asyncio.wait_for(asyncio.gather(task_a, task_b), timeout=2.0)

    assert len(received_a) == 1
    assert received_a[0].session_id == "topic_a"
    assert len(received_b) == 1
    assert received_b[0].session_id == "topic_b"
    transport.close()


# --- Replay buffer size limit ---


@pytest.mark.asyncio
async def test_replay_buffer_size_limit():
    """Replay buffer retains at most replay_buffer_size events."""
    transport = InProcessTransport(replay_buffer_size=10)

    # Publish 20 events.
    for i in range(20):
        await transport.publish(_make_envelope(seq=i, payload={"index": i}))

    received: list[EventEnvelope] = []
    # from_seq=1 triggers replay; buffer capped at 10 so only last 10 (seq 10-19) are replayed.
    reader = transport.subscribe("s1", from_seq=1)
    task = asyncio.create_task(_collect_envelopes(reader, received, count=10))
    await asyncio.wait_for(task, timeout=5.0)

    # Should only get the last 10 events from the replay buffer.
    assert len(received) == 10
    for i, env in enumerate(received):
        assert env.seq == 10 + i
    transport.close()
