"""Regression test: EventBus replay buffer must not deliver stale.

StreamCompleteEvent from a previous turn to the current turn's consumer.

Bug: When turn 2 subscribes to EventBus, the replay buffer from turn 1
(which contains StreamCompleteEvent) was being replayed. The consumer
saw the stale StreamCompleteEvent, broke out of the loop, and cancelled
the native runner via ``tg.cancel_scope.cancel()`` — causing a
CancelledError in ``agentlet.iter()`` before the LLM was ever called.

Fix: ``_run_turn_unlocked`` now calls ``event_bus.clear_replay_buffer()``
at the start of each turn, ensuring new subscribers only receive events
from the current turn.
"""

from __future__ import annotations

import anyio
import pytest

from wolfharness.agents.events.events import (
    PartDeltaEvent,
    RunErrorEvent,
    RunStartedEvent,
    StreamCompleteEvent,
)
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.core import EventBus, EventEnvelope


def _make_run_started(session_id: str = "s1") -> RunStartedEvent:
    return RunStartedEvent(run_id="r1", session_id=session_id)


def _make_stream_complete(session_id: str = "s1") -> StreamCompleteEvent:
    return StreamCompleteEvent(
        message=ChatMessage(role="assistant", content="done"),
        session_id=session_id,
    )


def _make_part_delta(session_id: str = "s1") -> PartDeltaEvent:
    return PartDeltaEvent.text(index=0, content="hello")


@pytest.mark.unit
async def test_clear_replay_buffer_prevents_stale_events() -> None:
    """After clear_replay_buffer, new subscribers must NOT receive events from previous turns."""
    bus = EventBus()

    for event in [_make_run_started(), _make_stream_complete()]:
        await bus._send("s1", EventEnvelope(event=event, source_session_id="s1"))

    assert "s1" in bus._replay_buffers
    assert len(bus._replay_buffers["s1"]) == 2

    bus.clear_replay_buffer("s1")
    assert "s1" not in bus._replay_buffers

    queue = await bus.subscribe("s1", scope="session")

    new_event = _make_run_started()
    await bus._send("s1", EventEnvelope(event=new_event, source_session_id="s1"))

    received: list = []
    with anyio.fail_after(1.0):
        envelope = await queue.get()
        received.append(envelope.event)

    assert len(received) == 1
    assert isinstance(received[0], RunStartedEvent)


@pytest.mark.unit
async def test_replay_buffer_replays_stale_without_clear() -> None:
    """Without clear_replay_buffer, new subscribers DO receive stale events."""
    bus = EventBus()

    for event in [_make_run_started(), _make_stream_complete()]:
        await bus._send("s1", EventEnvelope(event=event, source_session_id="s1"))

    queue = await bus.subscribe("s1", scope="session")

    received: list = []
    with anyio.fail_after(1.0):
        while True:
            envelope = await queue.get()
            received.append(envelope.event)
            if isinstance(envelope.event, (StreamCompleteEvent, RunErrorEvent)):
                break

    assert len(received) == 2
    assert isinstance(received[0], RunStartedEvent)
    assert isinstance(received[1], StreamCompleteEvent)


@pytest.mark.unit
async def test_clear_replay_buffer_preserves_active_subscribers() -> None:
    """clear_replay_buffer must NOT close active subscriber queues."""
    bus = EventBus()

    queue1 = await bus.subscribe("s1", scope="session")

    bus.clear_replay_buffer("s1")

    new_event = _make_part_delta()
    await bus._send("s1", EventEnvelope(event=new_event, source_session_id="s1"))

    received: list = []
    with anyio.fail_after(1.0):
        envelope = await queue1.get()
        received.append(envelope.event)

    assert len(received) == 1
    assert isinstance(received[0], PartDeltaEvent)


@pytest.mark.unit
async def test_clear_replay_buffer_idempotent() -> None:
    """clear_replay_buffer should be safe to call on non-existent session."""
    bus = EventBus()
    bus.clear_replay_buffer("nonexistent")
    bus.clear_replay_buffer("nonexistent")
