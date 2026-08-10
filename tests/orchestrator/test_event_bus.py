"""Unit tests for EventBus (SessionPool Group 2.10).

Tests pub/sub semantics, bounded stream dropping, QueueShutDown-based
shutdown, subscriber lifecycle management, and event coalescing infrastructure.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import anyio
from pydantic_ai import (
    PartEndEvent,
    TextPart,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPartDelta,
)
import pytest

from wolfharness.agents.events import (
    CompactionEvent,
    CustomEvent,
    PartDeltaEvent,
    PartStartEvent,
    PlanUpdateEvent,
    RunErrorEvent,
    RunFailedEvent,
    RunStartedEvent,
    SessionResumeEvent,
    SpawnSessionStart,
    StreamCompleteEvent,
    SubAgentEvent,
    TerminalContentItem,
    TextContentItem,
    ToolCallCompleteEvent,
    ToolCallDeferredEvent,
    ToolCallProgressEvent,
    ToolCallStartEvent,
    ToolResultMetadataEvent,
)
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.core import (  # type: ignore[attr-defined]
    EventBus,
    EventEnvelope,
    _is_immediate,
    _merge_envelopes,
    _merge_key,
    _merge_progress_events,
    _merge_text_deltas,
    _merge_thinking_deltas,
    _merge_tool_call_deltas,
    _rebind,
    drain_and_merge,
)


pytestmark = [pytest.mark.unit, pytest.mark.anyio]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def event_bus() -> EventBus:
    """Return a fresh EventBus with small buffer for deterministic tests."""
    return EventBus(max_queue_size=3)


@pytest.fixture
def sample_event() -> RunStartedEvent:
    """Return a sample RichAgentStreamEvent for publishing."""
    return RunStartedEvent(session_id="sess-1", run_id="run-1")


async def _drain_stream(stream: asyncio.Queue[Any]) -> list[Any]:
    """Drain all available items from a memory receive stream without blocking."""
    items: list[Any] = []
    while True:
        try:
            items.append(stream.get_nowait())
        except (asyncio.QueueEmpty, asyncio.QueueShutDown):
            break
    return items


async def _receive_one(stream: asyncio.Queue[Any], timeout: float = 0.5) -> Any | None:
    """Receive one item from a stream with a timeout."""
    try:
        with anyio.fail_after(timeout):
            return await stream.get()
    except TimeoutError:
        return None


# ---------------------------------------------------------------------------
# Subscribe / Unsubscribe
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_subscribe_creates_receive_stream(event_bus: EventBus) -> None:
    """subscribe() returns a memory object receive stream."""
    stream = await event_bus.subscribe("sess-1")
    assert hasattr(stream, "get")
    assert hasattr(stream, "get_nowait")


@pytest.mark.anyio
async def test_subscribe_multiple_streams_same_session(event_bus: EventBus) -> None:
    """Multiple subscribers for the same session each get their own stream."""
    s1 = await event_bus.subscribe("sess-1")
    s2 = await event_bus.subscribe("sess-1")
    counts = await event_bus.get_subscriber_counts()
    assert counts["sess-1"] == 2
    assert s1 is not s2


@pytest.mark.anyio
async def test_unsubscribe_removes_stream(event_bus: EventBus) -> None:
    """unsubscribe() removes the specific stream and cleans up empty lists."""
    s1 = await event_bus.subscribe("sess-1")
    s2 = await event_bus.subscribe("sess-1")
    await event_bus.unsubscribe("sess-1", s1)
    counts = await event_bus.get_subscriber_counts()
    assert counts["sess-1"] == 1
    await event_bus.unsubscribe("sess-1", s2)
    counts = await event_bus.get_subscriber_counts()
    assert "sess-1" not in counts


@pytest.mark.anyio
async def test_unsubscribe_unknown_session_noop(event_bus: EventBus) -> None:
    """Unsubscribing from a non-existent session is a no-op."""
    _recv: asyncio.Queue[Any] = asyncio.Queue(maxsize=10)
    await event_bus.unsubscribe("missing", _recv)
    counts = await event_bus.get_subscriber_counts()
    assert counts == {}


@pytest.mark.anyio
async def test_unsubscribe_wrong_stream_noop(event_bus: EventBus) -> None:
    """Unsubscribing a stream that was never subscribed is a no-op."""
    s_real = await event_bus.subscribe("sess-1")
    recv_fake: asyncio.Queue[Any] = asyncio.Queue(maxsize=10)
    await event_bus.unsubscribe("sess-1", recv_fake)
    counts = await event_bus.get_subscriber_counts()
    assert counts["sess-1"] == 1
    _ = s_real


# ---------------------------------------------------------------------------
# Publish - single & multiple subscribers
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_publish_single_subscriber(
    event_bus: EventBus,
    sample_event: RunStartedEvent,
) -> None:
    """A published event reaches the subscriber stream."""
    stream = await event_bus.subscribe("sess-1")
    await event_bus.publish("sess-1", sample_event)
    received = await _receive_one(stream)
    assert received is not None
    assert isinstance(received.event, RunStartedEvent)
    assert received.event.run_id == "run-1"


@pytest.mark.anyio
async def test_publish_multiple_subscribers(
    event_bus: EventBus,
    sample_event: RunStartedEvent,
) -> None:
    """Each subscriber receives an independent shallow copy of the event."""
    s1 = await event_bus.subscribe("sess-1")
    s2 = await event_bus.subscribe("sess-1")
    await event_bus.publish("sess-1", sample_event)
    ev1 = await _receive_one(s1)
    ev2 = await _receive_one(s2)
    assert ev1 is not None
    assert ev2 is not None
    assert ev1 == ev2
    assert isinstance(ev1.event, RunStartedEvent)
    assert isinstance(ev2.event, RunStartedEvent)
    assert ev1.event.run_id == ev2.event.run_id


@pytest.mark.anyio
async def test_publish_no_subscribers_is_noop(
    event_bus: EventBus,
    sample_event: RunStartedEvent,
) -> None:
    """Publishing to a session with no subscribers does not raise."""
    await event_bus.publish("sess-1", sample_event)
    counts = await event_bus.get_subscriber_counts()
    assert counts == {}


@pytest.mark.anyio
async def test_publish_different_sessions_isolated(
    event_bus: EventBus,
    sample_event: RunStartedEvent,
) -> None:
    """Events are only delivered to streams for the matching session_id."""
    s1 = await event_bus.subscribe("sess-1")
    s2 = await event_bus.subscribe("sess-2")
    await event_bus.publish("sess-1", sample_event)
    received = await _receive_one(s1)
    assert received is not None
    s2_items = await _drain_stream(s2)
    assert len(s2_items) == 0


# ---------------------------------------------------------------------------
# Bounded stream dropping
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_publish_drops_subscriber_when_buffer_full(
    sample_event: RunStartedEvent,
) -> None:
    """When a subscriber buffer is full and can't drain, subscriber is dropped."""
    bus = EventBus(max_queue_size=3, overflow_policy="drop_subscriber")
    stream = await bus.subscribe("sess-1")
    for i in range(3):
        await bus.publish("sess-1", RunStartedEvent(session_id="sess-1", run_id=f"ev{i}"))
    await bus.publish("sess-1", RunStartedEvent(session_id="sess-1", run_id="ev3"))
    await bus.publish("sess-1", RunStartedEvent(session_id="sess-1", run_id="ev4"))

    items = await _drain_stream(stream)
    run_ids = [e.event.run_id for e in items if isinstance(e.event, RunStartedEvent)]
    assert len(run_ids) <= 3
    counts = await bus.get_subscriber_counts()
    assert "sess-1" not in counts


@pytest.mark.anyio
async def test_publish_removes_dead_subscriber_on_broken_resource(
    event_bus: EventBus,
    sample_event: RunStartedEvent,
) -> None:
    """Subscribers with broken send streams are removed."""
    stream = await event_bus.subscribe("sess-1")
    stream.shutdown()
    await event_bus.publish("sess-1", sample_event)
    counts = await event_bus.get_subscriber_counts()
    assert "sess-1" not in counts


# ---------------------------------------------------------------------------
# close_session / EndOfStream shutdown
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_close_session_signals_end_of_stream(
    event_bus: EventBus,
    sample_event: RunStartedEvent,
) -> None:
    """close_session() closes send streams, causing QueueShutDown on consumers."""
    s1 = await event_bus.subscribe("sess-1")
    s2 = await event_bus.subscribe("sess-1")
    await event_bus.publish("sess-1", sample_event)
    await event_bus.close_session("sess-1")

    received1: list[Any] = []
    with contextlib.suppress(asyncio.QueueShutDown):
        while True:
            received1.append(s1.get_nowait())

    received2: list[Any] = []
    with contextlib.suppress(asyncio.QueueShutDown):
        while True:
            received2.append(s2.get_nowait())

    assert len(received1) >= 1
    assert len(received2) >= 1


@pytest.mark.anyio
async def test_close_session_removes_all_subscribers(
    event_bus: EventBus,
) -> None:
    """After close_session, no subscribers remain for that session."""
    await event_bus.subscribe("sess-1")
    await event_bus.subscribe("sess-1")
    await event_bus.close_session("sess-1")
    counts = await event_bus.get_subscriber_counts()
    assert "sess-1" not in counts


@pytest.mark.anyio
async def test_close_session_unknown_session_noop(event_bus: EventBus) -> None:
    """Closing a session that never had subscribers is a no-op."""
    await event_bus.close_session("missing")
    counts = await event_bus.get_subscriber_counts()
    assert counts == {}


# ---------------------------------------------------------------------------
# get_subscriber_counts
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_subscriber_counts_returns_snapshot(event_bus: EventBus) -> None:
    """get_subscriber_counts returns a snapshot of subscriber counts."""
    await event_bus.subscribe("sess-a")
    await event_bus.subscribe("sess-a")
    await event_bus.subscribe("sess-b")
    counts = await event_bus.get_subscriber_counts()
    assert counts == {"sess-a": 2, "sess-b": 1}


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_replay_buffer_bounds(event_bus: EventBus) -> None:
    """Publishing more events than replay_buffer_size drops oldest."""
    for i in range(150):
        await event_bus.publish("sess-1", RunStartedEvent(session_id="sess-1", run_id=f"ev{i}"))
    buffer = event_bus._replay_buffers["sess-1"]
    assert len(buffer) == 100
    run_ids = [e.event.run_id for e in buffer]
    assert run_ids[0] == "ev50"
    assert run_ids[-1] == "ev149"


@pytest.mark.anyio
async def test_replay_buffer_cleared_on_session_close(event_bus: EventBus) -> None:
    """close_session removes the replay buffer for the session."""
    await event_bus.publish("sess-1", RunStartedEvent(session_id="sess-1", run_id="ev1"))
    assert "sess-1" in event_bus._replay_buffers
    await event_bus.close_session("sess-1")
    assert "sess-1" not in event_bus._replay_buffers


@pytest.mark.anyio
async def test_replay_buffer_events_in_order(event_bus: EventBus) -> None:
    """Events in the replay buffer are stored oldest-to-newest."""
    for i in range(5):
        await event_bus.publish("sess-1", RunStartedEvent(session_id="sess-1", run_id=f"ev{i}"))
    buffer = event_bus._replay_buffers["sess-1"]
    assert len(buffer) == 5
    run_ids = [e.event.run_id for e in buffer]
    assert run_ids == ["ev0", "ev1", "ev2", "ev3", "ev4"]


@pytest.mark.anyio
async def test_replay_buffer_per_session_isolated(event_bus: EventBus) -> None:
    """Each session has its own independent replay buffer."""
    await event_bus.publish("sess-1", RunStartedEvent(session_id="sess-1", run_id="a"))
    await event_bus.publish("sess-2", RunStartedEvent(session_id="sess-2", run_id="b"))
    assert event_bus._replay_buffers["sess-1"][0].event.run_id == "a"
    assert event_bus._replay_buffers["sess-2"][0].event.run_id == "b"


@pytest.mark.anyio
async def test_replay_buffer_custom_size() -> None:
    """EventBus accepts a custom replay_buffer_size."""
    bus = EventBus(max_queue_size=3, replay_buffer_size=10)
    for i in range(15):
        await bus.publish("sess-1", RunStartedEvent(session_id="sess-1", run_id=f"ev{i}"))
    assert len(bus._replay_buffers["sess-1"]) == 10
    assert bus._replay_buffers["sess-1"][0].event.run_id == "ev5"
    assert bus._replay_buffers["sess-1"][-1].event.run_id == "ev14"


# ---------------------------------------------------------------------------
# Replay protocol
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_replay_protocol_new_subscriber_gets_historical() -> None:
    """New subscriber receives last N buffered events as replay."""
    bus = EventBus(max_queue_size=10)
    for i in range(5):
        await bus.publish("sess-1", RunStartedEvent(session_id="sess-1", run_id=f"ev{i}"))

    stream = await bus.subscribe("sess-1")
    received = await _drain_stream(stream)

    assert len(received) == 5
    run_ids = [e.event.run_id for e in received if isinstance(e.event, RunStartedEvent)]
    assert run_ids == ["ev0", "ev1", "ev2", "ev3", "ev4"]


@pytest.mark.anyio
async def test_replay_protocol_ordering() -> None:
    """Replayed events precede live events in the stream."""
    bus = EventBus(max_queue_size=10)
    for i in range(3):
        await bus.publish("sess-1", RunStartedEvent(session_id="sess-1", run_id=f"hist-{i}"))

    stream = await bus.subscribe("sess-1")

    for i in range(2):
        await bus.publish("sess-1", RunStartedEvent(session_id="sess-1", run_id=f"live-{i}"))

    received = await _drain_stream(stream)
    run_ids = [e.event.run_id for e in received if isinstance(e.event, RunStartedEvent)]
    assert run_ids == ["hist-0", "hist-1", "hist-2", "live-0", "live-1"]


@pytest.mark.anyio
async def test_replay_protocol_no_duplicates() -> None:
    """No duplicate events when publish happens during subscribe replay."""
    bus = EventBus(max_queue_size=10)
    for i in range(5):
        await bus.publish("sess-1", RunStartedEvent(session_id="sess-1", run_id=f"ev{i}"))

    stream = await bus.subscribe("sess-1")
    received = await _drain_stream(stream)

    run_ids = [e.event.run_id for e in received if isinstance(e.event, RunStartedEvent)]
    assert len(run_ids) == len(set(run_ids)), f"Duplicate run_ids found: {run_ids}"


@pytest.mark.anyio
async def test_replay_protocol_race_condition() -> None:
    """Subscribe concurrently with publishes; all events arrive in order."""
    bus = EventBus(max_queue_size=10)

    for i in range(3):
        await bus.publish("sess-1", RunStartedEvent(session_id="sess-1", run_id=f"hist-{i}"))

    subscribe_task = asyncio.create_task(bus.subscribe("sess-1"))
    publish_tasks = [
        asyncio.create_task(
            bus.publish("sess-1", RunStartedEvent(session_id="sess-1", run_id=f"race-{i}"))
        )
        for i in range(3)
    ]

    stream = await subscribe_task
    await asyncio.gather(*publish_tasks)

    for i in range(2):
        await bus.publish("sess-1", RunStartedEvent(session_id="sess-1", run_id=f"live-{i}"))

    received = await _drain_stream(stream)
    run_ids = [e.event.run_id for e in received if isinstance(e.event, RunStartedEvent)]

    assert len(run_ids) == 8, f"Expected 8 events, got {len(run_ids)}: {run_ids}"
    assert len(run_ids) == len(set(run_ids)), f"Duplicate run_ids found: {run_ids}"
    assert run_ids[:3] == ["hist-0", "hist-1", "hist-2"]


# ---------------------------------------------------------------------------
# SSE event ordering
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_event_ordering_replay_then_live() -> None:
    """Replayed PartStart→PartDelta→PartEnd events precede live events in stream."""
    bus = EventBus(max_queue_size=10)

    await bus.publish("sess-1", PartStartEvent(index=0, part=TextPart(content="hello")))
    await bus.publish(
        "sess-1", PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=" world"))
    )
    await bus.publish("sess-1", PartEndEvent(index=0, part=TextPart(content="hello world")))

    stream = await bus.subscribe("sess-1")

    await bus.publish("sess-1", PartStartEvent(index=1, part=TextPart(content="goodbye")))
    await bus.publish(
        "sess-1", PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=" world"))
    )
    await bus.publish("sess-1", PartEndEvent(index=1, part=TextPart(content="goodbye world")))

    received = await _drain_stream(stream)

    assert len(received) == 6

    assert isinstance(received[0].event, PartStartEvent)
    assert received[0].event.index == 0
    assert isinstance(received[1].event, PartDeltaEvent)
    assert received[1].event.index == 0
    assert isinstance(received[2].event, PartEndEvent)
    assert received[2].event.index == 0

    assert isinstance(received[3].event, PartStartEvent)
    assert received[3].event.index == 1
    assert isinstance(received[4].event, PartDeltaEvent)
    assert received[4].event.index == 1
    assert isinstance(received[5].event, PartEndEvent)
    assert received[5].event.index == 1


@pytest.mark.anyio
async def test_event_ordering_no_gaps_in_replay() -> None:
    """Replay buffer eviction drops oldest events; subscriber sees contiguous range."""
    bus = EventBus(max_queue_size=200, replay_buffer_size=100)

    for i in range(100):
        await bus.publish("sess-1", RunStartedEvent(session_id="sess-1", run_id=f"ev{i}"))

    for i in range(100, 150):
        await bus.publish("sess-1", RunStartedEvent(session_id="sess-1", run_id=f"ev{i}"))

    stream = await bus.subscribe("sess-1")
    received = await _drain_stream(stream)

    assert len(received) == 100

    run_ids = [e.event.run_id for e in received if isinstance(e.event, RunStartedEvent)]
    expected = [f"ev{i}" for i in range(50, 150)]
    assert run_ids == expected

    for i, rid in enumerate(run_ids):
        assert rid == f"ev{i + 50}"


@pytest.mark.anyio
async def test_event_ordering_concurrent_publish() -> None:
    """Concurrent publishers preserve per-task event ordering in replay buffer."""
    bus = EventBus(max_queue_size=200, replay_buffer_size=100)

    async def publisher(task_id: int, count: int) -> None:
        for i in range(count):
            await bus.publish(
                "sess-1",
                RunStartedEvent(session_id="sess-1", run_id=f"task{task_id}-ev{i}"),
            )

    tasks = [asyncio.create_task(publisher(tid, 20)) for tid in range(5)]
    await asyncio.gather(*tasks)

    stream = await bus.subscribe("sess-1")
    received = await _drain_stream(stream)

    assert len(received) == 100

    run_ids = [e.event.run_id for e in received if isinstance(e.event, RunStartedEvent)]
    assert len(run_ids) == 100
    assert len(run_ids) == len(set(run_ids)), f"Duplicate run_ids found: {run_ids}"

    for tid in range(5):
        task_events = [rid for rid in run_ids if rid.startswith(f"task{tid}-")]
        expected = [f"task{tid}-ev{i}" for i in range(20)]
        assert task_events == expected, f"Task {tid} events out of order: {task_events}"


@pytest.mark.anyio
async def test_event_ordering_mixed_sessions() -> None:
    """Events from different sessions are isolated; subscriber sees only its session."""
    bus = EventBus(max_queue_size=10)

    for i in range(5):
        await bus.publish("sess-1", RunStartedEvent(session_id="sess-1", run_id=f"s1-ev{i}"))
        await bus.publish("sess-2", RunStartedEvent(session_id="sess-2", run_id=f"s2-ev{i}"))
        await bus.publish("sess-3", RunStartedEvent(session_id="sess-3", run_id=f"s3-ev{i}"))

    stream = await bus.subscribe("sess-1")
    received = await _drain_stream(stream)

    assert len(received) == 5

    run_ids = [e.event.run_id for e in received if isinstance(e.event, RunStartedEvent)]
    assert run_ids == ["s1-ev0", "s1-ev1", "s1-ev2", "s1-ev3", "s1-ev4"]

    for e in received:
        if isinstance(e.event, RunStartedEvent):
            assert e.source_session_id == "sess-1"


# ---------------------------------------------------------------------------
# Descendants scope
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_child_events_visible_with_descendants_scope(event_bus: EventBus) -> None:
    """Parent subscriber with scope='descendants' receives child session events."""
    event_bus._session_tree["parent"] = ["child"]
    stream = await event_bus.subscribe("parent", scope="descendants")
    child_event = RunStartedEvent(session_id="child", run_id="run-child")
    await event_bus.publish("child", child_event)
    received = await _receive_one(stream)
    assert received is not None
    assert isinstance(received.event, RunStartedEvent)
    assert received.event.run_id == "run-child"


@pytest.mark.anyio
async def test_child_events_not_visible_with_session_scope(event_bus: EventBus) -> None:
    """Parent subscriber with scope='session' does NOT receive child session events."""
    event_bus._session_tree["parent"] = ["child"]
    stream = await event_bus.subscribe("parent", scope="session")
    child_event = RunStartedEvent(session_id="child", run_id="run-child")
    await event_bus.publish("child", child_event)
    items = await _drain_stream(stream)
    assert len(items) == 0


@pytest.mark.anyio
async def test_event_ordering_parent_and_child() -> None:
    """Events from parent and child arrive in correct interleaved order."""
    bus = EventBus(max_queue_size=10)
    bus._session_tree["parent"] = ["child"]
    stream = await bus.subscribe("parent", scope="descendants")
    events = [
        ("parent", "run-1"),
        ("child", "run-2"),
        ("parent", "run-3"),
        ("child", "run-4"),
        ("parent", "run-5"),
    ]
    for session_id, run_id in events:
        await bus.publish(session_id, RunStartedEvent(session_id=session_id, run_id=run_id))
    received: list[str] = []
    for _ in events:
        ev = await _receive_one(stream)
        assert ev is not None
        assert isinstance(ev.event, RunStartedEvent)
        received.append(ev.event.run_id)
    assert received == ["run-1", "run-2", "run-3", "run-4", "run-5"]


@pytest.mark.anyio
async def test_grandchild_events_visible_with_descendants_scope(
    event_bus: EventBus,
) -> None:
    """Parent subscriber with scope='descendants' receives grandchild events."""
    event_bus._session_tree["parent"] = ["child"]
    event_bus._session_tree["child"] = ["grandchild"]
    stream = await event_bus.subscribe("parent", scope="descendants")
    grandchild_event = RunStartedEvent(session_id="grandchild", run_id="run-grandchild")
    await event_bus.publish("grandchild", grandchild_event)
    received = await _receive_one(stream)
    assert received is not None
    assert isinstance(received.event, RunStartedEvent)
    assert received.event.run_id == "run-grandchild"


# ---------------------------------------------------------------------------
# Event coalescing infrastructure (Task 1)
# ---------------------------------------------------------------------------


# --- _is_immediate ---


@pytest.mark.parametrize(
    "event",
    [
        RunStartedEvent(session_id="s", run_id="r"),
        RunErrorEvent(message="err"),
        RunFailedEvent(run_id="r", session_id="s", exception=ValueError("test")),
        StreamCompleteEvent(message=ChatMessage(content="done", role="assistant")),
        SpawnSessionStart(
            child_session_id="c",
            parent_session_id="p",
            spawn_mechanism="task",
            source_name="agent",
            source_type="agent",
            description="test",
        ),
        CompactionEvent(session_id="s"),
        SessionResumeEvent(session_id="s", resolved_call_count=0),
        ToolCallStartEvent(tool_call_id="tc1", tool_name="bash", title="test"),
        ToolCallCompleteEvent(
            tool_name="bash",
            tool_call_id="tc1",
            tool_input={},
            tool_result="ok",
            agent_name="a",
            message_id="m",
        ),
        ToolCallDeferredEvent(
            tool_call_id="tc1",
            tool_name="bash",
            deferred_strategy="block",
            status="pending",
        ),
    ],
    ids=[
        "run_started",
        "run_error",
        "run_failed",
        "stream_complete",
        "spawn_session_start",
        "compaction",
        "session_resume",
        "tool_call_start",
        "tool_call_complete",
        "tool_call_deferred",
    ],
)
def test_immediate_returns_true_for_lifecycle_events(event: Any) -> None:
    """All 10 lifecycle event types are classified as immediate."""
    assert _is_immediate(event) is True


def test_immediate_returns_false_for_text_delta() -> None:
    """PartDeltaEvent with TextPartDelta is not immediate."""
    event = PartDeltaEvent.text(0, "hello")
    assert _is_immediate(event) is False


def test_immediate_returns_false_for_thinking_delta() -> None:
    """PartDeltaEvent with ThinkingPartDelta is not immediate."""
    event = PartDeltaEvent.thinking(0, "thinking")
    assert _is_immediate(event) is False


def test_immediate_returns_false_for_tool_call_delta() -> None:
    """PartDeltaEvent with ToolCallPartDelta is not immediate."""
    event = PartDeltaEvent.tool_call(0, "args", "tc1")
    assert _is_immediate(event) is False


def test_immediate_returns_false_for_tool_call_progress() -> None:
    """ToolCallProgressEvent is not immediate."""
    event = ToolCallProgressEvent(tool_call_id="tc1")
    assert _is_immediate(event) is False


def test_immediate_returns_false_for_plan_update() -> None:
    """PlanUpdateEvent is not immediate."""
    event = PlanUpdateEvent(entries=[])
    assert _is_immediate(event) is False


def test_immediate_returns_false_for_subagent_event() -> None:
    """SubAgentEvent is not immediate."""
    event = SubAgentEvent(
        source_name="agent",
        source_type="agent",
        event=RunStartedEvent(session_id="s", run_id="r"),
    )
    assert _is_immediate(event) is False


def test_immediate_returns_false_for_custom_event() -> None:
    """CustomEvent is not immediate."""
    event = CustomEvent(event_data="test")
    assert _is_immediate(event) is False


def test_immediate_returns_false_for_tool_result_metadata() -> None:
    """ToolResultMetadataEvent is not immediate."""
    event = ToolResultMetadataEvent(tool_call_id="tc1", metadata={})
    assert _is_immediate(event) is False


# --- _merge_key (classify) ---


def test_classify_text_delta() -> None:
    """PartDeltaEvent with TextPartDelta has merge key ('delta_text', '')."""
    event = PartDeltaEvent.text(0, "hello")
    assert _merge_key(event) == ("delta_text", "")


def test_classify_thinking_delta() -> None:
    """PartDeltaEvent with ThinkingPartDelta has merge key ('delta_thinking', '')."""
    event = PartDeltaEvent.thinking(0, "thinking")
    assert _merge_key(event) == ("delta_thinking", "")


def test_classify_tool_call_delta() -> None:
    """PartDeltaEvent with ToolCallPartDelta has merge key ('delta_tool_call', tool_call_id)."""
    event = PartDeltaEvent.tool_call(0, "args", "tc1")
    assert _merge_key(event) == ("delta_tool_call", "tc1")


def test_classify_tool_call_progress() -> None:
    """ToolCallProgressEvent has merge key ('progress', 'tool_call_id:status')."""
    event = ToolCallProgressEvent(tool_call_id="tc1", status="in_progress")
    assert _merge_key(event) == ("progress", "tc1:in_progress")


def test_classify_plan_update() -> None:
    """PlanUpdateEvent has merge key ('plan', '')."""
    event = PlanUpdateEvent(entries=[])
    assert _merge_key(event) == ("plan", "")


def test_classify_subagent_returns_none() -> None:
    """SubAgentEvent is passthrough (merge key None)."""
    event = SubAgentEvent(
        source_name="agent",
        source_type="agent",
        event=RunStartedEvent(session_id="s", run_id="r"),
    )
    assert _merge_key(event) is None


def test_classify_custom_returns_none() -> None:
    """CustomEvent is passthrough (merge key None)."""
    event = CustomEvent(event_data="test")
    assert _merge_key(event) is None


def test_classify_tool_result_metadata_returns_none() -> None:
    """ToolResultMetadataEvent is passthrough (merge key None)."""
    event = ToolResultMetadataEvent(tool_call_id="tc1", metadata={})
    assert _merge_key(event) is None


def test_classify_none_delta_returns_none() -> None:
    """PartDeltaEvent with delta=None is passthrough (merge key None)."""
    event: Any = PartDeltaEvent(index=0, delta=None)  # type: ignore[arg-type]
    assert _merge_key(event) is None


# --- _merge_text_deltas ---


def test_merge_text_deltas_concatenates_content() -> None:
    """Multiple text deltas are concatenated into a single content_delta."""
    events = [
        PartDeltaEvent.text(0, "hello "),
        PartDeltaEvent.text(0, "world"),
        PartDeltaEvent.text(0, "!"),
    ]
    merged = _merge_text_deltas(events)
    assert isinstance(merged, PartDeltaEvent)
    assert isinstance(merged.delta, TextPartDelta)
    assert merged.delta.content_delta == "hello world!"


def test_merge_text_deltas_uses_first_index() -> None:
    """Merged text delta uses the first event's index."""
    events = [
        PartDeltaEvent.text(5, "a"),
        PartDeltaEvent.text(7, "b"),
    ]
    merged = _merge_text_deltas(events)
    assert merged.index == 5


def test_merge_text_deltas_single_event() -> None:
    """Merging a single text delta returns the same content."""
    events = [PartDeltaEvent.text(0, "solo")]
    merged = _merge_text_deltas(events)
    assert isinstance(merged.delta, TextPartDelta)
    assert merged.delta.content_delta == "solo"


# --- _merge_thinking_deltas ---


def test_merge_thinking_deltas_concatenates_content() -> None:
    """Multiple thinking deltas are concatenated into a single content_delta."""
    events = [
        PartDeltaEvent.thinking(0, "think "),
        PartDeltaEvent.thinking(0, "more"),
    ]
    merged = _merge_thinking_deltas(events)
    assert isinstance(merged, PartDeltaEvent)
    assert isinstance(merged.delta, ThinkingPartDelta)
    assert merged.delta.content_delta == "think more"


def test_merge_thinking_deltas_uses_first_index() -> None:
    """Merged thinking delta uses the first event's index."""
    events = [
        PartDeltaEvent.thinking(3, "a"),
        PartDeltaEvent.thinking(7, "b"),
    ]
    merged = _merge_thinking_deltas(events)
    assert merged.index == 3


# --- _merge_tool_call_deltas ---


def test_merge_tool_call_deltas_concatenates_args() -> None:
    """Multiple tool_call deltas are concatenated into a single args_delta."""
    events = [
        PartDeltaEvent.tool_call(0, '{"path"', "tc1"),
        PartDeltaEvent.tool_call(0, ': "foo"}', "tc1"),
    ]
    merged = _merge_tool_call_deltas(events)
    assert isinstance(merged, PartDeltaEvent)
    assert isinstance(merged.delta, ToolCallPartDelta)
    assert merged.delta.args_delta == '{"path": "foo"}'


def test_merge_tool_call_deltas_uses_first_index_and_tool_call_id() -> None:
    """Merged tool_call delta uses first event's index and tool_call_id."""
    events = [
        PartDeltaEvent.tool_call(2, "a", "tc-first"),
        PartDeltaEvent.tool_call(5, "b", "tc-second"),
    ]
    merged = _merge_tool_call_deltas(events)
    assert merged.index == 2
    assert isinstance(merged.delta, ToolCallPartDelta)
    assert merged.delta.tool_call_id == "tc-first"


# --- _merge_progress_events ---


def test_merge_progress_events_concatenates_items() -> None:
    """Items from all progress events are concatenated."""
    events = [
        ToolCallProgressEvent(
            tool_call_id="tc1",
            status="in_progress",
            items=[TerminalContentItem(terminal_id="t1")],
        ),
        ToolCallProgressEvent(
            tool_call_id="tc1",
            status="in_progress",
            items=[TextContentItem(text="output")],
        ),
    ]
    merged = _merge_progress_events(events)
    assert len(merged.items) == 2
    assert isinstance(merged.items[0], TerminalContentItem)
    assert isinstance(merged.items[1], TextContentItem)


def test_merge_progress_events_uses_last_fields() -> None:
    """Merged progress event uses last event's title, status, replace_content, tool_name."""
    events = [
        ToolCallProgressEvent(
            tool_call_id="tc1",
            status="in_progress",
            title="first",
            replace_content=False,
            tool_name="bash",
            items=[],
        ),
        ToolCallProgressEvent(
            tool_call_id="tc1",
            status="completed",
            title="last",
            replace_content=True,
            tool_name="read",
            items=[],
        ),
    ]
    merged = _merge_progress_events(events)
    assert merged.title == "last"
    assert merged.status == "completed"
    assert merged.replace_content is True
    assert merged.tool_name == "read"


def test_merge_progress_events_keeps_duplicate_terminal_ids() -> None:
    """Duplicate terminal_id items are kept (no dedup)."""
    events = [
        ToolCallProgressEvent(
            tool_call_id="tc1",
            status="in_progress",
            items=[TerminalContentItem(terminal_id="t1")],
        ),
        ToolCallProgressEvent(
            tool_call_id="tc1",
            status="in_progress",
            items=[TerminalContentItem(terminal_id="t1")],
        ),
    ]
    merged = _merge_progress_events(events)
    assert len(merged.items) == 2
    # Both items should be TerminalContentItem with terminal_id="t1"
    for item in merged.items:
        assert isinstance(item, TerminalContentItem)
        assert item.terminal_id == "t1"


# --- _merge_envelopes ---


def test_merge_envelopes_groups_consecutive_text_deltas() -> None:
    """Consecutive text deltas are merged into a single envelope."""
    envelopes = [
        EventEnvelope(source_session_id="s1", event=PartDeltaEvent.text(0, "hello ")),
        EventEnvelope(source_session_id="s1", event=PartDeltaEvent.text(0, "world")),
    ]
    result = _merge_envelopes(envelopes)
    assert len(result) == 1
    assert isinstance(result[0].event, PartDeltaEvent)
    assert isinstance(result[0].event.delta, TextPartDelta)
    assert result[0].event.delta.content_delta == "hello world"
    assert result[0].source_session_id == "s1"


def test_merge_envelopes_type_change_creates_separate_groups() -> None:
    """Type change (text→thinking) creates two separate merged groups."""
    envelopes = [
        EventEnvelope(source_session_id="s1", event=PartDeltaEvent.text(0, "text")),
        EventEnvelope(source_session_id="s1", event=PartDeltaEvent.thinking(0, "think")),
    ]
    result = _merge_envelopes(envelopes)
    assert len(result) == 2
    assert isinstance(result[0].event.delta, TextPartDelta)
    assert result[0].event.delta.content_delta == "text"
    assert isinstance(result[1].event.delta, ThinkingPartDelta)
    assert result[1].event.delta.content_delta == "think"


def test_merge_envelopes_drops_none_delta() -> None:
    """PartDeltaEvent with delta=None is dropped, not merged or dispatched."""
    envelopes = [
        EventEnvelope(source_session_id="s1", event=PartDeltaEvent.text(0, "keep")),
        EventEnvelope(source_session_id="s1", event=PartDeltaEvent(index=0, delta=None)),  # type: ignore[arg-type]
        EventEnvelope(source_session_id="s1", event=PartDeltaEvent.text(0, "this")),
    ]
    result = _merge_envelopes(envelopes)
    # None delta is dropped; remaining two text deltas are merged
    assert len(result) == 1
    assert isinstance(result[0].event.delta, TextPartDelta)
    assert result[0].event.delta.content_delta == "keepthis"


def test_merge_envelopes_plan_last_wins() -> None:
    """PlanUpdateEvent groups use last-wins strategy."""
    envelopes = [
        EventEnvelope(source_session_id="s1", event=PlanUpdateEvent(entries=[])),
        EventEnvelope(source_session_id="s1", event=PlanUpdateEvent(entries=[])),
        EventEnvelope(source_session_id="s1", event=PlanUpdateEvent(entries=[])),
    ]
    result = _merge_envelopes(envelopes)
    assert len(result) == 1
    assert isinstance(result[0].event, PlanUpdateEvent)


def test_merge_envelopes_passthrough_extends_without_merging() -> None:
    """Passthrough events (SubAgentEvent, CustomEvent) are not merged."""
    sub_event = SubAgentEvent(
        source_name="agent",
        source_type="agent",
        event=RunStartedEvent(session_id="s", run_id="r"),
    )
    custom_event = CustomEvent(event_data="test")
    envelopes = [
        EventEnvelope(source_session_id="s1", event=sub_event),
        EventEnvelope(source_session_id="s1", event=custom_event),
    ]
    result = _merge_envelopes(envelopes)
    assert len(result) == 2
    assert isinstance(result[0].event, SubAgentEvent)
    assert isinstance(result[1].event, CustomEvent)


def test_merge_envelopes_empty_list_returns_empty() -> None:
    """Empty envelope list returns empty result."""
    result = _merge_envelopes([])
    assert result == []


def test_merge_envelopes_tool_call_progress_merged() -> None:
    """Consecutive ToolCallProgressEvents with same key are merged."""
    envelopes = [
        EventEnvelope(
            source_session_id="s1",
            event=ToolCallProgressEvent(
                tool_call_id="tc1",
                status="in_progress",
                title="first",
                items=[TerminalContentItem(terminal_id="t1")],
            ),
        ),
        EventEnvelope(
            source_session_id="s1",
            event=ToolCallProgressEvent(
                tool_call_id="tc1",
                status="in_progress",
                title="second",
                items=[TextContentItem(text="out")],
            ),
        ),
    ]
    result = _merge_envelopes(envelopes)
    assert len(result) == 1
    assert isinstance(result[0].event, ToolCallProgressEvent)
    assert len(result[0].event.items) == 2
    assert result[0].event.title == "second"


def test_merge_envelopes_different_tool_call_ids_not_merged() -> None:
    """ToolCallProgressEvents with different tool_call_id are not merged."""
    envelopes = [
        EventEnvelope(
            source_session_id="s1",
            event=ToolCallProgressEvent(
                tool_call_id="tc1",
                status="in_progress",
                items=[],
            ),
        ),
        EventEnvelope(
            source_session_id="s1",
            event=ToolCallProgressEvent(
                tool_call_id="tc2",
                status="in_progress",
                items=[],
            ),
        ),
    ]
    result = _merge_envelopes(envelopes)
    assert len(result) == 2


def test_merge_envelopes_different_status_not_merged() -> None:
    """ToolCallProgressEvents with different status are not merged."""
    envelopes = [
        EventEnvelope(
            source_session_id="s1",
            event=ToolCallProgressEvent(
                tool_call_id="tc1",
                status="in_progress",
                items=[],
            ),
        ),
        EventEnvelope(
            source_session_id="s1",
            event=ToolCallProgressEvent(
                tool_call_id="tc1",
                status="completed",
                items=[],
            ),
        ),
    ]
    result = _merge_envelopes(envelopes)
    assert len(result) == 2


def test_merge_envelopes_retains_event_type() -> None:
    """Merged events retain their original event type (no wrapper)."""
    envelopes = [
        EventEnvelope(source_session_id="s1", event=PartDeltaEvent.text(0, "a")),
        EventEnvelope(source_session_id="s1", event=PartDeltaEvent.text(0, "b")),
    ]
    result = _merge_envelopes(envelopes)
    assert len(result) == 1
    # The merged event should still be a PartDeltaEvent, not a wrapper
    assert type(result[0].event) is PartDeltaEvent


def test_merge_envelopes_non_consecutive_same_key_not_merged() -> None:
    """Events with same merge key but separated by different key are not merged."""
    envelopes = [
        EventEnvelope(source_session_id="s1", event=PartDeltaEvent.text(0, "a")),
        EventEnvelope(source_session_id="s1", event=PartDeltaEvent.thinking(0, "b")),
        EventEnvelope(source_session_id="s1", event=PartDeltaEvent.text(0, "c")),
    ]
    result = _merge_envelopes(envelopes)
    # Three groups: [text "a"], [thinking "b"], [text "c"]
    assert len(result) == 3
    assert isinstance(result[0].event.delta, TextPartDelta)
    assert result[0].event.delta.content_delta == "a"
    assert isinstance(result[1].event.delta, ThinkingPartDelta)
    assert result[1].event.delta.content_delta == "b"
    assert isinstance(result[2].event.delta, TextPartDelta)
    assert result[2].event.delta.content_delta == "c"


def test_merge_envelopes_preserves_source_session_id() -> None:
    """Merged envelopes preserve the source_session_id from the template."""
    envelopes = [
        EventEnvelope(source_session_id="custom-session", event=PartDeltaEvent.text(0, "a")),
        EventEnvelope(source_session_id="custom-session", event=PartDeltaEvent.text(0, "b")),
    ]
    result = _merge_envelopes(envelopes)
    assert len(result) == 1
    assert result[0].source_session_id == "custom-session"


# --- _rebind ---


def test_rebind_preserves_source_session_id() -> None:
    """_rebind creates new envelope with same source_session_id."""
    template = EventEnvelope(source_session_id="s1", event=PartDeltaEvent.text(0, "old"))
    new_event = PartDeltaEvent.text(0, "new")
    result = _rebind(template, new_event)
    assert result.source_session_id == "s1"
    assert result.event is new_event


def test_rebind_uses_new_event() -> None:
    """_rebind uses the provided new_event, not the template's event."""
    template = EventEnvelope(
        source_session_id="s1",
        event=RunStartedEvent(session_id="s", run_id="old"),
    )
    new_event = RunStartedEvent(session_id="s", run_id="new")
    result = _rebind(template, new_event)
    assert result.event is new_event
    assert result.event.run_id == "new"


def test_rebind_creates_new_envelope_instance() -> None:
    """_rebind returns a new EventEnvelope, not the template."""
    template = EventEnvelope(source_session_id="s1", event=PartDeltaEvent.text(0, "old"))
    new_event = PartDeltaEvent.text(0, "new")
    result = _rebind(template, new_event)
    assert result is not template


# ---------------------------------------------------------------------------
# Subscriber-side coalescing via drain_and_merge (Task 7)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_coalescing_type_change_flush() -> None:
    """Text deltas and thinking deltas are merged separately by drain_and_merge."""
    bus = EventBus(max_queue_size=100)
    stream = await bus.subscribe("sess-1")

    # Publish text deltas then thinking delta — all sent immediately via _send()
    await bus.publish("sess-1", PartDeltaEvent.text(0, "hello "))
    await bus.publish("sess-1", PartDeltaEvent.text(0, "world"))
    await bus.publish("sess-1", PartDeltaEvent.thinking(0, "thinking..."))
    await bus.close_session("sess-1")

    # Consumer drains and merges — text deltas merge into 1, thinking is separate
    results = [env async for env in drain_and_merge(stream)]
    assert len(results) == 2
    # First: merged text deltas
    assert isinstance(results[0].event, PartDeltaEvent)
    assert isinstance(results[0].event.delta, TextPartDelta)
    assert results[0].event.delta.content_delta == "hello world"
    # Second: thinking delta (single, no merge needed)
    assert isinstance(results[1].event, PartDeltaEvent)
    assert isinstance(results[1].event.delta, ThinkingPartDelta)
    assert results[1].event.delta.content_delta == "thinking..."


@pytest.mark.anyio
async def test_coalescing_immediate_event_drains_buffer() -> None:
    """Immediate event in drain batch delivered individually alongside merged batchable."""
    bus = EventBus(max_queue_size=100)
    stream = await bus.subscribe("sess-1")

    # Batchable events then immediate event — all sent immediately
    await bus.publish("sess-1", PartDeltaEvent.text(0, "hello "))
    await bus.publish("sess-1", PartDeltaEvent.text(0, "world"))
    await bus.publish(
        "sess-1", StreamCompleteEvent(message=ChatMessage(content="done", role="assistant"))
    )
    await bus.close_session("sess-1")

    # Consumer drains: text deltas merged, StreamCompleteEvent passthrough
    results = [env async for env in drain_and_merge(stream)]
    assert len(results) == 2
    # First: merged text deltas
    assert isinstance(results[0].event, PartDeltaEvent)
    assert isinstance(results[0].event.delta, TextPartDelta)
    assert results[0].event.delta.content_delta == "hello world"
    # Second: immediate event (passthrough, not merged with text)
    assert isinstance(results[1].event, StreamCompleteEvent)


@pytest.mark.anyio
async def test_coalescing_immediate_event_empty_buffer() -> None:
    """Immediate event with no batchable events is delivered individually."""
    bus = EventBus(max_queue_size=100)
    stream = await bus.subscribe("sess-1")

    await bus.publish(
        "sess-1", StreamCompleteEvent(message=ChatMessage(content="done", role="assistant"))
    )
    await bus.close_session("sess-1")

    results = [env async for env in drain_and_merge(stream)]
    assert len(results) == 1
    assert isinstance(results[0].event, StreamCompleteEvent)


@pytest.mark.anyio
async def test_coalescing_per_session_isolation() -> None:
    """Two sessions' consumers drain independently via drain_and_merge."""
    bus = EventBus(max_queue_size=100)
    stream_a = await bus.subscribe("sess-a")
    stream_b = await bus.subscribe("sess-b")

    # Publish text deltas to both sessions — all sent immediately
    await bus.publish("sess-a", PartDeltaEvent.text(0, "hello-a"))
    await bus.publish("sess-b", PartDeltaEvent.text(0, "hello-b"))
    await bus.close_session("sess-a")
    await bus.close_session("sess-b")

    # Each consumer drains independently
    results_a = [env async for env in drain_and_merge(stream_a)]
    results_b = [env async for env in drain_and_merge(stream_b)]

    assert len(results_a) == 1
    assert isinstance(results_a[0].event, PartDeltaEvent)
    assert results_a[0].event.delta.content_delta == "hello-a"  # type: ignore[union-attr]

    assert len(results_b) == 1
    assert isinstance(results_b[0].event, PartDeltaEvent)
    assert results_b[0].event.delta.content_delta == "hello-b"  # type: ignore[union-attr]


@pytest.mark.anyio
async def test_coalescing_passthrough_subagent_drains_buffer() -> None:
    """SubAgentEvent is passthrough, delivered individually by drain_and_merge."""
    bus = EventBus(max_queue_size=100)
    stream = await bus.subscribe("sess-1")

    await bus.publish("sess-1", PartDeltaEvent.text(0, "hello "))
    await bus.publish("sess-1", PartDeltaEvent.text(0, "world"))

    sub_event = SubAgentEvent(
        source_name="agent",
        source_type="agent",
        event=RunStartedEvent(session_id="s", run_id="r"),
    )
    await bus.publish("sess-1", sub_event)
    await bus.close_session("sess-1")

    results = [env async for env in drain_and_merge(stream)]
    assert len(results) == 2
    # First: merged text deltas
    assert isinstance(results[0].event, PartDeltaEvent)
    assert results[0].event.delta.content_delta == "hello world"  # type: ignore[union-attr]
    # Second: passthrough subagent event
    assert isinstance(results[1].event, SubAgentEvent)


@pytest.mark.anyio
async def test_coalescing_passthrough_custom_drains_buffer() -> None:
    """CustomEvent is passthrough, delivered individually by drain_and_merge."""
    bus = EventBus(max_queue_size=100)
    stream = await bus.subscribe("sess-1")

    await bus.publish("sess-1", PartDeltaEvent.text(0, "data"))
    await bus.publish("sess-1", CustomEvent(event_data="test"))
    await bus.close_session("sess-1")

    results = [env async for env in drain_and_merge(stream)]
    assert len(results) == 2
    # First: text delta
    assert isinstance(results[0].event, PartDeltaEvent)
    assert results[0].event.delta.content_delta == "data"  # type: ignore[union-attr]
    # Second: passthrough custom event
    assert isinstance(results[1].event, CustomEvent)


@pytest.mark.anyio
async def test_coalescing_passthrough_tool_result_metadata_drains_buffer() -> None:
    """ToolResultMetadataEvent is passthrough, delivered individually."""
    bus = EventBus(max_queue_size=100)
    stream = await bus.subscribe("sess-1")

    await bus.publish("sess-1", PartDeltaEvent.text(0, "data"))
    await bus.publish("sess-1", ToolResultMetadataEvent(tool_call_id="tc1", metadata={}))
    await bus.close_session("sess-1")

    results = [env async for env in drain_and_merge(stream)]
    assert len(results) == 2
    # First: text delta
    assert isinstance(results[0].event, PartDeltaEvent)
    assert results[0].event.delta.content_delta == "data"  # type: ignore[union-attr]
    # Second: passthrough tool result metadata
    assert isinstance(results[1].event, ToolResultMetadataEvent)


@pytest.mark.anyio
async def test_coalescing_non_consecutive_same_key_not_merged() -> None:
    """Non-consecutive same-key events are NOT merged (separated by different-type event)."""
    bus = EventBus(max_queue_size=100)
    stream = await bus.subscribe("sess-1")

    # text→thinking→text: the two text deltas are separated by thinking
    await bus.publish("sess-1", PartDeltaEvent.text(0, "a"))
    await bus.publish("sess-1", PartDeltaEvent.thinking(0, "b"))
    await bus.publish("sess-1", PartDeltaEvent.text(0, "c"))
    await bus.close_session("sess-1")

    # drain_and_merge groups consecutive same-key events
    # "a" is its own group (text), "b" is its own group (thinking), "c" is its own group (text)
    results = [env async for env in drain_and_merge(stream)]
    assert len(results) == 3
    assert isinstance(results[0].event.delta, TextPartDelta)
    assert results[0].event.delta.content_delta == "a"
    assert isinstance(results[1].event.delta, ThinkingPartDelta)
    assert results[1].event.delta.content_delta == "b"
    assert isinstance(results[2].event.delta, TextPartDelta)
    assert results[2].event.delta.content_delta == "c"


@pytest.mark.anyio
async def test_coalescing_none_delta_dropped() -> None:
    """PartDeltaEvent with delta=None is dropped by publish()."""
    bus = EventBus(max_queue_size=100)
    stream = await bus.subscribe("sess-1")

    none_delta: Any = PartDeltaEvent(index=0, delta=None)  # type: ignore[arg-type]
    await bus.publish("sess-1", none_delta)
    await bus.close_session("sess-1")

    # Stream closes immediately — no events to drain
    results = [env async for env in drain_and_merge(stream)]
    assert results == []


@pytest.mark.anyio
async def test_coalescing_plan_update_last_wins() -> None:
    """PlanUpdateEvent uses last-wins merge in drain batch."""
    bus = EventBus(max_queue_size=100)
    stream = await bus.subscribe("sess-1")

    # Publish multiple plan updates — all sent immediately
    await bus.publish("sess-1", PlanUpdateEvent(entries=[]))
    await bus.publish("sess-1", PlanUpdateEvent(entries=[]))
    await bus.publish("sess-1", PlanUpdateEvent(entries=[]))
    await bus.close_session("sess-1")

    # Consumer drains: plan updates merge to last-wins
    results = [env async for env in drain_and_merge(stream)]
    assert len(results) == 1
    assert isinstance(results[0].event, PlanUpdateEvent)


@pytest.mark.anyio
async def test_close_session_drains_buffer() -> None:
    """close_session closes streams; consumer drains remaining events via drain_and_merge."""
    bus = EventBus(max_queue_size=100)
    stream = await bus.subscribe("sess-1")

    # Publish text deltas — sent immediately
    await bus.publish("sess-1", PartDeltaEvent.text(0, "hello "))
    await bus.publish("sess-1", PartDeltaEvent.text(0, "world"))

    # Close session — closes send streams, consumer drains remaining events
    await bus.close_session("sess-1")

    results = [env async for env in drain_and_merge(stream)]
    assert len(results) == 1
    assert isinstance(results[0].event, PartDeltaEvent)
    assert results[0].event.delta.content_delta == "hello world"  # type: ignore[union-attr]


@pytest.mark.anyio
async def test_concurrent_publish_and_close_session_no_deadlock() -> None:
    """Concurrent publish() and close_session() complete without deadlock."""
    bus = EventBus(max_queue_size=100)
    _ = await bus.subscribe("sess-1")

    async def publish_loop() -> None:
        for i in range(10):
            await bus.publish("sess-1", PartDeltaEvent.text(0, f"chunk{i}"))

    async def close_loop() -> None:
        await bus.close_session("sess-1")

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:
            tg.start_soon(publish_loop)
            tg.start_soon(close_loop)


# ---------------------------------------------------------------------------
# Additional subscriber-side coalescing tests (Task 7)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_coalescing_100_consecutive_text_deltas() -> None:
    """100 consecutive text deltas merge into single event with no warning."""
    bus = EventBus(max_queue_size=200)
    stream = await bus.subscribe("sess-1")

    for i in range(100):
        await bus.publish("sess-1", PartDeltaEvent.text(0, f"chunk{i} "))
    await bus.close_session("sess-1")

    results = [env async for env in drain_and_merge(stream)]
    assert len(results) == 1
    assert isinstance(results[0].event, PartDeltaEvent)
    assert isinstance(results[0].event.delta, TextPartDelta)
    expected = "".join(f"chunk{i} " for i in range(100))
    assert results[0].event.delta.content_delta == expected


@pytest.mark.anyio
async def test_coalescing_lifecycle_alongside_batchable() -> None:
    """Lifecycle event in drain batch delivered individually alongside merged batchable."""
    bus = EventBus(max_queue_size=100)
    stream = await bus.subscribe("sess-1")

    await bus.publish("sess-1", PartDeltaEvent.text(0, "hello "))
    await bus.publish("sess-1", PartDeltaEvent.text(0, "world"))
    await bus.publish(
        "sess-1",
        ToolCallStartEvent(tool_name="bash", tool_call_id="tc1", title="Running bash"),
    )
    await bus.close_session("sess-1")

    results = [env async for env in drain_and_merge(stream)]
    assert len(results) == 2
    # First: merged text deltas
    assert isinstance(results[0].event, PartDeltaEvent)
    assert isinstance(results[0].event.delta, TextPartDelta)
    assert results[0].event.delta.content_delta == "hello world"
    # Second: lifecycle event (passthrough)
    assert isinstance(results[1].event, ToolCallStartEvent)


@pytest.mark.anyio
async def test_coalescing_passthrough_alongside_batchable() -> None:
    """Passthrough event in drain batch delivered individually alongside merged batchable."""
    bus = EventBus(max_queue_size=100)
    stream = await bus.subscribe("sess-1")

    await bus.publish("sess-1", PartDeltaEvent.text(0, "data1 "))
    await bus.publish("sess-1", PartDeltaEvent.text(0, "data2"))
    sub_event = SubAgentEvent(
        source_name="worker",
        source_type="agent",
        event=RunStartedEvent(session_id="s", run_id="r"),
    )
    await bus.publish("sess-1", sub_event)
    await bus.close_session("sess-1")

    results = [env async for env in drain_and_merge(stream)]
    assert len(results) == 2
    # First: merged text deltas
    assert isinstance(results[0].event, PartDeltaEvent)
    assert results[0].event.delta.content_delta == "data1 data2"  # type: ignore[union-attr]
    # Second: passthrough subagent event
    assert isinstance(results[1].event, SubAgentEvent)


@pytest.mark.anyio
async def test_coalescing_per_session_drain_isolation() -> None:
    """Two sessions' consumers drain independently with mixed event types."""
    bus = EventBus(max_queue_size=100)
    stream_a = await bus.subscribe("sess-a")
    stream_b = await bus.subscribe("sess-b")

    # Session A: text deltas + lifecycle event
    await bus.publish("sess-a", PartDeltaEvent.text(0, "a1 "))
    await bus.publish("sess-a", PartDeltaEvent.text(0, "a2"))
    await bus.publish("sess-a", RunStartedEvent(session_id="sess-a", run_id="r1"))

    # Session B: thinking deltas + lifecycle event
    await bus.publish("sess-b", PartDeltaEvent.thinking(0, "b1 "))
    await bus.publish("sess-b", PartDeltaEvent.thinking(0, "b2"))
    await bus.publish("sess-b", RunStartedEvent(session_id="sess-b", run_id="r2"))

    await bus.close_session("sess-a")
    await bus.close_session("sess-b")

    results_a = [env async for env in drain_and_merge(stream_a)]
    results_b = [env async for env in drain_and_merge(stream_b)]

    # Session A: merged text + lifecycle
    assert len(results_a) == 2
    assert isinstance(results_a[0].event, PartDeltaEvent)
    assert isinstance(results_a[0].event.delta, TextPartDelta)
    assert results_a[0].event.delta.content_delta == "a1 a2"
    assert isinstance(results_a[1].event, RunStartedEvent)

    # Session B: merged thinking + lifecycle
    assert len(results_b) == 2
    assert isinstance(results_b[0].event, PartDeltaEvent)
    assert isinstance(results_b[0].event.delta, ThinkingPartDelta)
    assert results_b[0].event.delta.content_delta == "b1 b2"
    assert isinstance(results_b[1].event, RunStartedEvent)


@pytest.mark.anyio
async def test_coalescing_plan_update_last_wins_in_drain() -> None:
    """PlanUpdateEvent last-wins merge in drain batch alongside other events."""
    bus = EventBus(max_queue_size=100)
    stream = await bus.subscribe("sess-1")

    # Publish plan updates interleaved with text deltas
    await bus.publish("sess-1", PartDeltaEvent.text(0, "text1 "))
    await bus.publish("sess-1", PlanUpdateEvent(entries=[]))
    await bus.publish("sess-1", PlanUpdateEvent(entries=[]))
    await bus.publish("sess-1", PartDeltaEvent.text(0, "text2"))
    await bus.close_session("sess-1")

    # drain_and_merge groups by consecutive merge_key:
    # text1 -> ("delta_text","") group, plan+plan -> ("plan","") group,
    # text2 -> ("delta_text","") group
    results = [env async for env in drain_and_merge(stream)]
    assert len(results) == 3
    # First: text1 (single text delta)
    assert isinstance(results[0].event, PartDeltaEvent)
    assert results[0].event.delta.content_delta == "text1 "  # type: ignore[union-attr]
    # Second: plan update (last-wins, 2 merged into 1)
    assert isinstance(results[1].event, PlanUpdateEvent)
    # Third: text2 (single text delta)
    assert isinstance(results[2].event, PartDeltaEvent)
    assert results[2].event.delta.content_delta == "text2"  # type: ignore[union-attr]


@pytest.mark.anyio
async def test_merge_helpers_callable_without_instance() -> None:
    """Merge helpers can be called directly without an EventBus instance."""
    # _merge_text_deltas
    text_events = [
        PartDeltaEvent.text(0, "hello "),
        PartDeltaEvent.text(0, "world"),
    ]
    merged_text = _merge_text_deltas(text_events)
    assert isinstance(merged_text.delta, TextPartDelta)
    assert merged_text.delta.content_delta == "hello world"

    # _merge_thinking_deltas
    thinking_events = [
        PartDeltaEvent.thinking(0, "think "),
        PartDeltaEvent.thinking(0, "ing"),
    ]
    merged_thinking = _merge_thinking_deltas(thinking_events)
    assert isinstance(merged_thinking.delta, ThinkingPartDelta)
    assert merged_thinking.delta.content_delta == "think ing"

    # _merge_tool_call_deltas
    tool_events = [
        PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta="arg1", tool_call_id="tc1")),
        PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta="arg2", tool_call_id="tc1")),
    ]
    merged_tool = _merge_tool_call_deltas(tool_events)
    assert isinstance(merged_tool.delta, ToolCallPartDelta)
    assert merged_tool.delta.args_delta == "arg1arg2"

    # _merge_envelopes
    envelopes = [
        EventEnvelope(source_session_id="s1", event=PartDeltaEvent.text(0, "a")),
        EventEnvelope(source_session_id="s1", event=PartDeltaEvent.text(0, "b")),
    ]
    merged_envs = _merge_envelopes(envelopes)
    assert len(merged_envs) == 1
    assert isinstance(merged_envs[0].event, PartDeltaEvent)
    assert merged_envs[0].event.delta.content_delta == "ab"  # type: ignore[union-attr]


@pytest.mark.anyio
async def test_coalescing_spawn_session_start_in_drain() -> None:
    """SpawnSessionStart is an immediate event that does not merge with batchable."""
    bus = EventBus(max_queue_size=100)
    stream = await bus.subscribe("sess-1")

    # Publish text deltas then SpawnSessionStart — all sent immediately
    await bus.publish("sess-1", PartDeltaEvent.text(0, "before "))
    await bus.publish("sess-1", PartDeltaEvent.text(0, "spawn"))
    spawn_event = SpawnSessionStart(
        child_session_id="sess-child",
        parent_session_id="sess-1",
        spawn_mechanism="spawn",
        source_name="worker",
        source_type="agent",
        description="Spawning worker agent",
    )
    await bus.publish("sess-1", spawn_event)
    await bus.close_session("sess-1")

    # drain_and_merge: text deltas merge, SpawnSessionStart is passthrough
    results = [env async for env in drain_and_merge(stream)]
    assert len(results) == 2
    # First: merged text deltas
    assert isinstance(results[0].event, PartDeltaEvent)
    assert isinstance(results[0].event.delta, TextPartDelta)
    assert results[0].event.delta.content_delta == "before spawn"
    # Second: SpawnSessionStart (passthrough/immediate, not merged)
    assert isinstance(results[1].event, SpawnSessionStart)


@pytest.mark.anyio
async def test_coalescing_noop_consumer_drains_queue() -> None:
    """When consumer skips processing, drain_and_merge still drains the queue."""
    bus = EventBus(max_queue_size=100)
    stream = await bus.subscribe("sess-1")

    # Publish events — all sent immediately via _send()
    await bus.publish("sess-1", PartDeltaEvent.text(0, "chunk0 "))
    await bus.publish("sess-1", PartDeltaEvent.text(0, "chunk1 "))
    await bus.publish("sess-1", PartDeltaEvent.text(0, "chunk2"))
    await bus.close_session("sess-1")

    # drain_and_merge consumes all items from the stream
    results = [env async for env in drain_and_merge(stream)]
    assert len(results) == 1
    assert results[0].event.delta.content_delta == "chunk0 chunk1 chunk2"

    # Stream is fully drained — no items remain
    remaining = await _drain_stream(stream)
    assert remaining == []


# ---------------------------------------------------------------------------
# drain_and_merge() tests
# ---------------------------------------------------------------------------


def _text_env(session: str, index: int, content: str) -> EventEnvelope:
    """Create an EventEnvelope wrapping a TextPartDelta event."""
    return EventEnvelope(
        source_session_id=session,
        event=PartDeltaEvent.text(index, content),
    )


def _thinking_env(session: str, index: int, content: str) -> EventEnvelope:
    """Create an EventEnvelope wrapping a ThinkingPartDelta event."""
    return EventEnvelope(
        source_session_id=session,
        event=PartDeltaEvent.thinking(index, content),
    )


async def test_drain_and_merge_consecutive_same_type_merges() -> None:
    """Consecutive same-type TextPartDelta events merge into 1 event."""
    recv: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
    recv.put_nowait(_text_env("s1", 0, "hello"))
    recv.put_nowait(_text_env("s1", 0, " "))
    recv.put_nowait(_text_env("s1", 0, "world"))
    recv.shutdown()

    results = [env async for env in drain_and_merge(recv)]

    assert len(results) == 1
    merged = results[0]
    assert isinstance(merged.event, PartDeltaEvent)
    assert isinstance(merged.event.delta, TextPartDelta)
    assert merged.event.delta.content_delta == "hello world"
    assert merged.source_session_id == "s1"


async def test_drain_and_merge_type_change_creates_separate_groups() -> None:
    """Type-change within a batch produces separate merged groups."""
    recv: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
    recv.put_nowait(_text_env("s1", 0, "foo"))
    recv.put_nowait(_text_env("s1", 0, "bar"))
    recv.put_nowait(_thinking_env("s1", 0, "think"))
    recv.put_nowait(_thinking_env("s1", 0, "ing"))
    recv.shutdown()

    results = [env async for env in drain_and_merge(recv)]

    assert len(results) == 2
    assert isinstance(results[0].event, PartDeltaEvent)
    assert isinstance(results[0].event.delta, TextPartDelta)
    assert results[0].event.delta.content_delta == "foobar"
    assert isinstance(results[1].event, PartDeltaEvent)
    assert isinstance(results[1].event.delta, ThinkingPartDelta)
    assert results[1].event.delta.content_delta == "thinking"


async def test_drain_and_merge_wouldblock_ends_batch() -> None:
    """QueueEmpty ends the current batch; merged result is yielded."""
    recv: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
    recv.put_nowait(_text_env("s1", 0, "x"))
    recv.put_nowait(_text_env("s1", 0, "y"))
    recv.put_nowait(_text_env("s1", 0, "z"))

    results: list[EventEnvelope] = []

    async def consumer() -> None:
        async for env in drain_and_merge(recv):
            results.append(env)
            recv.shutdown()

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:
            tg.start_soon(consumer)

    assert len(results) == 1
    assert isinstance(results[0].event, PartDeltaEvent)
    assert isinstance(results[0].event.delta, TextPartDelta)
    assert results[0].event.delta.content_delta == "xyz"


async def test_drain_and_merge_endofstream_mid_drain() -> None:
    """QueueShutDown mid-drain processes batch then terminates."""
    recv: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
    recv.put_nowait(_text_env("s1", 0, "a"))
    recv.put_nowait(_text_env("s1", 0, "b"))
    recv.put_nowait(_text_env("s1", 0, "c"))
    recv.shutdown()

    results = [env async for env in drain_and_merge(recv)]

    assert len(results) == 1
    assert isinstance(results[0].event, PartDeltaEvent)
    assert isinstance(results[0].event.delta, TextPartDelta)
    assert results[0].event.delta.content_delta == "abc"


async def test_drain_and_merge_endofstream_on_initial_receive() -> None:
    """QueueShutDown on initial receive terminates with no events yielded."""
    recv: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
    recv.shutdown()

    results = [env async for env in drain_and_merge(recv)]

    assert results == []


async def test_drain_and_merge_closed_resource_on_initial_receive() -> None:
    """QueueShutDown on initial receive terminates with no events."""
    recv: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
    recv.shutdown()

    results = [env async for env in drain_and_merge(recv)]

    assert results == []


async def test_drain_and_merge_raw_events_wrapped_and_merged() -> None:
    """Raw events (not EventEnvelope) are wrapped and merged correctly."""
    recv: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
    recv.put_nowait(PartDeltaEvent.text(0, "raw"))
    recv.put_nowait(PartDeltaEvent.text(0, "_"))
    recv.put_nowait(PartDeltaEvent.text(0, "event"))
    recv.shutdown()

    results = [env async for env in drain_and_merge(recv)]

    assert len(results) == 1
    merged = results[0]
    assert isinstance(merged, EventEnvelope)
    assert merged.source_session_id == ""
    assert isinstance(merged.event, PartDeltaEvent)
    assert isinstance(merged.event.delta, TextPartDelta)
    assert merged.event.delta.content_delta == "raw_event"


# ---------------------------------------------------------------------------
# Backpressure tests (merged from test_event_bus_backpressure.py)
#
# Verifies that the overflow policy (drop_oldest, drop_newest, drop_subscriber)
# works correctly with asyncio.Queue-based subscribers.
# ---------------------------------------------------------------------------


@pytest.fixture
def event_bus_bp() -> EventBus:
    """Return a fresh EventBus with small buffer for backpressure tests."""
    return EventBus(max_queue_size=5)


@pytest.fixture
def make_event() -> Any:
    """Return a factory for RunStartedEvent instances."""

    def _make(run_id: str = "run-1") -> RunStartedEvent:
        return RunStartedEvent(session_id="sess-bp", run_id=run_id)

    return _make


async def _drain_queue(queue: asyncio.Queue[Any]) -> list[Any]:
    """Drain all items from a queue until QueueShutDown."""
    items: list[Any] = []
    while True:
        try:
            items.append(await queue.get())
        except asyncio.QueueShutDown:
            break
    return items


async def test_backpressure_no_deadlock_with_slow_consumer(
    event_bus_bp: EventBus,
    make_event: Any,
) -> None:
    """Publishing 50 events to a slow consumer does not deadlock."""
    queue = await event_bus_bp.subscribe("sess-bp")

    async def publisher() -> None:
        for i in range(50):
            await event_bus_bp.publish(
                "sess-bp",
                RunStartedEvent(
                    session_id="sess-bp",
                    run_id=f"ev-{i}",
                ),
            )

    await publisher()

    # Shut down so drain terminates
    await event_bus_bp.close_session("sess-bp")
    received = await _drain_queue(queue)

    run_ids = [
        envelope.event.run_id
        for envelope in received
        if isinstance(envelope.event, RunStartedEvent)
    ]

    assert len(run_ids) > 0
    assert len(run_ids) <= 50


async def test_backpressure_retains_subscriber_with_drop_oldest(
    event_bus_bp: EventBus,
    make_event: Any,
) -> None:
    """A subscriber whose buffer is full survives with drop_oldest policy.

    With the default drop_oldest policy, the subscriber is NOT dropped —
    the oldest item is evicted to make room. This test verifies that
    the subscriber survives overflow.
    """
    await event_bus_bp.subscribe("sess-bp")

    for i in range(5):
        await event_bus_bp.publish(
            "sess-bp",
            RunStartedEvent(
                session_id="sess-bp",
                run_id=f"ev-{i}",
            ),
        )

    # Buffer is full (max_queue_size=5). With drop_oldest (default),
    # the subscriber survives — oldest item is evicted.
    await event_bus_bp.publish("sess-bp", make_event("overflow-1"))
    await event_bus_bp.publish("sess-bp", make_event("overflow-2"))

    counts = await event_bus_bp.get_subscriber_counts()
    # With drop_oldest (default), subscriber is NOT dropped.
    assert "sess-bp" in counts


async def test_subscribe_returns_asyncio_queue(
    event_bus_bp: EventBus,
) -> None:
    """subscribe() returns an asyncio.Queue."""
    queue = await event_bus_bp.subscribe("sess-bp")
    assert hasattr(queue, "get")
    assert hasattr(queue, "get_nowait")
    assert hasattr(queue, "shutdown")


async def test_unsubscribe_closes_queue(
    event_bus_bp: EventBus,
    make_event: Any,
) -> None:
    """unsubscribe() shuts down the queue, causing QueueShutDown on consumer."""
    queue = await event_bus_bp.subscribe("sess-bp")
    await event_bus_bp.publish("sess-bp", make_event("ev-1"))
    await event_bus_bp.unsubscribe("sess-bp", queue)

    received = await _drain_queue(queue)

    assert len(received) <= 1


async def test_close_session_signals_queue_shutdown(
    event_bus_bp: EventBus,
    make_event: Any,
) -> None:
    """close_session() shuts down all queues for the session."""
    queue = await event_bus_bp.subscribe("sess-bp")
    await event_bus_bp.publish("sess-bp", make_event("ev-1"))
    await event_bus_bp.close_session("sess-bp")

    received = await _drain_queue(queue)

    assert len(received) >= 0
    counts = await event_bus_bp.get_subscriber_counts()
    assert "sess-bp" not in counts


async def test_parallel_publishers_no_deadlock(
    event_bus_bp: EventBus,
) -> None:
    """Multiple concurrent publishers don't deadlock the EventBus."""
    queue = await event_bus_bp.subscribe("sess-bp")

    async def publisher(prefix: str) -> None:
        for i in range(20):
            await event_bus_bp.publish(
                "sess-bp",
                RunStartedEvent(
                    session_id="sess-bp",
                    run_id=f"{prefix}-{i}",
                ),
            )

    await asyncio.gather(
        publisher("a"),
        publisher("b"),
        publisher("c"),
    )

    await event_bus_bp.close_session("sess-bp")
    received = await _drain_queue(queue)

    run_ids = [
        envelope.event.run_id
        for envelope in received
        if isinstance(envelope.event, RunStartedEvent)
    ]

    assert len(run_ids) > 0
    assert len(run_ids) <= 60


async def test_replay_buffer_with_queue(
    event_bus_bp: EventBus,
    make_event: Any,
) -> None:
    """Replay buffer delivers historical events to new subscribers."""
    await event_bus_bp.publish("sess-bp", make_event("hist-1"))
    await event_bus_bp.publish("sess-bp", make_event("hist-2"))

    queue = await event_bus_bp.subscribe("sess-bp")

    received: list[str] = []
    try:
        with anyio.fail_after(0.5):
            while True:
                envelope = await queue.get()
                if isinstance(envelope.event, RunStartedEvent):
                    received.append(envelope.event.run_id)
    except TimeoutError:
        pass

    assert "hist-1" in received
    assert "hist-2" in received
