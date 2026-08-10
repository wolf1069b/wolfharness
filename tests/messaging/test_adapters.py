"""Tests for pydantic-graph adapters.

Consolidated from:
- test_signal_adapter.py (SignalEmittingGraphRun signal emission)
- test_streaming_adapter.py (Graph.iter() streaming adapter)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pydantic_graph.graph_builder import EndMarker, ErrorMarker, GraphTask
from pydantic_graph.id_types import ForkStack, NodeID, TaskID
import pytest

from wolfharness.agents.events import (
    PartStartEvent,
    RunErrorEvent,
    RunStartedEvent,
    StreamCompleteEvent,
    SubAgentEvent,
)
from wolfharness.messaging import ChatMessage
from wolfharness.messaging.messagenode import MessageNode
from wolfharness.messaging.messages import ChatMessage as Msg
from wolfharness.messaging.signal_adapter import SignalEmittingGraphRun
from wolfharness.messaging.streaming_adapter import (
    GraphStreamingAdapter,
    adapt_graph_run,
)
from wolfharness.talk import Talk


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence


# ============================================================================
# Shared mocks
# ============================================================================


class MockGraphRun:
    """Mock GraphRun that yields a configurable sequence of items."""

    def __init__(
        self,
        items: list[Sequence[GraphTask] | EndMarker[Any] | ErrorMarker],
        *,
        delay: float = 0.0,
    ) -> None:
        self._items = items
        self._index = 0
        self._delay = delay
        self.state = None

    def __aiter__(self) -> AsyncIterator[Sequence[GraphTask] | EndMarker[Any] | ErrorMarker]:
        return self

    async def __anext__(self) -> Sequence[GraphTask] | EndMarker[Any] | ErrorMarker:
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        return item


def _make_task(node_id: str, inputs: Any = None, task_id_offset: int = 0) -> GraphTask:
    return GraphTask(
        node_id=NodeID(node_id),
        inputs=inputs,
        fork_stack=ForkStack(()),
        task_id=TaskID(f"task:{node_id}:{task_id_offset}"),
    )


class DummyMessageNode(MessageNode[Any, Any]):
    """Minimal concrete MessageNode for signal capture."""

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.received: list[ChatMessage[Any]] = []
        self.sent: list[ChatMessage[Any]] = []
        self.message_received.connect(self._on_received)
        self.message_sent.connect(self._on_sent)

    async def run(self, *prompts: Any, **kwargs: Any) -> ChatMessage[Any]:
        return ChatMessage(content="ok", role="assistant")

    async def get_stats(self) -> Any:
        return None

    async def _empty_iter(self) -> AsyncIterator[ChatMessage[Any]]:
        if False:
            yield ChatMessage(content="", role="assistant")
        return

    def run_iter(self, *prompts: Any, **kwargs: Any) -> AsyncIterator[ChatMessage[Any]]:
        return self._empty_iter()

    def get_context(self, data: Any = None, input_provider: Any = None) -> Any:
        return None

    def _on_received(self, message: ChatMessage[Any]) -> None:
        self.received.append(message)

    def _on_sent(self, message: ChatMessage[Any]) -> None:
        self.sent.append(message)


class DummyTalk(Talk[Any]):
    """Minimal Talk subclass that records signal emissions."""

    def __init__(self, source: MessageNode[Any, Any], target: MessageNode[Any, Any]) -> None:
        super().__init__(source=source, targets=[target])
        self.forwarded: list[ChatMessage[Any]] = []
        self.processed: list[Talk.ConnectionProcessed] = []
        self.message_forwarded.connect(self._on_forwarded)
        self.connection_processed.connect(self._on_processed)

    def _on_forwarded(self, message: ChatMessage[Any]) -> None:
        self.forwarded.append(message)

    def _on_processed(self, event: Talk.ConnectionProcessed) -> None:
        self.processed.append(event)


# ============================================================================
# Signal adapter tests
# ============================================================================


@pytest.mark.anyio
async def test_message_received_before_step():
    """message_received is emitted when a GraphTask is first yielded."""
    node_a = DummyMessageNode("node_a")
    run = MockGraphRun([
        [_make_task("node_a", inputs="hello")],
        EndMarker("done"),
    ])

    adapter = SignalEmittingGraphRun(
        graph_run=run,  # type: ignore[arg-type]
        node_mapping={NodeID("node_a"): node_a},
    )

    _ = [event async for event in adapter]

    assert len(node_a.received) == 1
    assert node_a.received[0].content == "hello"
    assert node_a.received[0].role == "user"


@pytest.mark.anyio
async def test_message_sent_after_step():
    """message_sent is emitted on the next yield after a task was seen."""
    node_a = DummyMessageNode("node_a")
    run = MockGraphRun([
        [_make_task("node_a", inputs="hello")],
        EndMarker("done"),
    ])

    adapter = SignalEmittingGraphRun(
        graph_run=run,  # type: ignore[arg-type]
        node_mapping={NodeID("node_a"): node_a},
    )

    async for _ in adapter:
        pass

    assert len(node_a.sent) == 1
    assert node_a.sent[0].role == "assistant"


@pytest.mark.anyio
async def test_two_step_chain_signals():
    """A 2-step chain emits received/sent in the correct order."""
    node_a = DummyMessageNode("node_a")
    node_b = DummyMessageNode("node_b")

    run = MockGraphRun([
        [_make_task("node_a", inputs="step_a_input")],
        [_make_task("node_b", inputs="step_b_input")],
        EndMarker("final"),
    ])

    adapter = SignalEmittingGraphRun(
        graph_run=run,  # type: ignore[arg-type]
        node_mapping={
            NodeID("node_a"): node_a,
            NodeID("node_b"): node_b,
        },
    )

    async for _ in adapter:
        pass

    assert len(node_a.received) == 1
    assert len(node_a.sent) == 1
    assert len(node_b.received) == 1
    assert len(node_b.sent) == 1

    timeline: list[tuple[str, str, str]] = []
    timeline.extend(("node_a", "received", msg.content) for msg in node_a.received)
    timeline.extend(("node_a", "sent", msg.content) for msg in node_a.sent)
    timeline.extend(("node_b", "received", msg.content) for msg in node_b.received)
    timeline.extend(("node_b", "sent", msg.content) for msg in node_b.sent)

    expected = [
        ("node_a", "received", "step_a_input"),
        ("node_a", "sent", "step_a_input"),
        ("node_b", "received", "step_b_input"),
        ("node_b", "sent", "step_b_input"),
    ]
    assert timeline == expected


@pytest.mark.anyio
async def test_connection_processed_on_edge():
    """connection_processed is emitted when an edge traversal is detected."""
    node_a = DummyMessageNode("node_a")
    node_b = DummyMessageNode("node_b")
    talk_ab = DummyTalk(source=node_a, target=node_b)

    run = MockGraphRun([
        [_make_task("node_a", inputs="hello")],
        [_make_task("node_b", inputs="world")],
        EndMarker("done"),
    ])

    adapter = SignalEmittingGraphRun(
        graph_run=run,  # type: ignore[arg-type]
        node_mapping={
            NodeID("node_a"): node_a,
            NodeID("node_b"): node_b,
        },
        talk_mapping={
            (NodeID("node_a"), NodeID("node_b")): talk_ab,
        },
    )

    async for _ in adapter:
        pass

    assert len(talk_ab.processed) == 1
    event = talk_ab.processed[0]
    assert event.source == node_a
    assert event.targets == [node_b]
    assert event.message.content == "hello"
    assert event.connection_type == "run"
    assert not event.queued


@pytest.mark.anyio
async def test_message_forwarded_on_edge():
    """message_forwarded is emitted alongside connection_processed."""
    node_a = DummyMessageNode("node_a")
    node_b = DummyMessageNode("node_b")
    talk_ab = DummyTalk(source=node_a, target=node_b)

    run = MockGraphRun([
        [_make_task("node_a", inputs="hello")],
        [_make_task("node_b", inputs="world")],
        EndMarker("done"),
    ])

    adapter = SignalEmittingGraphRun(
        graph_run=run,  # type: ignore[arg-type]
        node_mapping={
            NodeID("node_a"): node_a,
            NodeID("node_b"): node_b,
        },
        talk_mapping={
            (NodeID("node_a"), NodeID("node_b")): talk_ab,
        },
    )

    async for _ in adapter:
        pass

    assert len(talk_ab.forwarded) == 1
    assert talk_ab.forwarded[0].content == "hello"


@pytest.mark.anyio
async def test_signal_session_id_injected():
    """ChatMessage payloads carry the configured session_id."""
    node_a = DummyMessageNode("node_a")
    run = MockGraphRun([
        [_make_task("node_a", inputs="hello")],
        EndMarker("done"),
    ])

    adapter = SignalEmittingGraphRun(
        graph_run=run,  # type: ignore[arg-type]
        node_mapping={NodeID("node_a"): node_a},
        session_id="sess-123",
    )

    async for _ in adapter:
        pass

    assert node_a.received[0].session_id == "sess-123"
    assert node_a.sent[0].session_id == "sess-123"


@pytest.mark.anyio
async def test_unmapped_node_id_skipped_gracefully():
    """Nodes not present in node_mapping are silently skipped."""
    node_a = DummyMessageNode("node_a")
    run = MockGraphRun([
        [_make_task("unknown_node", inputs="hello")],
        EndMarker("done"),
    ])

    adapter = SignalEmittingGraphRun(
        graph_run=run,  # type: ignore[arg-type]
        node_mapping={NodeID("node_a"): node_a},
    )

    async for _ in adapter:
        pass

    assert len(node_a.received) == 0
    assert len(node_a.sent) == 0


@pytest.mark.anyio
async def test_parallel_execution_signals():
    """Parallel tasks emit received/sent for each branch."""
    node_a = DummyMessageNode("node_a")
    node_b = DummyMessageNode("node_b")

    run = MockGraphRun([
        [_make_task("node_a", inputs="a"), _make_task("node_b", inputs="b")],
        EndMarker("done"),
    ])

    adapter = SignalEmittingGraphRun(
        graph_run=run,  # type: ignore[arg-type]
        node_mapping={
            NodeID("node_a"): node_a,
            NodeID("node_b"): node_b,
        },
    )

    async for _ in adapter:
        pass

    assert len(node_a.received) == 1
    assert len(node_a.sent) == 1
    assert len(node_b.received) == 1
    assert len(node_b.sent) == 1


@pytest.mark.anyio
async def test_is_completed_property():
    """is_completed becomes True after EndMarker is yielded."""
    run = MockGraphRun([EndMarker("done")])

    adapter = SignalEmittingGraphRun(
        graph_run=run,  # type: ignore[arg-type]
        node_mapping={},
    )

    assert not adapter.is_completed
    async for _ in adapter:
        pass
    assert adapter.is_completed


@pytest.mark.anyio
async def test_graph_run_property():
    """graph_run exposes the underlying GraphRun instance."""
    run = MockGraphRun([EndMarker("done")])

    adapter = SignalEmittingGraphRun(
        graph_run=run,  # type: ignore[arg-type]
        node_mapping={},
    )

    assert adapter.graph_run is run


# ============================================================================
# Streaming adapter tests
# ============================================================================


@pytest.mark.anyio
async def test_run_started_event():
    """Adapter always yields RunStartedEvent first."""
    run = MockGraphRun([EndMarker("done")])
    adapter = GraphStreamingAdapter(
        run,
        session_id="sess-1",
        agent_name="test-agent",
    )

    events = [e async for e in adapter]

    assert len(events) >= 1
    assert isinstance(events[0], RunStartedEvent)
    assert events[0].session_id == "sess-1"
    assert events[0].agent_name == "test-agent"


@pytest.mark.anyio
async def test_graph_task_to_part_start():
    """GraphTask yields map to PartStartEvent."""
    run = MockGraphRun([
        [_make_task("step_a", task_id_offset=0)],
        [_make_task("step_b", task_id_offset=1)],
        EndMarker("done"),
    ])
    adapter = GraphStreamingAdapter(
        run,
        session_id="sess-1",
        agent_name="test-agent",
    )

    events = [e async for e in adapter]

    part_starts = [e for e in events if isinstance(e, PartStartEvent)]
    assert len(part_starts) == 2
    assert part_starts[0].index == 0
    assert part_starts[1].index == 0


@pytest.mark.anyio
async def test_end_marker_to_stream_complete():
    """EndMarker yields map to StreamCompleteEvent."""
    run = MockGraphRun([EndMarker("final result")])
    adapter = GraphStreamingAdapter(
        run,
        session_id="sess-1",
        agent_name="test-agent",
        message_id="msg-1",
    )

    events = [e async for e in adapter]

    complete_events = [e for e in events if isinstance(e, StreamCompleteEvent)]
    assert len(complete_events) == 1
    assert complete_events[0].message.content == "final result"
    assert complete_events[0].message.session_id == "sess-1"
    assert complete_events[0].message.name == "test-agent"
    assert complete_events[0].message.message_id == "msg-1"


@pytest.mark.anyio
async def test_error_marker_raises():
    """ErrorMarker yields RunErrorEvent and re-raises the exception."""
    original_error = ValueError("boom")
    run = MockGraphRun([ErrorMarker(original_error)])
    adapter = GraphStreamingAdapter(
        run,
        session_id="sess-1",
        agent_name="test-agent",
    )

    events: list[Any] = []
    with pytest.raises(ValueError, match="boom"):  # noqa: PT012
        async for event in adapter:
            events.append(event)  # noqa: PERF401

    error_events = [e for e in events if isinstance(e, RunErrorEvent)]
    assert len(error_events) == 1
    assert error_events[0].message == "boom"
    assert error_events[0].agent_name == "test-agent"


@pytest.mark.anyio
async def test_step_event_collector_flat():
    """StepEventCollector emits events directly when depth is 0."""
    run = MockGraphRun([EndMarker("done")], delay=0.1)
    adapter = GraphStreamingAdapter(
        run,
        session_id="sess-1",
        agent_name="test-agent",
    )
    collector = adapter.create_collector("my_step", depth=0)

    async def emit_while_running() -> None:
        await asyncio.sleep(0.02)
        await collector.emit_text_delta(0, "hello")
        await collector.emit_text_delta(1, " world")

    _task = asyncio.create_task(emit_while_running())  # noqa: RUF006
    events = [e async for e in adapter]

    deltas = [e for e in events if hasattr(e, "delta")]
    assert len(deltas) == 2


@pytest.mark.anyio
async def test_step_event_collector_nested():
    """StepEventCollector wraps events in SubAgentEvent when depth > 0."""
    run = MockGraphRun([EndMarker("done")], delay=0.1)
    adapter = GraphStreamingAdapter(
        run,
        session_id="sess-1",
        agent_name="test-agent",
    )
    collector = adapter.create_collector("sub_agent", depth=1)

    async def emit_while_running() -> None:
        await asyncio.sleep(0.02)
        await collector.emit_text_delta(0, "nested text")

    _task = asyncio.create_task(emit_while_running())  # noqa: RUF006
    events = [e async for e in adapter]

    subagent_events = [e for e in events if isinstance(e, SubAgentEvent)]
    assert len(subagent_events) == 1
    assert subagent_events[0].source_name == "sub_agent"
    assert subagent_events[0].depth == 1


@pytest.mark.anyio
async def test_adapt_graph_run_convenience():
    """adapt_graph_run() yields the same events as the adapter class."""
    run = MockGraphRun([
        [_make_task("step_1", task_id_offset=0)],
        EndMarker("result"),
    ])

    events = [
        e
        async for e in adapt_graph_run(
            run,
            session_id="sess-2",
            agent_name="conv-agent",
        )
    ]

    assert any(isinstance(e, RunStartedEvent) for e in events)
    assert any(isinstance(e, PartStartEvent) for e in events)
    assert any(isinstance(e, StreamCompleteEvent) for e in events)


@pytest.mark.anyio
async def test_event_ordering():
    """Events are yielded in the order they are produced."""
    sync_queue: asyncio.Queue[str] = asyncio.Queue()

    class CoordinatedMockGraphRun:
        """Mock that waits for collector before yielding step_2."""

        def __init__(self) -> None:
            self._items = [
                [_make_task("step_1", task_id_offset=0)],
                [_make_task("step_2", task_id_offset=1)],
                EndMarker("done"),
            ]
            self._index = 0

        def __aiter__(self) -> AsyncIterator[Any]:
            return self

        async def __anext__(self) -> Any:
            if self._index >= len(self._items):
                raise StopAsyncIteration
            item = self._items[self._index]
            self._index += 1
            if self._index == 2:
                await sync_queue.get()
            return item

    run = CoordinatedMockGraphRun()
    adapter = GraphStreamingAdapter(
        run,
        session_id="sess-1",
        agent_name="test-agent",
    )
    collector = adapter.create_collector("step_1", depth=0)

    async def emit_after_step_1() -> None:
        await asyncio.sleep(0.02)
        await collector.emit_text_delta(0, "chunk")
        await sync_queue.put("done")

    _task = asyncio.create_task(emit_after_step_1())  # noqa: RUF006
    events = [e async for e in adapter]

    kinds = [type(e).__name__ for e in events]
    assert kinds[0] == "RunStartedEvent"
    assert kinds[-1] == "StreamCompleteEvent"
    part_start_indices = [i for i, e in enumerate(events) if isinstance(e, PartStartEvent)]
    delta_indices = [i for i, e in enumerate(events) if hasattr(e, "delta")]
    assert len(delta_indices) > 0, "Expected at least one delta event"
    assert part_start_indices[0] < delta_indices[0]


@pytest.mark.anyio
async def test_user_msg_parent_id():
    """StreamCompleteEvent carries parent_id from user_msg."""
    user_msg = Msg(content="hello", role="user", message_id="user-1")
    run = MockGraphRun([EndMarker("done")])
    adapter = GraphStreamingAdapter(
        run,
        session_id="sess-1",
        agent_name="test-agent",
        user_msg=user_msg,
    )

    events = [e async for e in adapter]
    complete = next(e for e in events if isinstance(e, StreamCompleteEvent))
    assert complete.message.parent_id == "user-1"


@pytest.mark.anyio
async def test_cancellation():
    """Adapter cancels cleanly when consumer breaks early."""
    run = MockGraphRun([
        [_make_task("step_a", task_id_offset=0)],
        [_make_task("step_b", task_id_offset=1)],
        EndMarker("done"),
    ])
    adapter = GraphStreamingAdapter(
        run,
        session_id="sess-1",
        agent_name="test-agent",
    )

    events: list[Any] = []
    async for event in adapter:
        events.append(event)
        if len(events) >= 2:
            break

    assert len(events) == 2
