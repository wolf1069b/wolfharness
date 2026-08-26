"""Unit tests for ProtocolEventConsumerMixin.

Tests consumer lifecycle, event dispatch, graceful shutdown,
and hook invocation for the mixin.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

from pydantic_ai import TextPartDelta
import pytest

from wolfharness.agents.events.events import (
    PartDeltaEvent,
    RunErrorEvent,
    SpawnSessionStart,
)
from wolfharness.orchestrator.core import EventBus, EventEnvelope
from wolfharness_server.mixins import (
    ConsumerShutdown,
    ProtocolEventConsumerMixin,
)


pytestmark = [pytest.mark.unit, pytest.mark.anyio]


class _TestConsumer(ProtocolEventConsumerMixin):
    """Concrete test subclass that records all hook calls."""

    def __init__(self, event_bus: Any) -> None:
        super().__init__()
        self._event_bus = event_bus
        self.handle_event_calls: list[tuple[str, Any]] = []
        self.before_loop_calls: list[str] = []
        self.after_loop_calls: list[str] = []
        self.spawn_session_start_calls: list[tuple[str, SpawnSessionStart]] = []

    @property
    def event_bus(self) -> Any:
        return self._event_bus

    async def _handle_event(self, session_id: str, event: Any) -> None:
        self.handle_event_calls.append((session_id, event))

    async def _before_consumer_loop(self, session_id: str) -> None:
        self.before_loop_calls.append(session_id)

    async def _after_consumer_loop(self, session_id: str) -> None:
        self.after_loop_calls.append(session_id)

    async def _on_spawn_session_start(self, session_id: str, event: EventEnvelope) -> None:
        assert isinstance(event.event, SpawnSessionStart)
        self.spawn_session_start_calls.append((session_id, event.event))


def _make_queue_and_mock_subscribe(
    mock_event_bus: AsyncMock,
) -> asyncio.Queue[EventEnvelope]:
    """Create a queue and wire subscribe to return it."""
    queue: asyncio.Queue[EventEnvelope] = asyncio.Queue(maxsize=100)
    mock_event_bus.subscribe = AsyncMock(return_value=queue)
    return queue


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """Return a mock EventBus with async subscribe/unsubscribe."""
    bus = AsyncMock(spec=EventBus)
    bus.unsubscribe = AsyncMock(return_value=None)
    return bus


@pytest.fixture
def mock_event_bus_with_queue(
    mock_event_bus: AsyncMock,
) -> tuple[AsyncMock, asyncio.Queue[EventEnvelope]]:
    """Return a mock EventBus with a real asyncio.Queue."""
    queue: asyncio.Queue[EventEnvelope] = asyncio.Queue(maxsize=100)
    mock_event_bus.subscribe = AsyncMock(return_value=queue)
    return mock_event_bus, queue


@pytest.mark.anyio
async def test_start_consumer_subscribes_and_runs_loop(
    mock_event_bus: AsyncMock,
) -> None:
    """Verify EventBus subscription and consumer task creation."""
    _make_queue_and_mock_subscribe(mock_event_bus)
    consumer = _TestConsumer(mock_event_bus)
    await consumer.start_event_consumer("sess-1")

    assert "sess-1" in consumer._session_groups
    assert "sess-1" in consumer._consumer_streams

    mock_event_bus.subscribe.assert_awaited_once_with(
        "sess-1", scope="descendants", replay=True, exclude_source=None
    )

    await consumer.stop_event_consumer("sess-1")


@pytest.mark.anyio
async def test_start_consumer_is_idempotent(
    mock_event_bus: AsyncMock,
) -> None:
    """Calling start_event_consumer twice does not create duplicate tasks."""
    _make_queue_and_mock_subscribe(mock_event_bus)
    consumer = _TestConsumer(mock_event_bus)
    await consumer.start_event_consumer("sess-1")

    await consumer.start_event_consumer("sess-1")

    assert len(consumer._session_groups) == 1
    assert "sess-1" in consumer._session_groups

    await consumer.stop_event_consumer("sess-1")


@pytest.mark.anyio
async def test_start_consumer_is_threadsafe(
    mock_event_bus: AsyncMock,
) -> None:
    """Concurrent calls for the same session are serialized."""
    _make_queue_and_mock_subscribe(mock_event_bus)
    consumer = _TestConsumer(mock_event_bus)

    async def start() -> None:
        await consumer.start_event_consumer("sess-1")

    await asyncio.gather(start(), start())

    assert len(consumer._session_groups) == 1

    await consumer.stop_event_consumer("sess-1")


@pytest.mark.anyio
async def test_stop_consumer_cancels_task_and_unsubscribes(
    mock_event_bus: AsyncMock,
) -> None:
    """Stopping a consumer cancels the task and unsubscribes from EventBus."""
    _make_queue_and_mock_subscribe(mock_event_bus)
    consumer = _TestConsumer(mock_event_bus)
    await consumer.start_event_consumer("sess-1")

    await consumer.stop_event_consumer("sess-1")

    assert "sess-1" not in consumer._session_groups
    assert mock_event_bus.unsubscribe.await_count >= 1


@pytest.mark.anyio
async def test_stop_consumer_is_safe_when_not_running(mock_event_bus: AsyncMock) -> None:
    """Calling stop_event_consumer without starting should not raise."""
    consumer = _TestConsumer(mock_event_bus)
    await consumer.stop_event_consumer("sess-1")


@pytest.mark.anyio
async def test_handle_event_dispatches_to_subclass(mock_event_bus: AsyncMock) -> None:
    """Verify abstract _handle_event is called with correct arguments."""
    queue = _make_queue_and_mock_subscribe(mock_event_bus)

    consumer = _TestConsumer(mock_event_bus)
    mock_handle = AsyncMock()
    consumer._handle_event = mock_handle  # type: ignore[method-assign]

    event = PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="hello"))
    envelope = EventEnvelope(source_session_id="sess-1", event=event)
    await queue.put(envelope)

    await consumer.start_event_consumer("sess-1")

    for _ in range(100):
        if mock_handle.called:
            break
        await asyncio.sleep(0.01)

    mock_handle.assert_awaited_once_with("sess-1", envelope)

    await consumer.stop_event_consumer("sess-1")


class _NoReplayConsumer(_TestConsumer):
    """Consumer that opts out of EventBus replay (OpenCode-style policy)."""

    def _get_subscription_replay(self) -> bool:
        return False

    def _get_subscription_exclude_source(self) -> frozenset[str] | None:
        return frozenset({"opencode_event_bridge"})


@pytest.mark.anyio
async def test_session_consumer_subscribes_replay_false(mock_event_bus: AsyncMock) -> None:
    """OpenCode-style consumer subscribes with replay=False + exclude_source."""
    _make_queue_and_mock_subscribe(mock_event_bus)
    consumer = _NoReplayConsumer(mock_event_bus)
    await consumer.start_event_consumer("sess-1")

    mock_event_bus.subscribe.assert_awaited_once_with(
        "sess-1",
        scope="descendants",
        replay=False,
        exclude_source=frozenset({"opencode_event_bridge"}),
    )

    await consumer.stop_event_consumer("sess-1")


class _RecoveryReplayConsumer(_TestConsumer):
    """Consumer that needs historical events (crash-recovery opt-in)."""

    def _get_subscription_replay(self) -> bool:
        return True


@pytest.mark.anyio
async def test_recovery_path_opt_in_replay_true(mock_event_bus: AsyncMock) -> None:
    """Recovery consumers explicitly opt back in to replay=True."""
    _make_queue_and_mock_subscribe(mock_event_bus)
    consumer = _RecoveryReplayConsumer(mock_event_bus)
    await consumer.start_event_consumer("sess-1")

    mock_event_bus.subscribe.assert_awaited_once_with(
        "sess-1", scope="descendants", replay=True, exclude_source=None
    )

    await consumer.stop_event_consumer("sess-1")


@pytest.mark.anyio
async def test_consumer_shutdown_gracefully_stops_loop(
    mock_event_bus: AsyncMock,
) -> None:
    """_handle_event raises ConsumerShutdown, loop exits gracefully."""
    queue = _make_queue_and_mock_subscribe(mock_event_bus)

    consumer = _TestConsumer(mock_event_bus)
    consumer._handle_event = AsyncMock(  # type: ignore[method-assign]
        side_effect=ConsumerShutdown()
    )

    event = RunErrorEvent(message="shutdown-test")
    envelope = EventEnvelope(source_session_id="sess-1", event=event)
    await queue.put(envelope)

    await consumer.start_event_consumer("sess-1")

    for _ in range(100):
        if "sess-1" in consumer.after_loop_calls:
            break
        await asyncio.sleep(0.01)

    assert "sess-1" in consumer.after_loop_calls

    await consumer.stop_event_consumer("sess-1")
    mock_event_bus.unsubscribe.assert_awaited()


@pytest.mark.anyio
async def test_unhandled_exception_propagates(
    mock_event_bus: AsyncMock,
) -> None:
    """_handle_event raises generic Exception; after hook runs."""
    queue = _make_queue_and_mock_subscribe(mock_event_bus)

    consumer = _TestConsumer(mock_event_bus)
    consumer._handle_event = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("boom")
    )

    event = RunErrorEvent(message="boom-test")
    envelope = EventEnvelope(source_session_id="sess-1", event=event)
    await queue.put(envelope)

    await consumer.start_event_consumer("sess-1")

    for _ in range(100):
        if "sess-1" in consumer.after_loop_calls:
            break
        await asyncio.sleep(0.01)

    assert "sess-1" in consumer.after_loop_calls


@pytest.mark.anyio
async def test_cancelled_error_reraised_after_cleanup(
    mock_event_bus: AsyncMock,
) -> None:
    """Cancel the consumer task mid-loop; CancelledError propagates, cleanup runs."""
    _make_queue_and_mock_subscribe(mock_event_bus)
    consumer = _TestConsumer(mock_event_bus)
    await consumer.start_event_consumer("sess-1")
    await asyncio.sleep(0.01)

    await consumer.stop_event_consumer("sess-1")

    assert "sess-1" not in consumer._session_groups
    mock_event_bus.unsubscribe.assert_awaited()


@pytest.mark.anyio
async def test_end_of_stream_stops_loop(mock_event_bus: AsyncMock) -> None:
    """Close send stream; loop exits gracefully and after hook runs."""
    queue = _make_queue_and_mock_subscribe(mock_event_bus)

    consumer = _TestConsumer(mock_event_bus)

    await consumer.start_event_consumer("sess-1")
    await asyncio.sleep(0.01)

    queue.shutdown()

    for _ in range(100):
        if "sess-1" in consumer.after_loop_calls:
            break
        await asyncio.sleep(0.01)

    assert "sess-1" in consumer.after_loop_calls


@pytest.mark.anyio
async def test_spawn_session_start_calls_hook(mock_event_bus: AsyncMock) -> None:
    """SpawnSessionStart triggers _on_spawn_session_start then _handle_event."""
    queue = _make_queue_and_mock_subscribe(mock_event_bus)

    consumer = _TestConsumer(mock_event_bus)

    event = SpawnSessionStart(
        child_session_id="child-1",
        parent_session_id="sess-1",
        source_name="test-agent",
        source_type="agent",
        depth=1,
        description="test spawn",
        spawn_mechanism="spawn",
    )
    envelope = EventEnvelope(source_session_id="sess-1", event=event)
    await queue.put(envelope)
    queue.shutdown()

    await consumer.start_event_consumer("sess-1")

    for _ in range(100):
        if len(consumer.spawn_session_start_calls) > 0:
            break
        await asyncio.sleep(0.01)

    assert len(consumer.spawn_session_start_calls) == 1
    assert consumer.spawn_session_start_calls[0] == ("sess-1", event)
    assert consumer.handle_event_calls == [("sess-1", envelope)]


@pytest.mark.anyio
async def test_before_after_hooks_called_in_order(mock_event_bus: AsyncMock) -> None:
    """_before_consumer_loop runs before loop, _after_consumer_loop after exit."""
    queue = _make_queue_and_mock_subscribe(mock_event_bus)

    consumer = _TestConsumer(mock_event_bus)

    await consumer.start_event_consumer("sess-1")
    await asyncio.sleep(0.05)

    assert consumer.before_loop_calls == ["sess-1"]

    queue.shutdown()

    for _ in range(100):
        if "sess-1" in consumer.after_loop_calls:
            break
        await asyncio.sleep(0.01)

    assert "sess-1" in consumer.after_loop_calls

    await consumer.stop_event_consumer("sess-1")
