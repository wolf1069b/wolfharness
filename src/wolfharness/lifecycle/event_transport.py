"""EventTransport implementation: InProcessTransport.

The EventTransport dimension abstracts the wire protocol between RunLoop
and external consumers. ``InProcessTransport`` is the default implementation
using in-process ``asyncio.Queue`` — it requires zero infrastructure and
passes Python objects directly (no serialization).

For MQ-backed or gRPC transports, see future milestones beyond M2.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from wolfharness.lifecycle.types import EventEnvelope


class InProcessTransport:
    """In-process EventTransport using per-subscriber ``asyncio.Queue``.

    Each subscriber gets its own queue, ensuring that every subscriber
    receives all events published to the topic (broadcast semantics).
    Events published before any subscriber exists are buffered in a
    per-topic backlog and delivered to the first subscriber.

    An optional replay buffer allows late subscribers to receive
    previously published events.

    Attributes:
        _replay_buffer_size: Maximum number of events retained per topic.
            ``0`` disables the replay buffer entirely.
        _subscribers: Per-topic set of subscriber queues for broadcasting.
        _replay_buffers: Per-topic lists of retained envelopes (only
            populated when ``_replay_buffer_size > 0``).
        _backlog: Per-topic lists of events published before any
            subscriber was registered.
        _closed: Whether ``close()`` has been called.
    """

    def __init__(self, replay_buffer_size: int = 0) -> None:
        """Initialize the transport.

        Args:
            replay_buffer_size: Maximum events retained per topic for
                late-subscriber replay. ``0`` disables replay (default).
        """
        self._replay_buffer_size: int = replay_buffer_size
        self._subscribers: dict[str, set[asyncio.Queue[EventEnvelope | None]]] = {}
        self._replay_buffers: dict[str, list[EventEnvelope]] = {}
        self._backlog: dict[str, list[EventEnvelope]] = {}
        self._closed: bool = False

    async def publish(self, envelope: EventEnvelope) -> None:
        """Publish an envelope to the transport.

        Broadcasts the envelope to all active subscriber queues for the
        topic. If no subscribers exist, the envelope is buffered in a
        per-topic backlog for delivery to the first subscriber.

        Args:
            envelope: The envelope to publish.

        Raises:
            RuntimeError: If the transport has been closed.
        """
        if self._closed:
            msg = "Cannot publish on a closed InProcessTransport."
            raise RuntimeError(msg)

        topic = envelope.session_id

        subs = self._subscribers.get(topic)
        if subs:
            for queue in subs:
                await queue.put(envelope)
        else:
            self._backlog.setdefault(topic, []).append(envelope)

        if self._replay_buffer_size > 0:
            buffer = self._replay_buffers.setdefault(topic, [])
            buffer.append(envelope)
            if len(buffer) > self._replay_buffer_size:
                del buffer[0]

    def subscribe(self, topic: str, from_seq: int = 0) -> AsyncIterator[EventEnvelope]:
        """Return an async iterator of envelopes for *topic*.

        If ``from_seq > 0`` and a replay buffer exists, replayed events
        with ``seq >= from_seq`` are yielded first, then backlog events
        (if any), then new events from the subscriber's own queue as
        they arrive.

        Args:
            topic: Topic identifier (typically ``session_id``).
            from_seq: Replay from this sequence number. Events with
                ``seq >= from_seq`` are replayed first.

        Returns:
            Async iterator yielding ``EventEnvelope`` objects.

        Raises:
            RuntimeError: If the transport has been closed.
        """
        if self._closed:
            msg = "Cannot subscribe on a closed InProcessTransport."
            raise RuntimeError(msg)

        return self._iterate(topic, from_seq)

    async def _iterate(self, topic: str, from_seq: int) -> AsyncIterator[EventEnvelope]:
        """Async generator yielding replayed, backlog, then new envelopes.

        Creates a per-subscriber queue so each subscriber independently
        receives all new events. Cleans up the queue on exit.

        Args:
            topic: Topic identifier.
            from_seq: Minimum sequence number for replay.

        Yields:
            ``EventEnvelope`` objects, replayed first then live.
        """
        queue: asyncio.Queue[EventEnvelope | None] = asyncio.Queue()
        self._subscribers.setdefault(topic, set()).add(queue)
        try:
            # Yield replayed events first.
            if from_seq > 0 and self._replay_buffer_size > 0:
                buffer = self._replay_buffers.get(topic, [])
                for envelope in buffer:
                    if envelope.seq is not None and envelope.seq >= from_seq:
                        yield envelope

            # Yield backlog events (published before any subscriber).
            backlog = self._backlog.pop(topic, [])
            for envelope in backlog:
                yield envelope

            # Yield new events from subscriber's own queue.
            while True:
                item: EventEnvelope | None = await queue.get()
                if item is None:  # sentinel from close()
                    break
                yield item
                queue.task_done()
        finally:
            # Clean up subscriber queue.
            subs = self._subscribers.get(topic)
            if subs is not None:
                subs.discard(queue)
                if not subs:
                    del self._subscribers[topic]

    def ack(self, seq: int) -> None:
        """Acknowledge that an event has been processed.

        For in-process transport, this is a no-op. MQ-backed transports
        would commit the consumer offset.

        Args:
            seq: The sequence number to acknowledge.
        """

    def close(self) -> None:
        """Release all transport resources.

        Sets ``_closed = True`` and puts a ``None`` sentinel on each
        subscriber queue to wake blocked consumers. Subsequent calls to
        ``publish()`` or ``subscribe()`` raise ``RuntimeError``.
        """
        self._closed = True
        for _topic, queues in list(self._subscribers.items()):
            for queue in queues:
                queue.put_nowait(None)
        self._subscribers.clear()


__all__ = ["InProcessTransport"]
