"""Lifecycle Protocols: TriggerSource, Journal, SnapshotStore, CommChannel, EventTransport.

These five ``@runtime_checkable`` Protocols define the contracts for the six
dimensions of the RunLoop lifecycle. Implementations are provided in separate
modules (triggers.py, journal.py, snapshot_store.py, comm_channel.py,
event_transport.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from wolfharness.lifecycle.types import (
        EventEnvelope,
        Feedback,
        Prompt,
        ResumeResult,
        RunState,
        ToolExecutionRecord,
    )


@runtime_checkable
class TriggerSource(Protocol):
    """Protocol defining how prompts arrive at the RunLoop.

    Implementations deliver prompts to the RunLoop via ``poll()``.
    The RunLoop calls ``subscribe()`` once during ``start()`` to
    attach itself, then calls ``poll()`` in the main loop.
    """

    def subscribe(self, run_loop: Any) -> None:
        """Attach to the RunLoop for prompt delivery.

        Called exactly once during ``RunLoop.start()``.

        Args:
            run_loop: The RunLoop instance to attach to.
        """
        ...

    def poll(self) -> Prompt | None:
        """Return the next ``Prompt`` if available, or ``None``.

        Must not block — returns ``None`` immediately if no prompt
        is pending.
        """
        ...

    def close(self) -> None:
        """Release all resources held by the TriggerSource."""
        ...


@runtime_checkable
class Journal(Protocol):
    """Protocol for event-layer persistence.

    Supports two write semantics: ``append()`` for delta events
    (each entry is a new record) and ``upsert(key, event)`` for
    entity-state events (latest state per key replaces previous).
    Both return a monotonically increasing sequence number.
    """

    def append(self, event: Any) -> int:
        """Create a new journal entry for a delta event.

        Args:
            event: The event to append.

        Returns:
            Monotonically increasing sequence number.
        """
        ...

    def upsert(self, key: str, event: Any) -> int:
        """Replace or create an entity-state entry by key.

        Args:
            key: Deduplication key (e.g. ``"tool_call:abc"``).
            event: The event to upsert.

        Returns:
            Monotonically increasing sequence number.
        """
        ...

    def replay(self, from_seq: int = 0, to_seq: int | None = None) -> AsyncIterator[Any]:
        """Return an async iterator of events in sequence order.

        Append entries are all returned, ordered by ``seq``.
        Upsert entries return only the latest state per key.

        Args:
            from_seq: Start sequence (inclusive). Defaults to 0.
            to_seq: End sequence (inclusive). ``None`` means no upper bound.
        """
        ...

    def resume(self, snapshot_store: SnapshotStore) -> ResumeResult | None:
        """Coordinate snapshot and journal for crash recovery.

        Loads the latest snapshot, replays journal events since the
        snapshot, and determines if a Turn was in-flight.

        Args:
            snapshot_store: The snapshot store to load state from.

        Returns:
            ``ResumeResult`` if state exists, ``None`` for fresh start.
        """
        ...

    def compact(self, before_seq: int) -> None:
        """Remove journal entries with ``seq < before_seq``.

        Called after a successful snapshot to prevent unbounded growth.

        Args:
            before_seq: Remove entries with seq below this value.
        """
        ...

    def clear(self) -> None:
        """Remove all journal entries, resetting the sequence counter."""
        ...

    def log_tool_execution(self, record: ToolExecutionRecord) -> None:
        """Store a tool execution record for idempotent crash recovery.

        Args:
            record: The tool execution record to store.
        """
        ...

    def get_tool_executions(self, turn_id: str) -> list[ToolExecutionRecord]:
        """Retrieve all tool execution records for a Turn.

        Args:
            turn_id: The Turn ID to query.

        Returns:
            List of tool execution records for the given Turn.
        """
        ...


@runtime_checkable
class SnapshotStore(Protocol):
    """Protocol for loop-layer state persistence.

    Persists full state images at Turn boundaries and provides
    idempotency keys via ``turn_id``.
    """

    def save(self, state: Any) -> int:
        """Persist a full state snapshot.

        Args:
            state: The RunState snapshot to persist.

        Returns:
            The sequence number of the snapshot.
        """
        ...

    def load(self) -> tuple[Any, int] | None:
        """Return the latest snapshot.

        Returns:
            Tuple of ``(state, last_journal_seq)`` if a snapshot exists,
            ``None`` otherwise.
        """
        ...

    def save_turn_result(self, turn_id: str, result: Any) -> None:
        """Persist a completed Turn's result for idempotency.

        Args:
            turn_id: The Turn ID.
            result: The Turn's result.
        """
        ...

    def has_turn_result(self, turn_id: str) -> bool:
        """Check whether a Turn was already completed.

        Args:
            turn_id: The Turn ID to check.

        Returns:
            ``True`` if the Turn result was saved, ``False`` otherwise.
        """
        ...

    def clear(self) -> None:
        """Remove all snapshots and turn results."""
        ...


@runtime_checkable
class CommChannel(Protocol):
    """Protocol abstracting event delivery and feedback reception.

    Owns the Journal reference and handles event persistence internally
    (append for deltas, upsert for entity-state events).
    """

    @property
    def journal(self) -> Journal:
        """The Journal instance owned by this channel.

        Every CommChannel owns a Journal. Callers (notably
        ``RunHandle.__post_init__``) can access this to reuse the same
        Journal instance, ensuring ``resume()`` reads the same events
        that ``publish()`` writes.

        Returns:
            The Journal instance owned by this channel.
        """
        ...

    def set_replaying(self, flag: bool) -> None:
        """Set the replaying flag.

        When ``True``, the channel skips journaling on ``publish()``.
        Used during crash recovery replay to avoid duplicate entries.

        Args:
            flag: ``True`` to enable replaying mode, ``False`` to disable.
        """
        ...

    @property
    def publishes_to_event_bus(self) -> bool:
        """Whether this channel publishes events to the EventBus internally.

        ``ProtocolChannel`` publishes to the EventBus inside its own
        ``publish()`` method, so the RunLoop must NOT also call
        ``event_bus.publish()`` directly to avoid double-publishing.

        ``DirectChannel`` does not publish to the EventBus, so the
        direct call in the RunLoop is required.

        Returns:
            ``True`` if the channel publishes to the EventBus
            internally, ``False`` otherwise.
        """
        ...

    def attach(self, run_loop: Any) -> None:
        """Store a reference to the RunLoop, enabling the feedback loop.

        Called by RunLoop during ``start()``.

        Args:
            run_loop: The RunLoop instance.
        """
        ...

    def on_state_change(self, state: RunState) -> None:
        """Receive state transitions via observer pattern.

        Called by RunLoop on every state transition (idle/running/done).

        Args:
            state: The new RunState.
        """
        ...

    async def publish(self, event: Any) -> None:
        """Journal and deliver an event.

        If ``_replaying`` is ``True``, journaling is skipped.
        Otherwise, the event is journaled (append or upsert) before
        delivery to the consumer.

        Args:
            event: The event to publish.
        """
        ...

    def recv(self) -> Feedback | None:
        """Return pending feedback, or ``None`` if none available.

        For unidirectional channels, always returns ``None``.
        Must not block.
        """
        ...

    def deliver_feedback(self, feedback: Feedback) -> bool:
        """Deliver feedback (steer/followup) to the RunLoop.

        Bidirectional channels (``ProtocolChannel``) enqueue the
        feedback and return ``True``. Unidirectional channels
        (``DirectChannel``) do not support feedback and return
        ``False`` so the caller can fall back to the queue-based path.

        Args:
            feedback: The feedback message to deliver.

        Returns:
            ``True`` if the feedback was handled, ``False`` otherwise.
        """
        ...

    def revoke(self, message_id: str) -> bool:
        """Revoke a pending feedback message by ID.

        Revocation operates at the CommChannel queue layer only. If the
        feedback is still pending in the channel's feedback queue (not
        yet delivered to the RunLoop), it is removed and marked as
        revoked.

        Once feedback is delivered to the agent runtime (dequeued by
        ``recv()`` and passed to ``agent_run``), it cannot be revoked.
        This is a deliberate design choice — post-delivery revocation
        would require deep integration with pydantic_ai's
        ``PendingMessage`` lifecycle, which is fragile and provides
        little value.

        Returns ``True`` if the message was revoked or is already gone
        (idempotent). Returns ``False`` if the message was already
        delivered to the agent runtime (past the revoke window).

        Unidirectional channels (``DirectChannel``) always return
        ``False`` since they have no feedback queue.

        Args:
            message_id: The ID of the feedback message to revoke.

        Returns:
            ``True`` if revoked or already gone, ``False`` if delivered.
        """
        ...

    def replace(self, message_id: str, new_content: str | list[Any]) -> bool:
        """Replace the content of a pending feedback message in-place.

        Updates the content of a feedback message that is still pending
        in the channel's feedback queue, preserving its position. When
        ``new_content`` is a ``list``, updates ``Feedback.content_blocks``;
        when ``str``, updates ``Feedback.content``.

        Returns ``False`` if the message has already been delivered
        (past CommChannel scope). Unidirectional channels always return
        ``False``.

        Args:
            message_id: The ID of the feedback message to replace.
            new_content: New content (``str`` or ``list[Any]``).

        Returns:
            ``True`` if replaced, ``False`` if past CommChannel scope.
        """
        ...

    def close(self) -> None:
        """Release all resources held by the CommChannel."""
        ...


@runtime_checkable
class EventTransport(Protocol):
    """Protocol abstracting the wire protocol between RunLoop and consumers.

    Enables language-agnostic protocol servers and MQ-based decoupling.
    """

    async def publish(self, envelope: EventEnvelope) -> None:
        """Publish an EventEnvelope to the transport.

        Args:
            envelope: The envelope to publish.
        """
        ...

    def subscribe(self, topic: str, from_seq: int = 0) -> AsyncIterator[EventEnvelope]:
        """Return an async iterator of envelopes for a topic.

        Args:
            topic: Topic identifier (typically session_id).
            from_seq: Replay from this sequence number.

        Returns:
            Async iterator yielding EventEnvelope objects.
        """
        ...

    def ack(self, seq: int) -> None:
        """Acknowledge that an event has been processed.

        For in-process transport, this is a no-op.
        For MQ-backed transport, this commits the consumer offset.

        Args:
            seq: The sequence number to acknowledge.
        """
        ...

    def close(self) -> None:
        """Release all transport resources."""
        ...


__all__ = [
    "CommChannel",
    "EventTransport",
    "Journal",
    "SnapshotStore",
    "TriggerSource",
]
