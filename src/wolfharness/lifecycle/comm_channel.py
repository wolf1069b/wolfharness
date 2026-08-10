"""CommChannel dimension: DirectChannel and ProtocolChannel.

The CommChannel abstracts event delivery and feedback reception for the
RunLoop. It owns the Journal reference and handles event persistence
internally (append for deltas, upsert for entity-state events).

Two implementations are provided:

- **DirectChannel** — unidirectional; publishes events to an internal
  ``asyncio.Queue`` that ``RunLoop.start()`` drains via ``get_nowait()``.
  ``recv()`` always returns ``None``.

- **ProtocolChannel** — bidirectional; publishes events to the
  ``EventBus`` for protocol server consumption and maintains a feedback
  queue for steer/followup messages from ``SessionController``.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING, Any

from wolfharness.agents.events import (
    MessageReplacementEvent,
    PlanUpdateEvent,
    StateUpdate,
    ToolCallUpdateEvent,
)
from wolfharness.log import get_logger


if TYPE_CHECKING:
    from wolfharness.lifecycle.protocols import Journal
    from wolfharness.lifecycle.types import Feedback, RunState
    from wolfharness.orchestrator.event_bus import EventBus


logger = get_logger(__name__)


def _derive_upsert_key(event: Any) -> str | None:
    """Derive the journal upsert key for an event.

    Entity-state events (tool call updates, state updates, message
    replacements, plan updates) use upsert semantics so only the latest
    state per entity is retained. Delta events return ``None`` to use
    append semantics.

    Args:
        event: The event to derive a key for.

    Returns:
        Deduplication key string, or ``None`` for append semantics.
    """
    match event:
        case ToolCallUpdateEvent(tool_call_id=tid) if tid:
            return f"tool_call:{tid}"
        case StateUpdate(session_id=sid) if sid:
            return f"state:{sid}"
        case MessageReplacementEvent(message_id=mid) if mid:
            return f"msg:{mid}"
        case PlanUpdateEvent(tool_call_id=tcid) if tcid is not None:
            return f"plan:{tcid}"
        case _:
            return None


class DirectChannel:
    """Unidirectional CommChannel for in-process event delivery.

    Publishes events to an internal ``asyncio.Queue``. The RunLoop's
    ``start()`` method drains this queue via ``get_nowait()`` to
    consume events. ``recv()`` always returns ``None`` since this
    channel does not support feedback.

    Events are journaled (append or upsert) before delivery, unless
    ``_replaying`` is ``True``.
    """

    def __init__(self, journal: Journal) -> None:
        """Initialize the direct channel.

        Args:
            journal: The Journal to persist events to.
        """
        self._journal: Journal = journal
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._replaying: bool = False
        self._closed: bool = False
        self._run_loop: Any = None
        self._state: RunState | None = None

    @property
    def queue(self) -> asyncio.Queue[Any]:
        """The internal event queue, accessible for RunLoop draining."""
        return self._queue

    @property
    def journal(self) -> Journal:
        """The Journal instance owned by this channel.

        Exposed so callers (notably ``RunHandle.__post_init__``) can
        reuse the same Journal instance instead of constructing a new
        one, which would break crash recovery (``journal.resume()``
        reads from an empty journal while events are written to a
        different instance).
        """
        return self._journal

    @property
    def publishes_to_event_bus(self) -> bool:
        """DirectChannel does not publish to the EventBus.

        Returns:
            Always ``False``.
        """
        return False

    def set_replaying(self, flag: bool) -> None:
        """Set the replaying flag.

        When ``True``, journaling is skipped during ``publish()``.

        Args:
            flag: ``True`` to enable replaying mode, ``False`` to disable.
        """
        self._replaying = flag

    def attach(self, run_loop: Any) -> None:
        """Store a reference to the RunLoop.

        No-op for DirectChannel since it does not route feedback.

        Args:
            run_loop: The RunLoop instance.
        """
        self._run_loop = run_loop

    def on_state_change(self, state: RunState) -> None:
        """Receive state transitions.

        No-op for DirectChannel but required for Protocol conformance.

        Args:
            state: The new RunState.
        """
        self._state = state

    async def publish(self, event: Any) -> None:
        """Journal and enqueue an event.

        If ``_replaying`` is ``True``, journaling is skipped.
        Otherwise, the event is journaled (append or upsert) before
        being enqueued to the internal queue.

        Args:
            event: The event to publish.

        Raises:
            RuntimeError: If the channel has been closed.
        """
        if self._closed:
            raise RuntimeError("DirectChannel is closed; cannot publish.")

        if not self._replaying:
            key = _derive_upsert_key(event)
            if key is not None:
                self._journal.upsert(key, event)
            else:
                self._journal.append(event)

        self._queue.put_nowait(event)

    def recv(self) -> Feedback | None:
        """Return ``None`` (unidirectional channel).

        DirectChannel does not support feedback reception.

        Returns:
            Always ``None``.
        """
        return None

    def deliver_feedback(self, feedback: Feedback) -> bool:
        """Reject feedback (unidirectional channel).

        DirectChannel does not support feedback delivery. Returns
        ``False`` so the caller can fall back to the queue-based path.

        Args:
            feedback: Ignored.

        Returns:
            Always ``False``.
        """
        return False

    def revoke(self, message_id: str) -> bool:
        """No-op revoke for unidirectional channel.

        DirectChannel has no feedback queue, so there is nothing to
        revoke. Always returns ``False``.

        Args:
            message_id: Ignored.

        Returns:
            Always ``False``.
        """
        return False

    def replace(self, message_id: str, new_content: str | list[Any]) -> bool:
        """No-op replace for unidirectional channel.

        DirectChannel has no feedback queue, so there is nothing to
        replace. Always returns ``False``.

        Args:
            message_id: Ignored.
            new_content: Ignored.

        Returns:
            Always ``False``.
        """
        return False

    def close(self) -> None:
        """Drain the queue and mark the channel as closed.

        After ``close()``, further calls to ``publish()`` raise
        ``RuntimeError``.
        """
        self._closed = True
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break


class ProtocolChannel:
    """Bidirectional CommChannel for protocol server event delivery.

    Publishes events to the ``EventBus`` for consumption by protocol
    servers (ACP, OpenCode, AG-UI, etc.). Maintains a feedback queue
    (``collections.deque``) for steer/followup messages injected by
    ``SessionController``.

    The feedback queue is backed by ``deque`` (not ``asyncio.Queue``)
    to support O(1) removal by value via ``revoke()``. Three tracking
    structures provide ID-based lifecycle management:

    - ``_pending``: ``dict[str, Feedback]`` — feedback waiting in the
      queue, keyed by ``message_id``.
    - ``_revoked``: ``set[str]`` — revoked message IDs; rejected if
      re-delivered.
    - ``_delivered``: ``set[str]`` — delivered message IDs; ``revoke()``
      returns ``False`` for these.

    Events are journaled (append or upsert) before delivery, unless
    ``_replaying`` is ``True``.
    """

    def __init__(
        self,
        journal: Journal,
        event_bus: EventBus,
        session_id: str = "",
    ) -> None:
        """Initialize the protocol channel.

        Args:
            journal: The Journal to persist events to.
            event_bus: The EventBus to publish events to.
            session_id: The session ID for EventBus routing.
        """
        self._journal: Journal = journal
        self._event_bus: EventBus = event_bus
        self._session_id: str = session_id
        self._feedback_queue: deque[Feedback] = deque()
        self._pending: dict[str, Feedback] = {}
        self._revoked: set[str] = set()
        self._delivered: set[str] = set()
        self._replaying: bool = False
        self._closed: bool = False
        self._run_loop: Any = None
        self._state: RunState | None = None

    def set_replaying(self, flag: bool) -> None:
        """Set the replaying flag.

        When ``True``, journaling is skipped during ``publish()``.

        Args:
            flag: ``True`` to enable replaying mode, ``False`` to disable.
        """
        self._replaying = flag

    @property
    def publishes_to_event_bus(self) -> bool:
        """ProtocolChannel publishes to the EventBus internally.

        Returns:
            Always ``True``.
        """
        return True

    @property
    def journal(self) -> Journal:
        """The Journal instance owned by this channel.

        Exposed so callers (notably ``RunHandle.__post_init__``) can
        reuse the same Journal instance instead of constructing a new
        one, which would break crash recovery (``journal.resume()``
        reads from an empty journal while events are written to a
        different instance).
        """
        return self._journal

    def attach(self, run_loop: Any) -> None:
        """Store a reference to the RunLoop for feedback routing.

        Args:
            run_loop: The RunLoop instance.
        """
        self._run_loop = run_loop

    def on_state_change(self, state: RunState) -> None:
        """Track RunLoop state for steer/followup routing.

        Args:
            state: The new RunState.
        """
        self._state = state

    async def publish(self, event: Any) -> None:
        """Journal and deliver an event to the EventBus.

        If ``_replaying`` is ``True``, journaling is skipped.
        Otherwise, the event is journaled (append or upsert) before
        being published to the EventBus.

        ``StateUpdate`` events are journaled but NOT published to the
        EventBus. They are internal lifecycle signals (state machine
        transitions) that protocol servers do not need to receive.
        This preserves backward compatibility with tests and protocol
        handlers that do not expect ``StateUpdate`` on the EventBus.

        ``UserMessageInsertedEvent`` events during replay (``_replaying``
        is ``True``) are journaled but NOT published to the EventBus.
        During crash-recovery replay, these events were already delivered
        to the EventBus during the original run. Republishing would cause
        duplicate user message rendering in protocol frontends.

        Args:
            event: The event to publish.

        Raises:
            RuntimeError: If the channel has been closed.
        """
        if self._closed:
            raise RuntimeError("ProtocolChannel is closed; cannot publish.")

        if not self._replaying:
            key = _derive_upsert_key(event)
            if key is not None:
                self._journal.upsert(key, event)
            else:
                self._journal.append(event)

        # StateUpdate events are internal lifecycle signals — journal
        # them but do not publish to EventBus. This prevents protocol
        # servers and EventBus subscribers from receiving state machine
        # transitions they don't know how to handle.
        if isinstance(event, StateUpdate):
            return

        # P2: During replay, skip EventBus publish for
        # UserMessageInsertedEvent to avoid duplicate user message
        # rendering. The event was already delivered during the
        # original run; replay only needs journaling.
        if self._replaying and type(event).__name__ == "UserMessageInsertedEvent":
            logger.debug("Skipping EventBus publish for replayed UserMessageInsertedEvent")
            return

        await self._event_bus.publish(self._session_id, event)

    def recv(self) -> Feedback | None:
        """Non-blocking dequeue from the feedback queue.

        Transitions the dequeued feedback from ``_pending`` to
        ``_delivered`` to prevent revoking already-delivered messages.

        Returns:
            The next ``Feedback`` if available, or ``None``.
        """
        if not self._feedback_queue:
            return None
        feedback = self._feedback_queue.popleft()
        msg_id = feedback.message_id
        self._pending.pop(msg_id, None)
        self._delivered.add(msg_id)
        return feedback

    def deliver_feedback(self, feedback: Feedback) -> bool:
        """Enqueue feedback from SessionController.

        This is how steer/followup messages arrive at the RunLoop.
        Revoked messages are rejected.

        Args:
            feedback: The feedback to enqueue.

        Returns:
            Always ``True`` (ProtocolChannel supports feedback delivery).
        """
        if feedback.message_id in self._revoked:
            return True
        self._feedback_queue.append(feedback)
        self._pending[feedback.message_id] = feedback
        return True

    def revoke(self, message_id: str) -> bool:
        """Revoke a pending feedback message by ID.

        Revocation only operates at the CommChannel queue layer. If the
        feedback is still in the queue (not yet delivered to the
        RunLoop), it is removed and marked as revoked. If already
        delivered, revocation is not possible and ``False`` is returned.

        Design decision: Once feedback is dequeued by ``recv()`` and
        delivered to the agent runtime (via ``_idle_loop`` →
        ``_execute_turn`` → ``agent_run``), it cannot be revoked. This
        is a deliberate design choice — post-delivery revocation would
        require deep integration with pydantic_ai's ``PendingMessage``
        lifecycle, which is fragile and provides little value (the
        message is already being processed). Future work: If ACP v2
        requires post-delivery revocation, a PydanticAI capability hook
        could be added to support it.

        Args:
            message_id: The ID of the feedback message to revoke.

        Returns:
            ``True`` if revoked or already gone, ``False`` if delivered.
        """
        # Already delivered — cannot revoke.
        if message_id in self._delivered:
            return False

        # Still pending in CommChannel queue — remove and mark revoked.
        if message_id in self._pending:
            feedback = self._pending.pop(message_id)
            self._feedback_queue.remove(feedback)
            self._revoked.add(message_id)
            return True

        # Unknown message_id — idempotent success.
        return True

    def replace(self, message_id: str, new_content: str | list[Any]) -> bool:
        """Replace the content of a pending feedback message in-place.

        Updates the content of a feedback message that is still pending
        in the channel's feedback queue, preserving its position. When
        ``new_content`` is a ``list``, updates ``Feedback.content_blocks``;
        when ``str``, updates ``Feedback.content``.

        Returns ``False`` if the message has already been delivered
        (past CommChannel scope).

        Args:
            message_id: The ID of the feedback message to replace.
            new_content: New content (``str`` or ``list[Any]``).

        Returns:
            ``True`` if replaced, ``False`` if past CommChannel scope.
        """
        # Cannot replace if already delivered.
        if message_id in self._delivered:
            return False

        if message_id not in self._pending:
            # Unknown message_id — nothing to replace.
            return False

        feedback = self._pending[message_id]
        if isinstance(new_content, list):
            feedback.content_blocks = new_content
            feedback.content = ""
        else:
            feedback.content = new_content
            feedback.content_blocks = None
        return True

    def close(self) -> None:
        """Clean up all tracking structures and mark as closed.

        After ``close()``, further calls to ``publish()`` raise
        ``RuntimeError``.
        """
        self._closed = True
        self._feedback_queue.clear()
        self._pending.clear()
        self._revoked.clear()
        self._delivered.clear()


__all__ = [
    "DirectChannel",
    "ProtocolChannel",
]
