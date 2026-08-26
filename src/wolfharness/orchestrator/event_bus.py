"""EventBus infrastructure for cross-turn event streaming.

Extracted from orchestrator/core.py as part of the thin-wrapper refactor.
Provides pub/sub event routing with replay buffers and overflow handling.

Uses ``asyncio.Queue`` with configurable overflow policies instead of
anyio memory streams. The ``block`` policy is rejected on the publish
path to prevent run-loop deadlocks.
"""

from __future__ import annotations

import asyncio
from collections import deque
import contextlib
from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic_ai import TextPartDelta, ThinkingPartDelta, ToolCallPartDelta

from wolfharness.agents.events import (
    CompactionEvent,
    ElicitationDeferredEvent,
    PartDeltaEvent,
    PlanUpdateEvent,
    RunErrorEvent,
    RunFailedEvent,
    RunStartedEvent,
    SessionResumeEvent,
    SpawnSessionStart,
    StreamCompleteEvent,
    ToolCallCompleteEvent,
    ToolCallContentItem,
    ToolCallDeferredEvent,
    ToolCallProgressEvent,
    ToolCallStartEvent,
)
from wolfharness.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from wolfharness.orchestrator.session_controller import SessionController


logger = get_logger(__name__)

DEFAULT_QUEUE_MAXSIZE: Final[int] = 1000

OverflowPolicy = Literal["drop_oldest", "drop_newest", "drop_subscriber"]

_VALID_OVERFLOW_POLICIES: frozenset[str] = frozenset({
    "drop_oldest",
    "drop_newest",
    "drop_subscriber",
})


@dataclass(frozen=True)
class EventEnvelope:
    """Wrapper for events published through EventBus.

    Carries routing metadata (source_session_id) separately from the event
    payload so consumers can determine the event's origin without mutating
    the event object.

    Attribute access is transparently forwarded to the wrapped event,
    so consumers can use ``envelope.delta`` or ``envelope.event_kind``
    without unwrapping.
    """

    source_session_id: str
    """The session that produced this event."""
    event: Any
    """The original event payload (unmodified)."""
    event_id: int = 0
    """Monotonic event ID assigned at publish time (0 for legacy/test envelopes)."""
    source_hint: str | None = None
    """Provenance tag set at publish time (e.g. ``"opencode_event_bridge"``).

    Lets subscribers exclude events published by a specific source via
    ``subscribe(exclude_source=...)``, preventing a producer from
    re-consuming its own output (loopback isolation).
    """

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the wrapped event."""
        return getattr(self.event, name)

    def __repr__(self) -> str:
        return f"EventEnvelope(source_session_id={self.source_session_id!r}, event={self.event!r})"


# ---------------------------------------------------------------------------
# Event coalescing infrastructure
# ---------------------------------------------------------------------------


def _is_immediate(event: Any) -> bool:
    """Check if an event is a lifecycle event that bypasses coalescing.

    Immediate events are dispatched right away and trigger a buffer drain
    of any pending batchable events for the session.

    Returns:
        True if the event is an immediate lifecycle event.
    """
    match event:
        case (
            RunStartedEvent()
            | RunErrorEvent()
            | RunFailedEvent()
            | StreamCompleteEvent()
            | SpawnSessionStart()
            | CompactionEvent()
            | SessionResumeEvent()
            | ToolCallStartEvent()
            | ToolCallCompleteEvent()
            | ToolCallDeferredEvent()
            | ElicitationDeferredEvent()
        ):
            return True
        case _:
            return False


def _merge_key(event: Any) -> tuple[str, str | None] | None:  # noqa: PLR0911
    """Compute the coalescing merge key for an event.

    Returns:
        A tuple key for batchable events, or None for passthrough events.
        Passthrough events are dispatched individually (after draining the buffer).
    """
    match event:
        case PartDeltaEvent(delta=TextPartDelta()):
            return ("delta_text", "")
        case PartDeltaEvent(delta=ThinkingPartDelta()):
            return ("delta_thinking", "")
        case PartDeltaEvent(delta=ToolCallPartDelta(tool_call_id=tcid)):
            return ("delta_tool_call", tcid)
        case PartDeltaEvent():
            return None
        case ToolCallProgressEvent(tool_call_id=tcid, status=status):
            return ("progress", f"{tcid}:{status}")
        case PlanUpdateEvent():
            return ("plan", "")
        case _:
            return None


def _merge_text_deltas(events: list[PartDeltaEvent]) -> PartDeltaEvent:
    """Concatenate TextPartDelta content_delta strings. Uses first event's index."""
    parts = [
        event.delta.content_delta
        for event in events
        if isinstance(event.delta, TextPartDelta) and event.delta.content_delta is not None
    ]
    return PartDeltaEvent(
        index=events[0].index,
        delta=TextPartDelta(content_delta="".join(parts)),
    )


def _merge_thinking_deltas(events: list[PartDeltaEvent]) -> PartDeltaEvent:
    """Concatenate ThinkingPartDelta content_delta strings. Uses first event's index."""
    parts = [
        event.delta.content_delta
        for event in events
        if isinstance(event.delta, ThinkingPartDelta) and event.delta.content_delta is not None
    ]
    return PartDeltaEvent(
        index=events[0].index,
        delta=ThinkingPartDelta(content_delta="".join(parts)),
    )


def _merge_tool_call_deltas(events: list[PartDeltaEvent]) -> PartDeltaEvent:
    """Concatenate ToolCallPartDelta args_delta strings.

    Uses first event's index and tool_call_id.
    """
    parts: list[str] = []
    tool_call_id: str | None = None
    for event in events:
        delta = event.delta
        if isinstance(delta, ToolCallPartDelta) and isinstance(delta.args_delta, str):
            parts.append(delta.args_delta)
            if not tool_call_id:
                tool_call_id = delta.tool_call_id
    return PartDeltaEvent(
        index=events[0].index,
        delta=ToolCallPartDelta(args_delta="".join(parts), tool_call_id=tool_call_id),
    )


def _merge_progress_events(events: list[ToolCallProgressEvent]) -> ToolCallProgressEvent:
    """Concatenate items sequences from progress events.

    Uses last event's title, status, replace_content, and tool_name.
    Items with duplicate TerminalContentItem.terminal_id are kept (consumer handles dedup).
    """
    all_items: list[ToolCallContentItem] = []
    for event in events:
        all_items.extend(event.items)
    last = events[-1]
    return ToolCallProgressEvent(
        tool_call_id=last.tool_call_id,
        status=last.status,
        title=last.title,
        items=all_items,
        replace_content=last.replace_content,
        tool_name=last.tool_name,
        progress=last.progress,
        total=last.total,
        message=last.message,
        tool_input=last.tool_input,
        session_id=last.session_id,
    )


def _rebind(template: EventEnvelope, new_event: Any) -> EventEnvelope:
    """Create new EventEnvelope with merged event, preserving source_session_id and event_id."""
    return EventEnvelope(
        source_session_id=template.source_session_id,
        event=new_event,
        event_id=template.event_id,
        source_hint=template.source_hint,
    )


def _merge_envelopes(envelopes: list[EventEnvelope]) -> list[EventEnvelope]:
    """Merge a list of envelopes using itertools.groupby.

    Groups consecutive envelopes by _merge_key. For batchable groups, merges
    events into a single event. For passthrough groups (key is None), extends
    without merging. For the ("plan", "") key, keeps the last event (last-wins).
    Drops PartDeltaEvent instances where delta is None.

    Returns:
        List of merged (or passthrough) envelopes ready for dispatch.
    """
    filtered: list[EventEnvelope] = [
        env
        for env in envelopes
        if not (isinstance(env.event, PartDeltaEvent) and env.event.delta is None)
    ]

    result: list[EventEnvelope] = []
    for key, group in groupby(filtered, key=lambda env: _merge_key(env.event)):
        group_list = list(group)
        if key is None:
            result.extend(group_list)
        elif key[0] == "plan":
            result.append(group_list[-1])
        else:
            events = [env.event for env in group_list]
            merged: PartDeltaEvent | ToolCallProgressEvent
            match key[0]:
                case "delta_text":
                    merged = _merge_text_deltas(events)
                case "delta_thinking":
                    merged = _merge_thinking_deltas(events)
                case "delta_tool_call":
                    merged = _merge_tool_call_deltas(events)
                case "progress":
                    merged = _merge_progress_events(events)
                case _:
                    result.extend(group_list)
                    continue
            result.append(_rebind(group_list[0], merged))
    return result


async def drain_and_merge(
    queue: asyncio.Queue[EventEnvelope],
) -> AsyncIterator[EventEnvelope]:
    """Drain all queued events from a subscriber queue and merge consecutive same-type events.

    Performs subscriber-side coalescing: blocks on ``await queue.get()``
    until at least one item is available, then drains all immediately-available
    items via ``get_nowait()`` until ``QueueEmpty``. The resulting batch is
    merged via ``_merge_envelopes()`` and each merged envelope is yielded.
    Repeats until the queue is shut down (``QueueShutDown``).

    Raw events (not wrapped in ``EventEnvelope``) are automatically wrapped with
    an empty ``source_session_id`` for compatibility with test queues.

    Args:
        queue: The ``asyncio.Queue`` to drain. Items must be
            ``EventEnvelope`` instances.

    Yields:
        Merged ``EventEnvelope`` instances ready for dispatch.
    """
    while True:
        try:
            first = await queue.get()
        except asyncio.QueueShutDown:
            return

        if not isinstance(first, EventEnvelope):
            first = EventEnvelope(source_session_id="", event=first, event_id=0)

        batch: list[EventEnvelope] = [first]

        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            except asyncio.QueueShutDown:
                for env in _merge_envelopes(batch):
                    yield env
                return
            if not isinstance(item, EventEnvelope):
                item = EventEnvelope(source_session_id="", event=item, event_id=0)
            batch.append(item)

        for env in _merge_envelopes(batch):
            yield env


_SubscriberEntry = tuple[asyncio.Queue[EventEnvelope], str, frozenset[str]]


class EventBus:
    """PubSub event bus for cross-turn event streaming.

    Decouples event producers (agents) from consumers (protocol handlers).
    Events are broadcast to all subscribers for a given session via ``_send()``
    with no publish-side buffering or coalescing.

    Event coalescing/merging is the subscriber's responsibility. Subscribers
    should drain their queue using ``drain_and_merge()`` to batch-merge
    consecutive same-type events (e.g., ``PartDeltaEvent`` text chunks) for
    efficient processing.

    Safety features:
    - Bounded ``asyncio.Queue`` with configurable overflow policies
    - Automatic cleanup of dead subscribers
    - ``Queue.shutdown()``-based shutdown (no sentinel None)
    """

    def __init__(
        self,
        max_queue_size: int = DEFAULT_QUEUE_MAXSIZE,
        replay_buffer_size: int = 100,
        session_controller: SessionController | None = None,
        overflow_policy: OverflowPolicy = "drop_oldest",
    ) -> None:
        """Initialize the event bus.

        Args:
            max_queue_size: Maximum buffer size for subscriber queues.
            replay_buffer_size: Maximum number of events retained per session for replay.
            session_controller: Optional session controller for hierarchy queries.
            overflow_policy: Policy for handling full subscriber queues.
                One of ``drop_oldest``, ``drop_newest``, ``drop_subscriber``.
                ``block`` is NOT supported (would deadlock the run loop).

        Raises:
            ValueError: If ``overflow_policy`` is ``"block"`` or not a valid policy.
        """
        if overflow_policy not in _VALID_OVERFLOW_POLICIES:
            if str(overflow_policy) == "block":
                raise ValueError(
                    "overflow_policy='block' is not supported on the publish path — "
                    "it would deadlock the run loop. Use 'drop_oldest', 'drop_newest', "
                    "or 'drop_subscriber' instead."
                )
            raise ValueError(
                f"Invalid overflow_policy={overflow_policy!r}. "
                f"Must be one of: {sorted(_VALID_OVERFLOW_POLICIES)}"
            )

        self._subscribers: dict[str, list[_SubscriberEntry]] = {}
        self._session_tree: dict[str, list[str]] = {}
        self._lock = asyncio.Lock()
        self._max_queue_size = max_queue_size
        self._replay_buffer_size = replay_buffer_size
        self._session_controller = session_controller
        self._replay_buffers: dict[str, deque[EventEnvelope]] = {}
        self._overflow_policy: OverflowPolicy = overflow_policy
        self._event_counter: int = 0

    async def subscribe(
        self,
        session_id: str,
        scope: str = "session",
        *,
        replay: bool = True,
        last_event_id: int | None = None,
        exclude_source: frozenset[str] | None = None,
    ) -> asyncio.Queue[EventEnvelope]:
        """Subscribe to events for a session.

        New subscribers receive replayed historical events from the replay
        buffer before live events. Events published during the replay phase
        are drained and re-inserted after historical events to preserve
        ordering and avoid loss.

        Conditional replay is supported via ``replay`` and ``last_event_id``:

        - ``replay=False``: Skip the replay buffer entirely (zero historical
          events). Only live events published after subscription are delivered.
        - ``replay=True, last_event_id=None``: Replay all buffered events
          (default — backward compatible).
        - ``replay=True, last_event_id=N``: Replay only events with
          ``event_id > N``. If the buffer's oldest event has
          ``event_id > N + 1`` (gap detected, meaning events N+1..oldest-1
          have been evicted), fall back to replaying all events from that
          buffer to avoid silent data loss.

        For ``scope="all"``, the filtering applies per-buffer after collecting
        from all buffers.

        ``exclude_source`` filters at publish fanout time: events published
        with a matching ``source_hint`` are never queued to this subscriber
        (live or replayed). This lets a producer subscribe to a bus without
        re-consuming its own output (loopback isolation).

        Args:
            session_id: The session to subscribe to.
            scope: Subscription scope - "session" (exact match),
                "descendants" (self + children), or "subtree" (self + parent + siblings).

                !!! warning "Deprecated: descendants scope"
                    The "descendants" scope is deprecated for protocol server use.
                    It has known issues with replay buffer data loss, O(N) recursive
                    traversal, and duplicate deliveries. Protocol servers should use
                    "session" scope with explicit child consumers via
                    `ProtocolEventConsumerMixin._on_spawn_session_start()` instead.
                    The "descendants" enum value is retained for backward compatibility.
            replay: If False, skip replay buffer (no historical events).
            last_event_id: If provided with ``replay=True``, only replay
                events with ``event_id > last_event_id``. Gap detection
                falls back to full replay when the buffer is missing
                contiguous events.
            exclude_source: Provenance tags whose published events this
                subscriber must not receive.

        Returns:
            An ``asyncio.Queue`` to consume events from.
        """
        queue: asyncio.Queue[EventEnvelope] = asyncio.Queue(maxsize=self._max_queue_size)

        async with self._lock:
            self._subscribers.setdefault(session_id, []).append((
                queue,
                scope,
                exclude_source or frozenset(),
            ))
            if scope == "all":
                historical_events: list[EventEnvelope] = []
                for buffer in self._replay_buffers.values():
                    historical_events.extend(buffer)
            else:
                buffer = self._replay_buffers.get(session_id, deque())
                historical_events = list(buffer)

        # Apply conditional replay filtering
        if not replay:
            historical_events = []
        elif last_event_id is not None and historical_events:
            # Gap detection: if the oldest event in the buffer has an event_id
            # greater than last_event_id + 1, the contiguous range has been
            # broken (events were evicted). Fall back to full replay from
            # this buffer to avoid silent data loss.
            oldest_id = historical_events[0].event_id
            if oldest_id > 0 and oldest_id > last_event_id + 1:
                # Gap detected — keep all events (full replay fallback)
                pass
            else:
                historical_events = [
                    env for env in historical_events if env.event_id > last_event_id
                ]

        # Apply source exclusion to replayed events too (same rule as live
        # fanout in _send), so a subscriber never sees its own output even
        # through the replay buffer.
        if exclude_source and historical_events:
            historical_events = [
                env
                for env in historical_events
                if env.source_hint is None or env.source_hint not in exclude_source
            ]

        for envelope in historical_events:
            try:
                queue.put_nowait(envelope)
            except asyncio.QueueFull:
                break

        return queue

    def clear_replay_buffer(self, session_id: str) -> None:
        """Clear the replay buffer for a session.

        Removes all historical events from the replay buffer so that
        new subscribers only receive events from this point forward.
        This should be called at the start of each turn to prevent
        stale events (including terminal events like StreamCompleteEvent)
        from previous turns being replayed to new subscribers.

        Args:
            session_id: The session whose replay buffer to clear.
        """
        self._replay_buffers.pop(session_id, None)

    async def unsubscribe(
        self,
        session_id: str,
        queue: asyncio.Queue[EventEnvelope],
    ) -> None:
        """Unsubscribe from events.

        Shuts down the subscriber's queue so the consumer receives
        ``QueueShutDown``. Cleans up empty subscriber lists to prevent
        memory leaks.

        Args:
            session_id: The session to unsubscribe from.
            queue: The queue returned by subscribe().
        """
        async with self._lock:
            if session_id in self._subscribers:
                self._subscribers[session_id] = [
                    (q, sc, ex) for q, sc, ex in self._subscribers[session_id] if q is not queue
                ]
                if not self._subscribers[session_id]:
                    del self._subscribers[session_id]

        queue.shutdown()

    def _get_parent(self, session_id: str) -> str | None:
        """Find the parent of a session in the session tree."""
        if self._session_controller is not None:
            parent_state = self._session_controller.get_parent(session_id)
            if parent_state is not None:
                return parent_state.session_id
        for parent_id, children in self._session_tree.items():
            if session_id in children:
                return parent_id
        return None

    def _is_descendant(self, child_id: str, parent_id: str) -> bool:
        """Check if child_id is a descendant of parent_id."""
        if self._session_controller is not None:
            children = self._session_controller.get_children(parent_id)
        else:
            children = self._session_tree.get(parent_id, [])
        return child_id in children or any(
            self._is_descendant(child_id, child) for child in children
        )

    def _are_siblings(self, sid1: str, sid2: str) -> bool:
        """Check if two sessions share the same parent."""
        parent1 = self._get_parent(sid1)
        parent2 = self._get_parent(sid2)
        return parent1 is not None and parent1 == parent2

    def _should_receive(self, published_sid: str, subscriber_sid: str, scope: str) -> bool:
        """Determine if a published event should reach a subscriber."""
        if scope == "session":
            return published_sid == subscriber_sid
        if scope == "descendants":
            return published_sid == subscriber_sid or self._is_descendant(
                published_sid, subscriber_sid
            )
        if scope == "subtree":
            return (
                published_sid == subscriber_sid
                or published_sid == self._get_parent(subscriber_sid)
                or self._are_siblings(published_sid, subscriber_sid)
            )
        if scope == "all":
            return True
        return published_sid == subscriber_sid

    def _enqueue(self, queue: asyncio.Queue[EventEnvelope], envelope: EventEnvelope) -> bool:
        """Enqueue an envelope using the configured overflow policy.

        Returns:
            True if the envelope was enqueued (or dropped by policy),
            False if the subscriber should be removed (dead queue).
        """
        try:
            queue.put_nowait(envelope)
        except asyncio.QueueShutDown:
            return False
        except asyncio.QueueFull:
            match self._overflow_policy:
                case "drop_oldest":
                    with contextlib.suppress(asyncio.QueueEmpty):
                        queue.get_nowait()
                    try:
                        queue.put_nowait(envelope)
                    except asyncio.QueueFull:
                        return False
                    else:
                        return True
                case "drop_newest":
                    return True
                case "drop_subscriber":
                    return False
        return True

    async def _send(self, session_id: str, envelope: EventEnvelope) -> None:
        """Send a single envelope to all matching subscribers.

        Appends to the replay buffer, collects target subscribers under
        ``_lock``, then sends to each target with the configured overflow
        policy. Cleans up dead queues afterwards.

        Args:
            session_id: The session that produced the event.
            envelope: The pre-constructed event envelope to broadcast.
        """
        async with self._lock:
            if session_id not in self._replay_buffers:
                self._replay_buffers[session_id] = deque(maxlen=self._replay_buffer_size)
            self._replay_buffers[session_id].append(envelope)

            targets: list[tuple[asyncio.Queue[EventEnvelope], str]] = []
            for subscriber_sid, subscribers in self._subscribers.items():
                for queue, scope, excluded in subscribers:
                    if envelope.source_hint is not None and envelope.source_hint in excluded:
                        continue
                    if self._should_receive(session_id, subscriber_sid, scope):
                        targets.append((queue, scope))

        dead_queues: list[asyncio.Queue[EventEnvelope]] = []
        for queue, _scope in targets:
            if not self._enqueue(queue, envelope):
                dead_queues.append(queue)

        if dead_queues:
            dead_set = {id(q) for q in dead_queues}
            async with self._lock:
                for subscriber_sid in list(self._subscribers):
                    self._subscribers[subscriber_sid] = [
                        item
                        for item in self._subscribers[subscriber_sid]
                        if id(item[0]) not in dead_set
                    ]
                    if not self._subscribers[subscriber_sid]:
                        del self._subscribers[subscriber_sid]

        for queue in dead_queues:
            with contextlib.suppress(Exception):
                queue.shutdown()

    async def publish(
        self,
        session_id: str,
        event: Any,
        *,
        source_hint: str | None = None,
    ) -> None:
        """Publish an event to all subscribers for a session.

        Wraps the event in an EventEnvelope and sends it directly via _send().
        ``PartDeltaEvent`` with ``delta=None`` is dropped (no content to deliver).
        Coalescing is handled subscriber-side by ``drain_and_merge()``.

        Each published event is assigned a monotonically increasing ``event_id``
        so that subscribers can request conditional replay via ``last_event_id``.

        Args:
            session_id: The session that produced the event.
            event: The event to broadcast.
            source_hint: Optional provenance tag so subscribers can exclude
                this publisher's events (see ``subscribe(exclude_source=...)``).
        """
        if isinstance(event, PartDeltaEvent) and event.delta is None:
            return
        self._event_counter += 1
        envelope = EventEnvelope(
            source_session_id=session_id,
            event=event,
            event_id=self._event_counter,
            source_hint=source_hint,
        )
        await self._send(session_id, envelope)

    async def close_session(self, session_id: str) -> None:
        """Close all subscriptions for a session.

        Shuts down all subscriber queues to signal ``QueueShutDown`` to
        consumers, and clears the replay buffer.

        Args:
            session_id: The session to close subscriptions for.
        """
        self._replay_buffers.pop(session_id, None)

        async with self._lock:
            subscribers = self._subscribers.pop(session_id, [])
            queues = [queue for queue, _scope, _ex in subscribers]

        for queue in queues:
            with contextlib.suppress(Exception):
                queue.shutdown()

    async def get_subscriber_counts(self) -> dict[str, int]:
        """Get subscriber counts per session.

        Returns:
            A snapshot mapping session IDs to subscriber counts.
        """
        async with self._lock:
            return {sid: len(items) for sid, items in self._subscribers.items()}
