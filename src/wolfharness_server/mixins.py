"""Protocol server mixins.

Shared utility mixins for AgentPool protocol server implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import contextlib
from typing import TYPE_CHECKING

import anyio

from wolfharness.agents.events.events import SpawnSessionStart


if TYPE_CHECKING:
    from wolfharness.orchestrator.core import EventBus, EventEnvelope


class ConsumerShutdown(Exception):  # noqa: N818
    """Signal raised by _handle_event() to request graceful consumer loop shutdown."""


class ProtocolEventConsumerMixin(ABC):
    """Mixin providing EventBus consumer lifecycle management for protocol servers.

    This mixin extracts the common pattern of subscribing to the EventBus,
    running an async consumer loop, and cleaning up on shutdown. Protocol
    handlers (ACP, OpenCode, AG-UI, etc.) can inherit from it and implement
    protocol-specific event handling via abstract hooks.

    Subclasses MUST call super().__init__() if they override __init__.

    !!! note
        The mixin does not automatically create child consumers when a
        SpawnSessionStart event is received. Subclasses that want child
        consumers must override _on_spawn_session_start() and call
        start_event_consumer(child_session_id) themselves.
    """

    # Set to True in subclasses that don't need event processing
    # (e.g. stateless HTTP servers like AG-UI and OpenAI API).
    # SpawnSessionStart detection still works regardless of this flag.
    _skip_event_processing: bool = False

    # Set to True during crash recovery replay to skip side-effectful
    # actions (e.g. steer split in OpenCode). Subclasses can check
    # self._replaying to guard against duplicate side effects.
    _replaying: bool = False

    def __init__(self) -> None:
        """Initialize mixin state.

        Sets up internal tracking for CancelScopes, TaskGroups, and
        per-session receive streams.
        """
        super().__init__()
        self._session_scopes: dict[str, anyio.CancelScope] = {}
        self._session_groups: dict[str, anyio.abc.TaskGroup] = {}  # ty: ignore[unresolved-attribute]
        self._consumer_streams: dict[str, asyncio.Queue[EventEnvelope]] = {}
        self._consumer_locks: dict[str, asyncio.Lock] = {}
        self._consumer_tasks: dict[str, asyncio.Task[None] | None] = {}
        self._consumer_done_events: dict[str, anyio.Event] = {}
        self._consumer_task_refs: list[asyncio.Task[None]] = []
        self._consumer_lock_creation_lock: asyncio.Lock = asyncio.Lock()

    @property
    @abstractmethod
    def event_bus(self) -> EventBus:
        """Return the EventBus instance to subscribe to."""

    def _get_subscription_scope(self) -> str:
        """Return the EventBus subscription scope.

        Defaults to "descendants" so that child session events are
        received automatically. Subclasses may override to return
        "session" or "subtree" for different visibility.

        Returns:
            The subscription scope string.
        """
        return "descendants"

    def _get_subscription_replay(self) -> bool:
        """Return whether the session consumer replays buffered events on subscribe.

        Defaults to ``True`` (historical behavior). Protocol servers whose
        client reloads full state via an explicit sync (e.g. OpenCode) may
        override to ``False`` so the consumer aligns with the SSE endpoint's
        first-connect policy — replaying buffered events would re-convert
        native events into duplicate projections for messages the client has
        already loaded. Crash-recovery paths that genuinely need historical
        events must override this back to ``True``.

        Returns:
            Whether the consumer should replay the EventBus replay buffer.
        """
        return True

    def _get_subscription_exclude_source(self) -> frozenset[str] | None:
        """Return sources whose published events this consumer must not receive.

        Defaults to ``None``. Protocol servers that republish their own
        projections into the EventBus (loopback) should override to exclude
        that source so a producer can never re-consume its own output.

        !!! note
            Currently inert in production: the OpenCode server delivers
            projections direct-wire to SSE queues and no producer calls
            ``publish(..., source_hint=...)`` anymore (the loopback bridge
            that did was removed). The hook is kept as defense-in-depth for
            any future producer that re-enters the EventBus; unit tests
            exercise both the hook and the ``exclude_source`` filter.

        Returns:
            A frozenset of ``source_hint`` values to exclude, or ``None``.
        """
        return None

    async def _before_consumer_loop(self, session_id: str) -> None:  # noqa: B027
        """Hook called before the consumer loop starts reading from the stream.

        Subclasses may override to set up per-session context (e.g.
        creating an event converter or adapter).

        Args:
            session_id: The session whose consumer is starting.
        """

    async def _after_consumer_loop(self, session_id: str) -> None:  # noqa: B027
        """Hook called after the consumer loop exits and unsubscribes.

        Only called if the consumer had actually started (i.e.
        _before_consumer_loop completed without raising). Subclasses
        may override to perform per-session cleanup.

        Args:
            session_id: The session whose consumer has stopped.
        """

    async def _on_spawn_session_start(  # noqa: B027
        self, session_id: str, envelope: EventEnvelope
    ) -> None:
        """Hook called when a SpawnSessionStart event is received.

        The default implementation is a no-op. Subclasses may override
        to start child consumers or perform other setup (e.g. registering
        a ToolPart for the subagent in OpenCode).

        !!! note
            This hook is called BEFORE _handle_event() for the same
            SpawnSessionStart event. Exceptions raised here are NOT
            caught by the mixin and will propagate out, triggering
            cleanup in the finally block.

        Args:
            session_id: The session whose consumer received the event.
            envelope: The event envelope containing the spawn session start event.
        """

    @abstractmethod
    async def _handle_event(self, session_id: str, envelope: EventEnvelope) -> None:
        """Handle a single event from the EventBus.

        Subclasses MUST implement this method with protocol-specific
        conversion and delivery logic.

        Args:
            session_id: The session whose consumer received the event.
            envelope: The event envelope to handle.

        Raises:
            ConsumerShutdown: To request graceful loop shutdown.
        """

    async def start_event_consumer(self, session_id: str) -> None:
        """Start an event consumer for a given session.

        This method is idempotent: if a consumer is already running for
        session, it returns immediately. Concurrent calls for the
        same session are serialized by a per-session lock.

        Args:
            session_id: The session to start consuming events for.
        """
        async with self._consumer_lock_creation_lock:
            if session_id not in self._consumer_locks:
                self._consumer_locks[session_id] = asyncio.Lock()

        async with self._consumer_locks[session_id]:
            if session_id in self._session_groups:
                return

            receive_stream = await self.event_bus.subscribe(
                session_id,
                scope=self._get_subscription_scope(),
                replay=self._get_subscription_replay(),
                exclude_source=self._get_subscription_exclude_source(),
            )
            self._consumer_streams[session_id] = receive_stream

            cancel_scope = anyio.CancelScope()
            task_group = anyio.create_task_group()

            self._session_scopes[session_id] = cancel_scope
            self._session_groups[session_id] = task_group

            tg = task_group
            scope = cancel_scope
            done_event = anyio.Event()

            async def _run_consumer() -> None:
                with scope:
                    async with tg:
                        tg.start_soon(self._event_consumer_loop, session_id)
                        await done_event.wait()

            task = asyncio.ensure_future(_run_consumer())
            self._consumer_tasks[session_id] = task
            self._consumer_done_events[session_id] = done_event

    async def stop_event_consumer(self, session_id: str) -> None:
        """Stop an event consumer for a given session.

        Cancels the session's CancelScope, exits the TaskGroup,
        unsubscribes from the EventBus, and cleans up internal state.
        Safe to call even if no consumer is running for the session.

        Args:
            session_id: The session to stop consuming events for.
        """
        cancel_scope = self._session_scopes.get(session_id)
        if cancel_scope is not None:
            cancel_scope.cancel()

        # Wait for the consumer task to fully exit so its finally block
        # (which pops _session_groups etc.) has run before we return.
        # Without this, a subsequent start_event_consumer races with the
        # old finally block, which clobbers the new state.
        consumer_task = self._consumer_tasks.pop(session_id, None)
        if consumer_task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await consumer_task

        # Cleanup after CancelScope ensures proper termination
        self._session_scopes.pop(session_id, None)
        self._session_groups.pop(session_id, None)
        stream = self._consumer_streams.pop(session_id, None)
        if stream is not None:
            with contextlib.suppress(Exception):
                await self.event_bus.unsubscribe(session_id, stream)

        self._consumer_locks.pop(session_id, None)
        self._consumer_done_events.pop(session_id, None)

    async def _event_consumer_loop(self, session_id: str) -> None:
        """Read events from the subscription stream and dispatch to hooks.

        Uses ``drain_and_merge()`` to consume events from the subscription
        stream, which handles subscriber-side event coalescing by batching
        and merging consecutive same-type events (e.g., ``PartDeltaEvent``
        text chunks) before they reach ``_handle_event()``.

        The loop exits gracefully when the receive stream reaches
        EndOfStream (send stream closed), when ConsumerShutdown is raised
        from _handle_event(), or when the task is cancelled.

        SpawnSessionStart events are dispatched to BOTH
        _on_spawn_session_start() AND _handle_event(). All other
        events go only to _handle_event().

        Cleanup (_after_consumer_loop) is performed in a finally
        block regardless of how the loop exits.

        Args:
            session_id: The session whose events to consume.
        """
        stream = self._consumer_streams.get(session_id)
        if stream is None:
            return

        started = False
        try:
            await self._before_consumer_loop(session_id)
            started = True

            # Deferred import to avoid circular dependency:
            # wolfharness.orchestrator.core -> wolfharness_server.* -> mixins.py
            from wolfharness.orchestrator.core import drain_and_merge

            async for envelope in drain_and_merge(stream):
                if isinstance(envelope.event, SpawnSessionStart):
                    await self._on_spawn_session_start(session_id, envelope)

                if not self._skip_event_processing:
                    try:
                        await self._handle_event(session_id, envelope)
                    except ConsumerShutdown:
                        break
        finally:
            done_event = self._consumer_done_events.pop(session_id, None)
            if done_event is not None:
                done_event.set()
            self._session_scopes.pop(session_id, None)
            self._session_groups.pop(session_id, None)
            stream = self._consumer_streams.pop(session_id, None)
            if stream is not None:
                with contextlib.suppress(Exception):
                    await self.event_bus.unsubscribe(session_id, stream)
            if started:
                await self._after_consumer_loop(session_id)
