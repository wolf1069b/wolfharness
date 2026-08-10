"""Event bridge that publishes OpenCode events to the SessionPool EventBus.

Provides :class:`OpenCodeEventBridge` which republishes events destined for
OpenCode clients to the SessionPool's :class:`EventBus` for EventBus-based
routing to all consumers.

**Event flow**

1. Caller invokes ``state.broadcast_event(event)`` (or ``bridge.publish(event)``).
2. Bridge extracts ``session_id`` from the event's ``properties``.
3. If a session_id is present, the event is wrapped in a
   :class:`CustomEvent` and published to the EventBus for that session.
4. EventBus subscribers (status bridges, protocol adapters, SSE clients)
   receive the wrapped event.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from wolfharness.agents.events.events import CustomEvent
from wolfharness.log import get_logger


if TYPE_CHECKING:
    from wolfharness.orchestrator.core import EventBus
    from wolfharness_server.opencode_server.models.events import Event
    from wolfharness_server.opencode_server.state import ServerState


logger = get_logger(__name__)


class OpenCodeEventBridge:
    """Bridge that publishes OpenCode events to the SessionPool EventBus.

    Every event published through this bridge is made available on the
    SessionPool EventBus so that all consumers (SSE clients, status bridges,
    protocol adapters) receive events via EventBus subscriptions.

    Args:
        state: The OpenCode server state.
        event_bus: The SessionPool EventBus to republish events into.
    """

    def __init__(self, state: ServerState, event_bus: EventBus) -> None:
        """Initialize the bridge."""
        self._state = state
        self._event_bus = event_bus

    async def publish(self, event: Event) -> None:
        """Publish an event to the EventBus.

        Extracts ``session_id`` from the event properties. If a session_id
        is present, the event is wrapped in a :class:`CustomEvent` and
        published to the EventBus for that session. EventBus subscribers
        (SSE clients, status bridges, protocol adapters) receive the
        wrapped event.

        Args:
            event: An OpenCode protocol event (e.g. ``SessionStatusEvent``,
                ``PartUpdatedEvent``, ``MessageUpdatedEvent``).
        """
        # Step 1: extract session_id
        session_id = self._extract_session_id(event)
        if session_id is None:
            return

        # Step 2: wrap and republish to EventBus
        wrapped = self._wrap_event(event)
        try:
            await self._event_bus.publish(session_id, wrapped)
        except Exception:
            logger.exception(
                "Failed to republish event to EventBus",
                session_id=session_id,
                event_type=getattr(event, "type", "unknown"),
            )

    @staticmethod
    def _extract_session_id(event: Event) -> str | None:
        """Extract session_id from an OpenCode event's properties.

        Most session-scoped events inherit from ``SessionIdProperties`` and
        expose ``properties.session_id`` directly.  However, some event types
        nest session_id deeper:

        - ``PartUpdatedEvent`` → ``properties.part.session_id``
        - ``SessionCreatedEvent`` / ``SessionUpdatedEvent`` → ``properties.info.id``
        - ``MessageUpdatedEvent`` → ``properties.info.session_id``

        This method tries each known path in order and returns the first
        non-None ``str`` result.

        Args:
            event: The OpenCode event to inspect.

        Returns:
            The session ID string, or ``None`` if the event is global.
        """
        properties = getattr(event, "properties", None)
        if properties is None:
            return None

        # Fast path: direct session_id (SessionIdProperties subclasses)
        session_id = getattr(properties, "session_id", None)
        if isinstance(session_id, str):
            return session_id

        # PartUpdatedEvent: session_id is at properties.part.session_id
        part = getattr(properties, "part", None)
        if part is not None:
            sid = getattr(part, "session_id", None)
            if isinstance(sid, str):
                return sid

        # SessionCreated / SessionUpdated: session_id is at properties.info.id
        info = getattr(properties, "info", None)
        if info is not None:
            sid = getattr(info, "id", None)
            if isinstance(sid, str):
                return sid

        return None

    @staticmethod
    def _wrap_event(event: Event) -> CustomEvent[Any]:
        """Wrap an OpenCode event in a :class:`CustomEvent`.

        The wrapped event preserves the original event as ``event_data`` and
        uses the OpenCode event type (prefixed with ``opencode:``) as the
        custom event type.  This makes it easy for EventBus consumers to
        distinguish OpenCode protocol events from native agent events.

        Args:
            event: The OpenCode event to wrap.

        Returns:
            A :class:`CustomEvent` carrying the original OpenCode event.
        """
        event_type = getattr(event, "type", "opencode:unknown")
        return CustomEvent(
            event_data=event,
            event_type=f"opencode:{event_type}",
            source="opencode_event_bridge",
        )
