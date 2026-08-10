"""Global routes (health, events)."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import TYPE_CHECKING, Any, cast

import anyio
from fastapi import APIRouter, Query
from sse_starlette.sse import EventSourceResponse

from wolfharness import log
from wolfharness.agents.events.events import CustomEvent
from wolfharness_server.opencode_server.dependencies import StateDep
from wolfharness_server.opencode_server.models import Event, GlobalEvent, HealthResponse
from wolfharness_server.opencode_server.models.app import (
    DiagnosticResponse,
    DisposeResponse,
    UpgradeResponse,
)
from wolfharness_server.opencode_server.models.events import (
    MessageUpdatedEvent,
    PartUpdatedEvent,
    ServerConnectedEvent,
    ServerHeartbeatEvent,
    SessionCreatedEvent,
    SessionErrorEvent,
    SessionIdProperties,
    SessionUpdatedEvent,
)
from wolfharness_server.opencode_server.routes.routing import (
    RoutingCheckResponse,
    tui_event_filter,
)


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from wolfharness_server.opencode_server.state import ServerState


logger = log.get_logger(__name__)
router = APIRouter(tags=["global"])

VERSION = "0.1.0"


@router.get("/global/health")
async def get_health() -> HealthResponse:
    """Get server health status."""
    return HealthResponse(healthy=True, version=VERSION)


@router.get("/global/diagnostic")
async def get_diagnostic(state: StateDep) -> DiagnosticResponse:
    """Get server diagnostic information.

    Returns directory, project, subscriber count, and server version.
    """
    if state.working_dir is None:
        return DiagnosticResponse(
            directory=None,
            project="",
            subscribers=len(state.event_subscribers),
            server_version=VERSION,
        )

    factory = state.get_event_factory()
    return DiagnosticResponse(
        directory=state.working_dir,
        project=factory._project,
        subscribers=len(state.event_subscribers),
        server_version=VERSION,
    )


@router.post("/global/dispose")
async def post_global_dispose() -> DisposeResponse:
    """Acknowledge OpenCode dispose requests without stopping the server."""
    return DisposeResponse(message="dispose acknowledged (no-op)")


@router.post("/global/upgrade")
async def post_global_upgrade() -> UpgradeResponse:
    """Acknowledge OpenCode upgrade requests without performing an upgrade."""
    return UpgradeResponse(message="upgrade not supported (stub)")


def _extract_session_id(event: Event) -> str | None:
    """Extract session ID from various event types.

    Uses a combination of:
    - ``isinstance(event.properties, SessionIdProperties)`` for the common
      case where session_id lives directly on the properties model.
    - Explicit match arms for special-path events where session_id is
      nested deeper (e.g., ``properties.info.id``,
      ``properties.info.session_id``, ``properties.part.session_id``).
    - ``SessionErrorEvent`` handled separately because its session_id is
      ``str | None`` (not ``str``), so its properties don't inherit from
      ``SessionIdProperties``.

    Unrecognized event types return None without warning, since many
    events genuinely have no session association.
    """
    # Special-path events: session_id is nested, not at properties.session_id
    match event:
        case SessionCreatedEvent(properties=props):
            session_id: str | None = props.info.id  # ty: ignore[unresolved-attribute]
        case SessionUpdatedEvent(properties=props):
            session_id = props.info.id
        case MessageUpdatedEvent(properties=props):
            session_id = props.info.session_id
        case PartUpdatedEvent(properties=props):
            session_id = props.part.session_id
        case SessionErrorEvent(properties=props):
            session_id = props.session_id
        case _:
            # Common path: properties inherit from SessionIdProperties
            if isinstance(event.properties, SessionIdProperties):
                return event.properties.session_id
            # No session association (server events, file events, pty events, etc.)
            return None

    return session_id


class GlobalEventFactory:
    """Creates GlobalEvent envelope JSON from Event instances.

    Stored on ServerState since directory/project don't change during
    the server's lifetime. Created lazily on first access.
    """

    def __init__(self, directory: str, project: str) -> None:
        """Initialize with directory and project routing metadata.

        Args:
            directory: Working directory for event routing
            project: Project identifier for event routing
        """
        self._directory = directory
        self._project = project

    def wrap(self, event: Event) -> str:
        """Wrap an Event in a GlobalEvent envelope JSON string.

        Args:
            event: The event to wrap

        Returns:
            JSON string with directory, project, and payload keys.
        """
        payload = _event_to_dict(event)
        envelope: dict[str, Any] = {
            "directory": self._directory,
            "project": self._project,
            "payload": payload,
        }
        return json.dumps(envelope, ensure_ascii=False)


def _event_to_dict(event: Event) -> dict[str, Any]:
    """Convert an Event to a dict with sessionId injected at top level.

    This is the dict-building half of serialization; the caller decides
    whether to wrap it in a payload envelope and when to call json.dumps.

    Injects sessionId (lowercase 'd') at the top level for subagent
    session tracking, separate from the alias-converted sessionID that
    appears inside properties.

    Args:
        event: The event to convert

    Returns:
        Dict with the event data and optional sessionId field.
    """
    event_data = event.model_dump(by_alias=True, exclude_none=True)
    session_id = _extract_session_id(event)
    if session_id is not None:
        event_data["sessionId"] = session_id
    return event_data


def _serialize_event(event: Event, wrap_payload: bool = False) -> str:
    r"""Serialize event, optionally wrapping in payload structure.

    Thin convenience wrapper around _event_to_dict + json.dumps.
    Uses ensure_ascii=False to preserve Unicode characters (Chinese, emoji, etc.)
    in the JSON output instead of escaping them as \uXXXX sequences.

    Args:
        event: The event to serialize
        wrap_payload: Whether to wrap in a {"payload": ...} structure

    Returns:
        JSON string of the serialized event data.
    """
    event_data = _event_to_dict(event)
    if wrap_payload:
        return json.dumps({"payload": event_data}, ensure_ascii=False)
    return json.dumps(event_data, ensure_ascii=False)


async def _event_generator(  # noqa: PLR0915
    state: ServerState, *, wrap_payload: bool = False, last_event_id: str | None = None
) -> AsyncGenerator[dict[str, Any]]:
    """Generate SSE events for connected clients.

    Events are received from the SessionPool EventBus via a global
    subscription (``"__global_sse__"``) and streamed to SSE clients.

    Args:
        state: The server state holding subscribers and event factory
        wrap_payload: Whether to wrap events in GlobalEvent envelopes
        last_event_id: The last event ID received by the client, for replay
    """
    factory = state.get_event_factory() if wrap_payload else None
    # Track connection for diagnostic subscriber counting.
    # The queue is maintained only for the event_subscribers counter;
    # actual event delivery now flows exclusively through the EventBus.
    _sentinel_queue: asyncio.Queue[Event] = asyncio.Queue()
    state.event_subscribers.append(_sentinel_queue)
    subscriber_count = len(state.event_subscribers)
    logger.info("SSE: New client connected (total subscribers: %s)", subscriber_count)

    # Trigger first subscriber callback if this is the first connection.
    if (
        subscriber_count == 1
        and not state._first_subscriber_triggered
        and state.on_first_subscriber is not None
    ):
        state._first_subscriber_triggered = True
        state.create_background_task(state.on_first_subscriber(), name="on_first_subscriber")

    # Subscribe to EventBus for global SSE events.
    # When the client provides a Last-Event-ID, use conditional replay: only
    # events with event_id > last_event_id are replayed. When no Last-Event-ID
    # is provided, replay all buffered events (default behavior).
    parsed_last_event_id: int | None = int(last_event_id) if last_event_id is not None else None
    event_bus_stream: asyncio.Queue[Any] | None = None
    session_controller = getattr(state, "session_controller", None)
    if session_controller is not None:
        session_pool = getattr(state.pool, "session_pool", None)
        if session_pool is not None:
            event_bus = session_pool.event_bus
            event_bus_stream = await event_bus.subscribe(
                "__global_sse__",
                scope="all",
                replay=parsed_last_event_id is not None,
                last_event_id=parsed_last_event_id,
            )

    try:
        # Send initial connected event
        connected = ServerConnectedEvent()
        data = _serialize_event(connected, wrap_payload=wrap_payload)
        logger.info("SSE: Sending connected event", data=data)
        yield {"data": data, "id": "0"}

        if event_bus_stream is None:
            while True:
                await asyncio.sleep(10.0)
                heartbeat = ServerHeartbeatEvent()
                data = _serialize_event(heartbeat, wrap_payload=wrap_payload)
                yield {"data": data}
        else:
            while True:
                try:
                    with anyio.fail_after(10.0):
                        raw_event = await event_bus_stream.get()
                except TimeoutError:
                    heartbeat = ServerHeartbeatEvent()
                    data = _serialize_event(heartbeat, wrap_payload=wrap_payload)
                    yield {"data": data}
                    continue
                except asyncio.QueueShutDown:
                    break

                from wolfharness.orchestrator.core import EventEnvelope

                inner_event: Any
                inner_event = raw_event.event if isinstance(raw_event, EventEnvelope) else raw_event

                if isinstance(inner_event, CustomEvent):
                    if inner_event.event_data is None:
                        continue
                    event = cast(Event, inner_event.event_data)
                else:
                    event = inner_event

                if not hasattr(event, "type"):
                    continue
                if factory is not None and not isinstance(
                    event, ServerHeartbeatEvent | ServerConnectedEvent
                ):
                    data = factory.wrap(event)
                elif wrap_payload:
                    data = _serialize_event(event, wrap_payload=True)
                else:
                    data = _serialize_event(event)
                logger.debug(
                    "SSE: Sending event",
                    event_type=getattr(event, "type", "unknown"),
                    session_id=_extract_session_id(event) or "-",
                )
                event_id = raw_event.event_id if isinstance(raw_event, EventEnvelope) else 0
                yield {"data": data, "id": str(event_id)}
    finally:
        if event_bus_stream is not None:
            session_pool = getattr(state.pool, "session_pool", None)
            if session_pool is not None:
                with contextlib.suppress(Exception):
                    await session_pool.event_bus.unsubscribe("__global_sse__", event_bus_stream)
        # Remove from subscriber count
        with contextlib.suppress(ValueError):
            state.event_subscribers.remove(_sentinel_queue)
        # Cancel any pending questions when the SSE client disconnects
        if session_controller is not None:
            cancelled = session_controller.cancel_all_pending_questions()
        else:
            cancelled = state.cancel_all_pending_questions()
        if cancelled:
            logger.info(
                "SSE: Cancelled pending questions on disconnect",
                question_ids=cancelled,
            )
        logger.info(
            "SSE: Client disconnected",
            remaining_subscribers=len(state.event_subscribers),
        )


@router.get("/global/event")
async def get_global_events(
    state: StateDep,
    last_event_id: str | None = Query(None),
) -> EventSourceResponse:
    """Get global events as SSE stream (uses payload wrapper)."""
    return EventSourceResponse(
        _event_generator(state, wrap_payload=True, last_event_id=last_event_id),
        sep="\n",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/event")
async def get_events(
    state: StateDep,
    last_event_id: str | None = Query(None),
) -> EventSourceResponse:
    """Get events as SSE stream (no payload wrapper)."""
    return EventSourceResponse(
        _event_generator(state, wrap_payload=False, last_event_id=last_event_id),
        sep="\n",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/global/routing-check", response_model=RoutingCheckResponse)
async def get_routing_check(
    state: StateDep,
    directory: str,
    project_directory: str | None = None,
) -> RoutingCheckResponse:
    """Check whether an event would pass the OpenCode TUI routing filter.

    Diagnostic endpoint that constructs a synthetic GlobalEvent with the
    given directory and runs it through the 3-rule TUI event
    routing filter. Returns whether the event would pass and why.

    Args:
        state: Server state (injected dependency).
        directory: The event's directory field.
        project_directory: The project directory to match against
            (defaults to state.working_dir).

    Returns:
        RoutingCheckResponse with would_pass and reason fields.
    """
    effective_project_dir = (
        project_directory if project_directory is not None else state.working_dir
    )
    event = GlobalEvent(directory=directory, payload={})
    would_pass, reason = tui_event_filter(event, effective_project_dir)
    return RoutingCheckResponse(would_pass=would_pass, reason=reason)
