"""Tests for _serialize_event, GlobalEvent model, and GlobalEventFactory."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from wolfharness_server.opencode_server.models import GlobalEvent
from wolfharness_server.opencode_server.models.app import ProjectTime
from wolfharness_server.opencode_server.models.common import TimeCreated
from wolfharness_server.opencode_server.models.events import (
    CommandExecutedEvent,
    Event,
    FileEditedEvent,
    FileWatcherUpdatedEvent,
    LspClientDiagnosticsEvent,
    LspUpdatedEvent,
    McpToolsChangedEvent,
    MessageRemovedEvent,
    MessageUpdatedEvent,
    PartDeltaEvent,
    PartRemovedEvent,
    PartUpdatedEvent,
    PermissionRequestEvent,
    PermissionResolvedEvent,
    PermissionUpdatedEvent,
    Project,
    ProjectUpdatedEvent,
    PtyCreatedEvent,
    PtyDeletedEvent,
    PtyExitedEvent,
    PtyUpdatedEvent,
    QuestionAskedEvent,
    QuestionRejectedEvent,
    QuestionRepliedEvent,
    ServerConnectedEvent,
    ServerHeartbeatEvent,
    SessionCompactedEvent,
    SessionCreatedEvent,
    SessionDeletedEvent,
    SessionDiffEvent,
    SessionErrorEvent,
    SessionIdleEvent,
    SessionIdProperties,
    SessionStatusEvent,
    SessionUpdatedEvent,
    Todo,
    TodoUpdatedEvent,
    TuiCommandExecuteEvent,
    TuiPromptAppendEvent,
    TuiSessionSelectEvent,
    TuiToastShowEvent,
    VcsBranchUpdatedEvent,
)
from wolfharness_server.opencode_server.models.message import (
    UserMessage,
)
from wolfharness_server.opencode_server.models.parts import Part, TextPart
from wolfharness_server.opencode_server.models.pty import PtyInfo
from wolfharness_server.opencode_server.models.question import (
    QuestionInfo,
    QuestionOption,
)
from wolfharness_server.opencode_server.models.session import (
    Session,
    TimeCreatedUpdated,
)
from wolfharness_server.opencode_server.routes.global_routes import (
    GlobalEventFactory,
    _event_generator,
    _extract_session_id,
    _serialize_event,
)
from wolfharness_server.opencode_server.state import ServerState


if TYPE_CHECKING:
    from httpx import AsyncClient


# =============================================================================
# _serialize_event baseline tests
# =============================================================================


def test_serialize_event_session_id_injection() -> None:
    """SessionId is injected at top level when event has a session."""
    event = SessionStatusEvent.create(session_id="abc", status_type="busy")
    result = _serialize_event(event, wrap_payload=False)
    data = json.loads(result)
    assert data["sessionId"] == "abc"


def test_serialize_event_no_session_id() -> None:
    """No sessionId key when event has no session."""
    event = ServerConnectedEvent()
    result = _serialize_event(event, wrap_payload=False)
    data = json.loads(result)
    assert "sessionId" not in data


def test_serialize_event_wrap_payload_true() -> None:
    """wrap_payload=True nests event data under 'payload' key."""
    event = ServerConnectedEvent()
    result = _serialize_event(event, wrap_payload=True)
    data = json.loads(result)
    assert "payload" in data
    assert data["payload"]["type"] == "server.connected"


def test_serialize_event_unicode_preserved() -> None:
    r"""Unicode characters are preserved (not \uXXXX escaped)."""
    event = SessionStatusEvent.create(session_id="你好", status_type="idle")
    result = _serialize_event(event, wrap_payload=False)
    assert "你好" in result
    assert "\\u" not in result


def test_serialize_event_camel_case_aliases() -> None:
    """Model fields use camelCase aliases in serialized output."""
    event = SessionStatusEvent.create(session_id="abc", status_type="busy")
    result = _serialize_event(event, wrap_payload=False)
    data = json.loads(result)
    # session_id → sessionID via convert() alias generator
    props = data["properties"]
    assert "sessionID" in props


# =============================================================================
# GlobalEvent model and GlobalEventFactory tests
# =============================================================================


def test_global_event_model_construction() -> None:
    """GlobalEvent stores directory, project, and payload correctly."""
    payload = {"type": "test"}
    event = GlobalEvent(directory="/tmp/test", project="abc123", payload=payload)
    dumped = event.model_dump(by_alias=True, exclude_none=True)
    assert dumped["directory"] == "/tmp/test"
    assert dumped["project"] == "abc123"
    assert dumped["payload"] == payload


def test_global_event_factory_wrap() -> None:
    """Factory.wrap() produces JSON with directory, project, payload."""
    factory = GlobalEventFactory(directory="/tmp", project="abc")
    event = SessionStatusEvent.create(session_id="sid1", status_type="idle")
    result = factory.wrap(event)
    data = json.loads(result)
    assert data["directory"] == "/tmp"
    assert data["project"] == "abc"
    assert isinstance(data["payload"], dict)


def test_global_event_factory_session_id_in_payload() -> None:
    """SessionId is injected inside payload by GlobalEventFactory.wrap."""
    factory = GlobalEventFactory(directory="/tmp", project="abc")
    event = SessionStatusEvent.create(session_id="sid1", status_type="idle")
    result = factory.wrap(event)
    data = json.loads(result)
    assert data["payload"]["sessionId"] == "sid1"


def test_global_event_factory_unicode_preserved() -> None:
    """Factory.wrap() preserves Unicode characters in output."""
    factory = GlobalEventFactory(directory="/tmp", project="abc")
    event = SessionStatusEvent.create(session_id="会话1", status_type="busy")
    result = factory.wrap(event)
    assert "会话1" in result
    assert "\\u" not in result


def test_global_event_model_project_global() -> None:
    """GlobalEvent with project='global' preserves the value."""
    event = GlobalEvent(directory="/tmp", project="global", payload={})
    dumped = event.model_dump(by_alias=True, exclude_none=True)
    assert dumped["project"] == "global"


def test_global_event_factory_wrap_returns_string() -> None:
    """Factory.wrap() returns a str (JSON string)."""
    factory = GlobalEventFactory(directory="/tmp", project="abc")
    event = ServerHeartbeatEvent()
    result = factory.wrap(event)
    assert isinstance(result, str)


# =============================================================================
# _event_generator integration tests
# =============================================================================


class _MockEventBus:
    """Minimal EventBus stub that supports subscribe/unsubscribe.

    Supports scope="all" subscriptions which receive events from any session_id,
    matching the real EventBus._should_receive behavior.

    Uses asyncio.Queue (matching the real EventBus) so subscribers receive
    objects with .get()/.get_nowait() instead of .receive()/.receive_nowait().
    """

    _STREAM_BUFFER_SIZE: int = 1024

    def __init__(self) -> None:
        self._subscribers: dict[str, list[tuple[asyncio.Queue[Any], str]]] = {}

    async def subscribe(
        self,
        session_id: str,
        scope: str = "session",
        *,
        replay: bool = True,
        last_event_id: int | None = None,
    ) -> asyncio.Queue[Any]:
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=self._STREAM_BUFFER_SIZE)
        self._subscribers.setdefault(session_id, []).append((queue, scope))
        return queue

    async def unsubscribe(self, session_id: str, queue: asyncio.Queue[Any]) -> None:
        if session_id in self._subscribers:
            self._subscribers[session_id] = [
                (q, sc) for q, sc in self._subscribers[session_id] if q is not queue
            ]
            if not self._subscribers[session_id]:
                del self._subscribers[session_id]
        with contextlib.suppress(asyncio.QueueShutDown):
            queue.shutdown()

    async def publish(self, session_id: str, event: Any) -> None:
        from wolfharness.orchestrator.core import EventEnvelope

        envelope = EventEnvelope(source_session_id=session_id, event=event)
        for subscriber_sid, subscribers in self._subscribers.items():
            for queue, scope in subscribers:
                if scope == "all" or subscriber_sid == session_id:
                    with contextlib.suppress(asyncio.QueueFull):
                        queue.put_nowait(envelope)


class _MockSessionPool:
    """Minimal SessionPool stub providing event_bus."""

    def __init__(self, event_bus: _MockEventBus) -> None:
        self.event_bus = event_bus


class _MockPool:
    """Minimal pool stub providing session_pool."""

    def __init__(self, session_pool: _MockSessionPool) -> None:
        self.session_pool = session_pool


class _MockSessionController:
    """Minimal session controller stub."""

    def cancel_all_pending_questions(self) -> list[str]:
        return []


class _MockState:
    """Minimal ServerState-like object for _event_generator tests."""

    def __init__(self, working_dir: str = "/tmp/test_wd") -> None:
        self.working_dir = working_dir
        self.event_subscribers: list[asyncio.Queue[Event]] = []
        self._event_factory: GlobalEventFactory | None = None
        self._first_subscriber_triggered = False
        self.on_first_subscriber: Any = None
        # Provide EventBus-based infrastructure so _event_generator uses
        # the EventBus path instead of the heartbeat-only fallback.
        self._mock_event_bus = _MockEventBus()
        self.session_controller = _MockSessionController()
        self.pool = _MockPool(_MockSessionPool(self._mock_event_bus))

    def get_event_factory(self) -> GlobalEventFactory:
        if self._event_factory is None:
            from wolfharness_storage.opencode_provider import helpers

            self._event_factory = GlobalEventFactory(
                directory=self.working_dir,
                project=helpers.compute_project_id(self.working_dir),
            )
        return self._event_factory

    def create_background_task(self, coro: Any, name: str = "") -> asyncio.Task[Any]:
        return asyncio.ensure_future(coro)

    def cancel_all_pending_questions(self) -> list[str]:
        """No-op mock for SSE disconnect handler."""
        return []

    # get_next_event_id() removed — event_id now comes from EventBus
    # (EventEnvelope.event_id assigned at publish time, not ServerState).


async def _collect_events(
    state: _MockState,
    wrap_payload: bool,
    events_to_send: list[Event],
) -> list[dict[str, Any]]:
    """Collect SSE items from _event_generator with given events."""
    results: list[dict[str, Any]] = []
    gen = _event_generator(state, wrap_payload=wrap_payload)
    # Get the initial connected event
    item = await gen.__anext__()
    results.append(json.loads(item["data"]))
    # Send additional events through the mock EventBus
    for event in events_to_send:
        await state._mock_event_bus.publish("__global_sse__", event)
        item = await gen.__anext__()
        results.append(json.loads(item["data"]))
    return results


@pytest.mark.anyio
async def test_global_event_server_connected_is_payload_wrapped() -> None:
    """First event from /global/event keeps only the payload wrapper."""
    state = _MockState()
    events = await _collect_events(state, wrap_payload=True, events_to_send=[])
    # Only the connected event
    assert len(events) == 1
    connected = events[0]
    assert connected["payload"]["type"] == "server.connected"
    assert "directory" not in connected
    assert "project" not in connected


@pytest.mark.anyio
async def test_global_event_wraps_regular_events_in_envelope() -> None:
    """/global/event wraps SessionStatusEvent in GlobalEvent envelope."""
    state = _MockState()
    session_evt = SessionStatusEvent.create(session_id="s1", status_type="busy")
    events = await _collect_events(state, wrap_payload=True, events_to_send=[session_evt])
    assert len(events) == 2
    wrapped = events[1]
    assert "directory" in wrapped
    assert "project" in wrapped
    assert "payload" in wrapped
    assert wrapped["payload"]["type"] == "session.status"


@pytest.mark.anyio
async def test_global_event_heartbeat_is_payload_wrapped() -> None:
    """/global/event keeps only the payload wrapper for heartbeat."""
    state = _MockState()
    hb = ServerHeartbeatEvent()
    events = await _collect_events(state, wrap_payload=True, events_to_send=[hb])
    assert len(events) == 2
    heartbeat = events[1]
    assert heartbeat["payload"]["type"] == "server.heartbeat"
    assert "directory" not in heartbeat
    assert "project" not in heartbeat


@pytest.mark.anyio
async def test_event_endpoint_all_events_are_bare() -> None:
    """/event sends connected, heartbeat, and session events all as bare."""
    state = _MockState()
    hb = ServerHeartbeatEvent()
    session_evt = SessionStatusEvent.create(session_id="s2", status_type="idle")
    events = await _collect_events(state, wrap_payload=False, events_to_send=[hb, session_evt])
    assert len(events) == 3
    for evt in events:
        # None should have envelope wrapper keys
        assert "directory" not in evt
        assert "project" not in evt
        assert "payload" not in evt
    assert events[0]["type"] == "server.connected"
    assert events[1]["type"] == "server.heartbeat"
    assert events[2]["type"] == "session.status"


@pytest.mark.anyio
async def test_global_events_have_no_session_id() -> None:
    """ServerConnectedEvent and ServerHeartbeatEvent lack sessionId."""
    state = _MockState()
    hb = ServerHeartbeatEvent()
    events = await _collect_events(state, wrap_payload=True, events_to_send=[hb])
    assert "sessionId" not in events[0]  # top-level envelope
    assert "sessionId" not in events[0]["payload"]  # server.connected payload
    assert "sessionId" not in events[1]  # top-level envelope
    assert "sessionId" not in events[1]["payload"]  # server.heartbeat payload


@pytest.mark.anyio
async def test_global_event_directory_matches_working_dir() -> None:
    """Envelope directory field matches the server working directory."""
    wd = "/custom/working/dir"
    state = _MockState(working_dir=wd)
    session_evt = SessionStatusEvent.create(session_id="s3", status_type="retry")
    events = await _collect_events(state, wrap_payload=True, events_to_send=[session_evt])
    wrapped = events[1]
    assert wrapped["directory"] == wd


@pytest.mark.anyio
async def test_multiple_events_maintain_correct_wrapping() -> None:
    """Sequence of wrapped events all have correct format."""
    state = _MockState()
    session_evt = SessionStatusEvent.create(session_id="s4", status_type="busy")
    hb = ServerHeartbeatEvent()
    session_evt2 = SessionStatusEvent.create(session_id="s5", status_type="idle")
    events = await _collect_events(
        state,
        wrap_payload=True,
        events_to_send=[session_evt, hb, session_evt2],
    )
    assert len(events) == 4
    # [0] connected — payload wrapped, no routing metadata
    assert events[0]["payload"]["type"] == "server.connected"
    assert "directory" not in events[0]
    # [1] session status — wrapped
    assert "payload" in events[1]
    assert events[1]["payload"]["type"] == "session.status"
    # [2] heartbeat — payload wrapped, no routing metadata
    assert events[2]["payload"]["type"] == "server.heartbeat"
    assert "directory" not in events[2]
    # [3] session status — wrapped
    assert "payload" in events[3]
    assert events[3]["payload"]["type"] == "session.status"


# =============================================================================
# /event endpoint backward compatibility tests
# =============================================================================


@pytest.mark.anyio
async def test_event_endpoint_no_global_event_fields() -> None:
    """wrap_payload=False events have no directory/project."""
    state = _MockState()
    session_evt = SessionStatusEvent.create(session_id="bc1", status_type="busy")
    events = await _collect_events(state, wrap_payload=False, events_to_send=[session_evt])
    for evt in events:
        assert "directory" not in evt
        assert "project" not in evt


@pytest.mark.anyio
async def test_event_endpoint_no_payload_wrapper() -> None:
    """No payload wrapper key; event data is at top level."""
    state = _MockState()
    session_evt = SessionStatusEvent.create(session_id="bc2", status_type="idle")
    events = await _collect_events(state, wrap_payload=False, events_to_send=[session_evt])
    session_data = events[1]
    assert "payload" not in session_data
    # Event fields directly at top level
    assert session_data["type"] == "session.status"


@pytest.mark.anyio
async def test_event_endpoint_session_id_at_top_level() -> None:
    """SessionId present at top level for session events."""
    state = _MockState()
    session_evt = SessionStatusEvent.create(session_id="bc3", status_type="busy")
    events = await _collect_events(state, wrap_payload=False, events_to_send=[session_evt])
    session_data = events[1]
    assert session_data["sessionId"] == "bc3"


@pytest.mark.anyio
async def test_event_endpoint_unicode_preserved() -> None:
    r"""Unicode characters not escaped as \uXXXX in /event output."""
    state = _MockState()
    session_evt = SessionStatusEvent.create(session_id="会话测试", status_type="idle")
    gen = _event_generator(state, wrap_payload=False)
    # Consume connected event
    await gen.__anext__()
    # Send unicode session event via mock EventBus
    await state._mock_event_bus.publish("__global_sse__", session_evt)
    item = await gen.__anext__()
    raw_data = item["data"]
    assert "会话测试" in raw_data
    assert "\\u" not in raw_data


# =============================================================================
# SSE integration tests for /global/event
# =============================================================================


async def _collect_real_events(
    state: ServerState,
    wrap_payload: bool,
    events_to_send: list[Event],
) -> list[dict[str, Any]]:
    """Collect SSE items from _event_generator with real ServerState."""
    results: list[dict[str, Any]] = []
    gen = _event_generator(state, wrap_payload=wrap_payload)
    # Get the initial connected event
    item = await gen.__anext__()
    results.append(json.loads(item["data"]))
    # Send additional events through the real broadcast system
    for event in events_to_send:
        await state.broadcast_event(event)
        # Yield control so the event can propagate through subscriber queues
        await asyncio.sleep(0.01)
        item = await gen.__anext__()
        results.append(json.loads(item["data"]))
    return results


@pytest.mark.integration
@pytest.mark.anyio
async def test_global_event_integration_envelope_fields(
    server_state: ServerState,
) -> None:
    """Test /global/event returns SSE with GlobalEvent envelope."""
    event = SessionStatusEvent.create(session_id="s1", status_type="busy")
    results = await _collect_real_events(server_state, wrap_payload=True, events_to_send=[event])
    assert len(results) == 2
    received = results[1]
    assert "directory" in received
    assert "project" in received
    assert "payload" in received


@pytest.mark.integration
@pytest.mark.anyio
async def test_global_event_integration_directory_matches_working_dir(
    server_state: ServerState,
) -> None:
    """Test directory field matches the resolved server working directory."""
    event = SessionStatusEvent.create(session_id="s2", status_type="idle")
    results = await _collect_real_events(server_state, wrap_payload=True, events_to_send=[event])
    received = results[1]
    assert received["directory"] == str(Path(server_state.working_dir).resolve())


@pytest.mark.integration
@pytest.mark.anyio
async def test_global_event_integration_project_computed(
    server_state: ServerState,
) -> None:
    """Test project field is computed via compute_project_id."""
    from wolfharness_storage.opencode_provider.helpers import compute_project_id

    event = SessionStatusEvent.create(session_id="s3", status_type="busy")
    results = await _collect_real_events(server_state, wrap_payload=True, events_to_send=[event])
    received = results[1]
    expected_project = compute_project_id(server_state.working_dir)
    assert received["project"] == expected_project


@pytest.mark.integration
@pytest.mark.anyio
async def test_global_event_routing_ignores_agent_execution_cwd(
    server_state: ServerState,
) -> None:
    """Routing metadata stays anchored to server working_dir, not agent env.cwd."""
    server_state.agent.env.cwd = "/tmp/non-exists-dir"

    event = SessionStatusEvent.create(session_id="s5", status_type="busy")
    results = await _collect_real_events(server_state, wrap_payload=True, events_to_send=[event])
    received = results[1]

    assert received["directory"] == str(Path(server_state.working_dir).resolve())


@pytest.mark.integration
@pytest.mark.anyio
async def test_global_event_integration_session_id_in_payload(
    server_state: ServerState,
) -> None:
    """Test sessionId injection at top level of GlobalEvent payload."""
    event = SessionStatusEvent.create(session_id="injected-sid", status_type="busy")
    results = await _collect_real_events(server_state, wrap_payload=True, events_to_send=[event])
    received = results[1]
    payload = received["payload"]
    assert payload["sessionId"] == "injected-sid"


@pytest.mark.integration
@pytest.mark.anyio
async def test_global_event_integration_unicode_preserved(
    server_state: ServerState,
) -> None:
    r"""Test unicode characters preserved in SSE output (not \uXXXX escaped)."""
    event = SessionStatusEvent.create(session_id="会话测试", status_type="idle")
    results = await _collect_real_events(server_state, wrap_payload=True, events_to_send=[event])
    received = results[1]
    payload = received["payload"]
    assert payload["sessionId"] == "会话测试"


# =============================================================================
# on_first_subscriber callback tests
# =============================================================================


@pytest.mark.anyio
async def test_on_first_subscriber_fires_once() -> None:
    """Callback fires exactly once on first subscriber."""
    state = _MockState()
    callback = AsyncMock()
    state.on_first_subscriber = callback

    gen = _event_generator(state, wrap_payload=False)
    await gen.__anext__()  # consume connected event

    assert state._first_subscriber_triggered is True
    await asyncio.sleep(0.05)
    callback.assert_called_once()


@pytest.mark.anyio
async def test_on_first_subscriber_does_not_fire_on_second_subscriber() -> None:
    """Callback does not fire again on second subscriber."""
    state = _MockState()
    callback = AsyncMock()
    state.on_first_subscriber = callback

    gen1 = _event_generator(state, wrap_payload=False)
    await gen1.__anext__()  # consume connected event

    await asyncio.sleep(0.05)
    assert callback.call_count == 1

    gen2 = _event_generator(state, wrap_payload=False)
    await gen2.__anext__()  # consume connected event

    await asyncio.sleep(0.05)
    # Callback should still have been called only once
    callback.assert_called_once()


@pytest.mark.anyio
async def test_on_first_subscriber_flag_set_and_stays_true() -> None:
    """First subscriber flag is set to True after first subscriber and stays True."""
    state = _MockState()
    callback = AsyncMock()
    state.on_first_subscriber = callback

    assert state._first_subscriber_triggered is False

    gen1 = _event_generator(state, wrap_payload=False)
    await gen1.__anext__()  # consume connected event

    assert state._first_subscriber_triggered is True

    gen2 = _event_generator(state, wrap_payload=False)
    await gen2.__anext__()  # consume connected event

    # Flag must remain True, never reset
    assert state._first_subscriber_triggered is True


@pytest.mark.anyio
async def test_on_first_subscriber_fires_before_events_delivered() -> None:
    """Callback fires before the generator yields any events beyond connected."""
    state = _MockState()
    callback = AsyncMock()
    state.on_first_subscriber = callback

    gen = _event_generator(state, wrap_payload=False)
    # Consuming the connected event should have already triggered the callback
    await gen.__anext__()

    # The flag is set synchronously before yielding the connected event
    assert state._first_subscriber_triggered is True
    await asyncio.sleep(0.05)
    # The background task created by the callback should have been scheduled
    callback.assert_called_once()


@pytest.mark.anyio
async def test_on_first_subscriber_no_callback_set() -> None:
    """No callback invocation when on_first_subscriber is None."""
    state = _MockState()
    # on_first_subscriber is None by default
    assert state.on_first_subscriber is None

    gen = _event_generator(state, wrap_payload=False)
    await gen.__anext__()  # consume connected event

    # Flag should not be set because there is no callback
    assert state._first_subscriber_triggered is False


# =============================================================================
# Client disconnect cleanup tests
# =============================================================================


@pytest.mark.anyio
async def test_disconnect_queue_removed_from_subscribers() -> None:
    """Queue is removed from event_subscribers when client disconnects."""
    state = _MockState()
    gen = _event_generator(state, wrap_payload=False)
    await gen.__anext__()  # consume connected event
    assert len(state.event_subscribers) == 1

    await gen.aclose()
    assert len(state.event_subscribers) == 0


@pytest.mark.anyio
async def test_disconnect_events_not_delivered() -> None:
    """After disconnect, broadcast_event does not deliver to disconnected client."""
    state = _MockState()
    gen1 = _event_generator(state, wrap_payload=False)
    await gen1.__anext__()  # consume connected event
    # Add a second subscriber that stays connected to verify isolation
    gen2 = _event_generator(state, wrap_payload=False)
    await gen2.__anext__()  # consume connected event
    assert len(state.event_subscribers) == 2

    queue1 = state.event_subscribers[0]
    queue2 = state.event_subscribers[1]

    # Disconnect first client
    await gen1.aclose()
    assert len(state.event_subscribers) == 1
    assert queue1 not in state.event_subscribers
    assert queue2 in state.event_subscribers

    # Put an event via mock EventBus — only gen2 should receive it
    event = SessionStatusEvent.create(session_id="disc1", status_type="busy")
    await state._mock_event_bus.publish("__global_sse__", event)
    item = await gen2.__anext__()
    data = json.loads(item["data"])
    assert data["type"] == "session.status"


@pytest.mark.anyio
async def test_disconnect_finally_block_executes() -> None:
    """The finally block in _event_generator runs on disconnect, removing the queue."""
    state = _MockState()
    gen = _event_generator(state, wrap_payload=False)
    await gen.__anext__()  # consume connected event
    queue_before = state.event_subscribers[-1]
    assert queue_before in state.event_subscribers

    await gen.aclose()

    # The finally block removed the queue
    assert queue_before not in state.event_subscribers
    assert len(state.event_subscribers) == 0


@pytest.mark.anyio
async def test_disconnect_abrupt_cleanup() -> None:
    """Abrupt disconnect (task cancellation) still triggers cleanup."""
    state = _MockState()

    async def consume() -> None:
        gen = _event_generator(state, wrap_payload=False)
        with contextlib.suppress(StopAsyncIteration):
            async for _ in gen:
                pass

    task = asyncio.create_task(consume())
    # Let the generator start and consume the connected event
    await asyncio.sleep(0.05)
    assert len(state.event_subscribers) == 1

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    # Cleanup should have removed the subscriber queue
    assert len(state.event_subscribers) == 0


@pytest.mark.anyio
async def test_disconnect_no_memory_leak() -> None:
    """Multiple connect/disconnect cycles do not cause subscriber list growth."""
    state = _MockState()

    for _ in range(5):
        gen = _event_generator(state, wrap_payload=False)
        await gen.__anext__()  # consume connected event
        assert len(state.event_subscribers) == 1
        await gen.aclose()
        assert len(state.event_subscribers) == 0

    # After 5 cycles, no leaked subscribers
    assert len(state.event_subscribers) == 0


# =============================================================================
# _extract_session_id exhaustiveness tests
# =============================================================================


def _make_session(session_id: str = "test-sid") -> Session:
    """Create a minimal Session for event construction."""
    return Session(
        id=session_id,
        project_id="proj1",
        directory="/tmp",
        title="Test",
        time=TimeCreatedUpdated(created=0, updated=0),
    )


def _make_part(session_id: str = "test-sid") -> Part:
    """Create a minimal Part for event construction."""
    return TextPart(
        id="part1",
        message_id="msg1",
        session_id=session_id,
        text="hello",
    )


# All 22 handled event types with their constructors
_HANDLED_EVENT_FACTORIES: list[tuple[str, type]] = [
    ("session.deleted", SessionDeletedEvent),
    ("session.status", SessionStatusEvent),
    ("session.idle", SessionIdleEvent),
    ("session.compacted", SessionCompactedEvent),
    ("message.removed", MessageRemovedEvent),
    ("message.part.removed", PartRemovedEvent),
    ("permission.asked", PermissionRequestEvent),
    ("permission.replied", PermissionResolvedEvent),
    ("question.asked", QuestionAskedEvent),
    ("question.replied", QuestionRepliedEvent),
    ("question.rejected", QuestionRejectedEvent),
    ("todo.updated", TodoUpdatedEvent),
    ("session.error", SessionErrorEvent),
    ("session.created", SessionCreatedEvent),
    ("session.updated", SessionUpdatedEvent),
    ("message.part.updated", PartUpdatedEvent),
    ("session.diff", SessionDiffEvent),
    ("message.part.delta", PartDeltaEvent),
    ("permission.updated", PermissionUpdatedEvent),
    ("command.executed", CommandExecutedEvent),
    ("tui.session.select", TuiSessionSelectEvent),
    ("message.updated", MessageUpdatedEvent),
]


def _build_handled_event(event_type: type) -> Event:  # noqa: PLR0911
    """Build a handled event with session_id='abc' using the appropriate constructor."""
    sid = "abc"
    if event_type is SessionDeletedEvent:
        return SessionDeletedEvent.create(session_id=sid)
    if event_type is SessionStatusEvent:
        return SessionStatusEvent.create(session_id=sid, status_type="busy")
    if event_type is SessionIdleEvent:
        return SessionIdleEvent.create(session_id=sid)
    if event_type is SessionCompactedEvent:
        return SessionCompactedEvent.create(session_id=sid)
    if event_type is MessageRemovedEvent:
        return MessageRemovedEvent.create(session_id=sid, message_id="m1")
    if event_type is PartRemovedEvent:
        return PartRemovedEvent.create(session_id=sid, message_id="m1", part_id="p1")
    if event_type is PermissionRequestEvent:
        return PermissionRequestEvent.create(
            session_id=sid,
            permission_id="perm1",
            tool_name="bash",
            args_preview="ls",
            message="Allow?",
        )
    if event_type is PermissionResolvedEvent:
        return PermissionResolvedEvent.create(
            session_id=sid,
            request_id="perm1",
            reply="once",
        )
    if event_type is QuestionAskedEvent:
        return QuestionAskedEvent.create(
            request_id="q1",
            session_id=sid,
            questions=[
                QuestionInfo(
                    question="Continue?",
                    header="Confirm",
                    options=[QuestionOption(label="Yes", description="Proceed")],
                )
            ],
        )
    if event_type is QuestionRepliedEvent:
        return QuestionRepliedEvent.create(
            session_id=sid,
            request_id="q1",
            answers=[["Yes"]],
        )
    if event_type is QuestionRejectedEvent:
        return QuestionRejectedEvent.create(
            session_id=sid,
            request_id="q1",
        )
    if event_type is TodoUpdatedEvent:
        return TodoUpdatedEvent.create(session_id=sid, todos=[])
    if event_type is SessionErrorEvent:
        return SessionErrorEvent.create(session_id=sid, error_name="TestError")
    if event_type is SessionCreatedEvent:
        return SessionCreatedEvent.create(session=_make_session(sid))
    if event_type is SessionUpdatedEvent:
        return SessionUpdatedEvent.create(session=_make_session(sid))
    if event_type is PartUpdatedEvent:
        return PartUpdatedEvent.create(part=_make_part(sid))
    if event_type is SessionDiffEvent:
        return SessionDiffEvent.create(session_id=sid, diff=[])
    if event_type is PartDeltaEvent:
        return PartDeltaEvent.create(session_id=sid, message_id="m1", part_id="p1", delta="hi")
    if event_type is PermissionUpdatedEvent:
        return PermissionUpdatedEvent.create(
            session_id=sid,
            permission_id="perm1",
            tool_name="bash",
            patterns=["bash: *"],
            metadata={},
        )
    if event_type is CommandExecutedEvent:
        return CommandExecutedEvent.create(
            name="test",
            session_id=sid,
            arguments="",
            message_id="m1",
        )
    if event_type is TuiSessionSelectEvent:
        return TuiSessionSelectEvent.create(session_id=sid)
    if event_type is MessageUpdatedEvent:
        return MessageUpdatedEvent.create(
            message=UserMessage(
                id="m1",
                session_id=sid,
                time=TimeCreated(created=0),
            ),
        )
    msg = f"Unhandled event type in test helper: {event_type}"
    raise ValueError(msg)


@pytest.mark.parametrize(
    ("event_type_name", "event_type"),
    [(name, cls) for name, cls in _HANDLED_EVENT_FACTORIES],
    ids=[name for name, _ in _HANDLED_EVENT_FACTORIES],
)
def test_extract_session_id_handled_events(
    event_type_name: str,
    event_type: type,
) -> None:
    """All 22 handled event types extract sessionId correctly."""
    event = _build_handled_event(event_type)
    result = _extract_session_id(event)
    assert result == "abc", f"Expected 'abc' for {event_type_name}, got {result!r}"


def test_extract_session_id_session_error_nullable() -> None:
    """SessionErrorEvent with None session_id returns None."""
    event = SessionErrorEvent.create(session_id=None, error_name="TestError")
    result = _extract_session_id(event)
    assert result is None


def test_extract_session_id_no_session_events_return_none() -> None:
    """Explicitly-listed no-session event types return None without warnings."""
    no_session_events: list[Event] = [
        ServerConnectedEvent(),
        ServerHeartbeatEvent(),
        FileEditedEvent.create(file="/tmp/test.py"),
        FileWatcherUpdatedEvent.create(file="/tmp/test.py", event="change"),
        McpToolsChangedEvent.create(server="test_server"),
        PtyCreatedEvent.create(
            info=PtyInfo(
                id="p1",
                title="test",
                command="echo",
                args=[],
                cwd="/tmp",
                status="running",
                pid=1234,
            ),
        ),
        PtyUpdatedEvent.create(
            info=PtyInfo(
                id="p1",
                title="test",
                command="echo",
                args=[],
                cwd="/tmp",
                status="running",
                pid=1234,
            ),
        ),
        PtyExitedEvent.create(pty_id="p1", exit_code=0),
        PtyDeletedEvent.create(pty_id="p1"),
        LspUpdatedEvent(),
        LspClientDiagnosticsEvent.create(server_id="s1", path="/tmp"),
        ProjectUpdatedEvent.create(
            project=Project(id="test", worktree="/tmp", time=ProjectTime(created=0)),
        ),
        VcsBranchUpdatedEvent.create(branch="main"),
        TuiPromptAppendEvent.create(text="hello"),
        TuiCommandExecuteEvent.create(command="test"),
        TuiToastShowEvent.create(message="hi"),
    ]
    for event in no_session_events:
        result = _extract_session_id(event)
        assert result is None, f"Expected None for {type(event).__name__}, got {result!r}"


def test_extract_session_id_no_warning_for_no_session_events(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No warning logged for explicitly-listed no-session event types."""
    event = FileEditedEvent.create(file="/tmp/test.py")
    with caplog.at_level("WARNING"):
        result = _extract_session_id(event)
    assert result is None
    assert "Unhandled event type in _extract_session_id" not in caplog.text


def test_extract_session_id_no_warning_for_unknown_event_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unknown event types return None without logging warnings.

    Since many events genuinely have no session association, the
    isinstance-based approach returns None silently for unrecognized types.
    """
    mock_event = MagicMock()
    mock_event.__class__.__name__ = "FutureUnknownEvent"
    # Make properties NOT be a SessionIdProperties instance
    mock_event.properties = MagicMock()
    with caplog.at_level("WARNING"):
        result = _extract_session_id(mock_event)  # type: ignore[arg-type]
    assert result is None
    assert "Unhandled event type in _extract_session_id" not in caplog.text


def test_extract_session_id_no_warning_for_handled(caplog: pytest.LogCaptureFixture) -> None:
    """No warning logged for handled event types."""
    event = SessionStatusEvent.create(session_id="no-warn", status_type="idle")
    with caplog.at_level("WARNING"):
        _extract_session_id(event)
    assert "Unhandled event type in _extract_session_id" not in caplog.text


def test_extract_session_id_exhaustiveness() -> None:
    """All Event union members are covered by _extract_session_id.

    Uses SessionIdProperties base class for automatic coverage:
    - Events whose properties inherit from SessionIdProperties are auto-covered
    - Special-path events (SessionCreated/Updated, MessageUpdated, PartUpdated,
      SessionError) have explicit extraction logic
    - Remaining events have no session association and return None

    Catches future regressions: if a new event type with session_id is added
    to the Event union but its properties don't inherit from SessionIdProperties
    and it's not in the special-path list, this test fails.
    """
    # Special-path events: session_id is nested (not at properties.session_id)
    special_path_types: set[type] = {
        SessionCreatedEvent,
        SessionUpdatedEvent,
        MessageUpdatedEvent,
        PartUpdatedEvent,
        SessionErrorEvent,
    }

    # Events whose properties inherit from SessionIdProperties
    # (automatically covered by isinstance check in _extract_session_id)
    session_id_props_types: set[type] = set()
    for event_type in Event.__args__:
        props_type = event_type.model_fields["properties"].annotation
        # Resolve annotations that may be stringified or wrapped
        if (
            props_type is not None
            and isinstance(props_type, type)
            and issubclass(props_type, SessionIdProperties)
        ):
            session_id_props_types.add(event_type)

    # Events with no session association
    no_session_types: set[type] = set()
    for event_type in Event.__args__:
        if event_type not in special_path_types and event_type not in session_id_props_types:
            no_session_types.add(event_type)

    # Verify every Event union member is accounted for
    event_union_args: set[type] = set(Event.__args__)
    covered = special_path_types | session_id_props_types | no_session_types

    missing = event_union_args - covered
    assert not missing, (
        f"New event types not covered by _extract_session_id: "
        f"{sorted(t.__name__ for t in missing)}. "
        f"Either make their properties inherit from SessionIdProperties, "
        f"add to special_path_types, or add to no_session_types."
    )

    # No extra types that aren't in the union
    extra = covered - event_union_args
    assert not extra, (
        f"Types listed in test but not in Event union: {sorted(t.__name__ for t in extra)}"
    )

    # Verify SessionIdProperties subclasses exist (not just empty set)
    assert session_id_props_types, (
        "No event types have properties inheriting from SessionIdProperties. "
        "The base class refactoring may not have been applied."
    )


def test_session_id_properties_base_class_hierarchy() -> None:
    """Verify SessionIdProperties is properly used by event properties models.

    Ensures that properties models with session_id: str inherit from
    SessionIdProperties, and that SessionErrorProperties (which has
    session_id: str | None) does NOT inherit from it.
    """
    from wolfharness_server.opencode_server.models.events import SessionErrorProperties

    # SessionIdProperties must have session_id: str
    assert "session_id" in SessionIdProperties.model_fields
    field = SessionIdProperties.model_fields["session_id"]
    # The field annotation should be str (not str | None)
    assert field.annotation is str

    # SessionErrorProperties must NOT inherit from SessionIdProperties
    # because its session_id is str | None, not str
    assert not issubclass(SessionErrorProperties, SessionIdProperties)


def test_extract_session_id_uses_isinstance() -> None:
    """_extract_session_id uses isinstance(SessionIdProperties) for common path.

    Verifies that the isinstance-based dispatch works correctly by testing
    a representative SessionIdProperties-inheriting event type.
    """
    # SessionStatusEvent properties inherit from SessionIdProperties
    event = SessionStatusEvent.create(session_id="isinstance-test", status_type="busy")
    assert isinstance(event.properties, SessionIdProperties)
    result = _extract_session_id(event)
    assert result == "isinstance-test"

    # ServerHeartbeatEvent properties do NOT inherit from SessionIdProperties
    event2 = ServerHeartbeatEvent()
    assert not isinstance(event2.properties, SessionIdProperties)
    result2 = _extract_session_id(event2)
    assert result2 is None


# =============================================================================
# Concurrent subscriber tests
# =============================================================================


@pytest.mark.anyio
async def test_concurrent_two_subscribers_both_receive_events() -> None:
    """Two SSE clients both receive a broadcast event."""
    state = _MockState()

    gen1 = _event_generator(state, wrap_payload=True)
    gen2 = _event_generator(state, wrap_payload=True)

    # Consume initial connected events
    await gen1.__anext__()
    await gen2.__anext__()

    assert len(state.event_subscribers) == 2

    # Broadcast event to both subscribers via mock EventBus
    event = SessionStatusEvent.create(session_id="s_concurrent", status_type="busy")
    await state._mock_event_bus.publish("__global_sse__", event)

    item1 = await gen1.__anext__()
    item2 = await gen2.__anext__()

    data1 = json.loads(item1["data"])
    data2 = json.loads(item2["data"])

    assert data1["payload"]["type"] == "session.status"
    assert data2["payload"]["type"] == "session.status"
    assert data1["payload"]["sessionId"] == "s_concurrent"
    assert data2["payload"]["sessionId"] == "s_concurrent"


@pytest.mark.anyio
async def test_concurrent_subscribers_receive_same_content() -> None:
    """Both subscribers get identical GlobalEvent envelopes."""
    state = _MockState()

    gen1 = _event_generator(state, wrap_payload=True)
    gen2 = _event_generator(state, wrap_payload=True)

    await gen1.__anext__()
    await gen2.__anext__()

    event = SessionStatusEvent.create(session_id="s_same", status_type="idle")
    await state._mock_event_bus.publish("__global_sse__", event)

    item1 = await gen1.__anext__()
    item2 = await gen2.__anext__()

    # Both envelopes must have identical directory, project, and payload
    data1 = json.loads(item1["data"])
    data2 = json.loads(item2["data"])

    assert data1["directory"] == data2["directory"]
    assert data1["project"] == data2["project"]
    assert data1["payload"] == data2["payload"]


@pytest.mark.anyio
async def test_concurrent_event_ordering_preserved() -> None:
    """Broadcast 3 events; each subscriber receives them in order."""
    state = _MockState()

    gen1 = _event_generator(state, wrap_payload=True)
    gen2 = _event_generator(state, wrap_payload=True)

    await gen1.__anext__()
    await gen2.__anext__()

    events = [
        SessionStatusEvent.create(session_id="ord1", status_type="busy"),
        SessionStatusEvent.create(session_id="ord2", status_type="idle"),
        SessionStatusEvent.create(session_id="ord3", status_type="retry"),
    ]

    for ev in events:
        await state._mock_event_bus.publish("__global_sse__", ev)

    # Collect all 3 events from each subscriber
    received1 = [json.loads((await gen1.__anext__())["data"]) for _ in range(3)]
    received2 = [json.loads((await gen2.__anext__())["data"]) for _ in range(3)]

    expected_order = ["ord1", "ord2", "ord3"]
    ids1 = [r["payload"]["sessionId"] for r in received1]
    ids2 = [r["payload"]["sessionId"] for r in received2]

    assert ids1 == expected_order
    assert ids2 == expected_order


@pytest.mark.anyio
async def test_concurrent_subscriber_receives_after_another_disconnects() -> None:
    """Subscriber B still receives events after subscriber A disconnects."""
    state = _MockState()

    gen_a = _event_generator(state, wrap_payload=True)
    gen_b = _event_generator(state, wrap_payload=True)

    await gen_a.__anext__()
    await gen_b.__anext__()

    assert len(state.event_subscribers) == 2

    # Disconnect subscriber A
    await gen_a.aclose()
    assert len(state.event_subscribers) == 1

    # Send event via mock EventBus — remaining subscriber B should receive it
    event = SessionStatusEvent.create(session_id="s_survive", status_type="busy")
    await state._mock_event_bus.publish("__global_sse__", event)

    item_b = await gen_b.__anext__()
    data_b = json.loads(item_b["data"])

    assert data_b["payload"]["type"] == "session.status"
    assert data_b["payload"]["sessionId"] == "s_survive"


@pytest.mark.anyio
async def test_concurrent_all_get_server_connected() -> None:
    """Each subscriber gets the initial payload-wrapped server.connected event."""
    state = _MockState()

    gen1 = _event_generator(state, wrap_payload=True)
    gen2 = _event_generator(state, wrap_payload=True)
    gen3 = _event_generator(state, wrap_payload=True)

    item1 = await gen1.__anext__()
    item2 = await gen2.__anext__()
    item3 = await gen3.__anext__()

    for item in [item1, item2, item3]:
        data = json.loads(item["data"])
        assert data["payload"]["type"] == "server.connected"
        assert "directory" not in data
        assert "project" not in data
        assert "payload" in data


# =============================================================================
# ServerState.broadcast_event direct tests
# =============================================================================


def _make_broadcast_state() -> ServerState:
    """Create a ServerState with a minimal mock agent for broadcast_event tests."""
    from unittest.mock import AsyncMock, Mock

    mock_env = Mock()
    mock_env.get_fs = Mock(return_value=Mock())
    mock_agent = Mock()
    mock_agent.env = mock_env
    state = ServerState(working_dir="/test", agent=mock_agent)
    # Set up event_bridge mock so broadcast_event has a destination
    state.event_bridge = Mock()
    state.event_bridge.publish = AsyncMock()
    return state


@pytest.mark.anyio
async def test_broadcast_event_delegates_to_bridge() -> None:
    """Broadcast delegates event to event_bridge.publish."""
    state = _make_broadcast_state()

    event = SessionStatusEvent.create(session_id="abc", status_type="busy")
    await state.broadcast_event(event)

    state.event_bridge.publish.assert_called_once_with(event)


@pytest.mark.anyio
async def test_broadcast_event_no_bridge_no_error() -> None:
    """Broadcast with no event_bridge does not raise."""
    from unittest.mock import Mock

    mock_env = Mock()
    mock_env.get_fs = Mock(return_value=Mock())
    mock_agent = Mock()
    mock_agent.env = mock_env
    state = ServerState(working_dir="/test", agent=mock_agent)
    assert state.event_bridge is None

    event = SessionStatusEvent.create(session_id="abc", status_type="busy")
    await state.broadcast_event(event)  # Should not raise


@pytest.mark.anyio
async def test_broadcast_event_bridge_error_propagates() -> None:
    """If event_bridge.publish raises, broadcast_event propagates the error."""
    state = _make_broadcast_state()
    state.event_bridge.publish.side_effect = RuntimeError("bridge broken")

    event = SessionStatusEvent.create(session_id="abc", status_type="busy")
    # The error propagates since broadcast_event does not catch it
    with pytest.raises(RuntimeError, match="bridge broken"):
        await state.broadcast_event(event)


# =============================================================================
# /global/health endpoint tests
# =============================================================================


@pytest.mark.anyio
async def test_global_health_endpoint(async_client: AsyncClient) -> None:
    """GET /global/health returns 200 with HealthResponse body."""
    response = await async_client.get("/global/health")
    assert response.status_code == 200
    data = response.json()
    assert data["healthy"] is True
    assert "version" in data


@pytest.mark.anyio
async def test_global_health_endpoint_fields(async_client: AsyncClient) -> None:
    """GET /global/health returns correct healthy and version fields."""
    from wolfharness_server.opencode_server.routes.global_routes import VERSION

    response = await async_client.get("/global/health")
    data = response.json()
    assert data["healthy"] is True
    assert data["version"] == VERSION


# =============================================================================
# GlobalEvent edge case tests
# =============================================================================


def test_global_event_large_payload() -> None:
    r"""Large payload (100KB+) serializes and deserializes correctly.

    Uses a TodoUpdatedEvent with a very long todo content string to produce
    a payload exceeding 100KB. Verifies round-trip correctness via json.loads.
    """
    large_text = "A" * 100_000  # 100KB string
    todo = Todo(id="t1", content=large_text, status="pending", priority="high")
    event = TodoUpdatedEvent.create(session_id="large-sid", todos=[todo])
    factory = GlobalEventFactory(directory="/tmp", project="abc")
    result = factory.wrap(event)
    data = json.loads(result)
    # Payload contains the full large text
    assert data["payload"]["properties"]["todos"][0]["content"] == large_text
    # Round-trip: re-serialize and re-parse
    round_tripped = json.loads(json.dumps(data, ensure_ascii=False))
    assert round_tripped["payload"]["properties"]["todos"][0]["content"] == large_text


def test_global_event_special_characters() -> None:
    r"""Special characters (quotes, backslashes, control chars, emojis) preserved.

    Verifies that characters like '"', '\\', '\n', '\t', and emoji are correctly
    serialized and deserialized through _serialize_event and GlobalEventFactory.
    """
    special_sid = 'sid-with-"quotes"-and-\\backslash\\'
    event = SessionStatusEvent.create(session_id=special_sid, status_type="busy")
    result = _serialize_event(event, wrap_payload=False)
    data = json.loads(result)
    # sessionId at top level
    assert data["sessionId"] == special_sid
    # sessionID inside properties (alias-converted)
    assert data["properties"]["sessionID"] == special_sid

    # Also test via factory.wrap with emoji and control chars in Todo content
    emoji_text = "Hello 🔥🚀 world!\n\ttabbed line"
    todo = Todo(id="t2", content=emoji_text, status="in_progress", priority="medium")
    event2 = TodoUpdatedEvent.create(session_id="emoji-sid", todos=[todo])
    factory = GlobalEventFactory(directory="/tmp", project="abc")
    result2 = factory.wrap(event2)
    data2 = json.loads(result2)
    assert data2["payload"]["properties"]["todos"][0]["content"] == emoji_text


def test_global_event_empty_string_fields() -> None:
    r"""Empty string sessionId is preserved (not treated as None).

    An empty string is a valid value and should not be excluded by
    exclude_none=True (which only drops None, not empty strings).
    """
    event = SessionStatusEvent.create(session_id="", status_type="idle")
    result = _serialize_event(event, wrap_payload=False)
    data = json.loads(result)
    # sessionId injected at top level
    assert data["sessionId"] == ""
    # sessionID inside properties
    assert data["properties"]["sessionID"] == ""

    # Via factory.wrap
    factory = GlobalEventFactory(directory="/tmp", project="abc")
    wrapped = factory.wrap(event)
    wrapped_data = json.loads(wrapped)
    assert wrapped_data["payload"]["sessionId"] == ""


def test_global_event_unicode_multibyte() -> None:
    r"""CJK characters and multi-byte emoji sequences are preserved.

    Verifies that characters like 日本語 (Japanese) and complex emoji
    sequences like 👨‍👩‍👧‍👦 (family emoji, ZWJ sequence) survive
    serialization round-trip without \uXXXX escaping.
    """
    cjk_sid = "会話-日本語-テスト"
    event = SessionStatusEvent.create(session_id=cjk_sid, status_type="busy")
    result = _serialize_event(event, wrap_payload=False)
    # Raw string must contain CJK chars, not \uXXXX escapes
    assert "日本語" in result
    assert "テスト" in result
    assert "\\u" not in result

    # Round-trip via json.loads
    data = json.loads(result)
    assert data["sessionId"] == cjk_sid

    # ZWJ emoji in todo content
    family_emoji = "👨‍👩‍👧‍👦"
    todo = Todo(id="t3", content=f"Family: {family_emoji}", status="completed", priority="low")
    event2 = TodoUpdatedEvent.create(session_id=cjk_sid, todos=[todo])
    factory = GlobalEventFactory(directory="/tmp", project="abc")
    wrapped = factory.wrap(event2)
    assert family_emoji in wrapped
    assert "\\u" not in wrapped
    wrapped_data = json.loads(wrapped)
    assert wrapped_data["payload"]["properties"]["todos"][0]["content"] == f"Family: {family_emoji}"
