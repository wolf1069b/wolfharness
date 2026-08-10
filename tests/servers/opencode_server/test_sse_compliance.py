"""End-to-end SSE protocol compliance tests.

Validates that the SSE event stream conforms to the OpenCode TUI protocol:
- /global/event emits events that pass the TUI routing filter
- Directory in GlobalEvent envelope uses the configured server working_dir
- server.connected and server.heartbeat keep a payload wrapper on /global/event
- All other events are wrapped in GlobalEvent with correct directory
- sessionId appears at top level of payload for ALL event types
- UserMessage events use nested model.variant format
- FileDiff events use v1.4.0+ schema (patch, not before/after)
- /event endpoint still works for backward compatibility (raw events, no envelope)
- PartDeltaEvent streaming (most critical for "no response" symptom)
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest

from wolfharness_server.opencode_server.models import GlobalEvent
from wolfharness_server.opencode_server.models.common import (
    FileDiff,
    ModelRef,
    TimeCreated,
)
from wolfharness_server.opencode_server.models.events import (
    CommandExecutedEvent,
    MessageUpdatedEvent,
    PartDeltaEvent,
    PermissionRequestEvent,
    PermissionResolvedEvent,
    ServerConnectedEvent,
    ServerHeartbeatEvent,
    SessionCompactedEvent,
    SessionCreatedEvent,
    SessionDeletedEvent,
    SessionDiffEvent,
    SessionStatusEvent,
    TodoUpdatedEvent,
    TuiSessionSelectEvent,
)
from wolfharness_server.opencode_server.models.message import UserMessage
from wolfharness_server.opencode_server.models.parts import TextPart
from wolfharness_server.opencode_server.models.session import (
    Session,
    TimeCreatedUpdated as SessionTimeCreatedUpdated,
)
from wolfharness_server.opencode_server.routes.global_routes import (
    GlobalEventFactory,
    _event_generator,
    _extract_session_id,
    _serialize_event,
)
from wolfharness_server.opencode_server.routes.routing import tui_event_filter


if TYPE_CHECKING:
    from wolfharness_server.opencode_server.models.events import Event
    from wolfharness_server.opencode_server.models.parts import Part


# =============================================================================
# Test helpers (reusing _MockState / _collect_events pattern from test_global_event)
# =============================================================================


import contextlib


pytestmark = pytest.mark.integration


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

            directory = self.working_dir
            self._event_factory = GlobalEventFactory(
                directory=directory,
                project=helpers.compute_project_id(directory),
            )
        return self._event_factory

    def create_background_task(self, coro: Any, name: str = "") -> asyncio.Task[Any]:
        return asyncio.ensure_future(coro)

    # get_next_event_id() removed — event_id now comes from EventBus
    # (EventEnvelope.event_id assigned at publish time, not ServerState).

    def cancel_all_pending_questions(self) -> list[str]:
        return []


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


def _make_session(session_id: str = "test-sid") -> Session:
    """Create a minimal Session for event construction."""
    return Session(
        id=session_id,
        project_id="proj1",
        directory="/tmp",
        title="Test",
        time=SessionTimeCreatedUpdated(created=0, updated=0),
    )


def _make_part(session_id: str = "test-sid") -> Part:
    """Create a minimal Part for event construction."""
    return TextPart(
        id="part1",
        message_id="msg1",
        session_id=session_id,
        text="hello",
    )


# =============================================================================
# 1. TUI routing filter compliance — /global/event events pass the filter
# =============================================================================


@pytest.mark.anyio
async def test_tui_filter_session_status_event_passes() -> None:
    """SessionStatusEvent in GlobalEvent envelope passes tui_event_filter."""
    wd = "/tmp/compliance_test"
    state = _MockState(working_dir=wd)
    event = SessionStatusEvent.create(session_id="s1", status_type="busy")
    events = await _collect_events(state, wrap_payload=True, events_to_send=[event])

    wrapped = events[1]
    # Build a GlobalEvent from the wire data for filter testing
    ge = GlobalEvent(
        directory=wrapped["directory"],
        project=wrapped.get("project"),
        payload=wrapped["payload"],
    )
    passes, reason = tui_event_filter(ge, state.working_dir)
    assert passes, f"SessionStatusEvent should pass filter, got reason={reason}"


@pytest.mark.anyio
async def test_tui_filter_part_delta_event_passes() -> None:
    """PartDeltaEvent in GlobalEvent envelope passes tui_event_filter."""
    wd = "/tmp/compliance_delta"
    state = _MockState(working_dir=wd)
    event = PartDeltaEvent.create(session_id="s2", message_id="m1", part_id="p1", delta="hello")
    events = await _collect_events(state, wrap_payload=True, events_to_send=[event])

    wrapped = events[1]
    ge = GlobalEvent(
        directory=wrapped["directory"],
        project=wrapped.get("project"),
        payload=wrapped["payload"],
    )
    passes, reason = tui_event_filter(ge, state.working_dir)
    assert passes, f"PartDeltaEvent should pass filter, got reason={reason}"


@pytest.mark.anyio
async def test_tui_filter_all_session_events_pass() -> None:
    """Multiple session-scoped events in envelopes all pass tui_event_filter."""
    wd = "/tmp/compliance_multi"
    state = _MockState(working_dir=wd)

    session_events: list[Event] = [
        SessionStatusEvent.create(session_id="s1", status_type="busy"),
        SessionCompactedEvent.create(session_id="s1"),
        SessionDeletedEvent.create(session_id="s1"),
        TodoUpdatedEvent.create(session_id="s1", todos=[]),
        PartDeltaEvent.create(session_id="s1", message_id="m1", part_id="p1", delta="x"),
    ]
    events = await _collect_events(state, wrap_payload=True, events_to_send=session_events)

    for i, raw in enumerate(events[1:], start=0):
        ge = GlobalEvent(
            directory=raw["directory"],
            project=raw.get("project"),
            payload=raw["payload"],
        )
        passes, reason = tui_event_filter(ge, state.working_dir)
        assert passes, f"Event {i} ({raw['payload']['type']}) should pass, reason={reason}"


# =============================================================================
# 2. Directory routing metadata in GlobalEvent envelope
# =============================================================================


@pytest.mark.anyio
async def test_envelope_directory_matches_working_dir() -> None:
    """Envelope directory field matches the server working directory."""
    wd = "/custom/exact/working/dir/../dir"
    state = _MockState(working_dir=wd)
    event = SessionStatusEvent.create(session_id="dir1", status_type="busy")
    events = await _collect_events(state, wrap_payload=True, events_to_send=[event])

    wrapped = events[1]
    assert wrapped["directory"] == state.working_dir


@pytest.mark.anyio
async def test_envelope_directory_has_no_trailing_slash() -> None:
    """Directory emits without a trailing slash."""
    wd = "/tmp/no_trailing_slash"
    state = _MockState(working_dir=wd)
    event = PartDeltaEvent.create(session_id="dir2", message_id="m1", part_id="p1", delta="hi")
    events = await _collect_events(state, wrap_payload=True, events_to_send=[event])

    wrapped = events[1]
    assert wrapped["directory"] == state.working_dir
    assert not wrapped["directory"].endswith("/")


@pytest.mark.anyio
async def test_envelope_directory_different_from_mismatched_path() -> None:
    """TUI filter rejects event whose directory doesn't match project_directory."""
    wd = "/correct/dir"
    state = _MockState(working_dir=wd)
    event = SessionStatusEvent.create(session_id="mismatch1", status_type="idle")
    events = await _collect_events(state, wrap_payload=True, events_to_send=[event])

    wrapped = events[1]
    ge = GlobalEvent(
        directory=wrapped["directory"],
        project=wrapped.get("project"),
        payload=wrapped["payload"],
    )
    # Should pass with correct directory
    passes, _ = tui_event_filter(ge, wd)
    assert passes
    # Should fail with wrong directory
    passes2, reason2 = tui_event_filter(ge, "/wrong/dir")
    assert not passes2
    assert reason2 == "directory_mismatch"


# =============================================================================
# 3. server.connected and server.heartbeat keep only a payload wrapper on /global/event
# =============================================================================


@pytest.mark.anyio
async def test_server_connected_is_payload_wrapped() -> None:
    """Initial server.connected keeps only the payload wrapper."""
    state = _MockState()
    events = await _collect_events(state, wrap_payload=True, events_to_send=[])
    assert len(events) == 1
    connected = events[0]
    assert connected["payload"]["type"] == "server.connected"
    assert "directory" not in connected
    assert "project" not in connected


@pytest.mark.anyio
async def test_server_heartbeat_is_payload_wrapped() -> None:
    """ServerHeartbeatEvent keeps only the payload wrapper."""
    state = _MockState()
    hb = ServerHeartbeatEvent()
    events = await _collect_events(state, wrap_payload=True, events_to_send=[hb])
    assert len(events) == 2
    heartbeat = events[1]
    assert heartbeat["payload"]["type"] == "server.heartbeat"
    assert "directory" not in heartbeat
    assert "project" not in heartbeat


@pytest.mark.anyio
async def test_server_events_have_no_session_id() -> None:
    """server.connected and server.heartbeat lack sessionId at top level."""
    state = _MockState()
    hb = ServerHeartbeatEvent()
    events = await _collect_events(state, wrap_payload=True, events_to_send=[hb])
    assert "sessionId" not in events[0]  # envelope
    assert "sessionId" not in events[0]["payload"]  # server.connected payload
    assert "sessionId" not in events[1]  # envelope
    assert "sessionId" not in events[1]["payload"]  # server.heartbeat payload


# =============================================================================
# 4. All other events are wrapped in GlobalEvent with correct directory
# =============================================================================


@pytest.mark.anyio
async def test_session_created_is_wrapped() -> None:
    """SessionCreatedEvent is wrapped in GlobalEvent envelope."""
    state = _MockState(working_dir="/wrap/test")
    event = SessionCreatedEvent.create(session=_make_session("wrap1"))
    events = await _collect_events(state, wrap_payload=True, events_to_send=[event])

    wrapped = events[1]
    assert "directory" in wrapped
    assert "project" in wrapped
    assert "payload" in wrapped
    assert wrapped["directory"] == state.working_dir
    assert wrapped["payload"]["type"] == "session.created"


@pytest.mark.anyio
async def test_session_status_is_wrapped() -> None:
    """SessionStatusEvent is wrapped in GlobalEvent envelope."""
    state = _MockState(working_dir="/wrap/test2")
    event = SessionStatusEvent.create(session_id="wrap2", status_type="idle")
    events = await _collect_events(state, wrap_payload=True, events_to_send=[event])

    wrapped = events[1]
    assert "directory" in wrapped
    assert "payload" in wrapped
    assert wrapped["directory"] == state.working_dir


@pytest.mark.anyio
async def test_session_compacted_is_wrapped() -> None:
    """SessionCompactedEvent is wrapped in GlobalEvent envelope."""
    state = _MockState(working_dir="/wrap/compacted")
    event = SessionCompactedEvent.create(session_id="wrap3")
    events = await _collect_events(state, wrap_payload=True, events_to_send=[event])

    wrapped = events[1]
    assert "directory" in wrapped
    assert "payload" in wrapped
    assert wrapped["directory"] == state.working_dir


@pytest.mark.anyio
async def test_tui_session_select_is_wrapped() -> None:
    """TuiSessionSelectEvent is wrapped in GlobalEvent envelope."""
    state = _MockState(working_dir="/wrap/select")
    event = TuiSessionSelectEvent.create(session_id="wrap4")
    events = await _collect_events(state, wrap_payload=True, events_to_send=[event])

    wrapped = events[1]
    assert "directory" in wrapped
    assert "payload" in wrapped
    assert wrapped["directory"] == state.working_dir


# =============================================================================
# 5. sessionId appears at top level of payload for ALL event types
# =============================================================================


@pytest.mark.anyio
async def test_session_id_in_payload_session_status() -> None:
    """SessionStatusEvent payload has sessionId at top level."""
    event = SessionStatusEvent.create(session_id="top1", status_type="busy")
    result = _serialize_event(event, wrap_payload=False)
    data = json.loads(result)
    assert data["sessionId"] == "top1"


@pytest.mark.anyio
async def test_session_id_in_payload_part_delta() -> None:
    """PartDeltaEvent payload has sessionId at top level (critical for streaming)."""
    event = PartDeltaEvent.create(
        session_id="top2", message_id="m1", part_id="p1", delta="streaming text"
    )
    result = _serialize_event(event, wrap_payload=False)
    data = json.loads(result)
    assert data["sessionId"] == "top2"


@pytest.mark.anyio
async def test_session_id_in_payload_session_created() -> None:
    """SessionCreatedEvent payload has sessionId at top level."""
    event = SessionCreatedEvent.create(session=_make_session("top3"))
    result = _serialize_event(event, wrap_payload=False)
    data = json.loads(result)
    assert data["sessionId"] == "top3"


@pytest.mark.anyio
async def test_session_id_in_payload_message_updated() -> None:
    """MessageUpdatedEvent payload has sessionId at top level."""
    msg = UserMessage(id="m1", session_id="top4", time=TimeCreated(created=0))
    event = MessageUpdatedEvent.create(message=msg)
    result = _serialize_event(event, wrap_payload=False)
    data = json.loads(result)
    assert data["sessionId"] == "top4"


@pytest.mark.anyio
async def test_session_id_in_payload_command_executed() -> None:
    """CommandExecutedEvent payload has sessionId at top level."""
    event = CommandExecutedEvent.create(
        name="test", session_id="top5", arguments="", message_id="m1"
    )
    result = _serialize_event(event, wrap_payload=False)
    data = json.loads(result)
    assert data["sessionId"] == "top5"


@pytest.mark.anyio
async def test_session_id_in_payload_permission_events() -> None:
    """PermissionRequestEvent and PermissionResolvedEvent payloads have sessionId."""
    req = PermissionRequestEvent.create(
        session_id="top6",
        permission_id="p1",
        tool_name="bash",
        args_preview="ls",
        message="Allow?",
    )
    resolved = PermissionResolvedEvent.create(
        session_id="top6",
        request_id="p1",
        reply="once",
    )
    for event in (req, resolved):
        result = _serialize_event(event, wrap_payload=False)
        data = json.loads(result)
        assert data["sessionId"] == "top6"


@pytest.mark.anyio
async def test_session_id_absent_for_server_events() -> None:
    """ServerConnectedEvent and ServerHeartbeatEvent have no sessionId."""
    for event in (ServerConnectedEvent(), ServerHeartbeatEvent()):
        result = _serialize_event(event, wrap_payload=False)
        data = json.loads(result)
        assert "sessionId" not in data


@pytest.mark.anyio
async def test_session_id_in_global_event_payload_part_delta() -> None:
    """PartDeltaEvent in GlobalEvent envelope has sessionId inside payload."""
    state = _MockState()
    event = PartDeltaEvent.create(session_id="env1", message_id="m1", part_id="p1", delta="x")
    events = await _collect_events(state, wrap_payload=True, events_to_send=[event])
    wrapped = events[1]
    assert wrapped["payload"]["sessionId"] == "env1"


# =============================================================================
# 6. UserMessage events use nested model.variant format
# =============================================================================


def test_user_message_variant_nested_in_model() -> None:
    """UserMessage with model.variant serializes variant inside model object."""
    msg = UserMessage(
        id="m1",
        session_id="s1",
        time=TimeCreated(created=0),
        model=ModelRef(provider_id="openai", model_id="gpt-4o", variant="high"),
    )
    data = msg.model_dump(by_alias=True, exclude_none=True)
    assert "model" in data
    assert isinstance(data["model"], dict)
    assert data["model"]["variant"] == "high"
    # variant should NOT appear at top level
    assert "variant" not in data


def test_user_message_variant_not_at_top_level() -> None:
    """UserMessage serialization does NOT put variant at top level."""
    msg = UserMessage(
        id="m2",
        session_id="s2",
        time=TimeCreated(created=0),
        model=ModelRef(variant="medium"),
    )
    data = msg.model_dump(by_alias=True, exclude_none=True)
    assert "variant" not in data
    assert data["model"]["variant"] == "medium"


def test_user_message_variant_migration_from_top_level() -> None:
    """UserMessage validator migrates top-level variant into model.variant."""
    # Simulate old client sending variant at top level
    msg = UserMessage.model_validate({
        "id": "m3",
        "session_id": "s3",
        "time": {"created": 0},
        "variant": "low",
    })
    data = msg.model_dump(by_alias=True, exclude_none=True)
    # variant should now be inside model
    assert "variant" not in data
    assert data["model"]["variant"] == "low"


def test_user_message_no_variant_no_model() -> None:
    """UserMessage without variant or model has neither in output."""
    msg = UserMessage(
        id="m4",
        session_id="s4",
        time=TimeCreated(created=0),
    )
    data = msg.model_dump(by_alias=True, exclude_none=True)
    assert "variant" not in data
    assert "model" not in data


def test_message_updated_event_carries_nested_variant() -> None:
    """MessageUpdatedEvent with UserMessage preserves model.variant nesting."""
    msg = UserMessage(
        id="m5",
        session_id="s5",
        time=TimeCreated(created=0),
        model=ModelRef(variant="max"),
    )
    event = MessageUpdatedEvent.create(message=msg)
    result = _serialize_event(event, wrap_payload=False)
    data = json.loads(result)
    # sessionId at top level
    assert data["sessionId"] == "s5"
    # variant nested inside model in properties
    info = data["properties"]["info"]
    assert "variant" not in info  # no top-level variant
    assert info["model"]["variant"] == "max"


# =============================================================================
# 7. FileDiff events use v1.4.0+ schema (patch field, not before/after)
# =============================================================================


def test_file_diff_has_patch_field() -> None:
    """FileDiff model serializes with 'patch' field."""
    diff = FileDiff(
        file="src/main.py",
        patch="@@ -1,3 +1,4 @@\n-old line\n+new line\n+added line",
        additions=2,
        deletions=1,
        status="modified",
    )
    data = diff.model_dump(by_alias=True, exclude_none=True)
    assert "patch" in data
    assert data["patch"] == "@@ -1,3 +1,4 @@\n-old line\n+new line\n+added line"


def test_file_diff_no_before_after_fields() -> None:
    """FileDiff serialization does NOT contain 'before' or 'after' fields."""
    diff = FileDiff(
        file="src/util.py",
        patch="some patch text",
        additions=1,
        deletions=0,
        status="added",
    )
    data = diff.model_dump(by_alias=True, exclude_none=True)
    assert "before" not in data
    assert "after" not in data


def test_file_diff_patch_none_excluded() -> None:
    """FileDiff with patch=None excludes patch from output (exclude_none)."""
    diff = FileDiff(
        file="README.md",
        additions=0,
        deletions=0,
    )
    data = diff.model_dump(by_alias=True, exclude_none=True)
    assert "patch" not in data  # None excluded
    assert "before" not in data
    assert "after" not in data


def test_session_diff_event_serializes_patch_not_before_after() -> None:
    """SessionDiffEvent wraps FileDiff objects that have 'patch' not 'before'/'after'."""
    diff = FileDiff(
        file="app.py",
        patch="@@ -1 +1 @@\n-old\n+new",
        additions=1,
        deletions=1,
        status="modified",
    )
    event = SessionDiffEvent.create(session_id="diff1", diff=[diff])
    result = _serialize_event(event, wrap_payload=False)
    data = json.loads(result)

    diff_entries = data["properties"]["diff"]
    assert len(diff_entries) == 1
    entry = diff_entries[0]
    assert "patch" in entry
    assert entry["patch"] == "@@ -1 +1 @@\n-old\n+new"
    assert "before" not in entry
    assert "after" not in entry


def test_session_diff_in_global_event_has_patch() -> None:
    """SessionDiffEvent in GlobalEvent envelope serializes FileDiff with patch."""
    factory = GlobalEventFactory(directory="/tmp", project="abc")
    diff = FileDiff(
        file="config.yml",
        patch="@@ -1,2 +1,3 @@\n-key: old\n+key: new\n+key2: val",
        additions=2,
        deletions=1,
        status="modified",
    )
    event = SessionDiffEvent.create(session_id="envdiff1", diff=[diff])
    result = factory.wrap(event)
    data = json.loads(result)

    payload = data["payload"]
    entry = payload["properties"]["diff"][0]
    assert "patch" in entry
    assert "before" not in entry
    assert "after" not in entry


# =============================================================================
# 8. /event endpoint backward compatibility (raw events, no envelope)
# =============================================================================


@pytest.mark.anyio
async def test_event_endpoint_raw_events_no_envelope() -> None:
    """/event (wrap_payload=False) sends events without GlobalEvent wrapper."""
    state = _MockState()
    event = SessionStatusEvent.create(session_id="bc1", status_type="busy")
    events = await _collect_events(state, wrap_payload=False, events_to_send=[event])

    for evt in events:
        assert "directory" not in evt
        assert "project" not in evt
        assert "payload" not in evt


@pytest.mark.anyio
async def test_event_endpoint_session_id_at_top_level() -> None:
    """/event endpoint puts sessionId at top level for session events."""
    state = _MockState()
    event = SessionStatusEvent.create(session_id="bc2", status_type="idle")
    events = await _collect_events(state, wrap_payload=False, events_to_send=[event])
    session_data = events[1]
    assert session_data["sessionId"] == "bc2"


@pytest.mark.anyio
async def test_event_endpoint_part_delta_raw() -> None:
    """/event sends PartDeltaEvent as raw JSON with sessionId at top level."""
    state = _MockState()
    event = PartDeltaEvent.create(
        session_id="bc3", message_id="m1", part_id="p1", delta="hello world"
    )
    events = await _collect_events(state, wrap_payload=False, events_to_send=[event])
    delta_data = events[1]
    assert delta_data["type"] == "message.part.delta"
    assert delta_data["sessionId"] == "bc3"
    assert delta_data["properties"]["delta"] == "hello world"


@pytest.mark.anyio
async def test_event_endpoint_message_updated_raw() -> None:
    """/event sends MessageUpdatedEvent as raw JSON with sessionId at top level."""
    state = _MockState()
    msg = UserMessage(id="m1", session_id="bc4", time=TimeCreated(created=0))
    event = MessageUpdatedEvent.create(message=msg)
    events = await _collect_events(state, wrap_payload=False, events_to_send=[event])
    msg_data = events[1]
    assert msg_data["type"] == "message.updated"
    assert msg_data["sessionId"] == "bc4"


# =============================================================================
# 9. PartDeltaEvent streaming — the most critical event for "no response" symptom
# =============================================================================


@pytest.mark.anyio
async def test_part_delta_session_id_at_top_level() -> None:
    """PartDeltaEvent has sessionId injected at top level (was missing = no response)."""
    event = PartDeltaEvent.create(
        session_id="delta1", message_id="m1", part_id="p1", delta="text chunk"
    )
    result = _serialize_event(event, wrap_payload=False)
    data = json.loads(result)
    assert data["sessionId"] == "delta1"


@pytest.mark.anyio
async def test_part_delta_session_id_in_global_event_payload() -> None:
    """PartDeltaEvent wrapped in GlobalEvent has sessionId inside payload."""
    state = _MockState(working_dir="/delta/test")
    event = PartDeltaEvent.create(
        session_id="delta2", message_id="m1", part_id="p1", delta="streaming"
    )
    events = await _collect_events(state, wrap_payload=True, events_to_send=[event])

    wrapped = events[1]
    assert "directory" in wrapped
    assert wrapped["payload"]["sessionId"] == "delta2"
    assert wrapped["payload"]["type"] == "message.part.delta"


@pytest.mark.anyio
async def test_part_delta_extract_session_id() -> None:
    """_extract_session_id correctly extracts sessionId from PartDeltaEvent."""
    event = PartDeltaEvent.create(session_id="delta3", message_id="m1", part_id="p1", delta="x")
    sid = _extract_session_id(event)
    assert sid == "delta3"


@pytest.mark.anyio
async def test_part_delta_tui_filter_passes() -> None:
    """PartDeltaEvent in GlobalEvent passes the TUI routing filter."""
    wd = "/delta/filter"
    state = _MockState(working_dir=wd)
    event = PartDeltaEvent.create(session_id="delta4", message_id="m1", part_id="p1", delta="x")
    events = await _collect_events(state, wrap_payload=True, events_to_send=[event])

    wrapped = events[1]
    ge = GlobalEvent(
        directory=wrapped["directory"],
        project=wrapped.get("project"),
        payload=wrapped["payload"],
    )
    passes, reason = tui_event_filter(ge, wd)
    assert passes, f"PartDeltaEvent should pass filter, reason={reason}"


@pytest.mark.anyio
async def test_part_delta_multiple_deltas_stream() -> None:
    """Sequence of PartDeltaEvents (simulating streaming) all have correct sessionId."""
    state = _MockState(working_dir="/delta/stream")
    deltas = [
        PartDeltaEvent.create(
            session_id="stream1", message_id="m1", part_id="p1", delta=f"chunk{i}"
        )
        for i in range(5)
    ]
    events = await _collect_events(state, wrap_payload=True, events_to_send=deltas)

    # First event is payload-wrapped server.connected, then 5 wrapped deltas
    for i, wrapped in enumerate(events[1:], start=0):
        assert wrapped["payload"]["sessionId"] == "stream1"
        assert wrapped["payload"]["properties"]["delta"] == f"chunk{i}"


@pytest.mark.anyio
async def test_part_delta_properties_fields() -> None:
    """PartDeltaEvent payload has correct properties fields."""
    event = PartDeltaEvent.create(
        session_id="delta5",
        message_id="msg_abc",
        part_id="part_xyz",
        delta="hello text",
        field="text",
    )
    result = _serialize_event(event, wrap_payload=False)
    data = json.loads(result)
    assert data["type"] == "message.part.delta"
    assert data["sessionId"] == "delta5"
    props = data["properties"]
    assert props["sessionID"] == "delta5"  # camelCase alias
    assert props["messageID"] == "msg_abc"
    assert props["partID"] == "part_xyz"
    assert props["field"] == "text"
    assert props["delta"] == "hello text"


# =============================================================================
# Cross-cutting: mixed event sequences with correct wrapping
# =============================================================================


@pytest.mark.anyio
async def test_mixed_events_correct_wrapping_sequence() -> None:
    """Sequence of wrapped events all have correct format."""
    wd = "/mixed/test"
    state = _MockState(working_dir=wd)

    events_to_send: list[Event] = [
        SessionStatusEvent.create(session_id="mix1", status_type="busy"),
        ServerHeartbeatEvent(),
        PartDeltaEvent.create(session_id="mix2", message_id="m1", part_id="p1", delta="chunk"),
        ServerHeartbeatEvent(),
        SessionCompactedEvent.create(session_id="mix3"),
    ]
    events = await _collect_events(state, wrap_payload=True, events_to_send=events_to_send)

    # [0] server.connected — payload wrapped, no routing metadata
    assert events[0]["payload"]["type"] == "server.connected"
    assert "directory" not in events[0]

    # [1] session.status — wrapped
    assert "payload" in events[1]
    assert events[1]["payload"]["type"] == "session.status"
    assert events[1]["payload"]["sessionId"] == "mix1"
    assert events[1]["directory"] == state.working_dir

    # [2] server.heartbeat — payload wrapped, no routing metadata
    assert events[2]["payload"]["type"] == "server.heartbeat"
    assert "directory" not in events[2]

    # [3] part.delta — wrapped (CRITICAL)
    assert "payload" in events[3]
    assert events[3]["payload"]["type"] == "message.part.delta"
    assert events[3]["payload"]["sessionId"] == "mix2"
    assert events[3]["directory"] == state.working_dir

    # [4] server.heartbeat — payload wrapped, no routing metadata
    assert events[4]["payload"]["type"] == "server.heartbeat"
    assert "directory" not in events[4]

    # [5] session.compacted — wrapped
    assert "payload" in events[5]
    assert events[5]["payload"]["type"] == "session.compacted"
    assert events[5]["payload"]["sessionId"] == "mix3"
    assert events[5]["directory"] == state.working_dir


# =============================================================================
# Cross-cutting: sessionId consistency between top level and properties
# =============================================================================


def test_session_id_consistency_session_status() -> None:
    """SessionId at top level matches sessionID inside properties (alias)."""
    event = SessionStatusEvent.create(session_id="consist1", status_type="busy")
    result = _serialize_event(event, wrap_payload=False)
    data = json.loads(result)
    top_level_sid = data["sessionId"]
    props_sid = data["properties"]["sessionID"]  # camelCase alias
    assert top_level_sid == props_sid == "consist1"


def test_session_id_consistency_part_delta() -> None:
    """PartDeltaEvent: sessionId at top level matches sessionID in properties."""
    event = PartDeltaEvent.create(session_id="consist2", message_id="m1", part_id="p1", delta="x")
    result = _serialize_event(event, wrap_payload=False)
    data = json.loads(result)
    top_level_sid = data["sessionId"]
    props_sid = data["properties"]["sessionID"]
    assert top_level_sid == props_sid == "consist2"


def test_session_id_consistency_command_executed() -> None:
    """CommandExecutedEvent: sessionId at top level matches sessionID in properties."""
    event = CommandExecutedEvent.create(
        name="test", session_id="consist3", arguments="", message_id="m1"
    )
    result = _serialize_event(event, wrap_payload=False)
    data = json.loads(result)
    top_level_sid = data["sessionId"]
    props_sid = data["properties"]["sessionID"]
    assert top_level_sid == props_sid == "consist3"


# =============================================================================
# Exhaustive: all handled event types produce sessionId at top level
# =============================================================================


# Re-use the same event factories from test_global_event.py
_ALL_HANDLED_EVENTS_WITH_SID: list[tuple[str, Event]] = [
    ("session.deleted", SessionDeletedEvent.create(session_id="ex1")),
    ("session.status", SessionStatusEvent.create(session_id="ex2", status_type="busy")),
    ("session.idle", SessionCompactedEvent.create(session_id="ex3")),
    ("session.compacted", SessionCompactedEvent.create(session_id="ex4")),
    ("message.removed", SessionDeletedEvent.create(session_id="ex5")),
    (
        "message.part.delta",
        PartDeltaEvent.create(session_id="ex6", message_id="m1", part_id="p1", delta="x"),
    ),
    (
        "message.updated",
        MessageUpdatedEvent.create(
            message=UserMessage(id="m1", session_id="ex7", time=TimeCreated(created=0))
        ),
    ),
    ("session.created", SessionCreatedEvent.create(session=_make_session("ex8"))),
    ("session.diff", SessionDiffEvent.create(session_id="ex9", diff=[])),
    (
        "command.executed",
        CommandExecutedEvent.create(name="test", session_id="ex10", arguments="", message_id="m1"),
    ),
    ("tui.session.select", TuiSessionSelectEvent.create(session_id="ex11")),
]


@pytest.mark.parametrize(
    ("event_type_name", "event"),
    [(name, evt) for name, evt in _ALL_HANDLED_EVENTS_WITH_SID],
    ids=[name for name, _ in _ALL_HANDLED_EVENTS_WITH_SID],
)
def test_all_session_events_have_session_id_at_top_level(
    event_type_name: str,
    event: Event,
) -> None:
    """All session-scoped events produce sessionId at top level in serialized output."""
    result = _serialize_event(event, wrap_payload=False)
    data = json.loads(result)
    assert "sessionId" in data, f"{event_type_name} missing sessionId at top level"
    assert data["sessionId"] is not None, f"{event_type_name} has null sessionId"


# =============================================================================
# GlobalEventFactory.wrap() produces correct envelope structure
# =============================================================================


def test_factory_wrap_envelope_structure() -> None:
    """GlobalEventFactory.wrap() produces {directory, project, payload} structure."""
    factory = GlobalEventFactory(directory="/factory/test", project="proj123")
    event = PartDeltaEvent.create(session_id="fac1", message_id="m1", part_id="p1", delta="x")
    result = factory.wrap(event)
    data = json.loads(result)

    # Top-level keys
    assert set(data.keys()) >= {"directory", "project", "payload"}
    assert data["directory"] == "/factory/test"
    assert data["project"] == "proj123"

    # Payload structure
    payload = data["payload"]
    assert payload["type"] == "message.part.delta"
    assert payload["sessionId"] == "fac1"


def test_factory_wrap_session_created_has_session_id() -> None:
    """Factory.wrap(SessionCreatedEvent) has sessionId in payload."""
    factory = GlobalEventFactory(directory="/tmp", project="abc")
    event = SessionCreatedEvent.create(session=_make_session("fac2"))
    result = factory.wrap(event)
    data = json.loads(result)
    assert data["payload"]["sessionId"] == "fac2"


def test_factory_wrap_message_updated_has_session_id() -> None:
    """Factory.wrap(MessageUpdatedEvent) has sessionId in payload."""
    factory = GlobalEventFactory(directory="/tmp", project="abc")
    msg = UserMessage(id="m1", session_id="fac3", time=TimeCreated(created=0))
    event = MessageUpdatedEvent.create(message=msg)
    result = factory.wrap(event)
    data = json.loads(result)
    assert data["payload"]["sessionId"] == "fac3"


# =============================================================================
# Unicode preservation through the full SSE pipeline
# =============================================================================


@pytest.mark.anyio
async def test_unicode_part_delta_in_global_event() -> None:
    r"""PartDeltaEvent with CJK text preserves Unicode in GlobalEvent envelope."""
    state = _MockState(working_dir="/unicode/test")
    event = PartDeltaEvent.create(
        session_id="unicode1", message_id="m1", part_id="p1", delta="你好世界 🌍"
    )
    events = await _collect_events(state, wrap_payload=True, events_to_send=[event])
    wrapped = events[1]
    assert wrapped["payload"]["properties"]["delta"] == "你好世界 🌍"
    assert wrapped["payload"]["sessionId"] == "unicode1"


def test_unicode_file_diff_patch_preserved() -> None:
    r"""FileDiff with Unicode in patch preserves characters (not \uXXXX)."""
    diff = FileDiff(
        file="中文文件.py",
        patch="@@ -1 +1 @@\n-旧代码\n+新代码 🔥",
        additions=1,
        deletions=1,
        status="modified",
    )
    event = SessionDiffEvent.create(session_id="udiff1", diff=[diff])
    result = _serialize_event(event, wrap_payload=False)
    assert "中文文件" in result
    assert "新代码" in result
    assert "\\u" not in result
