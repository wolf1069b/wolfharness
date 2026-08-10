"""Tests for lifecycle types, protocols, and new event types."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from wolfharness.agents.events import (
    MessageReplacementEvent,
    RichAgentStreamEvent,
    StateUpdate,
    ToolCallUpdateEvent,
)
from wolfharness.lifecycle import (
    CommChannel,
    EventEnvelope,
    EventTransport,
    Feedback,
    Journal,
    Prompt,
    ResumeResult,
    RunState,
    SnapshotStore,
    ToolExecutionRecord,
    TriggerSource,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import Any

pytestmark = pytest.mark.unit


# --- RunState enum tests ---


def test_runstate_has_idle_value():
    """RunState.IDLE exists and has value 'idle'."""
    assert RunState.IDLE.value == "idle"


def test_runstate_has_running_value():
    """RunState.RUNNING exists and has value 'running'."""
    assert RunState.RUNNING.value == "running"


def test_runstate_has_done_value():
    """RunState.DONE exists and has value 'done'."""
    assert RunState.DONE.value == "done"


def test_runstate_is_enum_instance():
    """RunState members are instances of RunState and Enum."""
    assert isinstance(RunState.IDLE, RunState)
    assert isinstance(RunState.RUNNING, RunState)
    assert isinstance(RunState.DONE, RunState)


def test_runstate_has_exactly_three_members():
    """RunState has exactly 3 members."""
    assert len(list(RunState)) == 3


# --- Prompt dataclass tests ---


def test_prompt_construction_with_defaults():
    """Prompt with only content uses default priority and empty metadata."""
    prompt = Prompt(content="hello")
    assert prompt.content == "hello"
    assert prompt.priority == "normal"
    assert prompt.metadata == {}


def test_prompt_construction_with_priority():
    """Prompt accepts custom priority."""
    prompt = Prompt(content="urgent", priority="asap")
    assert prompt.priority == "asap"


def test_prompt_construction_with_metadata():
    """Prompt accepts custom metadata."""
    prompt = Prompt(content="hello", metadata={"source": "test"})
    assert prompt.metadata == {"source": "test"}


def test_prompt_default_metadata_is_independent():
    """Each Prompt instance gets its own metadata dict (no shared mutable default)."""
    p1 = Prompt(content="a")
    p2 = Prompt(content="b")
    p1.metadata["key"] = "val"
    assert "key" not in p2.metadata


# --- Feedback dataclass tests ---


def test_feedback_construction_steer():
    """Feedback with is_steer=True."""
    fb = Feedback(content="Use Python 3.13", is_steer=True)
    assert fb.content == "Use Python 3.13"
    assert fb.is_steer is True


def test_feedback_construction_followup():
    """Feedback with is_steer=False."""
    fb = Feedback(content="Check tests next", is_steer=False)
    assert fb.is_steer is False


# --- ResumeResult dataclass tests ---


def test_resume_result_construction_inflight():
    """ResumeResult for in-flight Turn."""
    result = ResumeResult(
        is_inflight=True,
        state=RunState.RUNNING,
        events=["event1", "event2"],
        inflight_turn_id="turn_001",
    )
    assert result.is_inflight is True
    assert result.state == RunState.RUNNING
    assert result.events == ["event1", "event2"]
    assert result.inflight_turn_id == "turn_001"


def test_resume_result_construction_no_inflight():
    """ResumeResult for normal recovery (no in-flight Turn)."""
    result = ResumeResult(
        is_inflight=False,
        state=RunState.IDLE,
        events=[],
        inflight_turn_id=None,
    )
    assert result.is_inflight is False
    assert result.inflight_turn_id is None


# --- ToolExecutionRecord dataclass tests ---


def test_tool_execution_record_construction():
    """ToolExecutionRecord with all fields."""
    record = ToolExecutionRecord(
        turn_id="turn_001",
        tool_name="bash",
        args={"command": "ls"},
        result="file1\nfile2",
        status="completed",
    )
    assert record.turn_id == "turn_001"
    assert record.tool_name == "bash"
    assert record.args == {"command": "ls"}
    assert record.result == "file1\nfile2"
    assert record.status == "completed"


def test_tool_execution_record_with_none_result():
    """ToolExecutionRecord with result=None (not completed)."""
    record = ToolExecutionRecord(
        turn_id="turn_002",
        tool_name="read",
        args={"path": "/tmp/test"},
        result=None,
        status="failed",
    )
    assert record.result is None
    assert record.status == "failed"


# --- EventEnvelope dataclass tests ---


def test_event_envelope_construction_with_defaults():
    """EventEnvelope with required fields, verifying defaults."""
    env = EventEnvelope(
        event_type="test",
        session_id="s1",
        timestamp="2026-01-01T00:00:00Z",
        payload={},
    )
    assert env.schema_version == "1.0.0"
    assert env.event_type == "test"
    assert env.session_id == "s1"
    assert env.turn_id is None
    assert env.timestamp == "2026-01-01T00:00:00Z"
    assert env.payload == {}
    assert env.seq is None
    assert env.metadata == {}


def test_event_envelope_construction_with_all_fields():
    """EventEnvelope with all fields populated."""
    env = EventEnvelope(
        schema_version="2.0.0",
        event_type="tool_call",
        session_id="s1",
        turn_id="t1",
        timestamp="2026-01-01T00:00:00Z",
        payload={"tool": "bash"},
        seq=42,
        metadata={"source": "test"},
    )
    assert env.schema_version == "2.0.0"
    assert env.turn_id == "t1"
    assert env.seq == 42
    assert env.metadata == {"source": "test"}


def test_event_envelope_default_metadata_is_independent():
    """Each EventEnvelope instance gets its own metadata dict."""
    e1 = EventEnvelope(event_type="a", session_id="s1", timestamp="t", payload={})
    e2 = EventEnvelope(event_type="b", session_id="s2", timestamp="t", payload={})
    e1.metadata["key"] = "val"
    assert "key" not in e2.metadata


def test_event_envelope_default_payload_is_independent():
    """Each EventEnvelope instance gets its own payload dict."""
    e1 = EventEnvelope(event_type="a", session_id="s1", timestamp="t", payload={})
    e2 = EventEnvelope(event_type="b", session_id="s2", timestamp="t", payload={})
    e1.payload["key"] = "val"
    assert "key" not in e2.payload


# --- Protocol isinstance tests ---


class _DummyTriggerSource:
    """Minimal TriggerSource implementation for isinstance testing."""

    def subscribe(self, run_loop: Any) -> None: ...

    def poll(self) -> Prompt | None:
        return None

    def close(self) -> None: ...


class _DummyJournal:
    """Minimal Journal implementation for isinstance testing."""

    def append(self, event: Any) -> int:
        return 1

    def upsert(self, key: str, event: Any) -> int:
        return 1

    async def replay(self, from_seq: int = 0, to_seq: int | None = None) -> AsyncIterator[Any]:
        yield  # type: ignore[misc]

    def resume(self, snapshot_store: Any) -> ResumeResult | None:
        return None

    def compact(self, before_seq: int) -> None: ...

    def clear(self) -> None: ...

    def log_tool_execution(self, record: ToolExecutionRecord) -> None: ...

    def get_tool_executions(self, turn_id: str) -> list[ToolExecutionRecord]:
        return []

    _replaying: bool = False


class _DummySnapshotStore:
    """Minimal SnapshotStore implementation for isinstance testing."""

    def save(self, state: Any) -> int:
        return 1

    def load(self) -> tuple[Any, int] | None:
        return None

    def save_turn_result(self, turn_id: str, result: Any) -> None: ...

    def has_turn_result(self, turn_id: str) -> bool:
        return False

    def clear(self) -> None: ...


class _DummyCommChannel:
    """Minimal CommChannel implementation for isinstance testing."""

    def set_replaying(self, flag: bool) -> None: ...

    @property
    def publishes_to_event_bus(self) -> bool:
        return False

    @property
    def journal(self):  # type: ignore[no-untyped-def]
        return None

    def attach(self, run_loop: Any) -> None: ...

    def on_state_change(self, state: RunState) -> None: ...

    async def publish(self, event: Any) -> None: ...

    def recv(self) -> Feedback | None:
        return None

    def deliver_feedback(self, feedback: Feedback) -> bool:
        return False

    def revoke(self, message_id: str) -> bool:
        return False

    def replace(self, message_id: str, new_content: str | list[Any]) -> bool:
        return False

    def close(self) -> None: ...


class _DummyEventTransport:
    """Minimal EventTransport implementation for isinstance testing."""

    async def publish(self, envelope: EventEnvelope) -> None: ...

    async def subscribe(self, topic: str, from_seq: int = 0) -> AsyncIterator[EventEnvelope]:
        yield  # type: ignore[misc]

    def ack(self, seq: int) -> None: ...

    def close(self) -> None: ...


def test_trigger_source_protocol_isinstance():
    """TriggerSource isinstance check passes for compliant implementation."""
    assert isinstance(_DummyTriggerSource(), TriggerSource)


def test_journal_protocol_isinstance():
    """Journal isinstance check passes for compliant implementation."""
    assert isinstance(_DummyJournal(), Journal)


def test_snapshot_store_protocol_isinstance():
    """SnapshotStore isinstance check passes for compliant implementation."""
    assert isinstance(_DummySnapshotStore(), SnapshotStore)


def test_comm_channel_protocol_isinstance():
    """CommChannel isinstance check passes for compliant implementation."""
    assert isinstance(_DummyCommChannel(), CommChannel)


def test_event_transport_protocol_isinstance():
    """EventTransport isinstance check passes for compliant implementation."""
    assert isinstance(_DummyEventTransport(), EventTransport)


def test_protocol_isinstance_fails_for_non_compliant():
    """Protocol isinstance check fails for non-compliant objects."""
    assert not isinstance(42, TriggerSource)
    assert not isinstance("hello", Journal)
    assert not isinstance([], SnapshotStore)
    assert not isinstance({}, CommChannel)
    assert not isinstance(None, EventTransport)


# --- StateUpdate event tests ---


def test_state_update_construction():
    """StateUpdate with session_id and state."""
    event = StateUpdate(session_id="s1", state=RunState.RUNNING)
    assert event.session_id == "s1"
    assert event.state == RunState.RUNNING
    assert event.stop_reason is None
    assert event.event_kind == "state_update"


def test_state_update_with_stop_reason():
    """StateUpdate with crash recovery stop_reason."""
    event = StateUpdate(
        session_id="s1",
        state=RunState.IDLE,
        stop_reason="crash_recovery",
    )
    assert event.stop_reason == "crash_recovery"


def test_state_update_requires_session_id():
    """StateUpdate without session_id raises TypeError."""
    with pytest.raises(TypeError):
        StateUpdate(state=RunState.IDLE)  # type: ignore[call-arg]


def test_state_update_requires_state():
    """StateUpdate without state raises TypeError."""
    with pytest.raises(TypeError):
        StateUpdate(session_id="s1")  # type: ignore[call-arg]


# --- ToolCallUpdateEvent tests ---


def test_tool_call_update_event_construction():
    """ToolCallUpdateEvent with tool_call_id and tool_name."""
    event = ToolCallUpdateEvent(
        tool_call_id="tc_001",
        tool_name="bash",
    )
    assert event.tool_call_id == "tc_001"
    assert event.tool_name == "bash"
    assert event.status == "in_progress"
    assert event.title is None
    assert event.tool_input == {}
    assert event.tool_result is None
    assert event.session_id == ""
    assert event.event_kind == "tool_call_update"


def test_tool_call_update_event_with_all_fields():
    """ToolCallUpdateEvent with all fields populated."""
    event = ToolCallUpdateEvent(
        tool_call_id="tc_002",
        tool_name="read",
        status="completed",
        title="Reading file",
        tool_input={"path": "/tmp/test"},
        tool_result="content",
        session_id="s1",
    )
    assert event.status == "completed"
    assert event.tool_result == "content"


def test_tool_call_update_event_requires_tool_call_id():
    """ToolCallUpdateEvent without tool_call_id raises TypeError."""
    with pytest.raises(TypeError):
        ToolCallUpdateEvent(tool_name="bash")  # type: ignore[call-arg]


# --- MessageReplacementEvent tests ---


def test_message_replacement_event_construction():
    """MessageReplacementEvent with message_id and content."""
    event = MessageReplacementEvent(
        message_id="msg_001",
        content="replacement text",
    )
    assert event.message_id == "msg_001"
    assert event.content == "replacement text"
    assert event.session_id == ""
    assert event.event_kind == "message_replacement"


def test_message_replacement_event_requires_message_id():
    """MessageReplacementEvent without message_id raises TypeError."""
    with pytest.raises(TypeError):
        MessageReplacementEvent(content="text")  # type: ignore[call-arg]


# --- RichAgentStreamEvent union tests ---


def test_state_update_in_rich_agent_stream_event_union():
    """StateUpdate is part of RichAgentStreamEvent union."""
    from typing import get_args

    # RichAgentStreamEvent is a generic type alias (TypeAliasType);
    # __value__ holds the actual union type.
    args = get_args(RichAgentStreamEvent.__value__)
    assert StateUpdate in args


def test_tool_call_update_event_in_rich_agent_stream_event_union():
    """ToolCallUpdateEvent is part of RichAgentStreamEvent union."""
    from typing import get_args

    args = get_args(RichAgentStreamEvent.__value__)
    assert ToolCallUpdateEvent in args


def test_message_replacement_event_in_rich_agent_stream_event_union():
    """MessageReplacementEvent is part of RichAgentStreamEvent union."""
    from typing import get_args

    args = get_args(RichAgentStreamEvent.__value__)
    assert MessageReplacementEvent in args
