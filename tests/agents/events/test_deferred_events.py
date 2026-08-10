"""Tests for ToolCallDeferredEvent and SessionResumeEvent JSON round-trip serialization."""

from __future__ import annotations

from dataclasses import asdict
import json

import pytest

from wolfharness.agents.events.events import (
    RichAgentStreamEvent,
    SessionResumeEvent,
    ToolCallDeferredEvent,
)


pytestmark = [pytest.mark.unit]


def test_tool_call_deferred_event_json_roundtrip() -> None:
    """ToolCallDeferredEvent survives JSON roundtrip with all 6 fields preserved."""
    event = ToolCallDeferredEvent(
        tool_call_id="tc-001",
        tool_name="bash",
        deferred_strategy="block",
        deferred_handle="uuid-abc-123",
        status="pending",
        session_id="sess-42",
    )

    # Serialize
    data = json.dumps(asdict(event))
    # Deserialize
    restored = ToolCallDeferredEvent(**json.loads(data))

    assert restored.tool_call_id == "tc-001"
    assert restored.tool_name == "bash"
    assert restored.deferred_strategy == "block"
    assert restored.deferred_handle == "uuid-abc-123"
    assert restored.status == "pending"
    assert restored.session_id == "sess-42"
    assert restored.event_kind == "tool_call_deferred"


def test_tool_call_deferred_event_defaults() -> None:
    """ToolCallDeferredEvent has empty-string defaults for deferred_handle and session_id."""
    event = ToolCallDeferredEvent(
        tool_call_id="tc-002",
        tool_name="read",
        deferred_strategy="continue",
        status="resolved",
    )

    assert event.deferred_handle == ""
    assert event.session_id == ""
    assert event.status == "resolved"


@pytest.mark.parametrize("strategy", ["block", "continue", "stream"])
def test_tool_call_deferred_event_all_strategies(strategy: str) -> None:
    """ToolCallDeferredEvent accepts all three deferred_strategy values."""
    event = ToolCallDeferredEvent(
        tool_call_id="tc-003",
        tool_name="grep",
        deferred_strategy=strategy,
        status="pending",
    )
    assert event.deferred_strategy == strategy


@pytest.mark.parametrize("status", ["pending", "resolved", "expired"])
def test_tool_call_deferred_event_all_statuses(status: str) -> None:
    """ToolCallDeferredEvent accepts all three status values."""
    event = ToolCallDeferredEvent(
        tool_call_id="tc-003",
        tool_name="grep",
        deferred_strategy="block",
        status=status,
    )
    assert event.status == status


def test_session_resume_event_json_roundtrip() -> None:
    """SessionResumeEvent survives JSON roundtrip with all 3 fields preserved."""
    event = SessionResumeEvent(
        session_id="sess-resume-1",
        resolved_call_count=5,
        source="tool_result",
    )

    # Serialize
    data = json.dumps(asdict(event))
    # Deserialize
    restored = SessionResumeEvent(**json.loads(data))

    assert restored.session_id == "sess-resume-1"
    assert restored.resolved_call_count == 5
    assert restored.source == "tool_result"
    assert restored.event_kind == "session_resume"


def test_session_resume_event_default_source() -> None:
    """SessionResumeEvent defaults source to empty string."""
    event = SessionResumeEvent(
        session_id="sess-default",
        resolved_call_count=0,
    )

    assert event.source == ""


def test_events_in_rich_agent_stream_event_union() -> None:
    """Both events are members of the RichAgentStreamEvent union type.

    RichAgentStreamEvent is a ``type`` alias, so isinstance() cannot be used.
    Instead verify that both event instances can be passed to a function
    expecting RichAgentStreamEvent — i.e., they type-check as union members.
    """

    def accept_event(e: RichAgentStreamEvent[object]) -> None:
        _ = e  # just verifying it compiles

    deferred = ToolCallDeferredEvent(
        tool_call_id="tc-union",
        tool_name="bash",
        deferred_strategy="block",
        status="pending",
    )
    resume = SessionResumeEvent(
        session_id="sess-union",
        resolved_call_count=1,
        source="tool_result",
    )

    # Both events should be valid arguments (no TypeError at runtime)
    accept_event(deferred)
    accept_event(resume)
