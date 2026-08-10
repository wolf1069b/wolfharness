"""Tests for EventMapper ToolCallProgressEvent emission when args differ."""

from __future__ import annotations

from pydantic_ai import FunctionToolCallEvent, PartStartEvent
from pydantic_ai.messages import ToolCallPart
import pytest

from wolfharness.agents.events.events import (
    ToolCallProgressEvent,
    ToolCallStartEvent,
)
from wolfharness.orchestrator.event_mapper import EventMapper


@pytest.mark.unit
def test_emit_progress_event_when_args_differ() -> None:
    """Given a duplicate tool call start with different args, returns ToolCallProgressEvent."""
    mapper = EventMapper(agent_name="test-agent", message_id="msg-001")

    # First event — initial args (partial, as during streaming)
    event1 = FunctionToolCallEvent(
        part=ToolCallPart(
            tool_name="bash",
            args={"command": "ls"},
            tool_call_id="tc-progress-001",
        ),
    )
    result1 = mapper.map_event(event1)
    assert result1 is not None
    assert isinstance(result1, ToolCallStartEvent)
    assert result1.tool_call_id == "tc-progress-001"
    assert result1.raw_input == {"command": "ls"}

    # Second event — same tool_call_id, but args now have more detail
    event2 = PartStartEvent(
        index=0,
        part=ToolCallPart(
            tool_name="bash",
            args={"command": "ls -la /tmp"},
            tool_call_id="tc-progress-001",
        ),
    )
    result2 = mapper.map_event(event2)

    assert result2 is not None
    assert isinstance(result2, ToolCallProgressEvent)
    assert result2.tool_call_id == "tc-progress-001"
    assert result2.status == "in_progress"
    assert result2.tool_name == "bash"
    assert result2.tool_input == {"command": "ls -la /tmp"}


@pytest.mark.unit
def test_returns_none_when_args_identical() -> None:
    """Given a duplicate tool call start with identical args, returns None (dedup)."""
    mapper = EventMapper(agent_name="test-agent", message_id="msg-001")

    tool_part = ToolCallPart(
        tool_name="read",
        args={"path": "/tmp/test.txt"},
        tool_call_id="tc-dedup-002",
    )

    event1 = FunctionToolCallEvent(part=tool_part)
    result1 = mapper.map_event(event1)
    assert result1 is not None
    assert isinstance(result1, ToolCallStartEvent)

    # Same tool_call_id, same args — should dedup to None
    event2 = PartStartEvent(index=0, part=tool_part)
    result2 = mapper.map_event(event2)

    assert result2 is None


@pytest.mark.unit
def test_progress_event_updates_stored_input() -> None:
    """After emitting a progress event, the stored input is updated for future comparisons."""
    mapper = EventMapper(agent_name="test-agent", message_id="msg-001")

    # Initial start with partial args
    event1 = FunctionToolCallEvent(
        part=ToolCallPart(
            tool_name="bash",
            args={"command": "ls"},
            tool_call_id="tc-chain-003",
        ),
    )
    mapper.map_event(event1)

    # Progress: args updated to v2
    event2 = PartStartEvent(
        index=0,
        part=ToolCallPart(
            tool_name="bash",
            args={"command": "ls -la"},
            tool_call_id="tc-chain-003",
        ),
    )
    result2 = mapper.map_event(event2)
    assert result2 is not None
    assert isinstance(result2, ToolCallProgressEvent)
    assert result2.tool_input == {"command": "ls -la"}

    # Third event with same args as v2 — should dedup to None now
    event3 = PartStartEvent(
        index=0,
        part=ToolCallPart(
            tool_name="bash",
            args={"command": "ls -la"},
            tool_call_id="tc-chain-003",
        ),
    )
    result3 = mapper.map_event(event3)
    assert result3 is None
