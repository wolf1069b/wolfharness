"""Tests for EventMapper — PydanticAI to AgentPool event translation."""

from __future__ import annotations

from pydantic_ai import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartStartEvent,
)
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
import pytest

from wolfharness.agents.events.events import (
    RunStartedEvent,
    ToolCallCompleteEvent,
    ToolCallStartEvent,
)
from wolfharness.orchestrator.event_mapper import EventMapper


@pytest.mark.unit
def test_function_tool_call_event_maps_to_tool_call_start() -> None:
    """Given a FunctionToolCallEvent, map_event returns a ToolCallStartEvent with correct fields."""
    mapper = EventMapper(agent_name="test-agent", message_id="msg-001")
    event = FunctionToolCallEvent(
        part=ToolCallPart(
            tool_name="bash",
            args={"command": "ls -la"},
            tool_call_id="tc-001",
        ),
    )

    result = mapper.map_event(event)

    assert result is not None
    assert isinstance(result, ToolCallStartEvent)
    assert result.tool_call_id == "tc-001"
    assert result.tool_name == "bash"
    assert result.title == "Executing: bash"
    assert result.kind == "other"
    assert result.raw_input == {"command": "ls -la"}


@pytest.mark.unit
def test_part_start_event_with_tool_call_maps_to_tool_call_start() -> None:
    """Given a PartStartEvent with BaseToolCallPart, map_event returns a ToolCallStartEvent."""
    mapper = EventMapper(agent_name="test-agent", message_id="msg-001")
    event = PartStartEvent(
        index=0,
        part=ToolCallPart(
            tool_name="read",
            args={"path": "/tmp/test.txt"},
            tool_call_id="tc-002",
        ),
    )

    result = mapper.map_event(event)

    assert result is not None
    assert isinstance(result, ToolCallStartEvent)
    assert result.tool_call_id == "tc-002"
    assert result.tool_name == "read"
    assert result.title == "Executing: read"
    assert result.raw_input == {"path": "/tmp/test.txt"}


@pytest.mark.unit
def test_function_tool_result_event_maps_to_tool_call_complete() -> None:
    """Given a FunctionToolResultEvent after a start, map_event returns a ToolCallCompleteEvent."""
    mapper = EventMapper(agent_name="test-agent", message_id="msg-001")

    # First, emit the start event so the mapper tracks the tool call
    start_event = FunctionToolCallEvent(
        part=ToolCallPart(
            tool_name="bash",
            args={"command": "echo hello"},
            tool_call_id="tc-003",
        ),
    )
    mapper.map_event(start_event)

    # Now emit the result event
    result_event = FunctionToolResultEvent(
        part=ToolReturnPart(
            tool_name="bash",
            tool_call_id="tc-003",
            content="hello\n",
        ),
    )

    result = mapper.map_event(result_event)

    assert result is not None
    assert isinstance(result, ToolCallCompleteEvent)
    assert result.tool_call_id == "tc-003"
    assert result.tool_name == "bash"
    assert result.tool_input == {"command": "echo hello"}
    assert result.tool_result == "hello\n"
    assert result.agent_name == "test-agent"
    assert result.message_id == "msg-001"


@pytest.mark.unit
def test_rich_agent_stream_event_passes_through() -> None:
    """Given an unmatched event that IS a RichAgentStreamEvent, map_event passes it through."""
    mapper = EventMapper(agent_name="test-agent", message_id="msg-001")
    event = RunStartedEvent(run_id="run-123", agent_name="test-agent")

    result = mapper.map_event(event)

    assert result is event


@pytest.mark.unit
def test_unknown_object_returns_none() -> None:
    """Given an object that is not a RichAgentStreamEvent, map_event returns None."""
    mapper = EventMapper(agent_name="test-agent", message_id="msg-001")

    result = mapper.map_event("not an event")  # type: ignore[arg-type]

    assert result is None


@pytest.mark.unit
def test_multiple_tool_calls_tracked_separately() -> None:
    """Given multiple tool calls with different IDs, each is tracked independently."""
    mapper = EventMapper(agent_name="test-agent", message_id="msg-001")

    # First tool call
    start1 = FunctionToolCallEvent(
        part=ToolCallPart(
            tool_name="bash",
            args={"command": "ls"},
            tool_call_id="tc-A",
        ),
    )
    result1 = mapper.map_event(start1)
    assert result1 is not None
    assert isinstance(result1, ToolCallStartEvent)
    assert result1.tool_call_id == "tc-A"

    # Second tool call (different ID)
    start2 = PartStartEvent(
        index=1,
        part=ToolCallPart(
            tool_name="read",
            args={"path": "/etc/hosts"},
            tool_call_id="tc-B",
        ),
    )
    result2 = mapper.map_event(start2)
    assert result2 is not None
    assert isinstance(result2, ToolCallStartEvent)
    assert result2.tool_call_id == "tc-B"

    # First result
    complete1 = FunctionToolResultEvent(
        part=ToolReturnPart(
            tool_name="bash",
            tool_call_id="tc-A",
            content="file1\nfile2\n",
        ),
    )
    result3 = mapper.map_event(complete1)
    assert result3 is not None
    assert isinstance(result3, ToolCallCompleteEvent)
    assert result3.tool_call_id == "tc-A"
    assert result3.tool_name == "bash"

    # Second result
    complete2 = FunctionToolResultEvent(
        part=ToolReturnPart(
            tool_name="read",
            tool_call_id="tc-B",
            content="127.0.0.1 localhost",
        ),
    )
    result4 = mapper.map_event(complete2)
    assert result4 is not None
    assert isinstance(result4, ToolCallCompleteEvent)
    assert result4.tool_call_id == "tc-B"
    assert result4.tool_name == "read"


@pytest.mark.unit
def test_duplicate_tool_call_start_returns_none() -> None:
    """Given a duplicate tool call start for the same ID, map_event returns None."""
    mapper = EventMapper(agent_name="test-agent", message_id="msg-001")

    tool_part = ToolCallPart(
        tool_name="bash",
        args={"command": "ls"},
        tool_call_id="tc-dedup",
    )
    event1 = FunctionToolCallEvent(part=tool_part)
    event2 = PartStartEvent(index=0, part=tool_part)

    result1 = mapper.map_event(event1)
    assert result1 is not None
    assert isinstance(result1, ToolCallStartEvent)

    result2 = mapper.map_event(event2)
    assert result2 is None


@pytest.mark.unit
def test_tool_kind_map_lookup() -> None:
    """Given tool_kind_map is populated, map_event uses it for the kind field."""
    mapper = EventMapper(agent_name="test-agent", message_id="msg-001")
    mapper.tool_kind_map = {"bash": "execute", "read": "read"}

    event = FunctionToolCallEvent(
        part=ToolCallPart(
            tool_name="bash",
            args={"command": "ls"},
            tool_call_id="tc-kind",
        ),
    )

    result = mapper.map_event(event)

    assert result is not None
    assert isinstance(result, ToolCallStartEvent)
    assert result.kind == "execute"


@pytest.mark.unit
def test_result_without_start_returns_none() -> None:
    """Given a FunctionToolResultEvent with no preceding start, map_event returns None."""
    mapper = EventMapper(agent_name="test-agent", message_id="msg-001")

    result_event = FunctionToolResultEvent(
        part=ToolReturnPart(
            tool_name="bash",
            tool_call_id="tc-orphan",
            content="orphan result",
        ),
    )

    result = mapper.map_event(result_event)

    assert result is None


@pytest.mark.unit
def test_string_args_parsed_to_dict() -> None:
    """Given a ToolCallPart with JSON string args, map_event parses them into raw_input dict."""
    mapper = EventMapper(agent_name="test-agent", message_id="msg-001")

    event = FunctionToolCallEvent(
        part=ToolCallPart(
            tool_name="bash",
            args='{"command": "echo hello"}',
            tool_call_id="tc-str-args",
        ),
    )

    result = mapper.map_event(event)

    assert result is not None
    assert isinstance(result, ToolCallStartEvent)
    assert result.raw_input == {"command": "echo hello"}


@pytest.mark.unit
def test_flush_cancelled_tool_calls_returns_error_events_for_pending() -> None:
    """flush_cancelled_tool_calls returns ToolCallCompleteEvent with error metadata."""
    mapper = EventMapper(agent_name="test-agent", message_id="msg-001")

    mapper.map_event(
        FunctionToolCallEvent(
            part=ToolCallPart(tool_name="bash", args={"command": "ls -la"}, tool_call_id="tc-001"),
        ),
    )
    mapper.map_event(
        FunctionToolCallEvent(
            part=ToolCallPart(
                tool_name="read", args={"path": "/tmp/test.txt"}, tool_call_id="tc-002"
            ),
        ),
    )

    events = mapper.flush_cancelled_tool_calls()

    assert len(events) == 2
    assert all(isinstance(e, ToolCallCompleteEvent) for e in events)
    assert events[0].tool_call_id == "tc-001"
    assert events[0].tool_name == "bash"
    assert events[0].metadata == {"is_error": True, "cancelled": True}
    assert "cancelled" in str(events[0].tool_result).lower()
    assert events[1].tool_call_id == "tc-002"
    assert events[1].metadata == {"is_error": True, "cancelled": True}


@pytest.mark.unit
def test_flush_cancelled_tool_calls_clears_pending_state() -> None:
    """After flushing, _pending_tool_calls and _pending_tool_inputs are cleared."""
    mapper = EventMapper(agent_name="test-agent", message_id="msg-001")

    mapper.map_event(
        FunctionToolCallEvent(
            part=ToolCallPart(tool_name="bash", args={"command": "ls"}, tool_call_id="tc-001"),
        ),
    )

    events = mapper.flush_cancelled_tool_calls()
    assert len(events) == 1
    assert mapper.flush_cancelled_tool_calls() == []
    assert mapper._pending_tool_calls == {}
    assert mapper._pending_tool_inputs == {}


@pytest.mark.unit
def test_flush_cancelled_tool_calls_empty_when_no_pending() -> None:
    """flush_cancelled_tool_calls returns empty list when no pending tools."""
    mapper = EventMapper(agent_name="test-agent", message_id="msg-001")
    assert mapper.flush_cancelled_tool_calls() == []


@pytest.mark.unit
def test_flush_cancelled_tool_calls_after_partial_completion() -> None:
    """flush_cancelled_tool_calls only returns events for uncompleted tools."""
    mapper = EventMapper(agent_name="test-agent", message_id="msg-001")

    mapper.map_event(
        FunctionToolCallEvent(
            part=ToolCallPart(tool_name="bash", args={"command": "ls"}, tool_call_id="tc-001"),
        ),
    )
    mapper.map_event(
        FunctionToolCallEvent(
            part=ToolCallPart(tool_name="read", args={"path": "/tmp"}, tool_call_id="tc-002"),
        ),
    )
    mapper.map_event(
        FunctionToolResultEvent(
            part=ToolReturnPart(tool_name="bash", content="file1.txt", tool_call_id="tc-001"),
        ),
    )

    events = mapper.flush_cancelled_tool_calls()
    assert len(events) == 1
    assert events[0].tool_call_id == "tc-002"
    assert events[0].tool_name == "read"
