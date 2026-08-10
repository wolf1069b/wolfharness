"""Mock subagent events for ACP event converter snapshot tests.

These fixtures provide test event sequences for both tool_box and inline modes.
They represent realistic subagent activity patterns.

Per RFC-0001:

**Tool Box Mode (Summary Updates Only)**:
- Title format: [{icon} {role}]: {update}
- Content field is NEVER sent in tool_box mode
- Summary updates only via title field
- Title updates track: Thinking, Call params, Tool completed

**Inline Mode (All Events as Tool Outputs)**:
- All events treated as tool outputs (ToolCallProgress/ToolCallStart)
- Subagents distinguished via title [{source_name}]
- No AgentMessageChunk.text events after header
- Avoids concurrency issues: Multiple agents thinking don't conflict
  (same event types, different titles)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)
import pytest

from wolfharness.agents.events import (
    PartDeltaEvent,
    PartStartEvent,
    SpawnSessionStart,
    StreamCompleteEvent,
    SubAgentEvent,
)


if TYPE_CHECKING:
    from collections.abc import Callable


def get_text_start_event(text: str, index: int = 0) -> PartStartEvent:
    """Create a text part start event."""
    return PartStartEvent(index=index, part=TextPart(content=text))


def get_text_delta_event(delta: str, index: int = 0) -> PartDeltaEvent:
    """Create a text part delta event."""
    return PartDeltaEvent(index=index, delta=TextPartDelta(content_delta=delta))


def get_thinking_start_event(thinking: str, index: int = 0) -> PartStartEvent:
    """Create a thinking part start event."""
    return PartStartEvent(index=index, part=ThinkingPart(content=thinking))


def get_thinking_delta_event(delta: str, index: int = 0) -> PartDeltaEvent:
    """Create a thinking part delta event."""
    return PartDeltaEvent(index=index, delta=ThinkingPartDelta(content_delta=delta))


def get_tool_call_start_event(
    tool_name: str,
    tool_call_id: str,
    args: dict[str, Any],
) -> FunctionToolCallEvent:
    """Create a function tool call start event."""
    # Create a proper ToolCallPart with required fields

    part = ToolCallPart(
        tool_name=tool_name,
        args=args,
        tool_call_id=tool_call_id,
    )
    return FunctionToolCallEvent(part=part)


def get_tool_result_event(
    tool_name: str,
    tool_call_id: str,
    result: str,
) -> FunctionToolResultEvent:
    """Create a function tool result event."""
    # ToolReturnPart: content + tool_name (tool_call_id is auto-generated)
    part = ToolReturnPart(content=result, tool_name=tool_name)
    return FunctionToolResultEvent(part=part)


def get_tool_error_event(
    tool_name: str,
    tool_call_id: str,
    error_message: str,
) -> FunctionToolResultEvent:
    """Create a function tool error event (RetryPromptPart)."""
    # RetryPromptPart: content + tool_name
    part = RetryPromptPart(content=error_message, tool_name=tool_name)
    return FunctionToolResultEvent(part=part)


def get_stream_complete_event() -> StreamCompleteEvent[Any]:
    """Create a stream complete event."""
    from wolfharness.messaging import ChatMessage

    message = ChatMessage(
        role="assistant",
        content="Test complete",
        model_name="test-model",
    )
    return StreamCompleteEvent(message=message)


def get_subagent_event(
    source_name: str,
    inner_event: Any,
    source_type: str = "agent",
    depth: int = 1,
    child_session_id: str | None = None,
) -> SubAgentEvent:
    """Wrap an event in a SubAgentEvent."""
    return SubAgentEvent(
        source_name=source_name,
        source_type=source_type,  # type: ignore[arg-type]
        event=inner_event,
        depth=depth,
        child_session_id=child_session_id,
    )


def get_spawn_session_start_event(
    child_session_id: str = "sub_001",
    source_name: str = "researcher",
    description: str = "Research task",
    spawn_mechanism: str = "spawn",
) -> SpawnSessionStart:
    """Create a minimal SpawnSessionStart for zed-mode testing.

    Args:
        child_session_id: The child session ID to use.
        source_name: Name of the agent being spawned.
        description: Human-readable description.
        spawn_mechanism: How the subagent was created.
    """
    return SpawnSessionStart(
        child_session_id=child_session_id,
        parent_session_id="parent_ses_001",
        source_name=source_name,
        source_type="agent",
        description=description,
        spawn_mechanism=spawn_mechanism,  # type: ignore[arg-type]
        depth=1,
    )


# ============================================================================
# Test Event Sequences
# ============================================================================


def text_stream_events(source_name: str = "assistant") -> list[SubAgentEvent]:
    """Simple text streaming event sequence.

    Events:
    1. Text start: "Hello"
    2. Text delta: " world"
    3. Text delta: "!"
    4. Stream complete

    Expected behavior:
    - Tool_box: Title updates only, no content sent
    - Inline: ToolCallProgress with raw_output for each delta
    """
    return [
        get_subagent_event(
            source_name=source_name,
            inner_event=get_text_start_event("Hello"),
        ),
        get_subagent_event(
            source_name=source_name,
            inner_event=get_text_delta_event(" world"),
        ),
        get_subagent_event(
            source_name=source_name,
            inner_event=get_text_delta_event("!"),
        ),
        get_subagent_event(
            source_name=source_name,
            inner_event=get_stream_complete_event(),
        ),
    ]


def thinking_stream_events(source_name: str = "researcher") -> list[SubAgentEvent]:
    """Thinking stream event sequence.

    Events:
    1. Thinking start: "Analyzing"
    2. Thinking delta: " the"
    3. Thinking delta: " problem"
    4. Stream complete

    Expected behavior:
    - Tool_box: Title updates with thinking summary, no content sent
    - Inline: ToolCallProgress with raw_output="Thinking: {delta}"
    """
    return [
        get_subagent_event(
            source_name=source_name,
            inner_event=get_thinking_start_event("Analyzing"),
        ),
        get_subagent_event(
            source_name=source_name,
            inner_event=get_thinking_delta_event(" the"),
        ),
        get_subagent_event(
            source_name=source_name,
            inner_event=get_thinking_delta_event(" problem"),
        ),
        get_subagent_event(
            source_name=source_name,
            inner_event=get_stream_complete_event(),
        ),
    ]


def tool_call_events(source_name: str = "coder") -> list[SubAgentEvent]:
    """Tool call event sequence.

    Events:
    1. Text start: "I'll search for files"
    2. Tool call start: "search" with args={"pattern": "*.py"}
    3. Tool result: "Found 3 files"
    4. Stream complete

    Expected behavior:
    - Tool_box: Title updates (initializing, calling tool, completed), no content
    - Inline: ToolCallStart for tool, ToolCallProgress for result
    """
    return [
        get_subagent_event(
            source_name=source_name,
            inner_event=get_text_start_event("I'll search for files"),
        ),
        get_subagent_event(
            source_name=source_name,
            inner_event=get_tool_call_start_event(
                tool_name="search",
                tool_call_id="call_001",
                args={"pattern": "*.py"},
            ),
        ),
        get_subagent_event(
            source_name=source_name,
            inner_event=get_tool_result_event(
                tool_name="search",
                tool_call_id="call_001",
                result="Found 3 files",
            ),
        ),
        get_subagent_event(
            source_name=source_name,
            inner_event=get_stream_complete_event(),
        ),
    ]


def mixed_events(source_name: str = "analyzer") -> list[SubAgentEvent]:
    """Mixed event sequence with text, thinking, and tool calls.

    Events:
    1. Thinking start: "Need to analyze"
    2. Text start: "Let me check"
    3. Tool call start: "grep" with args={"pattern": "error"}
    4. Tool result: "No errors found"
    5. Text delta: " - all good!"
    6. Stream complete

    Expected behavior:
    - Tool_box: Title tracks each event type, no content sent
    - Inline: Each event type yields ToolCallProgress/ToolCallStart with appropriate raw_output
    """
    return [
        get_subagent_event(
            source_name=source_name,
            inner_event=get_thinking_start_event("Need to analyze"),
        ),
        get_subagent_event(
            source_name=source_name,
            inner_event=get_text_start_event("Let me check"),
        ),
        get_subagent_event(
            source_name=source_name,
            inner_event=get_tool_call_start_event(
                tool_name="grep",
                tool_call_id="call_002",
                args={"pattern": "error"},
            ),
        ),
        get_subagent_event(
            source_name=source_name,
            inner_event=get_tool_result_event(
                tool_name="grep",
                tool_call_id="call_002",
                result="No errors found",
            ),
        ),
        get_subagent_event(
            source_name=source_name,
            inner_event=get_text_delta_event(" - all good!"),
        ),
        get_subagent_event(
            source_name=source_name,
            inner_event=get_stream_complete_event(),
        ),
    ]


def tool_call_error_events(source_name: str = "executor") -> list[SubAgentEvent]:
    """Tool call with error event sequence.

    Events:
    1. Text start: "Executing command"
    2. Tool call start: "bash" with args={"command": "make build"}
    3. Tool result: Error "Build failed: missing dependency"

    Expected behavior:
    - Tool_box: Title shows error, no content sent
    - Inline: ToolCallProgress with error status and error message
    """
    return [
        get_subagent_event(
            source_name=source_name,
            inner_event=get_text_start_event("Executing command"),
        ),
        get_subagent_event(
            source_name=source_name,
            inner_event=get_tool_call_start_event(
                tool_name="bash",
                tool_call_id="call_003",
                args={"command": "make build"},
            ),
        ),
        get_subagent_event(
            source_name=source_name,
            inner_event=get_tool_error_event(
                tool_name="bash",
                tool_call_id="call_003",
                error_message="Build failed: missing dependency",
            ),
        ),
        get_subagent_event(
            source_name=source_name,
            inner_event=get_stream_complete_event(),
        ),
    ]


def long_text_events(source_name: str = "writer") -> list[SubAgentEvent]:
    """Long text streaming event sequence.

    Events:
    1. Multiple text deltas forming a long message
    2. Stream complete

    This tests header emission on first event only.
    """
    text = "This is a long message that gets streamed in multiple chunks. "
    text += "Each chunk should be a separate delta event. "
    text += "The header should only be emitted once. "
    text += "Subsequent deltas should have no prefix repetition."

    return [
        get_subagent_event(
            source_name=source_name,
            inner_event=get_text_start_event(text[:50]),
        ),
        get_subagent_event(
            source_name=source_name,
            inner_event=get_text_delta_event(text[50:100]),
        ),
        get_subagent_event(
            source_name=source_name,
            inner_event=get_text_delta_event(text[100:150]),
        ),
        get_subagent_event(
            source_name=source_name,
            inner_event=get_text_delta_event(text[150:]),
        ),
        get_subagent_event(
            source_name=source_name,
            inner_event=get_stream_complete_event(),
        ),
    ]


def nested_subagent_events() -> list[SubAgentEvent]:
    """Nested subagent event sequence (parent and child agents).

    Events from "coordinator" (depth=1):
    1. Text: "Delegating to researcher"
    2. Stream complete

    Events from "researcher" (depth=2):
    3. Thinking: "Searching"
    4. Tool call: "search"
    5. Tool result: "Results found"
    6. Stream complete

    This tests depth-based indentation and title-based subagent distinction.
    """
    return [
        # Coordinator agent (depth=1)
        get_subagent_event(
            source_name="coordinator",
            inner_event=get_text_start_event("Delegating to researcher"),
            depth=1,
        ),
        get_subagent_event(
            source_name="coordinator",
            inner_event=get_stream_complete_event(),
            depth=1,
        ),
        # Researcher agent (depth=2)
        get_subagent_event(
            source_name="researcher",
            inner_event=get_thinking_start_event("Searching"),
            depth=2,
        ),
        get_subagent_event(
            source_name="researcher",
            inner_event=get_tool_call_start_event(
                tool_name="search",
                tool_call_id="call_nested_001",
                args={"query": "test"},
            ),
            depth=2,
        ),
        get_subagent_event(
            source_name="researcher",
            inner_event=get_tool_result_event(
                tool_name="search",
                tool_call_id="call_nested_001",
                result="Results found",
            ),
            depth=2,
        ),
        get_subagent_event(
            source_name="researcher",
            inner_event=get_stream_complete_event(),
            depth=2,
        ),
    ]


# ============================================================================
# Zed-Mode Event Sequences
# ============================================================================


def zed_full_lifecycle_events() -> list[object]:
    """Full zed-mode subagent lifecycle.

    Includes SpawnSessionStart as the first event to seed the converter's
    tool map, followed by SubAgentEvent text/thinking deltas and a final
    StreamCompleteEvent.

    Events:
    1. SpawnSessionStart for "researcher"
    2. SubAgentEvent(TextPart): "Hello"
    3. SubAgentEvent(TextPartDelta): " world"
    4. SubAgentEvent(ThinkingPart): "Analyzing"
    5. SubAgentEvent(ThinkingPartDelta): " the data"
    6. SubAgentEvent(StreamCompleteEvent)

    Expected converter output (zed mode):
    - ToolCallStart(status="pending", title="task", field_meta)
    - ToolCallProgress(status="pending", content, field_meta) x4
    - ToolCallProgress(status="completed", field_meta with message_end_index)
    """
    child_id = "sub_001"
    return [
        get_spawn_session_start_event(
            child_session_id=child_id,
            source_name="researcher",
            description="Research task",
        ),
        get_subagent_event(
            source_name="researcher",
            inner_event=get_text_start_event("Hello"),
            child_session_id=child_id,
        ),
        get_subagent_event(
            source_name="researcher",
            inner_event=get_text_delta_event(" world"),
            child_session_id=child_id,
        ),
        get_subagent_event(
            source_name="researcher",
            inner_event=get_thinking_start_event("Analyzing"),
            child_session_id=child_id,
        ),
        get_subagent_event(
            source_name="researcher",
            inner_event=get_thinking_delta_event(" the data"),
            child_session_id=child_id,
        ),
        get_subagent_event(
            source_name="researcher",
            inner_event=get_stream_complete_event(),
            child_session_id=child_id,
        ),
    ]


# ============================================================================
# Parameterized Test Data
# ============================================================================

TEST_EVENT_SEQUENCES: dict[str, Callable[..., list[SubAgentEvent]]] = {
    "text_stream": text_stream_events,
    "thinking_stream": thinking_stream_events,
    "tool_call": tool_call_events,
    "mixed_events": mixed_events,
    "tool_call_error": tool_call_error_events,
    "long_text": long_text_events,
    "nested_subagents": nested_subagent_events,
}

# Separate sequences for zed mode (include SpawnSessionStart which is not a SubAgentEvent)
ZED_EVENT_SEQUENCES: dict[str, Callable[[], list[object]]] = {
    "full_lifecycle": zed_full_lifecycle_events,
}


@pytest.fixture(params=TEST_EVENT_SEQUENCES.keys())
def subagent_event_sequence(request: pytest.FixtureRequest) -> tuple[str, list[SubAgentEvent]]:
    """Parametrized fixture providing all test event sequences."""
    sequence_name: str = request.param
    return sequence_name, TEST_EVENT_SEQUENCES[sequence_name]()
