"""Tests for the EventProcessor in OpenCode server.

Tests text handling and tool processing.
SubAgentEvent handling was moved to session_pool_integration in commit 4be3dd70b.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai.messages import (
    PartDeltaEvent as PydanticPartDeltaEvent,
    PartStartEvent,
    TextPart as PydanticTextPart,
    TextPartDelta,
)
import pytest

from wolfharness_server.opencode_server.event_processor import EventProcessor
from wolfharness_server.opencode_server.event_processor_context import (
    EventProcessorContext,
)
from wolfharness_server.opencode_server.models import (
    MessagePath,
    MessageTime,
    MessageWithParts,
    PartDeltaEvent,
    PartUpdatedEvent,
    SessionStatusEvent,
    TextPart,
)


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from wolfharness_server.opencode_server.state import ServerState


# =============================================================================
# Text Handling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_process_text_start_creates_text_part(server_state: ServerState) -> None:
    """Test that PartStartEvent with PydanticTextPart creates a text part.

    Verifies:
    - EventProcessor yields PartUpdatedEvent
    - context.text_part is set
    - text is in assistant_msg.parts
    """
    # GIVEN: empty context with assistant message
    processor = EventProcessor()
    assistant_msg = MessageWithParts.assistant(
        message_id="msg-1",
        session_id="test-session",
        time=MessageTime(created=0),
        agent_name="test-agent",
        model_id="test-model",
        parent_id="parent-1",
        provider_id="wolfharness",
        path=MessagePath(cwd="/tmp", root="/tmp"),
    )
    ctx = EventProcessorContext(
        session_id="test-session",
        assistant_msg_id="msg-1",
        assistant_msg=assistant_msg,
        state=server_state,
        working_dir="/tmp",
    )

    # WHEN: PartStartEvent with PydanticTextPart received
    event = PartStartEvent(index=0, part=PydanticTextPart(content="Hello, world!"))
    events = [e async for e in processor.process(event, ctx)]

    # THEN: PartUpdatedEvent is yielded
    assert len(events) == 1
    assert isinstance(events[0], PartUpdatedEvent)

    # AND: context.text_part is set
    assert ctx.text_part is not None
    assert ctx.text_part.text == "Hello, world!"

    # AND: text is in assistant_msg.parts
    assert len(assistant_msg.parts) == 1
    first_part = assistant_msg.parts[0]
    assert isinstance(first_part, TextPart)
    assert first_part.text == "Hello, world!"

    # AND: response_text is accumulated
    assert ctx.response_text == "Hello, world!"


@pytest.mark.asyncio
async def test_process_text_delta_accumulates_text(server_state: ServerState) -> None:
    """Test that PartDeltaEvent accumulates text onto existing text part.

    Verifies:
    - context.response_text accumulates the delta
    - PartUpdatedEvent is yielded
    - text_part is updated with accumulated text
    """
    # GIVEN: text has been started
    processor = EventProcessor()
    assistant_msg = MessageWithParts.assistant(
        message_id="msg-1",
        session_id="test-session",
        time=MessageTime(created=0),
        agent_name="test-agent",
        model_id="test-model",
        parent_id="parent-1",
        provider_id="wolfharness",
        path=MessagePath(cwd="/tmp", root="/tmp"),
    )
    ctx = EventProcessorContext(
        session_id="test-session",
        assistant_msg_id="msg-1",
        assistant_msg=assistant_msg,
        state=server_state,
        working_dir="/tmp",
    )

    # Start with initial text
    start_event = PartStartEvent(index=0, part=PydanticTextPart(content="Hello, "))
    async for _ in processor.process(start_event, ctx):
        pass

    # WHEN: PartDeltaEvent with TextPartDelta received
    delta_event = PydanticPartDeltaEvent(index=0, delta=TextPartDelta(content_delta="world!"))
    events = [e async for e in processor.process(delta_event, ctx)]

    # THEN: PartDeltaEvent is yielded (not PartUpdatedEvent for deltas)
    assert len(events) == 1
    assert isinstance(events[0], PartDeltaEvent)

    # AND: context.response_text accumulated the delta
    assert ctx.response_text == "Hello, world!"

    # AND: text_part is updated with accumulated text
    assert ctx.text_part is not None
    assert ctx.text_part.text == "Hello, world!"

    # AND: assistant_msg.parts is updated
    assert len(assistant_msg.parts) == 1
    first_part = assistant_msg.parts[0]
    assert isinstance(first_part, TextPart)
    assert first_part.text == "Hello, world!"


@pytest.mark.asyncio
async def test_process_text_delta_without_start(server_state: ServerState) -> None:
    """Test that PartDeltaEvent without prior PartStartEvent creates text part.

    This tests the fallback behavior when delta arrives before start.
    """
    # GIVEN: no text part started yet
    processor = EventProcessor()
    assistant_msg = MessageWithParts.assistant(
        message_id="msg-1",
        session_id="test-session",
        time=MessageTime(created=0),
        agent_name="test-agent",
        model_id="test-model",
        parent_id="parent-1",
        provider_id="wolfharness",
        path=MessagePath(cwd="/tmp", root="/tmp"),
    )
    ctx = EventProcessorContext(
        session_id="test-session",
        assistant_msg_id="msg-1",
        assistant_msg=assistant_msg,
        state=server_state,
        working_dir="/tmp",
    )

    # WHEN: PartDeltaEvent without prior PartStartEvent
    delta_event = PydanticPartDeltaEvent(index=0, delta=TextPartDelta(content_delta="Some text"))
    events = [e async for e in processor.process(delta_event, ctx)]

    # THEN: PartUpdatedEvent is yielded
    assert len(events) == 1
    assert isinstance(events[0], PartUpdatedEvent)

    # AND: text_part is created with accumulated text
    assert ctx.text_part is not None
    assert ctx.text_part.text == "Some text"

    # AND: assistant_msg.parts contains the text part
    assert len(assistant_msg.parts) == 1
    first_part = assistant_msg.parts[0]
    assert isinstance(first_part, TextPart)
    assert first_part.text == "Some text"


# =============================================================================
# StreamCompleteEvent Cancellation Tests
# =============================================================================


def _make_stream_complete_context(server_state: ServerState) -> EventProcessorContext:
    """Create a minimal EventProcessorContext for StreamCompleteEvent tests."""
    assistant_msg = MessageWithParts.assistant(
        message_id="msg-1",
        session_id="test-session",
        time=MessageTime(created=0),
        agent_name="test-agent",
        model_id="test-model",
        parent_id="parent-1",
        provider_id="wolfharness",
        path=MessagePath(cwd="/tmp", root="/tmp"),
    )
    return EventProcessorContext(
        session_id="test-session",
        assistant_msg_id="msg-1",
        assistant_msg=assistant_msg,
        state=server_state,
        working_dir="/tmp",
    )


@pytest.mark.asyncio
async def test_stream_complete_emits_idle_status(server_state: ServerState) -> None:
    """Test that a completed StreamCompleteEvent emits SessionStatusEvent(idle)."""
    from wolfharness.agents.events import StreamCompleteEvent
    from wolfharness.messaging import ChatMessage

    processor = EventProcessor()
    ctx = _make_stream_complete_context(server_state)

    msg = ChatMessage[str](content="done", role="assistant")
    event = StreamCompleteEvent(message=msg, cancelled=False)

    events = [e async for e in processor.process(event, ctx)]

    status_events = [e for e in events if isinstance(e, SessionStatusEvent)]
    assert len(status_events) == 1
    assert status_events[0].properties.status.type == "idle"


@pytest.mark.asyncio
async def test_stream_complete_emits_cancelled_status(server_state: ServerState) -> None:
    """Test that a cancelled StreamCompleteEvent emits SessionStatusEvent(cancelled)."""
    from wolfharness.agents.events import StreamCompleteEvent
    from wolfharness.messaging import ChatMessage

    processor = EventProcessor()
    ctx = _make_stream_complete_context(server_state)

    msg = ChatMessage[str](content="partial", role="assistant")
    event = StreamCompleteEvent(message=msg, cancelled=True)

    events = [e async for e in processor.process(event, ctx)]

    status_events = [e for e in events if isinstance(e, SessionStatusEvent)]
    assert len(status_events) == 1
    assert status_events[0].properties.status.type == "cancelled"


# =============================================================================
# McpToolsChangedEvent Tests
# =============================================================================


def test_create_mcp_tools_changed_event() -> None:
    """EventProcessor.create_mcp_tools_changed_event creates correct event."""
    from wolfharness_server.opencode_server.models.events import McpToolsChangedEvent

    processor = EventProcessor()
    event = processor.create_mcp_tools_changed_event(server="my_mcp_server")

    assert isinstance(event, McpToolsChangedEvent)
    assert event.type == "mcp.tools.changed"
    assert event.properties.server == "my_mcp_server"


@pytest.mark.anyio
async def test_mcp_tools_changed_event_from_change_event() -> None:
    """Full wiring: ChangeEvent(kind='tools_changed') → McpToolsChangedEvent.

    Simulates the flow:
    1. McpServerCap.on_change() yields ChangeEvent(kind="tools_changed")
    2. EventProcessor.create_mcp_tools_changed_event() converts to McpToolsChangedEvent
    """
    from wolfharness.capabilities.change_event import ChangeEvent
    from wolfharness_server.opencode_server.models.events import McpToolsChangedEvent

    processor = EventProcessor()

    # Simulate a ChangeEvent from McpServerCap._on_tools_changed()
    change_event = ChangeEvent(
        capability_name="my_mcp_server",
        kind="tools_changed",
        source_uri="mcp://my_mcp_server",
    )

    # The server's _watch_mcp_tool_changes task would do this conversion
    oc_event = processor.create_mcp_tools_changed_event(
        server=change_event.capability_name,
    )

    assert isinstance(oc_event, McpToolsChangedEvent)
    assert oc_event.properties.server == "my_mcp_server"
    assert oc_event.type == "mcp.tools.changed"
