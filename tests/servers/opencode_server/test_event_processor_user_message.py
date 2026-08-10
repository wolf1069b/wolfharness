"""Tests for UserMessageInsertedEvent handling in EventProcessor.

Tests that steer/followup user messages are correctly converted to
OpenCode UserMessage objects and broadcast as SSE events.

In the single-path display architecture, EventProcessor uses the ``meta``
field (OpenCodeUserMessageMeta) to reconstruct rich parts, falling back
to text-only ``content`` when meta is None.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from wolfharness.agents.events import UserMessageInsertedEvent
from wolfharness_server.opencode_server.event_processor import (
    EventProcessor,
    OpenCodeUserMessageMeta,
)
from wolfharness_server.opencode_server.event_processor_context import (
    EventProcessorContext,
)
from wolfharness_server.opencode_server.models import (
    MessagePath,
    MessageTime,
    MessageUpdatedEvent,
    MessageWithParts,
    PartUpdatedEvent,
    TextPart,
)


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from wolfharness_server.opencode_server.state import ServerState


def _make_ctx(server_state: ServerState) -> EventProcessorContext:
    """Create a minimal EventProcessorContext for testing."""
    assistant_msg = MessageWithParts.assistant(
        message_id="msg-assistant-1",
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
        assistant_msg_id="msg-assistant-1",
        assistant_msg=assistant_msg,
        state=server_state,
        working_dir="/tmp",
    )


# =============================================================================
# UserMessageInsertedEvent Tests
# =============================================================================


@pytest.mark.asyncio
async def test_user_message_inserted_creates_user_message(
    server_state: ServerState,
) -> None:
    """Test that UserMessageInsertedEvent creates a UserMessage and SSE events.

    Verifies:
    - MessageUpdatedEvent is yielded with UserMessage info
    - PartUpdatedEvent is yielded for the text part
    - Message is appended to session state
    """
    processor = EventProcessor()
    ctx = _make_ctx(server_state)

    event = UserMessageInsertedEvent(
        session_id="test-session",
        message_id="user-msg-1",
        content="Steer this conversation",
        delivery="steer",
        source="accepted",
        timestamp=1700000000.0,
    )
    events = [e async for e in processor.process(event, ctx)]

    assert len(events) == 2
    assert isinstance(events[0], MessageUpdatedEvent)
    assert isinstance(events[1], PartUpdatedEvent)

    msg_info = events[0].properties.info
    assert msg_info.id == "user-msg-1"
    assert msg_info.session_id == "test-session"
    assert msg_info.role == "user"
    assert msg_info.time.created == 1700000000000

    part = events[1].properties.part
    assert isinstance(part, TextPart)
    assert part.text == "Steer this conversation"
    assert part.message_id == "user-msg-1"

    messages = server_state.messages.get("test-session", [])
    assert len(messages) == 1
    assert messages[0].info.id == "user-msg-1"


@pytest.mark.asyncio
async def test_user_message_inserted_with_meta_reconstructs_parts(
    server_state: ServerState,
) -> None:
    """Test that meta.parts are used to reconstruct the user message.

    Verifies:
    - When meta is OpenCodeUserMessageMeta, parts are deserialized from dicts
    - Each part yields a PartUpdatedEvent
    - Part text matches the serialized data
    """
    processor = EventProcessor()
    ctx = _make_ctx(server_state)

    parts_data = [
        {
            "type": "text",
            "id": "part-meta-1",
            "message_id": "user-msg-meta",
            "session_id": "test-session",
            "text": "Reconstructed from meta",
        }
    ]
    meta = OpenCodeUserMessageMeta(parts=parts_data)

    event = UserMessageInsertedEvent(
        session_id="test-session",
        message_id="user-msg-meta",
        content="Reconstructed from meta",
        delivery="initial",
        source="accepted",
        timestamp=1700000000.0,
        meta=meta,
    )
    events = [e async for e in processor.process(event, ctx)]

    assert len(events) == 2
    assert isinstance(events[0], MessageUpdatedEvent)
    assert isinstance(events[1], PartUpdatedEvent)

    part = events[1].properties.part
    assert isinstance(part, TextPart)
    assert part.text == "Reconstructed from meta"
    assert part.message_id == "user-msg-meta"


@pytest.mark.asyncio
async def test_user_message_inserted_multimodal_content(
    server_state: ServerState,
) -> None:
    """Test that list content with text blocks creates multiple text parts.

    Verifies:
    - Each text item in the list creates a separate TextPart
    - All parts are yielded as PartUpdatedEvent
    - Dict items with "text" key are also converted
    """
    processor = EventProcessor()
    ctx = _make_ctx(server_state)

    event = UserMessageInsertedEvent(
        session_id="test-session",
        message_id="user-msg-multi",
        content=["First part", {"text": "Second part"}, "Third part"],
        delivery="followup",
        source="accepted",
        timestamp=1700000001.0,
    )
    events = [e async for e in processor.process(event, ctx)]

    assert len(events) == 4
    assert isinstance(events[0], MessageUpdatedEvent)

    text_parts = [e for e in events[1:] if isinstance(e, PartUpdatedEvent)]
    assert len(text_parts) == 3
    assert text_parts[0].properties.part.text == "First part"
    assert text_parts[1].properties.part.text == "Second part"
    assert text_parts[2].properties.part.text == "Third part"


@pytest.mark.asyncio
async def test_user_message_inserted_empty_string_no_parts(
    server_state: ServerState,
) -> None:
    """Test that empty string content creates no text parts.

    Verifies:
    - MessageUpdatedEvent is still yielded
    - No PartUpdatedEvent for empty content
    """
    processor = EventProcessor()
    ctx = _make_ctx(server_state)

    event = UserMessageInsertedEvent(
        session_id="test-session",
        message_id="user-msg-empty",
        content="",
        delivery="initial",
        timestamp=1700000002.0,
    )
    events = [e async for e in processor.process(event, ctx)]

    assert len(events) == 1
    assert isinstance(events[0], MessageUpdatedEvent)


# =============================================================================
# Tests for source-based handling (5.15, 5.16)
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_event_processor_skips_processed_source(
    server_state: ServerState,
) -> None:
    """EventProcessor returns empty for source="processed" event.

    source="processed" means the event is from the model drain time
    (EnqueuedMessagesEvent). It should NOT create a UserMessage or
    yield any SSE events — display was already handled by the
    source="accepted" event at routing time.
    """
    processor = EventProcessor()
    ctx = _make_ctx(server_state)

    event = UserMessageInsertedEvent(
        session_id="test-session",
        message_id="user-msg-processed",
        content="processed source text",
        delivery="steer",
        source="processed",
        timestamp=1700000003.0,
    )
    events = [e async for e in processor.process(event, ctx)]

    assert len(events) == 0, (
        f"Expected 0 events for source='processed', got {len(events)}: {events}"
    )

    # No message should be appended to session state
    messages = server_state.messages.get("test-session", [])
    assert len(messages) == 0, f"Expected no messages in session state, got {len(messages)}"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_event_processor_processes_accepted_source(
    server_state: ServerState,
) -> None:
    """EventProcessor creates UserMessage for source="accepted" event.

    source="accepted" means the event is from the accept time (routing).
    It SHOULD create a UserMessage and yield SSE events for display.
    """
    processor = EventProcessor()
    ctx = _make_ctx(server_state)

    event = UserMessageInsertedEvent(
        session_id="test-session",
        message_id="user-msg-accepted",
        content="accepted source text",
        delivery="steer",
        source="accepted",
        timestamp=1700000004.0,
    )
    events = [e async for e in processor.process(event, ctx)]

    assert len(events) == 2
    assert isinstance(events[0], MessageUpdatedEvent)
    assert isinstance(events[1], PartUpdatedEvent)

    msg_info = events[0].properties.info
    assert msg_info.id == "user-msg-accepted"
    assert msg_info.role == "user"

    messages = server_state.messages.get("test-session", [])
    assert len(messages) == 1
    assert messages[0].info.id == "user-msg-accepted"
