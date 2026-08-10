"""Tests for P1: PartUpdatedEvent always emitted regardless of source.

P1 removed the ``if source != "protocol":`` guard so that
PartUpdatedEvent is always yielded for each part in a
UserMessageInsertedEvent, regardless of whether the source is
"protocol", "background_task", or "internal".

The TUI has no optimistic mechanism — it relies entirely on SSE events
for parts. Without PartUpdatedEvent, user messages appear empty after
the initial sync() (which only runs once per session).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from wolfharness.agents.events import UserMessageInsertedEvent
from wolfharness_server.opencode_server.event_processor import EventProcessor
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


@pytest.mark.asyncio
async def test_protocol_message_emits_part_updated_event(
    server_state: ServerState,
) -> None:
    """P1: source="protocol" messages must emit PartUpdatedEvent.

    Given: A UserMessageInsertedEvent with source="protocol" and text content.
    When: Processed through _process_user_message_inserted().
    Then: PartUpdatedEvent IS yielded for each text part.
    """
    processor = EventProcessor()
    ctx = _make_ctx(server_state)

    event = UserMessageInsertedEvent(
        session_id="test-session",
        message_id="user-msg-protocol",
        content="Hello from protocol",
        delivery="initial",
        source="protocol",
        timestamp=1700000000.0,
    )
    events = [e async for e in processor.process(event, ctx)]

    # Should have: 1 MessageUpdatedEvent + 1 PartUpdatedEvent
    assert len(events) == 2
    assert isinstance(events[0], MessageUpdatedEvent)
    assert isinstance(events[1], PartUpdatedEvent)

    part = events[1].properties.part
    assert isinstance(part, TextPart)
    assert part.text == "Hello from protocol"
    assert part.message_id == "user-msg-protocol"


@pytest.mark.asyncio
async def test_non_protocol_message_emits_part_updated_event(
    server_state: ServerState,
) -> None:
    """P1: source="internal" messages must also emit PartUpdatedEvent (existing behavior).

    Given: A UserMessageInsertedEvent with source="internal" and text content.
    When: Processed through _process_user_message_inserted().
    Then: PartUpdatedEvent IS yielded for each text part.
    """
    processor = EventProcessor()
    ctx = _make_ctx(server_state)

    event = UserMessageInsertedEvent(
        session_id="test-session",
        message_id="user-msg-internal",
        content="Hello from internal",
        delivery="steer",
        source="internal",
        timestamp=1700000001.0,
    )
    events = [e async for e in processor.process(event, ctx)]

    # Should have: 1 MessageUpdatedEvent + 1 PartUpdatedEvent
    assert len(events) == 2
    assert isinstance(events[0], MessageUpdatedEvent)
    assert isinstance(events[1], PartUpdatedEvent)

    part = events[1].properties.part
    assert isinstance(part, TextPart)
    assert part.text == "Hello from internal"
    assert part.message_id == "user-msg-internal"


@pytest.mark.asyncio
async def test_protocol_multimodal_emits_all_parts(
    server_state: ServerState,
) -> None:
    """P1: source="protocol" with multimodal content emits PartUpdatedEvent for each part.

    Given: A UserMessageInsertedEvent with source="protocol" and list content.
    When: Processed through _process_user_message_inserted().
    Then: PartUpdatedEvent IS yielded for each text part in the list.
    """
    processor = EventProcessor()
    ctx = _make_ctx(server_state)

    event = UserMessageInsertedEvent(
        session_id="test-session",
        message_id="user-msg-multi",
        content=["First", "Second", "Third"],
        delivery="initial",
        source="protocol",
        timestamp=1700000002.0,
    )
    events = [e async for e in processor.process(event, ctx)]

    # Should have: 1 MessageUpdatedEvent + 3 PartUpdatedEvents
    assert len(events) == 4
    assert isinstance(events[0], MessageUpdatedEvent)

    part_events = [e for e in events[1:] if isinstance(e, PartUpdatedEvent)]
    assert len(part_events) == 3
    texts = [e.properties.part.text for e in part_events]
    assert texts == ["First", "Second", "Third"]
