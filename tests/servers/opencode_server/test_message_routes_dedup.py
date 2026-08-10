"""Tests for single-path user message display architecture.

Verifies that the EventProcessor correctly handles UserMessageInsertedEvent
with meta (OpenCodeUserMessageMeta) and without meta (text-only fallback).
The dedup set has been removed — there is only one publication path now.
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
    MessageWithParts,
)


pytestmark = pytest.mark.unit

if TYPE_CHECKING:
    from wolfharness_server.opencode_server.state import ServerState


def _make_ctx(server_state: ServerState) -> EventProcessorContext:
    """Create a minimal EventProcessorContext for testing."""
    assistant_msg = MessageWithParts.assistant(
        message_id="msg-assistant",
        session_id="test-session",
        time=MessageTime(created=0),
        agent_name="test-agent",
        model_id="test-model",
        parent_id="parent",
        provider_id="wolfharness",
        path=MessagePath(cwd="/tmp", root="/tmp"),
    )
    return EventProcessorContext(
        session_id="test-session",
        assistant_msg_id="msg-assistant",
        assistant_msg=assistant_msg,
        state=server_state,
        working_dir="/tmp",
    )


@pytest.mark.asyncio
async def test_event_processor_emits_with_meta(
    server_state: ServerState,
) -> None:
    """EventProcessor uses meta.parts to reconstruct user message.

    Given: A UserMessageInsertedEvent with OpenCodeUserMessageMeta containing
        serialized TextPart data.
    When: EventProcessor processes the event.
    Then: It emits MessageUpdatedEvent and PartUpdatedEvent for each part.
    """
    processor = EventProcessor()
    ctx = _make_ctx(server_state)
    session_id = "test-session"
    message_id = "user-msg-with-meta"

    parts_data = [
        {
            "type": "text",
            "id": "part-1",
            "message_id": message_id,
            "session_id": session_id,
            "text": "Hello from meta",
        }
    ]
    meta = OpenCodeUserMessageMeta(parts=parts_data)

    event = UserMessageInsertedEvent(
        session_id=session_id,
        message_id=message_id,
        content="Hello from meta",
        delivery="initial",
        source="internal",
        meta=meta,
    )

    events = [e async for e in processor.process(event, ctx)]

    # Should emit MessageUpdatedEvent + PartUpdatedEvent for each part
    assert len(events) >= 2


@pytest.mark.asyncio
async def test_event_processor_emits_without_meta(
    server_state: ServerState,
) -> None:
    """EventProcessor falls back to text-only content when meta is None.

    Given: A UserMessageInsertedEvent with meta=None.
    When: EventProcessor processes the event.
    Then: It creates a TextPart from the content string and emits events.
    """
    processor = EventProcessor()
    ctx = _make_ctx(server_state)
    session_id = "test-session"
    message_id = "user-msg-no-meta"

    event = UserMessageInsertedEvent(
        session_id=session_id,
        message_id=message_id,
        content="Text-only fallback",
        delivery="steer",
        source="internal",
        meta=None,
    )

    events = [e async for e in processor.process(event, ctx)]

    # Should emit MessageUpdatedEvent + PartUpdatedEvent for the text part
    assert len(events) >= 2
