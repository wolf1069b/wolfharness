"""L2 integration tests for UserMessageInsertedEvent handling in EventProcessor.

These tests use REAL ``EventProcessor`` and ``EventProcessorContext`` instances
(no mocking of ``process()``) to verify the full event → OpenCode SSE event
conversion pipeline. They exercise the real ``_process_user_message_inserted``
code path including meta-based part reconstruction and text-only fallback.

See ``test_event_processor_user_message.py`` for the L1 unit tests.
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


pytestmark = [pytest.mark.integration, pytest.mark.anyio]


if TYPE_CHECKING:
    from wolfharness_server.opencode_server.state import ServerState


def _make_ctx(server_state: ServerState) -> EventProcessorContext:
    """Create a minimal EventProcessorContext for integration testing."""
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
# Test: Real EventProcessor creates UserMessage from UserMessageInsertedEvent
# =============================================================================


@pytest.mark.asyncio
async def test_event_processor_creates_user_message(
    server_state: ServerState,
) -> None:
    """Real EventProcessor converts UserMessageInsertedEvent to SSE events.

    Given: A real ``EventProcessor`` and a ``UserMessageInsertedEvent``
        with string content.
    When: ``processor.process(event, ctx)`` is called.
    Then: Yields ``MessageUpdatedEvent`` with a ``UserMessage`` and a
        ``PartUpdatedEvent`` with a ``TextPart``.
    """
    processor = EventProcessor()
    ctx = _make_ctx(server_state)

    event = UserMessageInsertedEvent(
        session_id="s1",
        message_id="msg_1",
        content="hello world",
        delivery="steer",
        source="background_task",
    )
    events = [e async for e in processor.process(event, ctx)]

    msg_events = [e for e in events if isinstance(e, MessageUpdatedEvent)]
    assert len(msg_events) == 1
    msg_info = msg_events[0].properties.info
    assert msg_info.id == "msg_1"
    assert msg_info.role == "user"

    part_events = [e for e in events if isinstance(e, PartUpdatedEvent)]
    assert len(part_events) == 1
    part = part_events[0].properties.part
    assert isinstance(part, TextPart)
    assert part.text == "hello world"
    assert part.message_id == "msg_1"

    messages = server_state.messages.get("test-session", [])
    assert len(messages) == 1
    assert messages[0].info.id == "msg_1"


# =============================================================================
# Test: Real EventProcessor handles meta-based part reconstruction
# =============================================================================


@pytest.mark.asyncio
async def test_event_processor_with_meta_reconstructs_parts(
    server_state: ServerState,
) -> None:
    """Real EventProcessor uses meta.parts to reconstruct user message parts.

    Given: A real ``EventProcessor`` and a ``UserMessageInsertedEvent`` with
        ``OpenCodeUserMessageMeta`` containing serialized TextPart data.
    When: ``processor.process(event, ctx)`` is called.
    Then: Yields ``MessageUpdatedEvent`` and ``PartUpdatedEvent`` with parts
        deserialized from the meta data.
    """
    processor = EventProcessor()
    ctx = _make_ctx(server_state)

    parts_data = [
        {
            "type": "text",
            "id": "part-meta-int-1",
            "message_id": "msg_meta",
            "session_id": "test-session",
            "text": "Reconstructed from meta",
        }
    ]
    meta = OpenCodeUserMessageMeta(parts=parts_data)

    event = UserMessageInsertedEvent(
        session_id="s1",
        message_id="msg_meta",
        content="Reconstructed from meta",
        delivery="initial",
        source="internal",
        meta=meta,
    )
    events = [e async for e in processor.process(event, ctx)]

    msg_events = [e for e in events if isinstance(e, MessageUpdatedEvent)]
    assert len(msg_events) == 1
    assert msg_events[0].properties.info.id == "msg_meta"

    part_events = [e for e in events if isinstance(e, PartUpdatedEvent)]
    assert len(part_events) == 1
    part = part_events[0].properties.part
    assert isinstance(part, TextPart)
    assert part.text == "Reconstructed from meta"


# =============================================================================
# Test: Real EventProcessor handles multimodal content list
# =============================================================================


@pytest.mark.asyncio
async def test_event_processor_content_list_multimodal(
    server_state: ServerState,
) -> None:
    """Real EventProcessor handles list content with text and non-text items.

    Given: A real ``EventProcessor`` and a ``UserMessageInsertedEvent`` with
        content as a list containing a text string and an image dict.
    When: ``processor.process(event, ctx)`` is called.
    Then: ``MessageUpdatedEvent`` is yielded, and a ``PartUpdatedEvent`` with
        ``TextPart`` is yielded for the text item. The image dict (no "text"
        key) is silently skipped by the current implementation.
    """
    processor = EventProcessor()
    ctx = _make_ctx(server_state)

    event = UserMessageInsertedEvent(
        session_id="s1",
        message_id="msg_multi",
        content=["text part", {"type": "image", "url": "http://example.com/img.png"}],
        delivery="steer",
        source="background_task",
    )
    events = [e async for e in processor.process(event, ctx)]

    msg_events = [e for e in events if isinstance(e, MessageUpdatedEvent)]
    assert len(msg_events) == 1
    assert msg_events[0].properties.info.id == "msg_multi"
    assert msg_events[0].properties.info.role == "user"

    part_events = [e for e in events if isinstance(e, PartUpdatedEvent)]
    assert len(part_events) == 1
    part = part_events[0].properties.part
    assert isinstance(part, TextPart)
    assert part.text == "text part"

    messages = server_state.messages.get("test-session", [])
    assert len(messages) == 1
    assert len(messages[0].parts) == 1
