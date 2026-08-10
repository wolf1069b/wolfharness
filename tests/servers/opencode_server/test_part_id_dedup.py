"""L2 test: verify part ID mismatch between DB and SSE doesn't cause duplication.

Root cause: ``chat_message_to_opencode`` generates NEW part IDs when
reconstructing messages from DB. The EventProcessor's ``_deserialize_part``
preserves the ORIGINAL part IDs from ``meta.parts``. When the TUI receives
both (from ``sync.session.sync()`` and SSE ``message.part.updated``),
the binary search by part ID fails to deduplicate, causing the text to
appear twice.

P1 fix: ``PartUpdatedEvent`` is now ALWAYS yielded regardless of source.
The TUI has no optimistic mechanism — it relies entirely on SSE events
for parts. Without ``PartUpdatedEvent``, user messages appear empty after
the initial sync() (which only runs once per session). Part IDs from
``_deserialize_part()`` preserve the original IDs from meta, so they
match DB-stored parts — no duplicate risk for new messages.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wolfharness_server.opencode_server.event_processor import (
    EventProcessor,
    OpenCodeUserMessageMeta,
)
from wolfharness_server.opencode_server.models.message import (
    MessagePath,
    MessageTime,
    MessageWithParts,
)


def _make_ctx(session_id: str = "test-session") -> Any:
    """Create a minimal EventProcessorContext for testing."""
    from wolfharness_server.opencode_server.event_processor_context import (
        EventProcessorContext,
    )

    assistant_msg = MessageWithParts.assistant(
        message_id="msg_assistant_001",
        session_id=session_id,
        time=MessageTime(created=0),
        agent_name="test-agent",
        model_id="test-model",
        parent_id="parent-001",
        provider_id="wolfharness",
        path=MessagePath(cwd="/tmp", root="/tmp"),
    )
    ctx = EventProcessorContext(
        session_id=session_id,
        assistant_msg_id="msg_assistant_001",
        assistant_msg=assistant_msg,
        state=MagicMock(),
        working_dir="/tmp",
    )
    ctx.state.messages = {}
    return ctx


@pytest.mark.unit
async def test_accepted_source_does_not_yield_part_updated_events() -> None:
    """source=accepted now YIELDS PartUpdatedEvent (P1 fix).

    P1: PartUpdatedEvent is always yielded regardless of source because
    the TUI has no optimistic mechanism. Parts come from SSE events.
    """
    processor = EventProcessor()
    ctx = _make_ctx("test-session")
    ctx.state = MagicMock()
    ctx.state.messages = {}

    # Mock append_message_to_session to avoid actual state mutation
    with patch(
        "wolfharness_server.opencode_server.opencode_message_bridge.append_message_to_session",
        new_callable=AsyncMock,
    ):
        meta = OpenCodeUserMessageMeta(
            parts=[
                {
                    "type": "text",
                    "id": "part_original_001",
                    "text": "hello world",
                    "message_id": "",
                    "session_id": "",
                }
            ],
        )
        events = []
        async for e in processor._process_user_message_inserted(
            ctx,
            message_id="msg_test_001",
            content="hello world",
            timestamp=1000.0,
            meta=meta,
            source="accepted",
        ):
            events.append(e)  # noqa: PERF401

    # P1: Should yield 2 events: MessageUpdatedEvent + PartUpdatedEvent
    assert len(events) == 2, f"Expected 2 events for accepted source, got {len(events)}: {events}"
    assert events[0].type == "message.updated"
    assert events[1].type == "message.part.updated"


@pytest.mark.unit
async def test_accepted_source_yields_part_updated_events() -> None:
    """source="accepted" SHOULD yield PartUpdatedEvent.

    Accepted messages have no sync() to load parts from DB, so parts
    must come via SSE.
    """
    processor = EventProcessor()
    ctx = _make_ctx("test-session")
    ctx.state = MagicMock()
    ctx.state.messages = {}

    with patch(
        "wolfharness_server.opencode_server.opencode_message_bridge.append_message_to_session",
        new_callable=AsyncMock,
    ):
        meta = OpenCodeUserMessageMeta(
            parts=[
                {
                    "type": "text",
                    "id": "part_accepted_001",
                    "text": "background task result",
                    "message_id": "",
                    "session_id": "",
                }
            ],
        )
        events = []
        async for e in processor._process_user_message_inserted(
            ctx,
            message_id="msg_test_002",
            content="background task result",
            timestamp=1000.0,
            meta=meta,
            source="accepted",
        ):
            events.append(e)  # noqa: PERF401

    # Should yield 2 events: MessageUpdatedEvent + PartUpdatedEvent
    assert len(events) == 2, f"Expected 2 events for accepted source, got {len(events)}"
    assert events[0].type == "message.updated"
    assert events[1].type == "message.part.updated"


@pytest.mark.unit
async def test_accepted_source_part_ids_differ_from_db_reconstruction() -> None:
    """Verify the root cause: DB reconstruction creates different part IDs.

    This test demonstrates WHY sending PartUpdatedEvent for accepted
    sources causes duplication: ``chat_message_to_opencode`` generates
    new part IDs, different from the original parts in meta.
    """
    from wolfharness.messaging.messages import ChatMessage
    from wolfharness_server.opencode_server.converters import chat_message_to_opencode

    # Create a ChatMessage as stored in DB
    chat_msg = ChatMessage[str](
        message_id="msg_test_003",
        session_id="test-session",
        content="hello world",
        role="user",
        timestamp=MagicMock(),
    )

    # Convert back to OpenCode format (as GET /message would)
    db_msg = chat_message_to_opencode(
        chat_msg,
        session_id="test-session",
        agent_name="test-agent",
    )

    # DB-reconstructed part has a NEW ID (not the original)
    db_part_id = db_msg.parts[0].id
    original_part_id = "part_original_001"

    # The IDs are DIFFERENT — this is the root cause of duplication
    assert db_part_id != original_part_id, (
        "DB reconstruction should generate a new part ID, "
        "different from the original. If they match, the duplication "
        "bug would not occur."
    )


@pytest.mark.unit
async def test_accepted_source_no_part_updated_with_text_content() -> None:
    """source=accepted with text-only content (no meta) now yields PartUpdatedEvent (P1 fix)."""
    processor = EventProcessor()
    ctx = _make_ctx("test-session")
    ctx.state = MagicMock()
    ctx.state.messages = {}

    with patch(
        "wolfharness_server.opencode_server.opencode_message_bridge.append_message_to_session",
        new_callable=AsyncMock,
    ):
        events = []
        async for e in processor._process_user_message_inserted(
            ctx,
            message_id="msg_test_004",
            content="plain text message",
            timestamp=1000.0,
            meta=None,
            source="accepted",
        ):
            events.append(e)  # noqa: PERF401

    # P1: Should yield 2 events: MessageUpdatedEvent + PartUpdatedEvent
    assert len(events) == 2
    assert events[0].type == "message.updated"
    assert events[1].type == "message.part.updated"


@pytest.mark.unit
async def test_accepted_source_text_content_yields_part_updated() -> None:
    """source="accepted" with text-only content SHOULD yield PartUpdatedEvent."""
    processor = EventProcessor()
    ctx = _make_ctx("test-session")
    ctx.state = MagicMock()
    ctx.state.messages = {}

    with patch(
        "wolfharness_server.opencode_server.opencode_message_bridge.append_message_to_session",
        new_callable=AsyncMock,
    ):
        events = []
        async for e in processor._process_user_message_inserted(
            ctx,
            message_id="msg_test_005",
            content="accepted steer message",
            timestamp=1000.0,
            meta=None,
            source="accepted",
        ):
            events.append(e)  # noqa: PERF401

    # Should yield 2 events: MessageUpdatedEvent + PartUpdatedEvent
    assert len(events) == 2
    assert events[0].type == "message.updated"
    assert events[1].type == "message.part.updated"
