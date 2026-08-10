"""Unit tests for OpenCode EventProcessor UserMessageInsertedEvent handling.

Verifies that:
- source="accepted" events produce UserMessage + SSE (display)
- source="processed" events are skipped (no UserMessage creation)
- Two events with different message_ids both produce events
- Events with empty message_id are always emitted
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wolfharness.agents.events.events import UserMessageInsertedEvent
from wolfharness_server.opencode_server.event_processor import EventProcessor
from wolfharness_server.opencode_server.event_processor_context import (
    EventProcessorContext,
)
from wolfharness_server.opencode_server.models.message import (
    MessagePath,
    MessageTime,
    MessageWithParts,
)


pytestmark = pytest.mark.unit


def _make_ctx() -> EventProcessorContext:
    """Create a minimal EventProcessorContext for testing."""
    assistant_msg = MessageWithParts.assistant(
        message_id="msg_assistant",
        session_id="test-session",
        time=MessageTime(created=0),
        agent_name="agent",
        model_id="default",
        parent_id="",
        provider_id="wolfharness",
        path=MessagePath(cwd="/tmp", root="/tmp"),
    )
    state = MagicMock()
    return EventProcessorContext(
        session_id="test-session",
        assistant_msg_id="msg_assistant",
        assistant_msg=assistant_msg,
        state=state,
        working_dir="/tmp",
    )


_BRIDGE_PATH = (
    "wolfharness_server.opencode_server.opencode_message_bridge.append_message_to_session"
)


async def test_accepted_source_produces_events() -> None:
    """source="accepted" event produces UserMessage + SSE events."""
    processor = EventProcessor()
    ctx = _make_ctx()
    event = UserMessageInsertedEvent(
        session_id="test-session",
        message_id="msg-1",
        content="first message",
        delivery="steer",
        source="accepted",
        timestamp=time.time(),
    )

    with patch(_BRIDGE_PATH, new_callable=AsyncMock):
        events = [e async for e in processor.process(event, ctx)]

    assert len(events) > 0, "Accepted event should produce output"


async def test_processed_source_skipped() -> None:
    """source="processed" event is skipped — no UserMessage creation."""
    processor = EventProcessor()
    ctx = _make_ctx()
    event = UserMessageInsertedEvent(
        session_id="test-session",
        message_id="msg-1",
        content="first message",
        delivery="steer",
        source="processed",
        timestamp=time.time(),
    )

    with patch(_BRIDGE_PATH, new_callable=AsyncMock):
        events = [e async for e in processor.process(event, ctx)]

    assert len(events) == 0, "Processed event should be skipped (no UserMessage)"


async def test_dedup_different_message_ids_both_emitted() -> None:
    """Two accepted events with different message_ids → both produce events."""
    processor = EventProcessor()
    ctx = _make_ctx()

    with patch(_BRIDGE_PATH, new_callable=AsyncMock):
        event1 = UserMessageInsertedEvent(
            session_id="test-session",
            message_id="msg-1",
            content="first",
            delivery="steer",
            source="accepted",
            timestamp=time.time(),
        )
        event2 = UserMessageInsertedEvent(
            session_id="test-session",
            message_id="msg-2",
            content="second",
            delivery="steer",
            source="accepted",
            timestamp=time.time(),
        )
        events1 = [e async for e in processor.process(event1, ctx)]
        events2 = [e async for e in processor.process(event2, ctx)]

    assert len(events1) > 0, "First event should produce output"
    assert len(events2) > 0, "Second event with different ID should produce output"


async def test_dedup_empty_message_id_not_tracked() -> None:
    """Events with empty message_id are always emitted (no dedup)."""
    processor = EventProcessor()
    ctx = _make_ctx()

    with patch(_BRIDGE_PATH, new_callable=AsyncMock):
        event1 = UserMessageInsertedEvent(
            session_id="test-session",
            message_id="",
            content="no-id-first",
            delivery="steer",
            source="accepted",
            timestamp=time.time(),
        )
        event2 = UserMessageInsertedEvent(
            session_id="test-session",
            message_id="",
            content="no-id-second",
            delivery="steer",
            source="accepted",
            timestamp=time.time(),
        )
        events1 = [e async for e in processor.process(event1, ctx)]
        events2 = [e async for e in processor.process(event2, ctx)]

    assert len(events1) > 0
    assert len(events2) > 0, "Empty message_id should not be deduped"
