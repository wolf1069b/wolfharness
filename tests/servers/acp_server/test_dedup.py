"""Unit tests for ACPEventConverter UserMessageInsertedEvent dedup.

Verifies that:
- Two UserMessageInsertedEvent with the same message_id → only the first
  produces a UserMessageChunk.
- Two UserMessageInsertedEvent with different message_ids → both produce
  UserMessageChunk.
- The dedup set (_displayed_message_ids) persists across reset() calls.
"""

from __future__ import annotations

import time

import pytest

from acp.schema import UserMessageChunk
from wolfharness.agents.events.events import UserMessageInsertedEvent
from wolfharness_server.acp_server.event_converter import ACPEventConverter


pytestmark = pytest.mark.unit


def _make_event(message_id: str, content: str = "hello") -> UserMessageInsertedEvent:
    """Create a UserMessageInsertedEvent for testing."""
    return UserMessageInsertedEvent(
        session_id="test-session",
        message_id=message_id,
        content=content,
        delivery="steer",
        source="accepted",
        timestamp=time.time(),
    )


async def test_dedup_same_message_id_skips_second() -> None:
    """Two events with the same message_id → only first produces UserMessageChunk."""
    converter = ACPEventConverter()
    event1 = _make_event("msg-1", "first message")
    event2 = _make_event("msg-1", "first message")

    updates1 = [u async for u in converter.convert(event1)]
    updates2 = [u async for u in converter.convert(event2)]

    assert len(updates1) == 1
    assert isinstance(updates1[0], UserMessageChunk)
    assert updates1[0].content.text == "first message"
    assert len(updates2) == 0, "Second event with same message_id should be skipped"


async def test_dedup_different_message_ids_both_emitted() -> None:
    """Two events with different message_ids → both produce UserMessageChunk."""
    converter = ACPEventConverter()
    event1 = _make_event("msg-1", "first message")
    event2 = _make_event("msg-2", "second message")

    updates1 = [u async for u in converter.convert(event1)]
    updates2 = [u async for u in converter.convert(event2)]

    assert len(updates1) == 1
    assert updates1[0].content.text == "first message"
    assert len(updates2) == 1
    assert updates2[0].content.text == "second message"


async def test_dedup_persists_across_reset() -> None:
    """_displayed_message_ids is NOT cleared in reset() — persists for session."""
    converter = ACPEventConverter()
    event1 = _make_event("msg-persistent", "first")

    updates1 = [u async for u in converter.convert(event1)]
    assert len(updates1) == 1

    converter.reset()

    event2 = _make_event("msg-persistent", "first")
    updates2 = [u async for u in converter.convert(event2)]
    assert len(updates2) == 0, "Dedup should persist across reset()"


async def test_dedup_empty_message_id_not_tracked() -> None:
    """Events with empty message_id are NOT deduped (always emitted)."""
    converter = ACPEventConverter()
    event1 = _make_event("", "no-id-first")
    event2 = _make_event("", "no-id-second")

    updates1 = [u async for u in converter.convert(event1)]
    updates2 = [u async for u in converter.convert(event2)]

    assert len(updates1) == 1
    assert len(updates2) == 1


@pytest.mark.unit
async def test_acp_converter_skips_processed_source() -> None:
    """ACPEventConverter returns empty for source="processed" event.

    source="processed" means the event is from the model drain time
    (EnqueuedMessagesEvent). The ACP converter should NOT produce a
    UserMessageChunk for it — display was already handled by the
    source="accepted" event at routing time.
    """
    converter = ACPEventConverter()
    event = UserMessageInsertedEvent(
        session_id="test-session",
        message_id="msg-processed",
        content="processed source text",
        delivery="steer",
        source="processed",
        timestamp=time.time(),
    )

    updates = [u async for u in converter.convert(event)]

    assert len(updates) == 0, (
        f"Expected 0 updates for source='processed', got {len(updates)}: {updates}"
    )
