"""Unit tests for ``ACPEventConverter`` handling ``UserMessageInsertedEvent``.

Tests v1-only path: ``UserMessageInsertedEvent`` → ``UserMessageChunk``
emission, meta-based content block reconstruction, and text-only fallback.

The dedup set has been removed — there is only one publication path now.
"""

from __future__ import annotations

from typing import Any

import pytest

from acp.schema import UserMessageChunk
from wolfharness.agents.events.events import UserMessageInsertedEvent
from wolfharness_server.acp_server.event_converter import (
    ACPEventConverter,
    ACPUserMessageMeta,
)


pytestmark = [pytest.mark.unit, pytest.mark.anyio]


async def _collect(converter: ACPEventConverter, event: Any) -> list[Any]:
    """Collect all notifications yielded by ``converter.convert(event)``."""
    results: list[Any] = []
    results.extend([update async for update in converter.convert(event)])
    return results


async def test_user_message_inserted_emits_user_message_chunk() -> None:
    """``UserMessageInsertedEvent`` with text content emits ``UserMessageChunk``.

    Given: A converter and a ``UserMessageInsertedEvent`` with string content.
    When: The event is converted.
    Then: Exactly one ``UserMessageChunk`` is yielded with the event's text.
    """
    converter = ACPEventConverter()
    event = UserMessageInsertedEvent(
        session_id="s1",
        message_id="msg_001",
        content="Hello from steer!",
        delivery="steer",
        source="protocol",
    )

    results = await _collect(converter, event)

    assert len(results) == 1
    chunk = results[0]
    assert isinstance(chunk, UserMessageChunk)
    assert chunk.message_id == "msg_001"
    assert chunk.content.text == "Hello from steer!"


async def test_user_message_inserted_with_meta_reconstructs_blocks() -> None:
    """``ACPUserMessageMeta`` content_blocks are used to reconstruct the message.

    Given: A converter and a ``UserMessageInsertedEvent`` with
        ``ACPUserMessageMeta`` containing serialized TextContentBlock data.
    When: The event is converted.
    Then: ``UserMessageChunk`` notifications are yielded for each text block.
    """
    converter = ACPEventConverter()
    content_blocks = [
        {"type": "text", "text": "Reconstructed from meta"},
    ]
    meta = ACPUserMessageMeta(content_blocks=content_blocks)

    event = UserMessageInsertedEvent(
        session_id="s1",
        message_id="msg_meta",
        content="Reconstructed from meta",
        delivery="initial",
        source="protocol",
        meta=meta,
    )

    results = await _collect(converter, event)

    assert len(results) == 1
    chunk = results[0]
    assert isinstance(chunk, UserMessageChunk)
    assert chunk.content.text == "Reconstructed from meta"


async def test_user_message_inserted_multimodal_content() -> None:
    """Multi-modal content (list with text dicts) converts to multiple chunks.

    Given: A ``UserMessageInsertedEvent`` with ``content`` as a list of
        dicts with ``"type": "text"``.
    When: The event is converted.
    Then: One ``UserMessageChunk`` per text block is yielded.
    """
    converter = ACPEventConverter()
    event = UserMessageInsertedEvent(
        session_id="s1",
        message_id="msg_multi",
        content=[
            {"type": "text", "text": "First part"},
            {"type": "text", "text": "Second part"},
        ],
        delivery="steer",
        source="protocol",
    )

    results = await _collect(converter, event)

    assert len(results) == 2
    assert all(isinstance(r, UserMessageChunk) for r in results)
    assert results[0].content.text == "First part"
    assert results[1].content.text == "Second part"
    assert all(r.message_id == "msg_multi" for r in results)


async def test_user_message_inserted_skips_non_text_items_in_list() -> None:
    """Non-text items in a multi-modal list are silently skipped (v1).

    Given: A ``UserMessageInsertedEvent`` with content containing an image
        dict and a text dict.
    When: The event is converted.
    Then: Only the text block yields a ``UserMessageChunk``.
    """
    converter = ACPEventConverter()
    event = UserMessageInsertedEvent(
        session_id="s1",
        message_id="msg_mixed",
        content=[
            {"type": "image", "data": "base64..."},
            {"type": "text", "text": "Only text matters"},
        ],
        delivery="steer",
        source="protocol",
    )

    results = await _collect(converter, event)

    assert len(results) == 1
    assert isinstance(results[0], UserMessageChunk)
    assert results[0].content.text == "Only text matters"


async def test_user_message_inserted_empty_string_content_yields_nothing() -> None:
    """Empty string content yields no chunks.

    Given: A ``UserMessageInsertedEvent`` with ``content=""``.
    When: The event is converted.
    Then: No notifications are yielded.
    """
    converter = ACPEventConverter()
    event = UserMessageInsertedEvent(
        session_id="s1",
        message_id="msg_empty",
        content="",
        delivery="steer",
        source="protocol",
    )

    results = await _collect(converter, event)

    assert results == []
