"""L2 integration tests for UserMessageInsertedEvent handling in ACPEventConverter.

These tests use a REAL ``ACPEventConverter`` instance (no mocking of
``convert()``) to verify the full event → ACP notification conversion
pipeline for ``UserMessageInsertedEvent``.

ACP v1 path: ``UserMessageInsertedEvent`` → ``UserMessageChunk`` with
``TextContentBlock``.

ACP v2 path (whole-message ``UserMessage`` upsert) is not yet implemented
in the schema or converter — Test 16 is skipped until v2 support lands.

See ``test_event_converter_user_message.py`` for the L1 unit tests.
"""

from __future__ import annotations

from typing import Any

import pytest

from acp.schema import TextContentBlock, UserMessageChunk
from wolfharness.agents.events.events import UserMessageInsertedEvent
from wolfharness_server.acp_server.event_converter import ACPEventConverter


pytestmark = [pytest.mark.integration, pytest.mark.anyio]


async def _collect(converter: ACPEventConverter, event: Any) -> list[Any]:
    """Collect all notifications yielded by ``converter.convert(event)``."""
    results: list[Any] = []
    results.extend([update async for update in converter.convert(event)])
    return results


# =============================================================================
# Test 15: Real ACPEventConverter (v1) emits UserMessageChunk
# =============================================================================


async def test_acp_v1_converter_emits_user_message_chunk() -> None:
    """Real ACPEventConverter emits UserMessageChunk for UserMessageInsertedEvent.

    Given: A real ``ACPEventConverter`` with default settings (v1 protocol).
    When: A ``UserMessageInsertedEvent`` with text content is converted.
    Then: Yields a ``UserMessageChunk`` with ``TextContentBlock(text="steer text")``
        and ``message_id="msg_acp"``.
    """
    # GIVEN: Real ACPEventConverter with default settings
    converter = ACPEventConverter()

    # WHEN: UserMessageInsertedEvent with steer text
    event = UserMessageInsertedEvent(
        session_id="s1",
        message_id="msg_acp",
        content="steer text",
        delivery="steer",
        source="protocol",
    )

    results = await _collect(converter, event)

    # THEN: Exactly one UserMessageChunk yielded
    assert len(results) == 1
    chunk = results[0]
    assert isinstance(chunk, UserMessageChunk)

    # AND: Chunk has TextContentBlock with the steer text
    assert isinstance(chunk.content, TextContentBlock)
    assert chunk.content.text == "steer text"

    # AND: Chunk message_id matches the event
    assert chunk.message_id == "msg_acp"


# =============================================================================
# Test 16: Real ACPEventConverter (v2) emits UserMessage (whole-message upsert)
# =============================================================================


@pytest.mark.skip(
    reason=(
        "ACP v2 UserMessage schema not yet implemented. "
        "ACPEventConverter has no protocol_version parameter and the "
        "UserMessage model does not exist in acp.schema. "
        "Skip until v2 whole-message upsert support lands."
    ),
)
async def test_acp_v2_converter_emits_user_message() -> None:
    """Real ACPEventConverter (v2) emits UserMessage for UserMessageInsertedEvent.

    Given: A real ``ACPEventConverter`` configured for ACP v2 protocol.
    When: A ``UserMessageInsertedEvent`` with text content is converted.
    Then: Yields a ``UserMessage`` (whole-message upsert) with
        ``content=[TextContentBlock(text="steer text")]``.
    """
    # This test is skipped because ACP v2 UserMessage is not yet implemented.
    # When v2 support is added:
    # 1. Add protocol_version parameter to ACPEventConverter
    # 2. Define UserMessage model in acp.schema.session_updates
    # 3. Implement v2 branch in ACPEventConverter.convert()
    # 4. Remove the skip marker and implement the assertions below.

    converter = ACPEventConverter()
    event = UserMessageInsertedEvent(
        session_id="s1",
        message_id="msg_acp",
        content="steer text",
        delivery="steer",
        source="protocol",
    )

    results = await _collect(converter, event)

    # When v2 is implemented, assert:
    # assert len(results) == 1
    # assert isinstance(results[0], UserMessage)
    # assert results[0].message_id == "msg_acp"
    # assert results[0].content == [TextContentBlock(text="steer text")]
    assert results == []  # placeholder — v2 not implemented
