"""Unit test for D14 user_msg_id / assistant_msg_id separation.

Verifies that ``route_message()`` stores ``assistant_msg_id`` in
``_pending_message_ids`` (for D14 event bridge reuse) while passing
``message_id`` (the user message ID) to ``send_message()`` (for
``UserMessageInsertedEvent`` dedup correlation).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from wolfharness_server.opencode_server.session_pool_integration import (
    OpenCodeSessionPoolIntegration,
)


pytestmark = [pytest.mark.unit, pytest.mark.anyio]


async def test_route_message_separates_user_and_assistant_ids() -> None:
    """route_message() stores assistant_msg_id for D14, passes user_msg_id for dedup.

    Given: An OpenCodeSessionPoolIntegration with a mocked session_pool.
    When: route_message() is called with message_id="user-1" and
        assistant_msg_id="assistant-1".
    Then: _pending_message_ids["session"] == "assistant-1" (D14).
    And: send_message() is called with message_id="user-1" (UserMessageInsertedEvent).
    """
    session_pool = MagicMock()
    session_pool.send_message = AsyncMock(return_value="user-1")
    server_state = MagicMock()

    integration = OpenCodeSessionPoolIntegration(
        session_pool=session_pool,
        server_state=server_state,
    )
    # Prevent _start_event_consumer from trying to use a mock EventBus
    integration._start_event_consumer = AsyncMock()  # type: ignore[method-assign]

    await integration.route_message(
        session_id="test-session",
        content="hello",
        message_id="user-1",
        assistant_msg_id="assistant-1",
    )

    # D14: assistant_msg_id stored for event bridge reuse
    assert integration._pending_message_ids.get("test-session") == "assistant-1"

    # UserMessageInsertedEvent: user_msg_id passed to send_message
    send_call = session_pool.send_message.call_args
    assert send_call is not None
    assert send_call.kwargs.get("message_id") == "user-1"


async def test_route_message_falls_back_to_message_id_for_d14() -> None:
    """When assistant_msg_id is None, message_id is used for D14 fallback.

    Given: route_message() called with only message_id (no assistant_msg_id).
    Then: _pending_message_ids falls back to message_id for backward compat.
    """
    session_pool = MagicMock()
    session_pool.send_message = AsyncMock(return_value="msg-fb")
    server_state = MagicMock()

    integration = OpenCodeSessionPoolIntegration(
        session_pool=session_pool,
        server_state=server_state,
    )
    integration._start_event_consumer = AsyncMock()  # type: ignore[method-assign]

    await integration.route_message(
        session_id="test-session-fb",
        content="hello",
        message_id="msg-fb",
    )

    # Fallback: message_id used for both purposes
    assert integration._pending_message_ids.get("test-session-fb") == "msg-fb"
