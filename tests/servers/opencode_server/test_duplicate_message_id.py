"""Tests for duplicate message ID handling in append_message_to_session.

Bug: In the sync path (POST /message), the REST handler pre-stores the
assistant message at ``message_routes.py:392`` before calling
``route_message``. The event bridge then tries to store the same message
again (same canonical ID via ``_pending_message_ids``), causing
``ValueError: Duplicate message ID`` from the storage provider.

Fix: ``append_message_to_session`` catches ``ValueError`` for duplicate
message IDs and treats the write as idempotent.

See: https://github.com/Leoyzen/wolfharness/issues/229
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from wolfharness_server.opencode_server.models.message import (
    MessagePath,
    MessageTime,
    MessageWithParts,
)
from wolfharness_server.opencode_server.opencode_message_bridge import (
    append_message_to_session,
)


if TYPE_CHECKING:
    from wolfharness_server.opencode_server.state import ServerState


pytestmark = pytest.mark.unit


def _make_assistant_msg(msg_id: str, session_id: str) -> MessageWithParts:
    """Create a minimal MessageWithParts.assistant for testing."""
    return MessageWithParts.assistant(
        message_id=msg_id,
        session_id=session_id,
        time=MessageTime(created=0),
        agent_name="test-agent",
        model_id="test-model",
        parent_id=session_id,
        provider_id="test-provider",
        path=MessagePath(cwd="/tmp", root="/tmp"),
        mode="test-agent",
    )


class TestMockAppendMessageDuplicateCheck:
    """Verify the conftest mock _mock_append_message catches duplicates.

    This is a meta-test: it verifies the test infrastructure itself
    can detect the bug. Without this check, the mock silently accepts
    duplicate writes and the bug goes undetected.
    """

    @pytest.mark.anyio
    async def test_mock_append_message_raises_on_duplicate(
        self,
        server_state: ServerState,
    ) -> None:
        """The mock session_pool.append_message must raise ValueError for duplicates."""
        session_id = "test-duplicate-session"
        msg_id = "msg_test_duplicate_001"
        chat_msg = Mock()
        chat_msg.message_id = msg_id

        sp = server_state.pool.session_pool  # type: ignore[union-attr]
        # First write succeeds
        await sp.append_message(session_id, chat_msg)
        # Second write with same ID must raise
        with pytest.raises(ValueError, match="Duplicate message ID"):
            await sp.append_message(session_id, chat_msg)


class TestAppendMessageToSessionIdempotent:
    """Verify append_message_to_session handles duplicate writes gracefully.

    The sync path (POST /message) pre-stores the assistant message before
    the event bridge tries to store it. The second write should be silently
    skipped, not raise ValueError.
    """

    @pytest.mark.anyio
    async def test_duplicate_assistant_message_does_not_raise(
        self,
        server_state: ServerState,
    ) -> None:
        """Double-write of the same assistant message must not raise ValueError."""
        session_id = "test-double-write-session"
        msg_id = "msg_test_double_write_001"
        assistant_msg = _make_assistant_msg(msg_id, session_id)

        # Simulate REST handler pre-store (message_routes.py:392)
        await append_message_to_session(server_state, session_id, assistant_msg)

        # Simulate event bridge second write (opencode_event_bridge.py:449)
        # This must NOT raise ValueError
        await append_message_to_session(server_state, session_id, assistant_msg)

        # Verify the message is in the in-memory dict exactly once
        messages = server_state.messages.get(session_id, [])
        assistant_messages = [m for m in messages if m.info.id == msg_id]
        assert len(assistant_messages) == 1, (
            f"Expected exactly 1 assistant message in memory, got {len(assistant_messages)}"
        )

    @pytest.mark.anyio
    async def test_different_messages_both_stored(
        self,
        server_state: ServerState,
    ) -> None:
        """Non-duplicate messages must still be stored normally."""
        session_id = "test-different-messages-session"
        msg1 = _make_assistant_msg("msg_different_001", session_id)
        msg2 = _make_assistant_msg("msg_different_002", session_id)

        await append_message_to_session(server_state, session_id, msg1)
        await append_message_to_session(server_state, session_id, msg2)

        messages = server_state.messages.get(session_id, [])
        assert len(messages) == 2

    @pytest.mark.anyio
    async def test_user_message_then_assistant_message_no_error(
        self,
        server_state: ServerState,
    ) -> None:
        """User message and assistant message with different IDs both stored."""
        session_id = "test-user-then-assistant-session"

        user_msg = MessageWithParts.assistant(
            message_id="msg_user_001",
            session_id=session_id,
            time=MessageTime(created=0),
            agent_name="test-agent",
            model_id="test-model",
            parent_id=session_id,
            provider_id="test-provider",
            path=MessagePath(cwd="/tmp", root="/tmp"),
            mode="test-agent",
        )
        assistant_msg = _make_assistant_msg("msg_assistant_001", session_id)

        await append_message_to_session(server_state, session_id, user_msg)
        await append_message_to_session(server_state, session_id, assistant_msg)

        messages = server_state.messages.get(session_id, [])
        assert len(messages) == 2


class TestRouteMessagePassesMessageId:
    """Verify the mock route_message sets _pending_message_ids on the event bridge.

    This is critical for reproducing the duplicate-write bug: without
    _pending_message_ids being set, the event bridge generates its own
    message ID (different from the REST handler's), so no duplicate occurs.
    """

    @pytest.mark.anyio
    async def test_route_message_sets_pending_message_ids(
        self,
        server_state: ServerState,
    ) -> None:
        """route_message must set _pending_message_ids on the event bridge."""
        integration = server_state.session_pool_integration

        # Call route_message with a message_id
        await integration.route_message(
            session_id="test-pending-session",
            content="test prompt",
            priority="when_idle",
            input_provider=None,
            message_id="msg_test_pending_001",
            model_id="test-model",
            provider_id="test-provider",
        )

        # The integration should have the pending message ID
        assert hasattr(integration, "_pending_message_ids")
        pending = integration._pending_message_ids.get("test-pending-session")
        assert pending == "msg_test_pending_001"

        # And the pending metadata
        assert hasattr(integration, "_pending_message_metadata")
        meta = integration._pending_message_metadata.get("test-pending-session")
        assert meta is not None
        assert meta["model_id"] == "test-model"
        assert meta["provider_id"] == "test-provider"
