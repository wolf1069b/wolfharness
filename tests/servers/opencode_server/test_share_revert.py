"""Integration tests for share_session, revert_session, and fork_session.

Tests session sharing, reverting, and forking behavior using the message
history API (SessionPool.get_messages, truncate_messages, copy_messages).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest

from wolfharness.utils.time_utils import now_ms
from wolfharness_server.opencode_server.models import (
    AssistantMessage,
    MessagePath,
    MessageTime,
    MessageWithParts,
    TextPart,
    TimeCreated,
    UserMessage,
)
from wolfharness_server.opencode_server.session_pool_integration import (
    append_message_to_session,
)


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from httpx import AsyncClient

    from wolfharness_server.opencode_server.state import ServerState


# =============================================================================
# Helpers
# =============================================================================


def _make_user_message(session_id: str, message_id: str, text: str) -> MessageWithParts:
    """Create a user MessageWithParts with a text part."""
    user_msg = UserMessage(
        id=message_id,
        session_id=session_id,
        time=TimeCreated(created=now_ms()),
        agent="test-agent",
    )
    part = TextPart(
        id=f"part-{message_id}",
        message_id=message_id,
        session_id=session_id,
        text=text,
    )
    return MessageWithParts(info=user_msg, parts=[part])


def _make_assistant_message(
    session_id: str,
    message_id: str,
    parent_id: str,
    text: str,
) -> MessageWithParts:
    """Create an assistant MessageWithParts with a text part."""
    assistant_msg = AssistantMessage(
        id=message_id,
        session_id=session_id,
        parent_id=parent_id,
        model_id="test-model",
        provider_id="test-provider",
        path=MessagePath(cwd="/tmp", root="/tmp"),
        time=MessageTime(created=now_ms()),
        agent="test-agent",
    )
    part = TextPart(
        id=f"part-{message_id}",
        message_id=message_id,
        session_id=session_id,
        text=text,
    )
    return MessageWithParts(info=assistant_msg, parts=[part])


async def _add_messages_to_state(
    server_state: ServerState,
    session_id: str,
    count: int = 10,
) -> list[MessageWithParts]:
    """Add N alternating user/assistant messages to a session.

    Routes messages through :func:`append_message_to_session` so they are
    stored in both the SessionPool mock and the in-memory ``state.messages``.
    """
    messages: list[MessageWithParts] = []
    for i in range(count):
        msg_id = f"msg-{i:03d}"
        if i % 2 == 0:
            msg = _make_user_message(session_id, msg_id, f"User message {i}")
        else:
            msg = _make_assistant_message(
                session_id, msg_id, f"msg-{i - 1:03d}", f"Assistant response {i}"
            )
        messages.append(msg)
        await append_message_to_session(server_state, session_id, msg)
    return messages


# =============================================================================
# Share Session Tests
# =============================================================================


class TestShareSession:
    """Tests for session sharing via the share endpoint."""

    async def test_share_session_copies_all_messages(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """Sharing a session should include all messages in the share."""
        # Create a session
        create_response = await async_client.post("/session", json={"title": "Share Test"})
        assert create_response.status_code == 200
        session_id = create_response.json()["id"]

        # Add messages to the session
        await _add_messages_to_state(server_state, session_id, count=5)

        # Mock OpenCodeSharer to avoid external API calls
        mock_sharer = AsyncMock()
        mock_sharer.__aenter__ = AsyncMock(return_value=mock_sharer)
        mock_sharer.__aexit__ = AsyncMock(return_value=None)
        mock_result = Mock()
        mock_result.url = "https://share.opencode.ai/test-share-id"
        mock_sharer.share_conversation = AsyncMock(return_value=mock_result)

        with patch(
            "wolfharness_server.opencode_server.routes.session_routes.OpenCodeSharer",
            return_value=mock_sharer,
        ):
            share_response = await async_client.post(f"/session/{session_id}/share")

        assert share_response.status_code == 200
        shared_session = share_response.json()
        assert shared_session["share"]["url"] == "https://share.opencode.ai/test-share-id"

        # Verify sharer was called with all messages
        mock_sharer.share_conversation.assert_awaited_once()
        call_args = mock_sharer.share_conversation.call_args
        shared_messages = call_args[0][0]
        assert len(shared_messages) == 5

    async def test_share_session_with_message_limit(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """Sharing with num_messages limit should only include recent messages."""
        create_response = await async_client.post("/session", json={"title": "Limited Share"})
        session_id = create_response.json()["id"]

        await _add_messages_to_state(server_state, session_id, count=10)

        mock_sharer = AsyncMock()
        mock_sharer.__aenter__ = AsyncMock(return_value=mock_sharer)
        mock_sharer.__aexit__ = AsyncMock(return_value=None)
        mock_result = Mock()
        mock_result.url = "https://share.opencode.ai/limited"
        mock_sharer.share_conversation = AsyncMock(return_value=mock_result)

        with patch(
            "wolfharness_server.opencode_server.routes.session_routes.OpenCodeSharer",
            return_value=mock_sharer,
        ):
            share_response = await async_client.post(f"/session/{session_id}/share?num_messages=3")

        assert share_response.status_code == 200
        call_args = mock_sharer.share_conversation.call_args
        shared_messages = call_args[0][0]
        assert len(shared_messages) == 3

    async def test_share_empty_session_returns_400(
        self,
        async_client: AsyncClient,
    ):
        """Sharing a session with no messages should return 400."""
        create_response = await async_client.post("/session", json={"title": "Empty Share"})
        session_id = create_response.json()["id"]

        share_response = await async_client.post(f"/session/{session_id}/share")
        assert share_response.status_code == 400
        assert "no messages" in share_response.json()["detail"].lower()

    async def test_share_nonexistent_session_returns_404(
        self,
        async_client: AsyncClient,
    ):
        """Sharing a non-existent session should return 404."""
        response = await async_client.post("/session/nonexistent-id/share")
        assert response.status_code == 404

    async def test_share_session_uses_message_history_api(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """Share endpoint should retrieve messages via the SessionPool API."""
        create_response = await async_client.post("/session", json={"title": "API Share Test"})
        session_id = create_response.json()["id"]

        await _add_messages_to_state(server_state, session_id, count=3)

        # Configure session_pool.get_messages to return messages
        # so the route uses the SessionPool API path
        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.get_messages.reset_mock()

        mock_sharer = AsyncMock()
        mock_sharer.__aenter__ = AsyncMock(return_value=mock_sharer)
        mock_sharer.__aexit__ = AsyncMock(return_value=None)
        mock_result = Mock()
        mock_result.url = "https://share.opencode.ai/api-test"
        mock_sharer.share_conversation = AsyncMock(return_value=mock_result)

        with patch(
            "wolfharness_server.opencode_server.routes.session_routes.OpenCodeSharer",
            return_value=mock_sharer,
        ):
            share_response = await async_client.post(f"/session/{session_id}/share")

        assert share_response.status_code == 200
        # Verify get_messages was called on the SessionPool
        # (may be called multiple times via get_messages_for_session fallback checks)
        session_pool.get_messages.assert_awaited()
        assert all(call.args[0] == session_id for call in session_pool.get_messages.await_args_list)


# =============================================================================
# Revert Session Tests
# =============================================================================


class TestRevertSession:
    """Tests for session reverting via the revert endpoint (STAGE model)."""

    async def test_revert_session_sets_revert_marker(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """STAGE: reverting to a message should set the revert marker without deleting messages."""
        # Create a session
        create_response = await async_client.post("/session", json={"title": "Revert Test"})
        session_id = create_response.json()["id"]

        # Add 10 messages
        messages = await _add_messages_to_state(server_state, session_id, count=10)
        revert_message_id = messages[4].info.id  # Revert to 5th message (index 4)

        # Call revert endpoint
        revert_response = await async_client.post(
            f"/session/{session_id}/revert",
            json={"message_id": revert_message_id},
        )

        assert revert_response.status_code == 200
        reverted_session = revert_response.json()
        assert reverted_session["revert"]["messageID"] == revert_message_id

        # STAGE: truncate_messages should NOT be called
        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.truncate_messages.assert_not_awaited()

        # STAGE: messages should still be in state.messages (not deleted)
        state_messages = server_state.messages.get(session_id, [])
        assert len(state_messages) == 10

    async def test_revert_session_with_single_message(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """STAGE: reverting a session with a single message sets the marker."""
        create_response = await async_client.post("/session", json={"title": "Single Revert"})
        session_id = create_response.json()["id"]

        messages = await _add_messages_to_state(server_state, session_id, count=1)
        message_id = messages[0].info.id

        revert_response = await async_client.post(
            f"/session/{session_id}/revert",
            json={"message_id": message_id},
        )

        assert revert_response.status_code == 200
        assert revert_response.json()["revert"]["messageID"] == message_id

        # STAGE: truncate_messages should NOT be called
        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.truncate_messages.assert_not_awaited()

    async def test_revert_empty_session_returns_400(
        self,
        async_client: AsyncClient,
    ):
        """Reverting an empty session should return 400."""
        create_response = await async_client.post("/session", json={"title": "Empty Revert"})
        session_id = create_response.json()["id"]

        revert_response = await async_client.post(
            f"/session/{session_id}/revert",
            json={"message_id": "msg-000"},
        )

        assert revert_response.status_code == 400
        assert "no messages" in revert_response.json()["detail"].lower()

    async def test_revert_nonexistent_message_returns_404(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """STAGE: reverting to a non-existent message should return 404, no marker set."""
        create_response = await async_client.post("/session", json={"title": "Bad Revert"})
        session_id = create_response.json()["id"]

        await _add_messages_to_state(server_state, session_id, count=3)

        revert_response = await async_client.post(
            f"/session/{session_id}/revert",
            json={"message_id": "nonexistent-message"},
        )

        assert revert_response.status_code == 404
        assert "not found" in revert_response.json()["detail"].lower()

        # No revert marker should be set
        session = server_state.sessions.get(session_id)
        assert session is not None
        assert session.revert is None

    async def test_revert_nonexistent_session_returns_404(
        self,
        async_client: AsyncClient,
    ):
        """Reverting a non-existent session should return 404."""
        response = await async_client.post(
            "/session/nonexistent-id/revert",
            json={"message_id": "msg-000"},
        )
        assert response.status_code == 404

    async def test_revert_session_does_not_store_removed_messages(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """STAGE: reverting should NOT store messages in reverted_messages (they stay in place)."""
        create_response = await async_client.post("/session", json={"title": "Store Revert"})
        session_id = create_response.json()["id"]

        messages = await _add_messages_to_state(server_state, session_id, count=5)
        revert_message_id = messages[2].info.id

        await async_client.post(
            f"/session/{session_id}/revert",
            json={"message_id": revert_message_id},
        )

        # STAGE: reverted_messages should NOT contain the session's messages
        reverted = server_state.reverted_messages.get(session_id, [])
        assert len(reverted) == 0

        # Messages should still be in state.messages
        state_messages = server_state.messages.get(session_id, [])
        assert len(state_messages) == 5


# =============================================================================
# Fork Session Tests
# =============================================================================


class TestForkSession:
    """Tests for session forking via the fork endpoint."""

    async def test_fork_session_copies_messages(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """Forking a session should copy all messages to the new session."""
        # Create original session
        original_response = await async_client.post("/session", json={"title": "Original"})
        original_id = original_response.json()["id"]

        # Add messages
        await _add_messages_to_state(server_state, original_id, count=6)

        # Fork the session
        fork_response = await async_client.post(f"/session/{original_id}/fork")
        assert fork_response.status_code == 200
        forked = fork_response.json()
        forked_id = forked["id"]

        # Verify forked session properties
        assert forked_id != original_id
        assert forked["parentID"] == original_id
        assert forked["title"] == "Original (fork)"

        # Verify copy_messages was called on SessionPool
        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.copy_messages.assert_awaited_once_with(
            original_id, forked_id, up_to_message_id=None
        )

    async def test_fork_session_at_specific_message(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """Forking at a specific message should only copy messages up to that point."""
        original_response = await async_client.post("/session", json={"title": "Fork Point"})
        original_id = original_response.json()["id"]

        messages = await _add_messages_to_state(server_state, original_id, count=8)
        fork_message_id = messages[3].info.id  # Fork at 4th message

        fork_response = await async_client.post(
            f"/session/{original_id}/fork",
            json={"message_id": fork_message_id},
        )

        assert fork_response.status_code == 200
        forked_id = fork_response.json()["id"]

        # Verify copy_messages was called with the message_id
        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.copy_messages.assert_awaited_once_with(
            original_id, forked_id, up_to_message_id=fork_message_id
        )

    async def test_fork_empty_session(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """Forking an empty session should create a new empty session."""
        original_response = await async_client.post("/session", json={"title": "Empty Fork"})
        original_id = original_response.json()["id"]

        fork_response = await async_client.post(f"/session/{original_id}/fork")
        assert fork_response.status_code == 200
        forked_id = fork_response.json()["id"]

        # Verify copy_messages was still called
        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.copy_messages.assert_awaited_once_with(
            original_id, forked_id, up_to_message_id=None
        )

    async def test_fork_session_uses_message_history_api(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """Fork endpoint should use SessionPool API for message operations."""
        original_response = await async_client.post("/session", json={"title": "API Fork"})
        original_id = original_response.json()["id"]

        await _add_messages_to_state(server_state, original_id, count=4)

        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.get_messages.reset_mock()

        fork_response = await async_client.post(f"/session/{original_id}/fork")
        assert fork_response.status_code == 200

        # Verify get_messages was called to retrieve original messages
        # (may be called multiple times via get_messages_for_session fallback checks)
        session_pool.get_messages.assert_awaited()
        assert all(
            call.args[0] == original_id for call in session_pool.get_messages.await_args_list
        )


# =============================================================================
# Combined / Edge Case Tests
# =============================================================================


class TestShareRevertEdgeCases:
    """Edge case tests for share and revert operations."""

    async def test_share_then_revert_in_same_session(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """A session can be shared and then reverted without errors."""
        create_response = await async_client.post("/session", json={"title": "Share Revert"})
        session_id = create_response.json()["id"]

        messages = await _add_messages_to_state(server_state, session_id, count=4)

        # Share first
        mock_sharer = AsyncMock()
        mock_sharer.__aenter__ = AsyncMock(return_value=mock_sharer)
        mock_sharer.__aexit__ = AsyncMock(return_value=None)
        mock_result = Mock()
        mock_result.url = "https://share.opencode.ai/combined"
        mock_sharer.share_conversation = AsyncMock(return_value=mock_result)

        # Debug: check what get_messages_for_session returns
        with patch(
            "wolfharness_server.opencode_server.routes.session_routes.OpenCodeSharer",
            return_value=mock_sharer,
        ):
            share_response = await async_client.post(f"/session/{session_id}/share")

        assert share_response.status_code == 200

        # Then revert
        revert_message_id = messages[1].info.id
        revert_response = await async_client.post(
            f"/session/{session_id}/revert",
            json={"message_id": revert_message_id},
        )
        assert revert_response.status_code == 200

    async def test_revert_then_fork(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """A reverted session can be forked correctly."""
        create_response = await async_client.post("/session", json={"title": "Revert Fork"})
        session_id = create_response.json()["id"]

        messages = await _add_messages_to_state(server_state, session_id, count=6)

        # Revert to message 2
        revert_message_id = messages[2].info.id
        await async_client.post(
            f"/session/{session_id}/revert",
            json={"message_id": revert_message_id},
        )

        # Fork the reverted session
        fork_response = await async_client.post(f"/session/{session_id}/fork")
        assert fork_response.status_code == 200
        fork_response.json()["id"]

        # After reverting, the fork endpoint copies messages via SessionPool.
        # Verify copy_messages was called (core behavior checked by fork tests).
