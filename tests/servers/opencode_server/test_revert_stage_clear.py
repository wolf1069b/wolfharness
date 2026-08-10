"""Unit tests for STAGE (revert_session) and CLEAR (unrevert_session).

Tests the soft-marker model:
- STAGE: sets a revert marker, rolls back files, broadcasts hide events, but
  does NOT delete messages from the database or in-memory cache.
- CLEAR: clears the revert marker, restores file changes. Messages were never
  deleted so no restoration is needed.

Covers OpenSpec tasks 3.10-3.13 (STAGE) and 4.7-4.11 (CLEAR).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, Mock

import pytest

from wolfharness.utils.streams import FileOpsTracker
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
    count: int = 5,
) -> list[MessageWithParts]:
    """Add N alternating user/assistant messages to a session."""
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
# STAGE (revert_session) Tests — Tasks 3.10-3.13
# =============================================================================


class TestStageRevert:
    """STAGE unit tests for revert_session (soft-marker model)."""

    async def test_stage_sets_marker_and_broadcasts_hide_events(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
        event_capture,
    ):
        """3.10: STAGE sets revert marker, broadcasts MessageRemovedEvent and PartRemovedEvent,.

        but does NOT delete messages from DB.
        """
        create_response = await async_client.post("/session", json={"title": "STAGE Test"})
        session_id = create_response.json()["id"]

        messages = await _add_messages_to_state(server_state, session_id, count=5)
        revert_message_id = messages[2].info.id

        event_capture.clear()

        revert_response = await async_client.post(
            f"/session/{session_id}/revert",
            json={"message_id": revert_message_id},
        )

        assert revert_response.status_code == 200
        reverted_session = revert_response.json()
        assert reverted_session["revert"]["messageID"] == revert_message_id

        # MessagedRemovedEvent should be broadcast for each soft-hidden message (indices 2, 3, 4)
        removed_events = event_capture.get_events_by_type("message.removed")
        removed_msg_ids = {e.properties.message_id for e in removed_events}
        expected_hidden = {messages[i].info.id for i in range(2, 5)}
        assert removed_msg_ids == expected_hidden

        # PartRemovedEvent should be broadcast for each part of each soft-hidden message.
        # We check by message_id since append_message_to_session may add extra parts.
        part_removed_events = event_capture.get_events_by_type("message.part.removed")
        removed_part_msg_ids = {e.properties.message_id for e in part_removed_events}
        assert removed_part_msg_ids == expected_hidden

        # Messages should NOT be deleted from state.messages
        state_messages = server_state.messages.get(session_id, [])
        assert len(state_messages) == 5

        # truncate_messages should NOT be called
        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.truncate_messages.assert_not_awaited()

    async def test_stage_with_unknown_message_id_returns_404(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """3.11: STAGE with unknown message_id -> HTTP 404, no marker set, no file rollback."""
        create_response = await async_client.post("/session", json={"title": "Bad STAGE"})
        session_id = create_response.json()["id"]

        await _add_messages_to_state(server_state, session_id, count=3)

        file_ops = cast(FileOpsTracker, server_state.pool.file_ops)
        file_ops.record_change(
            path="/tmp/test_file.py",
            old_content="old",
            new_content="new",
            operation="edit",
            message_id="msg-001",
        )
        changes_before = len(file_ops.changes)

        revert_response = await async_client.post(
            f"/session/{session_id}/revert",
            json={"message_id": "nonexistent-msg-id"},
        )

        assert revert_response.status_code == 404
        assert "not found" in revert_response.json()["detail"].lower()

        # No revert marker should be set
        session = server_state.sessions.get(session_id)
        assert session is not None
        assert session.revert is None

        # File changes should NOT have been rolled back
        assert len(file_ops.changes) == changes_before

    async def test_stage_clears_queues(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """3.12: STAGE clears prompt_queue and feedback_queue."""
        create_response = await async_client.post("/session", json={"title": "Queue STAGE"})
        session_id = create_response.json()["id"]

        await _add_messages_to_state(server_state, session_id, count=3)

        # Set up a mock session state with populated queues
        from wolfharness.orchestrator.session_controller import SessionState

        mock_session_state = SessionState(session_id=session_id, agent_name="test-agent")
        mock_session_state.prompt_queue.put_nowait("queued prompt 1")
        mock_session_state.prompt_queue.put_nowait("queued prompt 2")
        from wolfharness.lifecycle.types import Feedback

        mock_session_state.feedback_queue.put_nowait(Feedback(content="feedback 1", is_steer=False))

        # Wire it into the session controller
        session_controller = Mock()
        session_controller.get_session = Mock(return_value=mock_session_state)
        server_state.session_controller = session_controller

        assert not mock_session_state.prompt_queue.empty()
        assert not mock_session_state.feedback_queue.empty()

        revert_response = await async_client.post(
            f"/session/{session_id}/revert",
            json={"message_id": "msg-001"},
        )

        assert revert_response.status_code == 200

        # Both queues should be empty after STAGE
        assert mock_session_state.prompt_queue.empty()
        assert mock_session_state.feedback_queue.empty()

    async def test_stage_double_stage_overwrites_marker(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """3.13: Double STAGE - STAGE at point A, then STAGE at point B -> marker overwritten to B,.

        previous reverted_changes cleared.
        """
        create_response = await async_client.post("/session", json={"title": "Double STAGE"})
        session_id = create_response.json()["id"]

        messages = await _add_messages_to_state(server_state, session_id, count=6)
        point_a = messages[1].info.id
        point_b = messages[3].info.id

        file_ops = cast(FileOpsTracker, server_state.pool.file_ops)
        # Record "edit" changes (not "create") so file rollback writes content
        # instead of trying to delete non-existent files.
        file_ops.record_change(
            path="/tmp/file_a.py",
            old_content="original_a",
            new_content="content_a",
            operation="edit",
            message_id=point_a,
        )

        # Pre-create the file so file rollback (writing old_content) succeeds
        import pathlib

        pathlib.Path("/tmp/file_a.py").write_text("dummy")

        # STAGE at point A
        revert_response_a = await async_client.post(
            f"/session/{session_id}/revert",
            json={"message_id": point_a},
        )
        assert revert_response_a.status_code == 200
        assert revert_response_a.json()["revert"]["messageID"] == point_a

        # After first STAGE, reverted_changes should contain changes from point_a onwards
        assert len(file_ops.reverted_changes) == 1
        assert len(file_ops.changes) == 0

        # Simulate new work after STAGE: record new file changes
        file_ops.record_change(
            path="/tmp/file_b.py",
            old_content="original_b",
            new_content="content_b",
            operation="edit",
            message_id=point_b,
        )
        pathlib.Path("/tmp/file_b.py").write_text("dummy")

        # STAGE at point B (double-STAGE)
        revert_response_b = await async_client.post(
            f"/session/{session_id}/revert",
            json={"message_id": point_b},
        )
        assert revert_response_b.status_code == 200
        assert revert_response_b.json()["revert"]["messageID"] == point_b

        # After double STAGE, the previous reverted_changes should be cleared,
        # and only the new change (from point_b) should be in reverted_changes.
        assert len(file_ops.reverted_changes) == 1  # Only point_b's change


# =============================================================================
# CLEAR (unrevert_session) Tests — Tasks 4.7-4.11
# =============================================================================


class TestClearUnrevert:
    """CLEAR unit tests for unrevert_session (soft-marker model)."""

    async def test_clear_clears_marker_and_restores_files(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """4.7: CLEAR clears marker, restores files, messages are still visible (never deleted)."""
        create_response = await async_client.post("/session", json={"title": "CLEAR Test"})
        session_id = create_response.json()["id"]

        messages = await _add_messages_to_state(server_state, session_id, count=4)
        revert_message_id = messages[1].info.id

        file_ops = cast(FileOpsTracker, server_state.pool.file_ops)
        file_ops.record_change(
            path="/tmp/clear_test.py",
            old_content="original",
            new_content="modified",
            operation="edit",
            message_id=revert_message_id,
        )

        # STAGE first
        stage_response = await async_client.post(
            f"/session/{session_id}/revert",
            json={"message_id": revert_message_id},
        )
        assert stage_response.status_code == 200
        assert stage_response.json()["revert"]["messageID"] == revert_message_id

        # After STAGE, reverted_changes should have the change
        assert len(file_ops.reverted_changes) == 1

        # CLEAR
        clear_response = await async_client.post(
            f"/session/{session_id}/unrevert",
        )
        assert clear_response.status_code == 200
        cleared_session = clear_response.json()
        assert cleared_session["revert"] is None

        # After CLEAR, reverted_changes should be restored to changes
        assert len(file_ops.reverted_changes) == 0
        assert len(file_ops.changes) == 1

        # Messages should still be in state.messages (never deleted)
        state_messages = server_state.messages.get(session_id, [])
        assert len(state_messages) == 4

    async def test_clear_without_revert_marker_returns_400(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """4.8: CLEAR without revert marker -> HTTP 400.

        Detail: 'No reverted messages to restore'.
        """
        create_response = await async_client.post("/session", json={"title": "No Marker CLEAR"})
        session_id = create_response.json()["id"]

        await _add_messages_to_state(server_state, session_id, count=3)

        clear_response = await async_client.post(
            f"/session/{session_id}/unrevert",
        )

        assert clear_response.status_code == 400
        assert "no reverted messages" in clear_response.json()["detail"].lower()

    async def test_clear_when_session_not_found_returns_404(
        self,
        async_client: AsyncClient,
    ):
        """4.9: CLEAR when session not found -> HTTP 404."""
        clear_response = await async_client.post(
            "/session/nonexistent-session-id/unrevert",
        )
        assert clear_response.status_code == 404

    async def test_clear_while_busy_auto_cancels(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """4.10: CLEAR while busy -> auto-cancel -> CLEAR succeeds."""
        create_response = await async_client.post("/session", json={"title": "Busy CLEAR"})
        session_id = create_response.json()["id"]

        messages = await _add_messages_to_state(server_state, session_id, count=3)
        revert_message_id = messages[0].info.id

        # STAGE first
        stage_response = await async_client.post(
            f"/session/{session_id}/revert",
            json={"message_id": revert_message_id},
        )
        assert stage_response.status_code == 200

        # Simulate a busy session
        from wolfharness.orchestrator.session_controller import SessionState

        mock_session_state = SessionState(session_id=session_id, agent_name="test-agent")
        mock_session_state.current_run_id = "run-busy-123"

        session_controller = Mock()
        session_controller.get_session = Mock(return_value=mock_session_state)
        server_state.session_controller = session_controller

        # Configure session_pool to support cancel
        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.cancel_run = Mock()
        session_pool.wait_for_completion = AsyncMock(return_value=session_id)

        # CLEAR should auto-cancel and succeed
        clear_response = await async_client.post(
            f"/session/{session_id}/unrevert",
        )
        assert clear_response.status_code == 200
        assert clear_response.json()["revert"] is None

        # Verify cancel was called
        session_pool.cancel_run.assert_called_once_with("run-busy-123")

    async def test_clear_does_not_restore_cleared_queues(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """4.11: CLEAR does not restore cleared queues.

        STAGE clears queues, CLEAR does not restore them.
        """
        create_response = await async_client.post("/session", json={"title": "Queue CLEAR"})
        session_id = create_response.json()["id"]

        messages = await _add_messages_to_state(server_state, session_id, count=3)
        revert_message_id = messages[0].info.id

        # Set up a mock session state with populated queues
        from wolfharness.lifecycle.types import Feedback
        from wolfharness.orchestrator.session_controller import SessionState

        mock_session_state = SessionState(session_id=session_id, agent_name="test-agent")
        mock_session_state.prompt_queue.put_nowait("queued prompt")
        mock_session_state.feedback_queue.put_nowait(Feedback(content="feedback", is_steer=False))

        session_controller = Mock()
        session_controller.get_session = Mock(return_value=mock_session_state)
        server_state.session_controller = session_controller

        # STAGE - should clear queues
        stage_response = await async_client.post(
            f"/session/{session_id}/revert",
            json={"message_id": revert_message_id},
        )
        assert stage_response.status_code == 200
        assert mock_session_state.prompt_queue.empty()
        assert mock_session_state.feedback_queue.empty()

        # CLEAR - should NOT restore the queues
        clear_response = await async_client.post(
            f"/session/{session_id}/unrevert",
        )
        assert clear_response.status_code == 200

        # Queues should still be empty
        assert mock_session_state.prompt_queue.empty()
        assert mock_session_state.feedback_queue.empty()


# =============================================================================
# Close Session with Revert Marker Tests
# =============================================================================


class TestCloseSessionWithRevert:
    """Tests for closing a session with a revert marker set.

    Closing (DELETE /{session_id}) a session with a revert marker should
    remove the session entirely - no COMMIT should fire since the session
    is being deleted, not continued.
    """

    async def test_delete_session_with_revert_marker_clears_marker_without_commit(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """Given: a session with a revert marker (STAGE).

        When: the session is deleted (DELETE /{session_id}).
        Then: the revert marker is gone (session deleted),
              truncate_messages was NOT called (no COMMIT on close).
        """
        # Create a session
        create_response = await async_client.post("/session", json={"title": "Close Revert Test"})
        session_id = create_response.json()["id"]

        # Add messages
        await _add_messages_to_state(server_state, session_id, count=5)

        # STAGE - set revert marker
        revert_response = await async_client.post(
            f"/session/{session_id}/revert",
            json={"message_id": "msg-002"},
        )
        assert revert_response.status_code == 200
        assert revert_response.json()["revert"]["messageID"] == "msg-002"

        # Reset the truncate_messages mock so we can assert on new calls during DELETE
        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.truncate_messages.reset_mock()

        # DELETE the session
        delete_response = await async_client.delete(f"/session/{session_id}")
        assert delete_response.status_code == 200
        assert delete_response.json() is True

        # Session should be deleted - no marker remains
        get_response = await async_client.get(f"/session/{session_id}")
        assert get_response.status_code == 404

        # truncate_messages should NOT have been called (no COMMIT on close)
        session_pool.truncate_messages.assert_not_awaited()
