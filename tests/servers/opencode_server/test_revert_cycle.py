"""Integration tests for the STAGE → COMMIT cycle and 500 bug regression.

Covers:
- Test 1: Full STAGE → COMMIT → STAGE → COMMIT cycle (primary user workflow).
- Test 2: Full HTTP cycle: POST /revert → POST /unrevert → POST /message (COMMIT
  does NOT fire after CLEAR).
- Test 3: Regression test for the original 500 bug — STAGE must NOT call
  truncate_messages, so even a provider that raises NotImplementedError
  returns 200, not 500.
- Test 4: STAGE → new message → verify new message is appended to the
  TRUNCATED history, not the original full history.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, Mock

import pytest

from wolfharness_server.opencode_server.models import (
    AssistantMessage,
    MessagePath,
    MessageRequest,
    MessageTime,
    MessageWithParts,
    TextPart,
    TextPartInput,
    TimeCreated,
    UserMessage,
)
from wolfharness_server.opencode_server.session_pool_integration import (
    append_message_to_session,
)


if TYPE_CHECKING:
    from httpx import AsyncClient

    from wolfharness_server.opencode_server.state import ServerState


pytestmark = pytest.mark.integration


# =============================================================================
# Helpers
# =============================================================================


def _make_user_message(session_id: str, message_id: str, text: str) -> MessageWithParts:
    """Create a user MessageWithParts with a text part."""
    user_msg = UserMessage(
        id=message_id,
        session_id=session_id,
        time=TimeCreated(created=0),
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
        time=MessageTime(created=0),
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
    count: int = 6,
) -> list[MessageWithParts]:
    """Add N alternating user/assistant messages to a session via the in-memory cache.

    Uses ``append_message_to_session`` so messages flow through the same path
    as production code (SessionPool + in-memory dict).
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


def _message_ids(server_state: ServerState, session_id: str) -> list[str]:
    """Return the list of message IDs currently in the in-memory cache."""
    return [m.info.id for m in server_state.messages.get(session_id, [])]


# =============================================================================
# Test 1: STAGE → COMMIT → STAGE → COMMIT cycle
# =============================================================================


class TestStageCommitCycle:
    """Full STAGE → COMMIT → STAGE → COMMIT cycle — the primary user workflow."""

    async def test_stage_commit_stage_commit_cycle(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ) -> None:
        """Given: A session with 6 messages (3 user + 3 assistant alternating).

        When: STAGE at msg-003, send new message (COMMIT), STAGE at msg-001,
              send another new message (COMMIT).
        Then: Final state has msg-000 + new_msg_1 + new_msg_2, all others gone.
        """
        # --- Setup: create session with 6 messages ---
        create_response = await async_client.post("/session", json={"title": "Cycle Test"})
        session_id = create_response.json()["id"]

        await _add_messages_to_state(server_state, session_id, count=6)
        assert _message_ids(server_state, session_id) == [
            "msg-000",
            "msg-001",
            "msg-002",
            "msg-003",
            "msg-004",
            "msg-005",
        ]

        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.truncate_messages.reset_mock()

        # --- STAGE 1: revert at msg-003 (undo to msg-002) ---
        stage_response_1 = await async_client.post(
            f"/session/{session_id}/revert",
            json={"message_id": "msg-003"},
        )
        assert stage_response_1.status_code == 200
        staged_session_1 = stage_response_1.json()
        assert staged_session_1["revert"]["messageID"] == "msg-003"

        # Marker is set on the in-memory session
        assert server_state.sessions[session_id].revert is not None
        assert server_state.sessions[session_id].revert.message_id == "msg-003"

        # Messages are NOT deleted from in-memory (soft-hide only)
        assert _message_ids(server_state, session_id) == [
            "msg-000",
            "msg-001",
            "msg-002",
            "msg-003",
            "msg-004",
            "msg-005",
        ]

        # STAGE must NOT call truncate_messages
        session_pool.truncate_messages.assert_not_awaited()

        # --- COMMIT 1: send a new message (triggers COMMIT) ---
        session_pool.truncate_messages.reset_mock()
        request_1 = MessageRequest(
            parts=[TextPartInput(text="new message after first revert")],
            agent="default",
            message_id="msg_new_1",
        )
        commit_response_1 = await async_client.post(
            f"/session/{session_id}/prompt_async",
            json=request_1.model_dump(mode="json"),
        )
        assert commit_response_1.status_code == 204

        # COMMIT fired: truncate_messages called with msg-003
        session_pool.truncate_messages.assert_awaited_once_with(session_id, "msg-003")

        # Messages 3, 4, 5 deleted from in-memory; marker cleared
        ids_after_commit_1 = _message_ids(server_state, session_id)
        assert "msg-003" not in ids_after_commit_1
        assert "msg-004" not in ids_after_commit_1
        assert "msg-005" not in ids_after_commit_1
        assert "msg-000" in ids_after_commit_1
        assert "msg-001" in ids_after_commit_1
        assert "msg-002" in ids_after_commit_1

        # New message exists
        assert "msg_new_1" in ids_after_commit_1

        # Marker cleared
        assert server_state.sessions[session_id].revert is None

        # --- STAGE 2: revert at msg-001 (undo to msg-000) ---
        session_pool.truncate_messages.reset_mock()
        stage_response_2 = await async_client.post(
            f"/session/{session_id}/revert",
            json={"message_id": "msg-001"},
        )
        assert stage_response_2.status_code == 200
        staged_session_2 = stage_response_2.json()
        assert staged_session_2["revert"]["messageID"] == "msg-001"

        # Marker set on truncated history
        assert server_state.sessions[session_id].revert is not None
        assert server_state.sessions[session_id].revert.message_id == "msg-001"

        # STAGE must NOT call truncate_messages
        session_pool.truncate_messages.assert_not_awaited()

        # --- COMMIT 2: send another new message (triggers COMMIT) ---
        session_pool.truncate_messages.reset_mock()
        request_2 = MessageRequest(
            parts=[TextPartInput(text="new message after second revert")],
            agent="default",
            message_id="msg_new_2",
        )
        commit_response_2 = await async_client.post(
            f"/session/{session_id}/prompt_async",
            json=request_2.model_dump(mode="json"),
        )
        assert commit_response_2.status_code == 204

        # COMMIT fired: truncate_messages called with msg-001
        session_pool.truncate_messages.assert_awaited_once_with(session_id, "msg-001")

        # Messages 1, 2 deleted; new_msg_1 also deleted (it was after msg-001)
        final_ids = _message_ids(server_state, session_id)
        assert "msg-001" not in final_ids
        assert "msg-002" not in final_ids
        assert "msg_new_1" not in final_ids

        # msg-000 survives (it was before the first revert point msg-001)
        assert "msg-000" in final_ids

        # New message exists
        assert "msg_new_2" in final_ids

        # Marker cleared
        assert server_state.sessions[session_id].revert is None

        # --- Final state: msg-000 + new_msg_2 ---
        assert final_ids == ["msg-000", "msg_new_2"]


# =============================================================================
# Test 2: Full HTTP cycle — POST /revert → POST /unrevert → POST /message
# =============================================================================


class TestHttpRevertUnrevertMessage:
    """Full HTTP cycle: STAGE → CLEAR → new message (COMMIT does NOT fire)."""

    async def test_revert_unrevert_message_no_commit(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ) -> None:
        """Given: A session with messages.

        When: POST /revert (STAGE) → POST /unrevert (CLEAR) → POST /message.
        Then: COMMIT does NOT fire (marker was cleared), all messages visible.
        """
        # --- Setup ---
        create_response = await async_client.post("/session", json={"title": "HTTP Cycle"})
        session_id = create_response.json()["id"]

        await _add_messages_to_state(server_state, session_id, count=4)
        original_ids = _message_ids(server_state, session_id)
        assert original_ids == ["msg-000", "msg-001", "msg-002", "msg-003"]

        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.truncate_messages.reset_mock()

        # --- STAGE: POST /revert ---
        revert_response = await async_client.post(
            f"/session/{session_id}/revert",
            json={"message_id": "msg-002"},
        )
        assert revert_response.status_code == 200
        assert revert_response.json()["revert"]["messageID"] == "msg-002"

        # Marker set
        assert server_state.sessions[session_id].revert is not None
        assert server_state.sessions[session_id].revert.message_id == "msg-002"

        # Messages still in memory (soft-hide)
        assert _message_ids(server_state, session_id) == original_ids

        # --- CLEAR: POST /unrevert ---
        unrevert_response = await async_client.post(
            f"/session/{session_id}/unrevert",
        )
        assert unrevert_response.status_code == 200
        assert unrevert_response.json()["revert"] is None

        # Marker cleared
        assert server_state.sessions[session_id].revert is None

        # Messages still present
        assert _message_ids(server_state, session_id) == original_ids

        # --- New message: POST /prompt_async ---
        session_pool.truncate_messages.reset_mock()
        request = MessageRequest(
            parts=[TextPartInput(text="message after clear")],
            agent="default",
            message_id="msg_new_clear",
        )
        msg_response = await async_client.post(
            f"/session/{session_id}/prompt_async",
            json=request.model_dump(mode="json"),
        )
        assert msg_response.status_code == 204

        # COMMIT did NOT fire — no truncate_messages call
        session_pool.truncate_messages.assert_not_awaited()

        # All original messages still visible + new message appended
        final_ids = _message_ids(server_state, session_id)
        assert "msg-000" in final_ids
        assert "msg-001" in final_ids
        assert "msg-002" in final_ids
        assert "msg-003" in final_ids
        assert "msg_new_clear" in final_ids


# =============================================================================
# Test 3: Regression test for original 500 bug
# =============================================================================


class TestRegression500Bug:
    """Regression test for the original 500 bug.

    The original bug was ``NotImplementedError: SQLModelProvider does not
    support truncating messages``. The fix ensures STAGE does NOT call
    truncate_messages — only COMMIT does, and COMMIT suppresses
    NotImplementedError. So STAGE always returns 200 even if the provider
    would raise NotImplementedError on truncate.
    """

    async def test_stage_does_not_500_with_failing_provider(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ) -> None:
        """Given: A session with messages and a provider that raises.

        NotImplementedError on truncate_messages.
        When: POST /revert (STAGE).
        Then: Response is 200 (not 500), truncate_messages NOT called.
        """
        # --- Setup ---
        create_response = await async_client.post("/session", json={"title": "500 Bug Regression"})
        session_id = create_response.json()["id"]

        await _add_messages_to_state(server_state, session_id, count=4)

        # Configure truncate_messages to raise NotImplementedError — simulating
        # the original SQLModelProvider that didn't support truncation.
        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.truncate_messages = AsyncMock(
            side_effect=NotImplementedError("SQLModelProvider does not support truncating messages")
        )

        # --- STAGE: POST /revert ---
        revert_response = await async_client.post(
            f"/session/{session_id}/revert",
            json={"message_id": "msg-002"},
        )

        # The original bug would have returned 500 here. The fix ensures STAGE
        # never calls truncate_messages, so the response is 200.
        assert revert_response.status_code == 200
        assert revert_response.json()["revert"]["messageID"] == "msg-002"

        # STAGE must NOT have called truncate_messages at all
        session_pool.truncate_messages.assert_not_awaited()

        # Marker is set
        assert server_state.sessions[session_id].revert is not None
        assert server_state.sessions[session_id].revert.message_id == "msg-002"

        # Messages still in memory (soft-hide, not deleted)
        assert len(server_state.messages[session_id]) == 4


# =============================================================================
# Test 4: STAGE → new message → verify appended to TRUNCATED history
# =============================================================================


class TestCommitTruncatesBeforeAppend:
    """Verify COMMIT truncates the history BEFORE the new message is appended.

    This is the critical invariant: after STAGE → new message, the in-memory
    list should be [messages_before_revert, new_message], NOT
    [all_original_messages, new_message].
    """

    async def test_new_message_appended_to_truncated_history(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ) -> None:
        """Given: A session with 6 messages, STAGE at msg-003.

        When: A new message is sent (triggers COMMIT).
        Then: in-memory messages = [msg-000, msg-001, msg-002, new_message]
              (NOT [msg-000, ..., msg-005, new_message]).
        """
        # --- Setup ---
        create_response = await async_client.post(
            "/session", json={"title": "Truncated Append Test"}
        )
        session_id = create_response.json()["id"]

        await _add_messages_to_state(server_state, session_id, count=6)
        assert _message_ids(server_state, session_id) == [
            "msg-000",
            "msg-001",
            "msg-002",
            "msg-003",
            "msg-004",
            "msg-005",
        ]

        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.truncate_messages.reset_mock()

        # --- STAGE at msg-003 ---
        stage_response = await async_client.post(
            f"/session/{session_id}/revert",
            json={"message_id": "msg-003"},
        )
        assert stage_response.status_code == 200

        # --- Send new message (triggers COMMIT) ---
        request = MessageRequest(
            parts=[TextPartInput(text="new message after revert")],
            agent="default",
            message_id="msg_new_truncated",
        )
        commit_response = await async_client.post(
            f"/session/{session_id}/prompt_async",
            json=request.model_dump(mode="json"),
        )
        assert commit_response.status_code == 204

        # COMMIT fired: truncate_messages called
        session_pool.truncate_messages.assert_awaited_once_with(session_id, "msg-003")

        # The in-memory list should be [msg-000, msg-001, msg-002, msg_new_truncated]
        # NOT [msg-000, msg-001, msg-002, msg-003, msg-004, msg-005, msg_new_truncated]
        final_ids = _message_ids(server_state, session_id)
        assert final_ids == ["msg-000", "msg-001", "msg-002", "msg_new_truncated"]

        # Explicitly verify the reverted messages are gone
        assert "msg-003" not in final_ids
        assert "msg-004" not in final_ids
        assert "msg-005" not in final_ids

        # Marker cleared
        assert server_state.sessions[session_id].revert is None


# =============================================================================
# Test 5: Sync path COMMIT — STAGE → POST /message (sync) → COMMIT fires
# =============================================================================


class TestSyncPathCommit:
    """Verify COMMIT fires on the SYNC path (POST /session/{id}/message).

    All previous COMMIT tests use prompt_async (async path). This test
    covers the sync path where _process_message must run COMMIT BEFORE
    creating the user message (fixed in commit 990c0069e).
    """

    async def test_sync_path_commit_before_user_message(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ) -> None:
        """Given: A session with 4 messages, STAGE at msg-002.

        When: POST /session/{id}/message (sync path) sends a new message.
        Then: COMMIT fires (truncate_messages called), the new user message
              exists in state.messages (NOT truncated by COMMIT), and
              messages at/after msg-002 are gone.
        """
        # --- Setup ---
        create_response = await async_client.post("/session", json={"title": "Sync Commit Test"})
        session_id = create_response.json()["id"]

        await _add_messages_to_state(server_state, session_id, count=4)
        assert _message_ids(server_state, session_id) == [
            "msg-000",
            "msg-001",
            "msg-002",
            "msg-003",
        ]

        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.truncate_messages.reset_mock()

        # --- STAGE at msg-002 ---
        stage_response = await async_client.post(
            f"/session/{session_id}/revert",
            json={"message_id": "msg-002"},
        )
        assert stage_response.status_code == 200

        # --- Send via SYNC path (POST /message, not /prompt_async) ---
        request = MessageRequest(
            parts=[TextPartInput(text="sync path message after revert")],
            agent="default",
            message_id="msg_sync_new",
        )
        # The sync path calls _process_message which calls pool.session_pool.send_message.
        # pool.session_pool.send_message is an AsyncMock that returns a Mock run_handle.
        # This means the sync path will return quickly (mocked agent).
        # We just need to verify COMMIT fired and the new message wasn't truncated.
        sync_response = await async_client.post(
            f"/session/{session_id}/message",
            json=request.model_dump(mode="json"),
        )
        # Sync path returns the created MessageWithParts
        assert sync_response.status_code == 200

        # COMMIT fired: truncate_messages called with msg-002
        session_pool.truncate_messages.assert_awaited_once_with(session_id, "msg-002")

        # The new user message exists in state.messages (NOT truncated by COMMIT)
        final_ids = _message_ids(server_state, session_id)
        assert "msg_sync_new" in final_ids, (
            f"New message msg_sync_new missing from state.messages: {final_ids}"
        )

        # Reverted messages are gone
        assert "msg-002" not in final_ids
        assert "msg-003" not in final_ids

        # Pre-revert messages preserved
        assert "msg-000" in final_ids
        assert "msg-001" in final_ids

        # Marker cleared
        assert server_state.sessions[session_id].revert is None


# =============================================================================
# Test 6: STAGE → CLEAR → STAGE → CLEAR (repeated undo/redo, task 9.1)
# =============================================================================


class TestStageClearCycle:
    """Repeated undo/redo cycle — messages are never deleted from in-memory."""

    async def test_stage_clear_stage_clear_cycle(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ) -> None:
        """Given: A session with 6 messages.

        When: STAGE → CLEAR → STAGE → CLEAR (two full undo/redo cycles).
        Then: All 6 messages remain in state.messages throughout (soft-hide only),
              truncate_messages is NEVER called, and the marker toggles correctly.
        """
        # --- Setup ---
        create_response = await async_client.post("/session", json={"title": "Undo/Redo Cycle"})
        session_id = create_response.json()["id"]

        await _add_messages_to_state(server_state, session_id, count=6)
        initial_ids = _message_ids(server_state, session_id)

        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.truncate_messages.reset_mock()

        # --- STAGE 1: revert at msg-004 ---
        stage_1 = await async_client.post(
            f"/session/{session_id}/revert",
            json={"message_id": "msg-004"},
        )
        assert stage_1.status_code == 200
        assert server_state.sessions[session_id].revert is not None
        assert server_state.sessions[session_id].revert.message_id == "msg-004"

        # Messages NOT deleted
        assert _message_ids(server_state, session_id) == initial_ids
        session_pool.truncate_messages.assert_not_awaited()

        # --- CLEAR 1: unrevert ---
        clear_1 = await async_client.post(
            f"/session/{session_id}/unrevert",
        )
        assert clear_1.status_code == 200
        assert server_state.sessions[session_id].revert is None

        # Messages still NOT deleted
        assert _message_ids(server_state, session_id) == initial_ids
        session_pool.truncate_messages.assert_not_awaited()

        # --- STAGE 2: revert at msg-002 (different point) ---
        stage_2 = await async_client.post(
            f"/session/{session_id}/revert",
            json={"message_id": "msg-002"},
        )
        assert stage_2.status_code == 200
        assert server_state.sessions[session_id].revert is not None
        assert server_state.sessions[session_id].revert.message_id == "msg-002"

        # Messages still NOT deleted
        assert _message_ids(server_state, session_id) == initial_ids
        session_pool.truncate_messages.assert_not_awaited()

        # --- CLEAR 2: unrevert ---
        clear_2 = await async_client.post(
            f"/session/{session_id}/unrevert",
        )
        assert clear_2.status_code == 200
        assert server_state.sessions[session_id].revert is None

        # Messages still NOT deleted
        assert _message_ids(server_state, session_id) == initial_ids
        session_pool.truncate_messages.assert_not_awaited()


# =============================================================================
# Test 7: Concurrent STAGE calls (task 9.6)
# =============================================================================


class TestConcurrentStage:
    """Two simultaneous POST /revert requests — verify no corruption."""

    async def test_concurrent_stage_calls_no_corruption(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ) -> None:
        """Given: A session with 6 messages.

        When: Two STAGE requests are sent concurrently (both targeting different
              message IDs).
        Then: Both complete without error, the session has a valid revert marker
              (one of the two), and messages are not corrupted.
        """
        import asyncio

        # --- Setup ---
        create_response = await async_client.post("/session", json={"title": "Concurrent STAGE"})
        session_id = create_response.json()["id"]

        await _add_messages_to_state(server_state, session_id, count=6)
        original_ids = _message_ids(server_state, session_id)

        # --- Fire two STAGE requests concurrently ---
        task_a = asyncio.create_task(
            async_client.post(
                f"/session/{session_id}/revert",
                json={"message_id": "msg-002"},
            )
        )
        task_b = asyncio.create_task(
            async_client.post(
                f"/session/{session_id}/revert",
                json={"message_id": "msg-004"},
            )
        )

        response_a, response_b = await asyncio.gather(task_a, task_b)

        # Both should succeed (200) — the session lock serializes them
        assert response_a.status_code == 200
        assert response_b.status_code == 200

        # The session should have a valid revert marker (the last one to acquire the lock wins)
        session = server_state.sessions.get(session_id)
        assert session is not None
        assert session.revert is not None
        assert session.revert.message_id in {"msg-002", "msg-004"}

        # Messages are not corrupted — all still present (soft-hide)
        assert _message_ids(server_state, session_id) == original_ids


# =============================================================================
# Test 8: Auto-cancel timeout integration (task 9.11)
# =============================================================================


class TestAutoCancelTimeoutIntegration:
    """STAGE while busy with a run that blocks >10s — timeout force-clears."""

    async def test_stage_proceeds_after_cancel_timeout(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ) -> None:
        """Given: A session with messages and a busy run that doesn't respond to cancel.

        When: POST /revert is called while the session is busy.
        Then: _ensure_session_idle times out (10s), force-clears current_run_id,
              and STAGE proceeds — revert marker is set.
        """
        # --- Setup ---
        create_response = await async_client.post("/session", json={"title": "Timeout Integration"})
        session_id = create_response.json()["id"]

        await _add_messages_to_state(server_state, session_id, count=4)

        # Simulate a busy session: set current_run_id on the session state
        from wolfharness.orchestrator.session_pool import SessionState as PoolSessionState

        busy_session = PoolSessionState(
            session_id=session_id,
            agent_name="test-agent",
            current_run_id="run-that-never-completes",
        )

        # Override get_session to return our busy session
        session_controller = server_state.session_controller
        if session_controller is not None:
            session_controller.get_session = Mock(return_value=busy_session)

        # Make cancel_run a no-op (doesn't actually cancel)
        session_controller.cancel_run_for_session = Mock()

        # Patch asyncio.wait_for to timeout immediately (can't wait 10s in a test)
        import asyncio as asyncio_mod

        original_wait_for = asyncio_mod.wait_for

        async def _quick_timeout(coro, timeout):
            coro.close()
            raise TimeoutError("Simulated timeout")

        asyncio_mod.wait_for = _quick_timeout

        try:
            # --- STAGE while busy ---
            stage_response = await async_client.post(
                f"/session/{session_id}/revert",
                json={"message_id": "msg-002"},
            )
            # STAGE should succeed despite the timeout
            assert stage_response.status_code == 200

            # Revert marker is set
            session = server_state.sessions.get(session_id)
            assert session is not None
            assert session.revert is not None
            assert session.revert.message_id == "msg-002"

            # current_run_id was force-cleared
            assert busy_session.current_run_id is None
        finally:
            # Restore original wait_for
            asyncio_mod.wait_for = original_wait_for


# =============================================================================
# Test 9: Stale RunHandle — current_run_id set but handle gone (task 2.3)
# =============================================================================


class TestStaleRunHandle:
    """_ensure_session_idle when current_run_id is set but the RunHandle is gone."""

    async def test_stale_run_handle_force_clears(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ) -> None:
        """Given: A session where current_run_id is stale (RunHandle gone).

        When: POST /revert is called.
        Then: _ensure_session_idle detects the stale reference, force-clears
              current_run_id, and STAGE proceeds without waiting.
        """
        # --- Setup ---
        create_response = await async_client.post("/session", json={"title": "Stale Handle"})
        session_id = create_response.json()["id"]

        await _add_messages_to_state(server_state, session_id, count=4)

        # Simulate stale run handle
        from wolfharness.orchestrator.session_pool import SessionState as PoolSessionState

        stale_session = PoolSessionState(
            session_id=session_id,
            agent_name="test-agent",
            current_run_id="stale-run-id",
        )

        session_controller = server_state.session_controller
        if session_controller is not None:
            session_controller.get_session = Mock(return_value=stale_session)
            # cancel_run_for_session is already a Mock (no-op from conftest)

        # --- STAGE ---
        stage_response = await async_client.post(
            f"/session/{session_id}/revert",
            json={"message_id": "msg-002"},
        )
        # Should succeed — stale run is force-cleared
        assert stage_response.status_code == 200

        # Revert marker set
        session = server_state.sessions.get(session_id)
        assert session is not None
        assert session.revert is not None
        assert session.revert.message_id == "msg-002"

        # current_run_id force-cleared
        assert stale_session.current_run_id is None
