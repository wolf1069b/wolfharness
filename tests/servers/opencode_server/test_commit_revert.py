"""Tests for COMMIT logic in new-message paths (tasks 5.4-5.10).

When a new message is sent to a session that has ``session.revert`` set
(STAGE), COMMIT truncates reverted messages from DB + in-memory + agent
history, clears the marker, and clears FileOps backup — BEFORE creating
the new user message.

These tests verify the DB-first ordering (D10), suppress scope, error
propagation, and interaction with STAGE/CLEAR.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from wolfharness.messaging import ChatMessage
from wolfharness.utils.streams import FileChange
from wolfharness.utils.time_utils import now_ms
from wolfharness_server.opencode_server.models import (
    AssistantMessage,
    MessagePath,
    MessageRequest,
    MessageTime,
    MessageWithParts,
    Session,
    SessionRevert,
    TextPart,
    TextPartInput,
    TimeCreated,
    TimeCreatedUpdated,
    UserMessage,
)
from wolfharness_server.opencode_server.routes.message_routes import (
    _commit_revert,
    _truncate_agent_history,
)


if TYPE_CHECKING:
    from wolfharness_server.opencode_server.state import ServerState


pytestmark = pytest.mark.unit


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


def _make_chat_message(message_id: str, role: str = "user") -> ChatMessage[str]:
    """Create a ChatMessage with a specific message_id."""
    return ChatMessage(
        content=f"content-{message_id}",
        role=role,  # type: ignore[arg-type]
        message_id=message_id,
    )


def _setup_staged_session(
    state: ServerState,
    session_id: str,
    num_messages: int = 5,
    revert_at_index: int = 2,
) -> tuple[str, list[MessageWithParts]]:
    """Set up a session in STAGED state with messages and a revert marker.

    Creates ``num_messages`` alternating user/assistant messages, then sets
    a revert marker at ``revert_at_index``.

    Returns ``(revert_message_id, messages)``.
    """
    messages: list[MessageWithParts] = []
    for i in range(num_messages):
        msg_id = f"msg-{i:03d}"
        if i % 2 == 0:
            msg = _make_user_message(session_id, msg_id, f"User message {i}")
        else:
            msg = _make_assistant_message(
                session_id, msg_id, f"msg-{i - 1:03d}", f"Assistant response {i}"
            )
        messages.append(msg)
        state.messages.setdefault(session_id, []).append(msg)

    revert_msg_id = messages[revert_at_index].info.id

    # Set revert marker on session
    session = state.sessions[session_id]
    revert_info = SessionRevert(message_id=revert_msg_id)
    state.sessions[session_id] = session.model_copy(update={"revert": revert_info})

    # Ensure get_or_load_session returns the cached session (with revert marker)
    # by making session_controller.get_session return a non-None value.
    # Otherwise get_or_load_session falls through to storage which doesn't
    # have the revert marker.
    session_pool = cast(Mock, state.pool.session_pool)
    mock_session_state = Mock()
    mock_session_state.agent = None  # No agent by default; _truncate_agent_history returns early
    session_pool.sessions.get_session = Mock(return_value=mock_session_state)

    # Store reverted messages (as revert_session does)
    state.reverted_messages[session_id] = list(messages[revert_at_index:])

    # Add some reverted file changes
    state.pool.file_ops.reverted_changes.append(
        FileChange(
            path="/tmp/test.py",
            old_content="old",
            new_content="new",
            operation="edit",
            message_id=revert_msg_id,
        )
    )

    return revert_msg_id, messages


def _setup_agent_history(
    state: ServerState,
    session_id: str,
    messages: list[MessageWithParts],
) -> list[ChatMessage[str]]:
    """Set up mock agent conversation history matching the message list.

    Creates ChatMessages with message_id matching the OpenCode message IDs,
    and wires them into the mock agent's conversation.
    """
    chat_messages: list[ChatMessage[str]] = []
    for mwp in messages:
        role = "user" if isinstance(mwp.info, UserMessage) else "assistant"
        chat_messages.append(_make_chat_message(mwp.info.id, role=role))

    session_pool = cast(Mock, state.pool.session_pool)
    session_state = session_pool.sessions.get_session.return_value
    if session_state is None:
        session_state = Mock()
        session_pool.sessions.get_session.return_value = session_state

    agent = Mock()
    conversation = Mock()
    conversation.chat_messages = list(chat_messages)

    def _set_history(history: list[ChatMessage[Any]]) -> None:
        conversation.chat_messages = list(history)

    conversation.set_history = _set_history
    agent.conversation = conversation
    session_state.agent = agent

    return chat_messages


# =============================================================================
# 5.4: Integration test — full COMMIT
# =============================================================================


class TestCommitFull:
    """5.4: Full COMMIT — messages deleted from DB, in-memory, agent history,.

    marker cleared, FileOps backup cleared, new message created.
    """

    async def test_commit_deletes_from_db_and_memory_and_agent(
        self,
        async_client,
        server_state: ServerState,
    ) -> None:
        """Given: Session in STAGED state with 5 messages, revert at index 2.

        When: _commit_revert is called.
        Then: DB truncated, in-memory truncated, agent history truncated,
              marker cleared, reverted_messages cleared, FileOps cleared.
        """
        session_id = "test-commit-full"
        _create_session_in_state(server_state, session_id)
        revert_msg_id, messages = _setup_staged_session(
            server_state, session_id, num_messages=5, revert_at_index=2
        )
        _setup_agent_history(server_state, session_id, messages)

        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.truncate_messages.reset_mock()

        await _commit_revert(server_state, session_id)

        # 1. DB truncate called with correct args
        session_pool.truncate_messages.assert_awaited_once_with(session_id, revert_msg_id)

        # 2. In-memory messages truncated (messages before index 2 remain)
        remaining = server_state.messages[session_id]
        assert len(remaining) == 2
        assert remaining[0].info.id == "msg-000"
        assert remaining[1].info.id == "msg-001"

        # 3. Agent history truncated
        session_state = session_pool.sessions.get_session.return_value
        agent_chat_messages = session_state.agent.conversation.chat_messages
        assert len(agent_chat_messages) == 2
        assert agent_chat_messages[0].message_id == "msg-000"
        assert agent_chat_messages[1].message_id == "msg-001"

        # 4. Revert marker cleared
        assert server_state.sessions[session_id].revert is None

        # 5. reverted_messages cleared
        assert session_id not in server_state.reverted_messages

        # 6. FileOps reverted_changes cleared
        assert server_state.pool.file_ops.reverted_changes == []


# =============================================================================
# 5.5: COMMIT with NotImplementedError — suppress path
# =============================================================================


class TestCommitSuppressPath:
    """5.5: COMMIT where truncate_messages raises NotImplementedError —.

    in-memory still correct, marker cleared, new message processed.
    """

    async def test_commit_suppresses_not_implemented_error(
        self,
        async_client,
        server_state: ServerState,
    ) -> None:
        """Given: truncate_messages raises NotImplementedError.

        When: _commit_revert is called.
        Then: Error suppressed, in-memory truncated, marker cleared.
        """
        session_id = "test-commit-nie"
        _create_session_in_state(server_state, session_id)
        _revert_msg_id, messages = _setup_staged_session(
            server_state, session_id, num_messages=5, revert_at_index=2
        )
        _setup_agent_history(server_state, session_id, messages)

        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.truncate_messages = AsyncMock(side_effect=NotImplementedError("nope"))

        await _commit_revert(server_state, session_id)

        # In-memory still truncated (suppress only wraps DB call)
        remaining = server_state.messages[session_id]
        assert len(remaining) == 2

        # Marker cleared
        assert server_state.sessions[session_id].revert is None

    async def test_commit_suppresses_key_error(
        self,
        async_client,
        server_state: ServerState,
    ) -> None:
        """Given: truncate_messages raises KeyError.

        When: _commit_revert is called.
        Then: Error suppressed, in-memory truncated, marker cleared.
        """
        session_id = "test-commit-ke"
        _create_session_in_state(server_state, session_id)
        _setup_staged_session(server_state, session_id, num_messages=3, revert_at_index=1)

        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.truncate_messages = AsyncMock(side_effect=KeyError("missing"))

        await _commit_revert(server_state, session_id)

        remaining = server_state.messages[session_id]
        assert len(remaining) == 1
        assert server_state.sessions[session_id].revert is None

    async def test_commit_suppresses_type_error(
        self,
        async_client,
        server_state: ServerState,
    ) -> None:
        """Given: truncate_messages raises TypeError.

        When: _commit_revert is called.
        Then: Error suppressed, in-memory truncated, marker cleared.
        """
        session_id = "test-commit-te"
        _create_session_in_state(server_state, session_id)
        _setup_staged_session(server_state, session_id, num_messages=3, revert_at_index=1)

        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.truncate_messages = AsyncMock(side_effect=TypeError("bad type"))

        await _commit_revert(server_state, session_id)

        remaining = server_state.messages[session_id]
        assert len(remaining) == 1
        assert server_state.sessions[session_id].revert is None


# =============================================================================
# 5.6: COMMIT with non-suppressed error — error propagates
# =============================================================================


class TestCommitNonSuppressedError:
    """5.6: COMMIT where truncate_messages raises a non-suppressed error —.

    error propagates, in-memory NOT truncated, marker NOT cleared.
    """

    async def test_commit_runtime_error_propagates(
        self,
        async_client,
        server_state: ServerState,
    ) -> None:
        """Given: truncate_messages raises RuntimeError.

        When: _commit_revert is called.
        Then: RuntimeError propagates, in-memory NOT truncated, marker NOT cleared.
        """
        session_id = "test-commit-runtime"
        _create_session_in_state(server_state, session_id)
        revert_msg_id, messages = _setup_staged_session(
            server_state, session_id, num_messages=5, revert_at_index=2
        )
        _setup_agent_history(server_state, session_id, messages)

        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.truncate_messages = AsyncMock(side_effect=RuntimeError("db down"))

        with pytest.raises(RuntimeError, match="db down"):
            await _commit_revert(server_state, session_id)

        # In-memory NOT truncated (all 5 messages remain)
        remaining = server_state.messages[session_id]
        assert len(remaining) == 5

        # Marker NOT cleared
        assert server_state.sessions[session_id].revert is not None
        assert server_state.sessions[session_id].revert.message_id == revert_msg_id

        # reverted_messages NOT cleared
        assert session_id in server_state.reverted_messages

        # FileOps reverted_changes NOT cleared
        assert len(server_state.pool.file_ops.reverted_changes) > 0

    async def test_commit_value_error_propagates(
        self,
        async_client,
        server_state: ServerState,
    ) -> None:
        """Given: truncate_messages raises ValueError (non-suppressed).

        When: _commit_revert is called.
        Then: ValueError propagates, in-memory NOT truncated.
        """
        session_id = "test-commit-value"
        _create_session_in_state(server_state, session_id)
        _setup_staged_session(server_state, session_id, num_messages=3, revert_at_index=1)

        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.truncate_messages = AsyncMock(side_effect=ValueError("bad value"))

        with pytest.raises(ValueError, match="bad value"):
            await _commit_revert(server_state, session_id)

        # In-memory NOT truncated
        assert len(server_state.messages[session_id]) == 3
        # Marker NOT cleared
        assert server_state.sessions[session_id].revert is not None


# =============================================================================
# 5.7: COMMIT ordering — DB before in-memory before marker clearing
# =============================================================================


class TestCommitOrdering:
    """5.7: Verify DB truncate called BEFORE in-memory truncation, and.

    in-memory truncation BEFORE marker clearing.
    """

    async def test_commit_db_before_memory_before_marker(
        self,
        async_client,
        server_state: ServerState,
    ) -> None:
        """Given: Session in STAGED state.

        When: _commit_revert is called with ordered mock tracking.
        Then: DB truncate happens first, then in-memory, then marker clear.
        """
        session_id = "test-commit-order"
        _create_session_in_state(server_state, session_id)
        _revert_msg_id, _messages = _setup_staged_session(
            server_state, session_id, num_messages=5, revert_at_index=2
        )

        call_order: list[str] = []

        session_pool = cast(Mock, server_state.pool.session_pool)

        original_truncate = session_pool.truncate_messages

        async def _tracking_truncate(sid: str, up_to: str) -> int:
            call_order.append("db_truncate")
            # At this point, in-memory should NOT be truncated yet
            assert len(server_state.messages[sid]) == 5
            # Marker should still be set
            assert server_state.sessions[sid].revert is not None
            return await original_truncate(sid, up_to)

        session_pool.truncate_messages = AsyncMock(side_effect=_tracking_truncate)

        # Patch broadcast_event to track marker clear timing
        original_broadcast = server_state.broadcast_event

        async def _tracking_broadcast(event: Any) -> None:
            from wolfharness_server.opencode_server.models import SessionUpdatedEvent

            if isinstance(event, SessionUpdatedEvent):
                call_order.append("broadcast_session_updated")
                # By broadcast time, in-memory should be truncated and marker cleared
                assert len(server_state.messages[session_id]) == 2
                assert server_state.sessions[session_id].revert is None
            await original_broadcast(event)

        server_state.broadcast_event = _tracking_broadcast  # type: ignore[method-assign]

        await _commit_revert(server_state, session_id)

        # Verify order: DB truncate happened, then broadcast (which is after
        # in-memory truncation and marker clearing)
        assert "db_truncate" in call_order
        assert "broadcast_session_updated" in call_order
        assert call_order.index("db_truncate") < call_order.index("broadcast_session_updated")


# =============================================================================
# 5.8: contextlib.suppress scope — only wraps truncate_messages
# =============================================================================


class TestCommitSuppressScope:
    """5.8: Verify suppress wraps ONLY truncate_messages call. Errors in.

    in-memory access or other steps are NOT suppressed.
    """

    async def test_key_error_in_messages_access_not_suppressed(
        self,
        async_client,
        server_state: ServerState,
    ) -> None:
        """Given: state.messages access raises KeyError (not from truncate).

        When: _commit_revert is called.
        Then: KeyError propagates (NOT suppressed by the DB suppress).

        Note: This tests that the suppress scope is narrow — it only wraps
        truncate_messages, not the entire COMMIT block. We simulate this by
        making state.messages a dict-like that raises on .get().
        """
        session_id = "test-commit-scope"
        _create_session_in_state(server_state, session_id)
        _setup_staged_session(server_state, session_id, num_messages=3, revert_at_index=1)

        # Replace state.messages with a dict that raises KeyError on get
        # (not on __getitem__ — on the .get() method itself)
        original_messages = server_state.messages

        class _RaisingMessagesDict(dict):
            def get(self, key, default=None):
                raise KeyError("injected error in messages access")

        server_state.messages = _RaisingMessagesDict(original_messages)  # type: ignore[assignment]

        with pytest.raises(KeyError, match="injected error"):
            await _commit_revert(server_state, session_id)

        # Marker should NOT be cleared (error happened before that step)
        assert server_state.sessions[session_id].revert is not None


# =============================================================================
# 5.9: Session reload from DB after COMMIT
# =============================================================================


class TestCommitDbReload:
    """5.9: After COMMIT, truncated messages are gone from DB, not just.

    in-memory. Verify by checking truncate_messages was called (DB path).
    """

    async def test_commit_calls_truncate_on_db(
        self,
        async_client,
        server_state: ServerState,
    ) -> None:
        """Given: Session in STAGED state with messages in DB.

        When: _commit_revert is called.
        Then: session_pool.truncate_messages is called with correct args,
              confirming DB truncation occurred.
        """
        session_id = "test-commit-dbreload"
        _create_session_in_state(server_state, session_id)
        revert_msg_id, _messages = _setup_staged_session(
            server_state, session_id, num_messages=5, revert_at_index=2
        )

        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.truncate_messages.reset_mock()

        await _commit_revert(server_state, session_id)

        # DB truncate was called — this is the proof that DB was truncated
        session_pool.truncate_messages.assert_awaited_once_with(session_id, revert_msg_id)

        # After COMMIT, a fresh get_messages call would return truncated list
        # (In mock setup, get_messages returns from _mock_chat_store, but
        # truncate_messages doesn't modify it. The real SQL provider would
        # have deleted rows. We verify the call was made.)


# =============================================================================
# 5.10: STAGE → CLEAR → new message — COMMIT does NOT fire
# =============================================================================


class TestCommitAfterClear:
    """5.10: After CLEAR (which clears the revert marker), a new message.

    does NOT trigger COMMIT. All messages visible, new message appended.
    """

    async def test_no_commit_after_clear(
        self,
        async_client,
        server_state: ServerState,
    ) -> None:
        """Given: Session was STAGED then CLEARed (revert marker = None).

        When: _commit_revert is called.
        Then: No truncation occurs, all messages remain visible.
        """
        session_id = "test-commit-after-clear"
        _create_session_in_state(server_state, session_id)
        _setup_staged_session(server_state, session_id, num_messages=5, revert_at_index=2)

        # Simulate CLEAR: clear the revert marker
        session = server_state.sessions[session_id]
        server_state.sessions[session_id] = session.model_copy(update={"revert": None})
        # CLEAR also restores reverted_messages back to state.messages
        # (but for this test, we just need revert=None to skip COMMIT)

        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.truncate_messages.reset_mock()

        await _commit_revert(server_state, session_id)

        # COMMIT did NOT fire — no DB truncate
        session_pool.truncate_messages.assert_not_awaited()

        # All messages still present
        assert len(server_state.messages[session_id]) == 5

        # Marker still None
        assert server_state.sessions[session_id].revert is None


# =============================================================================
# 5.3: _truncate_agent_history unit tests
# =============================================================================


class TestTruncateAgentHistory:
    """Unit tests for the _truncate_agent_history helper."""

    async def test_truncate_finds_message_by_id(
        self,
        server_state: ServerState,
    ) -> None:
        """Given: Agent has 5 ChatMessages with known message IDs.

        When: _truncate_agent_history is called with message_id at index 2.
        Then: ChatMessages from index 2 onwards are removed.
        """
        session_pool = cast(Mock, server_state.pool.session_pool)
        chat_messages = [
            _make_chat_message(f"msg-{i:03d}", role="user" if i % 2 == 0 else "assistant")
            for i in range(5)
        ]

        session_state = Mock()
        agent = Mock()
        conversation = Mock()
        conversation.chat_messages = list(chat_messages)

        def _set_history(history: list[ChatMessage[Any]]) -> None:
            conversation.chat_messages = list(history)

        conversation.set_history = _set_history
        agent.conversation = conversation
        session_state.agent = agent
        session_pool.sessions.get_session = Mock(return_value=session_state)

        await _truncate_agent_history(session_pool, "test-session", "msg-002")

        assert len(conversation.chat_messages) == 2
        assert conversation.chat_messages[0].message_id == "msg-000"
        assert conversation.chat_messages[1].message_id == "msg-001"

    async def test_truncate_no_matching_message_id(
        self,
        server_state: ServerState,
    ) -> None:
        """Given: Agent has 3 ChatMessages, none matching the target ID.

        When: _truncate_agent_history is called with a non-existent ID.
        Then: No truncation occurs, all messages remain.
        """
        session_pool = cast(Mock, server_state.pool.session_pool)
        chat_messages = [_make_chat_message(f"msg-{i:03d}") for i in range(3)]

        session_state = Mock()
        agent = Mock()
        conversation = Mock()
        conversation.chat_messages = list(chat_messages)
        conversation.set_history = Mock()
        agent.conversation = conversation
        session_state.agent = agent
        session_pool.sessions.get_session = Mock(return_value=session_state)

        await _truncate_agent_history(session_pool, "test-session", "nonexistent")

        # set_history should NOT have been called
        conversation.set_history.assert_not_called()
        assert len(conversation.chat_messages) == 3

    async def test_truncate_no_session(
        self,
        server_state: ServerState,
    ) -> None:
        """Given: Session does not exist in session_pool.

        When: _truncate_agent_history is called.
        Then: No error, no-op.
        """
        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.sessions.get_session = Mock(return_value=None)

        # Should not raise
        await _truncate_agent_history(session_pool, "no-such-session", "msg-001")

    async def test_truncate_no_agent(
        self,
        server_state: ServerState,
    ) -> None:
        """Given: Session exists but has no agent.

        When: _truncate_agent_history is called.
        Then: No error, no-op.
        """
        session_pool = cast(Mock, server_state.pool.session_pool)
        session_state = Mock()
        session_state.agent = None
        session_pool.sessions.get_session = Mock(return_value=session_state)

        await _truncate_agent_history(session_pool, "test-session", "msg-001")

    async def test_truncate_agent_without_conversation(
        self,
        server_state: ServerState,
    ) -> None:
        """Given: Agent has no conversation attribute.

        When: _truncate_agent_history is called.
        Then: No error, no-op (AttributeError caught).
        """
        session_pool = cast(Mock, server_state.pool.session_pool)
        session_state = Mock()
        agent = Mock(spec=[])  # No attributes
        session_state.agent = agent
        session_pool.sessions.get_session = Mock(return_value=session_state)

        await _truncate_agent_history(session_pool, "test-session", "msg-001")


# =============================================================================
# 5.2: prompt_async path COMMIT test
# =============================================================================


class TestCommitInPromptAsync:
    """5.2: Verify COMMIT fires in the prompt_async path too."""

    async def test_prompt_async_triggers_commit(
        self,
        async_client,
        server_state: ServerState,
    ) -> None:
        """Given: Session in STAGED state.

        When: POST /prompt_async is called.
        Then: COMMIT fires (truncate_messages called) before message routing.
        """
        response = await async_client.post("/session", json={"title": "Async Commit"})
        session_id = response.json()["id"]

        # Add messages and set revert marker
        _setup_staged_session(server_state, session_id, num_messages=5, revert_at_index=2)

        session_pool = cast(Mock, server_state.pool.session_pool)
        session_pool.truncate_messages.reset_mock()

        request = MessageRequest(
            parts=[TextPartInput(text="new message after revert")],
            agent="default",
            message_id="msg_new_1",
        )
        response = await async_client.post(
            f"/session/{session_id}/prompt_async",
            json=request.model_dump(mode="json"),
        )
        assert response.status_code == 204

        # COMMIT should have fired
        session_pool.truncate_messages.assert_awaited()

        # Marker should be cleared
        assert server_state.sessions[session_id].revert is None

        # In-memory messages should be truncated to 2 (pre-revert) + 1 new
        remaining = server_state.messages[session_id]
        assert len(remaining) == 3  # 2 messages before revert point + 1 new user message
        assert remaining[0].info.id == "msg-000"
        assert remaining[1].info.id == "msg-001"
        # The new message created by prompt_async
        assert remaining[2].info.id == "msg_new_1"


# =============================================================================
# Helpers
# =============================================================================


def _create_session_in_state(state: ServerState, session_id: str) -> None:
    """Create a minimal Session in server_state.sessions."""
    now = now_ms()
    state.sessions[session_id] = Session(
        id=session_id,
        project_id="default",
        directory="/tmp",
        title="Test Session",
        version="1",
        time=TimeCreatedUpdated(created=now, updated=now),
    )
    state.messages.setdefault(session_id, [])
    state.reverted_messages.setdefault(session_id, [])
