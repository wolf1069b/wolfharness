"""Unit tests for soft-hide message filtering via ``session.revert``.

Tests that ``get_messages_for_session()`` correctly filters out messages
at and after the revert point when ``session.revert`` is set, while
preserving all messages when no revert is active.

Covers tasks 6.1-6.4 from session-revert-stage-clear-commit.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from wolfharness.messaging.messages import ChatMessage
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
from wolfharness_server.opencode_server.models.common import TimeCreatedUpdated
from wolfharness_server.opencode_server.models.session import Session, SessionRevert
from wolfharness_server.opencode_server.opencode_message_bridge import (
    _apply_revert_filter,
    get_messages_for_session,
)
from wolfharness_server.opencode_server.state import ServerState


pytestmark = pytest.mark.unit


# =============================================================================
# Helpers
# =============================================================================


def _make_user_message(session_id: str, message_id: str, text: str) -> MessageWithParts:
    """Create a user ``MessageWithParts`` with a text part."""
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
    """Create an assistant ``MessageWithParts`` with a text part."""
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


def _make_messages(session_id: str, count: int) -> list[MessageWithParts]:
    """Create ``count`` alternating user/assistant messages."""
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
    return messages


def _make_session(
    session_id: str,
    *,
    revert: SessionRevert | None = None,
    parent_id: str | None = None,
) -> Session:
    """Create a minimal ``Session`` model for testing."""
    now = now_ms()
    return Session(
        id=session_id,
        project_id="default",
        directory="/tmp",
        title="Test Session",
        version="1",
        time=TimeCreatedUpdated(created=now, updated=now),
        parent_id=parent_id,
        revert=revert,
    )


def _make_mock_state(
    session_id: str,
    messages: list[MessageWithParts],
    session: Session | None,
) -> ServerState:
    """Create a minimal mock ``ServerState`` for in-memory path tests.

    The mock has ``session_pool = None`` so only the in-memory cache path
    is exercised.
    """
    state = Mock(spec=ServerState)
    state.messages = {session_id: messages}
    state.sessions = {session_id: session} if session is not None else {}
    # Pool with session_pool=None forces the in-memory fallback path.
    pool_mock = Mock()
    pool_mock.session_pool = None
    state.pool = pool_mock
    state.pool_or_none = pool_mock
    return state


# =============================================================================
# 6.3: Unit tests for soft-hide filtering
# =============================================================================


class TestSoftHideRevertFilter:
    """Tests for soft-hide filtering via ``session.revert``."""

    async def test_revert_set_filters_messages_at_and_after_revert_point(self):
        """When ``session.revert`` is set, messages at and after the revert.

        message_id are excluded; messages before it are included.
        """
        session_id = "test-session"
        messages = _make_messages(session_id, count=6)
        # Revert to msg-003 (index 3) — should hide messages 3, 4, 5
        revert = SessionRevert(message_id="msg-003")
        session = _make_session(session_id, revert=revert)
        state = _make_mock_state(session_id, messages, session)

        result = await get_messages_for_session(state, session_id)

        assert len(result) == 3
        result_ids = [m.info.id for m in result]
        assert result_ids == ["msg-000", "msg-001", "msg-002"]
        # The revert message and everything after must be excluded
        assert "msg-003" not in result_ids
        assert "msg-004" not in result_ids
        assert "msg-005" not in result_ids

    async def test_revert_none_returns_all_messages(self):
        """When ``session.revert`` is ``None``, all messages are returned.

        without filtering.
        """
        session_id = "test-session"
        messages = _make_messages(session_id, count=5)
        session = _make_session(session_id, revert=None)
        state = _make_mock_state(session_id, messages, session)

        result = await get_messages_for_session(state, session_id)

        assert len(result) == 5
        result_ids = [m.info.id for m in result]
        assert result_ids == [m.info.id for m in messages]

    async def test_revert_message_id_not_found_returns_all_messages(self):
        """When ``session.revert.message_id`` doesn't match any message,.

        all messages are returned (defensive fallback).
        """
        session_id = "test-session"
        messages = _make_messages(session_id, count=4)
        revert = SessionRevert(message_id="nonexistent-msg-id")
        session = _make_session(session_id, revert=revert)
        state = _make_mock_state(session_id, messages, session)

        result = await get_messages_for_session(state, session_id)

        assert len(result) == 4
        result_ids = [m.info.id for m in result]
        assert result_ids == [m.info.id for m in messages]

    async def test_revert_at_first_message_hides_all(self):
        """Reverting to the first message hides all messages."""
        session_id = "test-session"
        messages = _make_messages(session_id, count=3)
        revert = SessionRevert(message_id="msg-000")
        session = _make_session(session_id, revert=revert)
        state = _make_mock_state(session_id, messages, session)

        result = await get_messages_for_session(state, session_id)

        assert len(result) == 0

    async def test_revert_at_last_message_hides_only_last(self):
        """Reverting to the last message hides only that message."""
        session_id = "test-session"
        messages = _make_messages(session_id, count=4)
        revert = SessionRevert(message_id="msg-003")
        session = _make_session(session_id, revert=revert)
        state = _make_mock_state(session_id, messages, session)

        result = await get_messages_for_session(state, session_id)

        assert len(result) == 3
        result_ids = [m.info.id for m in result]
        assert result_ids == ["msg-000", "msg-001", "msg-002"]


# =============================================================================
# 6.4: Edge case — empty messages with revert set
# =============================================================================


class TestSoftHideEmptyMessages:
    """Tests for soft-hide with empty message list and revert set."""

    async def test_empty_messages_with_revert_returns_empty_list(self):
        """When ``state.messages[session_id]`` is empty and ``session.revert``.

        is set, the function returns an empty list, not crash.
        """
        session_id = "test-session"
        revert = SessionRevert(message_id="msg-042")
        session = _make_session(session_id, revert=revert)
        state = _make_mock_state(session_id, [], session)

        result = await get_messages_for_session(state, session_id)

        assert result == []

    async def test_missing_session_id_with_revert_returns_empty_list(self):
        """When ``state.messages`` doesn't have the session_id key at all.

        and ``session.revert`` is set, the function returns an empty list.
        """
        session_id = "test-session"
        revert = SessionRevert(message_id="msg-042")
        session = _make_session(session_id, revert=revert)
        state = Mock(spec=ServerState)
        state.messages = {}
        state.sessions = {session_id: session}
        pool_mock = Mock()
        pool_mock.session_pool = None
        state.pool = pool_mock
        state.pool_or_none = pool_mock

        result = await get_messages_for_session(state, session_id)

        assert result == []

    async def test_session_not_cached_returns_all_messages(self):
        """When the session is not in ``state.sessions`` (cached_session is.

        None), no filtering is applied and all messages are returned.
        """
        session_id = "test-session"
        messages = _make_messages(session_id, count=3)
        state = _make_mock_state(session_id, messages, session=None)

        result = await get_messages_for_session(state, session_id)

        assert len(result) == 3


# =============================================================================
# Direct tests for _apply_revert_filter helper
# =============================================================================


class TestApplyRevertFilterHelper:
    """Direct unit tests for the ``_apply_revert_filter`` helper."""

    def test_none_session_returns_messages_unchanged(self):
        """A ``None`` session means no revert info — return messages as-is."""
        messages = _make_messages("s1", count=3)
        result = _apply_revert_filter(None, messages)
        assert result is messages

    def test_none_revert_returns_messages_unchanged(self):
        """``session.revert is None`` means no revert active."""
        messages = _make_messages("s1", count=3)
        session = _make_session("s1", revert=None)
        result = _apply_revert_filter(session, messages)
        assert result is messages

    def test_revert_found_returns_truncated_list(self):
        """Revert message found at index 2 — return messages[:2]."""
        messages = _make_messages("s1", count=5)
        session = _make_session("s1", revert=SessionRevert(message_id="msg-002"))
        result = _apply_revert_filter(session, messages)
        assert len(result) == 2
        assert [m.info.id for m in result] == ["msg-000", "msg-001"]

    def test_revert_not_found_returns_messages_unchanged(self):
        """Revert message_id not in list — defensive fallback."""
        messages = _make_messages("s1", count=3)
        session = _make_session("s1", revert=SessionRevert(message_id="missing"))
        result = _apply_revert_filter(session, messages)
        assert result is messages

    def test_empty_messages_with_revert_returns_empty(self):
        """Empty message list with revert set — returns empty list."""
        session = _make_session("s1", revert=SessionRevert(message_id="msg-000"))
        result = _apply_revert_filter(session, [])
        assert result == []

    def test_revert_filter_does_not_mutate_input(self):
        """The filter must not modify the original message list."""
        messages = _make_messages("s1", count=4)
        original_len = len(messages)
        session = _make_session("s1", revert=SessionRevert(message_id="msg-002"))
        _ = _apply_revert_filter(session, messages)
        assert len(messages) == original_len


# =============================================================================
# 6.5: Soft-hide with SessionPool-backed messages
# =============================================================================


class TestSoftHideSessionPool:
    """Tests for soft-hide filtering when messages come from ``session_pool.get_messages()``."""

    async def test_revert_filter_with_non_empty_session_pool(
        self,
    ) -> None:
        """When ``state.messages`` is empty but ``session_pool.get_messages()``.

        returns ChatMessage objects, the revert filter is applied to the converted
        messages. Messages at and after the revert point are excluded.
        """
        session_id = "test-session"

        # Create ChatMessage objects to be returned by session_pool.get_messages()
        chat_messages: list[ChatMessage[str]] = []
        for i in range(5):
            role = "user" if i % 2 == 0 else "assistant"
            chat_messages.append(
                ChatMessage[str](
                    content=f"Message {i}",
                    role=role,
                    message_id=f"msg-{i:03d}",
                    timestamp=datetime(2024, 1, 1 + i),
                    name="test-agent",
                )
            )

        revert = SessionRevert(message_id="msg-002")
        session = _make_session(session_id, revert=revert)

        # Mock state: empty in-memory messages + session_pool that returns ChatMessages
        state = Mock(spec=ServerState)
        state.messages = {}
        state.sessions = {session_id: session}
        state.agent = Mock()
        state.agent.name = "test-agent"
        state.agent.model_name = None
        state.resolve_default_model_info = Mock(return_value=("default", "wolfharness"))
        state.working_dir = "/tmp"

        pool_mock = Mock()
        pool_mock.session_pool = Mock()
        pool_mock.session_pool.get_messages = AsyncMock(return_value=chat_messages)
        pool_mock.session_pool.sessions = Mock()
        pool_mock.session_pool.sessions.get_session_agent = Mock(return_value=None)
        state.pool = pool_mock
        state.pool_or_none = pool_mock

        result = await get_messages_for_session(state, session_id)

        # msg and msg-003, msg-004 should be hidden
        assert len(result) == 2
        result_ids = [m.info.id for m in result]
        assert result_ids == ["msg-000", "msg-001"]

        # Messages at and after revert point must be excluded
        assert "msg-002" not in result_ids
        assert "msg-003" not in result_ids
        assert "msg-004" not in result_ids

    async def test_session_pool_with_no_revert_returns_all(
        self,
    ) -> None:
        """When session.revert is None, all ChatMessage objects from.

        session_pool.get_messages() are returned unfiltered.
        """
        session_id = "test-session"

        chat_messages: list[ChatMessage[str]] = []
        for i in range(3):
            role = "user" if i % 2 == 0 else "assistant"
            chat_messages.append(
                ChatMessage[str](
                    content=f"Message {i}",
                    role=role,
                    message_id=f"msg-{i:03d}",
                    timestamp=datetime(2024, 1, 1 + i),
                    name="test-agent",
                )
            )

        session = _make_session(session_id, revert=None)
        state = Mock(spec=ServerState)
        state.messages = {}
        state.sessions = {session_id: session}
        state.agent = Mock()
        state.agent.name = "test-agent"
        state.agent.model_name = None
        state.resolve_default_model_info = Mock(return_value=("default", "wolfharness"))
        state.working_dir = "/tmp"

        pool_mock = Mock()
        pool_mock.session_pool = Mock()
        pool_mock.session_pool.get_messages = AsyncMock(return_value=chat_messages)
        pool_mock.session_pool.sessions = Mock()
        pool_mock.session_pool.sessions.get_session_agent = Mock(return_value=None)
        state.pool = pool_mock
        state.pool_or_none = pool_mock

        result = await get_messages_for_session(state, session_id)

        assert len(result) == 3
        result_ids = [m.info.id for m in result]
        assert result_ids == [f"msg-{i:03d}" for i in range(3)]
