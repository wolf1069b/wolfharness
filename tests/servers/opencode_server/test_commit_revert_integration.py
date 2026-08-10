"""Real DB integration tests for ``_commit_revert`` + ``SQLModelProvider``.

The existing ``test_commit_revert.py`` mocks ``truncate_messages`` everywhere.
Oracle review identified this as the #1 gap: "the core DB truncation is never
verified end-to-end." These tests wire ``_commit_revert`` to a real
``SQLModelProvider`` with in-memory SQLite and verify DB rows are actually
deleted by querying the database directly.

Key differences from ``test_commit_revert.py``:
- ``truncate_messages`` is NOT mocked — the real ``SQLModelProvider`` executes.
- DB state is verified via direct SQL queries (``select(Message)``), not mock
  assertions like ``assert_awaited_once_with``.
- The SQL layer is real in-memory SQLite, not a mock.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from wolfharness.messaging import ChatMessage
from wolfharness.storage import StorageManager
from wolfharness.utils.time_utils import now_ms
from wolfharness_config.storage import SQLStorageConfig, StorageConfig
from wolfharness_server.opencode_server.models import (
    AssistantMessage,
    MessagePath,
    MessageTime,
    MessageWithParts,
    Session,
    SessionRevert,
    TextPart,
    TimeCreated,
    TimeCreatedUpdated,
    UserMessage,
)
from wolfharness_server.opencode_server.routes.message_routes import _commit_revert
from wolfharness_server.opencode_server.routes.session_routes import (
    RevertRequest,
    revert_session,
)
from wolfharness_storage.sql_provider.models import Message
from wolfharness_storage.sql_provider.sql_provider import SQLModelProvider


if TYPE_CHECKING:
    from pathlib import Path

    from wolfharness_server.opencode_server.state import ServerState


pytestmark = pytest.mark.integration


# =============================================================================
# Helpers
# =============================================================================


def _make_chat_message(
    *,
    msg_id: str,
    session_id: str,
    timestamp: datetime,
    role: str = "user",
    content: str = "",
) -> ChatMessage[str]:
    """Build a ``ChatMessage`` suitable for ``provider.log_message``."""
    return ChatMessage(
        content=content,
        role=role,  # type: ignore[arg-type]
        message_id=msg_id,
        session_id=session_id,
        timestamp=timestamp,
    )


def _make_user_message_with_parts(
    session_id: str,
    message_id: str,
    text: str,
) -> MessageWithParts:
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


def _make_assistant_message_with_parts(
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


def _create_session_in_state(state: ServerState, session_id: str) -> None:
    """Create a minimal ``Session`` in ``state.sessions``."""
    now = now_ms()
    state.sessions[session_id] = Session(
        id=session_id,
        project_id="default",
        directory="/tmp",
        title="Integration Test Session",
        version="1",
        time=TimeCreatedUpdated(created=now, updated=now),
    )
    state.messages.setdefault(session_id, [])
    state.reverted_messages.setdefault(session_id, [])


async def _insert_messages_into_db(
    provider: SQLModelProvider,
    session_id: str,
    num_messages: int,
    base_time: datetime,
) -> list[ChatMessage[str]]:
    """Insert ``num_messages`` alternating user/assistant messages into the DB.

    Returns the list of inserted ``ChatMessage`` objects (for ID reference).
    """
    messages: list[ChatMessage[str]] = []
    for i in range(num_messages):
        role = "user" if i % 2 == 0 else "assistant"
        msg = _make_chat_message(
            msg_id=f"db-msg-{i:03d}",
            session_id=session_id,
            timestamp=base_time + timedelta(seconds=i),
            role=role,
            content=f"DB message {i} ({role})",
        )
        messages.append(msg)
        await provider.log_message(message=msg)
    return messages


async def _insert_single_message_into_db(
    provider: SQLModelProvider,
    msg_id: str,
    session_id: str,
    timestamp: datetime,
    role: str = "user",
    content: str = "",
) -> ChatMessage[str]:
    """Insert a single message into the DB and return it."""
    msg = _make_chat_message(
        msg_id=msg_id,
        session_id=session_id,
        timestamp=timestamp,
        role=role,
        content=content,
    )
    await provider.log_message(message=msg)
    return msg


async def _remaining_db_ids(provider: SQLModelProvider, session_id: str) -> list[str]:
    """Return the IDs of messages still present in the DB for a session.

    Ordered by timestamp ascending, then by ID ascending — matching the
    real ``get_session_messages`` ordering.
    """
    async with AsyncSession(provider.engine) as session:
        result = await session.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.timestamp.asc(), Message.id.asc())  # type: ignore[arg-type]
        )
        return [m.id for m in result.scalars().all()]


def _populate_state_messages(
    state: ServerState,
    session_id: str,
    num_messages: int,
) -> list[MessageWithParts]:
    """Populate ``state.messages[session_id]`` with ``MessageWithParts``.

    Uses the same ``db-msg-NNN`` ID scheme as ``_insert_messages_into_db``
    so in-memory and DB messages share the same IDs.
    """
    messages: list[MessageWithParts] = []
    for i in range(num_messages):
        msg_id = f"db-msg-{i:03d}"
        if i % 2 == 0:
            msg = _make_user_message_with_parts(session_id, msg_id, f"User message {i}")
        else:
            msg = _make_assistant_message_with_parts(
                session_id, msg_id, f"db-msg-{i - 1:03d}", f"Assistant response {i}"
            )
        messages.append(msg)
    state.messages[session_id] = list(messages)
    return messages


def _set_revert_marker(
    state: ServerState,
    session_id: str,
    revert_msg_id: str,
) -> None:
    """Set the ``SessionRevert`` marker on the session."""
    session = state.sessions[session_id]
    revert_info = SessionRevert(message_id=revert_msg_id)
    state.sessions[session_id] = session.model_copy(update={"revert": revert_info})


def _wire_real_truncate(
    state: ServerState,
    storage_manager: StorageManager,
) -> None:
    """Wire ``state.pool.session_pool.truncate_messages`` to the real storage.

    This replaces the mock ``AsyncMock(return_value=0)`` set by the conftest
    with a real async function that delegates to ``StorageManager.truncate_messages``,
    which in turn calls ``SQLModelProvider.truncate_messages`` — hitting the
    real in-memory SQLite database.
    """
    session_pool = cast(Mock, state.pool.session_pool)

    async def _real_truncate(session_id: str, up_to_message_id: str) -> int:
        return await storage_manager.truncate_messages(session_id, up_to_message_id)

    session_pool.truncate_messages = _real_truncate


def _wire_session_controller_noop(state: ServerState) -> None:
    """Replace ``state.session_controller`` with a mock that reports idle.

    ``_ensure_session_idle`` checks ``session_state.current_run_id``; setting
    it to ``None`` makes the function a no-op. ``get_or_load_session`` checks
    ``session_controller.get_session(session_id)`` for the fast path; returning
    a non-None mock ensures the cached session is returned.

    Uses real ``asyncio.Queue`` objects for ``prompt_queue`` and ``feedback_queue``
    so that ``get_nowait()`` raises ``asyncio.QueueEmpty`` on empty queues —
    matching production behavior. Using ``Mock()`` here would cause
    ``get_nowait()`` to return a truthy Mock, never raising QueueEmpty,
    and the queue-draining loop in ``revert_session`` would hang forever.
    """
    mock_session_state = Mock()
    mock_session_state.current_run_id = None
    mock_session_state.prompt_queue = asyncio.Queue()
    mock_session_state.feedback_queue = asyncio.Queue()

    state.session_controller = Mock()
    state.session_controller.get_session = Mock(return_value=mock_session_state)
    state.session_controller.cancel_session_pending_questions = Mock(return_value=[])


# =============================================================================
# Test 1: _commit_revert deletes DB rows via real SQLModelProvider
# =============================================================================


class TestCommitRevertRealDb:
    """Verify ``_commit_revert`` actually deletes rows from the DB when.

    wired to a real ``SQLModelProvider`` with in-memory SQLite.
    """

    async def test_commit_revert_deletes_db_rows(
        self,
        server_state: ServerState,
        tmp_path: Path,
    ) -> None:
        """Given: 5 messages in real DB, session in STAGED state (revert at index 2).

        When: ``_commit_revert`` is called.
        Then: DB rows from index 2 onwards are deleted, in-memory messages
              are truncated, and the revert marker is cleared.
        """
        # --- Setup: real SQLModelProvider with in-memory SQLite ---
        db_path = tmp_path / "test_commit_revert.db"
        sql_config = SQLStorageConfig(url=f"sqlite:///{db_path}", auto_migration=False)
        provider = SQLModelProvider(sql_config)
        storage_config = StorageConfig(providers=[sql_config])
        storage_manager = StorageManager(config=storage_config)

        session_id = "s-commit-real-db"
        base_time = datetime(2026, 7, 21, 10, 0, tzinfo=UTC)

        async with storage_manager:
            # --- Insert 5 real messages into DB ---
            await _insert_messages_into_db(provider, session_id, 5, base_time)

            # Verify all 5 are in the DB
            db_ids = await _remaining_db_ids(provider, session_id)
            assert db_ids == ["db-msg-000", "db-msg-001", "db-msg-002", "db-msg-003", "db-msg-004"]

            # --- Wire ServerState with real storage ---
            _wire_real_truncate(server_state, storage_manager)

            # --- Set up session in STAGED state ---
            _create_session_in_state(server_state, session_id)
            _populate_state_messages(server_state, session_id, 5)
            revert_msg_id = "db-msg-002"
            _set_revert_marker(server_state, session_id, revert_msg_id)

            # Store reverted messages (as revert_session does)
            server_state.reverted_messages[session_id] = list(server_state.messages[session_id][2:])

            # --- Call _commit_revert ---
            await _commit_revert(server_state, session_id)

            # --- Verify DB rows are ACTUALLY deleted ---
            db_remaining = await _remaining_db_ids(provider, session_id)
            assert db_remaining == ["db-msg-000", "db-msg-001"]

            # --- Verify in-memory messages are truncated ---
            mem_remaining = server_state.messages[session_id]
            assert len(mem_remaining) == 2
            assert mem_remaining[0].info.id == "db-msg-000"
            assert mem_remaining[1].info.id == "db-msg-001"

            # --- Verify revert marker is cleared ---
            assert server_state.sessions[session_id].revert is None

            # --- Verify reverted_messages is cleared ---
            assert session_id not in server_state.reverted_messages


# =============================================================================
# Test 2: Full STAGE → COMMIT flow with real DB
# =============================================================================


class TestStageCommitFlowRealDb:
    """Verify the full STAGE → COMMIT flow with a real DB.

    STAGE (``revert_session``) sets a soft marker — messages remain in the DB.
    COMMIT (``_commit_revert``) deletes messages from the DB.
    """

    async def test_stage_then_commit_with_real_db(
        self,
        server_state: ServerState,
        tmp_path: Path,
    ) -> None:
        """Given: 5 messages in real DB.

        When: ``revert_session`` (STAGE) is called at message index 2,
              then ``_commit_revert`` (COMMIT) is called,
              then a new message is inserted into the DB.
        Then: After STAGE, DB still has all 5 messages.
              After COMMIT, DB has only messages 0-1.
              After new message insert, DB has messages 0-1 + new message.
        """
        # --- Setup: real SQLModelProvider with in-memory SQLite ---
        db_path = tmp_path / "test_stage_commit.db"
        sql_config = SQLStorageConfig(url=f"sqlite:///{db_path}", auto_migration=False)
        provider = SQLModelProvider(sql_config)
        storage_config = StorageConfig(providers=[sql_config])
        storage_manager = StorageManager(config=storage_config)

        session_id = "s-stage-commit"
        base_time = datetime(2026, 7, 21, 11, 0, tzinfo=UTC)

        async with storage_manager:
            # --- Insert 5 real messages into DB ---
            await _insert_messages_into_db(provider, session_id, 5, base_time)

            # Verify all 5 are in the DB
            db_ids = await _remaining_db_ids(provider, session_id)
            assert len(db_ids) == 5

            # --- Wire ServerState ---
            _wire_real_truncate(server_state, storage_manager)
            _wire_session_controller_noop(server_state)

            # --- Set up session and in-memory messages ---
            _create_session_in_state(server_state, session_id)
            _populate_state_messages(server_state, session_id, 5)

            # --- STAGE: Call revert_session ---
            revert_msg_id = "db-msg-002"
            request = RevertRequest(message_id=revert_msg_id)
            updated_session = await revert_session(session_id, request, server_state)

            # --- Verify STAGE did NOT delete from DB ---
            db_after_stage = await _remaining_db_ids(provider, session_id)
            assert db_after_stage == [
                "db-msg-000",
                "db-msg-001",
                "db-msg-002",
                "db-msg-003",
                "db-msg-004",
            ], "STAGE must NOT delete messages from the DB"

            # --- Verify STAGE set the revert marker ---
            assert updated_session.revert is not None
            assert updated_session.revert.message_id == revert_msg_id
            assert server_state.sessions[session_id].revert is not None
            assert server_state.sessions[session_id].revert.message_id == revert_msg_id

            # --- COMMIT: Call _commit_revert ---
            await _commit_revert(server_state, session_id)

            # --- Verify COMMIT deleted from DB ---
            db_after_commit = await _remaining_db_ids(provider, session_id)
            assert db_after_commit == ["db-msg-000", "db-msg-001"], (
                "COMMIT must delete messages from the revert point onwards"
            )

            # --- Verify revert marker is cleared after COMMIT ---
            assert server_state.sessions[session_id].revert is None

            # --- Simulate new message after COMMIT ---
            new_msg_time = base_time + timedelta(seconds=10)
            await _insert_single_message_into_db(
                provider,
                msg_id="db-msg-new",
                session_id=session_id,
                timestamp=new_msg_time,
                role="user",
                content="New message after revert",
            )

            # --- Verify new message exists in DB alongside surviving messages ---
            db_final = await _remaining_db_ids(provider, session_id)
            assert db_final == ["db-msg-000", "db-msg-001", "db-msg-new"]

            # --- Verify the reverted messages are NOT in the DB ---
            assert "db-msg-002" not in db_final
            assert "db-msg-003" not in db_final
            assert "db-msg-004" not in db_final
