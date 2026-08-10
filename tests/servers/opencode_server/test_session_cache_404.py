"""Tests for session cache hit in get_or_load_session.

Bug: When the TUI creates multiple sessions and then sends messages to each,
POST /session/{id}/message returns 404 because get_or_load_session requires
agent.session_id == session_id for cache hits. Only the last-created session
matches. For other sessions, it falls through to agent.load_session() which
reads from StorageManager — but create_session only saves to
pool.sessions.store (MemoryStorageProvider), which is a DIFFERENT backend.

This means sessions that exist in state.sessions cache are invisible to
agent.load_session(), causing 404 errors for all but the most recently
created session.

The fix: get_or_load_session must return cached sessions even when
agent.session_id doesn't match, and gracefully handle the case where
agent.load_session() returns None for a cached session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from wolfharness_server.opencode_server.models import (
    MessageWithParts,
    Session,
    TextPart,
    TimeCreatedUpdated,
    UserMessage,
)
from wolfharness_server.opencode_server.routes.session_routes import get_or_load_session


if TYPE_CHECKING:
    from wolfharness_server.opencode_server.state import ServerState


def _make_session(session_id: str, parent_id: str | None = None) -> Session:
    """Create a minimal Session object."""
    return Session(
        id=session_id,
        project_id="test-project",
        directory="/tmp/test",
        title="Test Session",
        version="1",
        time=TimeCreatedUpdated(created=0, updated=0),
        parent_id=parent_id,
    )


def _make_dummy_message(session_id: str) -> MessageWithParts:
    """Create a minimal message for populating session history."""
    from wolfharness_server.opencode_server.models import TimeCreated

    return MessageWithParts(
        info=UserMessage(
            id="msg_dummy",
            session_id=session_id,
            time=TimeCreated(created=0),
            agent="test",
        ),
        parts=[TextPart(id="p1", message_id="msg_dummy", session_id=session_id, text="hello")],
    )


class TestGetOrLoadSessionCacheHit:
    """Tests for get_or_load_session returning cached sessions."""

    @pytest.mark.asyncio
    async def test_cached_session_returned_when_agent_has_different_session(
        self,
        server_state: ServerState,
    ) -> None:
        """A session cached in state.sessions must be returned even when.

        agent.session_id points to a different session.

        This is the core TUI bug: create 4 sessions, only the last one
        matches agent.session_id, the other 3 return 404.
        """
        state = server_state

        # Set up: agent is bound to session_s4 (the last one created)
        session_s4 = _make_session("ses_s4")
        state.sessions["ses_s4"] = session_s4
        state.messages["ses_s4"] = [_make_dummy_message("ses_s4")]
        state.agent.session_id = "ses_s4"

        # Also create s1, s2, s3 in cache (like create_session does)
        session_s1 = _make_session("ses_s1")
        session_s2 = _make_session("ses_s2")
        session_s3 = _make_session("ses_s3")
        state.sessions["ses_s1"] = session_s1
        state.sessions["ses_s2"] = session_s2
        state.sessions["ses_s3"] = session_s3
        state.messages["ses_s1"] = [_make_dummy_message("ses_s1")]
        state.messages["ses_s2"] = [_make_dummy_message("ses_s2")]
        state.messages["ses_s3"] = [_make_dummy_message("ses_s3")]

        # agent.load_session returns None (session saved to MemoryStorageProvider,
        # not to StorageManager that agent.load_session reads from)
        state.agent.load_session = AsyncMock(return_value=None)  # type: ignore[method-assign]

        # Request s1 — should NOT return None even though agent.session_id != s1
        result = await get_or_load_session(state, "ses_s1")
        assert result is not None, (
            "get_or_load_session returned None for cached session ses_s1. "
            "Sessions in state.sessions cache must be returned even when "
            "agent.session_id doesn't match and agent.load_session returns None."
        )
        assert result.id == "ses_s1"

        # Request s2
        result = await get_or_load_session(state, "ses_s2")
        assert result is not None, "Cached session ses_s2 should be returned"
        assert result.id == "ses_s2"

        # Request s3
        result = await get_or_load_session(state, "ses_s3")
        assert result is not None, "Cached session ses_s3 should be returned"
        assert result.id == "ses_s3"

    @pytest.mark.asyncio
    async def test_cached_session_with_empty_messages_when_load_fails(
        self,
        server_state: ServerState,
    ) -> None:
        """When a cached session exists but agent.load_session returns None,.

        the session should still be returned with its existing (possibly empty)

        message list.
        """
        state = server_state
        session_id = "ses_newly_created"

        # Simulate create_session: adds to cache with messages
        session = _make_session(session_id)
        state.sessions[session_id] = session
        state.messages[session_id] = [_make_dummy_message(session_id)]

        # Agent is bound to a different session
        state.agent.session_id = "ses_other"

        # agent.load_session returns None (session not in StorageManager)
        state.agent.load_session = AsyncMock(return_value=None)  # type: ignore[method-assign]

        result = await get_or_load_session(state, session_id)
        assert result is not None, (
            "Newly created session should be returned from cache even when "
            "agent.load_session returns None (StorageManager backend mismatch)"
        )
        assert result.id == session_id

    @pytest.mark.asyncio
    async def test_non_cached_session_returns_none(
        self,
        server_state: ServerState,
    ) -> None:
        """A session NOT in cache and NOT in storage should return None."""
        state = server_state
        state.agent.load_session = AsyncMock(return_value=None)  # type: ignore[method-assign]

        result = await get_or_load_session(state, "ses_nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_cached_session_reloads_history_when_available(
        self,
        server_state: ServerState,
    ) -> None:
        """When a cached session exists with messages in memory, it is.

        returned directly without calling load_session.

        With per-session agents, each session has its own agent instance
        that already owns the correct conversation history. If a session
        is cached in memory (in state.sessions and state.messages), it
        doesn't need to be reloaded from storage.
        """
        state = server_state
        session_id = "ses_cached_with_history"

        session = _make_session(session_id)
        state.sessions[session_id] = session
        state.messages[session_id] = [_make_dummy_message(session_id)]

        result = await get_or_load_session(state, session_id)
        assert result is not None
        assert result.id == session_id

        # With per-session agents, cached sessions are returned directly
        # without calling load_session — the session agent already has
        # the correct history.
        assert state.agent.load_session.call_count == 0


class TestConcurrentSessionCreation:
    """Integration-style tests for the TUI session creation pattern."""

    @pytest.mark.asyncio
    async def test_multiple_sessions_then_messages(
        self,
        async_client,  # type: ignore[valid-type]
        server_state: ServerState,
    ) -> None:
        """Simulate TUI pattern: create 4 sessions, then send message to each.

        This tests the full HTTP flow, not just get_or_load_session.
        """
        # Create 4 sessions
        session_ids = []
        for i in range(4):
            response = await async_client.post(
                "/session",
                json={"title": f"Test Session {i}"},
            )
            assert response.status_code == 200, f"Session {i} creation failed"
            session_ids.append(response.json()["id"])

        # Send message to each session sequentially
        for i, sid in enumerate(session_ids):
            response = await async_client.post(
                f"/session/{sid}/message",
                json={"parts": [{"text": f"hello {i}"}]},
            )
            assert response.status_code != 404, (
                f"Message to session {sid} (index {i}) returned 404. "
                "Sessions created via POST /session should be immediately "
                "accessible for messaging."
            )
