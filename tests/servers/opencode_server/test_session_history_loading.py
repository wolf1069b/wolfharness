"""Test for session history loading during session switches.

This test verifies that conversation history is correctly loaded when
switching between sessions, preventing cross-session contamination.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from wolfharness.sessions.models import SessionData
from wolfharness_server.opencode_server.routes.session_routes import get_or_load_session


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from pathlib import Path

    from httpx import AsyncClient

    from wolfharness_server.opencode_server.state import ServerState


class TestSessionHistoryLoading:
    """Tests for conversation history loading during session switches."""

    async def test_session_switch_reloads_history(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
        tmp_project_dir: Path,
        event_capture,
    ):
        """When switching sessions, agent should reload conversation history."""
        # Setup: Create session A
        response_a = await async_client.post("/session", json={"title": "Session A"})
        assert response_a.status_code == 200
        session_a_id = response_a.json()["id"]

        # Verify agent has session A loaded
        assert server_state.agent.session_id == session_a_id

        # Setup: Create session B (this switches agent to session B)
        response_b = await async_client.post("/session", json={"title": "Session B"})
        assert response_b.status_code == 200
        session_b_id = response_b.json()["id"]

        # Verify agent now has session B loaded
        assert server_state.agent.session_id == session_b_id

        # Setup: Clear session A from memory cache
        del server_state.sessions[session_a_id]
        server_state.messages.pop(session_a_id, None)

        # Prepare session A data
        now = datetime.now(UTC)
        session_a_data = SessionData(
            session_id=session_a_id,
            agent_name="test-agent",
            cwd=str(tmp_project_dir),
            created_at=now,
            last_active=now,
            metadata={"title": "Session A"},
        )

        # Track if load_session was called
        load_session_calls: list[str] = []

        async def mock_load_session(sid: str) -> SessionData | None:
            load_session_calls.append(sid)
            if sid == session_a_id:
                server_state.agent.session_id = session_a_id
                # Set up conversation mock with empty list
                server_state.agent.conversation = Mock()
                server_state.agent.conversation.chat_messages = []
                return session_a_data
            return None

        server_state.agent.load_session = mock_load_session  # type: ignore[method-assign]

        # Disable SessionPool path to test shared-agent fallback
        server_state._pool.session_pool = None

        # ACTION: Load session A
        loaded_session = await get_or_load_session(server_state, session_a_id)

        # VERIFY: load_session should have been called
        assert session_a_id in load_session_calls
        assert loaded_session is not None
        assert loaded_session.id == session_a_id
        assert server_state.agent.session_id == session_a_id
        status_events = [
            event
            for event in event_capture.get_events_by_type("session.status")
            if event.properties.session_id == session_a_id
        ]
        idle_events = [
            event
            for event in event_capture.get_events_by_type("session.idle")
            if event.properties.session_id == session_a_id
        ]
        assert status_events
        assert idle_events

    async def test_cached_session_with_wrong_agent_session_gets_reloaded(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
        tmp_project_dir: Path,
    ):
        """With per-session agents, a cached session is returned immediately.

        In the old shared-agent model, the agent might have the wrong
        session loaded and need a reload. With per-session agents, each
        session has its own agent instance, so a cached session is always
        correct and no reload is needed.

        Note: In test environments without NativeAgentConfig, the fallback
        is the shared agent, but the cache-fast-path still applies.
        """
        # Create session A
        response_a = await async_client.post("/session", json={"title": "Session A"})
        session_a_id = response_a.json()["id"]

        # Create session B
        response_b = await async_client.post("/session", json={"title": "Session B"})
        response_b.json()["id"]

        # Add session A to cache
        from wolfharness_server.opencode_server.models import (
            Session,
            TimeCreatedUpdated,
        )
        from wolfharness_storage.opencode_provider import helpers

        now = int(datetime.now(UTC).timestamp() * 1000)
        session_a = Session(
            id=session_a_id,
            project_id=helpers.compute_project_id(str(tmp_project_dir)),
            directory=str(tmp_project_dir),
            title="Session A",
            version="1",
            time=TimeCreatedUpdated(created=now, updated=now),
        )
        server_state.sessions[session_a_id] = session_a
        server_state.messages[session_a_id] = []

        # ACTION: Get session A (cached, no reload needed)
        loaded_session = await get_or_load_session(server_state, session_a_id)

        # VERIFY: session is returned without reload
        assert loaded_session is not None
        assert loaded_session.id == session_a_id

    async def test_same_session_no_reload(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """If agent already has the correct session loaded, no reload needed."""
        # Create session A
        response_a = await async_client.post("/session", json={"title": "Session A"})
        session_a_id = response_a.json()["id"]

        # Verify agent has session A loaded
        assert server_state.agent.session_id == session_a_id

        load_session_called = False

        async def mock_load_session(sid: str) -> SessionData | None:
            nonlocal load_session_called
            load_session_called = True
            return None

        server_state.agent.load_session = mock_load_session  # type: ignore[method-assign]

        # ACTION: Get session A (already loaded)
        loaded_session = await get_or_load_session(server_state, session_a_id)

        # VERIFY: load_session should NOT have been called
        assert loaded_session is not None
        assert loaded_session.id == session_a_id
        assert not load_session_called

    async def test_input_provider_set_on_session_switch(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
        tmp_project_dir: Path,
    ):
        """With per-session agents, each session has its own input provider.

        When a session is loaded from storage, its agent is set up with the
        correct input provider for that session.
        """
        # Create session A
        response_a = await async_client.post("/session", json={"title": "Session A"})
        session_a_id = response_a.json()["id"]
        input_provider_a = server_state.ensure_input_provider(session_a_id)

        # Create session B
        response_b = await async_client.post("/session", json={"title": "Session B"})
        session_b_id = response_b.json()["id"]
        input_provider_b = server_state.ensure_input_provider(session_b_id)

        # Each session has its own input provider
        assert input_provider_a is not input_provider_b

        # Clear session A from cache to force a reload
        del server_state.sessions[session_a_id]
        server_state.messages.pop(session_a_id, None)

        # Prepare session A data
        now_dt = datetime.now(UTC)
        session_a_data = SessionData(
            session_id=session_a_id,
            agent_name="test-agent",
            cwd=str(tmp_project_dir),
            created_at=now_dt,
            last_active=now_dt,
            metadata={"title": "Session A"},
        )

        async def mock_load_session(sid: str) -> SessionData | None:
            if sid == session_a_id:
                # Set up conversation mock with empty list
                server_state.agent.conversation = Mock()
                server_state.agent.conversation.chat_messages = []
                return session_a_data
            return None

        server_state.agent.load_session = mock_load_session  # type: ignore[method-assign]

        # Disable SessionPool path to test shared-agent fallback
        server_state._pool.session_pool = None

        # ACTION: Switch back to session A
        await get_or_load_session(server_state, session_a_id)

        # VERIFY: The session has an input provider after reload
        reloaded_provider = server_state.ensure_input_provider(session_a_id)
        assert reloaded_provider is not None
        assert reloaded_provider.session_id == session_a_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
