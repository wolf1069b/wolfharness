"""Test for session input_provider isolation with per-session agents.

With per-session agents, each session has its own agent instance with its
own input provider. This resolves the old "session switch" bug where
switching to an existing session caused the shared agent's input_provider
to point to the wrong session.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

import pytest

from wolfharness.sessions.models import SessionData
from wolfharness_server.opencode_server.input_provider import OpenCodeInputProvider
from wolfharness_server.opencode_server.routes.session_routes import get_or_load_session


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from pathlib import Path

    from httpx import AsyncClient

    from wolfharness_server.opencode_server.state import ServerState


class TestSessionSwitchInputProvider:
    """Tests for input_provider handling with per-session agents."""

    async def test_create_session_sets_input_provider(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """Creating a session should set input_provider for that session.

        This is the baseline - create_session() correctly sets up input_provider.
        """
        # Create a session
        response = await async_client.post("/session", json={"title": "Test Session"})
        assert response.status_code == 200
        session_id = response.json()["id"]

        # Verify input_provider can be obtained for this session
        input_provider = server_state.ensure_input_provider(session_id)
        assert isinstance(input_provider, OpenCodeInputProvider)
        assert input_provider.session_id == session_id

    async def test_get_or_load_session_preserves_input_provider(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
        tmp_project_dir: Path,
    ):
        """Loading an existing session via get_or_load_session preserves.

        the session's input provider.

        With per-session agents, each session has its own agent with its
        own input_provider. Loading session A doesn't affect session B's
        input provider.
        """
        # Setup: Create session A with input_provider
        response_a = await async_client.post("/session", json={"title": "Session A"})
        assert response_a.status_code == 200
        session_a_id = response_a.json()["id"]

        # Verify session A's input_provider is available
        input_provider_a = server_state.ensure_input_provider(session_a_id)
        assert isinstance(input_provider_a, OpenCodeInputProvider)

        # Setup: Create session B
        response_b = await async_client.post("/session", json={"title": "Session B"})
        assert response_b.status_code == 200
        session_b_id = response_b.json()["id"]

        # Verify session B's input_provider is available
        input_provider_b = server_state.ensure_input_provider(session_b_id)
        assert isinstance(input_provider_b, OpenCodeInputProvider)

        # Setup: Mock agent.load_session to return session A data
        now = datetime.now(UTC)
        session_a_data = SessionData(
            session_id=session_a_id,
            agent_name="test-agent",
            cwd=str(tmp_project_dir),
            created_at=now,
            last_active=now,
            metadata={"title": "Session A"},
        )

        # Clear session A from memory to simulate "switching to existing session"
        del server_state.sessions[session_a_id]

        # Mock load_session to return the session data
        server_state.agent.load_session = AsyncMock(return_value=session_a_data)  # type: ignore[method-assign]

        # Also need to mock conversation.chat_messages for the conversion
        server_state.agent.conversation = Mock()
        server_state.agent.conversation.chat_messages = []

        # ACTION: Call get_or_load_session to "switch" to session A
        loaded_session = await get_or_load_session(server_state, session_a_id)

        # Verify session was loaded
        assert loaded_session is not None
        assert loaded_session.id == session_a_id

        # Both sessions' input providers should still be available
        provider_a_after = server_state.ensure_input_provider(session_a_id)
        provider_b_after = server_state.ensure_input_provider(session_b_id)
        assert provider_a_after.session_id == session_a_id
        assert provider_b_after.session_id == session_b_id

    async def test_input_provider_per_session_isolation(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
        tmp_project_dir: Path,
    ):
        """With per-session agents, each session's input provider is independent.

        This verifies that the old "input provider session_id mismatch" bug
        is resolved by the per-session agent architecture: each session has
        its own agent with its own input provider, so there's no cross-session
        contamination.
        """
        # Create session A
        response_a = await async_client.post("/session", json={"title": "Session A"})
        session_a_id = response_a.json()["id"]

        # Create session B
        response_b = await async_client.post("/session", json={"title": "Session B"})
        session_b_id = response_b.json()["id"]

        # Each session has its own input provider with the correct session_id
        input_provider_a = server_state.ensure_input_provider(session_a_id)
        input_provider_b = server_state.ensure_input_provider(session_b_id)

        assert input_provider_a.session_id == session_a_id
        assert input_provider_b.session_id == session_b_id
        assert input_provider_a is not input_provider_b

        # Clear session A from memory
        del server_state.sessions[session_a_id]

        # Mock load_session
        now = datetime.now(UTC)
        session_a_data = SessionData(
            session_id=session_a_id,
            agent_name="test-agent",
            cwd=str(tmp_project_dir),
            created_at=now,
            last_active=now,
            metadata={"title": "Session A"},
        )
        server_state.agent.load_session = AsyncMock(return_value=session_a_data)  # type: ignore[method-assign]
        server_state.agent.conversation = Mock()
        server_state.agent.conversation.chat_messages = []

        # Switch to session A
        await get_or_load_session(server_state, session_a_id)

        # Both input providers still have their correct session IDs
        provider_a_after = server_state.ensure_input_provider(session_a_id)
        provider_b_after = server_state.ensure_input_provider(session_b_id)
        assert provider_a_after.session_id == session_a_id
        assert provider_b_after.session_id == session_b_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
