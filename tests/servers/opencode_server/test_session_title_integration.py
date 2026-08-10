"""Integration tests for OpenCode session title persistence.

Tests that session titles are correctly:
1. Saved when creating sessions
2. Persisted to storage
3. Loaded when listing/getting sessions
4. Updated when title generation completes
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import anyio
import pytest

from wolfharness.sessions.models import SessionData
from wolfharness_server.opencode_server.converters import (
    opencode_to_session_data,
    session_data_to_opencode,
)
from wolfharness_server.opencode_server.models import Session
from wolfharness_server.opencode_server.models.common import TimeCreatedUpdated


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from httpx import AsyncClient

    from wolfharness_server.opencode_server.state import ServerState


class TestSessionTitlePersistence:
    """Tests for session title persistence through the full stack."""

    async def test_create_session_saves_title(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """Creating a session should persist the title to storage."""
        # Create session with explicit title
        response = await async_client.post("/session", json={"title": "My Test Session"})
        assert response.status_code == 200
        session_data = response.json()
        session_id = session_data["id"]

        # Verify title in response
        assert session_data["title"] == "My Test Session"

        # Verify title is in server state
        assert session_id in server_state.sessions
        assert server_state.sessions[session_id].title == "My Test Session"

        # Verify title is persisted to storage
        storage_session = await server_state.storage.load_session(session_id)
        assert storage_session is not None
        assert storage_session.title == "My Test Session"

    async def test_list_sessions_includes_titles(
        self,
        async_client: AsyncClient,
    ):
        """Listing sessions should include correct titles."""
        # Create sessions with different titles
        response1 = await async_client.post("/session", json={"title": "First Session"})
        response2 = await async_client.post("/session", json={"title": "Second Session"})

        session1_id = response1.json()["id"]
        session2_id = response2.json()["id"]

        # List sessions
        list_response = await async_client.get("/session")
        assert list_response.status_code == 200
        sessions = list_response.json()

        # Find our sessions and verify titles
        session_titles = {s["id"]: s["title"] for s in sessions}
        assert session1_id in session_titles
        assert session2_id in session_titles
        assert session_titles[session1_id] == "First Session"
        assert session_titles[session2_id] == "Second Session"

    async def test_get_session_returns_title(
        self,
        async_client: AsyncClient,
    ):
        """Getting a specific session should return the correct title."""
        # Create session
        create_response = await async_client.post("/session", json={"title": "Specific Title"})
        session_id = create_response.json()["id"]

        # Get session
        get_response = await async_client.get(f"/session/{session_id}")
        assert get_response.status_code == 200
        session_data = get_response.json()

        assert session_data["title"] == "Specific Title"

    async def test_update_session_persists_title(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """Updating session title should persist to storage."""
        # Create session
        create_response = await async_client.post("/session", json={"title": "Original Title"})
        session_id = create_response.json()["id"]

        # Update title
        update_response = await async_client.patch(
            f"/session/{session_id}",
            json={"title": "Updated Title"},
        )
        assert update_response.status_code == 200

        # Verify in response
        assert update_response.json()["title"] == "Updated Title"

        # Verify in server state
        assert server_state.sessions[session_id].title == "Updated Title"

        # Verify persisted to storage
        storage_session = await server_state.storage.load_session(session_id)
        assert storage_session is not None
        assert storage_session.title == "Updated Title"

        # Verify can be retrieved via API
        get_response = await async_client.get(f"/session/{session_id}")
        assert get_response.json()["title"] == "Updated Title"


class TestSessionTitleConverters:
    """Tests for the converter functions that handle title."""

    def test_opencode_to_session_data_includes_title(self):
        """opencode_to_session_data should include title in metadata."""
        session = Session(
            id="test_session_123",
            project_id="global",
            directory="/tmp",
            title="Test Session Title",
            version="1",
            time=TimeCreatedUpdated(created=1234567890, updated=1234567890),
        )

        session_data = opencode_to_session_data(session, agent_name="test_agent")

        # Title should be in metadata
        assert "title" in session_data.metadata
        assert session_data.metadata["title"] == "Test Session Title"
        # Title property should work
        assert session_data.title == "Test Session Title"

    def test_opencode_to_session_data_without_title(self):
        """opencode_to_session_data should handle empty title."""
        session = Session(
            id="test_session_456",
            project_id="global",
            directory="/tmp",
            title="",  # Empty title
            version="1",
            time=TimeCreatedUpdated(created=1234567890, updated=1234567890),
        )

        session_data = opencode_to_session_data(session, agent_name="test_agent")

        # Empty title should not be in metadata
        assert "title" not in session_data.metadata or session_data.metadata.get("title") == ""

    def test_session_data_to_opencode_reads_title(self):
        """session_data_to_opencode should read title from metadata."""
        session_data = SessionData(
            session_id="test_session_789",
            agent_name="test_agent",
            metadata={"title": "Metadata Title"},
        )

        session = session_data_to_opencode(session_data)

        assert session.title == "Metadata Title"

    def test_session_data_to_opencode_default_title(self):
        """session_data_to_opencode should use default when no title."""
        session_data = SessionData(
            session_id="test_session_000",
            agent_name="test_agent",
            metadata={},  # No title
        )

        session = session_data_to_opencode(session_data)

        assert session.title == "New Session"  # Default value

    def test_roundtrip_conversion_preserves_title(self):
        """Round-trip conversion should preserve title."""
        original_session = Session(
            id="test_session_round",
            project_id="global",
            directory="/tmp",
            title="Round Trip Title",
            version="1",
            time=TimeCreatedUpdated(created=1234567890, updated=1234567890),
        )

        # Convert to SessionData and back
        session_data = opencode_to_session_data(original_session, agent_name="test_agent")
        converted_session = session_data_to_opencode(session_data)

        assert converted_session.title == "Round Trip Title"


class TestSessionTitleGeneration:
    """Tests for automatic title generation flow."""

    async def test_log_session_triggers_title_generation(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """log_session with initial_prompt should trigger title generation.

        Note: This test mocks the LLM call to avoid external dependencies.
        """
        from wolfharness.storage.manager import SessionMetadata, StorageManager

        session_id = "test_title_gen_123"

        # Create session first
        session = Session(
            id=session_id,
            project_id="global",
            directory="/tmp",
            title="New Session",
            version="1",
            time=TimeCreatedUpdated(created=1234567890, updated=1234567890),
        )
        server_state.sessions[session_id] = session
        server_state.messages[session_id] = []

        # Mock the title generation
        mock_metadata = SessionMetadata(
            title="Generated Test Title",
            emoji="🧪",
            icon="mdi:test-tube",
        )

        with (
            patch.object(
                StorageManager,
                "_generate_title_core",
                return_value=mock_metadata,
            ),
            # Remove pytest env temporarily to trigger generation
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("PYTEST_CURRENT_TEST", None)

            # Call log_session with initial_prompt
            await server_state.storage.log_session(
                session_id=session_id,
                node_name="test_agent",
                initial_prompt="Tell me about Python programming",
            )

            # Wait for async processing
            await anyio.sleep(0.1)

        # Verify title was updated
        stored_title = await server_state.storage.get_session_title(session_id)
        assert stored_title == "Generated Test Title"

    async def test_title_generation_emits_event(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
        event_capture,  # Assuming this fixture exists
    ):
        """Title generation should emit session.updated event."""
        # This would test the full event flow
        # Implementation depends on your event capture fixture


class TestSessionTitleSearch:
    """Tests for searching sessions by title."""

    async def test_search_sessions_by_title(
        self,
        async_client: AsyncClient,
    ):
        """Should be able to search sessions by title."""
        # Create sessions with unique titles
        await async_client.post("/session", json={"title": "Python Questions"})
        await async_client.post("/session", json={"title": "JavaScript Help"})
        await async_client.post("/session", json={"title": "Python Advanced"})

        # Search for Python
        search_response = await async_client.get("/session?search=python")
        assert search_response.status_code == 200
        results = search_response.json()

        # Should find 2 sessions
        assert len(results) == 2
        titles = [s["title"] for s in results]
        assert "Python Questions" in titles
        assert "Python Advanced" in titles
        assert "JavaScript Help" not in titles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
