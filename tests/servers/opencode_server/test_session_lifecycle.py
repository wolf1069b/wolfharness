"""Session lifecycle tests.

Ported from OpenCode's test/session/session.test.ts

Tests session creation, events, and lifecycle management.

Note: The OpenCode API uses camelCase field names with "ID" suffix:
- projectID (not project_id)
- parentID (not parent_id)
- Session IDs use "ses_" prefix (not "session_")
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from wolfharness.sessions.models import SessionData
from wolfharness_server.opencode_server.models import Session
from wolfharness_server.opencode_server.models.events import (
    SessionCreatedEvent,
    SessionIdleEvent,
    SessionStatusEvent,
)


if TYPE_CHECKING:
    from httpx import AsyncClient

    from tests.servers.opencode_server.conftest import EventCapture
    from wolfharness_server.opencode_server.state import ServerState


class TestSessionCreatedEvent:
    """Tests for session.created event emission."""

    async def test_should_emit_session_created_event_when_session_is_created(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
        event_capture: EventCapture,
    ):
        """Session creation should emit session.created event.

        Ported from: "should emit session.started event when session is created"
        """
        # Create a session via the API
        response = await async_client.post("/session", json={"title": "Test Session"})
        assert response.status_code == 200
        session_data = response.json()
        # Verify the session was created correctly
        assert "id" in session_data
        assert session_data["title"] == "Test Session"
        assert session_data["projectID"] == "global"  # Non-git directory returns "global"
        # Verify the session.created event was emitted
        created_events = event_capture.get_events_by_type("session.created")
        assert len(created_events) == 1
        event = created_events[0]
        assert isinstance(event, SessionCreatedEvent)
        assert event.properties.info.id == session_data["id"]
        assert event.properties.info.title == session_data["title"]
        assert event.properties.info.project_id == "global"  # Non-git directory returns "global"
        status_events = event_capture.get_events_by_type("session.status")
        idle_events = event_capture.get_events_by_type("session.idle")
        assert (
            len(status_events) == 2
        )  # set_session_status() + mark_session_idle() explicit broadcast
        assert len(idle_events) == 1
        assert isinstance(status_events[0], SessionStatusEvent)
        assert isinstance(idle_events[0], SessionIdleEvent)
        assert status_events[0].properties.status.type == "idle"

    async def test_session_created_event_should_be_emitted_before_session_updated(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
        event_capture: EventCapture,
    ):
        """Session.created event should be emitted before session.updated.

        Ported from: "session.started event should be emitted before session.updated"

        When a session is created and then updated, the created event must come first.
        """
        # Create a session
        create_response = await async_client.post("/session", json={"title": "Original Title"})
        assert create_response.status_code == 200
        session_id = create_response.json()["id"]
        # Update the session title
        update_response = await async_client.patch(
            f"/session/{session_id}",
            json={"title": "Updated Title"},
        )
        assert update_response.status_code == 200
        # Verify event order: created should come before updated
        event_types = [e.type for e in event_capture.events]
        assert "session.created" in event_types
        assert "session.updated" in event_types
        created_index = event_types.index("session.created")
        updated_index = event_types.index("session.updated")
        assert created_index < updated_index, (
            f"session.created (index {created_index}) should come before "
            f"session.updated (index {updated_index})"
        )


class TestSessionCRUD:
    """Tests for session CRUD operations."""

    async def test_create_session_returns_valid_session(
        self,
        async_client: AsyncClient,
        tmp_project_dir: Path,
        event_capture: EventCapture,
    ):
        """Creating a session should return a valid session object."""
        response = await async_client.post("/session", json={"title": "My Session"})
        assert response.status_code == 200
        session = response.json()
        # Verify required fields (using camelCase API format)
        assert "id" in session
        assert session["id"].startswith("ses_")  # Session IDs use "ses_" prefix
        assert session["title"] == "My Session"
        assert session["projectID"] == "global"  # Non-git directory returns "global"
        assert session["directory"] == str(Path(tmp_project_dir).resolve())
        assert session["version"] == "1"
        assert "time" in session
        assert "created" in session["time"]
        assert "updated" in session["time"]
        status_events = event_capture.get_events_by_type("session.status")
        idle_events = event_capture.get_events_by_type("session.idle")
        assert (
            len(status_events) == 2
        )  # set_session_status() + mark_session_idle() explicit broadcast
        assert len(idle_events) == 1

    async def test_create_session_with_parent_id(self, async_client: AsyncClient):
        """Creating a session with parent_id should set the parent."""
        # Create parent session
        parent_response = await async_client.post("/session", json={"title": "Parent"})
        parent_id = parent_response.json()["id"]
        # Create child session (API accepts snake_case due to populate_by_name)
        child_response = await async_client.post(
            "/session",
            json={"title": "Child", "parent_id": parent_id},
        )

        assert child_response.status_code == 200
        child = child_response.json()
        assert child["parentID"] == parent_id  # Response uses camelCase

    async def test_create_session_with_default_title(self, async_client: AsyncClient):
        """Creating a session without title should use default."""
        response = await async_client.post("/session", json={})
        assert response.status_code == 200
        session = response.json()
        assert session["title"] == "New Session"

    async def test_get_session_returns_created_session(self, async_client: AsyncClient):
        """Getting a session should return the correct session."""
        # Create a session
        create_response = await async_client.post("/session", json={"title": "Get Test"})
        session_id = create_response.json()["id"]
        # Get the session
        get_response = await async_client.get(f"/session/{session_id}")
        assert get_response.status_code == 200
        session = get_response.json()
        assert session["id"] == session_id
        assert session["title"] == "Get Test"

    async def test_get_nonexistent_session_returns_404(self, async_client: AsyncClient):
        """Getting a non-existent session should return 404."""
        response = await async_client.get("/session/nonexistent-session-id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.skip(
        reason="Production bug #191: now_ms() float truncation causes non-monotonic timestamps"
    )
    async def test_update_session_title(
        self,
        async_client: AsyncClient,
        event_capture: EventCapture,
    ):
        """Updating a session title should persist and emit event."""
        # Create a session
        create_response = await async_client.post("/session", json={"title": "Original"})
        session_id = create_response.json()["id"]
        original_created = create_response.json()["time"]["created"]
        # Update the title
        update_response = await async_client.patch(
            f"/session/{session_id}",
            json={"title": "Updated Title"},
        )

        assert update_response.status_code == 200
        updated = update_response.json()
        assert updated["title"] == "Updated Title"
        assert updated["time"]["created"] == original_created
        assert updated["time"]["updated"] >= original_created
        # Verify session.updated event was emitted
        updated_events = event_capture.get_events_by_type("session.updated")
        assert len(updated_events) >= 1
        last_update = updated_events[-1]
        assert last_update.properties.info.title == "Updated Title"

    async def test_update_nonexistent_session_returns_404(self, async_client: AsyncClient):
        """Updating a non-existent session should return 404."""
        response = await async_client.patch("/session/nonexistent-id", json={"title": "New Title"})
        assert response.status_code == 404

    async def test_delete_session(self, async_client: AsyncClient, event_capture: EventCapture):
        """Deleting a session should remove it and emit event."""
        # Create a session
        create_response = await async_client.post("/session", json={"title": "To Delete"})
        session_id = create_response.json()["id"]
        # Delete the session
        delete_response = await async_client.delete(f"/session/{session_id}")
        assert delete_response.status_code == 200
        assert delete_response.json() is True
        # Verify session is gone
        get_response = await async_client.get(f"/session/{session_id}")
        assert get_response.status_code == 404
        # Verify session.deleted event was emitted
        deleted_events = event_capture.get_events_by_type("session.deleted")
        assert len(deleted_events) == 1
        assert deleted_events[0].properties.session_id == session_id

    async def test_delete_nonexistent_session_returns_404(self, async_client: AsyncClient):
        """Deleting a non-existent session should return 404."""
        response = await async_client.delete("/session/nonexistent-id")

        assert response.status_code == 404

    async def test_get_session_messages_returns_messages(self, async_client: AsyncClient):
        """Getting session messages should return all messages for the session."""
        # Create a session
        create_response = await async_client.post("/session", json={"title": "Message Test"})
        session_id = create_response.json()["id"]

        # Initially, there should be no messages
        messages_response = await async_client.get(f"/session/{session_id}/message")
        assert messages_response.status_code == 200
        messages = messages_response.json()
        assert len(messages) == 0

    async def test_get_session_messages_nonexistent_session_returns_404(
        self, async_client: AsyncClient
    ):
        """Getting messages for a non-existent session should return 404."""
        response = await async_client.get("/session/nonexistent-session-id/message")
        assert response.status_code == 404

    async def test_list_sessions_empty(self, async_client: AsyncClient):
        """Listing sessions when none exist should return empty list."""
        response = await async_client.get("/session")

        assert response.status_code == 200
        assert response.json() == []

    async def test_list_sessions_returns_created_sessions(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """Listing sessions should return all created sessions."""
        # Create multiple sessions
        session_ids = []
        for i in range(3):
            response = await async_client.post("/session", json={"title": f"Session {i}"})
            session_ids.append(response.json()["id"])

        # Mock agent.list_sessions to return SessionData objects
        now = datetime.now(UTC)
        session_data_list = [
            SessionData(
                session_id=sid,
                agent_name="test-agent",
                cwd=server_state.working_dir,
                created_at=now,
                last_active=now,
                metadata={"title": f"Session {i}"},
            )
            for i, sid in enumerate(session_ids)
        ]
        server_state.agent.list_sessions.return_value = session_data_list  # ty: ignore[unresolved-attribute]
        # List sessions
        response = await async_client.get("/session")
        assert response.status_code == 200
        sessions = response.json()
        assert len(sessions) == 3
        returned_ids = {s["id"] for s in sessions}
        assert returned_ids == set(session_ids)

    async def test_list_sessions_uses_session_controller_when_available(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """list_sessions delegates to SessionController when session_controller is set."""
        from unittest.mock import Mock

        from wolfharness_server.opencode_server.models.common import TimeCreatedUpdated
        from wolfharness_server.opencode_server.models.session_info import SessionInfo

        # Pre-populate the session cache so the route does not need storage
        server_state.sessions["ses_ctrl_001"] = Session(
            id="ses_ctrl_001",
            project_id="default",
            directory=server_state.base_path,
            title="Controller Session",
            version="1",
            time=TimeCreatedUpdated(created=1234567890000, updated=1234567891000),
        )

        mock_controller = Mock()
        mock_controller.list_sessions.return_value = [
            SessionInfo(
                session_id="ses_ctrl_001",
                agent_name="test-agent",
                created_at=1234567890.0,
                last_active_at=1234567891.0,
                is_per_session_agent=False,
                status="idle",
            )
        ]
        server_state.session_controller = mock_controller

        response = await async_client.get("/session")
        assert response.status_code == 200
        sessions = response.json()
        assert len(sessions) == 1
        assert sessions[0]["id"] == "ses_ctrl_001"
        assert sessions[0]["title"] == "Controller Session"
        mock_controller.list_sessions.assert_called_once()


class TestSessionStatus:
    """Tests for session status management."""

    async def test_get_session_status_empty_when_all_idle(self, async_client: AsyncClient):
        """Getting status should return empty when all sessions are idle."""
        # Create a session (it starts as idle)
        await async_client.post("/session", json={"title": "Idle Session"})
        # Get status
        response = await async_client.get("/session/status")
        assert response.status_code == 200
        # Only non-idle sessions are returned
        assert response.json() == {}

    async def test_session_status_is_idle_by_default(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """Newly created sessions should have idle status."""
        # Create a session
        response = await async_client.post("/session", json={"title": "New Session"})
        session_id = response.json()["id"]
        # Verify via the public status endpoint: idle sessions are filtered out
        status_response = await async_client.get("/session/status")
        assert status_response.status_code == 200
        assert session_id not in status_response.json()

    async def test_abort_session(self, async_client: AsyncClient, server_state: ServerState):
        """Aborting a session should set status to idle."""
        # Create a session
        response = await async_client.post("/session", json={"title": "Running Session"})
        session_id = response.json()["id"]
        # Track broadcast events to verify idle status after abort
        status_events: list[SessionStatusEvent] = []
        original_broadcast = server_state.broadcast_event

        async def tracking_broadcast(event: object) -> None:
            if isinstance(event, SessionStatusEvent):
                status_events.append(event)
            await original_broadcast(event)

        server_state.broadcast_event = tracking_broadcast  # type: ignore[method-assign]
        # Abort the session
        abort_response = await async_client.post(f"/session/{session_id}/abort")
        assert abort_response.status_code == 200
        assert abort_response.json() is True
        # Verify idle status was broadcast
        idle_events = [e for e in status_events if e.properties.status.type == "idle"]
        assert len(idle_events) >= 1, f"Expected idle broadcast after abort, got {status_events}"

    async def test_abort_session_delegates_to_session_pool(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """Aborting should delegate run cancellation to SessionPool."""
        response = await async_client.post("/session", json={"title": "Abort Session"})
        session_id = response.json()["id"]
        server_state.agent.interrupt = AsyncMock()

        abort_response = await async_client.post(f"/session/{session_id}/abort")
        assert abort_response.status_code == 200
        assert abort_response.json() is True

        server_state.agent.interrupt.assert_awaited_once()
        # Verify SessionPool cancel_run_for_session was called
        session_pool = server_state.agent.host_context.session_pool
        session_pool.sessions.cancel_run_for_session.assert_called_once_with(session_id)

    async def test_abort_nonexistent_session_returns_404(self, async_client: AsyncClient):
        """Aborting a non-existent session should return 404."""
        response = await async_client.post("/session/nonexistent-id/abort")
        assert response.status_code == 404

    async def test_abort_native_agent_calls_interrupt_not_cancel_run(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """Aborting a native (per-session) agent should call interrupt() only.

        interrupt() internally calls cancel_run_for_session(), so the
        abort handler must NOT call session_pool.cancel_run() again.
        A double cancel() kills the start() generator (issue #182).
        """
        from wolfharness.orchestrator.core import SessionState

        response = await async_client.post("/session", json={"title": "Native Agent Session"})
        session_id = response.json()["id"]

        # Set up a mock per-session agent with interrupt
        per_session_agent = Mock()
        per_session_agent.interrupt = AsyncMock()
        server_state.agent.interrupt = AsyncMock()

        # Set up session controller with a native (per-session) session
        session_state = SessionState(
            session_id=session_id,
            agent_name="test-agent",
            is_per_session_agent=True,
            current_run_id="run-native-123",
        )
        session_controller = MagicMock()
        session_controller.get_session.return_value = session_state
        session_controller.get_session_agent.return_value = per_session_agent
        server_state.session_controller = session_controller

        abort_response = await async_client.post(f"/session/{session_id}/abort")
        assert abort_response.status_code == 200
        assert abort_response.json() is True

        # Per-session agent should be interrupted
        per_session_agent.interrupt.assert_awaited_once()
        # Shared agent should NOT be interrupted
        server_state.agent.interrupt.assert_not_awaited()
        # cancel_run should NOT be called separately — interrupt() handles
        # it internally via cancel_run_for_session(). Calling cancel_run()
        # again would double-cancel and kill the start() generator.
        session_pool = server_state.agent.host_context.session_pool
        session_pool.cancel_run.assert_not_called()

    async def test_abort_non_native_shared_agent_skips_interrupt(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """Aborting a non-native shared agent should cancel run but NOT interrupt shared agent."""
        from wolfharness.orchestrator.core import SessionState

        response = await async_client.post("/session", json={"title": "Shared Agent Session"})
        session_id = response.json()["id"]

        # Set up mock shared agent
        shared_agent = Mock()
        shared_agent.interrupt = AsyncMock()
        server_state.agent.interrupt = AsyncMock()

        # Set up session controller with a non-native (shared) session
        session_state = SessionState(
            session_id=session_id,
            agent_name="test-agent",
            is_per_session_agent=False,
            current_run_id="run-shared-456",
        )
        session_controller = MagicMock()
        session_controller.get_session.return_value = session_state
        session_controller.get_session_agent.return_value = shared_agent
        server_state.session_controller = session_controller

        abort_response = await async_client.post(f"/session/{session_id}/abort")
        assert abort_response.status_code == 200
        assert abort_response.json() is True

        # Shared agent should NOT be interrupted (would kill all sessions)
        shared_agent.interrupt.assert_not_awaited()
        server_state.agent.interrupt.assert_not_awaited()
        # cancel_run should still be called with the run_id
        session_pool = server_state.agent.host_context.session_pool
        session_pool.cancel_run.assert_called_once_with("run-shared-456")


class TestSessionFork:
    """Tests for session forking functionality."""

    async def test_fork_session_creates_new_session_with_parent(
        self,
        async_client: AsyncClient,
        event_capture: EventCapture,
    ):
        """Forking a session should create a new session with parent_id set."""
        # Create original session
        original_response = await async_client.post("/session", json={"title": "Original Session"})
        original_id = original_response.json()["id"]
        # Fork the session
        fork_response = await async_client.post(f"/session/{original_id}/fork")
        assert fork_response.status_code == 200
        forked = fork_response.json()
        assert forked["id"] != original_id
        assert forked["parentID"] == original_id  # camelCase in response
        assert forked["title"] == "Original Session (fork)"
        # Verify session.created event was emitted for the fork
        created_events = event_capture.get_events_by_type("session.created")
        # Should have 2: original + fork
        assert len(created_events) == 2
        fork_event = created_events[-1]
        assert fork_event.properties.info.id == forked["id"]
        assert fork_event.properties.info.parent_id == original_id  # Python attr
        status_events = event_capture.get_events_by_type("session.status")
        idle_events = event_capture.get_events_by_type("session.idle")
        assert status_events[-1].properties.session_id == forked["id"]
        assert idle_events[-1].properties.session_id == forked["id"]

    async def test_fork_nonexistent_session_returns_404(self, async_client: AsyncClient):
        """Forking a non-existent session should return 404."""
        response = await async_client.post("/session/nonexistent-id/fork")
        assert response.status_code == 404


class TestSessionTodos:
    """Tests for session todo management."""

    async def test_get_session_todos_empty_initially(self, async_client: AsyncClient):
        """Getting todos for a new session should return empty list."""
        # Create a session
        response = await async_client.post("/session", json={"title": "Todo Session"})
        session_id = response.json()["id"]
        # Get todos
        todos_response = await async_client.get(f"/session/{session_id}/todo")
        assert todos_response.status_code == 200
        assert todos_response.json() == []

    async def test_get_todos_for_nonexistent_session_returns_404(self, async_client: AsyncClient):
        """Getting todos for a non-existent session should return 404."""
        response = await async_client.get("/session/nonexistent-id/todo")
        assert response.status_code == 404


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
