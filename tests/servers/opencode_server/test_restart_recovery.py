"""Regression tests for cold-start recovery after server restart."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import TYPE_CHECKING

import pytest

from wolfharness.sessions.models import SessionData
from wolfharness_server.opencode_server.models import SessionIdleEvent
from wolfharness_server.opencode_server.routes.session_routes import get_or_load_session
from wolfharness_server.opencode_server.state import ServerState
from wolfharness_storage.opencode_provider import helpers


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import Mock


class TestRestartRecovery:
    """Verify persisted sessions recover correctly after a fresh server start."""

    async def test_get_or_load_session_restores_runtime_state_after_restart(
        self,
        server_state: ServerState,
        tmp_project_dir: Path,
        event_capture,
    ):
        """Cold-start loading should rebuild all runtime buckets for a persisted session."""
        session_id = "restart-session"
        now = datetime.now(UTC)
        session_data = SessionData(
            session_id=session_id,
            agent_name="test-agent",
            cwd=str(tmp_project_dir),
            created_at=now,
            last_active=now,
            metadata={"title": "Recovered Session"},
        )

        # Persist the session so the store-first path in get_or_load_session finds it.
        await server_state.pool.storage.save_session(session_data)

        loaded_session = await get_or_load_session(server_state, session_id)

        assert loaded_session is not None
        assert loaded_session.id == session_id
        assert loaded_session.directory == str(tmp_project_dir)
        assert session_id in server_state.sessions
        assert server_state.reverted_messages[session_id] == []
        # Session status is now broadcast via set_session_status() instead of
        # stored in the in-memory session_status dict.

        status_events = [
            event
            for event in event_capture.get_events_by_type("session.status")
            if event.properties.session_id == session_id
        ]
        idle_events = [
            event
            for event in event_capture.get_events_by_type("session.idle")
            if event.properties.session_id == session_id
        ]
        assert status_events
        assert idle_events

    def test_event_factory_uses_resolved_directory_for_restart_routing(
        self,
        tmp_project_dir: Path,
        mock_agent: Mock,
    ):
        """Global event routing metadata should stay stable across restart path aliases."""
        nested = tmp_project_dir / "nested"
        nested.mkdir()
        aliased_working_dir = str(nested / "..")

        state = ServerState(working_dir=aliased_working_dir, agent=mock_agent)
        payload = json.loads(state.get_event_factory().wrap(SessionIdleEvent.create("sess-1")))

        resolved_dir = str(tmp_project_dir.resolve())
        assert payload["directory"] == resolved_dir
        assert payload["project"] == helpers.compute_project_id(resolved_dir)
        assert payload["payload"]["type"] == "session.idle"
        assert payload["payload"]["sessionId"] == "sess-1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
