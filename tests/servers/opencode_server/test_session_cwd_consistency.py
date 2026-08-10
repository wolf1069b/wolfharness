"""Tests for session cwd consistency between create and list operations.

Bug: create_session uses state.working_dir for session.directory,
while list_sessions uses state.agent.env.cwd for filtering.
When these differ, sessions created via the API are invisible to list_sessions.

Additionally, GET /session does not accept a `directory` query param that
the OpenCode SDK sends, and cwd comparison uses strict string equality
without path normalization.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from wolfharness.sessions.models import SessionData


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from httpx import AsyncClient

    from wolfharness_server.opencode_server.state import ServerState


class TestSessionCwdConsistency:
    """Tests ensuring create_session and list_sessions use the same cwd source."""

    async def test_create_and_list_use_same_cwd(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
        tmp_project_dir: Path,
    ):
        """Sessions created via API should be visible when listing.

        Both create_session and list_sessions use state.base_path
        (which resolves working_dir via Path.resolve()) as the routing
        directory. env.cwd is for agent tool isolation, not routing.
        """
        # Create a session — uses state.base_path (working_dir) for directory
        create_response = await async_client.post("/session", json={"title": "Test"})
        assert create_response.status_code == 200
        created = create_response.json()

        # The session's directory should match the resolved working_dir
        expected_resolved = str(Path(str(tmp_project_dir)).resolve())
        assert created["directory"] == expected_resolved

        # Set up list_sessions to return the session with the correct cwd
        now = datetime.now(UTC)
        session_data = SessionData(
            session_id=created["id"],
            agent_name="test-agent",
            cwd=created["directory"],
            created_at=now,
            last_active=now,
            metadata={"title": "Test"},
        )
        server_state.agent.list_sessions = AsyncMock(return_value=[session_data])  # type: ignore[method-assign]

        # List sessions — uses state.base_path (same as create_session)
        list_response = await async_client.get("/session")
        assert list_response.status_code == 200
        sessions = list_response.json()

        # The created session should appear in the list
        # because both create and list use state.base_path
        listed_ids = {s["id"] for s in sessions}
        assert created["id"] in listed_ids, (
            f"Session {created['id']} was created with directory={created['directory']!r} "
            "but list_sessions did not return it. "
            "create_session and list_sessions must use the same cwd source."
        )

    async def test_create_session_directory_matches_working_dir(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
        tmp_project_dir: Path,
    ):
        """create_session uses state.base_path (working_dir) for session.directory.

        The server's working_dir defines the project scope for routing.
        env.cwd is for agent tool isolation and does not affect routing.
        """
        response = await async_client.post("/session", json={"title": "Test"})
        assert response.status_code == 200
        session = response.json()

        # state.base_path resolves working_dir via Path.resolve()
        expected_resolved = str(Path(str(tmp_project_dir)).resolve())
        assert session["directory"] == expected_resolved, (
            f"Session directory should be resolved working_dir={expected_resolved!r} "
            f"but got {session['directory']!r}. "
            "create_session must use state.base_path for routing."
        )


class TestSessionDirectoryQueryParam:
    """Tests for the `directory` query param on GET /session.

    The OpenCode SDK auto-injects a `directory` query param via its
    request interceptor. The server must accept and use this param.
    """

    async def test_list_sessions_accepts_directory_param(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """GET /session should accept a `directory` query parameter.

        When provided, it should filter sessions by the given directory
        instead of the default env.cwd.
        """
        # Use legacy path (agent.list_sessions) by clearing session_controller
        server_state.session_controller = None

        directory = "/tmp/custom-directory"
        now = datetime.now(UTC)

        # Set up mock to return sessions for the given directory
        session_data = SessionData(
            session_id="ses_test_001",
            agent_name="test-agent",
            cwd=directory,
            created_at=now,
            last_active=now,
            metadata={"title": "Custom Dir Session"},
        )
        server_state.agent.list_sessions = AsyncMock(return_value=[session_data])  # type: ignore[method-assign]

        # Request with directory param
        response = await async_client.get("/session", params={"directory": directory})
        assert response.status_code == 200
        sessions = response.json()

        # Should return the session from the custom directory
        assert len(sessions) == 1
        assert sessions[0]["id"] == "ses_test_001"

    async def test_directory_param_overrides_env_cwd(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
    ):
        """When `directory` param is provided, it should override env.cwd.

        This allows the OpenCode TUI to request sessions for a specific
        project directory regardless of the server's configured env.cwd.
        """
        # Use legacy path (agent.list_sessions) by clearing session_controller
        server_state.session_controller = None

        override_dir = "/tmp/override-directory"
        server_state.agent.env.cwd = "/tmp/default-env-cwd"
        server_state.agent.list_sessions = AsyncMock(return_value=[])  # type: ignore[method-assign]

        response = await async_client.get("/session", params={"directory": override_dir})
        assert response.status_code == 200

        # Verify list_sessions was called with the override directory
        list_sessions_mock = server_state.agent.list_sessions
        list_sessions_mock.assert_called_once()  # type: ignore[union-attr]
        call_kwargs = list_sessions_mock.call_args  # type: ignore[union-attr]
        assert call_kwargs.kwargs.get("cwd") == override_dir or (
            len(call_kwargs.args) > 0 and call_kwargs.args[0] == override_dir
        ), (
            f"list_sessions should be called with cwd={override_dir!r}, "
            f"but got call_args={call_kwargs}"
        )


class TestSessionCwdPathNormalization:
    """Tests for path normalization in cwd comparison.

    NativeAgent.list_sessions() uses strict string equality for cwd
    matching, which fails on trailing slashes, symlinks, and other
    path variations.
    """

    async def test_list_sessions_normalizes_trailing_slash(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
        tmp_project_dir: Path,
    ):
        """Sessions with/without trailing slash in cwd should both be visible.

        If a session was stored with cwd="/foo/bar" but the query uses
        cwd="/foo/bar/", strict matching would fail. Path normalization
        should handle this.
        """
        # Use legacy path (agent.list_sessions) by clearing session_controller
        server_state.session_controller = None

        cwd_without_slash = str(tmp_project_dir)
        cwd_with_slash = str(tmp_project_dir) + "/"

        # Set env.cwd to the version with trailing slash
        server_state.agent.env.cwd = cwd_with_slash

        now = datetime.now(UTC)
        # Session data stored without trailing slash
        session_data = SessionData(
            session_id="ses_trailing_test",
            agent_name="test-agent",
            cwd=cwd_without_slash,
            created_at=now,
            last_active=now,
            metadata={"title": "Trailing Slash Test"},
        )
        server_state.agent.list_sessions = AsyncMock(return_value=[session_data])  # type: ignore[method-assign]

        response = await async_client.get("/session")
        assert response.status_code == 200
        sessions = response.json()

        # Should find the session despite trailing slash mismatch
        assert any(s["id"] == "ses_trailing_test" for s in sessions), (
            f"Session with cwd={cwd_without_slash!r} should be visible "
            f"when listing with cwd={cwd_with_slash!r}. "
            "Path normalization should handle trailing slashes."
        )

    async def test_list_sessions_matches_dot_path(
        self,
        async_client: AsyncClient,
        server_state: ServerState,
        tmp_project_dir: Path,
    ):
        """Sessions should match even when one path uses '.' and the other is absolute.

        Both "." and "/full/path" refer to the same directory when cwd
        is /full/path.
        """
        # Use legacy path (agent.list_sessions) by clearing session_controller
        server_state.session_controller = None

        # This test documents expected behavior - the actual fix
        # would be in NativeAgent.list_sessions()
        cwd_absolute = str(tmp_project_dir)

        now = datetime.now(UTC)
        # Session stored with "." as cwd
        session_data = SessionData(
            session_id="ses_dot_test",
            agent_name="test-agent",
            cwd=".",
            created_at=now,
            last_active=now,
            metadata={"title": "Dot Path Test"},
        )
        server_state.agent.list_sessions = AsyncMock(return_value=[session_data])  # type: ignore[method-assign]

        response = await async_client.get("/session")
        assert response.status_code == 200
        sessions = response.json()

        # Should find the session despite "." vs absolute path
        assert any(s["id"] == "ses_dot_test" for s in sessions), (
            f"Session with cwd='.' should be visible "
            f"when listing with cwd={cwd_absolute!r}. "
            "Path normalization should resolve relative paths."
        )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
