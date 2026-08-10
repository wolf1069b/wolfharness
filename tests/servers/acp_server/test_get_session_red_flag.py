"""Red-flag test: ACPSessionManager.get_session() should not return None.

when the session exists in _acp_sessions but not yet in _session_controller.

This is a standalone test that uses a real AgentPool to avoid circular imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest


if TYPE_CHECKING:
    from wolfharness import AgentPool


pytestmark = pytest.mark.unit


def _make_manager_with_controller(pool: AgentPool) -> tuple[object, object]:
    """Create an ACPSessionManager with _session_controller set.

    Uses a real AgentPool — the SessionController returns None for
    non-existent sessions naturally.
    """
    from wolfharness_server.acp_server.session_manager import ACPSessionManager

    manager = ACPSessionManager(pool=pool)
    controller = pool.session_pool.sessions if pool.session_pool else None
    return manager, controller


def _make_manager_without_controller(pool: AgentPool) -> object:
    """Create an ACPSessionManager without _session_controller (no SessionPool).

    Sets pool._session_pool = None to simulate no SessionPool.
    Caller is responsible for restoring pool._session_pool after the test.
    """
    from wolfharness_server.acp_server.session_manager import ACPSessionManager

    pool._session_pool = None
    return ACPSessionManager(pool=pool)


class TestGetSessionRedFlag:
    """Red-flag tests for get_session() behavior with _session_controller."""

    def test_returns_session_when_not_yet_in_controller(self, minimal_pool: AgentPool):
        """RED FLAG: get_session() MUST return the ACPSession from _acp_sessions.

        even when _session_controller doesn't have it yet.

        This reproduces the bug: during session/new, create_session() adds the
        session to _acp_sessions but the orchestrator registers it with
        _session_controller asynchronously. If get_session() returns None
        during this window, create_task(send_available_commands_update())
        is skipped and available_commands_update is never sent.
        """
        manager, controller = _make_manager_with_controller(minimal_pool)
        session_id = "sess-red-flag-001"

        # Simulate what create_session() does: add to _acp_sessions
        mock_session: MagicMock = MagicMock()
        mock_session.session_id = session_id
        manager._acp_sessions[session_id] = mock_session

        # Precondition: _session_controller exists but doesn't have the session
        assert manager._session_controller is not None
        assert controller.get_session(session_id) is None, (
            "Precondition: session should NOT be in controller yet"
        )

        # THE BUG: get_session() returns None even though session is in _acp_sessions
        result = manager.get_session(session_id)

        assert result is not None, (
            f"RED FLAG: get_session({session_id!r}) returned None even though "
            f"the session exists in _acp_sessions.\n\n"
            f"This means create_task(session.send_available_commands_update()) "
            f"in new_session/load_session/resume_session is silently skipped, "
            f"and available_commands_update is never sent to the client."
        )
        assert result is mock_session

    def test_returns_none_when_not_in_either(self, minimal_pool: AgentPool):
        """get_session() should return None for nonexistent sessions."""
        manager, _ = _make_manager_with_controller(minimal_pool)
        result = manager.get_session("nonexistent")
        assert result is None

    def test_returns_session_when_in_both(self, minimal_pool: AgentPool):
        """get_session() should work when session is in both places."""
        manager, _controller = _make_manager_with_controller(minimal_pool)
        session_id = "sess-in-both-001"

        mock_session: MagicMock = MagicMock()
        mock_session.session_id = session_id
        manager._acp_sessions[session_id] = mock_session

        # Register with controller too (create a real session state)
        # The controller naturally has no session, so we just verify
        # that get_session returns from _acp_sessions
        result = manager.get_session(session_id)
        assert result is not None
        assert result is mock_session

    def test_returns_session_without_controller(self, minimal_pool: AgentPool):
        """get_session() should work when _session_controller is None.

        (no SessionPool active).

        """
        original_sp = minimal_pool._session_pool
        try:
            manager = _make_manager_without_controller(minimal_pool)
            session_id = "sess-no-controller-001"

            mock_session: MagicMock = MagicMock()
            mock_session.session_id = session_id
            manager._acp_sessions[session_id] = mock_session

            assert manager._session_controller is None
            result = manager.get_session(session_id)
            assert result is not None
            assert result is mock_session
        finally:
            minimal_pool._session_pool = original_sp
