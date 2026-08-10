"""Tests for list_session_ids cross-agent filtering.

Verifies that ``list_session_ids`` with ``cwd`` returns sessions across
ALL agent names for that working directory — not just the current agent.

This is critical for the TUI "switch session" dialog: when the default_agent
changes (e.g. ``technical_assistant`` → ``engineer``), old sessions must
remain visible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from wolfharness.sessions.models import SessionData
from wolfharness_config.storage import SQLStorageConfig
from wolfharness_storage.sql_provider import SQLModelProvider


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from pathlib import Path


def _make_session(
    session_id: str,
    *,
    agent_name: str = "agent_a",
    cwd: str = "/tmp/project",
) -> SessionData:
    """Create a single SessionData object for testing."""
    return SessionData(
        session_id=session_id,
        agent_name=agent_name,
        cwd=cwd,
        metadata={"title": f"Session {session_id}"},
    )


class TestListSessionIdsCrossAgent:
    """Tests for list_session_ids with cwd filtering across agent names."""

    @pytest.fixture
    def provider(self, tmp_path: Path) -> SQLModelProvider:
        """Create a SQL provider with temp database."""
        db_path = tmp_path / "test_cross_agent.db"
        config = SQLStorageConfig(url=f"sqlite:///{db_path}")
        return SQLModelProvider(config)

    async def test_list_session_ids_with_cwd_returns_all_agents(
        self, provider: SQLModelProvider
    ) -> None:
        """list_session_ids(cwd=...) should return sessions for ALL agent names."""
        shared_cwd = "/tmp/project_x"
        sessions = [
            _make_session("sess_a1", agent_name="agent_a", cwd=shared_cwd),
            _make_session("sess_a2", agent_name="agent_a", cwd=shared_cwd),
            _make_session("sess_b1", agent_name="agent_b", cwd=shared_cwd),
            _make_session("sess_other", agent_name="agent_c", cwd="/tmp/other_project"),
        ]

        async with provider:
            for s in sessions:
                await provider.save_session(s)

            # With cwd filter — should return ALL sessions for shared_cwd
            result = await provider.list_session_ids(cwd=shared_cwd)

        result_set = set(result)
        assert "sess_a1" in result_set, "agent_a session missing from cwd filter"
        assert "sess_a2" in result_set, "agent_a session missing from cwd filter"
        assert "sess_b1" in result_set, "agent_b session missing from cwd filter"
        assert "sess_other" not in result_set, "other-project session should be excluded"

    async def test_list_session_ids_cwd_no_agent_name_filter(
        self, provider: SQLModelProvider
    ) -> None:
        """list_session_ids(cwd=..., agent_name=None) returns all agents for that cwd."""
        shared_cwd = "/tmp/project_y"
        sessions = [
            _make_session("sess_x1", agent_name="engineer", cwd=shared_cwd),
            _make_session("sess_x2", agent_name="technical_assistant", cwd=shared_cwd),
        ]

        async with provider:
            for s in sessions:
                await provider.save_session(s)

            result = await provider.list_session_ids(cwd=shared_cwd, agent_name=None)

        assert set(result) == {"sess_x1", "sess_x2"}

    async def test_list_session_ids_cwd_combined_with_agent_name(
        self, provider: SQLModelProvider
    ) -> None:
        """list_session_ids(cwd=..., agent_name=...) applies both filters."""
        shared_cwd = "/tmp/project_z"
        sessions = [
            _make_session("sess_a1", agent_name="agent_a", cwd=shared_cwd),
            _make_session("sess_b1", agent_name="agent_b", cwd=shared_cwd),
        ]

        async with provider:
            for s in sessions:
                await provider.save_session(s)

            result = await provider.list_session_ids(cwd=shared_cwd, agent_name="agent_a")

        assert result == ["sess_a1"]

    async def test_list_session_ids_cwd_no_match(self, provider: SQLModelProvider) -> None:
        """list_session_ids(cwd=...) with non-existent cwd returns empty list."""
        sessions = [
            _make_session("sess_a1", agent_name="agent_a", cwd="/tmp/real_project"),
        ]

        async with provider:
            for s in sessions:
                await provider.save_session(s)

            result = await provider.list_session_ids(cwd="/tmp/nonexistent")

        assert result == []

    async def test_list_session_ids_without_cwd_still_works(
        self, provider: SQLModelProvider
    ) -> None:
        """Backward compatibility: list_session_ids() without cwd still works."""
        sessions = [
            _make_session("sess_a1", agent_name="agent_a", cwd="/tmp/p1"),
            _make_session("sess_b1", agent_name="agent_b", cwd="/tmp/p2"),
        ]

        async with provider:
            for s in sessions:
                await provider.save_session(s)

            # No cwd, no agent_name — return all
            result_all = await provider.list_session_ids()
            # With agent_name only — backward compat
            result_a = await provider.list_session_ids(agent_name="agent_a")

        assert set(result_all) == {"sess_a1", "sess_b1"}
        assert result_a == ["sess_a1"]
