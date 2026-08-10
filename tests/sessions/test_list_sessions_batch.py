"""Tests for load_sessions_batch — N+1 query elimination.

Verifies that StorageProvider.load_sessions_batch loads all sessions in a
single call without falling back to per-ID load_session (N+1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from wolfharness.sessions.models import SessionData
from wolfharness_config.storage import MemoryStorageConfig, SQLStorageConfig
from wolfharness_storage.memory_provider import MemoryStorageProvider
from wolfharness_storage.sql_provider import SQLModelProvider


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from pathlib import Path


def _make_sessions(
    n: int, *, agent_name: str = "test_agent", prefix: str = "sess"
) -> list[SessionData]:
    """Create N SessionData objects for testing."""
    return [
        SessionData(
            session_id=f"{prefix}_{i}",
            agent_name=agent_name,
            cwd=f"/tmp/project_{i}",
            metadata={"title": f"Session {i}"},
        )
        for i in range(n)
    ]


class TestSQLModelProviderLoadSessionsBatch:
    """Tests for SQLModelProvider.load_sessions_batch."""

    @pytest.fixture
    def provider(self, tmp_path: Path) -> SQLModelProvider:
        """Create a SQL provider with temp database."""
        db_path = tmp_path / "test_batch.db"
        config = SQLStorageConfig(url=f"sqlite:///{db_path}")
        return SQLModelProvider(config)

    async def test_batch_returns_all_sessions(self, provider: SQLModelProvider) -> None:
        """Batch load should return all saved sessions."""
        sessions = _make_sessions(5)
        session_ids = [s.session_id for s in sessions]

        async with provider:
            for s in sessions:
                await provider.save_session(s)
            result = await provider.load_sessions_batch(session_ids)

        assert len(result) == 5
        result_ids = {s.session_id for s in result}
        assert result_ids == set(session_ids)

    async def test_batch_returns_sessions_with_titles(self, provider: SQLModelProvider) -> None:
        """Batch load should return SessionData with titles populated from metadata."""
        sessions = _make_sessions(3)
        session_ids = [s.session_id for s in sessions]

        async with provider:
            for s in sessions:
                await provider.save_session(s)
            result = await provider.load_sessions_batch(session_ids)

        for s in result:
            assert s.title is not None
            assert s.title.startswith("Session")

    async def test_batch_empty_ids(self, provider: SQLModelProvider) -> None:
        """Batch load with empty list should return empty list."""
        async with provider:
            result = await provider.load_sessions_batch([])

        assert result == []

    async def test_batch_nonexistent_ids(self, provider: SQLModelProvider) -> None:
        """Batch load with nonexistent IDs should return empty list."""
        async with provider:
            result = await provider.load_sessions_batch(["nonexistent_1", "nonexistent_2"])

        assert result == []

    async def test_batch_partial_ids(self, provider: SQLModelProvider) -> None:
        """Batch load with mix of existing and nonexistent IDs returns only found sessions."""
        sessions = _make_sessions(3)
        session_ids = [s.session_id for s in sessions]

        async with provider:
            for s in sessions:
                await provider.save_session(s)
            result = await provider.load_sessions_batch([
                session_ids[0],
                "nonexistent",
                session_ids[2],
            ])

        assert len(result) == 2
        result_ids = {s.session_id for s in result}
        assert result_ids == {session_ids[0], session_ids[2]}

    async def test_batch_does_not_call_load_session_per_id(
        self, provider: SQLModelProvider
    ) -> None:
        """Batch load must NOT call load_session N times (N+1 query test)."""
        sessions = _make_sessions(5)
        session_ids = [s.session_id for s in sessions]

        async with provider:
            for s in sessions:
                await provider.save_session(s)

            # Patch load_session to track calls — it should NOT be called
            with patch.object(
                provider,
                "load_session",
                new_callable=AsyncMock,
            ) as mock_load:
                result = await provider.load_sessions_batch(session_ids)

                # load_session should not be called at all — batch does its own query
                mock_load.assert_not_called()

        assert len(result) == 5

    async def test_batch_filter_by_agent_name(self, provider: SQLModelProvider) -> None:
        """Batch load with agent_name filter returns only matching sessions."""
        agent_a_sessions = _make_sessions(3, agent_name="agent_a", prefix="a")
        agent_b_sessions = _make_sessions(2, agent_name="agent_b", prefix="b")
        all_sessions = agent_a_sessions + agent_b_sessions
        all_ids = [s.session_id for s in all_sessions]

        async with provider:
            for s in all_sessions:
                await provider.save_session(s)
            result = await provider.load_sessions_batch(all_ids, agent_name="agent_a")

        assert len(result) == 3
        assert all(s.agent_name == "agent_a" for s in result)


class TestMemoryProviderLoadSessionsBatch:
    """Tests for MemoryStorageProvider.load_sessions_batch."""

    @pytest.fixture
    def provider(self) -> MemoryStorageProvider:
        """Create a memory storage provider."""
        return MemoryStorageProvider(MemoryStorageConfig())

    async def test_batch_returns_all_sessions(self, provider: MemoryStorageProvider) -> None:
        """Batch load should return all saved sessions."""
        sessions = _make_sessions(5)
        session_ids = [s.session_id for s in sessions]

        async with provider:
            for s in sessions:
                await provider.save_session(s)
            result = await provider.load_sessions_batch(session_ids)

        assert len(result) == 5
        result_ids = {s.session_id for s in result}
        assert result_ids == set(session_ids)

    async def test_batch_empty_ids(self, provider: MemoryStorageProvider) -> None:
        """Batch load with empty list returns empty list."""
        async with provider:
            result = await provider.load_sessions_batch([])

        assert result == []

    async def test_batch_filter_by_agent_name(self, provider: MemoryStorageProvider) -> None:
        """Batch load with agent_name filter returns only matching sessions."""
        agent_a_sessions = _make_sessions(3, agent_name="agent_a", prefix="a")
        agent_b_sessions = _make_sessions(2, agent_name="agent_b", prefix="b")
        all_sessions = agent_a_sessions + agent_b_sessions
        all_ids = [s.session_id for s in all_sessions]

        async with provider:
            for s in all_sessions:
                await provider.save_session(s)
            result = await provider.load_sessions_batch(all_ids, agent_name="agent_a")

        assert len(result) == 3
        assert all(s.agent_name == "agent_a" for s in result)
