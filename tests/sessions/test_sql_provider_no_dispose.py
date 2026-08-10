"""Tests that SQLModelProvider session methods use self.engine directly.

After the elimination of SQLSessionStore, SQLModelProvider session methods
use self.engine directly. These tests verify repeated operations do not
dispose the shared engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from wolfharness.sessions import SessionData
from wolfharness_config.storage import SQLStorageConfig
from wolfharness_storage.sql_provider import SQLModelProvider


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from pathlib import Path


class TestSQLProviderEngineNotDisposed:
    """Assert repeated session method calls do not dispose the shared engine."""

    @pytest.fixture
    def provider(self, tmp_path: Path) -> SQLModelProvider:
        """Create a SQL provider with temp database."""
        db_path = tmp_path / "test_engine_not_disposed.db"
        config = SQLStorageConfig(url=f"sqlite:///{db_path}")
        return SQLModelProvider(config)

    async def test_multiple_loads_do_not_dispose_engine(self, provider: SQLModelProvider) -> None:
        """Calling load_session multiple times must not dispose self.engine."""
        data = SessionData(session_id="s1", agent_name="agent1")
        async with provider:
            await provider.save_session(data)
            # Load multiple times — if engine were disposed between calls,
            # subsequent loads would fail.
            for _ in range(5):
                loaded = await provider.load_session("s1")
                assert loaded is not None
                assert loaded.session_id == "s1"

    async def test_engine_usable_after_multiple_session_ops(
        self, provider: SQLModelProvider
    ) -> None:
        """Engine must remain usable after save/load/delete/list cycles."""
        async with provider:
            # Save
            await provider.save_session(SessionData(session_id="s1", agent_name="a1", pool_id="p1"))
            await provider.save_session(SessionData(session_id="s2", agent_name="a2", pool_id="p1"))

            # Load
            loaded = await provider.load_session("s1")
            assert loaded is not None

            # List
            ids = await provider.list_session_ids(pool_id="p1")
            assert len(ids) == 2

            # Delete
            deleted = await provider.delete_session("s1")
            assert deleted is True

            # Engine still works for another query
            loaded2 = await provider.load_session("s2")
            assert loaded2 is not None
