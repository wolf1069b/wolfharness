"""Tests for session persistence across close/shutdown/restart.

Verifies that:
- SQLModelProvider save/load roundtrips the `status` field
- close_session() marks sessions as "closed" instead of deleting them
- Sessions survive SessionPool.shutdown() and can be loaded afterwards
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from wolfharness.orchestrator.core import SessionController
from wolfharness.sessions.models import SessionData
from wolfharness_config.storage import SQLStorageConfig
from wolfharness_storage.sql_provider import SQLModelProvider


if TYPE_CHECKING:
    from wolfharness import AgentPool


pytestmark = pytest.mark.unit


def make_session_data(
    session_id: str = "sess-persist-1",
    agent_name: str = "test-agent",
    status: str = "active",
) -> SessionData:
    """Create SessionData for persistence tests."""
    return SessionData(
        session_id=session_id,
        agent_name=agent_name,
        status=status,
    )


@pytest.fixture
def _clear_engine_cache():
    """Clear engine cache to avoid cross-test contamination."""
    from wolfharness_config.storage import _engine_cache

    _engine_cache.clear()
    yield
    _engine_cache.clear()


# ===================================================================
# SQLModelProvider status field roundtrip
# ===================================================================


@pytest.mark.usefixtures("_clear_engine_cache")
class TestSQLModelProviderStatusRoundtrip:
    """SQLModelProvider should persist and restore the status field."""

    @pytest.mark.anyio
    async def test_save_load_status_active(self) -> None:
        """Status 'active' survives a save/load roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            config = SQLStorageConfig(url=f"sqlite:///{db_path}")

            async with SQLModelProvider(config) as store:
                data = make_session_data(status="active")
                await store.save_session(data)
                loaded = await store.load_session(data.session_id)

            assert loaded is not None
            assert loaded.status == "active"

    @pytest.mark.anyio
    async def test_save_load_status_closed(self) -> None:
        """Status 'closed' survives a save/load roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            config = SQLStorageConfig(url=f"sqlite:///{db_path}")

            async with SQLModelProvider(config) as store:
                data = make_session_data(status="closed")
                await store.save_session(data)
                loaded = await store.load_session(data.session_id)

            assert loaded is not None
            assert loaded.status == "closed"

    @pytest.mark.anyio
    async def test_save_load_status_checkpointed(self) -> None:
        """Status 'checkpointed' survives a save/load roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            config = SQLStorageConfig(url=f"sqlite:///{db_path}")

            async with SQLModelProvider(config) as store:
                data = make_session_data(status="checkpointed")
                await store.save_session(data)
                loaded = await store.load_session(data.session_id)

            assert loaded is not None
            assert loaded.status == "checkpointed"

    @pytest.mark.anyio
    async def test_status_update_via_save(self) -> None:
        """Saving with a new status updates the existing record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            config = SQLStorageConfig(url=f"sqlite:///{db_path}")

            async with SQLModelProvider(config) as store:
                session_id = "sess-update-1"
                await store.save_session(make_session_data(session_id=session_id, status="active"))

                data = await store.load_session(session_id)
                assert data is not None
                assert data.status == "active"

                updated = data.model_copy(update={"status": "closed"})
                await store.save_session(updated)

                reloaded = await store.load_session(session_id)
                assert reloaded is not None
                assert reloaded.status == "closed"


# ===================================================================
# close_session marks as closed instead of deleting
# ===================================================================


class TestCloseSessionMarksClosed:
    """close_session() should mark sessions as 'closed', not delete them."""

    @pytest.mark.anyio
    async def test_close_marks_closed_not_deleted(self, minimal_pool: AgentPool) -> None:
        """close_session() saves with status='closed' and does NOT call delete."""
        mock_store = MagicMock()
        mock_store.load_session = AsyncMock(return_value=make_session_data())
        mock_store.save_session = AsyncMock(return_value=None)
        mock_store.delete_session = AsyncMock(return_value=True)

        ctrl = SessionController(pool=minimal_pool, store=mock_store)
        await ctrl.get_or_create_session("sess-1")
        await ctrl.close_session("sess-1")

        mock_store.delete_session.assert_not_awaited()
        closed_saves = [
            call
            for call in mock_store.save_session.await_args_list
            if call[0][0].status == "closed"
        ]
        assert len(closed_saves) >= 1

    @pytest.mark.anyio
    async def test_close_unlocked_marks_closed_not_deleted(self, minimal_pool: AgentPool) -> None:
        """_close_session_unlocked() also marks closed instead of deleting."""
        mock_store = MagicMock()
        mock_store.load_session = AsyncMock(return_value=make_session_data())
        mock_store.save_session = AsyncMock(return_value=None)
        mock_store.delete_session = AsyncMock(return_value=True)

        ctrl = SessionController(pool=minimal_pool, store=mock_store)
        await ctrl.get_or_create_session("sess-1")
        await ctrl._close_session_unlocked("sess-1")

        mock_store.delete_session.assert_not_awaited()
        closed_saves = [
            call
            for call in mock_store.save_session.await_args_list
            if call[0][0].status == "closed"
        ]
        assert len(closed_saves) >= 1

    @pytest.mark.anyio
    async def test_close_noop_when_not_in_store(self, minimal_pool: AgentPool) -> None:
        """_mark_session_closed does nothing if session is not in store."""
        mock_store = MagicMock()
        mock_store.load_session = AsyncMock(return_value=None)
        mock_store.save_session = AsyncMock(return_value=None)

        ctrl = SessionController(pool=minimal_pool, store=mock_store)
        await ctrl._mark_session_closed("nonexistent-session")

        mock_store.save_session.assert_not_awaited()


# ===================================================================
# Session survives shutdown via SQLModelProvider
# ===================================================================


@pytest.mark.usefixtures("_clear_engine_cache")
class TestSessionSurvivesShutdown:
    """Sessions should survive a full shutdown/restart cycle via SQL store."""

    @pytest.mark.anyio
    async def test_session_data_survives_shutdown_restart(self) -> None:
        """After shutdown + restart, session data is still loadable from the DB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "survive.db"
            config = SQLStorageConfig(url=f"sqlite:///{db_path}")

            async with SQLModelProvider(config) as store:
                await store.save_session(
                    make_session_data(session_id="sess-survive", status="active")
                )

            async with SQLModelProvider(config) as store2:
                loaded = await store2.load_session("sess-survive")

            assert loaded is not None
            assert loaded.session_id == "sess-survive"
            assert loaded.status == "active"

    @pytest.mark.anyio
    async def test_closed_session_survives_restart(self) -> None:
        """A session marked 'closed' survives restart and can be resumed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "closed.db"
            config = SQLStorageConfig(url=f"sqlite:///{db_path}")

            async with SQLModelProvider(config) as store:
                await store.save_session(
                    make_session_data(session_id="sess-closed", status="closed")
                )

            async with SQLModelProvider(config) as store2:
                loaded = await store2.load_session("sess-closed")

            assert loaded is not None
            assert loaded.status == "closed"

    @pytest.mark.anyio
    async def test_list_sessions_after_restart(self) -> None:
        """list_session_ids() returns sessions that were saved before restart."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "list.db"
            config = SQLStorageConfig(url=f"sqlite:///{db_path}")

            async with SQLModelProvider(config) as store:
                await store.save_session(make_session_data(session_id="sess-a", status="closed"))
                await store.save_session(make_session_data(session_id="sess-b", status="active"))

            async with SQLModelProvider(config) as store2:
                ids = await store2.list_session_ids()

            assert "sess-a" in ids
            assert "sess-b" in ids
