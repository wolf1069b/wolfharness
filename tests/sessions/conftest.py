"""Test fixtures for session storage tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from wolfharness.sessions.models import SessionData
from wolfharness_config.storage import SQLStorageConfig
from wolfharness_storage.sql_provider import SQLModelProvider


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def _clear_engine_cache():
    """Clear engine cache to avoid cross-test contamination."""
    from wolfharness_config.storage import _engine_cache

    _engine_cache.clear()
    yield
    _engine_cache.clear()


@pytest.fixture
async def sql_model_provider(tmp_path: Path) -> SQLModelProvider:
    """Create a SQLModelProvider with a temp SQLite DB.

    Yields an initialized provider (entered as async context manager).
    Cleans up the engine on exit.
    """
    db_path = tmp_path / "test_sessions.db"
    config = SQLStorageConfig(url=f"sqlite:///{db_path}")
    provider = SQLModelProvider(config)
    async with provider:
        yield provider


@pytest.fixture
def sample_session_data() -> SessionData:
    """Create a SessionData with all fields populated."""
    return SessionData(
        session_id="test-rt-001",
        agent_name="test-agent",
        pool_id="test-pool",
        project_id="test-project",
        parent_id=None,
        version="1",
        cwd="/tmp/test",
        agent_type="native",
        sdk_session_id="sdk-123",
        metadata={"title": "Test Session", "custom_key": "custom_value"},
        status="active",
    )
