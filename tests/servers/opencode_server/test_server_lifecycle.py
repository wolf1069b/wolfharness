"""Tests for server bootstrap and shutdown lifecycle wiring.

Validates that the server lifespan correctly:
- Cleans up all per-session agents during shutdown
- Cancels and waits for background tasks during shutdown
- Preserves pool/storage access after ServerState creation
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from wolfharness_server.opencode_server.state import ServerState


pytestmark = pytest.mark.integration


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_env() -> Mock:
    """Create a mock agent environment."""
    from upathtools.filesystems import AsyncLocalFileSystem

    env = Mock()
    fs = AsyncLocalFileSystem()
    env.get_fs = Mock(return_value=fs)
    env.cwd = "/tmp/test"
    return env


@pytest.fixture
def mock_pool() -> Mock:
    """Create a mock agent pool with minimal attributes."""
    pool = Mock()
    pool.manifest = Mock()
    pool.manifest.agents = {}
    return pool


@pytest.fixture
def shared_agent(mock_env: Mock, mock_pool: Mock) -> Mock:
    """Create the shared (default) mock agent."""
    agent = Mock()
    agent.name = "test-agent"
    agent.env = mock_env
    agent._input_provider = None
    agent.agent_pool = mock_pool
    agent.host_context = mock_pool
    agent._agent_pool = mock_pool  # state.py resolves _pool via agent._agent_pool
    agent.storage = None
    return agent


@pytest.fixture
def state(shared_agent: Mock, tmp_path: Any) -> ServerState:
    """Create a ServerState with a shared mock agent.

    Patches ``_create_session_agent`` so each call returns a fresh mock
    agent, enabling per-session cleanup testing without a real
    ``NativeAgentConfig``.
    """
    import tempfile

    with tempfile.TemporaryDirectory(prefix="lifecycle-test-") as tmpdir:
        st = ServerState(working_dir=tmpdir, agent=shared_agent)
        call_count = 0

        def _fake_create(session_id: str) -> Mock:
            nonlocal call_count
            call_count += 1
            agent = Mock()
            agent.name = f"session-agent-{call_count}"
            agent.session_id = session_id
            agent.__aexit__ = AsyncMock(return_value=False)
            return agent

        st._create_session_agent = _fake_create  # type: ignore[method-assign]
        yield st


# =============================================================================
# Test: shutdown cancels background tasks
# =============================================================================


async def test_shutdown_cancels_background_tasks(state: ServerState) -> None:
    """Shutdown sequence cancels and waits for all background tasks.

    Background tasks created via create_background_task must be cancelled
    and gathered during shutdown so no orphaned coroutines remain.
    """
    completed_normally = False

    async def long_running() -> None:
        """A task that runs until cancelled."""
        nonlocal completed_normally
        await asyncio.sleep(3600)
        completed_normally = True

    # Create a background task
    task = state.create_background_task(long_running(), name="test-bg-task")

    # Simulate the shutdown sequence from server.py lifespan
    await state.cleanup_tasks()

    # Task should have been cancelled (not completed normally)
    assert task.cancelled() or task.done()
    assert not completed_normally

    # Background tasks set should be empty after cleanup
    assert len(state.background_tasks) == 0


# =============================================================================
# Test: bootstrap preserves pool/storage access
# =============================================================================


async def test_bootstrap_preserves_pool_access(shared_agent: Mock, mock_pool: Mock) -> None:
    """ServerState creation caches pool and storage for later access.

    After __post_init__, state.pool and state.storage must return the
    cached references without going through the shared agent.
    """
    from wolfharness.storage import StorageManager
    from wolfharness_config.storage import MemoryStorageConfig, StorageConfig

    storage_manager = StorageManager(config=StorageConfig(providers=[MemoryStorageConfig()]))
    shared_agent.storage = storage_manager

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        st = ServerState(working_dir=tmpdir, agent=shared_agent)

        # Pool and storage should be accessible
        assert st.pool is mock_pool
        assert st.storage is storage_manager

        # They should return the same cached objects on repeated access
        assert st.pool is st._pool
        assert st.storage is st._storage
