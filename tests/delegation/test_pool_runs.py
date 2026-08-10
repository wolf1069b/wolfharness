"""Tests for AgentPool run facade methods.

Covers list_active_runs, cancel_run, and get_run delegation to SessionPool,
including graceful handling when no session pool is available.
"""

from __future__ import annotations

import pytest

from wolfharness import AgentPool, AgentsManifest, NativeAgentConfig
from wolfharness.orchestrator import RunHandle


@pytest.fixture
def basic_manifest() -> AgentsManifest:
    """Create a minimal manifest with one agent."""
    agent_config = NativeAgentConfig(
        name="test_agent",
        model="test",
        system_prompt="You are a test agent",
    )
    return AgentsManifest(agents={"test_agent": agent_config})


# =============================================================================
# Standalone mode (no session pool)
# =============================================================================


class TestAgentPoolRunFacadeStandalone:
    """Test facade methods when session_pool is None."""

    def test_agent_pool_list_active_runs_standalone_returns_empty(
        self,
        basic_manifest: AgentsManifest,
    ) -> None:
        """list_active_runs should return [] when session_pool is None."""
        pool = AgentPool(basic_manifest)
        assert pool.session_pool is None
        result = pool.list_active_runs()
        assert result == []

    def test_agent_pool_cancel_run_standalone_raises(
        self,
        basic_manifest: AgentsManifest,
    ) -> None:
        """cancel_run should raise RuntimeError when session_pool is None."""
        pool = AgentPool(basic_manifest)
        assert pool.session_pool is None
        with pytest.raises(RuntimeError, match="No session pool available"):
            pool.cancel_run("some-run-id")

    def test_agent_pool_get_run_standalone_returns_none(
        self,
        basic_manifest: AgentsManifest,
    ) -> None:
        """get_run should return None when session_pool is None."""
        pool = AgentPool(basic_manifest)
        assert pool.session_pool is None
        result = pool.get_run("some-run-id")
        assert result is None


# =============================================================================
# Delegation to SessionPool
# =============================================================================


class TestAgentPoolRunFacadeDelegation:
    """Test facade methods delegate to SessionPool when available."""

    @pytest.mark.integration
    async def test_agent_pool_list_active_runs_delegates_to_session_pool(
        self,
        basic_manifest: AgentsManifest,
    ) -> None:
        """list_active_runs should delegate to session_pool.active_runs."""
        async with AgentPool(basic_manifest) as pool:
            assert pool.session_pool is not None
            result = pool.list_active_runs()
            # No active runs in a fresh pool
            assert result == []

    @pytest.mark.integration
    async def test_agent_pool_get_run_delegates_to_session_pool(
        self,
        basic_manifest: AgentsManifest,
    ) -> None:
        """get_run should delegate to session_pool.get_run."""
        async with AgentPool(basic_manifest) as pool:
            assert pool.session_pool is not None
            result = pool.get_run("nonexistent-run-id")
            assert result is None

    @pytest.mark.integration
    async def test_agent_pool_cancel_run_delegates_to_session_pool(
        self,
        basic_manifest: AgentsManifest,
    ) -> None:
        """cancel_run should delegate to session_pool.cancel_run."""
        async with AgentPool(basic_manifest) as pool:
            assert pool.session_pool is not None
            with pytest.raises(ValueError, match="No active run found"):
                pool.cancel_run("nonexistent-run-id")

    @pytest.mark.integration
    async def test_agent_pool_list_active_runs_returns_run_handles(
        self,
        basic_manifest: AgentsManifest,
    ) -> None:
        """list_active_runs should return a list of RunHandle objects."""
        async with AgentPool(basic_manifest) as pool:
            result = pool.list_active_runs()
            assert isinstance(result, list)
            for handle in result:
                assert isinstance(handle, RunHandle)
