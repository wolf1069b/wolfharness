"""Tests for config/agent discovery routes after per-session agent isolation.

Verifies:
1. Config and agent discovery paths return correct global metadata
2. All route modules import cleanly (no syntax errors, no conflict markers)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from wolfharness.models.manifest import AgentsManifest
from wolfharness.storage import StorageManager
from wolfharness.utils.streams import FileOpsTracker
from wolfharness.utils.todos import TodoTracker
from wolfharness_server.opencode_server.dependencies import get_state
from wolfharness_server.opencode_server.routes.agent_routes import router as agent_router
from wolfharness_server.opencode_server.routes.config_routes import router as config_router
from wolfharness_server.opencode_server.routes.global_routes import router as global_router
from wolfharness_server.opencode_server.state import ServerState


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _server_state(tmp_path: Path) -> ServerState:
    """Build a ServerState with a mock agent for discovery route tests."""
    from wolfharness_config.storage import MemoryStorageConfig, StorageConfig

    storage_manager = StorageManager(config=StorageConfig(providers=[MemoryStorageConfig()]))
    file_ops = FileOpsTracker()
    todos = TodoTracker()
    manifest = AgentsManifest(config_file_path="/tmp/test-pool")

    pool = Mock()
    pool.storage = storage_manager
    pool.file_ops = file_ops
    pool.todos = todos
    pool.manifest = manifest
    pool.manifest.agents = {}
    pool.skill_provider = None
    pool.skills = None

    env = Mock()
    env.cwd = str(tmp_path)

    agent = Mock()
    agent.name = "test-agent"
    agent.env = env
    agent._input_provider = None
    agent.agent_pool = pool
    agent.host_context = pool
    agent._agent_pool = pool  # state.py resolves _pool via agent._agent_pool
    agent.storage = storage_manager
    agent.get_available_models = AsyncMock(return_value=[])
    agent.get_mcp_server_info = AsyncMock(return_value={})
    agent.list_sessions = AsyncMock(return_value=[])
    agent.set_model = AsyncMock()

    return ServerState(working_dir=str(tmp_path), agent=agent)


@pytest.fixture
def client(_server_state: ServerState) -> TestClient:
    """Create a test client with config, agent, and global routers."""
    app = FastAPI()
    app.include_router(config_router)
    app.include_router(agent_router)
    app.include_router(global_router)
    app.dependency_overrides[get_state] = lambda: _server_state
    return TestClient(app)


# ---------------------------------------------------------------------------
# Test 1: Config/agent discovery paths return correct global metadata
# ---------------------------------------------------------------------------


class TestConfigDiscoveryMetadata:
    """Verify config/agent discovery routes return global metadata."""

    def test_get_config_returns_consistent_metadata(self, client: TestClient) -> None:
        """GET /config and GET /global/config return the same metadata."""
        config_resp = client.get("/config")
        global_resp = client.get("/global/config")
        assert config_resp.status_code == 200
        assert global_resp.status_code == 200
        assert config_resp.json() == global_resp.json()

    def test_get_providers_uses_pool_manifest(self, client: TestClient) -> None:
        """GET /config/providers returns data derived from pool manifest."""
        resp = client.get("/config/providers")
        assert resp.status_code == 200
        data = resp.json()
        # Should have "providers" and "default" keys
        assert "providers" in data
        assert "default" in data
        # With an empty mock manifest, providers may be empty but the
        # response structure must always be present.
        assert isinstance(data["providers"], list)

    def test_list_providers_returns_consistent_structure(self, client: TestClient) -> None:
        """GET /provider returns all/default/connected structure."""
        resp = client.get("/provider")
        assert resp.status_code == 200
        data = resp.json()
        assert "all" in data
        assert "default" in data
        assert "connected" in data

    def test_list_agents_returns_pool_agents(self, client: TestClient) -> None:
        """GET /agent returns agents from the pool."""
        resp = client.get("/agent")
        assert resp.status_code == 200
        data = resp.json()
        # With empty all_agents, the fallback default agent is returned
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["name"] == "default"

    def test_get_health_returns_version(self, client: TestClient) -> None:
        """GET /global/health returns healthy status with version."""
        resp = client.get("/global/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["healthy"] is True
        assert "version" in data


# ---------------------------------------------------------------------------
# Test 2: All route modules import cleanly (no syntax errors, no markers)
# ---------------------------------------------------------------------------


class TestRouteModuleImports:
    """Verify all route modules can be imported without errors or conflict markers."""

    @pytest.mark.parametrize(
        "module_name",
        [
            "wolfharness_server.opencode_server.routes.config_routes",
            "wolfharness_server.opencode_server.routes.agent_routes",
            "wolfharness_server.opencode_server.routes.global_routes",
            "wolfharness_server.opencode_server.routes.pty_routes",
        ],
    )
    def test_module_imports_cleanly(self, module_name: str) -> None:
        """Each route module should import without errors."""
        mod = importlib.import_module(module_name)
        assert mod is not None

    @pytest.mark.parametrize(
        "module_name",
        [
            "wolfharness_server.opencode_server.routes.config_routes",
            "wolfharness_server.opencode_server.routes.agent_routes",
            "wolfharness_server.opencode_server.routes.global_routes",
            "wolfharness_server.opencode_server.routes.pty_routes",
        ],
    )
    def test_no_merge_conflict_markers(self, module_name: str) -> None:
        """Route modules must not contain merge-conflict markers."""
        import inspect

        mod = importlib.import_module(module_name)
        source = inspect.getsource(mod)
        conflict_markers = ["<<<<<<<", "=======", ">>>>>>>"]
        for marker in conflict_markers:
            assert marker not in source, f"Merge conflict marker {marker!r} found in {module_name}"


# ---------------------------------------------------------------------------
# Test 3: Per-session agent model propagation via PATCH /config
# ---------------------------------------------------------------------------


class TestConfigModelPropagation:
    """Verify model changes via PATCH /config propagate to per-session agents."""

    def test_patch_config_model_propagates_to_shared_agent(
        self,
        _server_state: ServerState,  # noqa: PT019
        client: TestClient,
    ) -> None:
        """PATCH /config with model should update the shared server agent."""
        # Keep a reference to the shared agent's set_model mock
        shared_set_model: AsyncMock = _server_state.agent.set_model  # type: ignore[assignment]

        # Patch the model via config
        resp = client.patch("/config", json={"model": "openai/gpt-4o"})
        assert resp.status_code == 200

        # Shared agent should have set_model called
        shared_set_model.assert_awaited_once_with("openai/gpt-4o")
