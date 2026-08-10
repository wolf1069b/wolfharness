"""Tests for OpenCode 1.4.4+ global compatibility routes.

Covers:
- GET /global/config  (delegates to /config)
- PATCH /global/config  (delegates to /config)
- POST /global/dispose  (stub no-op)
- POST /global/upgrade  (stub no-op)
"""

from __future__ import annotations

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
from wolfharness_server.opencode_server.routes.config_routes import router as config_router
from wolfharness_server.opencode_server.routes.global_routes import router as global_router
from wolfharness_server.opencode_server.state import ServerState


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def _server_state(tmp_path: Path) -> ServerState:
    """Build a ServerState with a mock agent for config route tests."""
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

    return ServerState(working_dir=str(tmp_path), agent=agent)


@pytest.fixture
def client(_server_state: ServerState) -> TestClient:
    """Create a test client with both global and config routers."""
    app = FastAPI()
    app.include_router(config_router)
    app.include_router(global_router)
    app.dependency_overrides[get_state] = lambda: _server_state
    return TestClient(app)


class TestGlobalConfigRoutes:
    """Tests for GET/PATCH /global/config."""

    def test_get_global_config_returns_config(self, client: TestClient) -> None:
        """GET /global/config should return a Config object."""
        resp = client.get("/global/config")
        assert resp.status_code == 200
        data = resp.json()
        # Config should have at least keybinds and watcher fields
        assert "keybinds" in data or "model" in data

    def test_get_global_config_matches_get_config(self, client: TestClient) -> None:
        """GET /global/config should return the same data as GET /config."""
        global_resp = client.get("/global/config")
        config_resp = client.get("/config")
        assert global_resp.status_code == 200
        assert config_resp.status_code == 200
        assert global_resp.json() == config_resp.json()

    def test_patch_global_config_updates_model(self, client: TestClient) -> None:
        """PATCH /global/config should update config fields."""
        # First, get current config
        get_resp = client.get("/global/config")
        assert get_resp.status_code == 200

        # Patch the theme
        patch_resp = client.patch("/global/config", json={"theme": "dark"})
        assert patch_resp.status_code == 200
        data = patch_resp.json()
        assert data.get("theme") == "dark"

    def test_patch_global_config_matches_patch_config(self, client: TestClient) -> None:
        """PATCH /global/config should behave identically to PATCH /config."""
        # Set via /config
        r1 = client.patch("/config", json={"theme": "light"})
        assert r1.status_code == 200

        # Set via /global/config
        r2 = client.patch("/global/config", json={"theme": "dark"})
        assert r2.status_code == 200

        # Both should update the same underlying state
        final = client.get("/global/config")
        assert final.json().get("theme") == "dark"


class TestGlobalDisposeRoute:
    """Tests for POST /global/dispose."""

    def test_global_dispose_returns_success(self, client: TestClient) -> None:
        """POST /global/dispose should return a success stub response."""
        resp = client.post("/global/dispose")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "no-op" in data["message"]

    def test_global_dispose_does_not_crash_server(self, client: TestClient) -> None:
        """Server should still respond after /global/dispose."""
        client.post("/global/dispose")
        # Server should still work
        resp = client.get("/global/health")
        assert resp.status_code == 200


class TestGlobalUpgradeRoute:
    """Tests for POST /global/upgrade."""

    def test_global_upgrade_returns_stub(self, client: TestClient) -> None:
        """POST /global/upgrade should return a stub response."""
        resp = client.post("/global/upgrade")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["upgraded"] is False

    def test_global_upgrade_does_not_crash_server(self, client: TestClient) -> None:
        """Server should still respond after /global/upgrade."""
        client.post("/global/upgrade")
        resp = client.get("/global/health")
        assert resp.status_code == 200
