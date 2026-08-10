"""Compatibility tests for OpenCode experimental workspace routes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def test_list_workspaces_returns_local_response(client: TestClient) -> None:
    """GET /experimental/workspace should not fall through to the Web UI proxy."""
    response = client.get("/experimental/workspace")

    assert response.status_code == 200
    assert response.json() == []


def test_get_experimental_capabilities(client: TestClient) -> None:
    """GET /experimental/capabilities reports no background-subagent support.

    OpenCode requires the response body to include the ``backgroundSubagents``
    field with a boolean value.
    """
    response = client.get("/experimental/capabilities")

    assert response.status_code == 200
    assert response.json() == {"backgroundSubagents": False}


def test_get_workspace_status_returns_empty_array(client: TestClient) -> None:
    """GET /experimental/workspace/status returns an empty status array.

    We do not maintain per-workspace event-stream connections, so the result
    is always an empty list matching OpenCode SDK's
    ``Array<WorkspaceEventConnectionStatus>`` contract.
    """
    response = client.get(
        "/experimental/workspace/status",
        params={"directory": "src", "workspace": "worktree-foo"},
    )

    assert response.status_code == 200
    assert response.json() == []
