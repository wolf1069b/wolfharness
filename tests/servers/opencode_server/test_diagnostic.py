"""Tests for GET /global/diagnostic endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest

from wolfharness_server.opencode_server.routes.global_routes import (
    VERSION,
    GlobalEventFactory,
)
from wolfharness_server.opencode_server.state import ServerState


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    import asyncio

    from httpx import AsyncClient


# =============================================================================
# _MockState for unit tests (reuses pattern from test_global_event.py)
# =============================================================================


class _MockState:
    """Minimal ServerState-like object for diagnostic endpoint tests."""

    def __init__(self, working_dir: str | None = "/tmp/test_wd") -> None:
        self.working_dir = working_dir
        self.event_subscribers: list[asyncio.Queue[Any]] = []
        self._event_factory: GlobalEventFactory | None = None

    def get_event_factory(self) -> GlobalEventFactory:
        if self._event_factory is None:
            from wolfharness_storage.opencode_provider import helpers

            self._event_factory = GlobalEventFactory(
                directory=self.working_dir or "",
                project=helpers.compute_project_id(self.working_dir or ""),
            )
        return self._event_factory


# =============================================================================
# Integration tests using async_client (real FastAPI app)
# =============================================================================


@pytest.mark.anyio
async def test_diagnostic_returns_200(async_client: AsyncClient) -> None:
    """GET /global/diagnostic returns 200 status code."""
    response = await async_client.get("/global/diagnostic")
    assert response.status_code == 200


@pytest.mark.anyio
async def test_diagnostic_has_required_fields(async_client: AsyncClient) -> None:
    """GET /global/diagnostic returns JSON with directory, project, subscribers, serverVersion."""
    response = await async_client.get("/global/diagnostic")
    data = response.json()
    assert "directory" in data
    assert "project" in data
    assert "subscribers" in data
    assert "serverVersion" in data


@pytest.mark.anyio
async def test_diagnostic_directory_matches_working_dir(
    async_client: AsyncClient,
    server_state: ServerState,
) -> None:
    """Directory field equals state.working_dir."""
    response = await async_client.get("/global/diagnostic")
    data = response.json()
    assert data["directory"] == server_state.working_dir


@pytest.mark.anyio
async def test_diagnostic_subscribers_is_non_negative_integer(
    async_client: AsyncClient,
) -> None:
    """Subscribers field is an integer >= 0."""
    response = await async_client.get("/global/diagnostic")
    data = response.json()
    assert isinstance(data["subscribers"], int)
    assert data["subscribers"] >= 0


@pytest.mark.anyio
async def test_diagnostic_server_version_matches_constant(
    async_client: AsyncClient,
) -> None:
    """ServerVersion field matches the VERSION constant from global_routes."""
    response = await async_client.get("/global/diagnostic")
    data = response.json()
    assert data["serverVersion"] == VERSION


@pytest.mark.anyio
async def test_diagnostic_project_is_computed(async_client: AsyncClient) -> None:
    """Project field is computed via compute_project_id."""
    response = await async_client.get("/global/diagnostic")
    data = response.json()
    # project should be a non-empty string
    assert isinstance(data["project"], str)
    assert len(data["project"]) > 0


# =============================================================================
# Edge case: working_dir=None
# =============================================================================


def _make_state_with_none_working_dir() -> ServerState:
    """Create a ServerState with working_dir=None for edge-case testing."""
    mock_env = Mock()
    mock_env.get_fs = Mock(return_value=Mock())
    mock_agent = Mock()
    mock_agent.env = mock_env
    return ServerState(working_dir=None, agent=mock_agent)  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_diagnostic_working_dir_none_returns_directory_null() -> None:
    """When working_dir is None, directory is null (not crashing)."""
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from wolfharness_server.opencode_server.dependencies import get_state
    from wolfharness_server.opencode_server.routes.global_routes import router as global_router

    state = _make_state_with_none_working_dir()
    app = FastAPI()
    app.include_router(global_router)
    app.dependency_overrides[get_state] = lambda: state

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/global/diagnostic")
        assert response.status_code == 200
        data = response.json()
        assert data["directory"] is None


# =============================================================================
# Subscriber count reflects real subscribers
# =============================================================================


@pytest.mark.anyio
async def test_diagnostic_subscribers_reflects_real_count(
    async_client: AsyncClient,
    server_state: ServerState,
) -> None:
    """Subscribers count equals len(state.event_subscribers)."""
    response = await async_client.get("/global/diagnostic")
    data = response.json()
    assert data["subscribers"] == len(server_state.event_subscribers)
