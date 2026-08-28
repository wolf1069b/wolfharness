"""Regression tests for GET /experimental/resource (@resource picker).

Covers the fix where ``McpResource.name`` must be the human-readable basename
(not the full resource URI): the opencode TUI picker displays and fuzzy-matches
on ``name``, so a full ``viking://...`` URI made @-mention quick-match useless
and rendered very long rows (scroll flicker).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from wolfharness.capabilities.resource_protocols import ResourceAccess, ResourceEntry
from wolfharness_server.opencode_server.dependencies import get_state


if TYPE_CHECKING:
    from wolfharness_server.opencode_server.state import ServerState


pytestmark = [pytest.mark.asyncio]


class _FakeVikingCap(ResourceAccess):
    """ResourceAccess whose list_resources returns fixed viking entries."""

    def __init__(self, entries: list[ResourceEntry]) -> None:
        self._entries = entries

    @property
    def server_name(self) -> str:
        return "viking"

    async def supports_resources(self) -> bool:
        return True

    async def list_resources(self) -> list[ResourceEntry]:
        return self._entries


def _make_app(state: ServerState) -> FastAPI:
    from wolfharness_server.opencode_server.routes import agent_router

    app = FastAPI()
    app.include_router(agent_router)
    app.dependency_overrides[get_state] = lambda: state
    return app


async def _fetch_resources(state: ServerState) -> dict[str, object]:
    app = _make_app(state)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/experimental/resource")
        assert resp.status_code == 200
        return resp.json()


def _state_with_cap(server_state: ServerState, cap: ResourceAccess) -> ServerState:
    """Point the mock agent's registry at a registry returning ``cap``."""
    registry = Mock()
    registry.get_mcp_resource_providers = Mock(return_value=[cap])
    server_state.agent.host_context.extension_registry = registry
    return server_state


async def test_resource_name_is_basename_not_uri(
    server_state: ServerState,
) -> None:
    """The @resource picker must match against the basename, not the full URI."""
    cap = _FakeVikingCap(
        [
            ResourceEntry(
                uri="viking://resources/810test/OP/OpA/xxx/opa-gap-E460-fault.md",
                name="opa-gap-E460-fault.md",
                description="/OP/OpA/xxx",
                mime_type="text/markdown",
            )
        ]
    )
    state = _state_with_cap(server_state, cap)

    data = await _fetch_resources(state)

    assert len(data) == 1
    resource = next(iter(data.values()))
    assert resource["name"] == "opa-gap-E460-fault.md"
    assert resource["uri"] == "viking://resources/810test/OP/OpA/xxx/opa-gap-E460-fault.md"


async def test_resource_name_falls_back_to_uri_when_empty(
    server_state: ServerState,
) -> None:
    """Empty basenames fall back to the URI so entries never show blank."""
    cap = _FakeVikingCap(
        [
            ResourceEntry(
                uri="viking://resources/810test/README.md",
                name="",
                description="",
                mime_type="text/markdown",
            )
        ]
    )
    state = _state_with_cap(server_state, cap)

    data = await _fetch_resources(state)

    assert len(data) == 1
    resource = next(iter(data.values()))
    assert resource["name"] == "viking://resources/810test/README.md"


async def test_duplicate_basenames_get_distinct_keys(
    server_state: ServerState,
) -> None:
    """Same basename in different directories must not silently drop entries."""
    cap = _FakeVikingCap(
        [
            ResourceEntry(
                uri="viking://resources/810test/chapters/a/chapter.md",
                name="chapter.md",
                description="/chapters/a",
                mime_type="text/markdown",
            ),
            ResourceEntry(
                uri="viking://resources/810test/chapters/b/chapter.md",
                name="chapter.md",
                description="/chapters/b",
                mime_type="text/markdown",
            ),
        ]
    )
    state = _state_with_cap(server_state, cap)

    data = await _fetch_resources(state)

    assert len(data) == 2
    by_uri = {r["uri"]: r for r in data.values()}
    assert set(by_uri) == {
        "viking://resources/810test/chapters/a/chapter.md",
        "viking://resources/810test/chapters/b/chapter.md",
    }


async def test_erroring_capability_is_skipped(server_state: ServerState) -> None:
    """A failing list_resources must not break the whole picker."""
    breaking = Mock()
    breaking.list_resources = AsyncMock(side_effect=RuntimeError("viking down"))
    breaking.supports_resources = AsyncMock(return_value=True)
    breaking.server_name = "breaking"
    good = _FakeVikingCap(
        [
            ResourceEntry(
                uri="viking://resources/810test/ok.md",
                name="ok.md",
                description="/",
                mime_type="text/markdown",
            )
        ]
    )
    registry = Mock()
    registry.get_mcp_resource_providers = Mock(return_value=[breaking, good])
    server_state.agent.host_context.extension_registry = registry

    data = await _fetch_resources(server_state)

    assert len(data) == 1
    assert data[next(iter(data))]["uri"] == "viking://resources/810test/ok.md"
