"""L2 integration tests for MCP status reporting through the real agent path.

Exercises the full route → agent → manager → cap → client path with a real
``AgentPool`` and a real ``BaseAgent._get_mcp_server_info`` method. No mocking
of ``agent.get_mcp_server_info`` — the method under test is exercised for real.

A mock ``MCPClient`` is injected into ``pool.mcp.providers`` so no real
subprocess is spawned. The ``MCPManager.setup_server`` dedup guard skips the
real connection because the matching ``client_id`` is already in ``providers``.

See ``openspec/changes/fix-mcp-status-reporting/design.md`` Decision 5 for the
test strategy rationale.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from mcp.types import Tool
import pytest
import yamling

from wolfharness import AgentPool, AgentsManifest
from wolfharness.capabilities.mcp_server_cap import McpServerCap
from wolfharness_server.opencode_server.dependencies import get_state
from wolfharness_server.opencode_server.routes import agent_router
from wolfharness_server.opencode_server.state import ServerState


if TYPE_CHECKING:
    from wolfharness_config.mcp_server import MCPServerConfig


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "mcp_status_config.yaml"


def _load_manifest() -> AgentsManifest:
    """Load the MCP status test config from the fixtures directory."""
    raw = yamling.load_yaml(FIXTURE_PATH.read_text(), verify_type=dict)
    return AgentsManifest.model_validate(raw)


def _make_mock_client(
    server_name: str,
    server_version: str,
    tool_name: str,
) -> Mock:
    """Create a mock ``MCPClient`` with configurable server info and tools.

    The mock satisfies the interface used by ``McpServerCap`` and
    ``MCPManager.get_server_status``: ``server_info`` property returns a dict,
    ``list_tools`` returns MCP ``Tool`` objects with ``name``, ``description``,
    and ``inputSchema`` attributes.
    """
    client = Mock()
    client.server_info = {"name": server_name, "version": server_version}
    client.list_tools = AsyncMock(
        return_value=[
            Tool(
                name=tool_name,
                description=f"{tool_name} tool",
                inputSchema={"type": "object", "properties": {}},
            )
        ]
    )
    return client


def _enable_config(config: MCPServerConfig) -> MCPServerConfig:
    """Return a copy of ``config`` with ``enabled=True``.

    The YAML fixture uses ``enabled: false`` so ``MCPManager.__aenter__`` skips
    the server (no subprocess). The test flips it to ``enabled=True`` so
    ``get_server_status`` reports ``connected`` instead of ``disabled``, then
    relies on the dedup guard to prevent the real connection.
    """
    return config.model_copy(update={"enabled": True})


def _inject_mock_cap(
    manager: object,
    config: MCPServerConfig,
    client: Mock,
    name: str,
) -> McpServerCap:
    """Append a mock ``McpServerCap`` to ``manager.providers``.

    The cap uses the same ``client_id`` as ``config`` so the dedup guard in
    ``setup_server`` skips the real connection on pool entry.
    """
    cap = McpServerCap(config=config, client=client, name=name)
    manager.providers.append(cap)  # type: ignore[attr-defined]
    return cap


async def _make_server_state(
    pool: AgentPool[None],
    agent_name: str,
    working_dir: str,
) -> tuple[ServerState, FastAPI]:
    """Create a real agent via ``SessionPool``, wire it into ``ServerState``.

    Returns the state and a FastAPI app with only ``agent_router`` mounted —
    enough for ``GET /mcp`` without pulling in session/message/file routers.
    """
    sp = pool.session_pool
    assert sp is not None
    await sp.create_session("mcp_status_session", agent_name=agent_name)
    agent = await sp.sessions.get_or_create_session_agent("mcp_status_session")
    state = ServerState(working_dir=working_dir, agent=agent)
    app = FastAPI()
    app.include_router(agent_router)
    app.dependency_overrides[get_state] = lambda: state
    return state, app


async def test_mcp_status_connected_server_via_mock_client(
    tmp_path: Path,
) -> None:
    """``GET /mcp`` returns ``connected`` with tools when a mock cap is injected.

    Given: a pool-level MCP server with ``enabled: false`` in YAML.
    When: the test flips ``enabled`` to True, injects a mock ``McpServerCap``
        into ``pool.mcp.providers`` before entering, then enters the pool —
        ``setup_server`` hits the dedup guard and skips the real connection.
    Then: ``GET /mcp`` returns ``status="connected"``, ``tools=["search_kb"]``,
        and the internal ``MCPServerStatus.server_name`` is ``"fake"``.

    This exercises the real ``BaseAgent._get_mcp_server_info`` →
    ``MCPManager.get_server_status`` → ``McpServerCap.list_tools`` →
    ``MCPClient.server_info`` path without mocking the method under test.
    """
    manifest = _load_manifest()
    pool = AgentPool(manifest)

    fake_cfg = pool.mcp.servers[0]
    fake_enabled = _enable_config(fake_cfg)
    pool.mcp.servers[0] = fake_enabled

    fake_client = _make_mock_client(
        server_name="fake",
        server_version="1.0",
        tool_name="search_kb",
    )
    _inject_mock_cap(pool.mcp, fake_enabled, fake_client, name="fake_kb")

    async with pool:
        state, app = await _make_server_state(pool, "test_agent", str(tmp_path))

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.get("/mcp")

        assert response.status_code == 200
        data = response.json()
        assert "fake_kb" in data
        server = data["fake_kb"]
        assert server["status"] == "connected"
        assert server["tools"] == ["search_kb"]
        assert server["displayName"] == "fake_kb"

        # Verify internal MCPServerStatus carries server_name/server_version
        # (not exposed via the OpenCode MCPStatus API but part of the spec).
        internal = await state.agent.get_mcp_server_info()
        status = internal["fake_kb"]
        assert status.server_name == "fake"
        assert status.server_version == "1.0"


async def test_mcp_status_agent_scoped_wins_on_key_collision(tmp_path: Path) -> None:
    """Agent-scoped MCP server takes precedence on ``display_name`` collision.

    Given: pool-level server ``kb`` and agent-scoped server ``kb``
        share the same ``display_name`` (``kb``) and ``client_id`` (``echo_kb``).
    When: ``GET /mcp`` is called on the agent with its own ``MCPManager``.
    Then: ``BaseAgent._get_mcp_server_info`` merges pool-level and agent-scoped
        statuses, with agent-scoped keys overwriting pool-level on collision.
        The response reports the agent-scoped server's tools.
    """
    manifest = _load_manifest()
    pool = AgentPool(manifest)

    # Pool-level "kb" server (client_id = "echo_kb").
    pool_cfg = pool.mcp.servers[1]
    assert pool_cfg.client_id == "echo_kb"
    pool_enabled = _enable_config(pool_cfg)
    pool.mcp.servers[1] = pool_enabled

    pool_client = _make_mock_client(
        server_name="pool_server",
        server_version="1.0",
        tool_name="pool_tool",
    )
    _inject_mock_cap(pool.mcp, pool_enabled, pool_client, name="kb")

    async with pool:
        state, app = await _make_server_state(pool, "collision_agent", str(tmp_path))

        # The collision_agent has its own MCPManager (not shared with pool)
        # because its config declares agent-level ``mcp_servers``.
        agent = state.agent
        assert agent.mcp is not pool.mcp

        # Agent-scoped "kb" server (same client_id "echo_kb", same display_name "kb").
        agent_cfg = agent.mcp.servers[0]
        assert agent_cfg.client_id == "echo_kb"
        agent_enabled = _enable_config(agent_cfg)
        agent.mcp.servers[0] = agent_enabled

        agent_client = _make_mock_client(
            server_name="agent_server",
            server_version="2.0",
            tool_name="agent_tool",
        )
        _inject_mock_cap(agent.mcp, agent_enabled, agent_client, name="kb")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.get("/mcp")

        assert response.status_code == 200
        data = response.json()
        # The collision key "kb" must be present with the agent-scoped
        # server's values (agent-scoped overwrites pool-level on collision).
        assert "kb" in data
        server = data["kb"]
        # Agent-scoped wins: its tools are surfaced.
        assert server["tools"] == ["agent_tool"]
        assert server["status"] == "connected"

        # Verify the internal status came from the agent-scoped manager.
        internal = await agent.get_mcp_server_info()
        status = internal["kb"]
        assert status.server_name == "agent_server"
        assert status.server_version == "2.0"
