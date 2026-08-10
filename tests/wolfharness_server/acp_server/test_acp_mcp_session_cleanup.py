"""Tests for AcpMcpConnectionManager session connection tracking.

Verifies per-session MCP connection registration, deduplication,
cleanup, shared connection preservation, and integration with
``connect_acp_mcp_server()``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from acp.schema.mcp import AcpMcpServer
from wolfharness import Agent
from wolfharness.delegation import AgentPool
from wolfharness.models.agents import NativeAgentConfig
from wolfharness.models.manifest import AgentsManifest
from wolfharness_server.acp_server.acp_agent import AgentPoolACPAgent
from wolfharness_server.acp_server.acp_mcp_manager import (
    AcpMcpConnection,
    AcpMcpConnectionManager,
)


# --- Shared fixtures ---


@pytest.fixture
def server_config() -> AcpMcpServer:
    """Create a test ACP MCP server configuration."""
    return AcpMcpServer(name="test-server", id="test-id")


@pytest.fixture
def send_to_client() -> AsyncMock:
    """Create an AsyncMock send_to_client callable."""
    return AsyncMock(return_value=None)


@pytest.fixture
def mock_connection() -> Mock:
    """Create a mock ACP connection."""
    return Mock()


@pytest.fixture
def default_test_agent() -> Agent:
    """Create a simple test agent with a pool backed by manifest config."""

    def simple_callback(message: str) -> str:
        return f"Test response: {message}"

    manifest = AgentsManifest(agents={"test_agent": NativeAgentConfig(model="test")})
    pool = AgentPool(manifest)
    return Agent.from_callback(name="test_agent", callback=simple_callback, agent_pool=pool)


@pytest.fixture
def acp_agent(mock_connection: Mock, default_test_agent: Agent) -> AgentPoolACPAgent:
    """Create an AgentPoolACPAgent for integration tests."""
    return AgentPoolACPAgent(client=mock_connection, default_agent=default_test_agent)


# --- Unit tests (7) ---


@pytest.mark.unit
@pytest.mark.asyncio
async def test_register_session_connection_adds_to_set(
    server_config: AcpMcpServer,
    send_to_client: AsyncMock,
) -> None:
    """register_session_connection adds (connection_id, session_key) to the session's set."""
    manager = AcpMcpConnectionManager()
    conn = await manager.create_connection("conn-1", server_config, send_to_client)
    _pair, session_key = conn.register_session()

    manager.register_session_connection("session-1", "conn-1", session_key)

    assert "session-1" in manager._session_connections
    assert ("conn-1", session_key) in manager._session_connections["session-1"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_register_deduplicates_same_tuple(
    server_config: AcpMcpServer,
    send_to_client: AsyncMock,
) -> None:
    """Registering the same (session_id, connection_id, session_key) twice yields one entry."""
    manager = AcpMcpConnectionManager()
    conn = await manager.create_connection("conn-1", server_config, send_to_client)
    _pair, session_key = conn.register_session()

    manager.register_session_connection("session-1", "conn-1", session_key)
    manager.register_session_connection("session-1", "conn-1", session_key)

    assert len(manager._session_connections["session-1"]) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cleanup_session_unregisters_streams(
    server_config: AcpMcpServer,
    send_to_client: AsyncMock,
) -> None:
    """cleanup_session unregisters session stream pairs from their connections."""
    manager = AcpMcpConnectionManager()
    conn = await manager.create_connection("conn-1", server_config, send_to_client)
    _pair, session_key = conn.register_session()
    manager.register_session_connection("session-1", "conn-1", session_key)

    assert conn.has_active_sessions()

    await manager.cleanup_session("session-1")

    assert not conn.has_active_sessions()
    assert "session-1" not in manager._session_connections


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cleanup_preserves_shared_connection(
    server_config: AcpMcpServer,
    send_to_client: AsyncMock,
) -> None:
    """Cleaning up one session preserves a connection still used by another session."""
    manager = AcpMcpConnectionManager()
    conn = await manager.create_connection("conn-1", server_config, send_to_client)

    _pair_a, key_a = conn.register_session()
    _pair_b, key_b = conn.register_session()
    manager.register_session_connection("session-a", "conn-1", key_a)
    manager.register_session_connection("session-b", "conn-1", key_b)

    await manager.cleanup_session("session-a")

    # Connection should still have one active session
    assert conn.has_active_sessions()
    assert manager.get_connection("conn-1") is conn
    # session-b's entry should still be tracked
    assert "session-b" in manager._session_connections


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cleanup_removes_connection_with_no_sessions(
    server_config: AcpMcpServer,
    send_to_client: AsyncMock,
) -> None:
    """Cleaning up the last session on a connection removes the connection."""
    manager = AcpMcpConnectionManager()
    conn = await manager.create_connection("conn-1", server_config, send_to_client)
    _pair, session_key = conn.register_session()
    manager.register_session_connection("session-1", "conn-1", session_key)

    await manager.cleanup_session("session-1")

    assert not conn.has_active_sessions()
    assert manager.get_connection("conn-1") is None
    assert "conn-1" not in manager


@pytest.mark.unit
@pytest.mark.asyncio
async def test_has_active_sessions_true_when_streams_exist(
    server_config: AcpMcpServer,
    send_to_client: AsyncMock,
) -> None:
    """has_active_sessions returns True after register_session()."""
    conn = AcpMcpConnection("conn-1", server_config, send_to_client)

    assert not conn.has_active_sessions()

    conn.register_session()

    assert conn.has_active_sessions()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_has_active_sessions_false_when_empty(
    server_config: AcpMcpServer,
    send_to_client: AsyncMock,
) -> None:
    """has_active_sessions returns False on a fresh connection with no sessions."""
    conn = AcpMcpConnection("conn-1", server_config, send_to_client)

    assert not conn.has_active_sessions()


# --- Integration test (1) ---


@pytest.mark.unit
@pytest.mark.asyncio
async def test_connect_acp_mcp_server_registers_session(
    acp_agent: AgentPoolACPAgent,
    server_config: AcpMcpServer,
) -> None:
    """connect_acp_mcp_server calls register_session_connection with correct IDs.

    Verifies that after calling ``connect_acp_mcp_server(server, session_id)``,
    the manager's ``_session_connections`` dict contains an entry mapping
    the session_id to the returned (connection_id, session_key) tuple.
    """
    send_request_mock = AsyncMock(return_value={"connectionId": "conn-reg-1"})
    acp_agent.client.send_request = send_request_mock  # type: ignore[method-assign]

    connection_id, session_key = await acp_agent.connect_acp_mcp_server(
        server_config, "session-integration"
    )

    assert connection_id == "conn-reg-1"
    assert isinstance(session_key, int)

    # Verify register_session_connection was called with correct args
    manager = acp_agent._mcp_manager
    assert "session-integration" in manager._session_connections
    assert ("conn-reg-1", session_key) in manager._session_connections["session-integration"]
