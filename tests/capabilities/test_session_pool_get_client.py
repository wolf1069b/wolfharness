"""Unit tests for SessionConnectionPool.get_client().

Tests cover:
- Client construction
- MCP handshake (client is entered)
- Lazy init: no connection at construct
- Transport reuse: same config → same transport
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wolfharness.mcp_server.session_pool import SessionConnectionPool
from wolfharness_config.mcp_server import StdioMCPServerConfig


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Test: get_client returns a connected MCPClient
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_client_returns_connected_client() -> None:
    """get_client() returns a client that has been entered via __aenter__."""
    pool = SessionConnectionPool(session_id="test")

    config = StdioMCPServerConfig(command="echo", args=["hello"])

    # Patch MCPClient to avoid real subprocess
    mock_client_instance = MagicMock()
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=None)

    with (
        patch(
            "wolfharness.mcp_server.session_pool._create_transport",
            return_value=MagicMock(),
        ),
        patch(
            "wolfharness.mcp_server.client.MCPClient",
            return_value=mock_client_instance,
        ),
    ):
        client = await pool.get_client(config)

    assert client is mock_client_instance
    mock_client_instance.__aenter__.assert_called_once()
    await pool.cleanup()


# ---------------------------------------------------------------------------
# Test: transport reuse — same config shares transport
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_client_transport_reuse_same_config() -> None:
    """Two get_client() calls with same config share the same transport."""
    pool = SessionConnectionPool(session_id="test")

    config = StdioMCPServerConfig(command="echo", args=["hello"])

    transport = MagicMock()
    mock_client_1 = MagicMock()
    mock_client_1.__aenter__ = AsyncMock(return_value=mock_client_1)
    mock_client_1.__aexit__ = AsyncMock(return_value=None)
    mock_client_2 = MagicMock()
    mock_client_2.__aenter__ = AsyncMock(return_value=mock_client_2)
    mock_client_2.__aexit__ = AsyncMock(return_value=None)

    transport_creation_count = 0

    def fake_create_transport(cfg: Any) -> Any:
        nonlocal transport_creation_count
        transport_creation_count += 1
        return transport

    with (
        patch(
            "wolfharness.mcp_server.session_pool._create_transport",
            side_effect=fake_create_transport,
        ),
        patch(
            "wolfharness.mcp_server.client.MCPClient",
            side_effect=[mock_client_1, mock_client_2],
        ),
    ):
        client1 = await pool.get_client(config)
        client2 = await pool.get_client(config)

    # Same transport was reused (only created once)
    assert transport_creation_count == 1
    # Two different client instances wrapping the same transport
    assert client1 is mock_client_1
    assert client2 is mock_client_2
    await pool.cleanup()


# ---------------------------------------------------------------------------
# Test: different configs get separate transports
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_client_different_configs_separate_transports() -> None:
    """Two get_client() calls with different configs get separate transports."""
    pool = SessionConnectionPool(session_id="test")

    config1 = StdioMCPServerConfig(command="echo", args=["a"])
    config2 = StdioMCPServerConfig(command="echo", args=["b"])

    transport1 = MagicMock()
    transport2 = MagicMock()
    mock_client_1 = MagicMock()
    mock_client_1.__aenter__ = AsyncMock(return_value=mock_client_1)
    mock_client_1.__aexit__ = AsyncMock(return_value=None)
    mock_client_2 = MagicMock()
    mock_client_2.__aenter__ = AsyncMock(return_value=mock_client_2)
    mock_client_2.__aexit__ = AsyncMock(return_value=None)

    transports = [transport1, transport2]

    def fake_create_transport(cfg: Any) -> Any:
        return transports.pop(0)

    with (
        patch(
            "wolfharness.mcp_server.session_pool._create_transport",
            side_effect=fake_create_transport,
        ),
        patch(
            "wolfharness.mcp_server.client.MCPClient",
            side_effect=[mock_client_1, mock_client_2],
        ),
    ):
        client1 = await pool.get_client(config1)
        client2 = await pool.get_client(config2)

    assert client1 is mock_client_1
    assert client2 is mock_client_2
    # Two separate transports were created
    assert len(transports) == 0
    await pool.cleanup()


# ---------------------------------------------------------------------------
# Test: lazy init — no connection at construct
# ---------------------------------------------------------------------------


def test_session_pool_no_connection_at_construct() -> None:
    """SessionConnectionPool does not create connections at construction."""
    pool = SessionConnectionPool(session_id="test")
    assert len(pool._connections) == 0
