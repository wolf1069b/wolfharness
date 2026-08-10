"""Transport reuse tests — two McpServerCap instances with same config share transport.

Tests that SessionConnectionPool transport sharing works correctly when
multiple McpServerCap instances use the same config.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wolfharness.capabilities.mcp_server_cap import McpServerCap
from wolfharness.mcp_server.session_pool import SessionConnectionPool
from wolfharness_config.mcp_server import StdioMCPServerConfig


pytestmark = pytest.mark.unit


@pytest.mark.anyio
async def test_two_caps_same_config_share_transport() -> None:
    """Two McpServerCap instances with same config share one transport."""
    pool = SessionConnectionPool(session_id="test")
    config = StdioMCPServerConfig(command="echo", args=["hello"])

    transport = MagicMock()
    transport_creation_count = 0

    def fake_create_transport(cfg: Any) -> Any:
        nonlocal transport_creation_count
        transport_creation_count += 1
        return transport

    mock_client_1 = MagicMock()
    mock_client_1.__aenter__ = AsyncMock(return_value=mock_client_1)
    mock_client_1.__aexit__ = AsyncMock(return_value=None)
    mock_client_1._tool_change_callback = None
    mock_client_2 = MagicMock()
    mock_client_2.__aenter__ = AsyncMock(return_value=mock_client_2)
    mock_client_2.__aexit__ = AsyncMock(return_value=None)
    mock_client_2._tool_change_callback = None

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
        cap1 = McpServerCap(config=config, session_pool=pool)
        cap2 = McpServerCap(config=config, session_pool=pool)

        await cap1._ensure_client()
        await cap2._ensure_client()

    assert transport_creation_count == 1
    assert cap1._client is mock_client_1
    assert cap2._client is mock_client_2
    await pool.cleanup()


@pytest.mark.anyio
async def test_two_caps_different_configs_separate_transports() -> None:
    """Two McpServerCap instances with different configs get separate transports."""
    pool = SessionConnectionPool(session_id="test")
    config1 = StdioMCPServerConfig(command="echo", args=["a"])
    config2 = StdioMCPServerConfig(command="echo", args=["b"])

    transport1 = MagicMock()
    transport2 = MagicMock()
    transports = [transport1, transport2]
    transport_creation_count = 0

    def fake_create_transport(cfg: Any) -> Any:
        nonlocal transport_creation_count
        transport_creation_count += 1
        return transports.pop(0)

    mock_client_1 = MagicMock()
    mock_client_1.__aenter__ = AsyncMock(return_value=mock_client_1)
    mock_client_1.__aexit__ = AsyncMock(return_value=None)
    mock_client_1._tool_change_callback = None
    mock_client_2 = MagicMock()
    mock_client_2.__aenter__ = AsyncMock(return_value=mock_client_2)
    mock_client_2.__aexit__ = AsyncMock(return_value=None)
    mock_client_2._tool_change_callback = None

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
        cap1 = McpServerCap(config=config1, session_pool=pool)
        cap2 = McpServerCap(config=config2, session_pool=pool)

        await cap1._ensure_client()
        await cap2._ensure_client()

    assert transport_creation_count == 2
    assert cap1._client is mock_client_1
    assert cap2._client is mock_client_2
    await pool.cleanup()
