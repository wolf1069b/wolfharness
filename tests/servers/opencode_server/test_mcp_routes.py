"""Integration tests for OpenCode server MCP routes.

Tests MCP status endpoint response format and display_name field handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from wolfharness.common_types import MCPServerStatus


if TYPE_CHECKING:
    from unittest.mock import Mock

    from httpx import AsyncClient

    from wolfharness_server.opencode_server.state import ServerState


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_mcp_route_status_includes_display_name(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: Mock,
):
    """Test that MCP status endpoint response includes display_name field.

    Verifies the API response format contains the display_name field
    as specified in RFC-0019.
    """
    # Mock MCP server info with display_name
    mock_status = MCPServerStatus(
        name="test-server",
        status="connected",
        server_type="stdio",
        display_name="Test Server Display Name",
    )
    mock_agent.get_mcp_server_info = AsyncMock(return_value={"test-server": mock_status})

    # Add agent_router to test app for this test
    # Note: In actual tests, the router should be included in conftest.py app fixture
    response = await async_client.get("/mcp")

    # Verify response includes display_name field
    assert response.status_code == 200
    data = response.json()
    assert "test-server" in data
    server_data = data["test-server"]
    assert "displayName" in server_data


async def test_mcp_route_status_display_name_matches_configured_name(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: Mock,
):
    """Test that display_name matches the configured name when provided.

    When a display name is configured for an MCP server, it should be
    returned in the API response instead of the client_id.
    """
    configured_name = "My Custom MCP Server"
    mock_status = MCPServerStatus(
        name="custom-server-id",
        status="connected",
        server_type="sse",
        display_name=configured_name,
    )
    mock_agent.get_mcp_server_info = AsyncMock(return_value={"custom-server-id": mock_status})

    response = await async_client.get("/mcp")

    assert response.status_code == 200
    data = response.json()
    server_data = data["custom-server-id"]
    assert "displayName" in server_data
    assert server_data["displayName"] == configured_name


async def test_mcp_route_status_display_name_fallback(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: Mock,
):
    """Test that display_name falls back to client_id when name not provided.

    When no custom display name is configured, the display_name field
    should fall back to the client_id for consistent UI presentation.
    """
    client_id = "filesystem-mcp"
    mock_status = MCPServerStatus(
        name=client_id,
        status="connected",
        server_type="stdio",
        display_name=None,  # No custom name provided
    )
    mock_agent.get_mcp_server_info = AsyncMock(return_value={client_id: mock_status})

    response = await async_client.get("/mcp")

    assert response.status_code == 200
    data = response.json()
    server_data = data[client_id]
    assert "displayName" in server_data
    assert server_data["displayName"] == client_id


async def test_mcp_route_status_multiple_servers(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: Mock,
):
    """Test MCP status endpoint with multiple servers.

    Verifies that the endpoint correctly returns status for all
    configured MCP servers with their respective display names.
    """
    mock_statuses = {
        "server-1": MCPServerStatus(
            name="server-1",
            status="connected",
            server_type="stdio",
            display_name="File System Server",
        ),
        "server-2": MCPServerStatus(
            name="server-2",
            status="error",
            server_type="sse",
            display_name="Search Server",
            error="Connection refused",
        ),
        "server-3": MCPServerStatus(
            name="server-3",
            status="disconnected",
            server_type="http",
            display_name=None,  # No custom name
        ),
    }
    mock_agent.get_mcp_server_info = AsyncMock(return_value=mock_statuses)

    response = await async_client.get("/mcp")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3

    # Verify each server has display_name
    for server_id, server_data in data.items():
        assert "displayName" in server_data
        expected_name = mock_statuses[server_id].display_name or server_id
        assert server_data["displayName"] == expected_name
        assert "status" in server_data


async def test_mcp_route_status_empty_response(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: Mock,
):
    """Test MCP status endpoint when no servers are configured.

    Verifies that an empty dict is returned when no MCP servers are configured.
    """
    mock_agent.get_mcp_server_info = AsyncMock(return_value={})

    response = await async_client.get("/mcp")

    assert response.status_code == 200
    data = response.json()
    assert data == {}


async def test_mcp_route_status_includes_tools(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: Mock,
):
    """Test that MCP status includes tools list in response.

    Shape-only contract test: verifies the ``tools`` field exists and is a
    list. The mock ``MCPServerStatus`` does not populate ``tools``, so this
    test asserts ``tools == []``. Real tools verification (tools populated
    from a connected MCP server) lives in
    ``test_mcp_status_behavior.py::test_mcp_status_connected_server_via_mock_client``.
    """
    mock_status = MCPServerStatus(
        name="tools-server",
        status="connected",
        server_type="stdio",
        display_name="Tools Server",
    )
    mock_agent.get_mcp_server_info = AsyncMock(return_value={"tools-server": mock_status})

    response = await async_client.get("/mcp")

    assert response.status_code == 200
    data = response.json()
    server_data = data["tools-server"]
    assert "tools" in server_data
    assert isinstance(server_data["tools"], list)


async def test_mcp_route_status_includes_error_field(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: Mock,
):
    """Test that MCP status includes error field when server has error.

    Verifies the API response includes error information when
    an MCP server is in error state.
    """
    error_message = "Failed to connect: timeout"
    mock_status = MCPServerStatus(
        name="failing-server",
        status="error",
        server_type="sse",
        display_name="Failing Server",
        error=error_message,
    )
    mock_agent.get_mcp_server_info = AsyncMock(return_value={"failing-server": mock_status})

    response = await async_client.get("/mcp")

    assert response.status_code == 200
    data = response.json()
    server_data = data["failing-server"]
    assert "error" in server_data
    assert server_data["error"] == error_message
    assert server_data["status"] == "error"
