"""Tests for MCP server configuration display_name property.

Unit tests for the display_name property across all MCP server config types.
Tests cover custom names, fallback to client_id, and edge cases.
"""

from __future__ import annotations

from pydantic import HttpUrl
import pytest

from wolfharness_config.mcp_server import (
    AcpMCPServerConfig,
    SSEMCPServerConfig,
    StdioMCPServerConfig,
    StreamableHTTPMCPServerConfig,
)


pytestmark = pytest.mark.unit


# =============================================================================
# StdioMCPServerConfig Tests
# =============================================================================


def test_stdio_display_name_with_custom_name():
    """Test that display_name returns the custom name when set."""
    config = StdioMCPServerConfig(
        command="uv",
        args=["run", "server.py"],
        name="my_stdio_server",
    )

    assert config.display_name == "my_stdio_server"


def test_stdio_display_name_fallback_to_client_id():
    """Test that display_name falls back to client_id when name is None."""
    config = StdioMCPServerConfig(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem"],
        name=None,
    )

    assert config.display_name == config.client_id


def test_stdio_display_name_fallback_empty_string():
    """Test that display_name falls back to client_id when name is empty string."""
    config = StdioMCPServerConfig(
        command="python",
        args=["-m", "mcp_server"],
        name="",
    )

    assert config.display_name == config.client_id


def test_stdio_display_name_fallback_whitespace():
    """Test that display_name falls back to client_id when name is whitespace."""
    config = StdioMCPServerConfig(
        command="node",
        args=["server.js"],
        name="   ",
    )

    assert config.display_name == config.client_id


def test_stdio_display_name_strips_whitespace():
    """Test that display_name strips leading and trailing whitespace from name."""
    config = StdioMCPServerConfig(
        command="uvx",
        args=["mcp-server-fetch"],
        name="  fetch_server  ",
    )

    assert config.display_name == "fetch_server"


# =============================================================================
# SSEMCPServerConfig Tests
# =============================================================================


def test_sse_display_name_with_custom_name():
    """Test that display_name returns the custom name when set."""
    config = SSEMCPServerConfig(
        url=HttpUrl("http://localhost:8080/sse"),
        name="my_sse_server",
    )

    assert config.display_name == "my_sse_server"


def test_sse_display_name_fallback_to_client_id():
    """Test that display_name falls back to client_id when name is None."""
    config = SSEMCPServerConfig(
        url=HttpUrl("https://api.example.com/events"),
        name=None,
    )

    assert config.display_name == config.client_id


def test_sse_display_name_fallback_empty_string():
    """Test that display_name falls back to client_id when name is empty string."""
    config = SSEMCPServerConfig(
        url=HttpUrl("http://localhost:3000/sse"),
        name="",
    )

    assert config.display_name == config.client_id


def test_sse_display_name_fallback_whitespace():
    """Test that display_name falls back to client_id when name is whitespace."""
    config = SSEMCPServerConfig(
        url=HttpUrl("http://192.168.1.100:8080/sse"),
        name="   ",
    )

    assert config.display_name == config.client_id


def test_sse_display_name_strips_whitespace():
    """Test that display_name strips leading and trailing whitespace from name."""
    config = SSEMCPServerConfig(
        url=HttpUrl("http://localhost:9000/sse"),
        name="  sse_server  ",
    )

    assert config.display_name == "sse_server"


# =============================================================================
# StreamableHTTPMCPServerConfig Tests
# =============================================================================


def test_streamable_http_display_name_with_custom_name():
    """Test that display_name returns the custom name when set."""
    config = StreamableHTTPMCPServerConfig(
        url=HttpUrl("http://localhost:8080/mcp"),
        name="my_http_server",
    )

    assert config.display_name == "my_http_server"


def test_streamable_http_display_name_fallback_to_client_id():
    """Test that display_name falls back to client_id when name is None."""
    config = StreamableHTTPMCPServerConfig(
        url=HttpUrl("https://api.example.com/mcp"),
        name=None,
    )

    assert config.display_name == config.client_id


def test_streamable_http_display_name_fallback_empty_string():
    """Test that display_name falls back to client_id when name is empty string."""
    config = StreamableHTTPMCPServerConfig(
        url=HttpUrl("http://localhost:3000/mcp"),
        name="",
    )

    assert config.display_name == config.client_id


def test_streamable_http_display_name_fallback_whitespace():
    """Test that display_name falls back to client_id when name is whitespace."""
    config = StreamableHTTPMCPServerConfig(
        url=HttpUrl("http://192.168.1.100:8080/mcp"),
        name="   ",
    )

    assert config.display_name == config.client_id


def test_streamable_http_display_name_strips_whitespace():
    """Test that display_name strips leading and trailing whitespace from name."""
    config = StreamableHTTPMCPServerConfig(
        url=HttpUrl("http://localhost:9000/mcp"),
        name="  http_server  ",
    )

    assert config.display_name == "http_server"


# =============================================================================
# to_transport() tests
# =============================================================================


def test_to_transport_stdio():
    """StdioMCPServerConfig.to_transport() should return StdioTransport with correct fields."""
    from fastmcp.client.transports import StdioTransport

    config = StdioMCPServerConfig(
        command="python",
        args=["-m", "my_server"],
        env={"FOO": "bar"},
    )
    transport = config.to_transport()

    assert isinstance(transport, StdioTransport)
    assert transport.command == "python"
    assert transport.args == ["-m", "my_server"]
    assert transport.env is not None
    assert transport.env["FOO"] == "bar"


def test_to_transport_sse():
    """SSEMCPServerConfig.to_transport() should return SSETransport with correct URL/headers."""
    from fastmcp.client.transports import SSETransport

    config = SSEMCPServerConfig(
        url=HttpUrl("http://localhost:8080/sse"),
        headers={"Authorization": "Bearer token"},
    )
    transport = config.to_transport()

    assert isinstance(transport, SSETransport)
    assert transport.url == "http://localhost:8080/sse"
    assert transport.headers == {"Authorization": "Bearer token"}


def test_to_transport_http():
    """StreamableHTTPMCPServerConfig.to_transport() should return StreamableHttpTransport."""
    from fastmcp.client.transports import StreamableHttpTransport

    config = StreamableHTTPMCPServerConfig(
        url=HttpUrl("https://api.example.com/mcp"),
        headers={"X-Api-Key": "secret"},
    )
    transport = config.to_transport()

    assert isinstance(transport, StreamableHttpTransport)
    assert transport.url == "https://api.example.com/mcp"
    assert transport.headers == {"X-Api-Key": "secret"}


def test_to_transport_acp_raises():
    """AcpMCPServerConfig.to_transport() should raise NotImplementedError."""
    config = AcpMCPServerConfig(acp_id="my-acp-server")
    try:
        config.to_transport()
    except NotImplementedError:
        pass
    else:
        msg = "Expected NotImplementedError from AcpMCPServerConfig.to_transport()"
        raise AssertionError(msg)
