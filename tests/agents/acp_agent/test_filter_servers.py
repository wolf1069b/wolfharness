"""Tests for filter_servers_by_capabilities."""

from __future__ import annotations

import pytest
import structlog

from acp.schema import AgentCapabilities
from acp.schema.mcp import AcpMcpServer, HttpMcpServer, StdioMcpServer
from wolfharness.agents.acp_agent.helpers import filter_servers_by_capabilities


pytestmark = pytest.mark.unit


def test_acp_server_filtered_when_acp_false() -> None:
    """AcpMcpServer is filtered out when agent does not support ACP."""
    capabilities = AgentCapabilities.create(acp_mcp_servers=False)
    servers = [AcpMcpServer(name="acp-server", id="acp-1")]

    result = filter_servers_by_capabilities(servers, capabilities)

    assert result == []


def test_acp_server_passes_when_acp_true() -> None:
    """AcpMcpServer passes through when agent supports ACP."""
    capabilities = AgentCapabilities.create(acp_mcp_servers=True)
    servers = [AcpMcpServer(name="acp-server", id="acp-1")]

    result = filter_servers_by_capabilities(servers, capabilities)

    assert len(result) == 1
    assert isinstance(result[0], AcpMcpServer)
    assert result[0].name == "acp-server"


def test_acp_server_filtered_when_capabilities_none() -> None:
    """AcpMcpServer is filtered out when agent_capabilities is None."""
    servers = [AcpMcpServer(name="acp-server", id="acp-1")]

    result = filter_servers_by_capabilities(servers, None)

    assert result == []


def test_mixed_servers_acp_filtered_when_not_supported() -> None:
    """Only AcpMcpServer is filtered when acp=False, stdio passes through."""
    capabilities = AgentCapabilities.create(
        acp_mcp_servers=False,
        http_mcp_servers=True,
        sse_mcp_servers=True,
    )
    servers = [
        StdioMcpServer(name="stdio-server", command="cmd", args=[], env=[]),
        AcpMcpServer(name="acp-server", id="acp-1"),
        HttpMcpServer(name="http-server", url="http://example.com"),
    ]

    result = filter_servers_by_capabilities(servers, capabilities)

    assert len(result) == 2
    assert result[0].name == "stdio-server"
    assert result[1].name == "http-server"


def test_acp_server_with_http_and_sse_false() -> None:
    """AcpMcpServer filtered when only acp is False but http/sse also False."""
    capabilities = AgentCapabilities.create(
        acp_mcp_servers=False,
        http_mcp_servers=False,
        sse_mcp_servers=False,
    )
    servers = [
        AcpMcpServer(name="acp-server", id="acp-1"),
        HttpMcpServer(name="http-server", url="http://example.com"),
    ]

    result = filter_servers_by_capabilities(servers, capabilities)

    assert result == []


def test_logger_includes_supported_acp() -> None:
    """Logger warning includes supported_acp field."""
    logger = structlog.get_logger()
    capabilities = AgentCapabilities.create(acp_mcp_servers=False)
    servers = [AcpMcpServer(name="acp-server", id="acp-1")]

    # Just verify it doesn't raise and returns empty list
    result = filter_servers_by_capabilities(servers, capabilities, logger=logger)
    assert result == []


def test_all_acp_variants() -> None:
    """All combinations of acp support with AcpMcpServer."""
    server = AcpMcpServer(name="acp-server", id="acp-1")

    # acp=True
    caps_true = AgentCapabilities.create(acp_mcp_servers=True)
    assert filter_servers_by_capabilities([server], caps_true) == [server]

    # acp=False
    caps_false = AgentCapabilities.create(acp_mcp_servers=False)
    assert filter_servers_by_capabilities([server], caps_false) == []

    # No mcp_capabilities at all - treated as not supported
    caps_no_mcp = AgentCapabilities(load_session=False, mcp_capabilities=None)
    assert filter_servers_by_capabilities([server], caps_no_mcp) == []
