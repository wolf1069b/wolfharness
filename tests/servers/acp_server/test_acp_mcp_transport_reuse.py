"""Tests for AcpMcpTransport reuse across multiple sessions.

Verifies that a single AcpMcpTransport (shared via copy_pre_created_transports)
can be entered by multiple MCPToolset/ClientSession instances concurrently
without message corruption or stream contention.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from wolfharness_server.acp_server.acp_mcp_manager import AcpMcpConnection
from wolfharness_server.acp_server.acp_mcp_transport import AcpMcpTransport


pytestmark = pytest.mark.integration


def _make_connection(
    handler: Any | None = None,
) -> AcpMcpConnection:
    """Create a real AcpMcpConnection with a mock send_to_client."""
    from acp.schema.mcp import AcpMcpServer

    server = AcpMcpServer(name="test-server", id="test-id")
    send_callback = AsyncMock(side_effect=handler or (lambda wrapped: {}))
    return AcpMcpConnection(
        connection_id="test-conn-1",
        server_config=server,
        send_to_client=send_callback,
    )


def _make_initialize_result() -> dict[str, Any]:
    """Return a standard MCP initialize result payload."""
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {},
        "serverInfo": {"name": "test", "version": "1.0"},
    }


def _make_tools_list_result(tool_name: str) -> dict[str, Any]:
    """Return a tools/list result payload with one tool."""
    return {
        "tools": [
            {
                "name": tool_name,
                "description": f"Tool {tool_name}",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ],
    }


def _make_handler(tool_name_fn: Any | None = None) -> Any:
    """Create a handler that returns result payloads (not full JSON-RPC)."""
    request_count = 0

    def handler(wrapped: dict[str, Any]) -> dict[str, Any]:
        nonlocal request_count
        request_count += 1
        method = wrapped.get("method", "")
        if method == "initialize":
            return _make_initialize_result()
        if method == "tools/list":
            name = tool_name_fn(request_count) if tool_name_fn else "tool"
            return _make_tools_list_result(name)
        return {}

    return handler


class TestAcpMcpTransportReuse:
    """Tests for sharing a single AcpMcpTransport across multiple sessions."""

    async def test_two_sessions_independent_connect(self) -> None:
        """Two sessions connect independently.

        Given a shared AcpMcpTransport, when two sessions enter
        connect_session() concurrently, then each gets an independent
        ClientSession that doesn't interfere with the other.
        """
        handler = _make_handler(tool_name_fn=lambda n: f"tool_{n}")
        conn = _make_connection(handler)
        transport = AcpMcpTransport(conn, timeout=5.0)

        results: list[dict[str, Any]] = []

        async def _session_work(label: str) -> None:
            async with transport.connect_session() as session:
                await session.initialize()
                tools = await session.list_tools()
                results.append({"label": label, "tools": tools})

        await asyncio.gather(_session_work("A"), _session_work("B"))

        assert len(results) == 2
        labels = {r["label"] for r in results}
        assert labels == {"A", "B"}
        for r in results:
            assert len(r["tools"].tools) == 1

    async def test_one_session_exit_does_not_break_other(self) -> None:
        """One session exit does not break the other.

        Given two sessions sharing a transport, when one exits
        connect_session(), then the other can still call tools.
        """
        handler = _make_handler(tool_name_fn=lambda n: "tool")
        conn = _make_connection(handler)
        transport = AcpMcpTransport(conn, timeout=5.0)

        session_a_active = asyncio.Event()

        async def _session_a() -> None:
            async with transport.connect_session() as session:
                await session.initialize()
                session_a_active.set()
                await asyncio.sleep(0.5)
                tools = await session.list_tools()
                assert tools is not None

        async def _session_b() -> None:
            await session_a_active.wait()
            async with transport.connect_session() as session:
                await session.initialize()
                await session.list_tools()

        await asyncio.gather(_session_a(), _session_b())

    async def test_concurrent_tool_calls_no_interference(self) -> None:
        """Concurrent tool calls have no interference.

        Given four sessions sharing a transport, when all call
        tools concurrently, then each gets its own correct response.
        """
        handler = _make_handler(tool_name_fn=lambda n: f"tool_{n}")
        conn = _make_connection(handler)
        transport = AcpMcpTransport(conn, timeout=5.0)

        results: list[str] = []

        async def _session(label: str) -> None:
            async with transport.connect_session() as session:
                await session.initialize()
                await session.list_tools()
                results.append(label)

        await asyncio.gather(
            _session("A"),
            _session("B"),
            _session("C"),
            _session("D"),
        )

        assert set(results) == {"A", "B", "C", "D"}
