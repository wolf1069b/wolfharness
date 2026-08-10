"""End-to-end integration tests for MCP-over-ACP message lifecycle.

Verifies the complete flow: mcp/connect -> mcp/message -> mcp/disconnect
through all layers of the ACP MCP integration.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from acp.schema.mcp import AcpMcpServer
from wolfharness_server.acp_server.acp_mcp_manager import (
    AcpMcpConnection,
    AcpMcpConnectionManager,
)
from wolfharness_server.acp_server.acp_mcp_transport import AcpMcpTransport


pytestmark = [pytest.mark.unit, pytest.mark.anyio]


@pytest.fixture
def server_config() -> AcpMcpServer:
    """Create a test ACP MCP server configuration."""
    return AcpMcpServer(name="test-server", id="test-id")


@pytest.fixture
def send_to_client() -> AsyncMock:
    """Create an AsyncMock send_to_client callable."""
    return AsyncMock(return_value=None)


# Test 1: Full Connection Lifecycle


async def test_full_connection_lifecycle(
    server_config: AcpMcpServer,
    send_to_client: AsyncMock,
) -> None:
    """Verify complete connection lifecycle from creation to removal."""
    # 1. Create an AcpMcpConnectionManager
    manager = AcpMcpConnectionManager()

    # 2. Create a connection with create_connection()
    conn = await manager.create_connection(
        connection_id="conn-1",
        server_config=server_config,
        send_to_client=send_to_client,
    )

    # 3. Verify connection is active via get_connection()
    assert manager.get_connection("conn-1") is conn
    assert "conn-1" in manager
    assert len(manager) == 1

    # 4. Register a session pair to verify per-session streams
    pair, _ = conn.register_session()
    assert pair.to_session_send is not None
    assert pair.to_session_receive is not None
    assert pair.from_session_send is not None
    assert pair.from_session_receive is not None

    # 5. Create an AcpMcpTransport with the connection
    transport = AcpMcpTransport(conn)

    # 6. Use connect_session() to establish a ClientSession
    with patch("mcp.client.session.ClientSession.initialize", new_callable=AsyncMock):
        async with transport.connect_session():
            # 7. Send a message via send_to_acp (mimics what the forwarder does)
            msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
            await conn.send_to_acp(msg, pair.to_session_send)

            # 8. Verify _send_to_client was called with the flattened message format
            conn._send_to_client.assert_awaited()
            call_args = conn._send_to_client.call_args[0][0]
            assert call_args["connectionId"] == "conn-1"
            assert "method" in call_args
            assert call_args["method"] == msg["method"]

        # 9. Close the session (context manager exit)
        # Session is closed when exiting the async with block

    # 10. Remove connection via manager.remove_connection()
    await manager.remove_connection("conn-1")

    # 11. Verify connection is gone
    assert manager.get_connection("conn-1") is None
    assert "conn-1" not in manager
    assert len(manager) == 0


# Test 2: Multiple Messages Over Same Connection


async def test_multiple_messages_over_same_connection(
    server_config: AcpMcpServer,
    send_to_client: AsyncMock,
) -> None:
    """Verify multiple MCP JSON-RPC messages are forwarded in order."""
    # 1. Create connection and register a session pair
    manager = AcpMcpConnectionManager()
    conn = await manager.create_connection(
        connection_id="conn-multi",
        server_config=server_config,
        send_to_client=send_to_client,
    )
    pair, _ = conn.register_session()

    # 2. Create transport and establish session
    transport = AcpMcpTransport(conn)

    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        },
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "read_file", "arguments": {"path": "/tmp/test"}},
        },
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/list",
            "params": {},
        },
    ]

    with patch("mcp.client.session.ClientSession.initialize", new_callable=AsyncMock):
        async with transport.connect_session():
            # 3. Send multiple MCP JSON-RPC messages via send_to_acp
            for msg in messages:
                await conn.send_to_acp(msg, pair.to_session_send)

            # 4. Verify each is forwarded to client in order
            assert conn._send_to_client.await_count == len(messages)

            # 5. Verify message format: {"connectionId": "...", "method": "...", "params": {...}}
            for i, expected in enumerate(messages):
                call_args = conn._send_to_client.call_args_list[i][0][0]
                assert call_args["connectionId"] == "conn-multi"
                assert "method" in call_args
                assert call_args["method"] == expected["method"]
                if "params" in expected:
                    assert call_args["params"] == expected["params"]

    # Cleanup
    await manager.remove_connection("conn-multi")


# Test 3: Connection Cleanup on Disconnect


async def test_connection_cleanup_on_disconnect(
    server_config: AcpMcpServer,
    send_to_client: AsyncMock,
) -> None:
    """Verify close_all closes and removes all connections."""
    # 1. Create multiple connections in the manager
    manager = AcpMcpConnectionManager()
    conn1 = await manager.create_connection(
        connection_id="conn-1",
        server_config=server_config,
        send_to_client=send_to_client,
    )
    conn2 = await manager.create_connection(
        connection_id="conn-2",
        server_config=server_config,
        send_to_client=send_to_client,
    )
    conn3 = await manager.create_connection(
        connection_id="conn-3",
        server_config=server_config,
        send_to_client=send_to_client,
    )

    assert len(manager) == 3
    assert manager.get_connection("conn-1") is conn1
    assert manager.get_connection("conn-2") is conn2
    assert manager.get_connection("conn-3") is conn3

    # 2. Call close_all()
    await manager.close_all()

    # 3. Verify all connections are closed and removed
    assert len(manager) == 0
    assert manager.get_connection("conn-1") is None
    assert manager.get_connection("conn-2") is None
    assert manager.get_connection("conn-3") is None
    assert conn1._closed is True
    assert conn2._closed is True
    assert conn3._closed is True


# Test 4: Error Handling - Closed Connection


async def test_error_handling_closed_connection(
    server_config: AcpMcpServer,
    send_to_client: AsyncMock,
) -> None:
    """Verify handle_client_message handles closed streams gracefully."""
    # 1. Create connection, register a session, then close it
    conn = AcpMcpConnection(
        connection_id="conn-closed",
        server_config=server_config,
        send_to_client=send_to_client,
    )
    conn.register_session()
    await conn.close()

    assert conn._closed is True

    # 2. Verify handle_client_message handles closed streams gracefully
    # (does not raise, logs debug message instead)
    await conn.handle_client_message({"jsonrpc": "2.0", "method": "test"})
