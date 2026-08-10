"""Tests for AcpMcpConnectionManager and AcpMcpConnection."""

from __future__ import annotations

from unittest.mock import AsyncMock

import anyio
import pytest

from acp.schema.mcp import AcpMcpServer
from wolfharness_server.acp_server.acp_mcp_manager import (
    AcpMcpConnection,
    AcpMcpConnectionManager,
)


pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
def server_config() -> AcpMcpServer:
    """Create a test ACP MCP server configuration."""
    return AcpMcpServer(name="test-server", id="test-id")


@pytest.fixture
def send_to_client() -> AsyncMock:
    """Create an AsyncMock send_to_client callable."""
    return AsyncMock(return_value=None)


# AcpMcpConnectionManager tests


async def test_create_connection_basic(
    server_config: AcpMcpServer,
    send_to_client: AsyncMock,
) -> None:
    """Creating a connection stores it and opens streams."""
    manager = AcpMcpConnectionManager()

    conn = await manager.create_connection("conn-1", server_config, send_to_client)

    assert conn.connection_id == "conn-1"
    assert conn.server_config == server_config
    assert manager.get_connection("conn-1") is conn
    assert "conn-1" in manager
    assert len(manager) == 1


async def test_create_connection_duplicate_raises(
    server_config: AcpMcpServer,
    send_to_client: AsyncMock,
) -> None:
    """Creating a connection with duplicate ID raises ValueError."""
    manager = AcpMcpConnectionManager()
    await manager.create_connection("conn-1", server_config, send_to_client)

    with pytest.raises(ValueError, match="MCP connection 'conn-1' already exists"):
        await manager.create_connection("conn-1", server_config, send_to_client)


async def test_get_connection_existing(
    server_config: AcpMcpServer,
    send_to_client: AsyncMock,
) -> None:
    """get_connection returns the connection for an existing ID."""
    manager = AcpMcpConnectionManager()
    created = await manager.create_connection("conn-1", server_config, send_to_client)

    result = manager.get_connection("conn-1")

    assert result is created


async def test_get_connection_missing() -> None:
    """get_connection returns None for a missing ID."""
    manager = AcpMcpConnectionManager()

    result = manager.get_connection("nonexistent")

    assert result is None


async def test_remove_connection(
    server_config: AcpMcpServer,
    send_to_client: AsyncMock,
) -> None:
    """remove_connection removes the connection and closes it."""
    manager = AcpMcpConnectionManager()
    await manager.create_connection("conn-1", server_config, send_to_client)

    await manager.remove_connection("conn-1")

    assert manager.get_connection("conn-1") is None
    assert "conn-1" not in manager
    assert len(manager) == 0


async def test_remove_connection_missing_does_not_raise(
    server_config: AcpMcpServer,
    send_to_client: AsyncMock,
) -> None:
    """remove_connection is a no-op for a missing ID."""
    manager = AcpMcpConnectionManager()
    await manager.create_connection("conn-1", server_config, send_to_client)

    await manager.remove_connection("missing")

    assert len(manager) == 1


async def test_close_all(
    server_config: AcpMcpServer,
    send_to_client: AsyncMock,
) -> None:
    """close_all closes and removes all connections."""
    manager = AcpMcpConnectionManager()
    await manager.create_connection("conn-1", server_config, send_to_client)
    await manager.create_connection("conn-2", server_config, send_to_client)

    await manager.close_all()

    assert len(manager) == 0
    assert manager.get_connection("conn-1") is None
    assert manager.get_connection("conn-2") is None


# AcpMcpConnection tests


async def test_register_session_creates_streams(
    server_config: AcpMcpServer,
    send_to_client: AsyncMock,
) -> None:
    """register_session creates a per-session stream pair."""
    conn = AcpMcpConnection("conn-1", server_config, send_to_client)

    pair, _ = conn.register_session()

    assert pair.to_session_send is not None
    assert pair.to_session_receive is not None
    assert pair.from_session_send is not None
    assert pair.from_session_receive is not None


async def test_connection_close_closes_streams(
    server_config: AcpMcpServer,
    send_to_client: AsyncMock,
) -> None:
    """Close closes all stream pairs and marks connection closed."""
    conn = AcpMcpConnection("conn-1", server_config, send_to_client)
    conn.register_session()

    await conn.close()

    assert conn._closed is True


async def test_connection_close_is_idempotent(
    server_config: AcpMcpServer,
    send_to_client: AsyncMock,
) -> None:
    """Close can be called multiple times without error."""
    conn = AcpMcpConnection("conn-1", server_config, send_to_client)
    conn.register_session()
    await conn.close()

    await conn.close()

    assert conn._closed is True


async def test_connection_handle_client_message_routes_to_session(
    server_config: AcpMcpServer,
    send_to_client: AsyncMock,
) -> None:
    """handle_client_message broadcasts to registered session pairs."""
    conn = AcpMcpConnection("conn-1", server_config, send_to_client)
    pair, _ = conn.register_session()

    message = {"jsonrpc": "2.0", "method": "test", "id": 1}

    # Stream has capacity 0 (handoff semantics), so send and receive must be concurrent
    async with anyio.create_task_group() as tg:
        tg.start_soon(conn.handle_client_message, message)
        received = await pair.to_session_receive.receive()

    from mcp.shared.message import SessionMessage

    assert isinstance(received, SessionMessage)
    assert received.message.root.method == "test"  # type: ignore[union-attr]


async def test_send_to_acp_formats_request(
    server_config: AcpMcpServer,
    send_to_client: AsyncMock,
) -> None:
    """send_to_acp extracts method/params and sends flattened ACP format."""
    conn = AcpMcpConnection("conn-1", server_config, send_to_client)
    pair, _ = conn.register_session()
    message = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2024-11-05"},
    }

    await conn.send_to_acp(message, pair.to_session_send)

    send_to_client.assert_awaited_once_with({
        "connectionId": "conn-1",
        "method": "initialize",
        "params": {"protocolVersion": "2024-11-05"},
    })


async def test_send_to_acp_formats_notification(
    server_config: AcpMcpServer,
    send_to_client: AsyncMock,
) -> None:
    """send_to_acp sends notification without id in flattened ACP format."""
    conn = AcpMcpConnection("conn-1", server_config, send_to_client)
    pair, _ = conn.register_session()
    message = {"jsonrpc": "2.0", "method": "notifications/initialized"}

    await conn.send_to_acp(message, pair.to_session_send)

    send_to_client.assert_awaited_once_with({
        "connectionId": "conn-1",
        "method": "notifications/initialized",
    })


async def test_send_to_acp_reconstructs_success_response(
    server_config: AcpMcpServer,
) -> None:
    """send_to_acp reconstructs JSON-RPC response from inner result payload."""
    send_mock = AsyncMock(
        return_value={"protocolVersion": "2024-11-05", "serverInfo": {"name": "test"}}
    )
    conn = AcpMcpConnection("conn-1", server_config, send_mock)
    pair, _ = conn.register_session()
    message = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2024-11-05"},
    }

    async with anyio.create_task_group() as tg:
        tg.start_soon(conn.send_to_acp, message, pair.to_session_send)
        received = await pair.to_session_receive.receive()

    from mcp.shared.message import SessionMessage

    assert isinstance(received, SessionMessage)
    assert received.message.root.result == {
        "protocolVersion": "2024-11-05",
        "serverInfo": {"name": "test"},
    }  # type: ignore[union-attr]


async def test_send_to_acp_reconstructs_error_response(
    server_config: AcpMcpServer,
) -> None:
    """send_to_acp reconstructs JSON-RPC error from inner error payload."""
    send_mock = AsyncMock(return_value={"error": {"code": -32600, "message": "Invalid Request"}})
    conn = AcpMcpConnection("conn-1", server_config, send_mock)
    pair, _ = conn.register_session()
    message = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}

    async with anyio.create_task_group() as tg:
        tg.start_soon(conn.send_to_acp, message, pair.to_session_send)
        received = await pair.to_session_receive.receive()

    from mcp.shared.message import SessionMessage

    assert isinstance(received, SessionMessage)
    assert received.message.root.error.code == -32600  # type: ignore[union-attr]
    assert received.message.root.error.message == "Invalid Request"  # type: ignore[union-attr]


async def test_connection_handle_client_message_flattened_format(
    server_config: AcpMcpServer,
    send_to_client: AsyncMock,
) -> None:
    """handle_client_message reconstructs JSON-RPC from flattened ACP format."""
    conn = AcpMcpConnection("conn-1", server_config, send_to_client)
    pair, _ = conn.register_session()
    flattened = {"connectionId": "conn-1", "method": "tools/list", "params": {}}

    async with anyio.create_task_group() as tg:
        tg.start_soon(conn.handle_client_message, flattened)
        received = await pair.to_session_receive.receive()

    from mcp.shared.message import SessionMessage

    assert isinstance(received, SessionMessage)
    assert received.message.root.method == "tools/list"  # type: ignore[union-attr]


async def test_connection_handle_client_message_backward_compat(
    server_config: AcpMcpServer,
    send_to_client: AsyncMock,
) -> None:
    """handle_client_message still accepts raw JSON-RPC messages."""
    conn = AcpMcpConnection("conn-1", server_config, send_to_client)
    pair, _ = conn.register_session()
    message = {"jsonrpc": "2.0", "method": "test", "id": 1}

    async with anyio.create_task_group() as tg:
        tg.start_soon(conn.handle_client_message, message)
        received = await pair.to_session_receive.receive()

    from mcp.shared.message import SessionMessage

    assert isinstance(received, SessionMessage)
    assert received.message.root.method == "test"  # type: ignore[union-attr]
