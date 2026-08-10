"""Red-flag regression tests for MCP-over-ACP message forwarding.

These tests guard against critical bugs in the MCP-over-ACP bridging layer
that can silently break tool discovery and all MCP operations.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock

import anyio
import pytest

from acp.schema.mcp import AcpMcpServer
from wolfharness import Agent
from wolfharness.delegation import AgentPool
from wolfharness.models.agents import NativeAgentConfig
from wolfharness.models.manifest import AgentsManifest
from wolfharness_server.acp_server.acp_agent import AgentPoolACPAgent
from wolfharness_server.acp_server.acp_mcp_manager import AcpMcpConnection


pytestmark = [pytest.mark.unit, pytest.mark.anyio]


@pytest.fixture
def mock_connection():
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
def acp_agent(mock_connection, default_test_agent: Agent) -> AgentPoolACPAgent:
    """Create a mock ACP agent for testing."""
    return AgentPoolACPAgent(client=mock_connection, default_agent=default_test_agent)


@pytest.fixture
def server_config() -> AcpMcpServer:
    """Create a test ACP MCP server configuration."""
    return AcpMcpServer(name="test-server", id="test-id")


async def test_no_double_wrap_on_mcp_message_forwarding(
    acp_agent: AgentPoolACPAgent,
    server_config: AcpMcpServer,
) -> None:
    """Regression test: mcp/message must use flattened ACP format.

    Bug: connect_acp_mcp_server() created a send_to_client callback that
    wrapped messages in {"connectionId": conn_id, "message": msg} (nested).
    But the ACP spec requires flattened format:
    {"connectionId": conn_id, "method": ..., "params": ...}.
    This caused MCP initialization to fail because the client couldn't
    parse the malformed nested request.

    Impact: When fastmcp ClientSession sends initialize internally, the
    message was nested. The client received a malformed request and
    silently failed to return tools.

    Fix: send_to_acp() now extracts method/params and sends flattened
    format per MCP-over-ACP RFD.

    This test simulates the real fastmcp flow via send_to_acp:
    1. ClientSession writes to pair.from_session_send
    2. Transport forwarder reads and calls connection.send_to_acp()
    3. send_to_acp() wraps as {"connectionId": id, "method": ..., "params": ...}
    4. The callback from connect_acp_mcp_server() passes through directly
    5. client.send_request("mcp/message", flattened) receives correct format
    """
    # Setup: mock client returns connectionId on mcp/connect
    send_request_mock = AsyncMock(return_value={"connectionId": "conn-redflag-1"})
    acp_agent.client.send_request = send_request_mock  # type: ignore[method-assign]

    # Step 1: Establish connection (this creates the callback)
    connection_id, _session_key = await acp_agent.connect_acp_mcp_server(
        server_config, "test-session-1"
    )
    assert connection_id == "conn-redflag-1"

    conn = acp_agent._mcp_manager.get_connection(connection_id)
    assert conn is not None

    # Step 2: Register a session pair to get per-session streams
    pair, _ = conn.register_session()
    raw_mcp_msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}

    # Step 3: Send via send_to_acp (what the transport forwarder calls internally)
    # Use a task group because the mock returns a value that send_to_acp tries to
    # route to the session stream (which has capacity 0, requiring a concurrent reader)
    async with anyio.create_task_group() as tg:
        tg.start_soon(conn.send_to_acp, raw_mcp_msg, pair.to_session_send)
        # Read the response to unblock send_to_acp
        await pair.to_session_receive.receive()

    # Step 4: Verify what the mock client received
    # Find the send_request call for "mcp/message"
    mcp_message_calls = [
        call for call in send_request_mock.call_args_list if call.args[0] == "mcp/message"
    ]
    assert len(mcp_message_calls) == 1, (
        f"Expected exactly one mcp/message call, got {len(mcp_message_calls)}"
    )

    _, params = mcp_message_calls[0].args

    # Params must be in flattened ACP format with connectionId, method, params
    assert "connectionId" in params, "params must contain connectionId"
    assert "method" in params, "params must contain method"
    assert params["connectionId"] == connection_id
    assert params["method"] == "tools/list"
    assert params.get("params") == {}

    # CRITICAL: There must NOT be a nested "message" key (old buggy format)
    assert "message" not in params, (
        f"Nested 'message' key detected — old buggy format! Got: {params}"
    )

    await conn.close()


async def test_send_to_client_forwards_response_to_session() -> None:
    """Regression test: send_to_acp forwards client response to session stream.

    When a request (message with `id`) is sent to the client and the client
    returns a response, that response must be forwarded back to the MCP
    session stream so ClientSession can process it.
    """
    mock_send = AsyncMock(return_value={"tools": []})
    server_cfg = AcpMcpServer(name="test", id="test-id")
    conn = AcpMcpConnection("conn-fwd", server_cfg, mock_send)
    pair, _ = conn.register_session()

    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}

    # Use task group to read from session stream concurrently
    # (unbuffered stream requires a receiver before send can complete)
    received_messages: list[Any] = []

    async with anyio.create_task_group() as tg:

        async def receiver() -> None:
            msg = await pair.to_session_receive.receive()
            received_messages.append(msg)

        tg.start_soon(receiver)
        await anyio.sleep(0)  # Let receiver start

        result = await conn.send_to_acp(request, pair.to_session_send)

    # _send_to_client was called with flattened ACP format
    mock_send.assert_awaited_once()
    call_args = mock_send.call_args[0][0]
    assert call_args["connectionId"] == "conn-fwd"
    assert call_args["method"] == "tools/list"

    # Result from client is returned
    assert result == {"tools": []}

    # Response was forwarded to session stream as SessionMessage
    from mcp.shared.message import SessionMessage

    assert len(received_messages) == 1
    assert isinstance(received_messages[0], SessionMessage)

    await conn.close()


async def test_send_to_client_error_forwards_error_to_session() -> None:
    """Regression test: send_to_acp errors forward error to session stream.

    When _send_to_client raises an exception, an error response must be
    forwarded to the session stream so ClientSession can handle it.
    """
    mock_send = AsyncMock(side_effect=RuntimeError("Connection lost"))
    server_cfg = AcpMcpServer(name="test", id="test-id")
    conn = AcpMcpConnection("conn-err", server_cfg, mock_send)
    pair, _ = conn.register_session()

    request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}

    # Use task group to read from session stream concurrently
    received_messages: list[Any] = []

    async with anyio.create_task_group() as tg:

        async def receiver() -> None:
            msg = await pair.to_session_receive.receive()
            received_messages.append(msg)

        tg.start_soon(receiver)
        await anyio.sleep(0)  # Let receiver start

        result = await conn.send_to_acp(request, pair.to_session_send)

    # Returns None on error
    assert result is None

    # Error response was forwarded to session stream as SessionMessage
    from mcp.shared.message import SessionMessage

    assert len(received_messages) == 1
    assert isinstance(received_messages[0], SessionMessage)

    await conn.close()


async def test_send_to_client_notification_no_session_forward() -> None:
    """Regression test: notifications (no id) don't forward to session stream.

    When a message has no `id` (notification), no response is forwarded
    to the session stream, since notifications are fire-and-forget.
    """
    mock_send = AsyncMock(return_value=None)
    server_cfg = AcpMcpServer(name="test", id="test-id")
    conn = AcpMcpConnection("conn-notif", server_cfg, mock_send)
    pair, _ = conn.register_session()

    notification = {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {}}
    await conn.send_to_acp(notification, pair.to_session_send)

    # Notification sent to client
    mock_send.assert_awaited_once()

    # No message forwarded to session stream (no id)
    with pytest.raises(anyio.WouldBlock):
        pair.to_session_receive.receive_nowait()

    await conn.close()


async def test_send_to_client_non_dict_returns_none() -> None:
    """Regression test: non-dict, non-SessionMessage returns None.

    If send_to_acp receives a message that is neither a dict nor a
    SessionMessage, it should return None without calling _send_to_client.
    """
    mock_send = AsyncMock()
    server_cfg = AcpMcpServer(name="test", id="test-id")
    conn = AcpMcpConnection("conn-nondict", server_cfg, mock_send)
    pair, _ = conn.register_session()

    result = await conn.send_to_acp("not a valid message", pair.to_session_send)

    assert result is None
    mock_send.assert_not_awaited()

    await conn.close()


async def test_send_to_client_closed_stream_handled_gracefully() -> None:
    """Regression test: BrokenResourceError when session stream is closed.

    If the session stream is already closed when trying to forward a
    response, the error should be handled gracefully without crashing.
    """
    mock_send = AsyncMock(return_value={"result": "ok"})
    server_cfg = AcpMcpServer(name="test", id="test-id")
    conn = AcpMcpConnection("conn-closed", server_cfg, mock_send)
    pair, _ = conn.register_session()

    # Close the session receive stream to simulate broken pipe
    await pair.to_session_receive.aclose()

    request = {"jsonrpc": "2.0", "id": 3, "method": "ping"}
    # Should not raise even though stream is closed
    result = await conn.send_to_acp(request, pair.to_session_send)
    assert result == {"result": "ok"}

    await conn.close()
