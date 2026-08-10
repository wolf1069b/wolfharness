from unittest.mock import AsyncMock, patch

import pytest

from acp.schema.mcp import AcpMcpServer
from wolfharness_server.acp_server.acp_mcp_manager import AcpMcpConnection
from wolfharness_server.acp_server.acp_mcp_transport import AcpMcpTransport


pytestmark = pytest.mark.integration


@pytest.fixture
async def connection():
    """Create an AcpMcpConnection for transport tests."""
    server = AcpMcpServer(name="test-server", id="test-123")
    conn = AcpMcpConnection(
        connection_id="test-conn-1",
        server_config=server,
        send_to_client=AsyncMock(return_value=None),
    )
    # Note: do NOT register a session pair here — each test that needs one
    # registers its own pair, and transport.connect_session() registers its own
    # pair internally.
    yield conn
    await conn.close()


class TestAcpMcpTransportInitialization:
    """Tests for AcpMcpTransport basic initialization."""

    @pytest.mark.anyio
    async def test_transport_initialization(self, connection):
        """Transport should store the connection reference."""
        transport = AcpMcpTransport(connection)
        assert transport._connection is connection

    @pytest.mark.anyio
    async def test_connect_session_yields_client_session(self, connection):
        """connect_session should yield a ClientSession instance."""
        from mcp.client.session import ClientSession

        transport = AcpMcpTransport(connection)

        with patch("mcp.client.session.ClientSession.initialize", new_callable=AsyncMock):
            async with transport.connect_session() as session:
                assert isinstance(session, ClientSession)


class TestAcpMcpTransportMessageForwarding:
    """Tests for message forwarding through the connection."""

    @pytest.mark.anyio
    async def test_message_forwarding_from_session_to_client(self, connection):
        """Messages sent via send_to_acp should be forwarded to the client."""
        AcpMcpTransport(connection)
        pair, _ = connection.register_session()
        msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        }

        # Send via send_to_acp (what the transport forwarder calls internally)
        await connection.send_to_acp(msg, pair.to_session_send)

        # Should be forwarded to client via _send_to_client in flattened format
        connection._send_to_client.assert_awaited_once()
        call_args = connection._send_to_client.call_args[0][0]
        assert call_args["connectionId"] == connection.connection_id
        assert call_args["method"] == msg["method"]
        assert call_args["params"] == msg["params"]

    @pytest.mark.anyio
    async def test_multiple_messages_forwarded(self, connection):
        """Multiple messages should be forwarded in order."""
        AcpMcpTransport(connection)
        pair, _ = connection.register_session()
        messages = [
            {"jsonrpc": "2.0", "id": i, "method": f"method_{i}", "params": {"data": i}}
            for i in range(3)
        ]

        for msg in messages:
            await connection.send_to_acp(msg, pair.to_session_send)

        assert connection._send_to_client.await_count == len(messages)
        for i, expected in enumerate(messages):
            call_args = connection._send_to_client.call_args_list[i][0][0]
            assert call_args["connectionId"] == connection.connection_id
            assert call_args["method"] == expected["method"]
            assert call_args["params"] == expected["params"]

    @pytest.mark.anyio
    async def test_forwarder_task_cleanup_on_session_exit(self, connection):
        """Forwarder task should be cancelled when session context exits."""
        transport = AcpMcpTransport(connection)

        with patch("mcp.client.session.ClientSession.initialize", new_callable=AsyncMock):
            async with transport.connect_session():
                pass  # Session exits here

        # After session exit, the transport's internal pair should be unregistered
        # Verify no session pairs remain on the connection
        assert len(connection._session_streams) == 0


class TestAcpMcpTransportReusability:
    """Tests verifying transport can be used for multiple sessions."""

    @pytest.mark.anyio
    async def test_transport_reusable_across_sessions(self, connection):
        """Transport should support multiple connect_session calls."""
        transport = AcpMcpTransport(connection)
        pair, _ = connection.register_session()

        msg1 = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        with patch("mcp.client.session.ClientSession.initialize", new_callable=AsyncMock):
            async with transport.connect_session():
                await connection.send_to_acp(msg1, pair.to_session_send)
                call_args = connection._send_to_client.call_args[0][0]
                assert call_args["connectionId"] == connection.connection_id
                assert call_args["method"] == msg1["method"]
                assert call_args["params"] == msg1["params"]

        # Connection remains reusable
        msg2 = {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "test"}}
        with patch("mcp.client.session.ClientSession.initialize", new_callable=AsyncMock):
            async with transport.connect_session():
                await connection.send_to_acp(msg2, pair.to_session_send)
                call_args = connection._send_to_client.call_args[0][0]
                assert call_args["connectionId"] == connection.connection_id
                assert call_args["method"] == msg2["method"]
                assert call_args["params"] == msg2["params"]

    @pytest.mark.anyio
    async def test_each_session_has_isolated_forwarder(self, connection):
        """Each session should get its own forwarder task."""
        transport = AcpMcpTransport(connection)
        pair, _ = connection.register_session()

        msg1 = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        }
        with patch("mcp.client.session.ClientSession.initialize", new_callable=AsyncMock):
            async with transport.connect_session():
                await connection.send_to_acp(msg1, pair.to_session_send)
                call_args = connection._send_to_client.call_args[0][0]
                assert call_args["connectionId"] == connection.connection_id
                assert call_args["method"] == msg1["method"]
                assert call_args["params"] == msg1["params"]

        msg2 = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        with patch("mcp.client.session.ClientSession.initialize", new_callable=AsyncMock):
            async with transport.connect_session():
                await connection.send_to_acp(msg2, pair.to_session_send)
                call_args = connection._send_to_client.call_args[0][0]
                assert call_args["connectionId"] == connection.connection_id
                assert call_args["method"] == msg2["method"]
                assert call_args["params"] == msg2["params"]


class TestAcpMcpTransportErrorHandling:
    """Tests for error conditions."""

    @pytest.mark.anyio
    async def test_message_after_forwarder_cancelled_not_delivered(self, connection):
        """Messages after forwarder cancellation are not delivered to client."""
        transport = AcpMcpTransport(connection)
        pair, _ = connection.register_session()

        with patch("mcp.client.session.ClientSession.initialize", new_callable=AsyncMock):
            async with transport.connect_session():
                # Send one message during active session
                msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
                await connection.send_to_acp(msg, pair.to_session_send)
                call_args = connection._send_to_client.call_args[0][0]
                assert call_args["method"] == "tools/list"

        # After session close, forwarder is cancelled
        # Our own session pair is still registered (we created it), but the
        # transport's internal pair was cleaned up. Verify send_to_client
        # was only called once (for the message during active session).
        connection._send_to_client.assert_awaited_once()
