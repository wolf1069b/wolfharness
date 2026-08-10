"""MCP-over-ACP connection manager.

Manages bidirectional MCP connections tunnelled over the ACP protocol.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

import anyio

from wolfharness.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Callable

    from anyio.streams.memory import MemoryObjectSendStream

    from acp.schema.mcp import AcpMcpServer

from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage
from pydantic import ValidationError

from wolfharness.mcp_server.session_stream_pair import SessionStreamPair


logger = get_logger(__name__)


class AcpMcpConnection:
    """Represents a single active MCP-over-ACP connection.

    Bridges MCP JSON-RPC messages between fastmcp's ClientSession
    and the ACP connection via mcp/message send_request calls.

    Supports multiple concurrent ClientSession instances sharing the
    same ACP connection. Each ``connect_session()`` call on the
    transport registers a per-session stream pair, and responses
    are routed back to the correct session via the forwarder task.
    """

    def __init__(
        self,
        connection_id: str,
        server_config: AcpMcpServer,
        send_to_client: Callable[[dict[str, Any]], Any],
    ) -> None:
        """Initialize an MCP-over-ACP connection.

        Args:
            connection_id: Unique identifier for this connection.
            server_config: The ACP MCP server configuration.
            send_to_client: Callable to send mcp/message to the client.
        """
        self.connection_id = connection_id
        self.server_config = server_config
        self._send_to_client = send_to_client
        self._closed = False
        # Per-session stream pairs for concurrent ClientSession support.
        self._session_streams: dict[int, SessionStreamPair] = {}
        self._next_session_key = 0

    async def close(self) -> None:
        """Close the connection and clean up all streams."""
        if self._closed:
            return
        self._closed = True
        # Close all per-session stream pairs
        for pair in list(self._session_streams.values()):
            await pair.close()
        self._session_streams.clear()
        logger.info("MCP-over-ACP connection closed", connection_id=self.connection_id)

    def register_session(self) -> tuple[SessionStreamPair, int]:
        """Create and register a per-session stream pair.

        Returns a ``(SessionStreamPair, int)`` tuple where the pair
        has independent streams for this session's ClientSession to
        read/write, and the int is the internal key used to store
        the pair in ``_session_streams``. The caller is responsible
        for calling ``unregister_session()`` when done.
        """
        key = self._next_session_key
        self._next_session_key += 1
        to_send, to_recv = anyio.create_memory_object_stream[dict[str, Any]](0)
        from_send, from_recv = anyio.create_memory_object_stream[dict[str, Any]](0)
        pair = SessionStreamPair(
            to_session_send=to_send,
            to_session_receive=to_recv,
            from_session_send=from_send,
            from_session_receive=from_recv,
        )
        self._session_streams[key] = pair
        return pair, key

    def unregister_session(self, pair: SessionStreamPair) -> None:
        """Remove a per-session stream pair from the connection."""
        for key, val in list(self._session_streams.items()):
            if val is pair:
                del self._session_streams[key]
                break

    def has_active_sessions(self) -> bool:
        """Return whether any per-session stream pairs are active."""
        return len(self._session_streams) > 0

    async def send_to_acp(
        self,
        message: Any,
        response_stream: MemoryObjectSendStream[dict[str, Any]],
    ) -> Any:
        """Send an MCP message to the ACP client and route the response.

        Like ``send_to_client()`` but writes the response to the
        caller's ``response_stream`` instead of the shared
        ``_to_session_send``. This enables per-session routing.

        Args:
            message: MCP JSON-RPC message dict or SessionMessage.
            response_stream: The caller session's ``to_session_send``.

        Returns:
            Response from client (for requests) or None (for notifications).
        """
        if isinstance(message, SessionMessage):
            message = message.message.model_dump(by_alias=True, mode="json", exclude_none=True)

        if not isinstance(message, dict):
            logger.warning(
                "Unexpected message type in send_to_acp",
                type=type(message).__name__,
                connection_id=self.connection_id,
            )
            return None

        original_id = message.get("id")
        method = message.get("method")
        params = message.get("params")

        wrapped: dict[str, Any] = {
            "connectionId": self.connection_id,
            "method": method,
        }
        if params is not None:
            wrapped["params"] = params

        try:
            result = await self._send_to_client(wrapped)
        except Exception as exc:
            is_optional_mcp = isinstance(method, str) and method.startswith((
                "resources/",
                "prompts/",
            ))
            if is_optional_mcp:
                logger.debug(
                    "MCP method not supported by client",
                    connection_id=self.connection_id,
                    method=method,
                    error=str(exc),
                )
            else:
                logger.exception(
                    "Error sending mcp/message to client",
                    connection_id=self.connection_id,
                    method=method,
                )
            if original_id is not None:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": original_id,
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {exc}",
                    },
                }
                with contextlib.suppress(anyio.BrokenResourceError, anyio.ClosedResourceError):
                    await response_stream.send(
                        SessionMessage(message=JSONRPCMessage.model_validate(error_response))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                    )
            return None

        if original_id is not None and isinstance(result, dict):
            if "error" in result:
                response = {
                    "jsonrpc": "2.0",
                    "id": original_id,
                    "error": result["error"],
                }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": original_id,
                    "result": result,
                }
            try:
                await response_stream.send(
                    SessionMessage(message=JSONRPCMessage.model_validate(response))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                )
            except ValidationError:
                logger.exception(
                    "Invalid JSON-RPC response from mcp/message",
                    response=response,
                    connection_id=self.connection_id,
                )
                if original_id is not None:
                    fallback = {
                        "jsonrpc": "2.0",
                        "id": original_id,
                        "error": {
                            "code": -32603,
                            "message": f"Invalid upstream response: {result.get('error', result)}",
                        },
                    }
                    with contextlib.suppress(anyio.BrokenResourceError, anyio.ClosedResourceError):
                        await response_stream.send(
                            SessionMessage(message=JSONRPCMessage.model_validate(fallback))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                        )
            except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                logger.debug(
                    "Cannot forward response: session stream already closed",
                    connection_id=self.connection_id,
                )

        return result

    async def broadcast_to_sessions(self, message: dict[str, Any]) -> None:
        """Broadcast a server-initiated notification to all active sessions.

        Used by ``handle_client_message()`` to deliver notifications
        (e.g. ``notifications/tools/list_changed``) to every active
        ClientSession.
        """
        session_msg: SessionMessage | None = None
        if isinstance(message, SessionMessage):
            session_msg = message
        elif isinstance(message, dict) and "jsonrpc" in message:
            session_msg = SessionMessage(message=JSONRPCMessage.model_validate(message))
        else:
            method = message.get("method")
            params = message.get("params")
            jsonrpc_message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
            if params is not None:
                jsonrpc_message["params"] = params
            session_msg = SessionMessage(message=JSONRPCMessage.model_validate(jsonrpc_message))

        for pair in list(self._session_streams.values()):
            with contextlib.suppress(anyio.BrokenResourceError, anyio.ClosedResourceError):
                await pair.to_session_send.send(session_msg)

    async def handle_client_message(self, message: dict[str, Any]) -> None:
        """Handle an incoming mcp/message from the client by broadcasting to all sessions."""
        await self.broadcast_to_sessions(message)


class AcpMcpConnectionManager:
    """Manages multiple MCP-over-ACP connections.

    Maps connection IDs to active AcpMcpConnection instances.
    """

    def __init__(self) -> None:
        """Initialize the connection manager."""
        self._connections: dict[str, AcpMcpConnection] = {}
        self._session_connections: dict[str, set[tuple[str, int]]] = {}
        self._lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()

    def register_session_connection(
        self,
        session_id: str,
        connection_id: str,
        session_key: int,
    ) -> None:
        """Register that a session has an active MCP connection.

        Tracks the (connection_id, session_key) tuple for the given
        session so that cleanup can find all connections belonging to
        a session. Uses a set for deduplication.

        Args:
            session_id: The ACP session ID.
            connection_id: The MCP connection ID.
            session_key: The int key used in AcpMcpConnection._session_streams.
        """
        self._session_connections.setdefault(session_id, set()).add(
            (connection_id, session_key),
        )

    async def cleanup_session(self, session_id: str) -> None:
        """Clean up all MCP connections associated with a session.

        Pops the session's ``(connection_id, session_key)`` pairs from
        the reverse index, unregisters each pair from its
        ``AcpMcpConnection``, and removes connections that have no
        remaining active sessions.

        This method is idempotent — calling it for a session that was
        already cleaned up (or never registered) is a no-op.

        Args:
            session_id: The ACP session ID to clean up.
        """
        async with self._cleanup_lock:
            entries = self._session_connections.pop(session_id, None)
            if entries is None:
                return

            # Group entries by connection_id to batch cleanup per connection.
            connections_to_check: set[str] = set()
            for connection_id, session_key in entries:
                conn = self._connections.get(connection_id)
                if conn is None:
                    continue
                pair = conn._session_streams.get(session_key)
                if pair is not None:
                    try:
                        conn.unregister_session(pair)
                    except Exception:
                        logger.exception(
                            "Failed to unregister session stream pair",
                            connection_id=connection_id,
                            session_key=session_key,
                        )
                connections_to_check.add(connection_id)

            # Remove connections with no remaining active sessions.
            for connection_id in connections_to_check:
                conn = self._connections.get(connection_id)
                if conn is not None and not conn.has_active_sessions():
                    try:
                        await self.remove_connection(connection_id)
                    except Exception:
                        logger.exception(
                            "Failed to remove idle MCP connection",
                            connection_id=connection_id,
                        )

    async def create_connection(
        self,
        connection_id: str,
        server_config: AcpMcpServer,
        send_to_client: Callable[[dict[str, Any]], Any],
    ) -> AcpMcpConnection:
        """Create and register a new MCP-over-ACP connection.

        Args:
            connection_id: Unique identifier for this connection.
            server_config: The ACP MCP server configuration.
            send_to_client: Callable to send mcp/message to the client.

        Returns:
            The newly created connection.

        Raises:
            ValueError: If a connection with the same ID already exists.
        """
        async with self._lock:
            if not connection_id:
                raise ValueError("connection_id cannot be empty")
            if connection_id in self._connections:
                raise ValueError(f"MCP connection '{connection_id}' already exists")
            conn = AcpMcpConnection(connection_id, server_config, send_to_client)
            self._connections[connection_id] = conn
            logger.info("MCP connection created", connection_id=connection_id)
            return conn

    async def remove_connection(self, connection_id: str) -> None:
        """Remove and close an MCP-over-ACP connection.

        Args:
            connection_id: The connection ID to remove.
        """
        async with self._lock:
            conn = self._connections.pop(connection_id, None)
        if conn is not None:
            try:
                await conn.close()
            except Exception:
                logger.exception("Failed to close MCP connection", connection_id=connection_id)
            logger.info("MCP connection removed", connection_id=connection_id)

    def get_connection(self, connection_id: str) -> AcpMcpConnection | None:
        """Get an active connection by ID.

        Args:
            connection_id: The connection ID to look up.

        Returns:
            The connection if found, None otherwise.
        """
        return self._connections.get(connection_id)

    async def close_all(self) -> None:
        """Close all active connections."""
        async with self._lock:
            connections = list(self._connections.values())
            self._connections.clear()
        for conn in connections:
            try:
                await conn.close()
            except Exception:
                logger.exception("Failed to close MCP connection", connection_id=conn.connection_id)
        logger.info("All MCP connections closed")

    def get_connection_ids(self) -> list[str]:
        """Return a snapshot of all active connection IDs."""
        return list(self._connections.keys())

    def __contains__(self, connection_id: str) -> bool:
        """Check if a connection ID is active."""
        return connection_id in self._connections

    def __len__(self) -> int:
        """Return the number of active connections."""
        return len(self._connections)
