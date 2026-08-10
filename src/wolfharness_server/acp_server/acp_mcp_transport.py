"""fastmcp ClientTransport implementation for MCP-over-ACP.

Bridges fastmcp's ClientSession to the ACP connection manager.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import anyio
from fastmcp.client.transports import ClientTransport
from mcp import ClientSession

from wolfharness.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from wolfharness_server.acp_server.acp_mcp_manager import AcpMcpConnection

logger = get_logger(__name__)

DEFAULT_READ_TIMEOUT_SECONDS: float = 600.0


class AcpMcpTransport(ClientTransport):
    """fastmcp ClientTransport that tunnels MCP over ACP.

    This transport is used by fastmcp's Client to communicate
    with an MCP server over the existing ACP connection.
    """

    def __init__(
        self,
        connection: AcpMcpConnection,
        timeout: float = DEFAULT_READ_TIMEOUT_SECONDS,
        on_session_registered: Callable[[str, int], None] | None = None,
    ) -> None:
        """Initialize the transport with an active ACP MCP connection.

        Args:
            connection: An active AcpMcpConnection with open streams.
            timeout: Read timeout in seconds for MCP session operations.
            on_session_registered: Optional callback invoked after
                ``register_session()`` with ``(connection_id, session_key)``.
                Used by callers to register the connection for cleanup
                tracking via
                ``AcpMcpConnectionManager.register_session_connection()``.
        """
        self._connection = connection
        self._timeout = timeout
        self._on_session_registered = on_session_registered

    @property
    def connection_id(self) -> str:
        """Return the ACP connection ID managed by this transport."""
        return self._connection.connection_id

    @asynccontextmanager
    async def connect_session(
        self,
        **session_kwargs: Any,
    ) -> AsyncIterator[ClientSession]:
        """Create a fastmcp ClientSession over ACP.

        Each call creates an independent per-session stream pair via
        ``register_session()``, so multiple ``ClientSession`` instances
        can share the same ``AcpMcpConnection`` without stream contention.

        Args:
            **session_kwargs: Additional arguments passed to ClientSession.

        Yields:
            A connected fastmcp ClientSession.
        """
        pair, _session_key = self._connection.register_session()

        if self._on_session_registered is not None:
            self._on_session_registered(self._connection.connection_id, _session_key)

        async def _forward_to_client() -> None:
            """Forward MCP requests from ClientSession to ACP client."""
            try:
                async for message in pair.from_session_receive:
                    await self._connection.send_to_acp(message, pair.to_session_send)
            except anyio.EndOfStream:
                pass
            except Exception:
                logger.exception("Error in MCP-over-ACP forwarder task")
                raise

        session_kwargs.pop("read_timeout_seconds", None)
        session = ClientSession(
            pair.to_session_receive,
            pair.from_session_send,
            read_timeout_seconds=timedelta(seconds=self._timeout),
            **session_kwargs,
        )

        forwarder = asyncio.create_task(_forward_to_client())
        try:
            async with session:
                yield session
        finally:
            forwarder.cancel()
            with suppress(asyncio.CancelledError):
                await forwarder
            self._connection.unregister_session(pair)
            await pair.close()
