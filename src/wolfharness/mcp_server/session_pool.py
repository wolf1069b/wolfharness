"""Per-session MCP connection pool for transport lifecycle isolation.

Each :class:`SessionConnectionPool` is scoped to a single session and manages
MCP :class:`~fastmcp.client.ClientTransport` instances keyed by
``(client_id, skill_name)``.  This ensures skill MCP servers are isolated
even within the same session.

Stdio transports use an owner-task pattern: a dedicated asyncio task holds
the transport, signals readiness via events, and shuts down cleanly when
signalled by :meth:`SessionConnectionPool.cleanup`.

HTTP/SSE transports are lightweight — created directly without an owner
task (no cancel-scope issues).

Pre-created transports (e.g. ``AcpMcpTransport``) are added via
:meth:`SessionConnectionPool.add_transport` and stored without an owner task.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
import threading
from typing import TYPE_CHECKING

from wolfharness.log import get_logger


if TYPE_CHECKING:
    from fastmcp.client import ClientTransport

    from wolfharness.mcp_server.client import MCPClient
    from wolfharness_config.mcp_server import BaseMCPServerConfig, MCPServerConfig


logger = get_logger(__name__)


@dataclass
class _SessionConnection:
    """Internal tracking for a per-session MCP transport."""

    transport: ClientTransport
    """The fastmcp transport instance."""

    owner_task: asyncio.Task[None] | None = None
    """Owner asyncio task for stdio transports; ``None`` for HTTP/SSE/pre-created."""

    ready_event: asyncio.Event | None = None
    """Set when the transport is ready for use (stdio only)."""

    close_event: asyncio.Event | None = None
    """Set to signal the owner task to shut down (stdio only)."""

    done_event: asyncio.Event | None = None
    """Set when the owner task has completed cleanup (stdio only)."""

    is_stdio: bool = False
    """Whether this connection uses a stdio transport."""

    is_pre_created: bool = False
    """Whether the transport was added via ``add_transport`` (e.g. ACP)."""


def _create_transport(config: BaseMCPServerConfig) -> ClientTransport:
    """Create a fastmcp ``ClientTransport`` from server config.

    Args:
        config: MCP server configuration.

    Returns:
        A fastmcp ``ClientTransport`` instance.

    Raises:
        NotImplementedError: For ACP-transport configs (use ``add_transport``).
    """
    from fastmcp.client import SSETransport, StreamableHttpTransport
    from fastmcp.client.transports import StdioTransport

    from wolfharness_config.mcp_server import (
        AcpMCPServerConfig,
        SSEMCPServerConfig,
        StdioMCPServerConfig,
        StreamableHTTPMCPServerConfig,
        _expand_headers,
    )

    match config:
        case StdioMCPServerConfig(command=command, args=args):
            env = config.get_env_vars()
            return StdioTransport(command=command, args=args, env=env)
        case SSEMCPServerConfig(url=url, headers=headers):
            from wolfharness_config.mcp_server import make_mcp_httpx_client_factory

            return SSETransport(
                url=str(url),
                headers=_expand_headers(headers),
                httpx_client_factory=make_mcp_httpx_client_factory(read_timeout=config.timeout),
            )
        case StreamableHTTPMCPServerConfig(url=url, headers=headers):
            from wolfharness_config.mcp_server import make_mcp_httpx_client_factory

            return StreamableHttpTransport(
                url=str(url),
                headers=_expand_headers(headers),
                httpx_client_factory=make_mcp_httpx_client_factory(read_timeout=config.timeout),
            )
        case AcpMCPServerConfig():
            raise NotImplementedError(
                "ACP-transport MCP servers must use add_transport(). "
                "The transport is created externally by AcpMcpTransport."
            )
        case _ as unreachable:
            raise NotImplementedError(
                f"Transport creation not supported for config type: {unreachable.type}"
            )


class SessionConnectionPool:
    """Per-session MCP transport pool with isolation.

    Each session gets its own pool of MCP transports.  Transports are keyed
    by ``(client_id, skill_name)`` to ensure skill MCP servers are isolated
    even within the same session.

    For stdio transports, an owner-task pattern manages the lifecycle:
    a dedicated asyncio task holds the transport, signals readiness via
    events, and shuts down cleanly when signalled.

    For HTTP/SSE transports, the transport is created directly without
    an owner task (lightweight, no cancel-scope issues).

    Pre-created transports (e.g. ``AcpMcpTransport``) are added via
    :meth:`add_transport` and stored without an owner task.
    """

    def __init__(self, session_id: str) -> None:
        """Initialize the session connection pool.

        Args:
            session_id: Unique identifier for the session that owns this pool.
        """
        self._session_id = session_id
        self._connections: dict[tuple[str, str | None], _SessionConnection] = {}
        self._lock = threading.Lock()

    async def get_transport(
        self,
        config: BaseMCPServerConfig,
        skill_name: str | None = None,
    ) -> ClientTransport:
        """Get or create a transport for the given config.

        If a transport for this ``(client_id, skill_name)`` pair already
        exists, returns the cached transport.  Otherwise creates a new one.

        For stdio transports, an owner task is spawned to manage the lifecycle.

        Args:
            config: MCP server configuration.
            skill_name: Optional skill name for skill-scoped MCP isolation.

        Returns:
            A fastmcp ``ClientTransport`` instance.
        """
        key = (config.client_id, skill_name)

        with self._lock:
            existing = self._connections.get(key)
            if existing is not None:
                conn = existing
            else:
                transport = _create_transport(config)
                is_stdio = config.type == "stdio"

                if is_stdio:
                    ready_event = asyncio.Event()
                    close_event = asyncio.Event()
                    done_event = asyncio.Event()
                    owner_task = asyncio.create_task(
                        _stdio_owner_task(
                            transport=transport,
                            ready_event=ready_event,
                            close_event=close_event,
                            done_event=done_event,
                        )
                    )
                    conn = _SessionConnection(
                        transport=transport,
                        owner_task=owner_task,
                        ready_event=ready_event,
                        close_event=close_event,
                        done_event=done_event,
                        is_stdio=True,
                        is_pre_created=False,
                    )
                else:
                    conn = _SessionConnection(
                        transport=transport,
                        is_stdio=False,
                        is_pre_created=False,
                    )
                self._connections[key] = conn

        # Wait for ready outside the lock (stdio only)
        if conn.ready_event is not None:
            await conn.ready_event.wait()

        return conn.transport

    async def get_client(
        self,
        config: MCPServerConfig,
        skill_name: str | None = None,
    ) -> MCPClient:
        """Get a pooled ``MCPClient`` for the given config.

        Wraps the pooled transport, constructs ``MCPClient`` on the
        transport, and performs the MCP handshake (initialize +
        capabilities exchange). The pool retains ownership of the
        transport — do NOT close the client's transport externally.

        Args:
            config: MCP server configuration.
            skill_name: Optional skill name for skill-scoped isolation.

        Returns:
            A connected ``MCPClient`` wrapping the pooled transport.
        """
        from wolfharness.mcp_server.client import MCPClient

        transport = await self.get_transport(config, skill_name)
        client = MCPClient(config=config, transport=transport)
        await client.__aenter__()
        return client

    async def add_transport(
        self,
        client_id: str,
        transport: ClientTransport,
        skill_name: str | None = None,
    ) -> None:
        """Add a pre-created transport to the pool.

        Used for transports created externally (e.g. ``AcpMcpTransport``
        which wraps an ACP JSON-RPC tunnel).  No owner task is spawned.

        Args:
            client_id: Client identifier for the transport.
            transport: Pre-created fastmcp ``ClientTransport``.
            skill_name: Optional skill name for skill-scoped MCP isolation.
        """
        key = (client_id, skill_name)
        with self._lock:
            self._connections[key] = _SessionConnection(
                transport=transport,
                is_stdio=False,
                is_pre_created=True,
            )

    async def copy_pre_created_transports(self, source: SessionConnectionPool) -> None:
        """Copy pre-created transports (e.g. ACP) from another pool.

        Used when creating child session agents: the child inherits the
        parent's ACP MCP configs via snapshot, but needs the corresponding
        transports to be available in its own pool for capability building.

        Only pre-created transports (``is_pre_created=True``) are copied.
        Stdio and HTTP/SSE transports are not shared — they have their own
        lifecycle managed by the owning pool.

        Args:
            source: The pool to copy pre-created transports from.
        """
        with source._lock:
            pre_created = {
                key: conn for key, conn in source._connections.items() if conn.is_pre_created
            }
        with self._lock:
            for key, conn in pre_created.items():
                if key not in self._connections:
                    self._connections[key] = _SessionConnection(
                        transport=conn.transport,
                        is_stdio=False,
                        is_pre_created=True,
                    )

    async def cleanup(self, timeout: float = 5.0) -> None:
        """Shut down all connections in the pool.

        Signals all owner-tasks to shut down, waits with timeout,
        and force-cancels remaining tasks.

        NOTE: This only cleans up stdio owner-task connections. Pre-created
        ACP transports (added via add_transport()) are NOT closed here —
        their underlying AcpMcpConnection lifecycle is managed by
        AcpMcpConnectionManager, not by this pool. This is intentional:
        child sessions share parent's ACP transport via
        copy_pre_created_transports(), and closing it here would break
        the parent's active sessions.

        Safe to call multiple times.

        Args:
            timeout: Maximum time in seconds to wait for graceful shutdown.
        """
        with self._lock:
            connections = list(self._connections.values())
            self._connections.clear()

        # Signal all owner tasks to shut down
        for conn in connections:
            if conn.close_event is not None:
                conn.close_event.set()

        # Wait for all owner tasks to complete with timeout
        owner_tasks = [conn.owner_task for conn in connections if conn.owner_task is not None]

        if owner_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*owner_tasks, return_exceptions=True),
                    timeout=timeout,
                )
            except TimeoutError:
                # Force-cancel remaining tasks
                for task in owner_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*owner_tasks, return_exceptions=True)

        logger.debug(
            "SessionConnectionPool cleaned up",
            session_id=self._session_id,
            connection_count=len(connections),
        )


async def _stdio_owner_task(
    transport: ClientTransport,
    ready_event: asyncio.Event,
    close_event: asyncio.Event,
    done_event: asyncio.Event,
) -> None:
    """Owner task for stdio transport lifecycle.

    Enters the transport's ``connect_session()`` context manager so the
    stdio subprocess is started, signals readiness, then waits for the
    close signal.  Provides clean shutdown coordination via events so
    that :meth:`SessionConnectionPool.cleanup` can gracefully shut down
    all stdio transports with a timeout and force-cancellation fallback.

    Args:
        transport: The stdio transport (held for lifecycle ownership).
        ready_event: Set when the transport is ready for use.
        close_event: Set to signal the owner task to shut down.
        done_event: Set when cleanup is complete.
    """
    try:
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(transport.connect_session())
            ready_event.set()
            await close_event.wait()
    except Exception:
        ready_event.set()  # set in case someone is waiting
        raise
    finally:
        done_event.set()
