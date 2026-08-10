"""Global connection pool for sharing MCP connections across sessions.

For HTTP/SSE servers: creates a fresh transport per ``get_transport()``
call. HTTP connections are cheap and stateless — no need to share.

For stdio servers: uses the owner-task pattern to keep the subprocess
alive. ``get_transport()`` returns a ``_SharedSessionTransport`` wrapper
whose ``connect_session()`` yields a shared ``ClientSession`` managed
by the owner task. This avoids multiple ``connect_session()`` calls
on the underlying transport.

Shared stdio connections live for the pool lifetime — no ref counting
or LRU eviction. Pool-level MCP servers are typically few (<10) and
long-lived, so the overhead of tracking references is unnecessary.
"""  # allow: SIZE_OK — single cohesive class with tightly coupled private state

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import logging
import threading
from typing import TYPE_CHECKING, Any

from fastmcp.client.transports import ClientTransport


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from wolfharness_config.mcp_server import BaseMCPServerConfig


logger = logging.getLogger(__name__)

#: Warn if the number of cached stdio connections exceeds this threshold.
_CONNECTION_COUNT_WARNING_THRESHOLD: int = 50


class _SharedSessionTransport(ClientTransport):
    """Transport wrapper that yields a shared ClientSession.

    Used for stdio connections where the owner task manages the
    underlying ``connect_session()`` context. Each call to this
    wrapper's ``connect_session()`` yields the same shared session
    without calling the underlying transport again.
    """

    def __init__(self, session: Any, ready_event: asyncio.Event) -> None:
        self._session = session
        self._ready_event = ready_event

    @asynccontextmanager
    async def connect_session(self, **kwargs: Any) -> AsyncIterator[Any]:
        """Yield the shared session."""
        await self._ready_event.wait()
        yield self._session


@dataclass
class _PooledConnection:
    """Internal state for a pooled stdio MCP connection.

    Only stdio connections are cached. HTTP/SSE transports are created
    fresh per ``get_transport()`` call and never stored here.
    """

    transport: ClientTransport
    owner_task: asyncio.Task[None] | None = None
    ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    close_event: asyncio.Event = field(default_factory=asyncio.Event)
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    shared_session_transport: _SharedSessionTransport | None = None


class GlobalConnectionPool:
    """Pool-level singleton for sharing MCP connections across sessions.

    Manages connections for pool-level and agent-level MCP servers.
    Assumes servers are stateless (safe to share across sessions/users).

    Uses the owner-task pattern (deer-flow) for stdio servers:
    - A dedicated asyncio.Task enters and exits the transport context manager.
    - Callers signal via Events; the owner task handles lifecycle.
    - This eliminates cross-task CancelScope errors.

    HTTP/SSE servers use direct transports (no cancel scope issues).

    Shared stdio connections live for the pool lifetime. No ref counting
    or LRU eviction is performed — pool-level MCP servers are typically
    few and long-lived.
    """

    # NOTE: threading.Lock is used (not asyncio.Lock) because all operations
    # inside the lock are synchronous dict operations with no await. If async
    # work is added inside the lock, switch to asyncio.Lock to avoid blocking
    # the event loop.

    def __init__(self) -> None:
        self._connections: dict[str, _PooledConnection] = {}
        self._lock = threading.Lock()

    async def get_transport(
        self,
        config: BaseMCPServerConfig,
    ) -> ClientTransport:
        """Get or create a shared transport for the given config.

        For stdio: uses owner-task pattern (dedicated asyncio.Task).
        For HTTP/SSE: creates direct transport (fresh per call, no caching).
        For ACP: raises NotImplementedError (handled by SessionConnectionPool).

        Args:
            config: MCP server configuration.

        Returns:
            A ClientTransport that can be used to construct MCPToolset.

        Raises:
            NotImplementedError: For ACP transport servers.
            RuntimeError: If the owner task fails to initialize the connection.
            TimeoutError: If the owner task does not become ready in time.
        """
        from wolfharness_config.mcp_server import (
            AcpMCPServerConfig,
            StdioMCPServerConfig,
        )

        if isinstance(config, AcpMCPServerConfig):
            raise NotImplementedError(
                "ACP transport is handled by SessionConnectionPool, not GlobalConnectionPool"
            )

        # HTTP/SSE: fresh transport per call — no sharing, no caching.
        if not isinstance(config, StdioMCPServerConfig):
            return config.to_transport()

        # stdio: shared via owner-task pattern
        client_id = config.client_id

        with self._lock:
            existing = self._connections.get(client_id)
            if existing is not None:
                # Reuse existing stdio connection (managed by owner task)
                ready_event = existing.ready_event
                owner_task = existing.owner_task
            else:
                if len(self._connections) >= _CONNECTION_COUNT_WARNING_THRESHOLD:
                    logger.warning(
                        "GlobalConnectionPool has %d cached stdio connections — "
                        "consider disabling unused MCP servers in YAML config",
                        len(self._connections),
                    )
                transport = config.to_transport()
                conn = _PooledConnection(transport=transport)
                conn.owner_task = asyncio.create_task(
                    self._run_session(conn, client_id),
                    name=f"mcp-owner-{client_id}",
                )
                self._connections[client_id] = conn
                ready_event = conn.ready_event
                owner_task = conn.owner_task

        # Wait for owner task to become ready (outside the lock).
        # Use the server's configured timeout (default 30s) so that
        # servers with longer startup times (e.g. npx wrappers) are
        # not killed prematurely.
        if owner_task is not None:
            ready_timeout = config.timeout
            try:
                await asyncio.shield(asyncio.wait_for(ready_event.wait(), timeout=ready_timeout))
            except TimeoutError:
                msg = f"Owner task for {client_id} did not become ready in {ready_timeout}s"
                logger.error(msg)  # noqa: TRY400
                raise TimeoutError(msg) from None

            if owner_task.done() and owner_task.exception() is not None:
                exc = owner_task.exception()
                msg = f"Owner task for {client_id} failed: {exc}"
                logger.error(msg)
                raise RuntimeError(msg) from exc

        # Return the shared session transport wrapper, not the raw
        # transport. This prevents callers from calling
        # connect_session() on the underlying transport again.
        existing = self._connections.get(client_id)
        if existing is not None and existing.shared_session_transport is not None:
            return existing.shared_session_transport

        # Fallback: should not happen if owner task succeeded
        return self._connections[client_id].transport

    async def _run_session(
        self,
        conn: _PooledConnection,
        client_id: str,
    ) -> None:
        """Owner-task body for stdio connections.

        Enters the transport context manager, captures the session,
        creates a _SharedSessionTransport wrapper, signals readiness,
        waits for close signal, then exits.

        On both clean shutdown and unexpected crash, the connection is
        removed from the pool so that subsequent ``get_transport()``
        calls will create a fresh connection instead of getting a dead
        entry.

        Args:
            conn: The pooled connection state.
            client_id: Cache key for logging.
        """
        logger.debug("Owner task starting for %s", client_id)
        try:
            async with conn.transport.connect_session() as session:
                conn.shared_session_transport = _SharedSessionTransport(
                    session=session,
                    ready_event=conn.ready_event,
                )
                conn.ready_event.set()
                logger.debug("Owner task ready for %s", client_id)
                await conn.close_event.wait()
                logger.debug("Owner task received close signal for %s", client_id)
        except Exception:
            logger.exception("Owner task error for %s", client_id)
            conn.ready_event.set()
            raise
        finally:
            # Remove from pool to prevent stale entries on crash.
            # Safe under lock: if shutdown_all() already removed it,
            # pop returns None and this is a no-op.
            with self._lock:
                if self._connections.get(client_id) is conn:
                    del self._connections[client_id]
            conn.done_event.set()
            logger.debug("Owner task done for %s", client_id)

    async def shutdown_all(self, timeout: float = 10.0) -> None:
        """Clean shutdown of all connections. Called on pool shutdown.

        Signals all owner tasks to shut down and waits for them to
        complete within the timeout.

        Args:
            timeout: Maximum time to wait for all connections to shut down.
        """
        with self._lock:
            items = list(self._connections.items())
            for _cid, conn in items:
                conn.close_event.set()

        stdio_tasks: list[asyncio.Task[None]] = [
            conn.owner_task for _, conn in items if conn.owner_task is not None
        ]

        for cid, _conn in items:
            logger.debug("Signaled shutdown for %s", cid)

        if not stdio_tasks:
            return

        _done, pending = await asyncio.wait(
            stdio_tasks,
            timeout=timeout,
            return_when=asyncio.ALL_COMPLETED,
        )

        for task in pending:
            logger.warning("Cancelling owner task that did not shut down: %s", task.get_name())
            task.cancel()

        for task in pending:
            with contextlib.suppress(asyncio.CancelledError):
                await task
            if not task.cancelled() and task.exception() is not None:
                logger.exception("Error during shutdown of owner task: %s", task.get_name())

        with self._lock:
            self._connections.clear()

        logger.info("GlobalConnectionPool shutdown complete")
