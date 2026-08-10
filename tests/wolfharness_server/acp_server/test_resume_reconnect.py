"""Integration test: resume after WebSocket reconnect creates fresh ACP connections.

Simulates the full disconnect → reconnect → resume cycle:

1. Create an ``AcpMcpConnectionManager`` and register a session connection.
2. Create an ``MCPManager`` per-session context (``get_or_create_session`` +
   ``add_acp_transport``).
3. Simulate WebSocket disconnect by calling ``cleanup_session()`` on the
   ``AcpMcpConnectionManager`` — this is what happens when the transport
   drops.
4. Verify old ACP connections are cleaned up (``_session_connections`` no
   longer has the session_id, the ``AcpMcpConnection`` is removed).
5. Simulate reconnect + resume by calling ``get_or_create_session`` again
   (creates a fresh ``McpSessionContext``) and registering a new ACP
   connection with a new ``connection_id``.
6. Verify the new connection is fresh — different object, different
   ``connection_id``, no stale references from the pre-disconnect session.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from acp.schema.mcp import AcpMcpServer
from wolfharness.mcp_server.manager import MCPManager
from wolfharness_server.acp_server.acp_mcp_manager import AcpMcpConnectionManager


class _FakeTransport:
    """Fake ``ClientTransport`` for testing ``add_acp_transport``."""

    def __init__(self, label: str = "fake-acp") -> None:
        self.label = label

    @asynccontextmanager
    async def connect_session(self):
        """Fake ``connect_session`` that yields immediately."""
        yield


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resume_after_reconnect() -> None:  # noqa: PLR0915
    """After WebSocket disconnect + reconnect + resume, ACP connections are fresh.

    Steps:
        1. Create AcpMcpConnectionManager + register session connection.
        2. Create MCPManager session context + add_acp_transport.
        3. Simulate disconnect via cleanup_session() on AcpMcpConnectionManager.
        4. Verify old connections cleaned up.
        5. Reconnect: get_or_create_session again + register fresh connection.
        6. Verify fresh connections, no stale references.
    """
    session_id = "test-reconnect-session"
    server_config = AcpMcpServer(name="test-server", id="test-id")
    send_to_client = AsyncMock(return_value=None)

    # --- AcpMcpConnectionManager (manages ACP-level connections) ---
    acp_manager = AcpMcpConnectionManager()

    # --- MCPManager (manages per-session MCP contexts) ---
    mcp_manager = MCPManager()
    # Wire the ACP manager into the MCP manager so cleanup_session delegates.
    mcp_manager._acp_mcp_manager = acp_manager

    # --- Step 1: Create initial ACP connection + register session ---
    old_connection_id = "conn-original"
    old_conn = await acp_manager.create_connection(old_connection_id, server_config, send_to_client)
    _old_pair, old_session_key = old_conn.register_session()
    acp_manager.register_session_connection(session_id, old_connection_id, old_session_key)

    # --- Step 2: Create MCP session context + add ACP transport ---
    old_ctx = mcp_manager.get_or_create_session(session_id)
    old_transport = _FakeTransport("original-transport")
    await mcp_manager.add_acp_transport(
        session_id,
        client_id="test-client",
        transport=old_transport,
        connection_id=old_connection_id,
        session_key=old_session_key,
    )

    # Verify initial state: session has registered connections
    assert session_id in acp_manager._session_connections
    assert (old_connection_id, old_session_key) in acp_manager._session_connections[session_id]
    assert old_conn.has_active_sessions()
    assert (old_connection_id, old_session_key) in old_ctx.acp_connection_ids
    assert old_ctx.connection_pool is not None

    # --- Step 3: Simulate WebSocket disconnect ---
    # cleanup_session() on AcpMcpConnectionManager pops the session from
    # _session_connections, unregisters stream pairs, and removes idle
    # connections. This is what happens when the WebSocket transport drops.
    await acp_manager.cleanup_session(session_id)

    # --- Step 4: Verify old ACP connections are cleaned up ---
    # _session_connections no longer has the session_id
    assert session_id not in acp_manager._session_connections
    # The old connection was the only one for this session, so it should
    # have been removed (no active sessions remaining)
    assert old_connection_id not in acp_manager
    assert acp_manager.get_connection(old_connection_id) is None
    # Old connection's session streams should be cleared
    assert not old_conn.has_active_sessions()

    # Also clean up the MCPManager session context (simulates full cleanup
    # as would happen via SessionController.close_session or ACPSession.close)
    await mcp_manager.cleanup_session(session_id)

    # MCPManager session context should be removed
    assert mcp_manager.get_session_context(session_id) is None

    # --- Step 5: Reconnect + resume — create fresh session context ---
    # get_or_create_session creates a NEW McpSessionContext with fresh state
    new_ctx = mcp_manager.get_or_create_session(session_id)

    # Create a fresh ACP connection (new connection_id, new session_key)
    new_connection_id = "conn-reconnected"
    new_conn = await acp_manager.create_connection(new_connection_id, server_config, send_to_client)
    _new_pair, new_session_key = new_conn.register_session()
    acp_manager.register_session_connection(session_id, new_connection_id, new_session_key)

    # Add fresh transport to the new session context
    new_transport = _FakeTransport("reconnected-transport")
    await mcp_manager.add_acp_transport(
        session_id,
        client_id="test-client",
        transport=new_transport,
        connection_id=new_connection_id,
        session_key=new_session_key,
    )

    # --- Step 6: Verify fresh connections, no stale references ---

    # 6a: New McpSessionContext is a different object from the old one
    assert new_ctx is not old_ctx

    # 6b: New context has fresh toolset_cache (empty, different object)
    assert new_ctx.toolset_cache is not old_ctx.toolset_cache
    assert len(new_ctx.toolset_cache) == 0

    # 6c: New context has a fresh connection_pool (different object)
    assert new_ctx.connection_pool is not None
    assert new_ctx.connection_pool is not old_ctx.connection_pool

    # 6d: New context's acp_connection_ids does NOT contain old connection
    assert (old_connection_id, old_session_key) not in new_ctx.acp_connection_ids
    assert (new_connection_id, new_session_key) in new_ctx.acp_connection_ids

    # 6e: AcpMcpConnectionManager has only the new connection
    assert old_connection_id not in acp_manager
    assert new_connection_id in acp_manager
    assert acp_manager.get_connection(new_connection_id) is new_conn

    # 6f: _session_connections has only the new (connection_id, session_key)
    assert session_id in acp_manager._session_connections
    assert (old_connection_id, old_session_key) not in acp_manager._session_connections[session_id]
    assert (new_connection_id, new_session_key) in acp_manager._session_connections[session_id]

    # 6g: New connection is active and distinct from old
    assert new_conn.has_active_sessions()
    assert new_conn is not old_conn
    assert new_conn.connection_id == new_connection_id

    # 6h: Old connection is fully closed (no active sessions)
    assert not old_conn.has_active_sessions()

    # --- Cleanup ---
    await mcp_manager.cleanup_session(session_id)
    await mcp_manager.cleanup()
