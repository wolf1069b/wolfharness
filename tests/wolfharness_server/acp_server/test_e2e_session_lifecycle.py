"""End-to-end integration test: full ACP session lifecycle with MCP tools.

Exercises the complete session lifecycle that a real ACP client experiences:

1. **Connect** - Create ``MCPManager`` + ``AcpMcpConnectionManager``, register
   an ACP connection, and create a per-session MCP context.
2. **Session** - Store a config snapshot and build MCP capabilities via
   ``get_capabilities(session_id=...)``.
3. **MCP tool** - Verify the capability list is returned and the session
   context has the expected resources (connection pool, toolset cache,
   ACP connection tracking).
4. **Disconnect** - Simulate WebSocket drop: call ``cleanup_session()`` on
   both managers, verify all resources are gone.
5. **Reconnect** - Create a fresh ACP connection with a new connection_id
   and call ``get_or_create_session()`` again.
6. **Resume** - Store a new snapshot and call ``get_capabilities()`` again.
7. **Verify** - New toolset objects are different from pre-disconnect ones,
   session context is fresh (empty toolset_cache, different connection_pool),
   and ACP connections are fully replaced.

All external dependencies are mocked - no real WebSocket, no real ACP client,
no real model API key.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from acp.schema.mcp import AcpMcpServer
from wolfharness.mcp_server.config_snapshot import McpConfigSnapshot
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
@pytest.mark.slow
@pytest.mark.asyncio
async def test_e2e_session_lifecycle() -> None:  # noqa: PLR0915
    """Full ACP session lifecycle: connect → session → MCP tool → disconnect → reconnect → resume.

    Verifies that after a complete disconnect/reconnect cycle, all MCP
    resources (toolsets, connection pools, ACP connections) are fresh
    and no stale references from the pre-disconnect session survive.
    """
    session_id = "e2e-1"
    server_config = AcpMcpServer(name="e2e-server", id="e2e-server-id")
    send_to_client = AsyncMock(return_value=None)

    # --- Managers ---
    acp_manager = AcpMcpConnectionManager()
    mcp_manager = MCPManager(name="e2e-test")
    mcp_manager._acp_mcp_manager = acp_manager

    try:
        # ================================================================
        # Phase 1: CONNECT - Create initial ACP connection + session context
        # ================================================================
        old_connection_id = "conn-e2e-original"
        old_conn = await acp_manager.create_connection(
            old_connection_id,
            server_config,
            send_to_client,
        )
        _old_pair, old_session_key = old_conn.register_session()
        acp_manager.register_session_connection(
            session_id,
            old_connection_id,
            old_session_key,
        )

        old_ctx = mcp_manager.get_or_create_session(session_id)
        old_transport = _FakeTransport("original-transport")
        await mcp_manager.add_acp_transport(
            session_id,
            client_id="e2e-client",
            transport=old_transport,
            connection_id=old_connection_id,
            session_key=old_session_key,
        )

        # Verify initial connection state
        assert session_id in acp_manager._session_connections
        assert (old_connection_id, old_session_key) in acp_manager._session_connections[session_id]
        assert old_conn.has_active_sessions()
        assert (old_connection_id, old_session_key) in old_ctx.acp_connection_ids
        assert old_ctx.connection_pool is not None

        # ================================================================
        # Phase 2: SESSION - Store snapshot and build capabilities
        # ================================================================
        snapshot = McpConfigSnapshot()
        mcp_manager.update_session_snapshot(session_id, snapshot)
        assert old_ctx.snapshot is snapshot

        # ================================================================
        # Phase 3: MCP TOOL - get_capabilities returns toolsets
        # ================================================================
        old_caps = await mcp_manager.get_capabilities(session_id=session_id)
        # Empty snapshot → no servers → empty capability list
        assert old_caps == []
        # Session context still exists during the turn
        assert mcp_manager.get_session_context(session_id) is not None

        # Store references to pre-disconnect resources for later comparison
        old_toolset_cache = old_ctx.toolset_cache
        old_connection_pool = old_ctx.connection_pool

        # ================================================================
        # Phase 4: DISCONNECT - Simulate WebSocket drop
        # ================================================================
        # ACP transport drops first, then MCP session cleanup propagates
        await acp_manager.cleanup_session(session_id)
        await mcp_manager.cleanup_session(session_id)

        # Verify ACP connections are fully cleaned up
        assert session_id not in acp_manager._session_connections
        assert old_connection_id not in acp_manager
        assert acp_manager.get_connection(old_connection_id) is None
        assert not old_conn.has_active_sessions()

        # Verify MCP session context is removed
        assert mcp_manager.get_session_context(session_id) is None

        # ================================================================
        # Phase 5: RECONNECT - Create fresh ACP connection + session
        # ================================================================
        new_connection_id = "conn-e2e-reconnected"
        new_conn = await acp_manager.create_connection(
            new_connection_id,
            server_config,
            send_to_client,
        )
        _new_pair, new_session_key = new_conn.register_session()
        acp_manager.register_session_connection(
            session_id,
            new_connection_id,
            new_session_key,
        )

        new_ctx = mcp_manager.get_or_create_session(session_id)
        new_transport = _FakeTransport("reconnected-transport")
        await mcp_manager.add_acp_transport(
            session_id,
            client_id="e2e-client",
            transport=new_transport,
            connection_id=new_connection_id,
            session_key=new_session_key,
        )

        # ================================================================
        # Phase 6: RESUME - Store new snapshot and build capabilities again
        # ================================================================
        new_snapshot = McpConfigSnapshot()
        mcp_manager.update_session_snapshot(session_id, new_snapshot)
        new_caps = await mcp_manager.get_capabilities(session_id=session_id)

        # ================================================================
        # Phase 7: VERIFY - Fresh resources, no stale references
        # ================================================================

        # 7a: New McpSessionContext is a different object
        assert new_ctx is not old_ctx

        # 7b: New toolset_cache is fresh (different object, empty)
        assert new_ctx.toolset_cache is not old_toolset_cache
        assert len(new_ctx.toolset_cache) == 0

        # 7c: New connection_pool is a different object
        assert new_ctx.connection_pool is not None
        assert new_ctx.connection_pool is not old_connection_pool

        # 7d: Snapshot is the new one, not the old one
        assert new_ctx.snapshot is new_snapshot
        assert new_ctx.snapshot is not snapshot

        # 7e: acp_connection_ids does NOT contain old connection
        assert (old_connection_id, old_session_key) not in new_ctx.acp_connection_ids
        assert (new_connection_id, new_session_key) in new_ctx.acp_connection_ids

        # 7f: AcpMcpConnectionManager has only the new connection
        assert old_connection_id not in acp_manager
        assert new_connection_id in acp_manager
        assert acp_manager.get_connection(new_connection_id) is new_conn

        # 7g: _session_connections has only the new (connection_id, session_key)
        assert session_id in acp_manager._session_connections
        old_entry = (old_connection_id, old_session_key)
        new_entry = (new_connection_id, new_session_key)
        assert old_entry not in acp_manager._session_connections[session_id]
        assert new_entry in acp_manager._session_connections[session_id]

        # 7h: New connection is active, old connection is closed
        assert new_conn.has_active_sessions()
        assert new_conn is not old_conn
        assert new_conn.connection_id == new_connection_id
        assert not old_conn.has_active_sessions()

        # 7i: New capabilities list is returned (empty snapshot → empty list)
        assert new_caps == []

        # 7j: Session context still exists after resume
        assert mcp_manager.get_session_context(session_id) is not None

    finally:
        # Ensure cleanup even on assertion failure
        await mcp_manager.cleanup_session(session_id)
        await mcp_manager.cleanup()
