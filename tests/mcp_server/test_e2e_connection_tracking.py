"""End-to-end integration tests for connection_id tracking.

Verifies the full cleanup chain from WebSocket disconnect through
ACPSessionManager, SessionController, MCPManager, and AcpMcpConnectionManager
using real component instances (not manually injected registries).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from acp.schema.mcp import AcpMcpServer
from wolfharness.mcp_server.config_snapshot import McpConfigSnapshot
from wolfharness.mcp_server.manager import MCPManager
from wolfharness.orchestrator.session_controller import SessionController, SessionState
from wolfharness_server.acp_server.acp_mcp_manager import AcpMcpConnectionManager
from wolfharness_server.acp_server.session_manager import ACPSessionManager


# ============================================================================
# G7: Full on_disconnect cleanup chain with real components
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_on_disconnect_full_cleanup_chain_with_real_components() -> None:  # noqa: PLR0915
    """Full on_disconnect chain empties all registries.

    Verifies the complete cleanup path:
    ``close_all_sessions_for_connection()``
    -> ``SessionController.close_session()`` (pops _sessions, _session_agents,
       calls ``agent.mcp.cleanup_session()``)
    -> ``ACPSession.close()`` (calls ``cleanup_session()`` idempotently)
    -> ``MCPManager.cleanup_session()`` (pops _session_contexts, delegates to
       ``AcpMcpConnectionManager.cleanup_session()``)

    After cleanup, ALL registries must be empty:
    _session_contexts, _session_connections, _connections,
    _acp_sessions, _connection_sessions, _sessions, _session_agents.
    """
    # --- Real MCP + ACP managers ---
    mcp_manager = MCPManager(name="test-g7")
    acp_mcp_manager = AcpMcpConnectionManager()
    mcp_manager._acp_mcp_manager = acp_mcp_manager

    # --- Mock pool with real SessionController ---
    mock_pool: MagicMock = MagicMock()
    mock_pool.manifest.agents = {"test_agent": MagicMock()}

    session_controller = SessionController(pool=mock_pool, store=None)
    mock_pool.session_pool = MagicMock()
    mock_pool.session_pool.sessions = session_controller
    mock_pool.session_pool.create_session = AsyncMock()

    async def _close_via_controller(sid: str) -> None:
        await session_controller.close_session(sid)

    mock_pool.session_pool.close_session = _close_via_controller
    mock_pool.storage.generate_session_id = Mock(return_value="g7-session-1")

    # --- Mock agent with .mcp = real MCPManager ---
    mock_agent: MagicMock = MagicMock()
    mock_agent.name = "test_agent"
    mock_agent.mcp = mcp_manager

    async def mock_get_agent(sid: str, agent_name: str | None = None) -> MagicMock:
        return mock_agent

    session_controller.get_or_create_session_agent = mock_get_agent  # type: ignore[method-assign]

    # --- Real ACPSessionManager ---
    acp_manager = ACPSessionManager(mock_pool)

    # --- Patch ACPSession to avoid heavy __post_init__ ---
    # mock_session.close() calls mcp_manager.cleanup_session (simulates T15)
    async def mock_session_close() -> None:
        await mcp_manager.cleanup_session("g7-session-1")

    mock_session: AsyncMock = AsyncMock()
    mock_session.session_id = "g7-session-1"
    mock_session.initialize = AsyncMock()
    mock_session.initialize_mcp_servers = AsyncMock()
    mock_session.register_update_callback = MagicMock()
    mock_session.close = mock_session_close

    try:
        with patch(
            "wolfharness_server.acp_server.session_manager.ACPSession",
            return_value=mock_session,
        ):
            session_id = await acp_manager.create_session(
                agent_name="test_agent",
                cwd="/tmp",
                client=MagicMock(),
                acp_agent=MagicMock(),
                connection_id="conn-1",
            )

        # --- Populate MCP + ACP resources for the session ---
        mcp_manager.get_or_create_session(session_id)

        server_config = AcpMcpServer(name="test-server", id="test-id")
        send_to_client: AsyncMock = AsyncMock(return_value=None)
        conn = await acp_mcp_manager.create_connection(
            "acp-conn-1",
            server_config,
            send_to_client,
        )
        _pair, session_key = conn.register_session()
        acp_mcp_manager.register_session_connection(
            session_id,
            "acp-conn-1",
            session_key,
        )

        # --- Populate SessionController registries ---
        session_state = SessionState(
            session_id=session_id,
            agent_name="test_agent",
        )
        session_controller._sessions[session_id] = session_state
        session_controller._session_agents[session_id] = mock_agent

        # --- Verify pre-state ---
        assert session_id in acp_manager._acp_sessions
        assert "conn-1" in acp_manager._connection_sessions
        assert session_id in acp_manager._connection_sessions["conn-1"]
        assert session_id in mcp_manager._session_contexts
        assert session_id in acp_mcp_manager._session_connections
        assert "acp-conn-1" in acp_mcp_manager._connections
        assert session_id in session_controller._sessions
        assert session_id in session_controller._session_agents

        # --- Trigger disconnect cleanup ---
        await acp_manager.close_all_sessions_for_connection("conn-1")

        # --- Assert all registries empty ---
        assert len(acp_manager._acp_sessions) == 0
        assert "conn-1" not in acp_manager._connection_sessions
        assert len(mcp_manager._session_contexts) == 0
        assert len(acp_mcp_manager._session_connections) == 0
        assert len(acp_mcp_manager._connections) == 0
        assert session_id not in session_controller._sessions
        assert session_id not in session_controller._session_agents
    finally:
        await mcp_manager.cleanup()


# ============================================================================
# G8: create_session(connection_id=...) populates _connection_sessions
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_connection_id_propagation_create_session_populates_connection_sessions() -> None:
    """create_session(connection_id=...) naturally populates _connection_sessions.

    This is the producer-side test: calling ``create_session(connection_id="conn-1")``
    must add the session_id to ``_connection_sessions["conn-1"]`` without any
    manual injection. This would have caught the original ``connection_id``
    propagation bug where ``_get_connection_id()`` was not wired.
    """
    # --- Mock pool ---
    mock_pool: MagicMock = MagicMock()
    mock_pool.manifest.agents = {"test_agent": MagicMock()}

    mock_sessions: MagicMock = MagicMock()
    mock_pool.session_pool.sessions = mock_sessions
    mock_pool.session_pool.create_session = AsyncMock()
    mock_pool.session_pool.close_session = AsyncMock()
    mock_pool.session_pool._get_resume_lock = AsyncMock(return_value=__import__("asyncio").Lock())
    mock_sessions.close_session = AsyncMock()

    # Mock session_store (returns None for load, save is no-op)
    mock_store: AsyncMock = AsyncMock()
    mock_store.load_session = AsyncMock(return_value=None)
    mock_store.save_session = AsyncMock()
    mock_sessions.store = mock_store

    mock_pool.storage.generate_session_id = Mock(return_value="g8-session-1")

    # Mock get_or_create_session_agent
    mock_agent: MagicMock = MagicMock()
    mock_agent.name = "test_agent"

    async def mock_get_agent(sid: str, agent_name: str | None = None) -> MagicMock:
        return mock_agent

    mock_sessions.get_or_create_session_agent = mock_get_agent

    # --- ACPSessionManager ---
    acp_manager = ACPSessionManager(mock_pool)

    # --- Patch ACPSession constructor ---
    mock_session: AsyncMock = AsyncMock()
    mock_session.session_id = "g8-session-1"
    mock_session.initialize = AsyncMock()
    mock_session.initialize_mcp_servers = AsyncMock()
    mock_session.register_update_callback = MagicMock()

    with patch(
        "wolfharness_server.acp_server.session_manager.ACPSession",
        return_value=mock_session,
    ):
        session_id = await acp_manager.create_session(
            agent_name="test_agent",
            cwd="/tmp",
            client=MagicMock(),
            acp_agent=MagicMock(),
            connection_id="conn-1",
        )

    # --- Assert _connection_sessions was populated naturally ---
    # Session ID is now generated via identifiers.generate_session_id() (Phase 4)
    # which produces sortable IDs with 'ses_' prefix.
    assert session_id.startswith("ses_")
    assert "conn-1" in acp_manager._connection_sessions
    assert session_id in acp_manager._connection_sessions["conn-1"]
    assert session_id in acp_manager._acp_sessions


# ============================================================================
# G9: get_capabilities() during concurrent cleanup_session()
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_capabilities_during_concurrent_cleanup() -> None:
    """get_capabilities() called concurrently with cleanup_session() must not crash.

    Verifies the GAP-11 race condition: ``get_capabilities(session_id)`` uses
    ``self._session_contexts.get(session_id)`` which returns None if
    ``cleanup_session()`` has already popped the context. The method must
    handle this gracefully by falling back to global-only capabilities.

    Either:
    - get_capabilities runs first: returns capabilities from snapshot (empty)
    - cleanup runs first: get_capabilities gets None, falls back to global-only
    Both outcomes are acceptable; no exception should be raised.
    """
    manager = MCPManager(name="test-g9")
    try:
        session_id = "test-concurrent-g9"

        # Populate session context with an empty snapshot
        manager.get_or_create_session(session_id)
        manager.update_session_snapshot(session_id, McpConfigSnapshot())

        assert session_id in manager._session_contexts

        # Fire both concurrently
        results: list[object] = await asyncio.gather(
            manager.get_capabilities(session_id=session_id),
            manager.cleanup_session(session_id),
            return_exceptions=True,
        )

        # Neither should have raised
        for result in results:
            assert not isinstance(result, Exception), (
                f"Concurrent get_capabilities/cleanup raised: {result!r}"
            )

        # Session must be removed from _session_contexts after both complete
        assert session_id not in manager._session_contexts
    finally:
        await manager.cleanup()


# ============================================================================
# G11: Multiple sessions on same connection_id, closing one preserves the other
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiple_sessions_same_connection_real_acpsessions() -> None:  # noqa: PLR0915
    """Two sessions on same connection_id both tracked; closing one preserves other.

    Creates two sessions via ``create_session(connection_id="conn-1")`` and
    verifies both session_ids are in ``_connection_sessions["conn-1"]``.
    Closing one session does not affect the other.
    """
    # --- Mock pool ---
    mock_pool: MagicMock = MagicMock()
    mock_pool.manifest.agents = {"test_agent": MagicMock()}

    mock_sessions: MagicMock = MagicMock()
    mock_pool.session_pool.sessions = mock_sessions
    mock_pool.session_pool.create_session = AsyncMock()
    mock_pool.session_pool.close_session = AsyncMock()
    mock_pool.session_pool._get_resume_lock = AsyncMock(return_value=__import__("asyncio").Lock())
    mock_sessions.close_session = AsyncMock()

    mock_store: AsyncMock = AsyncMock()
    mock_store.load_session = AsyncMock(return_value=None)
    mock_store.save_session = AsyncMock()
    mock_sessions.store = mock_store

    # Generate unique IDs for each session
    id_counter = 0

    def mock_generate_id() -> str:
        nonlocal id_counter
        id_counter += 1
        return f"g11-session-{id_counter}"

    mock_pool.storage.generate_session_id = mock_generate_id

    mock_agent: MagicMock = MagicMock()
    mock_agent.name = "test_agent"

    async def mock_get_agent(sid: str, agent_name: str | None = None) -> MagicMock:
        return mock_agent

    mock_sessions.get_or_create_session_agent = mock_get_agent

    # --- ACPSessionManager ---
    acp_manager = ACPSessionManager(mock_pool)

    # --- Patch ACPSession constructor with side_effect for two sessions ---
    mock_session_1: AsyncMock = AsyncMock()
    mock_session_1.session_id = "g11-session-1"
    mock_session_1.initialize = AsyncMock()
    mock_session_1.initialize_mcp_servers = AsyncMock()
    mock_session_1.register_update_callback = MagicMock()
    mock_session_1.close = AsyncMock()

    mock_session_2: AsyncMock = AsyncMock()
    mock_session_2.session_id = "g11-session-2"
    mock_session_2.initialize = AsyncMock()
    mock_session_2.initialize_mcp_servers = AsyncMock()
    mock_session_2.register_update_callback = MagicMock()
    mock_session_2.close = AsyncMock()

    with patch(
        "wolfharness_server.acp_server.session_manager.ACPSession",
        side_effect=[mock_session_1, mock_session_2],
    ):
        session_id_1 = await acp_manager.create_session(
            agent_name="test_agent",
            cwd="/tmp",
            client=MagicMock(),
            acp_agent=MagicMock(),
            connection_id="conn-1",
        )
        session_id_2 = await acp_manager.create_session(
            agent_name="test_agent",
            cwd="/tmp",
            client=MagicMock(),
            acp_agent=MagicMock(),
            connection_id="conn-1",
        )

    # --- Both sessions in _connection_sessions["conn-1"] ---
    assert "conn-1" in acp_manager._connection_sessions
    conn_sessions = acp_manager._connection_sessions["conn-1"]
    assert session_id_1 in conn_sessions
    assert session_id_2 in conn_sessions
    assert len(conn_sessions) == 2

    # Both in _acp_sessions
    assert session_id_1 in acp_manager._acp_sessions
    assert session_id_2 in acp_manager._acp_sessions

    # --- Close session 1 only ---
    await acp_manager.close_session(session_id_1)

    # Session 1 is gone
    assert session_id_1 not in acp_manager._acp_sessions

    # Session 2 survives
    assert session_id_2 in acp_manager._acp_sessions
    assert acp_manager._acp_sessions[session_id_2] is mock_session_2

    # Session 1 removed from _connection_sessions["conn-1"]
    # (close_session does NOT touch _connection_sessions — only
    # close_all_sessions_for_connection does. So both IDs remain.)
    # But session 2 must still be tracked.
    assert "conn-1" in acp_manager._connection_sessions
    assert session_id_2 in acp_manager._connection_sessions["conn-1"]

    # --- Clean up remaining session ---
    await acp_manager.close_session(session_id_2)
    assert session_id_2 not in acp_manager._acp_sessions
