"""Integration/unit tests for MCP session lifecycle wiring (Part 2).

Categories C + D: state consistency and error-path tests covering the
wiring between ``MCPManager``, ``AcpMcpConnectionManager``,
``ACPSessionManager``, and ``ACPSession`` during cleanup, resume, and
disconnect scenarios.

Category C — State Consistency (4 tests):
    C1: After cleanup_session all registries are consistent.
    C2: close_all_sessions_for_connection only removes target sessions.
    C3: resume_session cleans old connection and registers new.
    C4: AcpMcpConnectionManager.cleanup_session unregisters stream pairs.

Category D — Error Path Tests (5 tests):
    D1: cleanup_session with acp_manager raising still pops session.
    D2: close_all_sessions_for_connection one raises, others still closed.
    D3: ACPSession.close with mcp cleanup raising still exits acp_env.
    D4: resume_session with old close raising still creates new session.
    D5: cleanup_session with connection pool raising still pops session.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from acp.schema.mcp import AcpMcpServer
from wolfharness import Agent
from wolfharness.delegation import AgentPool
from wolfharness.mcp_server.manager import MCPManager
from wolfharness.models.agents import NativeAgentConfig
from wolfharness.models.manifest import AgentsManifest
from wolfharness.sessions import SessionData
from wolfharness_server.acp_server.acp_agent import AgentPoolACPAgent
from wolfharness_server.acp_server.acp_mcp_manager import AcpMcpConnectionManager
from wolfharness_server.acp_server.session import ACPSession
from wolfharness_server.acp_server.session_manager import ACPSessionManager


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _FakeTransport:
    """Fake ``ClientTransport`` for testing ``add_acp_transport``."""

    def __init__(self, label: str = "fake-acp") -> None:
        self.label = label

    @asynccontextmanager
    async def connect_session(self):
        """Fake ``connect_session`` that yields immediately."""
        yield


def _make_wired_managers() -> tuple[MCPManager, AcpMcpConnectionManager]:
    """Create wired MCPManager + AcpMcpConnectionManager for testing.

    Returns a tuple ``(mcp_manager, acp_manager)`` where
    ``mcp_manager._acp_mcp_manager`` is set to ``acp_manager``, simulating
    the production wiring done in ``ACPSession.__post_init__``.
    """
    acp_manager = AcpMcpConnectionManager()
    mcp_manager = MCPManager(name="test")
    mcp_manager._acp_mcp_manager = acp_manager
    return mcp_manager, acp_manager


def _make_mock_pool() -> MagicMock:
    """Create a mock AgentPool suitable for ACPSessionManager tests.

    Returns a MagicMock with ``session_pool.sessions`` configured as a
    SessionController mock, including ``store``, ``close_session``, and
    ``get_or_create_session_agent``.
    """
    mock_pool: MagicMock = MagicMock()
    mock_pool.manifest.agents = {"test_agent": MagicMock()}
    mock_sessions: MagicMock = MagicMock()
    mock_pool.session_pool.sessions = mock_sessions
    mock_sessions.close_session = AsyncMock()
    mock_pool.session_pool.close_session = AsyncMock()
    mock_pool.session_pool._get_resume_lock = AsyncMock(return_value=__import__("asyncio").Lock())
    return mock_pool


def _make_mock_session(session_id: str) -> AsyncMock:
    """Create a mock ACPSession with async close().

    Args:
        session_id: The session ID to assign to the mock session.

    Returns:
        An AsyncMock configured with ``session_id`` and an async ``close()``.
    """
    session: AsyncMock = AsyncMock()
    session.session_id = session_id
    session.close = AsyncMock()
    return session


# ============================================================================
# Category C: State Consistency
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_after_cleanup_all_registries_consistent() -> None:
    """After cleanup_session all registries across both managers are consistent.

    Creates a session with ACP transport and session connection, then
    calls ``cleanup_session()`` on the MCPManager. Verifies that the
    the session context is removed (``get_session_context()`` returns None),
    ``_session_connections``,
    and the ACP connection is removed from ``_connections`` when no other
    sessions reference it.
    """
    session_id = "c1-consistency"
    server_config = AcpMcpServer(name="c1-server", id="c1-server-id")
    send_to_client = AsyncMock(return_value=None)
    connection_id = "conn-c1"

    mcp_manager, acp_manager = _make_wired_managers()

    try:
        # Create ACP connection and register session connection
        conn = await acp_manager.create_connection(
            connection_id,
            server_config,
            send_to_client,
        )
        _pair, session_key = conn.register_session()
        acp_manager.register_session_connection(
            session_id,
            connection_id,
            session_key,
        )

        # Create MCP session context and add ACP transport
        mcp_manager.get_or_create_session(session_id)
        transport = _FakeTransport("c1-transport")
        await mcp_manager.add_acp_transport(
            session_id,
            client_id="c1-client",
            transport=transport,
            connection_id=connection_id,
            session_key=session_key,
        )

        # Verify pre-cleanup state
        assert mcp_manager.get_session_context(session_id) is not None
        assert session_id in acp_manager._session_connections
        assert connection_id in acp_manager._connections
        assert conn.has_active_sessions()

        # Cleanup via MCPManager (delegates to AcpMcpConnectionManager)
        await mcp_manager.cleanup_session(session_id)

        # Assert: session removed from MCPManager
        assert mcp_manager.get_session_context(session_id) is None

        # Assert: session removed from AcpMcpConnectionManager reverse index
        assert session_id not in acp_manager._session_connections

        # Assert: connection removed from _connections (no other sessions)
        assert connection_id not in acp_manager._connections
        assert not conn.has_active_sessions()
    finally:
        await mcp_manager.cleanup()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_after_close_all_sessions_for_connection_only_target_removed() -> None:
    """close_all_sessions_for_connection removes only target connection's sessions.

    Populates two connections (conn-1 with s1/s2, conn-2 with s3), then
    calls ``close_all_sessions_for_connection("conn-1")``. Verifies that
    conn-1's sessions are removed while conn-2's session survives.
    """
    mock_pool: MagicMock = _make_mock_pool()
    manager: ACPSessionManager = ACPSessionManager(mock_pool)

    # Populate _connection_sessions
    manager._connection_sessions["conn-1"] = {"s1", "s2"}
    manager._connection_sessions["conn-2"] = {"s3"}

    # Populate _acp_sessions with mock sessions
    session_s1: AsyncMock = _make_mock_session("s1")
    session_s2: AsyncMock = _make_mock_session("s2")
    session_s3: AsyncMock = _make_mock_session("s3")
    manager._acp_sessions["s1"] = session_s1
    manager._acp_sessions["s2"] = session_s2
    manager._acp_sessions["s3"] = session_s3

    # Call close_all_sessions_for_connection for conn-1 only
    await manager.close_all_sessions_for_connection("conn-1")

    # Assert: conn-1 removed, conn-2 preserved with s3
    assert "conn-1" not in manager._connection_sessions
    assert "conn-2" in manager._connection_sessions
    assert manager._connection_sessions["conn-2"] == {"s3"}

    # Assert: s1 and s2 removed, s3 still present
    assert "s1" not in manager._acp_sessions
    assert "s2" not in manager._acp_sessions
    assert "s3" in manager._acp_sessions
    assert manager._acp_sessions["s3"] is session_s3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_after_resume_session_old_connection_cleaned_new_registered() -> None:
    """resume_session removes session from old connection and registers in new.

    Pre-populates ``_connection_sessions["conn-1"] = {"session-1"}`` and
    ``_acp_sessions["session-1"]`` with a mock old session, then calls
    ``resume_session("session-1", ..., connection_id="conn-2")``.
    Verifies the session_id is removed from conn-1 and added to conn-2.
    """
    session_id = "session-1"

    mock_pool: MagicMock = _make_mock_pool()

    # Mock session_store.load to return session data
    mock_store: AsyncMock = AsyncMock()
    mock_store.load_session = AsyncMock(
        return_value=SessionData(
            session_id=session_id,
            agent_name="test_agent",
            cwd="/tmp",
        )
    )
    mock_pool.session_pool.sessions.store = mock_store

    # Mock get_or_create_session_agent
    mock_agent: MagicMock = MagicMock()
    mock_agent.name = "test_agent"

    async def mock_get_agent(sid: str, agent_name: str | None = None) -> MagicMock:
        return mock_agent

    mock_pool.session_pool.sessions.get_or_create_session_agent = mock_get_agent

    # Create ACPSessionManager
    acp_manager = ACPSessionManager(mock_pool)

    # Pre-populate old connection tracking and old session
    acp_manager._connection_sessions["conn-1"] = {session_id}

    old_session: AsyncMock = AsyncMock()
    old_session.session_id = session_id
    old_session.close = AsyncMock()
    acp_manager._acp_sessions[session_id] = old_session

    # Patch ACPSession constructor
    mock_new_session: AsyncMock = AsyncMock()
    mock_new_session.session_id = session_id
    mock_new_session.initialize = AsyncMock()
    mock_new_session.initialize_mcp_servers = AsyncMock()
    mock_new_session.register_update_callback = MagicMock()

    with patch(
        "wolfharness_server.acp_server.session_manager.ACPSession",
        return_value=mock_new_session,
    ):
        result = await acp_manager.resume_session(
            session_id=session_id,
            client=MagicMock(),
            acp_agent=MagicMock(),
            connection_id="conn-2",
        )

    assert result is mock_new_session

    # Assert: session_id removed from old connection
    old_conn_sessions = acp_manager._connection_sessions.get("conn-1", set())
    assert session_id not in old_conn_sessions

    # Assert: session_id registered in new connection
    new_conn_sessions = acp_manager._connection_sessions.get("conn-2", set())
    assert session_id in new_conn_sessions

    # Assert: old session replaced by new in _acp_sessions
    assert acp_manager._acp_sessions[session_id] is mock_new_session
    assert acp_manager._acp_sessions[session_id] is not old_session


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_mcp_manager_cleanup_unregisters_stream_pairs() -> None:
    """AcpMcpConnectionManager.cleanup_session unregisters per-session stream pairs.

    Creates a connection, registers a session to get a stream pair + key,
    registers the session connection for reverse-index tracking, then
    calls ``cleanup_session()``. Verifies the session key is removed from
    the connection's ``_session_streams``, ``has_active_sessions()``
    returns False, and the connection is removed from ``_connections``.
    """
    server_config = AcpMcpServer(name="c4-server", id="c4-server-id")
    send_to_client = AsyncMock(return_value=None)
    connection_id = "conn-c4"
    session_id = "c4-session"

    acp_manager = AcpMcpConnectionManager()

    # Create connection and register a session
    conn = await acp_manager.create_connection(
        connection_id,
        server_config,
        send_to_client,
    )
    _pair, session_key = conn.register_session()
    acp_manager.register_session_connection(
        session_id,
        connection_id,
        session_key,
    )

    # Verify pre-cleanup state
    assert session_key in conn._session_streams
    assert conn.has_active_sessions()
    assert connection_id in acp_manager._connections

    # Cleanup
    await acp_manager.cleanup_session(session_id)

    # Assert: session key removed from connection's _session_streams
    assert session_key not in conn._session_streams

    # Assert: no active sessions remain
    assert not conn.has_active_sessions()

    # Assert: connection removed from _connections
    assert connection_id not in acp_manager._connections


# ============================================================================
# Category D: Error Path Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cleanup_session_acp_manager_raises_session_still_popped() -> None:
    """cleanup_session catches ACP manager errors and still pops the session.

    Wires a mock AcpMcpConnectionManager whose ``cleanup_session`` raises
    ``RuntimeError``. Creates a session context, then calls
    ``cleanup_session()``. The session must still be removed from
    ``get_session_context()`` returns None and no exception should propagate.
    """
    manager = MCPManager(name="test")
    try:
        # Wire a mock ACP manager that raises on cleanup
        mock_acp_manager: AsyncMock = AsyncMock()
        mock_acp_manager.cleanup_session = AsyncMock(
            side_effect=RuntimeError("ACP cleanup failed"),
        )
        manager._acp_mcp_manager = mock_acp_manager

        # Create session context
        ctx = manager.get_or_create_session("d1-session")
        original_toolset_cache = ctx.toolset_cache

        # Populate toolset_cache to verify it was cleared before the error
        original_toolset_cache["fake-toolset"] = MagicMock()

        # Cleanup should NOT propagate the RuntimeError
        await manager.cleanup_session("d1-session")

        # Assert: session removed despite ACP manager error
        assert manager.get_session_context("d1-session") is None

        # Assert: ACP manager cleanup was called
        mock_acp_manager.cleanup_session.assert_called_once_with("d1-session")

        # Assert: toolset_cache was cleared before the error
        assert len(original_toolset_cache) == 0
    finally:
        await manager.cleanup()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_close_all_sessions_for_connection_one_raises_others_still_closed() -> None:
    """close_all_sessions_for_connection continues when one session.close() raises.

    Creates two sessions on the same connection. The first session's
    ``close()`` raises ``RuntimeError``. The second closes normally.
    Verifies both are removed from ``_acp_sessions``, the connection is
    removed from ``_connection_sessions``, and ``controller.close_session``
    is called for both.
    """
    mock_pool: MagicMock = _make_mock_pool()
    manager: ACPSessionManager = ACPSessionManager(mock_pool)

    session_id_1 = "d2-s1"
    session_id_2 = "d2-s2"
    connection_id = "conn-d2"

    # First session raises on close, second is normal
    session_1: AsyncMock = _make_mock_session(session_id_1)
    session_1.close = AsyncMock(side_effect=RuntimeError("close failed"))
    session_2: AsyncMock = _make_mock_session(session_id_2)

    manager._acp_sessions[session_id_1] = session_1
    manager._acp_sessions[session_id_2] = session_2
    manager._connection_sessions[connection_id] = {session_id_1, session_id_2}

    # Call close_all_sessions_for_connection
    await manager.close_all_sessions_for_connection(connection_id)

    # Assert: both sessions' close() was called (first raised, second succeeded)
    session_1.close.assert_awaited_once()
    session_2.close.assert_awaited_once()

    # Assert: both removed from _acp_sessions
    assert session_id_1 not in manager._acp_sessions
    assert session_id_2 not in manager._acp_sessions

    # Assert: connection removed from _connection_sessions
    assert connection_id not in manager._connection_sessions

    # Assert: controller.close_session called for both
    assert mock_pool.session_pool.close_session.await_count == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_session_close_mcp_cleanup_raises_acp_env_still_exits() -> None:
    """ACPSession.close() catches MCP cleanup errors and still exits acp_env.

    Creates a real ACPSession using the ``Agent.from_callback`` pattern,
    mocks ``agent.mcp.cleanup_session`` to raise ``RuntimeError``, and
    mocks ``acp_env.__aexit__``. Calls ``session.close()`` and verifies
    that ``acp_env.__aexit__`` is still called despite the MCP cleanup
    error, and no exception propagates from ``close()``.
    """
    session_id = "d3-session"

    def simple_callback(message: str) -> str:
        return f"Test: {message}"

    manifest = AgentsManifest(agents={"test_agent": NativeAgentConfig(model="test")})
    pool = AgentPool(manifest)
    agent = Agent.from_callback(
        name="test_agent",
        callback=simple_callback,
        agent_pool=pool,
    )
    acp_agent = AgentPoolACPAgent(client=MagicMock(), default_agent=agent)

    # Create ACPSession
    session = ACPSession(
        session_id=session_id,
        agent=agent,
        cwd="/tmp",
        client=MagicMock(),
        acp_agent=acp_agent,
    )

    # Mock agent.mcp.cleanup_session to raise
    agent.mcp.cleanup_session = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("MCP cleanup failed"),
    )

    # Mock acp_env.__aexit__
    session.acp_env.__aexit__ = AsyncMock(  # type: ignore[method-assign]
        return_value=None,
    )

    # close() should not propagate the RuntimeError
    await session.close()

    # Assert: agent.mcp.cleanup_session was called (and raised, but caught)
    agent.mcp.cleanup_session.assert_awaited_once_with(session_id)

    # Assert: acp_env.__aexit__ was called despite the MCP cleanup error
    session.acp_env.__aexit__.assert_awaited_once()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resume_session_old_close_raises_new_still_created() -> None:
    """resume_session creates new session even if old session close raises.

    Pre-populates an old session whose ``close()`` raises ``RuntimeError``.
    Calls ``resume_session()`` with a new connection_id. Verifies the old
    session's ``close()`` was called (raised, caught), the new session is
    created and present in ``_acp_sessions``, and the new session is
    registered in ``_connection_sessions["conn-2"]``.
    """
    session_id = "d4-session"

    mock_pool: MagicMock = _make_mock_pool()

    # Mock session_store.load
    mock_store: AsyncMock = AsyncMock()
    mock_store.load_session = AsyncMock(
        return_value=SessionData(
            session_id=session_id,
            agent_name="test_agent",
            cwd="/tmp",
        )
    )
    mock_pool.session_pool.sessions.store = mock_store

    # Mock get_or_create_session_agent
    mock_agent: MagicMock = MagicMock()
    mock_agent.name = "test_agent"

    async def mock_get_agent(sid: str, agent_name: str | None = None) -> MagicMock:
        return mock_agent

    mock_pool.session_pool.sessions.get_or_create_session_agent = mock_get_agent

    # Create ACPSessionManager
    acp_manager = ACPSessionManager(mock_pool)

    # Pre-populate old session that raises on close
    old_session: AsyncMock = AsyncMock()
    old_session.session_id = session_id
    old_session.close = AsyncMock(side_effect=RuntimeError("old close failed"))
    acp_manager._acp_sessions[session_id] = old_session
    acp_manager._connection_sessions["conn-1"] = {session_id}

    # Patch ACPSession to return mock new session
    mock_new_session: AsyncMock = AsyncMock()
    mock_new_session.session_id = session_id
    mock_new_session.initialize = AsyncMock()
    mock_new_session.initialize_mcp_servers = AsyncMock()
    mock_new_session.register_update_callback = MagicMock()

    with patch(
        "wolfharness_server.acp_server.session_manager.ACPSession",
        return_value=mock_new_session,
    ):
        result = await acp_manager.resume_session(
            session_id=session_id,
            client=MagicMock(),
            acp_agent=MagicMock(),
            connection_id="conn-2",
        )

    # Assert: old session's close() was called (raised, caught)
    old_session.close.assert_awaited_once()

    # Assert: new session created and in _acp_sessions
    assert result is not None
    assert result is mock_new_session
    assert acp_manager._acp_sessions[session_id] is mock_new_session

    # Assert: new session registered in _connection_sessions["conn-2"]
    new_conn_sessions = acp_manager._connection_sessions.get("conn-2", set())
    assert session_id in new_conn_sessions


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cleanup_session_connection_pool_raises_session_still_popped() -> None:
    """cleanup_session catches connection pool errors and still pops the session.

    Creates a real MCPManager with a mock connection pool whose
    ``cleanup()`` raises ``RuntimeError``, wires a mock ACP manager,
    creates a session, and calls ``cleanup_session()``. The session must
    still be removed (``get_session_context()`` returns None) and the ACP manager
    cleanup must still be called despite the pool error.
    """
    manager = MCPManager(name="test")
    try:
        # Wire a mock ACP manager
        mock_acp_manager: AsyncMock = AsyncMock()
        manager._acp_mcp_manager = mock_acp_manager

        # Create session context
        ctx = manager.get_or_create_session("d5-session")

        # Replace connection_pool with a mock that raises on cleanup
        mock_pool_obj: AsyncMock = AsyncMock()
        mock_pool_obj.cleanup = AsyncMock(side_effect=RuntimeError("pool cleanup failed"))
        ctx.connection_pool = mock_pool_obj  # type: ignore[method-assign]

        # Cleanup should NOT propagate the RuntimeError
        await manager.cleanup_session("d5-session")

        # Assert: session removed despite pool error
        assert manager.get_session_context("d5-session") is None

        # Assert: ACP manager cleanup still called despite pool error
        mock_acp_manager.cleanup_session.assert_called_once_with("d5-session")
    finally:
        await manager.cleanup()
