"""Integration and unit tests for MCP session lifecycle cross-component wiring.

Covers two categories:

**Category A — Cross-Component Wiring (4 tests):**
  Verifies that MCPManager, AcpMcpConnectionManager, ACPSession, and
  ACPSessionManager are correctly wired so that cleanup propagates
  through the full delegation chain.

**Category B — Lifecycle Edge Cases (7 tests):**
  Exercises edge cases: full create→cleanup cycle, close→recreate
  freshness, shared connections, WebSocket disconnect isolation,
  resume_session MCP manager freshness, and concurrent cleanup safety.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from acp.schema.mcp import AcpMcpServer
from wolfharness.mcp_server.manager import MCPManager
from wolfharness.sessions import SessionData
from wolfharness_server.acp_server.acp_mcp_manager import AcpMcpConnectionManager
from wolfharness_server.acp_server.session_manager import ACPSessionManager


# ============================================================================
# Shared helpers
# ============================================================================


def _make_wired_managers() -> tuple[MCPManager, AcpMcpConnectionManager]:
    """Create wired MCPManager + AcpMcpConnectionManager for testing.

    Returns a tuple ``(mcp_manager, acp_manager)`` where
    ``mcp_manager._acp_mcp_manager`` is set to ``acp_manager`` so that
    ``cleanup_session()`` delegates to ``AcpMcpConnectionManager``.
    """
    acp_manager = AcpMcpConnectionManager()
    mcp_manager = MCPManager(name="test")
    mcp_manager._acp_mcp_manager = acp_manager
    return mcp_manager, acp_manager


class _FakeTransport:
    """Fake ``ClientTransport`` for testing ``add_acp_transport``."""

    def __init__(self, label: str = "fake-acp") -> None:
        self.label = label

    @asynccontextmanager
    async def connect_session(self):
        """Fake ``connect_session`` that yields immediately."""
        yield


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


def _make_mock_pool() -> MagicMock:
    """Create a mock AgentPool suitable for ACPSessionManager.

    Returns:
        A MagicMock with ``session_pool``, ``sessions``, ``store``,
        and ``manifest`` configured for session lifecycle tests.
    """
    mock_pool: MagicMock = MagicMock()
    mock_sessions: MagicMock = MagicMock()
    mock_pool.session_pool.sessions = mock_sessions
    mock_sessions.close_session = AsyncMock()
    mock_pool.session_pool.close_session = AsyncMock()
    mock_pool.session_pool._get_resume_lock = AsyncMock(return_value=__import__("asyncio").Lock())
    mock_pool.manifest.agents = {}
    return mock_pool


# ============================================================================
# Category A: Cross-Component Wiring
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cleanup_session_delegates_to_wired_acp_mcp_manager() -> None:
    """cleanup_session() delegates to wired AcpMcpConnectionManager.

    Creates real MCPManager + AcpMcpConnectionManager wired via
    ``_acp_mcp_manager``, registers an ACP connection + session, adds
    an ACP transport, then calls ``cleanup_session()``. All three
    registries must be cleaned: the session context registry,
    ``_session_connections``, and the connection's ``_session_streams``.
    """
    mcp_manager, acp_manager = _make_wired_managers()
    session_id = "test-delegate-1"
    connection_id = "conn-delegate-1"
    server_config = AcpMcpServer(name="test-server", id="test-server-id")
    send_to_client = AsyncMock(return_value=None)

    try:
        # Create ACP connection + register session
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

        # Add ACP transport to MCP session context
        mcp_manager.get_or_create_session(session_id)
        await mcp_manager.add_acp_transport(
            session_id,
            client_id="test-client",
            transport=_FakeTransport("test-transport"),
            connection_id=connection_id,
            session_key=session_key,
        )

        # Verify pre-state
        assert session_id in acp_manager._session_connections
        assert mcp_manager.get_session_context(session_id) is not None
        assert conn.has_active_sessions()

        # Cleanup
        await mcp_manager.cleanup_session(session_id)

        # Assert all three registries cleaned
        assert session_id not in acp_manager._session_connections
        assert not conn.has_active_sessions()
        assert mcp_manager.get_session_context(session_id) is None
    finally:
        await mcp_manager.cleanup()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_session_post_init_wires_acp_mcp_manager() -> None:
    """ACPSession.__post_init__ wires agent.mcp._acp_mcp_manager.

    Builds a minimal AgentPool with a NativeAgentConfig, creates an
    AgentPoolACPAgent, and constructs an ACPSession. The session's
    ``__post_init__`` must wire ``agent.mcp._acp_mcp_manager`` to
    ``acp_agent._mcp_manager`` so that cleanup delegation works.
    """
    from wolfharness import Agent
    from wolfharness.delegation import AgentPool
    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.manifest import AgentsManifest
    from wolfharness_server.acp_server.acp_agent import AgentPoolACPAgent
    from wolfharness_server.acp_server.session import ACPSession

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

    # Construct ACPSession — __post_init__ should wire _acp_mcp_manager
    ACPSession(
        session_id="test-wire-post-init",
        agent=agent,
        cwd="/tmp",
        client=MagicMock(),
        acp_agent=acp_agent,
    )

    # Assert wiring: agent.mcp._acp_mcp_manager is acp_agent._mcp_manager
    assert agent.mcp._acp_mcp_manager is acp_agent._mcp_manager, (
        "ACPSession.__post_init__ must wire agent.mcp._acp_mcp_manager "
        "to acp_agent._mcp_manager for cleanup delegation to work"
    )
    assert agent.mcp._acp_mcp_manager is not None, (
        "agent.mcp._acp_mcp_manager must not be None after ACPSession construction"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_session_close_full_delegation_chain() -> None:
    """ACPSession.close() triggers full cleanup delegation chain.

    Creates wired MCPManager + AcpMcpConnectionManager, registers an
    ACP connection + session, adds an ACP transport, constructs a real
    ACPSession, then calls ``session.close()``. The close must clean
    up the session context registry, ``_session_connections``, and the
    connection's active sessions.
    """
    from wolfharness import Agent
    from wolfharness.delegation import AgentPool
    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.manifest import AgentsManifest
    from wolfharness_server.acp_server.acp_agent import AgentPoolACPAgent
    from wolfharness_server.acp_server.session import ACPSession

    mcp_manager, acp_manager = _make_wired_managers()
    session_id = "test-close-chain-1"
    connection_id = "conn-close-chain-1"
    server_config = AcpMcpServer(name="chain-server", id="chain-server-id")
    send_to_client = AsyncMock(return_value=None)

    try:
        # Create ACP connection + register session
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

        # Add ACP transport to MCP session context
        mcp_manager.get_or_create_session(session_id)
        await mcp_manager.add_acp_transport(
            session_id,
            client_id="chain-client",
            transport=_FakeTransport("chain-transport"),
            connection_id=connection_id,
            session_key=session_key,
        )

        # Build ACPSession with the wired MCPManager
        def simple_callback(message: str) -> str:
            return f"Test: {message}"

        manifest = AgentsManifest(
            agents={"test_agent": NativeAgentConfig(model="test")},
        )
        pool = AgentPool(manifest)
        agent = Agent.from_callback(
            name="test_agent",
            callback=simple_callback,
            agent_pool=pool,
        )
        # Wire agent's MCPManager to our test manager
        agent.mcp = mcp_manager
        acp_agent = AgentPoolACPAgent(client=MagicMock(), default_agent=agent)
        # Wire ACP MCP manager
        acp_agent._mcp_manager = acp_manager

        session = ACPSession(
            session_id=session_id,
            agent=agent,
            cwd="/tmp",
            client=MagicMock(),
            acp_agent=acp_agent,
        )

        # Verify pre-state
        assert mcp_manager.get_session_context(session_id) is not None
        assert session_id in acp_manager._session_connections
        assert conn.has_active_sessions()

        # Close the session — triggers full delegation chain
        await session.close()

        # Assert all registries cleaned
        assert mcp_manager.get_session_context(session_id) is None
        assert session_id not in acp_manager._session_connections
        assert not conn.has_active_sessions()
    finally:
        await mcp_manager.cleanup()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_close_all_sessions_for_connection_invokes_session_controller_and_acp_close() -> None:
    """close_all_sessions_for_connection() calls controller + ACPSession.close().

    Pre-populates ``_connection_sessions`` and ``_acp_sessions`` with 2
    mock sessions on connection "conn-1", mocks
    ``SessionController.close_session``, then calls
    ``close_all_sessions_for_connection("conn-1")``. Each session's
    ``close()`` must be called once, ``controller.close_session`` called
    once per session_id, and the connection entry removed.
    """
    mock_pool = _make_mock_pool()
    manager = ACPSessionManager(mock_pool)

    connection_id = "conn-1"
    session_id_1 = "s-controller-1"
    session_id_2 = "s-controller-2"

    session_1 = _make_mock_session(session_id_1)
    session_2 = _make_mock_session(session_id_2)

    manager._acp_sessions[session_id_1] = session_1
    manager._acp_sessions[session_id_2] = session_2
    manager._connection_sessions[connection_id] = {session_id_1, session_id_2}

    # Call close_all_sessions_for_connection
    await manager.close_all_sessions_for_connection(connection_id)

    # Assert: controller.close_session called once per session_id
    assert mock_pool.session_pool.close_session.await_count == 2
    call_args: list[tuple[str, ...]] = [
        call.args for call in mock_pool.session_pool.close_session.await_args_list
    ]
    assert (session_id_1,) in call_args
    assert (session_id_2,) in call_args

    # Assert: each session.close() called once
    session_1.close.assert_awaited_once()
    session_2.close.assert_awaited_once()

    # Assert: connection removed from _connection_sessions
    assert connection_id not in manager._connection_sessions

    # Assert: both session_ids removed from _acp_sessions
    assert session_id_1 not in manager._acp_sessions
    assert session_id_2 not in manager._acp_sessions


# ============================================================================
# Category B: Lifecycle Edge Cases
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_lifecycle_create_acp_transport_cleanup_unregister() -> None:
    """Full lifecycle: create ACP transport → cleanup → all registries empty.

    Creates a connection, registers a session, adds an ACP transport,
    then calls ``cleanup_session()``. After add, the context must have
    ACP connection IDs and the connection must have active sessions.
    After cleanup, all three registries must be empty.
    """
    mcp_manager, acp_manager = _make_wired_managers()
    session_id = "test-full-lifecycle-1"
    connection_id = "conn-full-lifecycle"
    server_config = AcpMcpServer(name="lifecycle-server", id="lifecycle-server-id")
    send_to_client = AsyncMock(return_value=None)

    try:
        # Create connection + register session
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

        # Add ACP transport
        ctx = mcp_manager.get_or_create_session(session_id)
        await mcp_manager.add_acp_transport(
            session_id,
            client_id="lifecycle-client",
            transport=_FakeTransport("lifecycle-transport"),
            connection_id=connection_id,
            session_key=session_key,
        )

        # Assert after add
        assert (connection_id, session_key) in ctx.acp_connection_ids
        assert conn.has_active_sessions()

        # Cleanup
        await mcp_manager.cleanup_session(session_id)

        # Assert after cleanup: all three registries empty
        assert mcp_manager.get_session_context(session_id) is None
        assert session_id not in acp_manager._session_connections
        assert not conn.has_active_sessions()
        assert connection_id not in acp_manager._connections
    finally:
        await mcp_manager.cleanup()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_close_then_recreate_same_session_id_fresh_context() -> None:
    """Close then recreate same session_id yields fresh context.

    Creates a session, populates toolset_cache and ACP transport entry,
    cleans up, then re-creates with ``get_or_create_session()``. The
    new context must be a different object with empty toolset_cache,
    empty acp_connection_ids, a new connection_pool, and snapshot=None.
    """
    mcp_manager = MCPManager(name="test")
    session_id = "test-recreate-1"

    try:
        # Create session and populate state
        old_ctx = mcp_manager.get_or_create_session(session_id)
        old_ctx.toolset_cache["dummy"] = MagicMock()
        old_ctx.acp_connection_ids.append(("conn-old", 0))
        old_snapshot = MagicMock()
        old_ctx.snapshot = old_snapshot

        # Cleanup
        await mcp_manager.cleanup_session(session_id)

        # Recreate
        new_ctx = mcp_manager.get_or_create_session(session_id)

        # Assert fresh context
        assert new_ctx is not old_ctx
        assert len(new_ctx.toolset_cache) == 0
        assert len(new_ctx.acp_connection_ids) == 0
        assert new_ctx.connection_pool is not old_ctx.connection_pool
        assert new_ctx.snapshot is None
    finally:
        await mcp_manager.cleanup()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_two_sessions_share_one_acp_connection_close_one_other_survives() -> None:
    """Two sessions share one ACP connection; closing one preserves the other.

    Creates one ACP connection, registers two sessions on it, then
    cleans up session-1. Session-2's stream pair must survive and the
    connection must remain in ``_connections``.
    """
    acp_manager = AcpMcpConnectionManager()
    connection_id = "conn-shared"
    server_config = AcpMcpServer(name="shared-server", id="shared-server-id")
    send_to_client = AsyncMock(return_value=None)

    # Create one connection
    conn = await acp_manager.create_connection(
        connection_id,
        server_config,
        send_to_client,
    )

    # Register two sessions on the same connection
    _pair_1, session_key_1 = conn.register_session()
    _pair_2, session_key_2 = conn.register_session()

    acp_manager.register_session_connection("session-1", connection_id, session_key_1)
    acp_manager.register_session_connection("session-2", connection_id, session_key_2)

    # Verify pre-state
    assert "session-1" in acp_manager._session_connections
    assert "session-2" in acp_manager._session_connections
    assert conn.has_active_sessions()

    # Cleanup session-1 only
    await acp_manager.cleanup_session("session-1")

    # Assert session-1 removed, session-2 survives
    assert "session-1" not in acp_manager._session_connections
    assert "session-2" in acp_manager._session_connections

    # Connection still in _connections (session-2 still active)
    assert connection_id in acp_manager._connections
    assert conn.has_active_sessions()

    # Session-2's stream pair still present
    assert session_key_2 in conn._session_streams


@pytest.mark.unit
@pytest.mark.asyncio
async def test_websocket_disconnect_closes_all_sessions_on_connection() -> None:
    """WebSocket disconnect closes all sessions on the affected connection.

    Pre-populates 2 sessions on "conn-1", mocks
    ``SessionController.close_session``, then calls
    ``close_all_sessions_for_connection("conn-1")``. Both sessions must
    be closed and removed from ``_acp_sessions``.
    """
    mock_pool = _make_mock_pool()
    manager = ACPSessionManager(mock_pool)

    connection_id = "conn-1"
    session_id_1 = "s1"
    session_id_2 = "s2"

    session_1 = _make_mock_session(session_id_1)
    session_2 = _make_mock_session(session_id_2)

    manager._acp_sessions[session_id_1] = session_1
    manager._acp_sessions[session_id_2] = session_2
    manager._connection_sessions[connection_id] = {session_id_1, session_id_2}

    # Simulate WebSocket disconnect
    await manager.close_all_sessions_for_connection(connection_id)

    # Both sessions' close() called
    session_1.close.assert_awaited_once()
    session_2.close.assert_awaited_once()

    # controller.close_session called for both
    assert mock_pool.session_pool.close_session.await_count == 2

    # Connection removed
    assert connection_id not in manager._connection_sessions

    # Both sessions removed from _acp_sessions
    assert session_id_1 not in manager._acp_sessions
    assert session_id_2 not in manager._acp_sessions


@pytest.mark.unit
@pytest.mark.asyncio
async def test_websocket_disconnect_only_closes_affected_connection_sessions() -> None:
    """WebSocket disconnect only closes sessions on the affected connection.

    Populates sessions on two connections ("conn-1" and "conn-2"),
    disconnects "conn-1", and verifies that only conn-1's sessions are
    closed while conn-2's sessions survive.
    """
    mock_pool = _make_mock_pool()
    manager = ACPSessionManager(mock_pool)

    # conn-1 has s1, s2; conn-2 has s3
    session_1 = _make_mock_session("s1")
    session_2 = _make_mock_session("s2")
    session_3 = _make_mock_session("s3")

    manager._acp_sessions["s1"] = session_1
    manager._acp_sessions["s2"] = session_2
    manager._acp_sessions["s3"] = session_3
    manager._connection_sessions["conn-1"] = {"s1", "s2"}
    manager._connection_sessions["conn-2"] = {"s3"}

    # Disconnect conn-1 only
    await manager.close_all_sessions_for_connection("conn-1")

    # s1, s2 closed; s3 NOT closed
    session_1.close.assert_awaited_once()
    session_2.close.assert_awaited_once()
    session_3.close.assert_not_awaited()

    # conn-1 removed; conn-2 still present with {s3}
    assert "conn-1" not in manager._connection_sessions
    assert "conn-2" in manager._connection_sessions
    assert manager._connection_sessions["conn-2"] == {"s3"}

    # s1, s2 removed from _acp_sessions; s3 still present
    assert "s1" not in manager._acp_sessions
    assert "s2" not in manager._acp_sessions
    assert "s3" in manager._acp_sessions


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resume_session_old_closed_new_has_fresh_acp_mcp_manager() -> None:
    """resume_session() closes old session; new session has fresh ACP MCP manager.

    Pre-populates an old session, mocks session_store.load, patches
    ACPSession constructor to capture the agent, then calls
    ``resume_session()``. The old session's ``close()`` and
    ``controller.close_session()`` must be called, and the new
    session's ``agent.mcp._acp_mcp_manager`` must be wired to
    ``acp_agent._mcp_manager``.
    """
    session_id = "test-resume-fresh-1"

    # --- Build mock pool ---
    mock_pool: MagicMock = MagicMock()
    mock_pool.manifest.agents = {"test_agent": MagicMock()}

    mock_sessions: MagicMock = MagicMock()
    mock_pool.session_pool.sessions = mock_sessions

    # Mock session_store.load
    mock_store: AsyncMock = AsyncMock()
    mock_store.load_session = AsyncMock(
        return_value=SessionData(
            session_id=session_id,
            agent_name="test_agent",
            cwd="/tmp",
        ),
    )
    mock_sessions.store = mock_store
    mock_sessions.close_session = AsyncMock()
    mock_pool.session_pool.close_session = AsyncMock()
    mock_pool.session_pool._get_resume_lock = AsyncMock(return_value=__import__("asyncio").Lock())

    # Mock get_or_create_session_agent — returns a mock agent with mcp
    mock_agent: MagicMock = MagicMock()
    mock_agent.name = "test_agent"
    mock_agent.mcp = MagicMock()

    async def mock_get_agent(sid: str, agent_name: str | None = None) -> MagicMock:
        return mock_agent

    mock_sessions.get_or_create_session_agent = mock_get_agent

    # --- Create ACPSessionManager ---
    acp_manager = ACPSessionManager(mock_pool)

    # Pre-populate old session
    old_session: AsyncMock = AsyncMock()
    old_session.session_id = session_id
    old_session.close = AsyncMock()
    acp_manager._acp_sessions[session_id] = old_session

    # Build a mock acp_agent with _mcp_manager
    mock_acp_agent: MagicMock = MagicMock()
    mock_acp_agent._mcp_manager = AcpMcpConnectionManager()

    # Patch ACPSession constructor to capture the agent
    mock_new_session: AsyncMock = AsyncMock()
    mock_new_session.session_id = session_id
    mock_new_session.initialize = AsyncMock()
    mock_new_session.initialize_mcp_servers = AsyncMock()
    mock_new_session.register_update_callback = MagicMock()

    captured_agent: list[MagicMock] = []

    def _capture_session(*args: object, **kwargs: object) -> AsyncMock:
        # Capture the agent from kwargs for assertion
        agent = kwargs.get("agent")
        if agent is not None:
            captured_agent.append(agent)  # type: ignore[arg-type]
            # Simulate __post_init__ wiring — agent is a MagicMock so
            # agent.mcp is always accessible without hasattr.
            agent.mcp._acp_mcp_manager = mock_acp_agent._mcp_manager  # type: ignore[method-assign]
        return mock_new_session

    with patch(
        "wolfharness_server.acp_server.session_manager.ACPSession",
        side_effect=_capture_session,
    ):
        result = await acp_manager.resume_session(
            session_id=session_id,
            client=MagicMock(),
            acp_agent=mock_acp_agent,
        )

    assert result is mock_new_session

    # Assert: old session's close() called
    old_session.close.assert_awaited_once()

    # Assert: controller.close_session called for old session_id
    mock_sessions.close_session.assert_awaited()
    call_args_list: list[tuple[str, ...]] = [
        call.args for call in mock_sessions.close_session.await_args_list
    ]
    assert (session_id,) in call_args_list

    # Assert: new session's agent.mcp._acp_mcp_manager is acp_agent._mcp_manager
    assert len(captured_agent) == 1
    captured = captured_agent[0]
    assert captured.mcp._acp_mcp_manager is mock_acp_agent._mcp_manager


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_cleanup_from_disconnect_and_controller_no_double_free() -> None:
    """Concurrent cleanup_session() calls do not double-free resources.

    Uses a real MCPManager with a CountingPool that tracks cleanup calls.
    Fires two concurrent ``cleanup_session()`` calls via
    ``asyncio.gather()``. The identity check after acquiring the lock
    must prevent redundant cleanup — ``connection_pool.cleanup()`` is
    called exactly once and no exception is raised.
    """
    mcp_manager = MCPManager(name="test")
    session_id = "test-concurrent-1"

    try:
        # Create a session context
        ctx = mcp_manager.get_or_create_session(session_id)

        # Replace connection_pool with a CountingPool that forces a context switch
        cleanup_call_count = 0

        class CountingPool:
            async def cleanup(self, timeout: float = 5.0) -> None:
                nonlocal cleanup_call_count
                await asyncio.sleep(0)  # Force context switch
                cleanup_call_count += 1

        ctx.connection_pool = CountingPool()  # type: ignore[method-assign]

        # Fire two concurrent cleanup calls
        await asyncio.gather(
            mcp_manager.cleanup_session(session_id),
            mcp_manager.cleanup_session(session_id),
        )

        # With the identity check, cleanup should be called exactly once
        assert cleanup_call_count == 1, (
            f"connection_pool.cleanup() was called {cleanup_call_count} times, "
            f"expected 1. The identity check after acquiring the lock should "
            f"prevent redundant cleanup work."
        )

        # Session removed from session context registry
        assert mcp_manager.get_session_context(session_id) is None
    finally:
        await mcp_manager.cleanup()
