"""TDD tests for code-review bugs in MCP session lifecycle.

Round 1 tests (Fixes #1/#2/#3) are GREEN — already fixed in df377db9b.
Round 2 tests (Fixes #4/#5/#6) are written BEFORE the fixes (RED phase).

Bugs covered:
- Fix #2 (DONE): get_capabilities(session_id) recreates a cleaned-up session context
- Fix #3 (DONE): cleanup_session() calls get_or_create_session() on non-existent sessions
- Fix #1 (DONE): resume_session() doesn't remove session_id from old connection's
  _connection_sessions, causing close_all_sessions_for_connection() to close
  the NEW session when the old connection disconnects.

Round 2:
- Fix #4: _acp_mcp_manager never wired — cleanup_session() never delegates
  to AcpMcpConnectionManager.cleanup_session(), leaking per-session ACP
  stream pairs and reverse-index entries.
- Fix #5: cleanup_session() lacks identity check after acquiring lock —
  concurrent callers may do redundant work (all ops are idempotent, but
  the check is a defensive optimization).
- Fix #6: get_capabilities() has 3 identical fallback loops — pure duplication.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wolfharness.mcp_server.manager import MCPManager
from wolfharness.sessions import SessionData
from wolfharness_server.acp_server.session_manager import ACPSessionManager


# ============================================================================
# Fix #2: get_capabilities() must not recreate a cleaned-up session context
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_capabilities_does_not_recreate_cleaned_session() -> None:
    """get_capabilities(session_id=...) must not recreate a cleaned-up context.

    Bug: ``get_capabilities(session_id=...)`` calls ``get_or_create_session()``
    which recreates an empty ``McpSessionContext`` if cleanup already popped it.
    This is a memory leak — the dead context lingers forever with an empty
    snapshot, and the ``KeyError`` fallback code is dead.

    Steps:
    1. Create MCPManager.
    2. get_or_create_session("test-leak") to create the context.
    3. cleanup_session("test-leak") to clean and pop it.
    4. Assert the context is gone from _session_contexts.
    5. Call get_capabilities(session_id="test-leak").
    6. Assert the context is STILL NOT in _session_contexts — the bug
       recreates it; the fix should not.
    7. cleanup().
    """
    manager = MCPManager(name="test")
    try:
        # 1-2. Create session context
        manager.get_or_create_session("test-leak")
        assert "test-leak" in manager._session_contexts

        # 3. Clean up the session
        await manager.cleanup_session("test-leak")

        # 4. Context should be gone
        assert "test-leak" not in manager._session_contexts

        # 5. Call get_capabilities with the cleaned-up session_id
        await manager.get_capabilities(session_id="test-leak")

        # 6. BUG: get_or_create_session() recreates the context.
        #    FIX: should use _session_contexts.get() so no recreation occurs.
        assert "test-leak" not in manager._session_contexts, (
            "get_capabilities() recreated a cleaned-up session context — "
            "memory leak. Should use _session_contexts.get() instead of "
            "get_or_create_session()."
        )
    finally:
        await manager.cleanup()


# ============================================================================
# Fix #3: cleanup_session() must not call get_or_create_session()
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cleanup_session_does_not_recreate_context() -> None:
    """cleanup_session() on a non-existent session must be a no-op.

    Bug: ``cleanup_session()`` calls ``get_or_create_session()`` which
    creates a throwaway ``SessionConnectionPool`` if none exists.  The
    pool is immediately cleaned up and the context popped in ``finally``,
    but the unnecessary creation is wasteful and can have side effects.

    Steps:
    1. Create MCPManager.
    2. Assert "test-no-create" is not in _session_contexts.
    3. Patch get_or_create_session to track if it was called.
    4. Call cleanup_session("test-no-create").
    5. Assert get_or_create_session was NOT called.
    6. Assert "test-no-create" is still not in _session_contexts.
    7. cleanup().
    """
    manager = MCPManager(name="test")
    try:
        # 2. Session never created
        assert "test-no-create" not in manager._session_contexts

        # 3-4. Wrap get_or_create_session to detect if it's called
        with patch.object(
            manager,
            "get_or_create_session",
            wraps=manager.get_or_create_session,
        ) as mock_get:
            await manager.cleanup_session("test-no-create")

            # 5. BUG: cleanup_session calls get_or_create_session, which
            #    creates a throwaway context.
            #    FIX: should use _session_contexts.get() instead.
            assert not mock_get.called, (
                "cleanup_session() called get_or_create_session() for a "
                "non-existent session — should use _session_contexts.get() "
                "instead to avoid creating a throwaway SessionConnectionPool."
            )

        # 6. Still not in _session_contexts
        assert "test-no-create" not in manager._session_contexts
    finally:
        await manager.cleanup()


# ============================================================================
# Fix #1: resume_session() must clean _connection_sessions for old connection
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resume_session_cleans_connection_sessions() -> None:
    """resume_session() must remove session_id from old connection's set.

    Bug: ``resume_session()`` does not remove ``session_id`` from the old
    connection's ``_connection_sessions`` entry before closing the old
    session.  If the old connection later disconnects,
    ``close_all_sessions_for_connection()`` finds the ``session_id`` and
    closes the NEW (resumed) session.

    Steps:
    1. Create ACPSessionManager with a mock pool.
    2. Manually populate _connection_sessions["conn-1"] = {"session-1"}.
    3. Manually populate _acp_sessions["session-1"] = mock old session.
    4. Mock session_store.load to return session data.
    5. Mock get_or_create_session_agent to return a mock agent.
    6. Patch ACPSession constructor to return a mock new session.
    7. Call resume_session("session-1", ..., connection_id="conn-2").
    8. Assert "session-1" NOT in _connection_sessions["conn-1"] (bug leaves it).
    9. Assert "session-1" IN _connection_sessions["conn-2"] (new connection).
    """
    session_id = "session-1"

    # --- 1. Build mock pool ---
    mock_pool: MagicMock = MagicMock()
    mock_pool.manifest.agents = {"test_agent": MagicMock()}

    mock_sessions: MagicMock = MagicMock()
    mock_pool.session_pool.sessions = mock_sessions
    mock_pool.session_pool._get_resume_lock = AsyncMock(return_value=asyncio.Lock())
    mock_pool.session_pool.close_session = AsyncMock()

    # 4. Mock session_store.load
    mock_store: AsyncMock = AsyncMock()
    mock_store.load_session = AsyncMock(
        return_value=SessionData(
            session_id=session_id,
            agent_name="test_agent",
            cwd="/tmp",
        )
    )
    mock_sessions.store = mock_store

    # Mock controller.close_session (called during resume)
    mock_sessions.close_session = AsyncMock()

    # 5. Mock get_or_create_session_agent
    mock_agent: MagicMock = MagicMock()
    mock_agent.name = "test_agent"

    async def mock_get_agent(sid: str, agent_name: str | None = None) -> MagicMock:
        return mock_agent

    mock_sessions.get_or_create_session_agent = mock_get_agent

    # --- 2. Create ACPSessionManager ---
    acp_manager = ACPSessionManager(mock_pool)

    # --- 3. Populate old connection tracking and old session ---
    acp_manager._connection_sessions["conn-1"] = {session_id}

    old_session: AsyncMock = AsyncMock()
    old_session.session_id = session_id
    old_session.close = AsyncMock()
    acp_manager._acp_sessions[session_id] = old_session

    # --- 6. Patch ACPSession constructor ---
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

    # Result should be the new session
    assert result is mock_new_session

    # --- 8. BUG: resume_session does NOT remove session_id from old conn.
    #    FIX: should remove session_id from _connection_sessions["conn-1"].
    old_conn_sessions = acp_manager._connection_sessions.get("conn-1", set())
    assert session_id not in old_conn_sessions, (
        f"resume_session() left session_id {session_id!r} in old connection "
        f"conn-1's _connection_sessions set. If conn-1 disconnects later, "
        f"close_all_sessions_for_connection() will close the NEW session."
    )

    # --- 9. New connection should track the session ---
    new_conn_sessions = acp_manager._connection_sessions.get("conn-2", set())
    assert session_id in new_conn_sessions, (
        f"resume_session() did not register session_id {session_id!r} "
        f"in new connection conn-2's _connection_sessions set."
    )


# ============================================================================
# Fix #4: _acp_mcp_manager never wired in production
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cleanup_session_delegates_to_acp_mcp_manager() -> None:
    """cleanup_session() must delegate to _acp_mcp_manager when wired.

    Bug: ``MCPManager._acp_mcp_manager`` is initialized to ``None`` and
    never set in production. ``cleanup_session()`` checks
    ``if self._acp_mcp_manager is not None:`` but it's always ``False``,
    so ``AcpMcpConnectionManager.cleanup_session()`` never runs, leaking
    per-session ACP stream pairs and reverse-index entries.

    This test validates the delegation path by manually wiring a mock
    and verifying cleanup_session() calls it.
    """
    manager = MCPManager(name="test")
    try:
        # Manually wire a mock ACP MCP manager
        mock_acp_manager: AsyncMock = AsyncMock()
        manager._acp_mcp_manager = mock_acp_manager

        # Create a session context
        manager.get_or_create_session("test-acp-delegate")

        # Clean up the session
        await manager.cleanup_session("test-acp-delegate")

        # Verify delegation occurred
        mock_acp_manager.cleanup_session.assert_called_once_with("test-acp-delegate")
    finally:
        await manager.cleanup()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_session_wires_acp_mcp_manager() -> None:
    """ACPSession.__post_init__ must wire agent.mcp._acp_mcp_manager.

    Bug: ``ACPSession.__post_init__`` never sets
    ``agent.mcp._acp_mcp_manager = acp_agent._mcp_manager``.
    This means cleanup_session() on the agent's MCPManager can never
    delegate to AcpMcpConnectionManager, leaking ACP connections.

    Fix: In ``__post_init__``, after ``self.agent.env = self.acp_env``,
    wire the manager for native agents.
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
    agent = Agent.from_callback(name="test_agent", callback=simple_callback, agent_pool=pool)
    acp_agent = AgentPoolACPAgent(client=MagicMock(), default_agent=agent)

    # Create ACPSession with minimal required fields
    ACPSession(
        session_id="test-wire",
        agent=agent,
        cwd="/tmp",
        client=MagicMock(),
        acp_agent=acp_agent,
    )

    # Verify wiring
    assert agent.mcp._acp_mcp_manager is acp_agent._mcp_manager, (
        "ACPSession.__post_init__ must wire agent.mcp._acp_mcp_manager "
        "to acp_agent._mcp_manager for cleanup delegation to work"
    )


# ============================================================================
# Fix #5: Identity check after acquiring cleanup lock
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cleanup_session_identity_check_prevents_redundant_work() -> None:
    """cleanup_session() must check context identity after acquiring lock.

    Bug: When two concurrent cleanup_session() calls race for the same
    session, the first caller pops the context in the ``finally`` block.
    The second caller, already past the ``ctx is None`` check, acquires
    the lock and runs cleanup on an already-popped context. All ops are
    idempotent, but the redundant work is wasteful.

    Fix: After ``async with ctx._cleanup_lock:``, check
    ``if self._session_contexts.get(session_id) is not ctx: return``.

    This test creates a session, then fires two concurrent
    cleanup_session() calls. With the fix, connection_pool.cleanup() is
    called only once. Without the fix, it may be called twice.
    """
    manager = MCPManager(name="test")
    try:
        # Create a session context
        ctx = manager.get_or_create_session("test-identity")

        # Replace connection_pool with a mock that counts cleanup calls.
        # The asyncio.sleep(0) forces a context switch so both tasks
        # actually race for the lock (without it, the first task runs
        # to completion synchronously without yielding).
        cleanup_call_count = 0

        class CountingPool:
            async def cleanup(self, timeout: float = 5.0) -> None:
                nonlocal cleanup_call_count
                await asyncio.sleep(0)  # Force context switch
                cleanup_call_count += 1

        ctx.connection_pool = CountingPool()

        # Fire two concurrent cleanup calls
        import asyncio

        await asyncio.gather(
            manager.cleanup_session("test-identity"),
            manager.cleanup_session("test-identity"),
        )

        # With the identity check, cleanup should be called exactly once.
        assert cleanup_call_count == 1, (
            f"connection_pool.cleanup() was called {cleanup_call_count} times, "
            f"expected 1. The identity check after acquiring the lock should "
            f"prevent redundant cleanup work."
        )
    finally:
        await manager.cleanup()
