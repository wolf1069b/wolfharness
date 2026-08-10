"""Integration test for ACPSessionManager.resume_session() lifecycle.

Verifies that resume_session() closes the old session (removing it from
_acp_sessions) and creates a fresh session with new MCP resources
(different McpSessionContext in MCPManager's session context).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wolfharness.mcp_server.manager import MCPManager
from wolfharness.sessions import SessionData
from wolfharness_server.acp_server.session_manager import ACPSessionManager


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resume_closes_old_session() -> None:
    """resume_session() closes old session and creates fresh MCP resources.

    Steps:
    1. Create an MCPManager and set up session-scoped MCP resources.
    2. Create an ACPSessionManager with a mock pool.
    3. Inject a mock old ACPSession into _acp_sessions.
    4. Call resume_session() with the same session_id.
    5. Verify old session was closed (popped from _acp_sessions, close called).
    6. Verify new session is a different object in _acp_sessions.
    7. Verify MCPManager's session context has a fresh McpSessionContext
       (different object, different toolset_cache, different connection_pool).
    """
    session_id = "test-resume-lifecycle-1"

    # --- 1. Create real MCPManager and set up old session context ---
    mcp_manager = MCPManager(name="test")
    old_ctx = mcp_manager.get_or_create_session(session_id)
    old_toolset_cache = old_ctx.toolset_cache
    old_connection_pool = old_ctx.connection_pool

    # --- 2. Build mock pool with session_pool, session_store, manifest ---
    mock_pool: MagicMock = MagicMock()
    mock_pool.manifest.agents = {"test_agent": MagicMock()}

    mock_sessions: MagicMock = MagicMock()
    mock_pool.session_pool.sessions = mock_sessions

    # Mock _get_resume_lock to return a real asyncio.Lock (awaitable)
    mock_pool.session_pool._get_resume_lock = AsyncMock(return_value=asyncio.Lock())

    # Mock session_store.load_session to return persisted SessionData
    mock_store: AsyncMock = AsyncMock()
    mock_store.load_session = AsyncMock(
        return_value=SessionData(
            session_id=session_id,
            agent_name="test_agent",
            cwd="/tmp",
        )
    )
    mock_sessions.store = mock_store

    # Mock controller.close_session (called during resume, before ACPSession.close)
    mock_sessions.close_session = AsyncMock()

    # Mock get_or_create_session_agent to create fresh MCP session context
    mock_agent: MagicMock = MagicMock()
    mock_agent.mcp = mcp_manager
    mock_agent.name = "test_agent"

    async def mock_get_agent(sid: str, agent_name: str | None = None) -> MagicMock:
        mcp_manager.get_or_create_session(sid)
        return mock_agent

    mock_sessions.get_or_create_session_agent = mock_get_agent

    # --- 3. Create ACPSessionManager and inject mock old session ---
    acp_manager = ACPSessionManager(mock_pool)

    old_session: AsyncMock = AsyncMock()
    old_session.session_id = session_id

    # Make old_session.close() actually clean up MCP session context
    async def mock_old_close() -> None:
        await mcp_manager.cleanup_session(session_id)

    old_session.close = mock_old_close
    acp_manager._acp_sessions[session_id] = old_session

    # --- 4. Patch ACPSession to avoid complex __post_init__ ---
    mock_new_session: AsyncMock = AsyncMock()
    mock_new_session.session_id = session_id
    mock_new_session.initialize = AsyncMock()
    mock_new_session.initialize_mcp_servers = AsyncMock()
    mock_new_session.register_update_callback = MagicMock()

    try:
        with patch(
            "wolfharness_server.acp_server.session_manager.ACPSession",
            return_value=mock_new_session,
        ):
            result = await acp_manager.resume_session(
                session_id=session_id,
                client=MagicMock(),
                acp_agent=MagicMock(),
            )
    finally:
        await mcp_manager.cleanup()

    # --- 5. Verify old session was closed ---
    # resume_session pops old session and calls close() on it
    # mock_old_close calls cleanup_session which removes the old session context
    assert result is mock_new_session

    # --- 6. Verify new session replaced old in _acp_sessions ---
    assert session_id in acp_manager._acp_sessions
    assert acp_manager._acp_sessions[session_id] is mock_new_session
    assert acp_manager._acp_sessions[session_id] is not old_session

    # --- 7. Verify fresh MCP resources ---
    # After cleanup_session (old) + get_or_create_session (new),
    # The session context should be completely new
    new_ctx = mcp_manager.get_session_context(session_id)
    assert new_ctx is not None
    assert new_ctx is not old_ctx
    assert new_ctx.toolset_cache is not old_toolset_cache
    assert new_ctx.connection_pool is not old_connection_pool
    assert new_ctx.connection_pool is not None
    assert new_ctx.toolset_cache == {}
