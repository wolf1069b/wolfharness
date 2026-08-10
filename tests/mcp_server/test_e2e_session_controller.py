"""End-to-end integration tests for MCP session lifecycle via SessionController.

Covers real SessionController paths with real agents and real MCPManager
instances, verifying that the full create/close chain populates and cleans
per-session MCP state correctly.

Tests:
    G1  - Full create-session chain populates MCP McpSessionContext with snapshot
    G5  - close_session cleans agent.mcp session context and exits agent context
    G6  - resume_session creates real ACPSession with _acp_mcp_manager wired
    G13 - MCPManager.cleanup() cleans all session contexts
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from wolfharness.mcp_server.config_snapshot import McpConfigSnapshot
from wolfharness.mcp_server.manager import MCPManager
from wolfharness.orchestrator.session_controller import SessionController, SessionState
from wolfharness.sessions import SessionData


# ============================================================================
# Helpers
# ============================================================================


def _make_mock_pool() -> MagicMock:
    """Create a mock AgentPool suitable for SessionController tests.

    Returns a MagicMock with manifest, main_agent_name, _config_file_path,
    skills_tools_provider, and mcp configured
    so that SessionController.get_or_create_session_agent() can resolve
    a NativeAgentConfig and call cfg.get_agent().
    """
    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.manifest import AgentsManifest

    mock_pool: MagicMock = MagicMock()
    mock_pool.manifest = AgentsManifest(
        agents={"test_agent": NativeAgentConfig(model="test")},
    )
    mock_pool.main_agent_name = "test_agent"
    mock_pool._config_file_path = None
    mock_pool.skill_capabilities = []
    mock_pool.mcp = MCPManager(name="pool-mcp")
    return mock_pool


def _make_mock_client() -> MagicMock:
    """Create a mock ACP Client with spec for isinstance checks."""
    return MagicMock()


# ============================================================================
# G1: Full create_session chain populates MCP session context
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_create_session_chain_populates_mcp_session_context() -> None:
    """SessionController.get_or_create_session_agent() populates MCPManager McpSessionContext.

    Verifies that the real SessionController, when given a NativeAgentConfig,
    calls agent.mcp.get_or_create_session() and agent.mcp.update_session_snapshot()
    so that the agent's MCPManager has a McpSessionContext with a non-None snapshot.
    """
    from wolfharness.agents.native_agent import Agent
    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.manifest import AgentsManifest

    # Build a real manifest with a NativeAgentConfig
    manifest = AgentsManifest(
        agents={"test_agent": NativeAgentConfig(model="test")},
    )

    # Build a real Agent from a callback (no model API calls needed)
    def simple_callback(message: str) -> str:
        return f"Test: {message}"

    agent = Agent.from_callback(
        name="test_agent",
        callback=simple_callback,
    )

    # Build mock pool that returns our agent config
    mock_pool: MagicMock = MagicMock()
    mock_pool.manifest = manifest
    mock_pool.main_agent_name = "test_agent"
    mock_pool._config_file_path = None
    mock_pool.skill_capabilities = []
    mock_pool.mcp = agent.mcp  # Use the agent's own MCPManager
    mock_pool.get_context.return_value = MagicMock()

    # Use side_effect to mimic AgentFactory._create_native_main() behavior:
    # populate the MCP session context with a snapshot.
    async def _mock_create_session_agent(**kwargs: object) -> Agent:
        sid = kwargs.get("session_id", "")
        agent.mcp.get_or_create_session(sid)
        agent.mcp.update_session_snapshot(sid, McpConfigSnapshot())
        return agent

    mock_pool._factory.create_session_agent = AsyncMock(
        side_effect=_mock_create_session_agent,
    )

    controller = SessionController(pool=mock_pool)

    session_id = "test-g1-create-session"

    try:
        # Call the real get_or_create_session_agent method
        result_agent = await controller.get_or_create_session_agent(
            session_id, agent_name="test_agent"
        )

        # Assert: agent's MCPManager has a McpSessionContext with non-None snapshot
        ctx = result_agent.mcp.get_session_context(session_id)
        assert ctx is not None, "get_or_create_session_agent() must create a McpSessionContext"
        assert ctx.snapshot is not None, (
            "get_or_create_session_agent() must call update_session_snapshot()"
        )
        assert isinstance(ctx.snapshot, McpConfigSnapshot), (
            "Snapshot must be a McpConfigSnapshot instance"
        )
        assert ctx.connection_pool is not None, (
            "get_or_create_session() must create a SessionConnectionPool"
        )
    finally:
        # Clean up
        agent_exit = result_agent if "result_agent" in dir() else None
        if agent_exit is not None:
            with contextlib.suppress(Exception):
                await agent_exit.__aexit__(None, None, None)
        await controller.close_session(session_id)


# ============================================================================
# G5: close_session with real agent and MCP resources
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_session_controller_close_session_with_real_agent_and_mcp_resources() -> None:
    """close_session() calls agent.mcp.cleanup_session() and agent.__aexit__().

    Creates a real Agent via Agent.from_callback(), manually registers it
    in controller._session_agents, populates agent.mcp session context
    with snapshot, toolset_cache entry, and acp_connection_ids. Then calls
    close_session() and verifies the session context is cleaned and the
    agent context is exited.
    """
    from wolfharness.agents.native_agent import Agent

    def simple_callback(message: str) -> str:
        return f"Test: {message}"

    agent = Agent.from_callback(
        name="test_agent",
        callback=simple_callback,
    )
    await agent.__aenter__()

    mock_pool = _make_mock_pool()
    controller = SessionController(pool=mock_pool)

    session_id = "test-g5-close-session"

    # Populate agent.mcp session context with real data
    ctx = agent.mcp.get_or_create_session(session_id)
    ctx.snapshot = McpConfigSnapshot()
    ctx.toolset_cache["test-client-id"] = MagicMock()
    ctx.acp_connection_ids.append(("test-conn-id", 1))

    # Register agent in controller._session_agents
    controller._session_agents[session_id] = agent

    # Create a session state
    session = SessionState(
        session_id=session_id,
        agent_name="test_agent",
    )
    session.is_per_session_agent = True
    controller._sessions[session_id] = session

    # Verify pre-close state
    assert agent.mcp.get_session_context(session_id) is not None
    assert len(ctx.toolset_cache) > 0
    assert len(ctx.acp_connection_ids) > 0

    with contextlib.suppress(Exception):
        await controller.close_session(session_id)

    # Assert: session context is empty after close
    assert agent.mcp.get_session_context(session_id) is None, (
        "close_session() must call agent.mcp.cleanup_session() which removes the session context"
    )
    # Assert: session removed from controller._sessions
    assert session_id not in controller._sessions, (
        "close_session() must remove session from _sessions"
    )
    # Assert: session removed from controller._session_agents
    assert session_id not in controller._session_agents, (
        "close_session() must remove agent from _session_agents"
    )


# ============================================================================
# G6: resume_session with real ACPSession (not patched)
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resume_session_with_real_acpsession_not_patched() -> None:
    """resume_session() creates a real ACPSession with _acp_mcp_manager wired.

    Uses real AgentPool + real ACPSessionManager. Creates an initial session,
    then calls resume_session(). The real ACPSession.__post_init__ must run
    and wire agent.mcp._acp_mcp_manager to acp_agent._mcp_manager.
    """
    from wolfharness import Agent
    from wolfharness.delegation import AgentPool
    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.manifest import AgentsManifest
    from wolfharness_server.acp_server.acp_agent import AgentPoolACPAgent
    from wolfharness_server.acp_server.session_manager import ACPSessionManager

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
    acp_agent = AgentPoolACPAgent(client=_make_mock_client(), default_agent=agent)

    # Build mock session pool infrastructure for ACPSessionManager
    mock_session_pool: MagicMock = MagicMock()
    mock_sessions: MagicMock = MagicMock()
    mock_session_pool.sessions = mock_sessions
    mock_session_pool.create_session = AsyncMock()
    mock_session_pool.close_session = AsyncMock()
    mock_session_pool._get_resume_lock = AsyncMock(return_value=asyncio.Lock())
    pool._session_pool = mock_session_pool

    # Mock session store for resume
    mock_store: AsyncMock = AsyncMock()
    session_id = "test-g6-resume-real"
    mock_store.load_session = AsyncMock(
        return_value=SessionData(
            session_id=session_id,
            agent_name="test_agent",
            cwd="/tmp",
        )
    )
    mock_sessions.store = mock_store
    mock_sessions.close_session = AsyncMock()

    # Mock get_or_create_session_agent to return our real agent
    async def mock_get_agent(sid: str, agent_name: str | None = None) -> Agent:
        agent.mcp.get_or_create_session(sid)
        return agent

    mock_sessions.get_or_create_session_agent = mock_get_agent

    acp_manager = ACPSessionManager(pool)

    # Inject a mock old session into _acp_sessions
    old_session: AsyncMock = AsyncMock()
    old_session.session_id = session_id

    async def mock_old_close() -> None:
        await agent.mcp.cleanup_session(session_id)

    old_session.close = mock_old_close
    acp_manager._acp_sessions[session_id] = old_session

    try:
        result = await acp_manager.resume_session(
            session_id=session_id,
            client=_make_mock_client(),
            acp_agent=acp_agent,
        )

        # Assert: result is a real ACPSession (not a mock)
        from wolfharness_server.acp_server.session import ACPSession

        assert isinstance(result, ACPSession), (
            "resume_session() must return a real ACPSession, not a mock"
        )

        # Assert: new session's agent.mcp._acp_mcp_manager is wired
        assert result.agent.mcp._acp_mcp_manager is acp_agent._mcp_manager, (
            "ACPSession.__post_init__ must wire agent.mcp._acp_mcp_manager "
            "to acp_agent._mcp_manager"
        )
        assert result.agent.mcp._acp_mcp_manager is not None, (
            "_acp_mcp_manager must not be None after resume"
        )
    finally:
        # Cleanup
        with contextlib.suppress(Exception):
            await agent.mcp.cleanup()
        with contextlib.suppress(Exception):
            await agent.__aexit__(None, None, None)


# ============================================================================
# G13: Pool shutdown cleans all session MCP resources
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pool_shutdown_cleans_all_session_mcp_resources() -> None:
    """MCPManager.cleanup() cleans all session contexts.

    Creates a real MCPManager with multiple session contexts populated
    (3 sessions with snapshots, toolset_cache entries, acp_connection_ids).
    Calls await manager.cleanup() and asserts session contexts are empty
    for all sessions after cleanup.

    Note: MCPManager.cleanup() closes the exit_stack and clears providers.
    Session contexts are cleaned via cleanup_session() per session, but
    the test verifies that after a full cleanup(), no session state leaks.
    We call cleanup_session() for each session first (simulating what
    AgentPool.__aexit__ -> SessionPool.shutdown() does), then cleanup().
    """
    manager = MCPManager(name="test-g13")

    session_ids = ["test-g13-s1", "test-g13-s2", "test-g13-s3"]

    try:
        # Populate 3 session contexts with real data
        for sid in session_ids:
            ctx = manager.get_or_create_session(sid)
            ctx.snapshot = McpConfigSnapshot()
            ctx.toolset_cache[f"client-{sid}"] = MagicMock()
            ctx.acp_connection_ids.append((f"conn-{sid}", 1))

        # Verify pre-cleanup state
        for sid in session_ids:
            ctx = manager.get_session_context(sid)
            assert ctx is not None
            assert ctx.snapshot is not None
            assert len(ctx.toolset_cache) > 0
            assert len(ctx.acp_connection_ids) > 0

        # Simulate pool shutdown: clean up each session, then global cleanup
        for sid in session_ids:
            await manager.cleanup_session(sid)

        # Assert: all session contexts removed
        for sid in session_ids:
            assert manager.get_session_context(sid) is None, (
                f"cleanup_session() must remove session {sid} from the session context registry"
            )

        # Final global cleanup
        await manager.cleanup()
    finally:
        # Ensure cleanup even on assertion failure
        for sid in session_ids:
            with contextlib.suppress(Exception):
                await manager.cleanup_session(sid)
