"""Tests for ACPSessionManager child-session path (RFC-0028 T13)."""

from __future__ import annotations

import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wolfharness import Agent
from wolfharness.delegation import AgentPool
from wolfharness.orchestrator.core import SessionPool
from wolfharness.sessions import SessionData
from wolfharness_server.acp_server.session_manager import ACPSessionManager
from wolfharness_storage.memory_provider.provider import MemoryStorageProvider


pytestmark = pytest.mark.unit


def _make_pool_with_sessions() -> tuple[AgentPool, Agent, SessionPool, MemoryStorageProvider]:
    """Create a pool with a real SessionPool backed by MemoryStorageProvider."""
    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.manifest import AgentsManifest

    manifest = AgentsManifest(agents={"test_agent": NativeAgentConfig(model="test")})

    pool = AgentPool(manifest)

    def simple_callback(message: str) -> str:
        return f"Test response: {message}"

    agent = Agent.from_callback(name="test_agent", callback=simple_callback, agent_pool=pool)
    # pool.register() removed; agent created from callback/config above
    store = MemoryStorageProvider()
    session_pool = SessionPool(pool=pool, store=store)
    pool._session_pool = session_pool

    # Also wire up storage.generate_session_id for top-level path
    pool.storage.generate_session_id = MagicMock(return_value="session_top_001")  # type: ignore[assignment]

    return pool, agent, session_pool, store


def _make_acp_session_manager(pool: AgentPool) -> ACPSessionManager:
    """Create an ACPSessionManager with minimal mock ACP agent."""
    manager = ACPSessionManager(pool=pool)

    # Mock out ACPSession creation and initialization to avoid needing
    # a real ACP client and all the initialization machinery.
    with patch("wolfharness_server.acp_server.session_manager.ACPSession") as mock_session_cls:
        mock_session = MagicMock()
        mock_session.session_id = "session_top_001"
        mock_session.register_update_callback = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.initialize_mcp_servers = AsyncMock()
        mock_session_cls.return_value = mock_session

    return manager


async def test_top_level_session_has_no_parent():
    """Top-level ACP session (no parent_session_id) should have parent_id=None and a...."""
    pool, agent, _sessions, store = _make_pool_with_sessions()

    manager = ACPSessionManager(pool=pool)
    mock_client = MagicMock()
    mock_acp_agent = MagicMock()

    with (
        patch("wolfharness_server.acp_server.session_manager.ACPSession") as mock_session_cls,
        patch("wolfharness_server.acp_server.session_manager.ClientCapabilities"),
    ):
        mock_session = MagicMock()
        mock_session.register_update_callback = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.initialize_mcp_servers = AsyncMock()
        mock_session_cls.return_value = mock_session

        session_id = await manager.create_session(
            agent_name=agent.name,
            cwd=tempfile.gettempdir(),
            client=mock_client,
            acp_agent=mock_acp_agent,
        )

    # Verify the session was persisted via store
    data = await store.load_session(session_id)
    assert data is not None
    assert data.parent_id is None
    assert data.project_id is not None
    assert data.agent_name == "test_agent"


async def test_child_session_inherits_parent_project_id():
    """Child ACP session (with parent_session_id) should inherit project_id and cwd from the...."""
    pool, agent, _sessions, store = _make_pool_with_sessions()

    # Create a parent session in the store first
    from wolfharness_storage.opencode_provider.helpers import compute_project_id

    parent_cwd = tempfile.gettempdir()
    parent_project_id = compute_project_id(parent_cwd)
    parent_data = SessionData(
        session_id="parent_session_001",
        agent_name="test_agent",
        cwd=parent_cwd,
        project_id=parent_project_id,
    )
    await store.save_session(parent_data)

    manager = ACPSessionManager(pool=pool)
    mock_client = MagicMock()
    mock_acp_agent = MagicMock()

    with (
        patch("wolfharness_server.acp_server.session_manager.ACPSession") as mock_session_cls,
        patch("wolfharness_server.acp_server.session_manager.ClientCapabilities"),
    ):
        mock_session = MagicMock()
        mock_session.register_update_callback = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.initialize_mcp_servers = AsyncMock()
        mock_session_cls.return_value = mock_session

        session_id = await manager.create_session(
            agent_name=agent.name,
            cwd="/some/other/cwd",  # Different cwd — should be overridden by parent's
            client=mock_client,
            acp_agent=mock_acp_agent,
            parent_session_id="parent_session_001",
        )

    # Load child session data from store
    child_data = await store.load_session(session_id)
    assert child_data is not None
    # Child should inherit parent's project_id
    assert child_data.project_id == parent_project_id
    # Child should inherit parent's cwd
    assert child_data.cwd == parent_cwd
    # Child should reference parent
    assert child_data.parent_id == "parent_session_001"
    # Agent type should be "acp"
    assert child_data.agent_type == "acp"


async def test_child_session_uses_effective_cwd_for_acp_session():
    """When creating a child ACP session, the ACPSession object should receive the inherited...."""
    pool, agent, _sessions, store = _make_pool_with_sessions()

    parent_cwd = tempfile.gettempdir()
    from wolfharness_storage.opencode_provider.helpers import compute_project_id

    parent_project_id = compute_project_id(parent_cwd)
    parent_data = SessionData(
        session_id="parent_session_002",
        agent_name="test_agent",
        cwd=parent_cwd,
        project_id=parent_project_id,
    )
    await store.save_session(parent_data)

    manager = ACPSessionManager(pool=pool)
    mock_client = MagicMock()
    mock_acp_agent = MagicMock()

    with (
        patch("wolfharness_server.acp_server.session_manager.ACPSession") as mock_session_cls,
        patch("wolfharness_server.acp_server.session_manager.ClientCapabilities"),
    ):
        mock_session = MagicMock()
        mock_session.register_update_callback = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.initialize_mcp_servers = AsyncMock()
        mock_session_cls.return_value = mock_session

        await manager.create_session(
            agent_name=agent.name,
            cwd="/different/cwd",
            client=mock_client,
            acp_agent=mock_acp_agent,
            parent_session_id="parent_session_002",
        )

        # Verify ACPSession was constructed with the inherited cwd
        call_kwargs = mock_session_cls.call_args
        assert (
            call_kwargs.kwargs.get("cwd") == parent_cwd or call_kwargs[1].get("cwd") == parent_cwd
        )


async def test_no_parent_session_id_preserves_existing_behavior():
    """When parent_session_id is None, the existing top-level behavior (compute project_id...."""
    pool, agent, _sessions, store = _make_pool_with_sessions()

    manager = ACPSessionManager(pool=pool)
    mock_client = MagicMock()
    mock_acp_agent = MagicMock()

    with (
        patch("wolfharness_server.acp_server.session_manager.ACPSession") as mock_session_cls,
        patch("wolfharness_server.acp_server.session_manager.ClientCapabilities"),
    ):
        mock_session = MagicMock()
        mock_session.register_update_callback = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.initialize_mcp_servers = AsyncMock()
        mock_session_cls.return_value = mock_session

        cwd = tempfile.gettempdir()
        session_id = await manager.create_session(
            agent_name=agent.name,
            cwd=cwd,
            client=mock_client,
            acp_agent=mock_acp_agent,
            parent_session_id=None,
        )

    data = await store.load_session(session_id)
    assert data is not None
    assert data.parent_id is None
    assert data.project_id is not None
    # cwd should match what was provided
    assert data.cwd == cwd


async def test_child_session_without_pool_sessions_falls_back_to_top_level():
    """When pool.sessions is None but parent_session_id is provided, should fall back to...."""
    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.manifest import AgentsManifest

    manifest = AgentsManifest(agents={"test_agent": NativeAgentConfig(model="test")})

    pool = AgentPool(manifest)

    def simple_callback(message: str) -> str:
        return f"Test response: {message}"

    agent = Agent.from_callback(name="test_agent", callback=simple_callback, agent_pool=pool)
    # pool.register() removed; agent created from callback/config above
    pool._session_pool = None

    pool.storage.generate_session_id = MagicMock(return_value="session_fallback_001")  # type: ignore[assignment]

    manager = ACPSessionManager(pool=pool)
    mock_client = MagicMock()
    mock_acp_agent = MagicMock()

    with (
        patch("wolfharness_server.acp_server.session_manager.ACPSession") as mock_session_cls,
        patch("wolfharness_server.acp_server.session_manager.ClientCapabilities"),
    ):
        mock_session = MagicMock()
        mock_session.register_update_callback = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.initialize_mcp_servers = AsyncMock()
        mock_session_cls.return_value = mock_session

        cwd = tempfile.gettempdir()
        session_id = await manager.create_session(
            agent_name=agent.name,
            cwd=cwd,
            client=mock_client,
            acp_agent=mock_acp_agent,
            parent_session_id="some_parent_id",  # Will be ignored since pool.sessions is None
        )

    # Should have used the top-level path (computed project_id, no parent_id)
    # We can't check the store since there's no store configured, but
    # the session_id should be from generate_session_id (top-level path)
    assert session_id == "session_fallback_001"


# =============================================================================
# Red-flag tests: get_session() should return ACPSession even when
# _session_controller hasn't registered the session yet.
# =============================================================================


async def test_get_session_returns_session_when_not_yet_in_controller():
    """get_session() should return ACPSession from _acp_sessions even when not in controller.

    This is a red-flag test - it verifies the bug where get_session()
    returns None because _session_controller.get_session() returns None,
    even though the session IS in _acp_sessions.

    Bug: During session/new, create_session() adds the session to
    _acp_sessions but the orchestrator registers it with _session_controller
    asynchronously. If get_session() is called between these two steps,
    it returns None, causing create_task(session.send_available_commands_update())
    to be skipped — so available_commands_update is never sent.
    """
    pool, _agent, _sessions, _store = _make_pool_with_sessions()
    manager = ACPSessionManager(pool=pool)

    session_id = "sess-get-session-001"

    # Simulate what create_session() does: add to _acp_sessions but NOT
    # to _session_controller (which is populated asynchronously later).
    mock_session = MagicMock()
    mock_session.session_id = session_id
    manager._acp_sessions[session_id] = mock_session

    # _session_controller exists (from SessionPool) but doesn't have the session
    assert manager._session_controller is not None
    assert manager._session_controller.get_session(session_id) is None, (
        "Precondition: session should NOT be in controller yet"
    )

    # RED FLAG: get_session() should still return the session from _acp_sessions
    result = manager.get_session(session_id)
    assert result is not None, (
        f"get_session({session_id!r}) returned None even though the session "
        f"exists in _acp_sessions. This means create_task() calls in "
        f"new_session/load_session/resume_session are silently skipped, "
        f"and available_commands_update is never sent."
    )
    assert result is mock_session, (
        "get_session() should return the same session object from _acp_sessions"
    )


async def test_get_session_returns_none_when_not_in_either():
    """get_session() should return None when session is in neither _session_controller nor...."""
    pool, _agent, _sessions, _store = _make_pool_with_sessions()
    manager = ACPSessionManager(pool=pool)

    result = manager.get_session("nonexistent-session")
    assert result is None, "get_session() should return None for a session that doesn't exist"


async def test_get_session_returns_session_when_in_both():
    """get_session() should return the session when it exists in both _session_controller...."""
    pool, agent, sessions, _store = _make_pool_with_sessions()
    manager = ACPSessionManager(pool=pool)

    session_id = "sess-in-both-001"

    # Register in both
    mock_session = MagicMock()
    mock_session.session_id = session_id
    manager._acp_sessions[session_id] = mock_session

    # Register with session controller
    await sessions.create_session(
        session_id=session_id,
        agent_name=agent.name,
    )

    result = manager.get_session(session_id)
    assert result is not None, (
        f"get_session({session_id!r}) should return the session when it "
        f"exists in both _session_controller and _acp_sessions"
    )
    assert result is mock_session
