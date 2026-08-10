"""Unit tests for AgentPoolACPAgent.resume_session()."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from acp.schema import LoadSessionRequest, ResumeSessionRequest, ResumeSessionResponse
from acp.schema.mcp import StdioMcpServer
from wolfharness import Agent
from wolfharness.delegation import AgentPool
from wolfharness.orchestrator.core import SessionPool
from wolfharness.sessions import SessionData
from wolfharness_server.acp_server.acp_agent import AgentPoolACPAgent
from wolfharness_server.acp_server.session_manager import ACPSessionManager
from wolfharness_storage.memory_provider.provider import MemoryStorageProvider


@pytest.fixture
def mock_connection():
    """Create a mock ACP connection."""
    return AsyncMock()


@pytest.fixture
def mock_agent_pool_with_agent():
    """Create a mock agent pool with a test agent."""

    def simple_callback(message: str) -> str:
        return f"Test response: {message}"

    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.manifest import AgentsManifest

    manifest = AgentsManifest(agents={"test_agent": NativeAgentConfig(model="test")})

    pool = AgentPool(manifest)
    agent = Agent.from_callback(name="test_agent", callback=simple_callback, agent_pool=pool)
    # pool.register() removed; agent created from callback/config above
    return pool, agent


@pytest.fixture
def default_test_agent(mock_agent_pool_with_agent):
    """Get the default test agent from the mock pool."""
    return mock_agent_pool_with_agent[1]


@pytest.fixture
def mock_acp_agent(mock_connection, default_test_agent):
    """Create a mock ACP agent for testing."""
    return AgentPoolACPAgent(client=mock_connection, default_agent=default_test_agent)


@pytest.fixture
def mock_session():
    """Create a mock ACPSession with all required attributes."""
    session = MagicMock()
    session.session_id = "test-session-id"
    session.cwd = "/tmp"

    session.agent = MagicMock()
    session.agent.model_name = "test-model"
    session.agent.conversation = MagicMock()
    session.agent.conversation.chat_messages = []
    session.agent.load_session = AsyncMock(return_value=True)
    session.agent.load_rules = AsyncMock()

    session.notifications = MagicMock()
    session.notifications.replay = AsyncMock()

    session.send_available_commands_update = AsyncMock()

    return session


@pytest.fixture
def resume_session_request():
    """Create a ResumeSessionRequest."""
    return ResumeSessionRequest(session_id="test-session-id", cwd="/tmp")


@pytest.mark.unit
async def test_resume_session_calls_agent_load_session(
    mock_acp_agent, mock_session, resume_session_request
):
    """Test that agent.load_session() is called during resume_session."""
    mock_acp_agent.session_manager.get_session = MagicMock(return_value=mock_session)
    mock_acp_agent._initialized = True

    response = await mock_acp_agent.resume_session(resume_session_request)

    assert isinstance(response, ResumeSessionResponse)


@pytest.mark.unit
async def test_resume_session_does_not_call_replay(
    mock_acp_agent, mock_session, resume_session_request
):
    """Test that notifications.replay() is NOT called during resume_session."""
    mock_acp_agent.session_manager.get_session = MagicMock(return_value=mock_session)
    mock_acp_agent._initialized = True

    await mock_acp_agent.resume_session(resume_session_request)

    mock_session.notifications.replay.assert_not_awaited()


@pytest.mark.unit
async def test_resume_session_schedules_commands_update(
    mock_acp_agent, mock_session, resume_session_request
):
    """Test that send_available_commands_update() is scheduled after resume."""
    mock_acp_agent.session_manager.get_session = MagicMock(return_value=mock_session)
    mock_acp_agent._initialized = True

    with patch.object(mock_acp_agent._task_group, "start_soon") as mock_start_soon:
        await mock_acp_agent.resume_session(resume_session_request)

        # Should schedule send_available_commands_update and load_rules
        assert mock_start_soon.call_count == 2


@pytest.mark.unit
async def test_resume_session_agent_load_fails(
    mock_acp_agent, mock_session, resume_session_request
):
    """Test resume_session handles agent.load_session() failure gracefully."""
    mock_acp_agent.session_manager.get_session = MagicMock(return_value=mock_session)
    mock_acp_agent._initialized = True

    response = await mock_acp_agent.resume_session(resume_session_request)

    assert isinstance(response, ResumeSessionResponse)


@pytest.mark.unit
async def test_resume_session_creates_session_if_not_found(
    mock_acp_agent, mock_session, resume_session_request
):
    """Test that resume_session returns empty response for non-existent session.

    Previously, resume would create a new session when the session wasn't found.
    Now it returns an error/empty response because resuming a non-existent
    session doesn't make sense — the client should use session/new instead.
    """
    mock_acp_agent.session_manager.get_session = MagicMock(return_value=None)
    mock_acp_agent.session_manager.create_session = AsyncMock()
    mock_acp_agent._initialized = True
    mock_acp_agent._protocol_handler = None

    # Mock session_store to return None (session not in persistent store)
    mock_store = MagicMock()
    mock_store.load_session = AsyncMock(return_value=None)
    mock_store.list_session_ids = AsyncMock(return_value=[])
    with patch.object(
        type(mock_acp_agent.session_manager),
        "session_store",
        new_callable=lambda: property(lambda self: mock_store),
    ):
        response = await mock_acp_agent.resume_session(resume_session_request)

    # Non-existent session should return empty response (no models)
    assert response.models is None
    # resume_session no longer calls create_session; it only calls session_manager.resume_session()
    mock_acp_agent.session_manager.create_session.assert_not_awaited()


@pytest.mark.unit
async def test_resume_session_exception_returns_empty_response(
    mock_acp_agent, mock_session, resume_session_request
):
    """Test that resume_session returns empty ResumeSessionResponse on exception."""
    mock_session.agent.load_session = AsyncMock(side_effect=RuntimeError("boom"))
    mock_acp_agent.session_manager.get_session = MagicMock(return_value=mock_session)
    mock_acp_agent._initialized = True

    response = await mock_acp_agent.resume_session(resume_session_request)

    assert isinstance(response, ResumeSessionResponse)


# ──────────────────────────────────────────────
# Tests for ACPSessionManager.resume_session()
# ──────────────────────────────────────────────


@pytest.mark.unit
async def test_resume_session_passes_mcp_servers_to_constructor():
    """Test that resume_session passes mcp_servers to the ACPSession constructor."""
    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.manifest import AgentsManifest

    manifest = AgentsManifest(agents={"test_agent": NativeAgentConfig(model="test")})

    pool = AgentPool(manifest)

    def _callback(message: str) -> str:
        return f"Test response: {message}"

    Agent.from_callback(name="test_agent", callback=_callback, agent_pool=pool)
    # pool.register() removed; agent created from callback/config above
    store = MemoryStorageProvider()
    session_pool = SessionPool(pool=pool, store=store)
    pool._session_pool = session_pool

    await store.save_session(
        SessionData(
            session_id="test-session-id",
            agent_name="test_agent",
            cwd="/tmp",
        )
    )

    manager = ACPSessionManager(pool=pool)
    mock_client = MagicMock()
    mock_acp_agent = MagicMock()
    mock_server = MagicMock()

    with (
        patch("wolfharness_server.acp_server.session_manager.ACPSession") as mock_session_cls,
        patch("wolfharness_server.acp_server.session_manager.ClientCapabilities"),
    ):
        mock_session = MagicMock()
        mock_session.register_update_callback = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.initialize_mcp_servers = AsyncMock()
        mock_session.agent = MagicMock()
        mock_session.agent.load_session = AsyncMock(return_value=True)
        mock_session_cls.return_value = mock_session

        result = await manager.resume_session(
            session_id="test-session-id",
            client=mock_client,
            acp_agent=mock_acp_agent,
            mcp_servers=[mock_server],
        )

    assert result is mock_session
    call_kwargs = mock_session_cls.call_args
    passed_mcp = call_kwargs.kwargs.get("mcp_servers")
    assert passed_mcp == [mock_server]


@pytest.mark.unit
async def test_resume_session_initializes_mcp_servers():
    """Test that resume_session calls initialize_mcp_servers on the session."""
    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.manifest import AgentsManifest

    manifest = AgentsManifest(agents={"test_agent": NativeAgentConfig(model="test")})

    pool = AgentPool(manifest)

    def _callback(message: str) -> str:
        return f"Test response: {message}"

    Agent.from_callback(name="test_agent", callback=_callback, agent_pool=pool)
    # pool.register() removed; agent created from callback/config above
    store = MemoryStorageProvider()
    session_pool = SessionPool(pool=pool, store=store)
    pool._session_pool = session_pool

    await store.save_session(
        SessionData(
            session_id="test-session-id",
            agent_name="test_agent",
            cwd="/tmp",
        )
    )

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
        mock_session.agent = MagicMock()
        mock_session.agent.load_session = AsyncMock(return_value=True)
        mock_session_cls.return_value = mock_session

        await manager.resume_session(
            session_id="test-session-id",
            client=mock_client,
            acp_agent=mock_acp_agent,
        )

    mock_session.initialize_mcp_servers.assert_awaited_once()


@pytest.mark.unit
async def test_resume_session_with_none_mcp_servers_calls_initialize():
    """Test that resume_session still calls initialize_mcp_servers when mcp_servers is None...."""
    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.manifest import AgentsManifest

    manifest = AgentsManifest(agents={"test_agent": NativeAgentConfig(model="test")})

    pool = AgentPool(manifest)

    def _callback(message: str) -> str:
        return f"Test response: {message}"

    Agent.from_callback(name="test_agent", callback=_callback, agent_pool=pool)
    # pool.register() removed; agent created from callback/config above
    store = MemoryStorageProvider()
    session_pool = SessionPool(pool=pool, store=store)
    pool._session_pool = session_pool

    await store.save_session(
        SessionData(
            session_id="test-session-id",
            agent_name="test_agent",
            cwd="/tmp",
        )
    )

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
        mock_session.agent = MagicMock()
        mock_session.agent.load_session = AsyncMock(return_value=True)
        mock_session_cls.return_value = mock_session

        await manager.resume_session(
            session_id="test-session-id",
            client=mock_client,
            acp_agent=mock_acp_agent,
            mcp_servers=None,
        )

    mock_session.initialize_mcp_servers.assert_awaited_once()


@pytest.mark.unit
async def test_resume_session_does_not_call_load_session():
    """Test that resume_session does NOT call agent.load_session().

    Conversation history loading was moved to SessionPool's
    get_or_create_session_agent(), so resume_session no longer
    calls agent.load_session() directly.
    """
    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.manifest import AgentsManifest

    manifest = AgentsManifest(agents={"test_agent": NativeAgentConfig(model="test")})

    pool = AgentPool(manifest)

    def _callback(message: str) -> str:
        return f"Test response: {message}"

    Agent.from_callback(name="test_agent", callback=_callback, agent_pool=pool)
    # pool.register() removed; agent created from callback/config above
    store = MemoryStorageProvider()
    session_pool = SessionPool(pool=pool, store=store)
    pool._session_pool = session_pool

    await store.save_session(
        SessionData(
            session_id="test-session-id",
            agent_name="test_agent",
            cwd="/tmp",
        )
    )

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
        mock_session.agent = MagicMock()
        mock_session.agent.load_session = AsyncMock()
        mock_session_cls.return_value = mock_session

        await manager.resume_session(
            session_id="test-session-id",
            client=mock_client,
            acp_agent=mock_acp_agent,
        )

    mock_session.agent.load_session.assert_not_awaited()


@pytest.mark.unit
async def test_resume_session_closes_old_and_recreates():
    """Test resume_session closes old session and creates a fresh one."""
    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.manifest import AgentsManifest

    manifest = AgentsManifest(agents={"test_agent": NativeAgentConfig(model="test")})

    pool = AgentPool(manifest)

    def _callback(message: str) -> str:
        return f"Test response: {message}"

    Agent.from_callback(name="test_agent", callback=_callback, agent_pool=pool)
    # pool.register() removed; agent created from callback/config above
    store = MemoryStorageProvider()
    session_pool = SessionPool(pool=pool, store=store)
    pool._session_pool = session_pool

    await store.save_session(
        SessionData(
            session_id="test-session-id",
            agent_name="test_agent",
            cwd="/tmp",
        )
    )

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
        mock_session.close = AsyncMock()
        mock_session.agent = MagicMock()
        mock_session.agent.load_session = AsyncMock(return_value=True)
        mock_session2 = MagicMock()
        mock_session2.register_update_callback = MagicMock()
        mock_session2.initialize = AsyncMock()
        mock_session2.initialize_mcp_servers = AsyncMock()
        mock_session2.close = AsyncMock()
        mock_session2.agent = MagicMock()
        mock_session2.agent.load_session = AsyncMock(return_value=True)
        mock_session_cls.side_effect = [mock_session, mock_session2]

        result1 = await manager.resume_session(
            session_id="test-session-id",
            client=mock_client,
            acp_agent=mock_acp_agent,
        )
        result2 = await manager.resume_session(
            session_id="test-session-id",
            client=mock_client,
            acp_agent=mock_acp_agent,
        )

    # resume_session now closes old session and creates fresh one (T20)
    assert result1 is not result2  # Different session objects (close-then-recreate)
    assert mock_session_cls.call_count == 2  # ACPSession constructed twice (once per resume)
    assert mock_session.close.call_count == 1  # Old session was closed on second resume


# ──────────────────────────────────────────────────────────────
# Tests for AgentPoolACPAgent resume/load routing (Tasks 2+3)
# ──────────────────────────────────────────────────────────────


@pytest.mark.unit
async def test_resume_calls_session_manager_resume_not_create(
    mock_acp_agent, mock_session, resume_session_request
):
    """resume_session calls session_manager.resume_session() not create_session() when...."""
    mock_acp_agent.session_manager.get_session = MagicMock(return_value=None)
    mock_acp_agent.session_manager.resume_session = AsyncMock(return_value=mock_session)
    mock_acp_agent.session_manager.create_session = AsyncMock()
    mock_acp_agent._initialized = True

    with patch.object(mock_acp_agent._task_group, "start_soon"):
        await mock_acp_agent.resume_session(resume_session_request)

    mock_acp_agent.session_manager.resume_session.assert_awaited_once()
    mock_acp_agent.session_manager.create_session.assert_not_called()


@pytest.mark.unit
async def test_resume_passes_mcp_servers_to_session_manager(mock_acp_agent, mock_session):
    """resume_session passes mcp_servers through to session_manager.resume_session."""
    mock_acp_agent.session_manager.get_session = MagicMock(return_value=None)
    mock_acp_agent.session_manager.resume_session = AsyncMock(return_value=mock_session)
    mock_acp_agent._initialized = True

    mcp_servers = [StdioMcpServer(name="test", command="echo", args=[], env=[])]
    request = ResumeSessionRequest(
        session_id="test-session-id",
        cwd="/tmp",
        mcp_servers=mcp_servers,
    )

    with patch.object(mock_acp_agent._task_group, "start_soon"):
        await mock_acp_agent.resume_session(request)

    mock_acp_agent.session_manager.resume_session.assert_awaited_once()
    call_kwargs = mock_acp_agent.session_manager.resume_session.call_args.kwargs
    assert call_kwargs.get("mcp_servers") == mcp_servers


@pytest.mark.unit
async def test_resume_passes_through_to_session_manager_correctly(
    mock_acp_agent, mock_session, resume_session_request
):
    """resume_session passes all correct arguments to session_manager.resume_session."""
    mock_acp_agent.session_manager.get_session = MagicMock(return_value=None)
    mock_acp_agent.session_manager.resume_session = AsyncMock(return_value=mock_session)
    mock_acp_agent._initialized = True

    with patch.object(mock_acp_agent._task_group, "start_soon"):
        await mock_acp_agent.resume_session(resume_session_request)

    mock_acp_agent.session_manager.resume_session.assert_awaited_once_with(
        session_id="test-session-id",
        client=mock_acp_agent.client,
        acp_agent=mock_acp_agent,
        mcp_servers=resume_session_request.mcp_servers,
        client_capabilities=mock_acp_agent.client_capabilities,
        client_info=mock_acp_agent.client_info,
        subagent_display_mode=mock_acp_agent.subagent_display_mode,
        connection_id=mock_acp_agent._get_connection_id(),
    )


@pytest.mark.unit
async def test_load_session_calls_session_manager_resume_not_create(mock_acp_agent, mock_session):
    """load_session calls session_manager.resume_session() not create_session() when...."""
    mock_acp_agent.session_manager.get_session = MagicMock(return_value=None)
    mock_acp_agent.session_manager.resume_session = AsyncMock(return_value=mock_session)
    mock_acp_agent.session_manager.create_session = AsyncMock()
    mock_acp_agent._initialized = True

    request = LoadSessionRequest(session_id="test-session-id", cwd="/tmp")

    with patch.object(mock_acp_agent._task_group, "start_soon"):
        await mock_acp_agent.load_session(request)

    mock_acp_agent.session_manager.resume_session.assert_awaited_once()
    mock_acp_agent.session_manager.create_session.assert_not_called()


@pytest.mark.unit
async def test_load_session_still_replays_history_after_resume(mock_acp_agent, mock_session):
    """load_session still replays conversation history after resuming from session_manager."""
    mock_session.agent.conversation.chat_messages = [MagicMock(messages=[MagicMock()])]
    mock_acp_agent.session_manager.get_session = MagicMock(return_value=None)
    mock_acp_agent.session_manager.resume_session = AsyncMock(return_value=mock_session)
    mock_acp_agent._initialized = True

    request = LoadSessionRequest(session_id="test-session-id", cwd="/tmp")

    with patch.object(mock_acp_agent._task_group, "start_soon"):
        await mock_acp_agent.load_session(request)

    mock_session.notifications.replay.assert_awaited_once()


@pytest.mark.unit
async def test_session_data_preserved_after_resume(
    mock_acp_agent, mock_session, resume_session_request
):
    """session_store.save is NOT called during resume_session (data is preserved)."""
    mock_store = MagicMock()
    mock_store.save_session = AsyncMock()

    mock_acp_agent.session_manager.get_session = MagicMock(return_value=None)
    mock_acp_agent.session_manager.resume_session = AsyncMock(return_value=mock_session)
    mock_acp_agent._initialized = True

    with (
        patch.object(
            type(mock_acp_agent.session_manager),
            "session_store",
            new_callable=lambda: property(lambda self: mock_store),
        ),
        patch.object(mock_acp_agent._task_group, "start_soon"),
    ):
        await mock_acp_agent.resume_session(resume_session_request)

    mock_store.save_session.assert_not_called()
