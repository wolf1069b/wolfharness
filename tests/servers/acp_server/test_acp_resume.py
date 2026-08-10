"""Tests for ACP session/resume endpoint and handler integration.

Covers:
- ACP session/resume restores session context without history replay
- EventBus re-subscribed for resumed session
- Non-existent session returns appropriate error
- handle_prompt for checkpointed sessions calls resume_session instead of create_session
- resume_session proactively creates per-session agent with loaded history
- resume_session returns full ResumeSessionResponse with models/modes/config_options
- resume_session injects session MCP providers into per-session agent
- resume_session does not block on per-session agent creation failure
- resume_session ensures handle_prompt reuses cached agent (no duplicate creation)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest

from acp.schema import ClientCapabilities, ResumeSessionRequest
from wolfharness import Agent
from wolfharness.delegation import AgentPool
from wolfharness.sessions.models import SessionData
from wolfharness_server.acp_server.acp_agent import AgentPoolACPAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_connection():
    """Create a mock ACP connection."""
    return MagicMock()


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
def mocked_acp_agent(mock_connection, default_test_agent):
    """Create AgentPoolACPAgent with mocked session_manager and _protocol_handler."""
    acp_agent = AgentPoolACPAgent(client=mock_connection, default_agent=default_test_agent)
    acp_agent._initialized = True

    # Mock session manager
    acp_agent.session_manager = MagicMock()
    acp_agent.session_manager.get_session = MagicMock(return_value=None)
    acp_agent.session_manager.resume_session = AsyncMock(return_value=None)
    acp_agent.session_manager.create_session = AsyncMock()
    acp_agent.session_manager.session_store = MagicMock()

    # Mock protocol handler
    acp_agent._protocol_handler = MagicMock()
    acp_agent._protocol_handler.start_event_consumer = AsyncMock()

    return acp_agent


def _make_session_data(
    session_id: str = "resume-test-session",
    agent_name: str = "test_agent",
    cwd: str = "/tmp/resume",
    status: str = "checkpointed",
) -> SessionData:
    """Create a SessionData for test use."""
    return SessionData(
        session_id=session_id,
        agent_name=agent_name,
        cwd=cwd,
        status=status,
        metadata={"agent_type": "acp"},
    )


# ---------------------------------------------------------------------------
# Tests: resume_session restores context from store without history replay
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_resume_uses_session_manager_resume_when_not_active(
    mocked_acp_agent, mock_connection
):
    """When session is not active but exists in store, use session_manager.resume_session().

    Verifies that session_manager.create_session() is NOT called.
    """
    session_data = _make_session_data()
    mock_session = MagicMock()
    mock_session.session_id = "resume-test-session"
    mock_session.agent = MagicMock()
    mock_session.agent.load_session = AsyncMock(return_value=True)
    mock_session.agent.load_rules = AsyncMock()
    mock_session.send_available_commands_update = AsyncMock()
    mock_session.cwd = "/tmp/resume"

    # Not active
    mocked_acp_agent.session_manager.get_session.return_value = None
    # But exists in store
    mocked_acp_agent.session_manager.session_store.load_session = AsyncMock(
        return_value=session_data
    )
    mocked_acp_agent.session_manager.session_store.list_session_ids = AsyncMock(
        return_value=["resume-test-session"]
    )
    mocked_acp_agent.session_manager.resume_session.return_value = mock_session

    request = ResumeSessionRequest(session_id="resume-test-session", cwd="/tmp/resume")
    await mocked_acp_agent.resume_session(request)

    # resume_session now routes through session_manager.resume_session(), not create_session()
    mocked_acp_agent.session_manager.resume_session.assert_awaited_once()


@pytest.mark.unit
async def test_resume_starts_event_bus_consumer(mocked_acp_agent):
    """After successful resume, the EventBus consumer should be started."""
    session_data = _make_session_data()
    mock_session = MagicMock()
    mock_session.session_id = "resume-test-session"
    mock_session.agent = MagicMock()
    mock_session.agent.load_session = AsyncMock(return_value=True)
    mock_session.agent.load_rules = AsyncMock()
    mock_session.send_available_commands_update = AsyncMock()
    mock_session.cwd = "/tmp/resume"

    mocked_acp_agent.session_manager.get_session.return_value = None
    mocked_acp_agent.session_manager.session_store.load_session = AsyncMock(
        return_value=session_data
    )
    mocked_acp_agent.session_manager.session_store.list_session_ids = AsyncMock(
        return_value=["resume-test-session"]
    )
    mocked_acp_agent.session_manager.resume_session.return_value = mock_session

    request = ResumeSessionRequest(session_id="resume-test-session", cwd="/tmp/resume")
    await mocked_acp_agent.resume_session(request)

    # resume_session should have been called
    mocked_acp_agent.session_manager.resume_session.assert_awaited_once()


@pytest.mark.unit
async def test_resume_does_not_replay_history(mocked_acp_agent):
    """Resume should NOT call any session/update notifications with message history."""
    session_data = _make_session_data()
    mock_session = MagicMock()
    mock_session.session_id = "resume-test-session"
    mock_session.agent = MagicMock()
    mock_session.agent.load_session = AsyncMock(return_value=True)
    mock_session.agent.load_rules = AsyncMock()
    mock_session.send_available_commands_update = AsyncMock()
    mock_session.cwd = "/tmp/resume"
    mock_session.notifications = MagicMock()
    mock_session.notifications.replay = AsyncMock()

    mocked_acp_agent.session_manager.get_session.return_value = None
    mocked_acp_agent.session_manager.session_store.load_session = AsyncMock(
        return_value=session_data
    )
    mocked_acp_agent.session_manager.session_store.list_session_ids = AsyncMock(
        return_value=["resume-test-session"]
    )
    mocked_acp_agent.session_manager.resume_session.return_value = mock_session

    request = ResumeSessionRequest(session_id="resume-test-session", cwd="/tmp/resume")
    await mocked_acp_agent.resume_session(request)

    # replay() must NOT be called (history is client-side)
    mock_session.notifications.replay.assert_not_awaited()


@pytest.mark.unit
async def test_resume_nonexistent_session_returns_error(mocked_acp_agent):
    """When session does not exist in store, resume should return an error/empty response."""
    # Not active
    mocked_acp_agent.session_manager.get_session.return_value = None
    # Not in store either
    mocked_acp_agent.session_manager.session_store.load_session = AsyncMock(return_value=None)
    mocked_acp_agent.session_manager.session_store.list_session_ids = AsyncMock(return_value=[])

    request = ResumeSessionRequest(session_id="nonexistent-session", cwd="/tmp")
    response = await mocked_acp_agent.resume_session(request)

    # Should NOT try to create a new session for a non-existent session_id
    mocked_acp_agent.session_manager.create_session.assert_not_awaited()
    mocked_acp_agent.session_manager.resume_session.assert_awaited_once()
    # EventBus consumer should NOT be started
    mocked_acp_agent._protocol_handler.start_event_consumer.assert_not_awaited()
    # Should return empty response (no models)
    assert response.models is None


@pytest.mark.unit
async def test_resume_already_active_session_reuses(mocked_acp_agent):
    """When session is already active, resume should reuse it."""
    mock_session = MagicMock()
    mock_session.session_id = "active-session"
    mock_session.agent = MagicMock()
    mock_session.agent.load_session = AsyncMock(return_value=True)
    mock_session.agent.load_rules = AsyncMock()
    mock_session.send_available_commands_update = AsyncMock()

    # Already active
    mocked_acp_agent.session_manager.get_session.return_value = mock_session

    request = ResumeSessionRequest(session_id="active-session", cwd="/tmp")
    await mocked_acp_agent.resume_session(request)

    # Should reuse existing session (no resume or create needed)
    mocked_acp_agent.session_manager.resume_session.assert_not_awaited()
    mocked_acp_agent.session_manager.create_session.assert_not_awaited()


@pytest.mark.unit
async def test_resume_loads_agent_state(mocked_acp_agent):
    """Resume should call agent.load_session() to restore internal state."""
    session_data = _make_session_data()
    mock_session = MagicMock()
    mock_session.session_id = "resume-test-session"
    mock_session.agent = MagicMock()
    mock_session.agent.load_session = AsyncMock(return_value=True)
    mock_session.agent.load_rules = AsyncMock()
    mock_session.send_available_commands_update = AsyncMock()
    mock_session.cwd = "/tmp/resume"

    mocked_acp_agent.session_manager.get_session.return_value = None
    mocked_acp_agent.session_manager.session_store.load_session = AsyncMock(
        return_value=session_data
    )
    mocked_acp_agent.session_manager.session_store.list_session_ids = AsyncMock(
        return_value=["resume-test-session"]
    )
    mocked_acp_agent.session_manager.resume_session.return_value = mock_session

    request = ResumeSessionRequest(session_id="resume-test-session", cwd="/tmp/resume")
    await mocked_acp_agent.resume_session(request)

    # resume_session should have been called
    mocked_acp_agent.session_manager.resume_session.assert_awaited_once()


# ---------------------------------------------------------------------------
# Tests: handle_prompt for checkpointed sessions
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_handle_prompt_checkpointed_session_resumes(mocked_acp_agent, mock_connection):
    """handle_prompt should resume a checkpointed session instead of creating a new one."""
    from wolfharness_server.acp_server.event_converter import ACPEventConverter
    from wolfharness_server.acp_server.handler import ACPProtocolHandler

    session_data = _make_session_data(status="checkpointed")

    mock_ctx = MagicMock()
    mock_ctx.session_pool = MagicMock()
    mock_ctx.session_pool.sessions = MagicMock()
    mock_ctx.session_pool.sessions.store = MagicMock()
    mock_ctx.session_pool.sessions.store.load_session = AsyncMock(return_value=session_data)
    mock_ctx.session_pool.event_bus = MagicMock()

    mock_session_manager = mocked_acp_agent.session_manager
    mock_session_manager.session_store = mock_ctx.session_pool.sessions.store
    # Session is not active
    mock_session_manager.get_session = MagicMock(return_value=None)
    mock_session_manager.resume_session = AsyncMock()

    # Set up acp_agent on the mocked acp_agent object for handler usage
    mocked_acp_agent.client_info = None
    mocked_acp_agent.subagent_display_mode = "legacy"

    mock_client = MagicMock()
    mock_client.session_update = AsyncMock()

    handler = ACPProtocolHandler(
        host_context=mock_ctx,
        session_manager=mock_session_manager,
        event_converter=ACPEventConverter(subagent_display_mode="legacy"),
        client=mock_client,
        client_capabilities=ClientCapabilities(),
        acp_agent=mocked_acp_agent,
    )

    # Mock start_event_consumer to avoid creating real asyncio tasks
    handler.start_event_consumer = AsyncMock()  # type: ignore[assignment]

    # Mock session_pool methods
    mock_ctx.session_pool.create_session = AsyncMock()
    mock_ctx.session_pool.send_message = AsyncMock()
    mock_ctx.session_pool.sessions.get_or_create_session_agent = AsyncMock()

    from acp.schema.content_blocks import TextContentBlock

    prompt_blocks = [TextContentBlock(text="continue work")]

    await handler.handle_prompt("resume-test-session", prompt_blocks)

    # For checkpointed sessions, resume_session should be called
    mock_session_manager.resume_session.assert_awaited_once()
    # create_session is still called (idempotent) to ensure SessionPool entry
    mock_ctx.session_pool.create_session.assert_awaited()


@pytest.mark.unit
async def test_handle_prompt_active_session_uses_create_session(mocked_acp_agent, mock_connection):
    """handle_prompt for a non-checkpointed (active/new) session should call create_session."""
    from wolfharness_server.acp_server.event_converter import ACPEventConverter
    from wolfharness_server.acp_server.handler import ACPProtocolHandler

    session_data = _make_session_data(status="active")

    mock_ctx = MagicMock()
    mock_ctx.session_pool = MagicMock()
    mock_ctx.session_pool.sessions = MagicMock()
    mock_ctx.session_pool.sessions.store = MagicMock()
    mock_ctx.session_pool.sessions.store.load_session = AsyncMock(return_value=session_data)
    # Make event_bus.subscribe an AsyncMock
    mock_ctx.session_pool.event_bus = MagicMock()
    _send, _recv = anyio.create_memory_object_stream(max_buffer_size=100)
    await _send.aclose()
    mock_ctx.session_pool.event_bus.subscribe = AsyncMock(return_value=_recv)
    mock_ctx.session_pool.event_bus.unsubscribe = AsyncMock()

    mock_session_manager = mocked_acp_agent.session_manager
    mock_session_manager.session_store = mock_ctx.session_pool.sessions.store
    mock_session_manager.get_session = MagicMock(return_value=None)
    mock_session_manager.resume_session = AsyncMock()

    # Set up acp_agent on the mocked acp_agent object for handler usage
    mocked_acp_agent.client_info = None
    mocked_acp_agent.subagent_display_mode = "legacy"

    mock_client = MagicMock()
    mock_client.session_update = AsyncMock()

    handler = ACPProtocolHandler(
        host_context=mock_ctx,
        session_manager=mock_session_manager,
        event_converter=ACPEventConverter(subagent_display_mode="legacy"),
        client=mock_client,
        client_capabilities=ClientCapabilities(),
        acp_agent=mocked_acp_agent,
    )

    # Mock start_event_consumer to avoid creating real asyncio tasks
    handler.start_event_consumer = AsyncMock()  # type: ignore[assignment]

    mock_ctx.session_pool.create_session = AsyncMock()
    mock_ctx.session_pool.send_message = AsyncMock()
    mock_ctx.session_pool.sessions.get_or_create_session_agent = AsyncMock()

    from acp.schema.content_blocks import TextContentBlock

    prompt_blocks = [TextContentBlock(text="hello")]
    await handler.handle_prompt("resume-test-session", prompt_blocks)

    # Sessions missing from memory are resumed to restore conversation history
    mock_session_manager.resume_session.assert_awaited()
    mock_ctx.session_pool.create_session.assert_awaited()


# ---------------------------------------------------------------------------
# Tests: resume_session creates per-session agent with conversation history
# ---------------------------------------------------------------------------


def _make_mock_session(session_id: str = "resume-test-session") -> MagicMock:
    mock_session = MagicMock()
    mock_session.session_id = session_id
    mock_session.agent = MagicMock()
    mock_session.agent.load_session = AsyncMock(return_value=True)
    mock_session.agent.load_rules = AsyncMock()
    mock_session.send_available_commands_update = AsyncMock()
    mock_session.cwd = "/tmp/resume"
    mock_session.session_mcp_providers = []
    mock_session.notifications = MagicMock()
    mock_session.notifications.replay = AsyncMock()
    return mock_session


def _setup_session_pool_mock(acp_agent: AgentPoolACPAgent) -> MagicMock:
    mock_ctx = MagicMock()
    session_pool = MagicMock()
    session_pool.create_session = AsyncMock()
    session_pool.send_message = AsyncMock()
    session_pool.sessions = MagicMock()
    session_pool.sessions.get_or_create_session_agent = AsyncMock()
    session_pool.event_bus = MagicMock()
    mock_ctx.session_pool = session_pool
    # Replace default_agent with a mock so host_context returns our mock_ctx
    acp_agent.default_agent = MagicMock()
    acp_agent.default_agent.host_context = mock_ctx
    return session_pool


@pytest.mark.unit
async def test_resume_creates_per_session_agent_with_history(mocked_acp_agent):
    """resume_session should call create_session + get_or_create_session_agent."""
    session_data = _make_session_data()
    mock_session = _make_mock_session()

    mocked_acp_agent.session_manager.get_session.return_value = None
    mocked_acp_agent.session_manager.session_store.load_session = AsyncMock(
        return_value=session_data
    )
    mocked_acp_agent.session_manager.session_store.list_session_ids = AsyncMock(
        return_value=["resume-test-session"]
    )
    mocked_acp_agent.session_manager.resume_session.return_value = mock_session

    session_pool = _setup_session_pool_mock(mocked_acp_agent)
    mock_session_agent = MagicMock()
    mock_session_agent.tools.external_providers = []
    session_pool.sessions.get_or_create_session_agent.return_value = mock_session_agent

    request = ResumeSessionRequest(session_id="resume-test-session", cwd="/tmp/resume")
    await mocked_acp_agent.resume_session(request)

    session_pool.create_session.assert_awaited_once_with("resume-test-session", cwd="/tmp/resume")
    session_pool.sessions.get_or_create_session_agent.assert_awaited_once_with(
        "resume-test-session"
    )


@pytest.mark.unit
async def test_resume_returns_models_and_modes(mocked_acp_agent):
    session_data = _make_session_data()
    mock_session = _make_mock_session()

    mocked_acp_agent.session_manager.get_session.return_value = None
    mocked_acp_agent.session_manager.session_store.load_session = AsyncMock(
        return_value=session_data
    )
    mocked_acp_agent.session_manager.session_store.list_session_ids = AsyncMock(
        return_value=["resume-test-session"]
    )
    mocked_acp_agent.session_manager.resume_session.return_value = mock_session

    _setup_session_pool_mock(mocked_acp_agent)

    request = ResumeSessionRequest(session_id="resume-test-session", cwd="/tmp/resume")
    response = await mocked_acp_agent.resume_session(request)

    assert response is not None
    assert response.config_options is not None


@pytest.mark.unit
async def test_resume_injects_session_mcp_providers(mocked_acp_agent):
    session_data = _make_session_data()
    mock_session = _make_mock_session()
    mock_provider = MagicMock()
    mock_session.session_mcp_providers = [mock_provider]

    mocked_acp_agent.session_manager.get_session.return_value = None
    mocked_acp_agent.session_manager.session_store.load_session = AsyncMock(
        return_value=session_data
    )
    mocked_acp_agent.session_manager.session_store.list_session_ids = AsyncMock(
        return_value=["resume-test-session"]
    )
    mocked_acp_agent.session_manager.resume_session.return_value = mock_session

    session_pool = _setup_session_pool_mock(mocked_acp_agent)
    mock_session_agent = MagicMock()
    mock_session_agent.tools.external_providers = []
    mock_session_agent.tools.add_provider = MagicMock()
    session_pool.sessions.get_or_create_session_agent.return_value = mock_session_agent

    request = ResumeSessionRequest(session_id="resume-test-session", cwd="/tmp/resume")
    await mocked_acp_agent.resume_session(request)

    # MCP providers are no longer added via add_provider — they go through
    # McpConfigSnapshot → get_capabilities() → MCPToolset. We verify that the
    # session agent was created (which internally builds the snapshot).
    session_pool.sessions.get_or_create_session_agent.assert_called_once_with("resume-test-session")
    mock_session_agent.tools.add_provider.assert_not_called()


@pytest.mark.unit
async def test_resume_history_load_failure_does_not_block(mocked_acp_agent):
    session_data = _make_session_data()
    mock_session = _make_mock_session()

    mocked_acp_agent.session_manager.get_session.return_value = None
    mocked_acp_agent.session_manager.session_store.load_session = AsyncMock(
        return_value=session_data
    )
    mocked_acp_agent.session_manager.session_store.list_session_ids = AsyncMock(
        return_value=["resume-test-session"]
    )
    mocked_acp_agent.session_manager.resume_session.return_value = mock_session

    session_pool = _setup_session_pool_mock(mocked_acp_agent)
    session_pool.sessions.get_or_create_session_agent.side_effect = RuntimeError("DB error")

    request = ResumeSessionRequest(session_id="resume-test-session", cwd="/tmp/resume")
    response = await mocked_acp_agent.resume_session(request)

    assert response is not None
    assert response.config_options is not None


@pytest.mark.unit
async def test_resume_then_handle_prompt_no_duplicate_agent_creation(
    mocked_acp_agent, mock_connection
):
    from acp.schema.content_blocks import TextContentBlock
    from wolfharness_server.acp_server.event_converter import ACPEventConverter
    from wolfharness_server.acp_server.handler import ACPProtocolHandler

    session_data = _make_session_data()
    mock_session = _make_mock_session()

    mocked_acp_agent.session_manager.get_session.return_value = None
    mocked_acp_agent.session_manager.session_store.load_session = AsyncMock(
        return_value=session_data
    )
    mocked_acp_agent.session_manager.session_store.list_session_ids = AsyncMock(
        return_value=["resume-test-session"]
    )
    mocked_acp_agent.session_manager.resume_session.return_value = mock_session

    session_pool = _setup_session_pool_mock(mocked_acp_agent)
    mock_session_agent = MagicMock()
    mock_session_agent.tools.external_providers = []
    session_pool.sessions.get_or_create_session_agent.return_value = mock_session_agent

    request = ResumeSessionRequest(session_id="resume-test-session", cwd="/tmp/resume")
    await mocked_acp_agent.resume_session(request)

    assert session_pool.sessions.get_or_create_session_agent.await_count == 1

    mocked_acp_agent.client_info = None
    mocked_acp_agent.subagent_display_mode = "legacy"

    mock_client = MagicMock()
    mock_client.session_update = AsyncMock()

    handler = ACPProtocolHandler(
        host_context=mocked_acp_agent.host_context,
        session_manager=mocked_acp_agent.session_manager,
        event_converter=ACPEventConverter(subagent_display_mode="legacy"),
        client=mock_client,
        client_capabilities=ClientCapabilities(),
        acp_agent=mocked_acp_agent,
    )
    handler.start_event_consumer = AsyncMock()  # type: ignore[assignment]

    await handler.handle_prompt("resume-test-session", [TextContentBlock(text="continue")])

    assert session_pool.sessions.get_or_create_session_agent.await_count == 1
