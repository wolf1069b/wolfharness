"""Unit tests for AgentPoolACPAgent.load_session()."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from pydantic_ai import ModelRequest, UserPromptPart
import pytest

from acp.schema import (
    LoadSessionRequest,
    LoadSessionResponse,
    SessionMode,
    SessionModelState,
    SessionModeState,
)
from wolfharness import Agent
from wolfharness.delegation import AgentPool
from wolfharness.messaging import ChatMessage
from wolfharness_server.acp_server.acp_agent import AgentPoolACPAgent


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

    # Mock agent with conversation
    session.agent = MagicMock()
    session.agent.conversation = MagicMock()
    session.agent.conversation.chat_messages = []
    session.agent.load_session = AsyncMock(return_value=True)
    session.agent.load_rules = AsyncMock()

    # Mock notifications
    session.notifications = MagicMock()
    session.notifications.replay = AsyncMock()

    # Mock send_available_commands_update
    session.send_available_commands_update = AsyncMock()

    return session


@pytest.fixture
def load_session_request():
    """Create a LoadSessionRequest."""
    return LoadSessionRequest(session_id="test-session-id", cwd="/tmp")


@pytest.mark.unit
async def test_load_session_calls_agent_load_session(
    mock_acp_agent, mock_session, load_session_request
):
    """Test that session.agent.load_session() is called with the session ID.

    NOTE: load_session() no longer calls session.agent.load_session() directly.
    That call was moved into session_manager.resume_session().
    """
    mock_acp_agent.session_manager.get_session = MagicMock(return_value=mock_session)
    mock_acp_agent._initialized = True

    response = await mock_acp_agent.load_session(load_session_request)

    # load_session no longer calls agent.load_session directly; it's done in resume_session()
    assert isinstance(response, LoadSessionResponse)


@pytest.mark.unit
async def test_load_session_calls_replay_with_messages(
    mock_acp_agent, mock_session, load_session_request
):
    """Test that session.notifications.replay() is called with correct messages."""
    chat_msg = ChatMessage[str](
        content="Hello",
        role="user",
        messages=[ModelRequest(parts=[UserPromptPart(content="Hello")])],
    )
    mock_session.agent.conversation.chat_messages = [chat_msg]
    mock_acp_agent.session_manager.get_session = MagicMock(return_value=mock_session)
    mock_acp_agent._initialized = True

    await mock_acp_agent.load_session(load_session_request)

    mock_session.notifications.replay.assert_awaited_once()
    replay_args = mock_session.notifications.replay.call_args[0][0]
    assert len(replay_args) == 1
    assert isinstance(replay_args[0], ModelRequest)


@pytest.mark.unit
async def test_load_session_schedules_commands_update(
    mock_acp_agent, mock_session, load_session_request
):
    """Test that send_available_commands_update() is scheduled after load."""
    mock_acp_agent.session_manager.get_session = MagicMock(return_value=mock_session)
    mock_acp_agent._initialized = True

    with patch.object(mock_acp_agent._task_group, "start_soon") as mock_start_soon:
        await mock_acp_agent.load_session(load_session_request)

        # Should schedule send_available_commands_update and load_rules
        assert mock_start_soon.call_count == 2


@pytest.mark.unit
async def test_load_session_empty_conversation(mock_acp_agent, mock_session, load_session_request):
    """Test load_session with empty conversation (no messages to replay)."""
    mock_session.agent.conversation.chat_messages = []
    mock_acp_agent.session_manager.get_session = MagicMock(return_value=mock_session)
    mock_acp_agent._initialized = True

    await mock_acp_agent.load_session(load_session_request)

    # replay should NOT be called when there are no messages
    mock_session.notifications.replay.assert_not_awaited()


@pytest.mark.unit
async def test_load_session_agent_load_fails(mock_acp_agent, mock_session, load_session_request):
    """Test load_session failure handling when agent.load_session returns False."""
    mock_session.agent.load_session = AsyncMock(return_value=False)
    mock_acp_agent.session_manager.get_session = MagicMock(return_value=mock_session)
    mock_acp_agent._initialized = True

    response = await mock_acp_agent.load_session(load_session_request)

    assert isinstance(response, LoadSessionResponse)
    assert response.models is None
    assert response.modes is None
    assert response.config_options == []
    mock_session.notifications.replay.assert_not_awaited()


@pytest.mark.unit
async def test_load_session_response_contains_config_options(
    mock_acp_agent, mock_session, load_session_request
):
    """Test that LoadSessionResponse contains correct config_options."""

    class _MockMode:
        def __init__(self, mode_id: str, name: str, description: str) -> None:
            self.id = mode_id
            self.name = name
            self.description = description

    class _MockCategory:
        def __init__(self) -> None:
            self.id = "mode"
            self.name = "Mode"
            self.current_mode_id = "code"
            self.available_modes = [_MockMode("code", "Code", "Coding mode")]
            self.category = "mode"

    mock_session.agent.get_modes = AsyncMock(return_value=[_MockCategory()])
    mock_acp_agent.session_manager.get_session = MagicMock(return_value=mock_session)
    mock_acp_agent._initialized = True

    response = await mock_acp_agent.load_session(load_session_request)

    assert isinstance(response, LoadSessionResponse)
    assert len(response.config_options) > 0
    assert response.config_options[0].id == "mode"


@pytest.mark.unit
async def test_load_session_response_contains_config_options_empty_modes(
    mock_acp_agent, mock_session, load_session_request
):
    """Test that LoadSessionResponse handles empty modes correctly."""
    mock_session.agent.get_modes = AsyncMock(return_value=[])
    mock_acp_agent.session_manager.get_session = MagicMock(return_value=mock_session)
    mock_acp_agent._initialized = True

    response = await mock_acp_agent.load_session(load_session_request)

    assert isinstance(response, LoadSessionResponse)
    assert isinstance(response.config_options, list)


@pytest.mark.unit
async def test_load_session_creates_session_if_not_found(
    mock_acp_agent, mock_session, load_session_request
):
    """Test that load_session resumes via session_manager.resume_session() if session not found."""
    mock_acp_agent.session_manager.get_session = MagicMock(side_effect=[None, mock_session])
    mock_acp_agent.session_manager.resume_session = AsyncMock(return_value=mock_session)
    mock_acp_agent._initialized = True

    await mock_acp_agent.load_session(load_session_request)

    # load_session now calls resume_session(), not create_session()
    mock_acp_agent.session_manager.resume_session.assert_awaited_once()


@pytest.mark.unit
async def test_load_session_exception_returns_empty_response(
    mock_acp_agent, mock_session, load_session_request
):
    """Test that load_session returns empty LoadSessionResponse on exception."""
    mock_session.agent.load_session = AsyncMock(side_effect=RuntimeError("boom"))
    mock_acp_agent.session_manager.get_session = MagicMock(return_value=mock_session)
    mock_acp_agent._initialized = True

    response = await mock_acp_agent.load_session(load_session_request)

    assert isinstance(response, LoadSessionResponse)
    assert response.models is None
    assert response.modes is None


@pytest.mark.unit
async def test_load_session_with_nested_acp_agent(
    mock_acp_agent, mock_session, load_session_request
):
    """Test load_session with nested ACP agent populates models/modes from agent state."""
    from wolfharness.agents.acp_agent import ACPAgent

    nested_agent = MagicMock(spec=ACPAgent)
    nested_agent._state = MagicMock()
    nested_agent._state.modes = SessionModeState(
        available_modes=[SessionMode(id="chat", name="Chat", description="Chat mode")],
        current_mode_id="chat",
    )
    nested_agent._state.models = SessionModelState(available_models=[], current_model_id="gpt-4")
    nested_agent.load_session = AsyncMock(return_value=True)
    nested_agent.conversation = MagicMock()
    nested_agent.conversation.chat_messages = []
    nested_agent.get_modes = AsyncMock(return_value=[])
    nested_agent.host_context = mock_acp_agent.host_context

    mock_session.agent = nested_agent
    mock_acp_agent.session_manager.get_session = MagicMock(return_value=mock_session)
    mock_acp_agent._initialized = True

    response = await mock_acp_agent.load_session(load_session_request)

    assert isinstance(response, LoadSessionResponse)
    assert response.modes == nested_agent._state.modes
    assert response.models == nested_agent._state.models
