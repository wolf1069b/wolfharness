"""Unit tests for ACPAgent.load_session() client-side functionality."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from acp.schema import (
    AgentMessageChunk,
    LoadSessionResponse,
    SessionConfigOption,
    SessionModelState,
    SessionModeState,
    UserMessageChunk,
)
from wolfharness.agents.acp_agent import ACPAgent
from wolfharness.agents.acp_agent.session_state import ACPSessionState
from wolfharness.sessions.models import SessionData


@pytest.fixture
def mock_api():
    """Create a mock ACPAgentAPI."""
    api = MagicMock()
    api.load_session = AsyncMock()
    api.new_session = AsyncMock()
    api.prompt = AsyncMock()
    return api


@pytest.fixture
def mock_state():
    """Create an ACPSessionState for testing."""
    return ACPSessionState(session_id="")


@pytest.fixture
def acp_agent(mock_api, mock_state):
    """Create an ACPAgent with mocked internals."""
    agent = ACPAgent(command="test-cmd")
    agent._api = mock_api
    agent._state = mock_state
    agent._caps = None
    agent._cwd = "/tmp"
    agent._extra_mcp_servers = []
    agent._sessions_cache = None
    return agent


@pytest.mark.unit
async def test_load_session_calls_api_with_correct_params(acp_agent, mock_api):
    """Test that _api.load_session() is called with correct parameters."""
    mock_api.load_session = AsyncMock(
        return_value=LoadSessionResponse(
            models=SessionModelState(available_models=[], current_model_id="gpt-4"),
            modes=SessionModeState(available_modes=[], current_mode_id="chat"),
        )
    )

    await acp_agent.load_session("sess-123")

    mock_api.load_session.assert_awaited_once()
    call_args = mock_api.load_session.call_args[0]
    assert call_args[0] == "sess-123"
    assert call_args[1] == "/tmp"


@pytest.mark.unit
async def test_load_session_captures_updates_during_load(acp_agent, mock_api, mock_state):
    """Test that updates are captured during load (start_load/finish_load)."""

    # Simulate updates being added during load
    async def side_effect(*args, **kwargs):
        # During load, the state should be collecting updates
        assert mock_state.is_loading
        return LoadSessionResponse()

    mock_api.load_session = AsyncMock(side_effect=side_effect)

    await acp_agent.load_session("sess-123")

    # After load, is_loading should be False
    assert not mock_state.is_loading


@pytest.mark.unit
async def test_load_session_converts_updates_to_chat_messages(acp_agent, mock_api, mock_state):
    """Test that captured updates are converted to chat_messages."""

    # Add a mock update to the state during load
    async def side_effect(*args, **kwargs):
        mock_state.add_update(UserMessageChunk.text("Hello from history"))
        mock_state.add_update(AgentMessageChunk.text("Agent response"))
        return LoadSessionResponse()

    mock_api.load_session = AsyncMock(side_effect=side_effect)

    result = await acp_agent.load_session("sess-123")

    assert result is not None
    assert len(acp_agent.conversation.chat_messages) == 2


@pytest.mark.unit
async def test_load_session_updates_sdk_session_id(acp_agent, mock_api):
    """Test that sdk_session_id is updated after successful load."""
    mock_api.load_session = AsyncMock(
        return_value=LoadSessionResponse(
            models=SessionModelState(available_models=[], current_model_id="gpt-4"),
        )
    )

    await acp_agent.load_session("sess-123")

    assert acp_agent._sdk_session_id == "sess-123"


@pytest.mark.unit
async def test_load_session_updates_state_models(acp_agent, mock_api, mock_state):
    """Test that state.models is updated from response."""
    models = SessionModelState(available_models=[], current_model_id="gpt-4o")
    mock_api.load_session = AsyncMock(return_value=LoadSessionResponse(models=models))

    await acp_agent.load_session("sess-123")

    assert mock_state.models == models
    assert mock_state.current_model_id == "gpt-4o"


@pytest.mark.unit
async def test_load_session_updates_state_modes(acp_agent, mock_api, mock_state):
    """Test that state.modes is updated from response."""
    modes = SessionModeState(available_modes=[], current_mode_id="code")
    mock_api.load_session = AsyncMock(return_value=LoadSessionResponse(modes=modes))

    await acp_agent.load_session("sess-123")

    assert mock_state.modes == modes


@pytest.mark.unit
async def test_load_session_updates_state_config_options(acp_agent, mock_api, mock_state):
    """Test that state.config_options is updated from response."""
    config_options = [
        SessionConfigOption(
            id="mode",
            name="Mode",
            description="Session mode",
            category="mode",
            current_value="chat",
            options=[],
        )
    ]
    mock_api.load_session = AsyncMock(
        return_value=LoadSessionResponse(config_options=config_options)
    )

    await acp_agent.load_session("sess-123")

    assert mock_state.config_options == config_options


@pytest.mark.unit
async def test_load_session_returns_session_data(acp_agent, mock_api):
    """Test that load_session returns SessionData on success."""
    mock_api.load_session = AsyncMock(return_value=LoadSessionResponse())

    result = await acp_agent.load_session("sess-123")

    assert isinstance(result, SessionData)
    assert result.session_id == "sess-123"
    assert result.agent_name == acp_agent.name


@pytest.mark.unit
async def test_load_session_returns_none_when_not_connected(acp_agent, mock_api):
    """Test that load_session returns None when API is not connected."""
    acp_agent._api = None

    result = await acp_agent.load_session("sess-123")

    assert result is None


@pytest.mark.unit
async def test_load_session_returns_none_when_state_not_initialized(acp_agent, mock_api):
    """Test that load_session returns None when state is not initialized."""
    acp_agent._state = None

    result = await acp_agent.load_session("sess-123")

    assert result is None


@pytest.mark.unit
async def test_load_session_returns_none_on_api_exception(acp_agent, mock_api, mock_state):
    """Test failure handling when API raises exception."""
    mock_api.load_session = AsyncMock(side_effect=RuntimeError("API error"))

    result = await acp_agent.load_session("sess-123")

    assert result is None
    # is_loading should be reset even on failure
    assert not mock_state.is_loading


@pytest.mark.unit
async def test_load_session_clears_existing_chat_messages(acp_agent, mock_api, mock_state):
    """Test that existing chat_messages are cleared before adding loaded ones."""
    # Pre-populate with a message
    existing_msg = MagicMock()
    acp_agent.conversation.chat_messages.append(existing_msg)

    async def side_effect(*args, **kwargs):
        mock_state.add_update(UserMessageChunk.text("Loaded message"))
        return LoadSessionResponse()

    mock_api.load_session = AsyncMock(side_effect=side_effect)

    await acp_agent.load_session("sess-123")

    assert len(acp_agent.conversation.chat_messages) == 1
    assert acp_agent.conversation.chat_messages[0].content == "Loaded message"


@pytest.mark.unit
async def test_load_session_with_no_updates(acp_agent, mock_api):
    """Test load_session when response has no updates to replay."""
    mock_api.load_session = AsyncMock(return_value=LoadSessionResponse())

    result = await acp_agent.load_session("sess-123")

    assert isinstance(result, SessionData)
    assert len(acp_agent.conversation.chat_messages) == 0


@pytest.mark.unit
async def test_load_session_uses_session_cache_when_available(acp_agent, mock_api):
    """Test that load_session returns cached session info when available."""
    cached = SessionData(
        session_id="sess-123",
        agent_name="test_agent",
        cwd="/cached",
        metadata={"title": "Cached Session"},
    )
    acp_agent._sessions_cache = [cached]
    mock_api.load_session = AsyncMock(return_value=LoadSessionResponse())

    result = await acp_agent.load_session("sess-123")

    assert result is not None
    assert result.session_id == "sess-123"
