"""Tests for ACPClientHandler.session_update() behavior.

These tests document how various session update types are handled,
including a known bug with AvailableCommandsUpdate during load_session.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from acp.schema import (
    AgentMessageChunk,
    AvailableCommand,
    AvailableCommandsUpdate,
    ConfigOptionUpdate,
    CurrentModelUpdate,
    CurrentModeUpdate,
    SessionConfigOption,
    SessionMode,
    SessionModelState,
    SessionModeState,
    SessionNotification,
    ToolCallStart,
    UserMessageChunk,
)
from wolfharness.agents.acp_agent import ACPAgent
from wolfharness.agents.acp_agent.client_handler import ACPClientHandler
from wolfharness.agents.acp_agent.session_state import ACPSessionState


def _mock_agent() -> MagicMock:
    """Create a mocked ACPAgent with required attributes."""
    agent = MagicMock(spec=ACPAgent)
    agent.state_updated = MagicMock()
    agent.state_updated.emit = AsyncMock()
    agent.update_state = AsyncMock()
    agent.command_store = MagicMock()
    agent.command_store.register_command = MagicMock()
    agent.command_store.list_commands = MagicMock(return_value=[])
    # Mock _init_request with client_capabilities
    agent._init_request = MagicMock()
    agent._init_request.client_capabilities = MagicMock()
    agent._init_request.client_capabilities.fs = None
    agent._init_request.client_capabilities.terminal = False
    return agent


@pytest.fixture
def mock_agent() -> MagicMock:
    """Provide a mocked ACPAgent."""
    return _mock_agent()


@pytest.fixture
def session_state() -> ACPSessionState:
    """Provide a fresh ACPSessionState."""
    return ACPSessionState(session_id="test-session")


@pytest.fixture
def handler(mock_agent: MagicMock, session_state: ACPSessionState) -> ACPClientHandler:
    """Provide an ACPClientHandler with mocked agent and real state."""
    return ACPClientHandler(agent=mock_agent, state=session_state)  # type: ignore[reportAbstractUsage]


# =============================================================================
# Stream data updates (should be added to state.updates)
# =============================================================================


@pytest.mark.unit
async def test_user_message_chunk_added_to_updates(
    handler: ACPClientHandler, session_state: ACPSessionState
) -> None:
    """UserMessageChunk should be added to state.updates."""
    chunk = UserMessageChunk.text("hello")
    notification = SessionNotification(session_id="test-session", update=chunk)

    await handler.session_update(notification)

    assert len(session_state.updates) == 1
    assert session_state.updates[0] == chunk


@pytest.mark.unit
async def test_agent_message_chunk_added_to_updates(
    handler: ACPClientHandler, session_state: ACPSessionState
) -> None:
    """AgentMessageChunk should be added to state.updates."""
    chunk = AgentMessageChunk.text("response")
    notification = SessionNotification(session_id="test-session", update=chunk)

    await handler.session_update(notification)

    assert len(session_state.updates) == 1
    assert session_state.updates[0] == chunk


@pytest.mark.unit
async def test_tool_call_start_added_to_updates(
    handler: ACPClientHandler, session_state: ACPSessionState
) -> None:
    """ToolCallStart should be added to state.updates."""
    tool_call = ToolCallStart(tool_call_id="tc-1", title="Reading file")
    notification = SessionNotification(session_id="test-session", update=tool_call)

    await handler.session_update(notification)

    assert len(session_state.updates) == 1
    assert session_state.updates[0] == tool_call


# =============================================================================
# AvailableCommandsUpdate handling
# =============================================================================


@pytest.mark.unit
async def test_available_commands_stored_in_state(
    handler: ACPClientHandler, session_state: ACPSessionState
) -> None:
    """AvailableCommandsUpdate should be stored in state.available_commands."""
    cmd = AvailableCommand.create(name="test-cmd", description="A test command")
    update = AvailableCommandsUpdate(available_commands=[cmd])
    notification = SessionNotification(session_id="test-session", update=update)

    await handler.session_update(notification)

    assert session_state.available_commands is not None
    assert len(session_state.available_commands.available_commands) == 1
    assert session_state.available_commands.available_commands[0].name == "test-cmd"


@pytest.mark.unit
async def test_available_commands_added_to_updates(
    handler: ACPClientHandler, session_state: ACPSessionState
) -> None:
    """AvailableCommandsUpdate should now be added to state.updates (bug fixed)."""
    cmd = AvailableCommand.create(name="test-cmd", description="A test command")
    update = AvailableCommandsUpdate(available_commands=[cmd])
    notification = SessionNotification(session_id="test-session", update=update)

    await handler.session_update(notification)

    assert len(session_state.updates) == 1
    assert session_state.updates[0] == update


@pytest.mark.unit
async def test_available_commands_captured_in_load_updates(
    handler: ACPClientHandler, session_state: ACPSessionState
) -> None:
    """AvailableCommandsUpdate should be captured during load_session (bug fixed).

    Previously, session_update() returned early for AvailableCommandsUpdate,
    so it never called state.add_update(), causing _load_updates to miss it.
    """
    session_state.start_load()
    cmd = AvailableCommand.create(name="test-cmd", description="A test command")
    update = AvailableCommandsUpdate(available_commands=[cmd])
    notification = SessionNotification(session_id="test-session", update=update)

    await handler.session_update(notification)

    load_updates = session_state.finish_load()
    assert len(load_updates) == 1
    assert update in load_updates


@pytest.mark.unit
async def test_stream_updates_captured_in_load_updates(
    handler: ACPClientHandler, session_state: ACPSessionState
) -> None:
    """When is_loading=True, stream updates should be captured in _load_updates."""
    session_state.start_load()
    chunk = AgentMessageChunk.text("response during load")
    notification = SessionNotification(session_id="test-session", update=chunk)

    await handler.session_update(notification)

    load_updates = session_state.finish_load()
    assert len(load_updates) == 1
    assert load_updates[0] == chunk


@pytest.mark.unit
async def test_all_updates_captured_when_is_loading(
    handler: ACPClientHandler, session_state: ACPSessionState
) -> None:
    """When is_loading=True, all stream updates should be in _load_updates."""
    session_state.start_load()
    user_chunk = UserMessageChunk.text("hello")
    agent_chunk = AgentMessageChunk.text("hi")
    tool_call = ToolCallStart(tool_call_id="tc-1", title="Tool")

    for update in [user_chunk, agent_chunk, tool_call]:
        notification = SessionNotification(session_id="test-session", update=update)
        await handler.session_update(notification)

    load_updates = session_state.finish_load()
    assert len(load_updates) == 3
    assert load_updates[0] == user_chunk
    assert load_updates[1] == agent_chunk
    assert load_updates[2] == tool_call


# =============================================================================
# State update types (mode, model, config)
# =============================================================================


@pytest.mark.unit
async def test_current_mode_update_sets_state(
    handler: ACPClientHandler, session_state: ACPSessionState
) -> None:
    """CurrentModeUpdate should set state.current_mode_id and modes.current_mode_id."""
    session_state.modes = SessionModeState(
        available_modes=[SessionMode(id="chat", name="Chat", description="Chat mode")],
        current_mode_id="chat",
    )
    update = CurrentModeUpdate(current_mode_id="code")
    notification = SessionNotification(session_id="test-session", update=update)

    await handler.session_update(notification)

    assert session_state.current_mode_id == "code"
    assert session_state.modes.current_mode_id == "code"
    assert len(session_state.updates) == 0  # State updates don't go to updates queue


@pytest.mark.unit
async def test_current_mode_update_emits_signal(
    handler: ACPClientHandler, mock_agent: MagicMock
) -> None:
    """CurrentModeUpdate should emit state_updated signal with ModeInfo."""
    from wolfharness.agents.modes import ModeInfo

    session_state = ACPSessionState(session_id="test-session")
    session_state.modes = SessionModeState(
        available_modes=[SessionMode(id="chat", name="Chat", description="Chat mode")],
        current_mode_id="chat",
    )
    handler = ACPClientHandler(agent=mock_agent, state=session_state)  # type: ignore[reportAbstractUsage]
    update = CurrentModeUpdate(current_mode_id="chat")
    notification = SessionNotification(session_id="test-session", update=update)

    await handler.session_update(notification)

    mock_agent.state_updated.emit.assert_awaited_once()
    emitted = mock_agent.state_updated.emit.call_args[0][0]
    assert isinstance(emitted, ModeInfo)
    assert emitted.id == "chat"


@pytest.mark.unit
async def test_current_model_update_sets_state(
    handler: ACPClientHandler, session_state: ACPSessionState
) -> None:
    """CurrentModelUpdate should set state.current_model_id and models.current_model_id."""
    from acp.schema import ModelInfo as ACPModelInfo

    session_state.models = SessionModelState(
        available_models=[ACPModelInfo(model_id="gpt-4", name="GPT-4", description="")],
        current_model_id="gpt-4",
    )
    update = CurrentModelUpdate(current_model_id="gpt-3")
    notification = SessionNotification(session_id="test-session", update=update)

    await handler.session_update(notification)

    assert session_state.current_model_id == "gpt-3"
    assert session_state.models.current_model_id == "gpt-3"
    assert len(session_state.updates) == 0


@pytest.mark.unit
async def test_current_model_update_emits_signal(
    handler: ACPClientHandler, mock_agent: MagicMock
) -> None:
    """CurrentModelUpdate should emit state_updated signal with ModelInfo."""
    from tokonomics.model_discovery.model_info import ModelInfo

    session_state = ACPSessionState(session_id="test-session")
    from acp.schema import ModelInfo as ACPModelInfo

    session_state.models = SessionModelState(
        available_models=[ACPModelInfo(model_id="gpt-4", name="GPT-4", description="The model")],
        current_model_id="gpt-4",
    )
    handler = ACPClientHandler(agent=mock_agent, state=session_state)  # type: ignore[reportAbstractUsage]
    from acp.schema import ModelInfo as ACPModelInfo

    session_state.models = SessionModelState(
        available_models=[ACPModelInfo(model_id="gpt-4", name="GPT-4", description="The model")],
        current_model_id="gpt-4",
    )
    update = CurrentModelUpdate(current_model_id="gpt-4")
    notification = SessionNotification(session_id="test-session", update=update)

    await handler.session_update(notification)

    mock_agent.state_updated.emit.assert_awaited_once()
    emitted = mock_agent.state_updated.emit.call_args[0][0]
    assert isinstance(emitted, ModelInfo)
    assert emitted.id == "gpt-4"


@pytest.mark.unit
async def test_config_option_update_sets_state(
    handler: ACPClientHandler, session_state: ACPSessionState
) -> None:
    """ConfigOptionUpdate should update the matching config option's current_value."""
    session_state.config_options = [
        SessionConfigOption(
            id="theme",
            name="Theme",
            description="UI theme",
            category="other",
            current_value="dark",
            options=[],
        )
    ]
    update = ConfigOptionUpdate(config_id="theme", value_id="light", config_options=[])
    notification = SessionNotification(session_id="test-session", update=update)

    await handler.session_update(notification)

    assert session_state.config_options[0].current_value == "light"
    assert len(session_state.updates) == 0


@pytest.mark.unit
async def test_config_option_update_calls_agent_update_state(
    handler: ACPClientHandler, mock_agent: MagicMock
) -> None:
    """ConfigOptionUpdate should call agent.update_state()."""
    session_state = ACPSessionState(session_id="test-session")
    session_state.config_options = [
        SessionConfigOption(
            id="theme",
            name="Theme",
            description="UI theme",
            category="other",
            current_value="dark",
            options=[],
        )
    ]
    handler = ACPClientHandler(agent=mock_agent, state=session_state)  # type: ignore[reportAbstractUsage]
    update = ConfigOptionUpdate(config_id="theme", value_id="light", config_options=[])
    notification = SessionNotification(session_id="test-session", update=update)

    await handler.session_update(notification)

    mock_agent.update_state.assert_awaited_once_with(config_id="theme", value_id="light")


# =============================================================================
# Proposed fix for AvailableCommandsUpdate bug
# =============================================================================


@pytest.mark.unit
async def test_available_commands_captured_in_load_updates_full(
    mock_agent: MagicMock,
) -> None:
    """Verify AvailableCommandsUpdate is properly captured during load after fix."""
    session_state = ACPSessionState(session_id="test-session")
    session_state.start_load()
    handler = ACPClientHandler(agent=mock_agent, state=session_state)

    cmd = AvailableCommand.create(name="test-cmd", description="A test command")
    update = AvailableCommandsUpdate(available_commands=[cmd])
    notification = SessionNotification(session_id="test-session", update=update)

    await handler.session_update(notification)

    load_updates = session_state.finish_load()
    assert update in load_updates
    assert len(load_updates) == 1
