"""Tests for mode field consistency in opencode_server.

The OpenCode protocol has two distinct concepts:
- Agent.mode: Literal["subagent", "primary", "all"] — agent category (visibility)
- AssistantMessage.mode: str — identifies which agent produced the message

The TUI uses Agent.mode to filter agents (exclude "subagent" from switcher).
The TUI uses AssistantMessage.agent (name) to resolve the agent for display.

These tests verify:
1. /agent endpoint returns mode="primary" for all wolfharness agents (correct)
2. Assistant messages created by _before_consumer_loop have mode=agent_name
3. chat_message_to_opencode preserves mode from ChatMessage.name
4. Subagent assistant messages have mode and agent matching the child agent
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from wolfharness.messaging import ChatMessage
from wolfharness_server.opencode_server.converters import chat_message_to_opencode


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# /agent endpoint mode field
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_agent_endpoint_mode_is_primary_for_all_agents() -> None:
    """GET /agent should return mode='primary' for all wolfharness agents.

    AgentMode is Literal['subagent', 'primary', 'all']. All wolfharness agents
    are primary (visible in switcher). This is correct — mode is a category,
    not an agent identifier.
    """
    from wolfharness_server.opencode_server.routes.agent_routes import list_agents

    agent1 = MagicMock()
    agent1.description = "Agent 1"
    agent1.display_name = None
    agent2 = MagicMock()
    agent2.description = "Agent 2"
    agent2.display_name = None

    ctx = MagicMock()
    ctx.main_agent_name = "agent1"
    ctx.manifest.agents = {"agent1": agent1, "agent2": agent2}

    state = MagicMock()
    state.agent.host_context = ctx

    agents = await list_agents(state)

    assert len(agents) == 2
    for agent in agents:
        assert agent.mode == "primary"
        assert agent.name in ("agent1", "agent2")


# ---------------------------------------------------------------------------
# _before_consumer_loop mode field
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_before_consumer_loop_sets_mode_to_agent_name() -> None:
    """_before_consumer_loop should set mode=agent_name on assistant message.

    The TUI uses the message's mode and agent fields to identify which agent
    produced the message. Both should be the agent's name.
    """
    from wolfharness_server.opencode_server.opencode_event_bridge import (
        OpenCodeEventBridgeMixin,
    )

    session_state = MagicMock()
    session_state.agent_name = "librarian"
    session_state.metadata = {}

    session_pool = MagicMock()
    session_pool.sessions.get_session = MagicMock(return_value=session_state)

    server_state = MagicMock()
    server_state.working_dir = "/tmp"
    server_state.resolve_default_model_info = MagicMock(return_value=("default", "wolfharness"))

    bridge = OpenCodeEventBridgeMixin.__new__(OpenCodeEventBridgeMixin)
    bridge.session_pool = session_pool
    bridge.server_state = server_state
    bridge._pending_message_ids = {}
    bridge._pending_message_metadata = {}
    bridge._contexts = {}
    bridge._adapters = {}
    bridge._message_registered = {}
    bridge.get_session_context_data = MagicMock(return_value=None)

    await bridge._before_consumer_loop("test-session")

    ctx = bridge._contexts["test-session"]
    assert ctx.assistant_msg.info.mode == "librarian"
    assert ctx.assistant_msg.info.agent == "librarian"


# ---------------------------------------------------------------------------
# chat_message_to_opencode mode preservation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_chat_message_to_opencode_preserves_mode_from_name() -> None:
    """chat_message_to_opencode should set mode from msg.name, not default 'default'.

    When loading messages from storage, the mode field must be preserved
    so the TUI can identify which agent produced the message.
    """
    msg = ChatMessage(
        content="test response",
        role="assistant",
        name="librarian",
        model_name="gpt-4o",
        provider_name="openai",
    )
    msg.message_id = "msg_test_1"

    result = chat_message_to_opencode(
        msg,
        session_id="test-session",
        working_dir="/tmp",
        agent_name="librarian",
        model_id="gpt-4o",
        provider_id="openai",
    )

    assert result.info.agent == "librarian"
    assert result.info.mode == "librarian", f"mode should be 'librarian', got '{result.info.mode}'"


@pytest.mark.unit
def test_chat_message_to_opencode_mode_falls_back_to_agent_name_param() -> None:
    """When msg.name is None, mode should fall back to the agent_name parameter."""
    msg = ChatMessage(
        content="test response",
        role="assistant",
        name=None,
        model_name="gpt-4o",
        provider_name="openai",
    )
    msg.message_id = "msg_test_2"

    result = chat_message_to_opencode(
        msg,
        session_id="test-session",
        working_dir="/tmp",
        agent_name="engineer",
        model_id="gpt-4o",
        provider_id="openai",
    )

    assert result.info.agent == "engineer"
    assert result.info.mode == "engineer"


# ---------------------------------------------------------------------------
# Subagent mode propagation
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_subagent_assistant_mode_matches_child_agent_name() -> None:
    """Subagent assistant message mode should match the child agent's name.

    When a subagent session is created, the child session's assistant message
    should have mode=child_agent_name so the TUI can display the correct agent.
    """
    from wolfharness_server.opencode_server.opencode_event_bridge import (
        OpenCodeEventBridgeMixin,
    )

    child_session_state = MagicMock()
    child_session_state.agent_name = "historian"
    child_session_state.metadata = {}

    session_pool = MagicMock()
    session_pool.sessions.get_session = MagicMock(return_value=child_session_state)

    server_state = MagicMock()
    server_state.working_dir = "/tmp"
    server_state.resolve_default_model_info = MagicMock(return_value=("default", "wolfharness"))

    bridge = OpenCodeEventBridgeMixin.__new__(OpenCodeEventBridgeMixin)
    bridge.session_pool = session_pool
    bridge.server_state = server_state
    bridge._pending_message_ids = {}
    bridge._pending_message_metadata = {}
    bridge._contexts = {}
    bridge._adapters = {}
    bridge._message_registered = {}
    bridge.get_session_context_data = MagicMock(return_value=None)

    await bridge._before_consumer_loop("child-session")

    ctx = bridge._contexts["child-session"]
    assert ctx.assistant_msg.info.mode == "historian"
    assert ctx.assistant_msg.info.agent == "historian"
