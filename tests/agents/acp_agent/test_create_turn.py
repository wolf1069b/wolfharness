"""Unit tests for ACPAgent.create_turn()."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from acp import InitializeRequest
from acp.agent.acp_agent_api import ACPAgentAPI
from wolfharness.agents.acp_agent import ACPAgent
from wolfharness.agents.acp_agent.client_handler import TimeoutableEvent
from wolfharness.agents.acp_agent.session_state import ACPSessionState
from wolfharness.agents.acp_agent.turn import ACPClientProtocol, ACPTurn
from wolfharness.agents.context import AgentRunContext


@pytest.mark.unit
def test_acp_agent_create_turn_returns_acp_turn() -> None:
    """Given an ACPAgent with mocked API, create_turn() returns an ACPTurn."""
    init_request = MagicMock(spec=InitializeRequest)
    agent = ACPAgent(command="test-cmd", init_request=init_request)
    agent._api = MagicMock()
    agent._sdk_session_id = "test-session-id"

    run_ctx = AgentRunContext(session_id="test-run-ctx")
    turn = agent.create_turn(
        prompts=["hello"],
        run_ctx=run_ctx,
        message_history=[],
    )

    assert isinstance(turn, ACPTurn)


@pytest.mark.unit
def test_acp_agent_api_satisfies_acp_client_protocol() -> None:
    """ACPAgentAPI with state+event satisfies ACPClientProtocol (no cast needed)."""
    connection = MagicMock()
    state = ACPSessionState(session_id="test-session")
    update_event = TimeoutableEvent()
    api = ACPAgentAPI(connection, state=state, update_event=update_event)

    assert isinstance(api, ACPClientProtocol)


@pytest.mark.unit
def test_acp_agent_api_satisfies_protocol_without_state() -> None:
    """ACPAgentAPI satisfies ACPClientProtocol even without state (methods exist)."""
    connection = MagicMock()
    api = ACPAgentAPI(connection)

    assert isinstance(api, ACPClientProtocol)
