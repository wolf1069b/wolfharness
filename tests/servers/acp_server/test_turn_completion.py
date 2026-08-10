"""Tests for AgentPoolACPAgent.initialize() turn_complete gating.

Verifies that the agent only advertises turn_complete in the InitializeResponse
when the client explicitly declares support for it via client_capabilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from acp import ClientCapabilities, InitializeRequest


if TYPE_CHECKING:
    from wolfharness_server.acp_server.acp_agent import AgentPoolACPAgent


pytestmark = pytest.mark.unit


@pytest.fixture
def initialize_request() -> InitializeRequest:
    """Create a base InitializeRequest for testing."""
    return InitializeRequest(protocol_version=1)


@pytest.fixture
def agent_with_mock_pool(mock_acp_agent: AgentPoolACPAgent) -> AgentPoolACPAgent:
    """Return a mock ACP agent ready for initialize() tests."""
    return mock_acp_agent


class TestTurnCompleteGating:
    """Tests for turn_complete capability advertisement gating."""

    @pytest.mark.anyio
    async def test_turn_complete_advertised_when_client_supports_it(
        self,
        agent_with_mock_pool: AgentPoolACPAgent,
        initialize_request: InitializeRequest,
    ) -> None:
        """When client sends turn_complete=True, response advertises it."""
        request = initialize_request.model_copy(
            update={
                "client_capabilities": ClientCapabilities.create(
                    turn_complete=True,
                ),
            },
        )

        response = await agent_with_mock_pool.initialize(request)

        assert response.agent_capabilities is not None
        assert response.agent_capabilities.session_capabilities is not None
        assert response.agent_capabilities.session_capabilities.turn_complete is not None

    @pytest.mark.anyio
    async def test_turn_complete_not_advertised_when_client_disabled(
        self,
        agent_with_mock_pool: AgentPoolACPAgent,
        initialize_request: InitializeRequest,
    ) -> None:
        """When client sends turn_complete=False, response does NOT advertise it."""
        request = initialize_request.model_copy(
            update={
                "client_capabilities": ClientCapabilities.create(
                    turn_complete=False,
                ),
            },
        )

        response = await agent_with_mock_pool.initialize(request)

        assert response.agent_capabilities is not None
        assert response.agent_capabilities.session_capabilities is not None
        assert response.agent_capabilities.session_capabilities.turn_complete is None

    @pytest.mark.anyio
    async def test_turn_complete_not_advertised_when_field_missing(
        self,
        agent_with_mock_pool: AgentPoolACPAgent,
        initialize_request: InitializeRequest,
    ) -> None:
        """When client_capabilities lacks turn_complete field, response does NOT advertise it."""
        caps = ClientCapabilities.create()
        # Explicitly verify the default is False / not True
        assert not caps.turn_complete
        request = initialize_request.model_copy(
            update={"client_capabilities": caps},
        )

        response = await agent_with_mock_pool.initialize(request)

        assert response.agent_capabilities is not None
        assert response.agent_capabilities.session_capabilities is not None
        assert response.agent_capabilities.session_capabilities.turn_complete is None

    @pytest.mark.anyio
    async def test_turn_complete_not_advertised_when_capabilities_none(
        self,
        agent_with_mock_pool: AgentPoolACPAgent,
        initialize_request: InitializeRequest,
    ) -> None:
        """When client_capabilities is None, response does NOT advertise turn_complete."""
        request = initialize_request.model_copy(
            update={"client_capabilities": None},
        )

        response = await agent_with_mock_pool.initialize(request)

        assert response.agent_capabilities is not None
        assert response.agent_capabilities.session_capabilities is not None
        assert response.agent_capabilities.session_capabilities.turn_complete is None

    @pytest.mark.anyio
    async def test_client_capabilities_stored_on_agent(
        self,
        agent_with_mock_pool: AgentPoolACPAgent,
        initialize_request: InitializeRequest,
    ) -> None:
        """Client capabilities are persisted on the agent for later use."""
        caps = ClientCapabilities.create(turn_complete=True)
        request = initialize_request.model_copy(
            update={"client_capabilities": caps},
        )

        await agent_with_mock_pool.initialize(request)

        assert agent_with_mock_pool.client_capabilities is caps
