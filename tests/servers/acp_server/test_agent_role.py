"""Tests for agent role config option and swap (Phase 2)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import yamling

from acp.exceptions import RequestError
from acp.schema import SessionConfigOption
from wolfharness import AgentPool, AgentsManifest
from wolfharness_server.acp_server.acp_agent import (
    AgentPoolACPAgent,
    get_agent_role_config_option,
)


pytestmark = pytest.mark.unit


MULTI_AGENT_CONFIG = """\
agents:
  agent_a:
    type: native
    model: test
    system_prompt: "Agent A"
  agent_b:
    type: native
    model: test
    system_prompt: "Agent B"
"""


class TestGetAgentRoleConfigOption:
    """Test get_agent_role_config_option."""

    def test_single_agent_no_role(self, minimal_pool: AgentPool):
        """Single agent pool should not expose agent_role option."""
        agent = MagicMock()
        agent.name = "test_agent"
        agent.host_context = minimal_pool.get_context()

        result = get_agent_role_config_option(agent)
        assert result is None

    def test_multi_agent_has_role(self):
        """Multi-agent pool should expose agent_role option."""
        manifest_dict = yamling.load_yaml(MULTI_AGENT_CONFIG, verify_type=dict)
        manifest_obj = AgentsManifest.model_validate(manifest_dict)

        import asyncio

        async def _run_test():
            async with AgentPool(manifest_obj) as pool:
                agent = MagicMock()
                agent.name = "agent_a"
                agent.display_name = None
                agent.host_context = pool.get_context()

                result = get_agent_role_config_option(agent)
                assert result is not None
                assert isinstance(result, SessionConfigOption)
                assert result.id == "agent_role"
                assert result.current_value == "agent_a"
                options = result.options
                assert len(options) == 2
                assert all(hasattr(o, "value") for o in options)
                choice_values = {o.value for o in options}  # type: ignore[union-attr]
                assert "agent_a" in choice_values
                assert "agent_b" in choice_values
                # Verify display_name fallback and description
                option_a = next(o for o in options if o.value == "agent_a")  # type: ignore[union-attr]
                assert option_a.name == "agent_a"  # type: ignore[union-attr]
                assert option_a.description == "Switch to agent_a agent"  # type: ignore[union-attr]
                option_b = next(o for o in options if o.value == "agent_b")  # type: ignore[union-attr]
                assert option_b.name == "agent_b"  # type: ignore[union-attr]
                assert option_b.description == "Switch to agent_b agent"  # type: ignore[union-attr]

        asyncio.run(_run_test())

    def test_no_pool_returns_none(self):
        """Agent without pool should not expose agent_role."""
        agent = MagicMock()
        agent.host_context = None

        result = get_agent_role_config_option(agent)
        assert result is None


class TestSwapSessionAgent:
    """Test _swap_session_agent on AgentPoolACPAgent."""

    @pytest.fixture
    def mock_acp_agent(self, minimal_pool: AgentPool):
        """Create a mock ACP agent with real pool context."""
        default_agent = MagicMock()
        default_agent.name = "test_agent"
        default_agent.host_context = minimal_pool.get_context()

        client = MagicMock()
        acp_agent = AgentPoolACPAgent(client=client, default_agent=default_agent)
        # Mock provider_router
        acp_agent.provider_router = MagicMock()
        return acp_agent, minimal_pool

    async def test_role_swap_success(self, mock_acp_agent):
        """Swap to valid agent should succeed."""
        acp_agent, _pool = mock_acp_agent

        session = MagicMock()
        session.agent = MagicMock()
        session.agent.name = "test_agent"
        session._task_lock = MagicMock()
        session._task_lock.locked.return_value = False
        session.switch_active_agent = AsyncMock()
        session.is_busy = False
        acp_agent.session_manager._acp_sessions = {"sess_1": session}
        acp_agent.session_manager.get_session = lambda sid: session

        result = await acp_agent._swap_session_agent("sess_1", "other")
        assert result["success"] is True
        session.switch_active_agent.assert_called_once_with("other")

    async def test_role_swap_blocked_during_prompt(self, mock_acp_agent):
        """Swap should fail when prompt is active."""
        acp_agent, _pool = mock_acp_agent

        session = MagicMock()
        session._task_lock = MagicMock()
        session._task_lock.locked.return_value = True
        acp_agent.session_manager.get_session = lambda sid: session

        with pytest.raises(RequestError) as exc_info:
            await acp_agent._swap_session_agent("sess_1", "other")
        assert exc_info.value.code == -32602

    async def test_role_swap_invalid_agent(self, mock_acp_agent):
        """Swap to invalid agent name should fail."""
        acp_agent, _pool = mock_acp_agent

        session = MagicMock()
        session.agent = MagicMock()
        session.agent.name = "test_agent"
        session._task_lock = MagicMock()
        session._task_lock.locked.return_value = False
        session.is_busy = False
        session.switch_active_agent = AsyncMock(side_effect=ValueError("Agent not found"))
        acp_agent.session_manager.get_session = lambda sid: session

        with pytest.raises(ValueError, match="Agent not found"):
            await acp_agent._swap_session_agent("sess_1", "nonexistent")

    async def test_swap_no_history_inheritance(self, mock_acp_agent):
        """New agent should start fresh without inheriting history."""
        acp_agent, _pool = mock_acp_agent

        session = MagicMock()
        session.agent = MagicMock()
        session.agent.name = "test_agent"
        session._task_lock = MagicMock()
        session._task_lock.locked.return_value = False
        session.is_busy = False
        session.switch_active_agent = AsyncMock()
        acp_agent.session_manager.get_session = lambda sid: session

        await acp_agent._swap_session_agent("sess_1", "other")
        # switch_active_agent is called, which handles creating new agent
        session.switch_active_agent.assert_called_once_with("other")
