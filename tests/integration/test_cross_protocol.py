"""Cross-protocol integration validation (Task 17).

# TODO: L2 migration — test requires complex mock pool dependencies that
# cannot be easily replaced with a real pool. Needs investigation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from wolfharness.agents.modes import ModeCategory, ModeInfo
from wolfharness_server.acp_server.acp_agent import get_agent_role_config_option
from wolfharness_server.opencode_server.routes.config_routes import list_modes


pytestmark = pytest.mark.unit


class TestCrossProtocolAlignment:
    """Verify ACP and OpenCode protocols reflect aligned agent state."""

    async def test_agent_role_appears_when_multiple_modes(self):
        """agent_role config option appears when /mode returns multiple modes."""
        # Setup: agent with multiple modes
        agent = MagicMock()
        agent.name = "agent_a"
        agent.host_context = MagicMock()
        agent_b = MagicMock()
        agent_b.name = "agent_b"
        agent.host_context.manifest.agents = {"agent_a": agent, "agent_b": agent_b}
        agent.get_modes = AsyncMock(
            return_value=[
                ModeCategory(
                    id="mode",
                    name="Mode",
                    available_modes=[
                        ModeInfo(id="default", name="Default"),
                        ModeInfo(id="advanced", name="Advanced"),
                    ],
                    current_mode_id="default",
                    category="mode",
                )
            ]
        )

        # ACP: get agent_role config option
        agent_role_opt = get_agent_role_config_option(agent)
        assert agent_role_opt is not None
        assert agent_role_opt.id == "agent_role"

        # OpenCode: get modes
        state = MagicMock()
        state.agent = agent
        modes = await list_modes(state)  # type: ignore[arg-type]
        assert len(modes) == 2

    async def test_agent_role_hidden_when_single_mode(self):
        """agent_role config option hidden when /mode returns single mode."""
        agent = MagicMock()
        agent.name = "solo"
        agent.host_context = MagicMock()
        agent.host_context.manifest.agents = {"solo": agent}
        agent.get_modes = AsyncMock(
            return_value=[
                ModeCategory(
                    id="mode",
                    name="Mode",
                    available_modes=[ModeInfo(id="default", name="Default")],
                    current_mode_id="default",
                    category="mode",
                )
            ]
        )

        # ACP: no agent_role for single agent
        agent_role_opt = get_agent_role_config_option(agent)
        assert agent_role_opt is None

        # OpenCode: single mode (returns mode.id, not mode.name)
        state = MagicMock()
        state.agent = agent
        modes = await list_modes(state)  # type: ignore[arg-type]
        assert len(modes) == 1
        assert modes[0].name == "default"

    async def test_cross_protocol_model_alignment(self):
        """Both protocols reflect same underlying model state."""
        from wolfharness.models.model_configs import StringModelConfig
        from wolfharness_server.acp_server.provider_router import ProviderRouter
        from wolfharness_server.shared.model_utils import build_model_state_for_acp

        # Manifest with configured variants
        manifest = MagicMock()
        manifest.model_variants = {
            "fast": StringModelConfig(identifier="openai:gpt-4o-mini"),
        }
        pool = MagicMock()
        pool.manifest = manifest
        agent = MagicMock()
        agent.name = "test"
        agent.model_name = "fast"
        agent.host_context = pool
        agent.get_available_models = AsyncMock(return_value=[])

        router = ProviderRouter(manifest)  # type: ignore[arg-type]
        acp_state = await build_model_state_for_acp(agent, router)  # type: ignore[arg-type]

        assert acp_state is not None
        model_ids = {m.model_id for m in acp_state.available_models}
        assert "fast" in model_ids
        assert acp_state.current_model_id == "fast"
