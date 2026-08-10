"""Integration test for Claude ACP agent with toolsets exposed via MCP bridge.

This test creates an AgentPool with a Claude ACP agent configured with
the Subagent toolset, which gets exposed via an internal MCP server bridge.
The Claude agent can then use our internal tools through MCP.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from wolfharness import AgentPool
from wolfharness.agents.acp_agent import ACPAgent
from wolfharness.models.acp_agents.base import ACPAgentConfig
from wolfharness_config.toolsets import SubagentToolsetConfig


if not shutil.which("claude-code-acp"):
    pytest.skip("claude-code-acp CLI not available", allow_module_level=True)

pytestmark = [pytest.mark.integration]


@pytest.fixture
def claude_config_with_subagent() -> ACPAgentConfig:
    """Create Claude ACP config with Subagent toolset."""
    return ACPAgentConfig(
        name="claude_orchestrator",
        command="claude-code-acp",
        tools=[SubagentToolsetConfig()],
        env_vars={"ANTHROPIC_API_KEY": ""},  # Use subscription, not direct API key
    )


async def test_claude_acp_tool_bridge_mcp_config(claude_config_with_subagent: ACPAgentConfig):
    """Test that tool bridge MCP config is properly passed to session."""
    async with AgentPool() as pool:  # noqa: SIM117
        async with ACPAgent.from_config(claude_config_with_subagent, agent_pool=pool) as agent:
            # Verify extra MCP servers include our bridge
            assert len(agent._extra_mcp_servers) > 0
            # Find our toolset bridge server
            bridge_server = next(
                (s for s in agent._extra_mcp_servers if "tools" in s.name),
                None,
            )
            assert bridge_server is not None


async def test_claude_acp_multiple_toolsets():
    """Test Claude ACP agent with multiple toolsets."""
    from wolfharness_config.toolsets import DebugToolsetConfig

    tools = [SubagentToolsetConfig(), DebugToolsetConfig()]
    config = ACPAgentConfig(
        name="claude_multi", command="claude-code-acp", cwd=str(Path.cwd()), tools=tools
    )
    async with AgentPool() as pool, ACPAgent.from_config(config, agent_pool=pool) as agent:
        # All toolsets should be exposed via single bridge
        assert agent._tool_bridge is not None
        tool_names = {t.name for t in await agent.tools.get_tools()}
        # Should have tools from both toolsets
        # SubagentToolset provides: task
        assert "list_available_nodes" not in tool_names
        assert "task" in tool_names
        assert "execute_introspection" in tool_names


if __name__ == "__main__":
    pytest.main(["-v", __file__])
