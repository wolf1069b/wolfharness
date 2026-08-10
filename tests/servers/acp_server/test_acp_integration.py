"""Proper integration tests for ACP functionality."""

from __future__ import annotations

import tempfile
from unittest.mock import AsyncMock

import pytest

from acp import ClientCapabilities

# Add another agent to the pool for switching
from wolfharness import Agent
from wolfharness.delegation import AgentPool
from wolfharness_server.acp_server import ACPServer
from wolfharness_server.acp_server.session import ACPSession


pytestmark = pytest.mark.integration


@pytest.fixture
async def agent_pool():
    """Create a real agent pool from config."""
    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.manifest import AgentsManifest

    manifest = AgentsManifest(agents={"test_agent": NativeAgentConfig(model="test")})

    pool = AgentPool(manifest)

    # Create a simple test agent with pool reference
    def simple_callback(message: str) -> str:
        return f"Test response: {message}"

    Agent.from_callback(name="test_agent", callback=simple_callback, agent_pool=pool)
    # pool.register() removed; agent created from callback/config above
    return pool


async def test_acp_server_creation(agent_pool: AgentPool):
    """Test that ACP server can be created from agent pool."""
    server = ACPServer(pool=agent_pool)
    assert server.pool is agent_pool
    assert len(server.pool.manifest.agents) > 0


async def test_agent_switching_workflow(agent_pool: AgentPool, mock_acp_agent):
    """Test the complete agent switching workflow."""
    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.manifest import AgentsManifest

    config1 = NativeAgentConfig(name="agent1", model="test")
    config2 = NativeAgentConfig(name="agent2", model="test")
    manifest = AgentsManifest(agents={"agent1": config1, "agent2": config2})
    async with AgentPool(manifest) as multi_pool:
        agent1 = config1.get_agent(pool=multi_pool)

        mock_client = AsyncMock()
        capabilities = ClientCapabilities(fs=None, terminal=False)

        session = ACPSession(
            session_id="switching-test",
            agent=agent1,
            cwd=tempfile.gettempdir(),
            client=mock_client,
            acp_agent=mock_acp_agent,
            client_capabilities=capabilities,
        )

        # Should start with agent1
        assert session.agent.name == "agent1"

        # Switch to agent2
        await session.switch_active_agent("agent2")
        assert session.agent.name == "agent2"

        # Switching to non-existent agent should fail
        with pytest.raises(ValueError, match="not found"):
            await session.switch_active_agent("nonexistent")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
