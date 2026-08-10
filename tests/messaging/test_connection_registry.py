from __future__ import annotations

from pydantic_ai.models.test import TestModel
import pytest

from wolfharness import Agent, AgentPool
from wolfharness.models.agents import NativeAgentConfig
from wolfharness.models.manifest import AgentsManifest


pytestmark = pytest.mark.integration


@pytest.fixture
async def pool():
    """Create agent pool with test agents."""
    manifest = AgentsManifest(agents={"agent1": NativeAgentConfig(name="agent1", model="test")})
    pool = AgentPool(manifest)

    async with pool:
        yield pool


async def test_registry_captures_agent_interaction(pool: AgentPool):
    """Test that registry captures real agent interactions."""
    messages = []
    pool.connection_registry.message_flow.connect(messages.append)

    # Create agents directly with pool reference
    agent1 = Agent("agent1", model=TestModel(), agent_pool=pool)
    agent2 = Agent("agent2", model=TestModel())
    agent1.connect_to(agent2, name="test_talk")
    await agent1.run("Test message")

    # Verify flow was captured
    assert len(messages) == 1
    assert messages[0].source == agent1
    assert messages[0].targets == [agent2]


async def test_chained_communication(pool: AgentPool):
    """Test message flow through chain of agents."""
    messages = []
    pool.connection_registry.message_flow.connect(messages.append)

    # Create agents directly with pool reference
    agent1 = Agent("agent1", model=TestModel(), agent_pool=pool)
    agent2 = Agent("agent2", model=TestModel(), agent_pool=pool)
    agent3 = Agent("agent3", model=TestModel())

    # Create chain with named connections
    agent1.connect_to(agent2, name="chain1")
    agent2.connect_to(agent3, name="chain2")

    # Trigger chain
    await agent1.run("Start chain")

    # Should capture both flows
    assert len(messages) == 2
    assert messages[0].source == agent1
    assert messages[0].targets == [agent2]
    assert messages[1].source == agent2
    assert messages[1].targets == [agent3]


async def test_broadcast_communication(pool: AgentPool):
    """Test broadcasting to multiple agents."""
    messages = []
    pool.connection_registry.message_flow.connect(messages.append)

    # Create agents directly with pool reference
    agent1 = Agent("agent1", model=TestModel(), agent_pool=pool)
    agent2 = Agent("agent2", model=TestModel())
    agent3 = Agent("agent3", model=TestModel())

    # Create individual connections for broadcast
    agent1.connect_to(agent2, name="broadcast1")
    agent1.connect_to(agent3, name="broadcast2")

    # Send broadcast
    await agent1.run("Broadcast message")

    # Should capture two events, one for each target
    assert len(messages) == 2
    targets = {t for m in messages for t in m.targets}
    assert targets == {agent2, agent3}
    assert all(m.source == agent1 for m in messages)


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
