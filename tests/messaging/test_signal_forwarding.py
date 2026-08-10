from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from wolfharness import Agent, AgentPool, AgentsManifest


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from pathlib import Path


BASIC_FORWARDING = """\
agents:
    agent1:
        type: native
        model: test
        display_name: TestAgent 1
        connections:
            - type: node
              name: agent2

    agent2:
        type: native
        model: test
        display_name: TestAgent 2
        connections:
            - type: node
              name: agent3

    agent3:
        type: native
        model: test
        display_name: TestAgent 3
"""


INVALID_FORWARD = """\
agents:
    agent1:
        type: native
        model: test
        display_name: TestAgent
        connections:
            - type: node
              name: nonexistent
"""


PARTIAL_FORWARDING = """\
agents:
    agent1:
        type: native
        model: test
        display_name: TestAgent 1
        connections:
            - type: node
              name: agent2

    agent2:
        type: native
        model: test
        display_name: TestAgent 2
"""


@pytest.fixture
def basic_config(tmp_path: Path) -> Path:
    """Create a temporary config file with basic forwarding setup."""
    config_file = tmp_path / "agents.yml"
    config_file.write_text(BASIC_FORWARDING)
    return config_file


@pytest.fixture
def partial_config(tmp_path: Path) -> Path:
    """Create a temporary config file with partial forwarding setup."""
    config_file = tmp_path / "partial.yml"
    config_file.write_text(PARTIAL_FORWARDING)
    return config_file


@pytest.fixture
def invalid_config(tmp_path: Path) -> Path:
    """Create a temporary config file with invalid forwarding setup."""
    config_file = tmp_path / "invalid.yml"
    config_file.write_text(INVALID_FORWARD)
    return config_file


async def test_agent_forwarding(basic_config: Path):
    """Test that messages get forwarded through the agent chain."""
    manifest = AgentsManifest.from_file(basic_config)

    async with AgentPool(manifest) as pool:
        # Create agents directly from config names
        agent1 = Agent("agent1", model="test", agent_pool=pool)
        agent2 = Agent("agent2", model="test", agent_pool=pool)
        agent3 = Agent("agent3", model="test", agent_pool=pool)

        # Set up connections as defined in config
        agent1.connect_to(agent2, name="agent1->agent2")
        agent2.connect_to(agent3, name="agent2->agent3")

        responded_agents = set()
        received_messages = []

        def record_response(agent_name: str):
            def callback(message):
                responded_agents.add(agent_name)
                received_messages.append(f"{agent_name}: {message.content}")

            return callback

        agent1.message_sent.connect(record_response("agent1"))
        agent2.message_sent.connect(record_response("agent2"))
        agent3.message_sent.connect(record_response("agent3"))

        await agent1.run("test")
        # Wait for all forwarded messages to be processed
        if agent1._pending_tasks:
            await asyncio.gather(*agent1._pending_tasks, return_exceptions=True)
        if agent2._pending_tasks:
            await asyncio.gather(*agent2._pending_tasks, return_exceptions=True)
        if agent3._pending_tasks:
            await asyncio.gather(*agent3._pending_tasks, return_exceptions=True)
        assert responded_agents == {"agent1", "agent2", "agent3"}


async def test_partial_chain(partial_config: Path):
    """Test forwarding with only some agents loaded."""
    manifest = AgentsManifest.from_file(partial_config)

    async with AgentPool(manifest) as pool:
        # Create agents directly
        agent1 = Agent("agent1", model="test", agent_pool=pool)
        agent2 = Agent("agent2", model="test", agent_pool=pool)

        # Set up connections as defined in config
        agent1.connect_to(agent2, name="agent1->agent2")

        responded_agents = set()
        agent1.message_sent.connect(lambda _: responded_agents.add("agent1"))
        agent2.message_sent.connect(lambda _: responded_agents.add("agent2"))

        await agent1.run("test")
        if agent2._pending_tasks:
            await asyncio.gather(*agent2._pending_tasks, return_exceptions=True)
        assert responded_agents == {"agent1", "agent2"}


async def test_invalid_forward_target(invalid_config: Path):
    """Test error when forwarding to non-existent agent."""
    manifest = AgentsManifest.from_file(invalid_config)

    with pytest.raises(ValueError, match=r"Forward target.*not found"):
        async with AgentPool(manifest):
            pass


if __name__ == "__main__":
    pytest.main([__file__])
