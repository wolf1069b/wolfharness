"""Tests for parallel agent execution."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
import pytest

from wolfharness import AgentPool, AgentsManifest
from wolfharness.delegation.base_team import BaseTeam


pytestmark = pytest.mark.integration


class _TestOutput(BaseModel):
    """Expected output format."""

    message: str


def make_test_response(prompt: str) -> _TestOutput:
    """Callback for test agent responses."""
    return _TestOutput(message=f"Response to: {prompt}")


TEST_CONFIG = f"""\
responses:
  _TestOutput:
    response_schema:
        type: inline
        description: Simple test output
        fields:
            message:
                type: str
                description: Message from agent

agents:
  agent_1:
    type: native
    display_name: First Agent
    description: First test agent
    model:
      type: function
      function: {__name__}.make_test_response
    output_type: _TestOutput
    system_prompt: You are the first agent

  agent_2:
    display_name: Second Agent
    description: Second test agent
    model:
      type: function
      function: {__name__}.make_test_response
    output_type: _TestOutput
    system_prompt: You are the second agent
"""


async def test_parallel_execution():
    """Test parallel execution of multiple agents."""
    manifest = AgentsManifest.from_yaml(TEST_CONFIG)

    async with AgentPool(manifest) as pool:
        agent_1 = pool.manifest.agents["agent_1"].get_agent(pool=pool)
        agent_2 = pool.manifest.agents["agent_2"].get_agent(pool=pool)
        group = BaseTeam([agent_1, agent_2])

        async with agent_1, agent_2:
            prompt = "Test input"
            responses = await group.execute(prompt)
            # Verify execution
            assert len(responses) == 2
            assert all(r.success for r in responses)
            assert all(r.message.data.message == f"Response to: {prompt}" for r in responses)  # type: ignore

            # Verify agent names
            agent_names = {r.message.name for r in responses}  # type: ignore
            assert agent_names == {"agent_1", "agent_2"}


async def test_sequential_execution():
    """Test sequential execution through agent chain."""
    manifest: AgentsManifest = AgentsManifest.from_yaml(TEST_CONFIG)

    async with AgentPool(manifest) as pool:
        agent_1 = pool.manifest.agents["agent_1"].get_agent(pool=pool)
        agent_2 = pool.manifest.agents["agent_2"].get_agent(pool=pool)
        group: BaseTeam[Any, Any] = BaseTeam([agent_1, agent_2], mode="sequential")

        async with agent_1, agent_2:
            prompt = "Test input"
            responses = await group.execute(prompt)

            # Verify execution order
            assert len(responses) == 2
            assert all(r.success for r in responses)
            agent_order = [r.message.name for r in responses]  # type: ignore
            assert agent_order == ["agent_1", "agent_2"]

            # Verify message chain
            first_response = responses[0].message.data.message  # type: ignore
            assert first_response == f"Response to: {prompt}"

            second_response = responses[1].message.data.message  # type: ignore
            expected_input = "Response to: Test input"  # Just care about the content
            assert expected_input in second_response


if __name__ == "__main__":
    pytest.main(["-v", __file__])
