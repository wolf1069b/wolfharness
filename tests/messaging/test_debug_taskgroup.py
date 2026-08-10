"""Diagnostic test for whether calling agent.run() twice hangs (without pipeline).

If the second call hangs, the issue is in the Agent level.
If it works, the issue is specific to the pydantic-graph TaskGroup interaction.
"""

import asyncio

import pytest

from wolfharness import Agent


pytestmark = pytest.mark.unit


async def test_sequential_sync_agents_no_pipeline():
    """Two sequential sync agent runs WITHOUT a pipeline - should NOT hang."""
    agent1 = Agent.from_callback(lambda x: f"model1: {x}", name="agent1")
    agent2 = Agent.from_callback(lambda x: f"model2: {x}", name="agent2")

    # Run agent1
    result1 = await asyncio.wait_for(agent1.run("test"), timeout=10)
    print(f"Agent1 result: {result1.content}")

    # Run agent2 with agent1's output - does this hang?
    result2 = await asyncio.wait_for(agent2.run(str(result1.content)), timeout=10)
    print(f"Agent2 result: {result2.content}")

    assert "model1" in str(result1.content)
    assert "model2" in str(result2.content)


async def test_same_sync_agent_twice():
    """Run the SAME sync agent twice to check if second call hangs."""
    agent = Agent.from_callback(lambda x: f"processed: {x}", name="agent")

    result1 = await asyncio.wait_for(agent.run("first"), timeout=10)
    print(f"First result: {result1.content}")

    result2 = await asyncio.wait_for(agent.run("second"), timeout=10)
    print(f"Second result: {result2.content}")

    assert "first" in str(result1.content)
    assert "second" in str(result2.content)


async def test_pipeline_with_async_second():
    """Pipeline: sync first, async second - should work."""
    agent1 = Agent.from_callback(lambda x: f"sync: {x}", name="sync_agent")

    async def async_transform(x: str) -> str:
        return f"async: {x}"

    agent2 = Agent.from_callback(async_transform, name="async_agent")

    pipeline = agent1 | agent2
    result = await asyncio.wait_for(pipeline.execute("test"), timeout=10)
    assert result[0].message
    assert result[1].message


async def test_pipeline_with_sync_second():
    """Pipeline: sync first, sync second - THIS SHOULD HANG (the bug)."""
    agent1 = Agent.from_callback(lambda x: f"sync1: {x}", name="sync1")
    agent2 = Agent.from_callback(lambda x: f"sync2: {x}", name="sync2")

    pipeline = agent1 | agent2
    result = await asyncio.wait_for(pipeline.execute("test"), timeout=10)
    assert result[0].message
    assert result[1].message
