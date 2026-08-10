"""Minimal test to isolate the piping hang."""

import pytest

from wolfharness import Agent


pytestmark = pytest.mark.unit


async def test_sync_callback_pipe():
    """Test with sync named function."""

    def callback(text: str) -> str:
        return f"model: {text}"

    agent1 = Agent.from_callback(callback, name="agent1")
    agent2 = Agent.from_callback(callback, name="agent2")
    pipeline = agent1 | agent2
    result = await pipeline.execute("test")
    assert len(result) == 2


async def test_lambda_callback_pipe():
    """Test with lambda."""
    agent1 = Agent.from_callback(lambda x: f"model: {x}", name="agent1")
    agent2 = Agent.from_callback(lambda x: f"transform: {x}", name="agent2")
    pipeline = agent1 | agent2
    result = await pipeline.execute("test")
    assert len(result) == 2


async def test_async_callback_pipe():
    """Test with async named function."""

    async def callback(text: str) -> str:
        return f"model: {text}"

    agent1 = Agent.from_callback(callback, name="agent1")
    agent2 = Agent.from_callback(callback, name="agent2")
    pipeline = agent1 | agent2
    result = await pipeline.execute("test")
    assert len(result) == 2
