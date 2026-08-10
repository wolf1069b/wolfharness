"""Minimal test to isolate the piping hang."""

from pydantic_ai._utils import disable_threads
import pytest

from wolfharness import Agent


pytestmark = pytest.mark.unit


async def test_sync_callback_pipe_no_threads():
    """Test with sync named function, threads disabled."""

    def callback(text: str) -> str:
        return f"model: {text}"

    with disable_threads():
        agent1 = Agent.from_callback(callback, name="agent1")
        agent2 = Agent.from_callback(callback, name="agent2")
        pipeline = agent1 | agent2
        result = await pipeline.execute("test")
        assert len(result) == 2


async def test_sync_callback_pipe_with_threads():
    """Test with sync named function, threads enabled (default)."""

    def callback(text: str) -> str:
        return f"model: {text}"

    agent1 = Agent.from_callback(callback, name="agent1")
    agent2 = Agent.from_callback(callback, name="agent2")
    pipeline = agent1 | agent2
    result = await pipeline.execute("test")
    assert len(result) == 2
