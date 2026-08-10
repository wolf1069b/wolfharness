from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai.models.test import TestModel
import pytest

from wolfharness import Agent


pytestmark = pytest.mark.unit


class Result(BaseModel):
    """Structured response result."""

    is_positive: bool


async def test_structured_response():
    """Test that structured output_type produces typed result.data."""
    agent = Agent(
        name="summarizer",
        model=TestModel(seed=1),
        system_prompt="Summarize text in a structured way.",
        output_type=Result,
    )
    result = await agent.run("I love this new feature!")
    assert isinstance(result.data, Result)
    # TestModel(seed=1) should produce is_positive=True for boolean fields
    assert result.data.is_positive is True
