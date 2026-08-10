from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from schemez import OpenAIFunctionDefinition

from wolfharness.capabilities.function_toolset import FunctionToolsetCapability


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from pydantic_ai.capabilities import AbstractCapability


class MockProvider(FunctionToolsetCapability):
    """Mock provider for testing schema overrides."""

    async def my_tool(self, x: int, y: str) -> str:
        """Original description."""
        return f"{x} {y}"


@pytest.mark.asyncio
async def test_to_pydantic_ai_includes_parameter_descriptions_from_override():
    provider = MockProvider(name="mock")

    # Custom schema with parameter descriptions
    schema_override = OpenAIFunctionDefinition(
        name="my_tool",
        description="Overridden description",
        parameters={
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "Custom X description"},
                "y": {"type": "string", "description": "Custom Y description"},
            },
            "required": ["x", "y"],
        },
    )

    tool = provider.create_tool(provider.my_tool, schema_override=schema_override)
    pydantic_tool = tool.to_pydantic_ai()

    # When using Tool.from_schema, the custom schema is in function_schema.json_schema
    assert pydantic_tool.function_schema is not None

    # Verify that the parameter descriptions from our override are preserved
    params = pydantic_tool.function_schema.json_schema["properties"]
    assert params["x"]["description"] == "Custom X description"
    assert params["y"]["description"] == "Custom Y description"
    assert pydantic_tool.name == "my_tool"
    assert pydantic_tool.description == "Original description."

    def get_capabilities(self) -> AbstractCapability | None:
        """Return a pydantic-ai capability for this provider.

        Returns:
            A pydantic-ai AbstractCapability instance, or None.
        """
        return None
