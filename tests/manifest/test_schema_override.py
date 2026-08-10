from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from wolfharness.tools.base import Tool


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from schemez import OpenAIFunctionDefinition


def my_tool(arg1: str):
    """Original docstring."""
    return f"Hello {arg1}"


async def test_schema_override_propagation():
    """Test that schema overrides are merged into the PydanticAI tool's function_schema."""
    # Define a schema override
    override: OpenAIFunctionDefinition = {
        "name": "my_tool",
        "description": "Overridden description",
        "parameters": {
            "type": "object",
            "properties": {
                "arg1": {"type": "string", "description": "Overridden argument description"}
            },
            "required": ["arg1"],
        },
    }

    tool = Tool.from_callable(my_tool, schema_override=override)

    # In RFC-0002, schema_override is handled in Tool.to_pydantic_ai()
    # and merged into function_schema.
    pydantic_tool = tool.to_pydantic_ai()

    assert pydantic_tool.function_schema is not None, "function_schema was not set on the tool"

    # Check that description and parameter descriptions from override are in the schema
    json_schema = pydantic_tool.function_schema.json_schema
    assert json_schema is not None
    # The tool description itself is NOT overridden (stays as docstring)
    # But the json_schema's description IS overridden
    assert json_schema["description"] == "Overridden description"

    # Verify parameter descriptions are overridden
    if "properties" in json_schema and "arg1" in json_schema["properties"]:
        arg1_desc = json_schema["properties"]["arg1"]
        # Check that description matches the override
        if isinstance(arg1_desc, dict):
            assert arg1_desc.get("description") == "Overridden argument description"
