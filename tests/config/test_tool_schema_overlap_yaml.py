"""YAML configuration tests for ToolSchemaOverlapCapability (tasks 8.1, 8.2).

Verifies the full YAML round-trip: a ``type: tool-schema-overlap`` entry is
resolved through the ``wolfharness.capabilities`` entry-point registry,
instantiated via ``build_config_capabilities``, and applies its overrides to
the tool pipeline.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai._run_context import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RunUsage
import pytest

from wolfharness import AgentsManifest, NativeAgentConfig
from wolfharness.capabilities.tool_schema_overlap_capability import (
    SchemaOverrideToolset,
    ToolSchemaOverlapCapability,
)
from wolfharness.capabilities.tool_schema_overlap_config import (
    ORIGINAL_TOOL_NAME_METADATA_KEY,
    SERVER_NAME_METADATA_KEY,
)
from wolfharness.tools import FunctionTool
from wolfharness_config.capabilities import (
    EntryPointCapabilityConfig,
    build_config_capabilities,
)


pytestmark = pytest.mark.unit

CONFIG_YAML = """\
agents:
  test_agent:
    type: native
    model: openai:gpt-4o-mini
    capabilities:
      - type: tool-schema-overlap
        args:
          servers:
            weather:
              get_weather:
                name: fetch_weather
                description: Get the current weather for a city.
                param_names:
                  location: city
                param_removals:
                  - api_key
                param_overrides:
                  api_key:
                    default: sk-default
          global_overrides:
            search:
              description: Search the web.
"""


def _load_agent() -> NativeAgentConfig:
    manifest = AgentsManifest.from_yaml(CONFIG_YAML)
    agent = manifest.agents["test_agent"]
    assert isinstance(agent, NativeAgentConfig)
    return agent


class TestYamlParsing:
    """Task 8.1: YAML round-trip through the entry-point registry."""

    def test_parses_to_entrypoint_config(self) -> None:
        agent = _load_agent()

        assert len(agent.capabilities) == 1
        cap = agent.capabilities[0]
        assert isinstance(cap, EntryPointCapabilityConfig)
        assert cap.type == "tool-schema-overlap"
        assert "servers" in cap.args
        assert "global_overrides" in cap.args

    def test_build_instantiates_capability_with_config(self) -> None:
        agent = _load_agent()
        built = build_config_capabilities(agent.capabilities)

        assert len(built) == 1
        capability = built[0]
        assert isinstance(capability, ToolSchemaOverlapCapability)

        override = capability.config.servers["weather"]["get_weather"]
        assert override.name == "fetch_weather"
        assert override.description == "Get the current weather for a city."
        assert override.param_names == {"location": "city"}
        assert override.param_removals == {"api_key"}
        assert override.param_overrides["api_key"].default == "sk-default"
        assert capability.config.global_overrides["search"].description == "Search the web."


class TestYamlPipeline:
    """Task 8.2: a YAML-built capability applies overrides to the pipeline."""

    @pytest.mark.asyncio
    async def test_built_capability_applies_overrides(self) -> None:
        agent = _load_agent()
        built = build_config_capabilities(agent.capabilities)
        capability = built[0]
        assert isinstance(capability, ToolSchemaOverlapCapability)

        async def get_weather(**kwargs: Any) -> str:
            return "sunny"

        schema = {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "api_key": {"type": "string"},
            },
            "required": ["location", "api_key"],
        }
        tool = FunctionTool(
            name="get_weather",
            description="Original weather description.",
            schema_override={  # type: ignore[arg-type]
                "name": "get_weather",
                "description": "Original weather description.",
                "parameters": schema,
            },
            callable=get_weather,
            source="mcp",
        )
        tool.metadata[SERVER_NAME_METADATA_KEY] = "weather"
        tool.metadata[ORIGINAL_TOOL_NAME_METADATA_KEY] = "get_weather"

        # The YAML fixture also declares a global override for `search`; list a
        # matching identified tool so the healthy-pipeline unmatched-key check
        # passes and the global override can be asserted below.
        async def search(**kwargs: Any) -> str:
            return "results"

        search_tool = FunctionTool(
            name="search",
            description="Original search.",
            schema_override={  # type: ignore[arg-type]
                "name": "search",
                "description": "Original search.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            callable=search,
            source="mcp",
        )
        search_tool.metadata[SERVER_NAME_METADATA_KEY] = "weather"
        search_tool.metadata[ORIGINAL_TOOL_NAME_METADATA_KEY] = "search"
        inner = FunctionToolset([tool.to_pydantic_ai(), search_tool.to_pydantic_ai()])

        wrapper = capability.get_wrapper_toolset(inner)
        assert isinstance(wrapper, SchemaOverrideToolset)
        ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
        tools = await wrapper.get_tools(ctx)

        assert set(tools) == {"fetch_weather", "search"}
        tool_def = tools["fetch_weather"].tool_def
        assert tool_def.description == "Get the current weather for a city."
        properties = tool_def.parameters_json_schema["properties"]
        assert set(properties) == {"city"}
        assert tool_def.parameters_json_schema["required"] == ["city"]

        # The global override applies to the identified search tool as well.
        assert tools["search"].tool_def.description == "Search the web."
