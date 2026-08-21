"""Integration tests for ToolSchemaOverlapCapability composition.

Covers composition with ``ToolDisplayCapability`` (schema layer vs display
layer), per-server scoping across multiple ``McpServerCap`` servers, the
triple stack of ``tool_prefix`` + schema override + display rename, runtime
round-trips through the real ``McpServerCap`` pipeline with a fake MCP
client, and degradation when source identity metadata is absent.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import logging
from typing import Any
from unittest.mock import MagicMock

from pydantic_ai._run_context import RunContext
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import (
    AbstractToolset,
    CombinedToolset,
    FunctionToolset,
    PrefixedToolset,
    ToolsetTool,
)
from pydantic_ai.usage import RunUsage
from pydantic_core import SchemaValidator, core_schema
import pytest

from wolfharness.capabilities.mcp_server_cap import McpServerCap
from wolfharness.capabilities.tool_display_capability import ToolDisplayCapability
from wolfharness.capabilities.tool_schema_overlap_capability import (
    APPLIED_METADATA_KEY,
    SchemaOverrideToolset,
    ToolSchemaOverlapCapability,
)
from wolfharness.capabilities.tool_schema_overlap_config import (
    ORIGINAL_TOOL_NAME_METADATA_KEY,
    SERVER_NAME_METADATA_KEY,
)
from wolfharness.tools import FunctionTool


pytestmark = pytest.mark.integration

_CAPABILITY_LOGGER = "wolfharness.capabilities.tool_schema_overlap_capability"

WEATHER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "location": {"type": "string", "description": "City name"},
        "api_key": {"type": "string", "description": "API key"},
        "optional_param": {"type": "string"},
    },
    "required": ["location", "api_key"],
}


def _weather_schema() -> dict[str, Any]:
    return copy.deepcopy(WEATHER_SCHEMA)


def _ctx() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


def _call(tool_name: str, args: dict[str, Any] | None = None) -> ToolCallPart:
    return ToolCallPart(tool_name=tool_name, args=args or {}, tool_call_id=f"call_{tool_name}")


def _make_tool_def(
    name: str,
    *,
    server: str | None = None,
    original: str | None = None,
) -> ToolDefinition:
    metadata: dict[str, Any] | None = None
    if server is not None:
        metadata = {
            SERVER_NAME_METADATA_KEY: server,
            ORIGINAL_TOOL_NAME_METADATA_KEY: original if original is not None else name,
        }
    return ToolDefinition(
        name=name,
        description=f"Original description of {name}",
        parameters_json_schema=_weather_schema(),
        metadata=metadata,
    )


class RecordingToolset(AbstractToolset[Any]):
    """Leaf toolset that records calls instead of executing tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolsetTool[Any]] = {}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def add(self, name: str, *, server: str | None = None, original: str | None = None) -> None:
        self._tools[name] = ToolsetTool(
            toolset=self,
            tool_def=_make_tool_def(name, server=server, original=original),
            max_retries=1,
            args_validator=SchemaValidator(core_schema.any_schema()),
        )

    @property
    def id(self) -> str:
        return "recording"

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        return self._tools

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[Any], tool: ToolsetTool[Any]
    ) -> Any:
        self.calls.append((name, ctx.tool_name or "", dict(tool_args)))
        return "recorded"


@dataclass
class RawMcpTool:
    """Minimal stand-in for an MCP-server tool listing entry."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=_weather_schema)


class RecordingFakeMcpClient:
    """MCPClient double that builds real FunctionTools and records calls."""

    def __init__(self, raw_tools: list[RawMcpTool]) -> None:
        self._raw_tools = raw_tools
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[RawMcpTool]:
        return list(self._raw_tools)

    def convert_tool(self, tool: RawMcpTool) -> FunctionTool[str]:
        calls = self.calls
        raw_name = tool.name

        async def tool_callable(**kwargs: Any) -> str:
            calls.append((raw_name, dict(kwargs)))
            return "ok"

        schema_override = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }
        return FunctionTool.from_callable(
            tool_callable, source="mcp", schema_override=schema_override
        )  # type: ignore[arg-type]


def _display_stack() -> tuple[
    RecordingToolset,
    ToolSchemaOverlapCapability,
    AbstractToolset[Any],
    SchemaOverrideToolset[Any],
]:
    """Build recorder → schema wrapper → display wrapper (tasks 7.5, 7.12)."""
    recorder = RecordingToolset()
    recorder.add("get_weather", server="weather", original="get_weather")
    capability = ToolSchemaOverlapCapability(
        servers={
            "weather": {
                "get_weather": {"name": "fetch_weather", "param_names": {"location": "city"}}
            }
        }
    )
    schema_wrapper = capability.get_wrapper_toolset(recorder)
    assert isinstance(schema_wrapper, SchemaOverrideToolset)
    display = ToolDisplayCapability(name_map={"fetch_weather": "Fetch Weather"})
    outer = display.get_wrapper_toolset(schema_wrapper)
    assert outer is not None
    return recorder, capability, outer, schema_wrapper


class TestDisplayComposition:
    """Tasks 7.5 and 7.12: schema layer composes under the display layer."""

    @pytest.mark.asyncio
    async def test_both_layers_visible(self) -> None:
        _, _, outer, schema_wrapper = _display_stack()
        ctx = _ctx()

        schema_tools = await schema_wrapper.get_tools(ctx)
        display_tools = await outer.get_tools(ctx)

        # Schema layer exposes the model-facing rename with the mutated schema.
        assert set(schema_tools) == {"fetch_weather"}
        # Display layer renames for the UI on top of the schema layer.
        assert set(display_tools) == {"Fetch Weather"}
        tool_def = display_tools["Fetch Weather"].tool_def
        assert "city" in tool_def.parameters_json_schema["properties"]
        # Mutated schema and identity metadata survive the display rename.
        assert tool_def.metadata is not None
        assert tool_def.metadata[SERVER_NAME_METADATA_KEY] == "weather"
        assert tool_def.metadata[ORIGINAL_TOOL_NAME_METADATA_KEY] == "get_weather"
        assert tool_def.metadata[APPLIED_METADATA_KEY] is True

    @pytest.mark.asyncio
    async def test_model_display_upstream_names_round_trip(self) -> None:
        recorder, capability, outer, _ = _display_stack()
        ctx = _ctx()
        display_tools = await outer.get_tools(ctx)
        tool = display_tools["Fetch Weather"]

        async def handler(args: dict[str, Any]) -> Any:
            return await outer.call_tool("Fetch Weather", args, ctx, tool)

        result = await capability.wrap_tool_execute(
            ctx,
            call=_call("Fetch Weather", {"city": "Paris"}),
            tool_def=tool.tool_def,
            args={"city": "Paris"},
            handler=handler,
        )

        assert result == "recorded"
        # Display name in, upstream raw MCP name with original parameters out.
        assert recorder.calls == [("get_weather", "get_weather", {"location": "Paris"})]


class TestPerServerScoping:
    """Tasks 7.6 and 8.4: per-server overrides across overlapping names."""

    @pytest.mark.asyncio
    async def test_two_servers_independent_overrides(self) -> None:
        # Overlapping tool names require a prefix in production; web-a is
        # prefixed, web-b is not. Identity metadata carries the raw names.
        client_a = RecordingFakeMcpClient([RawMcpTool(name="search", description="A search")])
        client_b = RecordingFakeMcpClient([RawMcpTool(name="search", description="B search")])
        cap_a = McpServerCap(MagicMock(), name="web-a", client=client_a, tool_prefix="web_a")  # type: ignore[arg-type]
        cap_b = McpServerCap(MagicMock(), name="web-b", client=client_b)  # type: ignore[arg-type]
        inner_a = await cap_a.get_toolset()(_ctx())
        inner_b = await cap_b.get_toolset()(_ctx())
        assert inner_a is not None
        assert inner_b is not None
        inner = CombinedToolset([inner_a, inner_b])

        capability = ToolSchemaOverlapCapability(
            servers={
                "web-a": {"search": {"name": "web_search", "description": "Search via web-a"}},
                "web-b": {"search": {"description": "Search via web-b"}},
            }
        )
        wrapper = capability.get_wrapper_toolset(inner)
        assert wrapper is not None
        ctx = _ctx()
        tools = await wrapper.get_tools(ctx)

        assert set(tools) == {"web_search", "search"}
        assert tools["web_search"].tool_def.description == "Search via web-a"
        assert tools["search"].tool_def.description == "Search via web-b"

        async def call_a(args: dict[str, Any]) -> Any:
            return await wrapper.call_tool("web_search", args, ctx, tools["web_search"])

        async def call_b(args: dict[str, Any]) -> Any:
            return await wrapper.call_tool("search", args, ctx, tools["search"])

        await capability.wrap_tool_execute(
            ctx,
            call=_call("web_search"),
            tool_def=tools["web_search"].tool_def,
            args={},
            handler=call_a,
        )
        await capability.wrap_tool_execute(
            ctx,
            call=_call("search"),
            tool_def=tools["search"].tool_def,
            args={},
            handler=call_b,
        )

        # Each call reached the correct upstream server with its raw name.
        assert client_a.calls == [("search", {})]
        assert client_b.calls == [("search", {})]


class TestTripleStack:
    """Task 7.17: tool_prefix + schema rename + display rename together."""

    @pytest.mark.asyncio
    async def test_prefix_schema_display_stack(self) -> None:
        recorder = RecordingToolset()
        recorder.add("get_weather", server="weather", original="get_weather")
        prefixed = PrefixedToolset(wrapped=recorder, prefix="weather")
        capability = ToolSchemaOverlapCapability(
            servers={
                "weather": {
                    "get_weather": {"name": "fetch_weather", "param_names": {"location": "city"}}
                }
            }
        )
        schema_wrapper = capability.get_wrapper_toolset(prefixed)
        assert schema_wrapper is not None
        display = ToolDisplayCapability(name_map={"fetch_weather": "Fetch Weather"})
        outer = display.get_wrapper_toolset(schema_wrapper)
        assert outer is not None
        ctx = _ctx()

        schema_tools = await schema_wrapper.get_tools(ctx)
        assert set(schema_tools) == {"fetch_weather"}
        display_tools = await outer.get_tools(ctx)
        assert set(display_tools) == {"Fetch Weather"}

        tool = display_tools["Fetch Weather"]
        properties = tool.tool_def.parameters_json_schema["properties"]
        assert "city" in properties
        assert "location" not in properties

        async def handler(args: dict[str, Any]) -> Any:
            return await outer.call_tool("Fetch Weather", args, ctx, tool)

        result = await capability.wrap_tool_execute(
            ctx,
            call=_call("Fetch Weather", {"city": "Paris"}),
            tool_def=tool.tool_def,
            args={"city": "Paris"},
            handler=handler,
        )

        assert result == "recorded"
        # Leaf sees the raw MCP name and original params; ctx carries them too.
        assert recorder.calls == [("get_weather", "get_weather", {"location": "Paris"})]


class TestRuntimeRoundTrip:
    """Tasks 8.3 and 8.5: rename + param rename + defaults through McpServerCap."""

    @staticmethod
    def _override() -> dict[str, Any]:
        return {
            "name": "fetch_weather",
            "param_names": {"location": "city"},
            "param_removals": ["api_key"],
            "param_overrides": {"api_key": {"default": "sk-default"}},
            "param_additions": {
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                }
            },
        }

    async def _drive(self, prefix: str | None) -> tuple[RecordingFakeMcpClient, ToolsetTool[Any]]:
        raw = RawMcpTool(name="get_weather", description="Get the weather")
        client = RecordingFakeMcpClient([raw])
        server_cap = McpServerCap(MagicMock(), name="weather", client=client, tool_prefix=prefix)  # type: ignore[arg-type]
        inner = await server_cap.get_toolset()(_ctx())
        assert inner is not None

        capability = ToolSchemaOverlapCapability(
            servers={"weather": {"get_weather": self._override()}}
        )
        wrapper = capability.get_wrapper_toolset(inner)
        assert wrapper is not None
        ctx = _ctx()
        tools = await wrapper.get_tools(ctx)

        assert set(tools) == {"fetch_weather"}
        tool = tools["fetch_weather"]
        assert tool.tool_def.name == "fetch_weather"
        # The override removes only api_key; optional_param is not removed, so it stays.
        assert set(tool.tool_def.parameters_json_schema["properties"]) == {
            "city",
            "optional_param",
            "units",
        }
        assert tool.tool_def.metadata is not None
        assert tool.tool_def.metadata[APPLIED_METADATA_KEY] is True

        async def handler(args: dict[str, Any]) -> Any:
            return await wrapper.call_tool("fetch_weather", args, ctx, tool)

        result = await capability.wrap_tool_execute(
            ctx,
            call=_call("fetch_weather", {"city": "Paris"}),
            tool_def=tool.tool_def,
            args={"city": "Paris"},
            handler=handler,
        )

        assert result == "ok"
        assert client.calls == [
            ("get_weather", {"location": "Paris", "api_key": "sk-default", "units": "celsius"})
        ]
        return client, tool

    @pytest.mark.asyncio
    async def test_round_trip_without_prefix(self) -> None:
        await self._drive(prefix=None)

    @pytest.mark.asyncio
    async def test_round_trip_with_prefix(self) -> None:
        _, tool = await self._drive(prefix="weather")
        # Identity survives the prefix wrapper and reaches the model layer.
        assert tool.tool_def.metadata is not None
        assert tool.tool_def.metadata[SERVER_NAME_METADATA_KEY] == "weather"
        assert tool.tool_def.metadata[ORIGINAL_TOOL_NAME_METADATA_KEY] == "get_weather"


class TestDegradedPipeline:
    """Task 8.6: missing identity metadata degrades to pass-through."""

    @pytest.mark.asyncio
    async def test_missing_identity_passes_through_and_warns(self, caplog) -> None:
        async def noop(**kwargs: Any) -> str:
            return "ok"

        # Stamping bypassed: a FunctionTool without identity metadata.
        wolf_tool = FunctionTool(
            name="get_weather",
            description="Original description of get_weather",
            schema_override={
                "name": "get_weather",
                "description": "Original description of get_weather",
                "parameters": _weather_schema(),
            },  # type: ignore[arg-type]
            callable=noop,
            source="mcp",
        )
        inner = FunctionToolset([wolf_tool.to_pydantic_ai()])
        capability = ToolSchemaOverlapCapability(
            servers={"weather": {"get_weather": {"name": "fetch_weather"}}}
        )
        wrapper = capability.get_wrapper_toolset(inner)
        assert wrapper is not None
        ctx = _ctx()

        with caplog.at_level(logging.WARNING, logger=_CAPABILITY_LOGGER):
            tools = await wrapper.get_tools(ctx)

        # No rename, no mutation — the tool passes through unchanged.
        assert set(tools) == {"get_weather"}
        tool_def = tools["get_weather"].tool_def
        assert tool_def.name == "get_weather"
        assert tool_def.description == "Original description of get_weather"
        assert "location" in tool_def.parameters_json_schema["properties"]
        assert any("tool-schema-overlap" in record.message for record in caplog.records)

        received: dict[str, Any] | None = None

        async def handler(args: dict[str, Any]) -> Any:
            nonlocal received
            received = dict(args)
            return "ran"

        result = await capability.wrap_tool_execute(
            ctx,
            call=_call("get_weather", {"location": "Paris"}),
            tool_def=tool_def,
            args={"location": "Paris"},
            handler=handler,
        )

        assert result == "ran"
        assert received == {"location": "Paris"}
