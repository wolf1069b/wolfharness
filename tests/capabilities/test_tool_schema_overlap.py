"""Unit tests for ToolSchemaOverlapCapability and SchemaOverrideToolset.

Covers identity-driven schema transformation, first-listing fail-fast
validation, call routing, composition with ``tool_prefix``, degradation
when source identity metadata is absent, and the metadata survival chain
from ``McpServerCap`` through the toolset wrappers.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pydantic import ValidationError
from pydantic_ai._run_context import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import (
    AbstractToolset,
    CombinedToolset,
    FunctionToolset,
    PrefixedToolset,
    RenamedToolset,
    ToolsetTool,
)
from pydantic_ai.usage import RunUsage
from pydantic_core import SchemaValidator, core_schema
import pytest

from wolfharness.capabilities.mcp_server_cap import McpServerCap
from wolfharness.capabilities.tool_schema_overlap_capability import (
    APPLIED_METADATA_KEY,
    SchemaOverrideToolset,
    ToolSchemaOverlapCapability,
)
from wolfharness.capabilities.tool_schema_overlap_config import (
    ORIGINAL_TOOL_NAME_METADATA_KEY,
    SERVER_NAME_METADATA_KEY,
    ToolSchemaOverlapConfig,
)
from wolfharness.tools import FunctionTool


pytestmark = pytest.mark.unit

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
    description: str | None = None,
    schema: dict[str, Any] | None = None,
) -> ToolDefinition:
    metadata: dict[str, Any] | None = None
    if server is not None:
        metadata = {
            SERVER_NAME_METADATA_KEY: server,
            ORIGINAL_TOOL_NAME_METADATA_KEY: original if original is not None else name,
        }
    return ToolDefinition(
        name=name,
        description=description if description is not None else f"Original description of {name}",
        parameters_json_schema=schema if schema is not None else _weather_schema(),
        metadata=metadata,
    )


def _assert_identity(tool_def: ToolDefinition, server: str, original: str) -> None:
    assert tool_def.metadata is not None
    assert tool_def.metadata[SERVER_NAME_METADATA_KEY] == server
    assert tool_def.metadata[ORIGINAL_TOOL_NAME_METADATA_KEY] == original


class RecordingToolset(AbstractToolset[Any]):
    """Leaf toolset that records calls instead of executing tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolsetTool[Any]] = {}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def add(
        self,
        name: str,
        *,
        server: str | None = None,
        original: str | None = None,
        schema: dict[str, Any] | None = None,
    ) -> None:
        self._tools[name] = ToolsetTool(
            toolset=self,
            tool_def=_make_tool_def(name, server=server, original=original, schema=schema),
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


def _make_function_tool(
    name: str, *, server: str | None = None, callable_: Any = None
) -> FunctionTool[str]:
    async def _noop(**kwargs: Any) -> str:
        return "ok"

    tool = FunctionTool(
        name=name,
        description=f"Original description of {name}",
        schema_override={
            "name": name,
            "description": f"Original description of {name}",
            "parameters": _weather_schema(),
        },  # type: ignore[arg-type]
        callable=callable_ if callable_ is not None else _noop,
        source="mcp",
    )
    if server is not None:
        tool.metadata[SERVER_NAME_METADATA_KEY] = server
        tool.metadata[ORIGINAL_TOOL_NAME_METADATA_KEY] = name
    return tool


@dataclass
class RawMcpTool:
    """Minimal stand-in for an MCP-server tool listing entry."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=_weather_schema)


class FakeMcpClient:
    """MCPClient double whose ``convert_tool`` returns real FunctionTools."""

    def __init__(self, raw_tools: list[RawMcpTool]) -> None:
        self._raw_tools = raw_tools

    async def list_tools(self) -> list[RawMcpTool]:
        return list(self._raw_tools)

    def convert_tool(self, tool: RawMcpTool) -> FunctionTool[str]:
        async def tool_callable(**kwargs: Any) -> str:
            return "ok"

        schema_override = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }
        return FunctionTool.from_callable(
            tool_callable,
            source="mcp",
            schema_override=schema_override,  # type: ignore[arg-type]
        )


class TestMetadataSurvival:
    """Task 2.4: identity metadata survives the full tool pipeline chain."""

    async def _build_prefixed_toolset(self) -> PrefixedToolset[Any]:
        raw = RawMcpTool(name="get_weather", description="Get weather")
        cap = McpServerCap(
            MagicMock(),
            name="weather",
            client=FakeMcpClient([raw]),  # type: ignore[arg-type]
            tool_prefix="w",
        )
        builder = cap.get_toolset()
        toolset = await builder(_ctx())
        assert isinstance(toolset, PrefixedToolset)
        return toolset

    @pytest.mark.asyncio
    async def test_stamped_after_convert(self) -> None:
        toolset = await self._build_prefixed_toolset()
        inner_tools = await toolset.wrapped.get_tools(_ctx())
        assert set(inner_tools) == {"get_weather"}
        # Survives FunctionToolset.get_tools (including the schema-override prepare).
        _assert_identity(inner_tools["get_weather"].tool_def, "weather", "get_weather")

    @pytest.mark.asyncio
    async def test_survives_prefix(self) -> None:
        toolset = await self._build_prefixed_toolset()
        tools = await toolset.get_tools(_ctx())
        assert set(tools) == {"w_get_weather"}
        _assert_identity(tools["w_get_weather"].tool_def, "weather", "get_weather")

    @pytest.mark.asyncio
    async def test_survives_rename(self) -> None:
        toolset = await self._build_prefixed_toolset()
        renamed = RenamedToolset(wrapped=toolset, name_map={"aliased": "w_get_weather"})
        tools = await renamed.get_tools(_ctx())
        assert set(tools) == {"aliased"}
        _assert_identity(tools["aliased"].tool_def, "weather", "get_weather")


class TestSchemaTransformation:
    """Task 7.2: model-facing schema mutations resolved by identity."""

    @pytest.mark.asyncio
    async def test_full_override_transforms_schema(self) -> None:
        recorder = RecordingToolset()
        recorder.add("get_weather", server="weather")
        capability = ToolSchemaOverlapCapability(
            servers={
                "weather": {
                    "get_weather": {
                        "name": "fetch_weather",
                        "description": "Rewritten description",
                        "param_names": {"location": "city"},
                        "param_descriptions": {"location": "City name and country code"},
                        "param_overrides": {
                            "api_key": {"default": "sk-default"},
                            "optional_param": {"enum": ["a", "b"]},
                        },
                        "param_removals": ["api_key"],
                        "param_additions": {
                            "units": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Temperature unit",
                                "default": "celsius",
                            }
                        },
                    }
                }
            }
        )
        wrapper = SchemaOverrideToolset[Any](wrapped=recorder, config=capability.config)
        tools = await wrapper.get_tools(_ctx())

        assert set(tools) == {"fetch_weather"}
        tool_def = tools["fetch_weather"].tool_def
        assert tool_def.name == "fetch_weather"
        assert tool_def.description == "Rewritten description"
        schema = tool_def.parameters_json_schema
        assert set(schema["properties"]) == {"city", "optional_param", "units"}
        assert schema["required"] == ["city"]
        assert schema["properties"]["city"]["description"] == "City name and country code"
        assert "api_key" not in schema["properties"]
        assert schema["properties"]["optional_param"]["enum"] == ["a", "b"]
        assert schema["properties"]["units"]["default"] == "celsius"
        # Identity preserved and application marked.
        _assert_identity(tool_def, "weather", "get_weather")
        assert tool_def.metadata is not None
        assert tool_def.metadata[APPLIED_METADATA_KEY] is True
        # Routing table maps model name back to the inner visible name.
        assert wrapper.routing == {"fetch_weather": "get_weather"}
        # Inner tool untouched (mutations applied on a copy).
        inner_schema = recorder._tools["get_weather"].tool_def.parameters_json_schema
        assert inner_schema == _weather_schema()

    @pytest.mark.asyncio
    async def test_description_only_override_keeps_name(self) -> None:
        recorder = RecordingToolset()
        recorder.add("get_weather", server="weather")
        capability = ToolSchemaOverlapCapability(
            servers={"weather": {"get_weather": {"description": "Only description"}}}
        )
        wrapper = SchemaOverrideToolset[Any](wrapped=recorder, config=capability.config)
        tools = await wrapper.get_tools(_ctx())
        assert set(tools) == {"get_weather"}
        assert tools["get_weather"].tool_def.description == "Only description"
        # Renamed-less overrides still record an identity routing entry.
        assert wrapper.routing == {"get_weather": "get_weather"}

    @pytest.mark.asyncio
    async def test_identity_matched_toolset_tools(self) -> None:
        """Overrides resolve through real FunctionToolset tool definitions."""
        tool = _make_function_tool("get_weather", server="weather")
        inner = FunctionToolset([tool.to_pydantic_ai()])
        capability = ToolSchemaOverlapCapability(
            servers={"weather": {"get_weather": {"name": "fetch_weather"}}}
        )
        wrapper = SchemaOverrideToolset[Any](wrapped=inner, config=capability.config)
        tools = await wrapper.get_tools(_ctx())
        assert set(tools) == {"fetch_weather"}
        _assert_identity(tools["fetch_weather"].tool_def, "weather", "get_weather")

    @pytest.mark.asyncio
    async def test_global_override_applies_to_all_servers(self) -> None:
        recorder = RecordingToolset()
        recorder.add("accu_get_weather", server="weather-accu", original="get_weather")
        recorder.add("wmo_get_weather", server="weather-wmo", original="get_weather")
        capability = ToolSchemaOverlapCapability(
            global_overrides={"get_weather": {"description": "Shared description"}}
        )
        wrapper = SchemaOverrideToolset[Any](wrapped=recorder, config=capability.config)
        tools = await wrapper.get_tools(_ctx())
        assert tools["accu_get_weather"].tool_def.description == "Shared description"
        assert tools["wmo_get_weather"].tool_def.description == "Shared description"

    @pytest.mark.asyncio
    async def test_server_scope_takes_precedence_over_global(self) -> None:
        recorder = RecordingToolset()
        recorder.add("accu_get_weather", server="weather-accu", original="get_weather")
        recorder.add("wmo_get_weather", server="weather-wmo", original="get_weather")
        capability = ToolSchemaOverlapCapability(
            servers={"weather-accu": {"get_weather": {"description": "Accu only"}}},
            global_overrides={"get_weather": {"description": "Shared description"}},
        )
        wrapper = SchemaOverrideToolset[Any](wrapped=recorder, config=capability.config)
        tools = await wrapper.get_tools(_ctx())
        assert tools["accu_get_weather"].tool_def.description == "Accu only"
        assert tools["wmo_get_weather"].tool_def.description == "Shared description"


class TestPrepareTools:
    """Task 7.3: supplementary per-step ToolDefinition modification."""

    @pytest.mark.asyncio
    async def test_applies_override_and_marks(self) -> None:
        capability = ToolSchemaOverlapCapability(
            servers={"weather": {"get_weather": {"description": "New description"}}}
        )
        tool_def = _make_tool_def("get_weather", server="weather")
        prepared = await capability.prepare_tools(_ctx(), [tool_def])
        assert prepared[0].description == "New description"
        assert prepared[0].metadata is not None
        assert prepared[0].metadata[APPLIED_METADATA_KEY] is True

    @pytest.mark.asyncio
    async def test_second_application_is_skipped(self) -> None:
        capability = ToolSchemaOverlapCapability(
            servers={"weather": {"get_weather": {"description": "New description"}}}
        )
        tool_def = _make_tool_def("get_weather", server="weather")
        prepared = await capability.prepare_tools(_ctx(), [tool_def])
        prepared_again = await capability.prepare_tools(_ctx(), prepared)
        assert prepared_again[0] is prepared[0]

    @pytest.mark.asyncio
    async def test_unidentified_tool_passes_through(self) -> None:
        capability = ToolSchemaOverlapCapability(
            servers={"weather": {"get_weather": {"description": "New description"}}}
        )
        anonymous = _make_tool_def("get_weather")
        prepared = await capability.prepare_tools(_ctx(), [anonymous])
        assert prepared[0] is anonymous


class TestWrapToolExecute:
    """Task 7.4: runtime parameter desharing and default injection."""

    @pytest.mark.asyncio
    async def test_deshares_renames_and_injects_defaults(self) -> None:
        capability = ToolSchemaOverlapCapability(
            servers={
                "weather": {
                    "get_weather": {
                        "param_names": {"location": "city"},
                        "param_overrides": {"api_key": {"default": "sk-default"}},
                        "param_removals": ["api_key"],
                        "param_additions": {"units": {"type": "string", "default": "celsius"}},
                    }
                }
            }
        )
        tool_def = _make_tool_def("fetch_weather", server="weather", original="get_weather")
        handler = AsyncMock(return_value="ok")
        await capability.wrap_tool_execute(
            _ctx(),
            call=_call("fetch_weather", {"city": "Paris"}),
            tool_def=tool_def,
            args={"city": "Paris"},
            handler=handler,
        )
        handler.assert_awaited_once_with({
            "location": "Paris",
            "api_key": "sk-default",
            "units": "celsius",
        })

    @pytest.mark.asyncio
    async def test_dropped_removed_param_without_default_not_forwarded(self) -> None:
        capability = ToolSchemaOverlapCapability(
            servers={
                "weather": {
                    "get_weather": {
                        "param_removals": ["optional_param"],
                    }
                }
            }
        )
        tool_def = _make_tool_def("get_weather", server="weather")
        handler = AsyncMock(return_value="ok")
        await capability.wrap_tool_execute(
            _ctx(),
            call=_call("get_weather", {"location": "Paris", "optional_param": "x"}),
            tool_def=tool_def,
            args={"location": "Paris", "optional_param": "x"},
            handler=handler,
        )
        handler.assert_awaited_once_with({"location": "Paris"})

    @pytest.mark.asyncio
    async def test_unidentified_tool_args_pass_through(self) -> None:
        capability = ToolSchemaOverlapCapability(
            servers={"weather": {"get_weather": {"param_names": {"location": "city"}}}}
        )
        anonymous = _make_tool_def("get_weather")
        handler = AsyncMock(return_value="ok")
        await capability.wrap_tool_execute(
            _ctx(),
            call=_call("get_weather", {"location": "Paris"}),
            tool_def=anonymous,
            args={"location": "Paris"},
            handler=handler,
        )
        handler.assert_awaited_once_with({"location": "Paris"})

    @pytest.mark.asyncio
    async def test_tool_without_override_passes_through(self) -> None:
        capability = ToolSchemaOverlapCapability(
            servers={"weather": {"get_weather": {"param_names": {"location": "city"}}}}
        )
        tool_def = _make_tool_def("other_tool", server="weather", original="other_tool")
        handler = AsyncMock(return_value="ok")
        await capability.wrap_tool_execute(
            _ctx(),
            call=_call("other_tool", {"location": "Paris"}),
            tool_def=tool_def,
            args={"location": "Paris"},
            handler=handler,
        )
        handler.assert_awaited_once_with({"location": "Paris"})


class TestFirstListingFailFast:
    """Tasks 7.9/7.10/7.14: schema-dependent validation at agent startup."""

    @pytest.mark.asyncio
    async def test_required_removal_without_default_raises(self) -> None:
        recorder = RecordingToolset()
        recorder.add("get_weather", server="weather")
        capability = ToolSchemaOverlapCapability(
            servers={"weather": {"get_weather": {"param_removals": ["api_key"]}}}
        )
        wrapper = SchemaOverrideToolset[Any](wrapped=recorder, config=capability.config)
        with pytest.raises(ValidationError) as exc_info:
            await wrapper.get_tools(_ctx())
        message = str(exc_info.value)
        assert "weather" in message
        assert "get_weather" in message
        assert "api_key" in message
        assert "without a configured default" in message

    @pytest.mark.asyncio
    async def test_required_removal_with_default_allowed(self) -> None:
        recorder = RecordingToolset()
        recorder.add("get_weather", server="weather")
        capability = ToolSchemaOverlapCapability(
            servers={
                "weather": {
                    "get_weather": {
                        "param_removals": ["api_key"],
                        "param_overrides": {"api_key": {"default": "sk-default"}},
                    }
                }
            }
        )
        wrapper = SchemaOverrideToolset[Any](wrapped=recorder, config=capability.config)
        tools = await wrapper.get_tools(_ctx())
        assert "api_key" not in tools["get_weather"].tool_def.parameters_json_schema["properties"]

    @pytest.mark.asyncio
    async def test_param_rename_onto_existing_property_raises(self) -> None:
        recorder = RecordingToolset()
        recorder.add(
            "get_weather",
            server="weather",
            schema={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "city": {"type": "string"},
                },
                "required": ["location"],
            },
        )
        capability = ToolSchemaOverlapCapability(
            servers={"weather": {"get_weather": {"param_names": {"location": "city"}}}}
        )
        wrapper = SchemaOverrideToolset[Any](wrapped=recorder, config=capability.config)
        with pytest.raises(ValidationError, match="renames parameter"):
            await wrapper.get_tools(_ctx())

    @pytest.mark.asyncio
    async def test_misspelled_tool_key_raises(self) -> None:
        recorder = RecordingToolset()
        recorder.add("get_weather", server="weather")
        capability = ToolSchemaOverlapCapability(
            servers={"weather": {"get_weatherr": {"description": "typo"}}}
        )
        wrapper = SchemaOverrideToolset[Any](wrapped=recorder, config=capability.config)
        with pytest.raises(ValidationError, match="does not exist on server"):
            await wrapper.get_tools(_ctx())

    @pytest.mark.asyncio
    async def test_misspelled_server_key_raises_when_pipeline_healthy(self) -> None:
        recorder = RecordingToolset()
        recorder.add("get_weather", server="weather")
        capability = ToolSchemaOverlapCapability(
            servers={"weatherr": {"get_weather": {"description": "typo"}}}
        )
        wrapper = SchemaOverrideToolset[Any](wrapped=recorder, config=capability.config)
        with pytest.raises(ValidationError, match="matched no listed tool"):
            await wrapper.get_tools(_ctx())

    @pytest.mark.asyncio
    async def test_unmatched_global_key_raises_when_pipeline_healthy(self) -> None:
        recorder = RecordingToolset()
        recorder.add("get_weather", server="weather")
        capability = ToolSchemaOverlapCapability(
            global_overrides={"nonexistent": {"description": "typo"}}
        )
        wrapper = SchemaOverrideToolset[Any](wrapped=recorder, config=capability.config)
        with pytest.raises(ValidationError, match="matched no listed tool"):
            await wrapper.get_tools(_ctx())

    def test_enum_default_inconsistency_rejected_at_construction(self) -> None:
        with pytest.raises(ValidationError, match="not one of the configured enum values"):
            ToolSchemaOverlapCapability(
                servers={
                    "weather": {
                        "get_weather": {
                            "param_overrides": {"format": {"enum": ["json"], "default": "xml"}}
                        }
                    }
                }
            )


class TestAdversarialRouting:
    """Tasks 3.4/7.7/7.8/7.13: routing and conflict semantics."""

    @pytest.mark.asyncio
    async def test_call_tool_routes_model_name_to_inner_name(self) -> None:
        recorder = RecordingToolset()
        recorder.add("get_weather", server="weather")
        capability = ToolSchemaOverlapCapability(
            servers={"weather": {"get_weather": {"name": "fetch_weather"}}}
        )
        wrapper = SchemaOverrideToolset[Any](wrapped=recorder, config=capability.config)
        tools = await wrapper.get_tools(_ctx())
        await wrapper.call_tool(
            "fetch_weather", {"location": "Paris"}, _ctx(), tools["fetch_weather"]
        )
        assert recorder.calls == [("get_weather", "get_weather", {"location": "Paris"})]

    @pytest.mark.asyncio
    async def test_unrenamed_tool_call_delegates_unchanged(self) -> None:
        recorder = RecordingToolset()
        recorder.add("get_weather", server="weather")
        recorder.add("other_tool", server="weather", original="other_tool")
        capability = ToolSchemaOverlapCapability(
            servers={"weather": {"get_weather": {"name": "fetch_weather"}}}
        )
        wrapper = SchemaOverrideToolset[Any](wrapped=recorder, config=capability.config)
        tools = await wrapper.get_tools(_ctx())
        await wrapper.call_tool("other_tool", {"x": 1}, _ctx(), tools["other_tool"])
        assert recorder.calls == [("other_tool", "", {"x": 1})]

    @pytest.mark.asyncio
    async def test_two_servers_same_tool_name_calls_do_not_cross(self) -> None:
        recorder_a = RecordingToolset()
        recorder_a.add("web_a_search", server="web-a", original="search")
        recorder_b = RecordingToolset()
        recorder_b.add("search", server="web-b", original="search")
        combined = CombinedToolset([recorder_a, recorder_b])
        capability = ToolSchemaOverlapCapability(
            servers={"web-a": {"search": {"name": "web_search"}}}
        )
        wrapper = SchemaOverrideToolset[Any](wrapped=combined, config=capability.config)
        tools = await wrapper.get_tools(_ctx())
        assert set(tools) == {"web_search", "search"}
        _assert_identity(tools["web_search"].tool_def, "web-a", "search")
        _assert_identity(tools["search"].tool_def, "web-b", "search")

        await wrapper.call_tool("web_search", {"query": "a"}, _ctx(), tools["web_search"])
        await wrapper.call_tool("search", {"query": "b"}, _ctx(), tools["search"])
        assert recorder_a.calls == [("web_a_search", "web_a_search", {"query": "a"})]
        assert recorder_b.calls == [("search", "", {"query": "b"})]

    @pytest.mark.asyncio
    async def test_rename_onto_existing_tool_name_raises_user_error(self) -> None:
        recorder = RecordingToolset()
        recorder.add("get_weather", server="weather")
        recorder.add("fetch_weather", server="other")
        capability = ToolSchemaOverlapCapability(
            servers={"weather": {"get_weather": {"name": "fetch_weather"}}}
        )
        wrapper = SchemaOverrideToolset[Any](wrapped=recorder, config=capability.config)
        # The renamed tool is processed first; the pre-existing tool with the
        # same name then collides as an un-renamed entry (RenamedToolset
        # semantics mirrored verbatim).
        with pytest.raises(UserError, match="conflicts with previously renamed tool"):
            await wrapper.get_tools(_ctx())

    @pytest.mark.asyncio
    async def test_listing_is_idempotent(self) -> None:
        recorder = RecordingToolset()
        recorder.add("get_weather", server="weather")
        capability = ToolSchemaOverlapCapability(
            servers={
                "weather": {
                    "get_weather": {
                        "name": "fetch_weather",
                        "param_names": {"location": "city"},
                        "param_additions": {"units": {"type": "string", "default": "celsius"}},
                    }
                }
            }
        )
        wrapper = SchemaOverrideToolset[Any](wrapped=recorder, config=capability.config)
        first = await wrapper.get_tools(_ctx())
        second = await wrapper.get_tools(_ctx())
        assert (
            first["fetch_weather"].tool_def.parameters_json_schema
            == second["fetch_weather"].tool_def.parameters_json_schema
        )
        assert first["fetch_weather"].tool_def.name == second["fetch_weather"].tool_def.name
        # The inner schema is never cumulatively mutated.
        assert recorder._tools["get_weather"].tool_def.parameters_json_schema == _weather_schema()


class TestPrefixComposition:
    """Task 7.15: overrides resolve by raw name under tool_prefix."""

    @pytest.mark.asyncio
    async def test_prefixed_server_override_renames_and_routes(self) -> None:
        recorder = RecordingToolset()
        # The leaf holds the raw tool name; PrefixedToolset applies the prefix.
        recorder.add("get_weather", server="weather", original="get_weather")
        prefixed = PrefixedToolset(wrapped=recorder, prefix="weather")
        capability = ToolSchemaOverlapCapability(
            servers={
                "weather": {
                    "get_weather": {"name": "fetch_weather", "param_names": {"location": "city"}}
                }
            }
        )
        wrapper = SchemaOverrideToolset[Any](wrapped=prefixed, config=capability.config)
        tools = await wrapper.get_tools(_ctx())
        # The model sees the final renamed schema, not the prefixed raw form.
        assert set(tools) == {"fetch_weather"}
        properties = tools["fetch_weather"].tool_def.parameters_json_schema["properties"]
        assert "city" in properties
        assert "location" not in properties

        tool = tools["fetch_weather"]

        async def handler(args: dict[str, Any]) -> Any:
            return await wrapper.call_tool("fetch_weather", args, _ctx(), tool)

        await capability.wrap_tool_execute(
            _ctx(),
            call=_call("fetch_weather", {"city": "Paris"}),
            tool_def=tool.tool_def,
            args={"city": "Paris"},
            handler=handler,
        )
        # The leaf receives the raw tool name with original parameter names.
        assert recorder.calls == [("get_weather", "get_weather", {"location": "Paris"})]


class TestDegradation:
    """Task 7.16: tools without identity metadata are never guessed."""

    @pytest.mark.asyncio
    async def test_unidentified_tool_passes_through_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        recorder = RecordingToolset()
        recorder.add("get_weather")  # no identity metadata
        capability = ToolSchemaOverlapCapability(
            servers={"weather": {"get_weather": {"description": "Should not apply"}}}
        )
        wrapper = SchemaOverrideToolset[Any](wrapped=recorder, config=capability.config)
        with caplog.at_level(logging.WARNING, logger=_CAPABILITY_LOGGER):
            tools = await wrapper.get_tools(_ctx())
        assert set(tools) == {"get_weather"}
        # Without identity metadata the tool passes through unchanged — the
        # server-scoped override must never be applied by name-guessing.
        assert tools["get_weather"].tool_def.description == "Original description of get_weather"
        assert tools["get_weather"].tool_def.name == "get_weather"
        assert wrapper.routing == {}
        assert any("tool-schema-overlap" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_no_cross_server_bleed_on_partial_identity(self) -> None:
        recorder = RecordingToolset()
        recorder.add("accu_get_weather", server="weather-accu", original="get_weather")
        recorder.add("wmo_get_weather")  # identity lost
        capability = ToolSchemaOverlapCapability(
            servers={"weather-accu": {"get_weather": {"description": "Accu override"}}}
        )
        wrapper = SchemaOverrideToolset[Any](wrapped=recorder, config=capability.config)
        tools = await wrapper.get_tools(_ctx())
        assert tools["accu_get_weather"].tool_def.description == "Accu override"
        assert (
            tools["wmo_get_weather"].tool_def.description
            == "Original description of wmo_get_weather"
        )

    @pytest.mark.asyncio
    async def test_wrap_tool_execute_ignores_unidentified_tools(self) -> None:
        capability = ToolSchemaOverlapCapability(
            servers={"weather": {"get_weather": {"param_names": {"location": "city"}}}}
        )
        anonymous = _make_tool_def("weather_get_weather")
        handler = AsyncMock(return_value="ok")
        await capability.wrap_tool_execute(
            _ctx(),
            call=_call("weather_get_weather", {"location": "Paris"}),
            tool_def=anonymous,
            args={"location": "Paris"},
            handler=handler,
        )
        handler.assert_awaited_once_with({"location": "Paris"})


class TestConfigParsing:
    """Capability construction accepts raw nested dicts (YAML args shape)."""

    def test_raw_dicts_are_parsed_into_config(self) -> None:
        capability = ToolSchemaOverlapCapability(
            servers={"weather": {"get_weather": {"name": "fetch_weather"}}},
            global_overrides={},
        )
        assert isinstance(capability.config, ToolSchemaOverlapConfig)
        assert capability.config.servers["weather"]["get_weather"].name == "fetch_weather"

    def test_invalid_config_raises_at_construction(self) -> None:
        with pytest.raises(ValidationError):
            ToolSchemaOverlapCapability(servers={"weather": {"get_weather": {"unknown_field": 1}}})

    def test_ordering_wrapped_by_display_capability(self) -> None:
        from wolfharness.capabilities.tool_display_capability import ToolDisplayCapability

        ordering = ToolSchemaOverlapCapability().get_ordering()
        assert ordering is not None
        assert ToolDisplayCapability in ordering.wrapped_by
