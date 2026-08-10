"""Integration tests for ToolDisplayCapability — global decoration of assembled tools.

Validates that ToolDisplayCapability, wired alongside a concrete
capability (viking), decorates exactly the tools in ``name_map`` /
``emit_diff_for`` without altering the underlying tool semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets.renamed import RenamedToolset
import pytest

from wolfharness.capabilities.tool_display_capability import ToolDisplayCapability
from wolfharness.capabilities.viking import VikingCapability
from wolfharness.capabilities.viking.identity import VikingIdentity
from wolfharness_config.capabilities import VikingCapabilityConfig, build_capability


if TYPE_CHECKING:
    from wolfharness.agents.events import DiffContentItem


pytestmark = pytest.mark.integration


def _make_mock_client() -> AsyncMock:
    """Create a fully populated mock AsyncHTTPClient for the viking capability."""
    client = AsyncMock()
    client.initialize = AsyncMock()
    client.close = AsyncMock()
    client.read = AsyncMock(return_value="file content")
    client.write = AsyncMock(return_value={"status": "ok"})
    client.search = AsyncMock(return_value={"results": []})
    client.find = AsyncMock(return_value={"results": []})
    return client


def _build_viking_cap(client: AsyncMock) -> VikingCapability:
    """Build a viking capability from config with mocked client."""
    config = VikingCapabilityConfig(type="viking", mode="write")
    cap = build_capability(config)
    assert isinstance(cap, VikingCapability)
    cap._client = client
    cap._identity = VikingIdentity(account_id="acc", user_id="user")  # type: ignore[arg-type]
    return cap


def _make_request_context(messages: list[Any]) -> ModelRequestContext:
    """Build a minimal ModelRequestContext with a real TestModel."""
    return ModelRequestContext(
        model=TestModel(),
        messages=messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            native_tools=[],
        ),
    )


def test_viking_tools_are_wrapped_after_assembly() -> None:
    """Assembling viking + tool_display gives renamed tools via get_wrapper_toolset."""
    cap = _build_viking_cap(_make_mock_client())
    toolset = cap.get_toolset()

    display = ToolDisplayCapability(
        rename_mode=True,
        name_map={"viking_write": "write"},
        emit_diff=True,
        emit_diff_for={"viking_write", "viking_edit"},
    )

    wrapped = display.get_wrapper_toolset(toolset)  # type: ignore[arg-type]
    assert wrapped is not None
    assert isinstance(wrapped, RenamedToolset)


def test_assembled_rename_changes_tool_names() -> None:
    """RenamedToolset applied over viking write-mode tools renames viking_write."""
    cap = _build_viking_cap(_make_mock_client())
    toolset = cap.get_toolset()

    # RenamedToolset renames ToolDefinitions during preparation; the wrapper
    # carries the name_map regardless of runtime tool listing.
    display = ToolDisplayCapability(
        rename_mode=True,
        name_map={"viking_write": "write"},
        emit_diff=False,
    )
    wrapped = display.get_wrapper_toolset(toolset)  # type: ignore[arg-type]
    assert wrapped is not None

    assert isinstance(wrapped, RenamedToolset)
    # name_map is {original: display} in config, inverted to {display: original} for RenamedToolset
    assert "write" in wrapped.name_map


def test_wrap_execute_passes_through_for_untouched_tool() -> None:
    """A viking tool outside emit_diff_for executes unchanged with no event."""
    events = AsyncMock()
    deps = MagicMock()
    deps.events = events
    ctx = MagicMock()
    ctx.deps = deps

    display = ToolDisplayCapability(emit_diff=True, emit_diff_for={"viking_write"})
    handler = AsyncMock(return_value="viking_read result")

    import asyncio

    async def run() -> str:
        return await display.wrap_tool_execute(  # type: ignore[return-value]
            ctx,
            call=MagicMock(tool_name="viking_read", tool_call_id="c1"),
            tool_def=MagicMock(name="viking_read"),
            args={"uri": "viking://x.md"},
            handler=handler,  # type: ignore[arg-type]
        )

    result = asyncio.run(run())
    assert result == "viking_read result"
    handler.assert_awaited_once()
    events.tool_call_progress.assert_not_awaited()


def test_wrap_execute_injects_diff_for_match() -> None:
    """A viking tool in emit_diff_for gets a DiffContentItem event."""
    events = AsyncMock()
    deps = MagicMock()
    deps.events = events
    ctx = MagicMock()
    ctx.deps = deps

    display = ToolDisplayCapability(
        rename_mode=True,
        name_map={"viking_write": "write"},
        emit_diff=True,
        emit_diff_for={"viking_write", "viking_edit"},
    )
    handler = AsyncMock(return_value="Wrote 3 chars.")

    import asyncio

    async def run() -> str:
        return await display.wrap_tool_execute(  # type: ignore[return-value]
            ctx,
            call=MagicMock(tool_name="viking_write", tool_call_id="c1"),
            tool_def=MagicMock(name="viking_write"),
            args={"uri": "viking://x.md", "content": "abc"},
            handler=handler,  # type: ignore[arg-type]
        )

    result = asyncio.run(run())
    assert result == "Wrote 3 chars."
    events.tool_call_progress.assert_awaited_once()
    items: list[DiffContentItem] = events.tool_call_progress.await_args.kwargs["items"]
    assert items[0].path == "viking://x.md"
    assert items[0].new_text == "abc"
