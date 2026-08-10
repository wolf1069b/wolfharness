"""Semantic tests for ToolDisplayCapability — rename output, event identity, cross-layer matching.

These tests verify the three core behavioral contracts of the capability:

1. **Rename produces display names**: ``get_tools()`` on the wrapped toolset
   returns tools keyed by the *display* name, not the original.
2. **Injected events carry real tool_call_id**: ``wrap_tool_execute`` populates
   ``ctx.deps.tool_call_id`` before emitting, so events aren't dropped by
   downstream converters (capability tools skip ``tool_wrapping.py``).
3. **emit_diff_for matches across rename boundary**: when the model calls a
   display name, the capability resolves it back to the original for
   ``emit_diff_for`` matching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, RenamedToolset
import pytest

from wolfharness.capabilities.tool_display_capability import ToolDisplayCapability


if TYPE_CHECKING:
    from wolfharness.agents.events import DiffContentItem


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_call(tool_name: str, tool_call_id: str = "call_123") -> ToolCallPart:
    """Create a ToolCallPart with a real tool_call_id."""
    return ToolCallPart(tool_name=tool_name, args={}, tool_call_id=tool_call_id)


def _make_tool_def(name: str) -> ToolDefinition:
    return ToolDefinition(name=name, description="test tool")


def _make_handler(result: Any) -> AsyncMock:
    return AsyncMock(return_value=result)


class _FakeDeps:
    """Lightweight stand-in for AgentContext.

    Simulates the capability-tool scenario where ``tool_wrapping.py``
    never runs, so ``tool_call_id`` and ``tool_name`` start as ``None``.
    The ``events`` property returns whatever emitter is assigned.
    """

    def __init__(self, events: Any = None) -> None:
        self.tool_call_id: str | None = None
        self.tool_name: str | None = None
        self._events = events or AsyncMock()

    @property
    def events(self) -> Any:
        return self._events


def _make_ctx(deps: Any = None) -> Any:
    """Create a minimal ctx object with ``.deps``."""
    ctx = type("_Ctx", (), {})()
    ctx.deps = deps or _FakeDeps()
    return ctx


# ---------------------------------------------------------------------------
# 1. Rename produces display names in get_tools()
# ---------------------------------------------------------------------------


class TestRenameProducesDisplayNames:
    """``RenamedToolset.get_tools()`` must return tools keyed by display names.

    ``name_map`` is documented as ``{original: display}`` (what the user
    writes in YAML), but ``RenamedToolset`` expects ``{new: original}``.
    The capability must invert before construction.
    """

    @pytest.mark.asyncio
    async def test_get_tools_keys_by_display_name(self) -> None:
        """A tool originally named 'viking_write' appears as 'write' after wrapping."""
        from pydantic_ai._run_context import RunContext
        from pydantic_ai.models.test import TestModel
        from pydantic_ai.tools import Tool
        from pydantic_ai.toolsets import FunctionToolset
        from pydantic_ai.usage import RunUsage

        async def viking_write(ctx: Any, uri: str, content: str) -> str:
            return f"Wrote {len(content)} chars to {uri}."

        toolset: AbstractToolset[Any] = FunctionToolset(tools=[Tool(viking_write)])  # type: ignore[arg-type]

        cap = ToolDisplayCapability(
            rename_mode=True,
            name_map={"viking_write": "write"},
            emit_diff=False,
        )
        wrapped = cap.get_wrapper_toolset(toolset)
        assert wrapped is not None
        assert isinstance(wrapped, RenamedToolset)

        ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
        tools = await wrapped.get_tools(ctx)
        tool_names = set(tools.keys())

        assert "write" in tool_names, (
            f"Expected 'write' in tool names after rename, got {tool_names}"
        )
        assert "viking_write" not in tool_names, (
            f"'viking_write' should have been renamed away, got {tool_names}"
        )


# ---------------------------------------------------------------------------
# 2. Injected events carry real tool_call_id
# ---------------------------------------------------------------------------


class TestInjectedEventsCarryToolCallId:
    """``wrap_tool_execute`` must set ``ctx.deps.tool_call_id`` before emitting.

    For capability tools (viking, fsspec, etc.) the legacy ``tool_wrapping.py``
    layer never runs, so ``ctx.deps.tool_call_id`` stays ``None``. Without
    explicit population, ``StreamEventEmitter`` reads ``""`` and the ACP
    converter's ``if tool_call_id:`` guard drops the event silently.
    """

    @pytest.mark.asyncio
    async def test_populates_deps_tool_call_id_before_emit(self) -> None:
        """After wrap_tool_execute, deps.tool_call_id matches call.tool_call_id."""
        deps = _FakeDeps()
        ctx = _make_ctx(deps)

        cap = ToolDisplayCapability(
            emit_diff=True,
            emit_diff_for={"viking_write"},
        )

        result = await cap.wrap_tool_execute(
            ctx,
            call=_make_call("viking_write", tool_call_id="call_abc"),
            tool_def=_make_tool_def("viking_write"),
            args={"uri": "viking://x.md", "content": "abc"},
            handler=_make_handler("Wrote 3 chars."),
        )

        assert result == "Wrote 3 chars."
        assert deps.tool_call_id == "call_abc", (
            f"Expected deps.tool_call_id='call_abc', got '{deps.tool_call_id}'"
        )
        assert deps.tool_name == "viking_write", (
            f"Expected deps.tool_name='viking_write', got '{deps.tool_name}'"
        )


# ---------------------------------------------------------------------------
# 3. emit_diff_for matches across the rename boundary
# ---------------------------------------------------------------------------


class TestDiffMatchingAcrossRename:
    """When rename is active, ``call.tool_name`` is the *display* name.

    ``emit_diff_for`` is configured with *original* names (the names the
    user knows from the tool's source). The capability must resolve the
    display name back to the original via ``name_map`` before checking
    membership.
    """

    @pytest.mark.asyncio
    async def test_display_name_resolves_to_original_for_matching(self) -> None:
        """Model calls 'write' (display); emit_diff_for={'viking_write'} still matches."""
        events = AsyncMock()
        deps = _FakeDeps(events=events)
        ctx = _make_ctx(deps)

        cap = ToolDisplayCapability(
            rename_mode=True,
            name_map={"viking_write": "write"},
            emit_diff=True,
            emit_diff_for={"viking_write"},
        )

        result = await cap.wrap_tool_execute(
            ctx,
            call=_make_call("write", tool_call_id="call_xyz"),
            tool_def=_make_tool_def("write"),
            args={"uri": "viking://x.md", "content": "abc"},
            handler=_make_handler("Wrote 3 chars."),
        )

        assert result == "Wrote 3 chars."
        events.tool_call_progress.assert_awaited_once()
        items: list[DiffContentItem] = events.tool_call_progress.await_args.kwargs["items"]
        assert items[0].path == "viking://x.md"
        assert items[0].new_text == "abc"
