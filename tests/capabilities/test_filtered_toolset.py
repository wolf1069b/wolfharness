"""Tests for FilteredToolsetCapability."""

from __future__ import annotations

import dataclasses
from typing import Any

from pydantic_ai.toolsets import FilteredToolset
import pytest

from wolfharness.capabilities.filtered_toolset import FilteredToolsetCapability
from wolfharness.capabilities.function_toolset import FunctionToolsetCapability
from wolfharness.tools.base import Tool


pytestmark = pytest.mark.unit


def _make_test_tool(name: str) -> Tool[Any]:
    """Create a minimal Tool for testing."""

    def dummy_fn(x: int) -> int:
        """Return x doubled.

        Args:
            x: Input value
        """
        return x * 2

    return Tool.from_callable(dummy_fn, name_override=name)


@dataclasses.dataclass
class _DummyCtx:
    """Minimal dataclass stand-in for RunContext used in toolset function calls.

    FunctionToolset.get_tools() calls dataclasses.replace() on the context,
    so it must be a dataclass.
    """

    deps: Any = None
    retry: int = 0
    retries: dict[str, int] = dataclasses.field(default_factory=dict)
    max_retries: int = 1
    tool_name: str = ""
    tool_call_id: str = ""
    run_step: int = 0


async def _get_tool_names(toolset: Any) -> list[str]:
    """Extract tool names from an AbstractToolset."""
    ctx = _DummyCtx()
    tools = await toolset.get_tools(ctx)  # type: ignore[arg-type]
    return list(tools.keys())


# ---- get_toolset returns FilteredToolset wrapping concrete toolset ----


@pytest.mark.asyncio
async def test_get_toolset_returns_filtered_toolset() -> None:
    """Given a capability with a concrete toolset, get_toolset returns FilteredToolset."""
    tool_a = _make_test_tool("tool_a")
    tool_b = _make_test_tool("tool_b")
    inner = FunctionToolsetCapability([tool_a, tool_b], name="inner")

    async def allow_all(_ctx: Any, _tool_def: Any) -> bool:
        return True

    cap = FilteredToolsetCapability(inner, allow_all)
    toolset = cap.get_toolset()
    assert toolset is not None
    assert isinstance(toolset, FilteredToolset)


@pytest.mark.asyncio
async def test_filter_excludes_disallowed_tools() -> None:
    """Filtered toolset excludes tools where filter returns False."""
    tool_a = _make_test_tool("tool_a")
    tool_b = _make_test_tool("tool_b")
    tool_c = _make_test_tool("tool_c")
    inner = FunctionToolsetCapability([tool_a, tool_b, tool_c], name="inner")

    allowed = {"tool_a", "tool_c"}

    async def filter_func(_ctx: Any, tool_def: Any) -> bool:
        return tool_def.name in allowed

    cap = FilteredToolsetCapability(inner, filter_func)
    toolset = cap.get_toolset()
    assert toolset is not None
    assert isinstance(toolset, FilteredToolset)

    names = await _get_tool_names(toolset)
    assert "tool_a" in names
    assert "tool_c" in names
    assert "tool_b" not in names


@pytest.mark.asyncio
async def test_filter_includes_allowed_tools() -> None:
    """Filtered toolset includes tools where filter returns True."""
    tool_a = _make_test_tool("tool_a")
    tool_b = _make_test_tool("tool_b")
    inner = FunctionToolsetCapability([tool_a, tool_b], name="inner")

    async def filter_func(_ctx: Any, tool_def: Any) -> bool:
        return tool_def.name == "tool_a"

    cap = FilteredToolsetCapability(inner, filter_func)
    toolset = cap.get_toolset()
    assert toolset is not None

    names = await _get_tool_names(toolset)
    assert names == ["tool_a"]


@pytest.mark.asyncio
async def test_empty_filter_allows_all() -> None:
    """When filter always returns True, all tools pass through."""
    tool_a = _make_test_tool("tool_a")
    tool_b = _make_test_tool("tool_b")
    inner = FunctionToolsetCapability([tool_a, tool_b], name="inner")

    async def allow_all(_ctx: Any, _tool_def: Any) -> bool:
        return True

    cap = FilteredToolsetCapability(inner, allow_all)
    toolset = cap.get_toolset()
    assert toolset is not None

    names = await _get_tool_names(toolset)
    assert set(names) == {"tool_a", "tool_b"}


@pytest.mark.asyncio
async def test_filter_that_blocks_all() -> None:
    """When filter always returns False, no tools pass through."""
    tool_a = _make_test_tool("tool_a")
    tool_b = _make_test_tool("tool_b")
    inner = FunctionToolsetCapability([tool_a, tool_b], name="inner")

    async def block_all(_ctx: Any, _tool_def: Any) -> bool:
        return False

    cap = FilteredToolsetCapability(inner, block_all)
    toolset = cap.get_toolset()
    assert toolset is not None

    names = await _get_tool_names(toolset)
    assert names == []


# ---- Sync filter function support ----


@pytest.mark.asyncio
async def test_sync_filter_function_supported() -> None:
    """FilteredToolsetCapability accepts sync filter functions."""
    tool_a = _make_test_tool("tool_a")
    tool_b = _make_test_tool("tool_b")
    inner = FunctionToolsetCapability([tool_a, tool_b], name="inner")

    def sync_filter(_ctx: Any, tool_def: Any) -> bool:
        return tool_def.name == "tool_b"

    cap = FilteredToolsetCapability(inner, sync_filter)
    toolset = cap.get_toolset()
    assert toolset is not None

    names = await _get_tool_names(toolset)
    assert names == ["tool_b"]


# ---- get_toolset returns None when inner returns None ----


@pytest.mark.asyncio
async def test_get_toolset_none_when_inner_has_no_tools() -> None:
    """When wrapped capability has no tools, get_toolset returns None."""
    inner = FunctionToolsetCapability([], name="empty")
    cap = FilteredToolsetCapability(
        inner,
        lambda _ctx, _td: True,
    )
    toolset = cap.get_toolset()
    assert toolset is None


# ---- get_toolset wraps ToolsetFunc lazily ----


@pytest.mark.asyncio
async def test_get_toolset_wraps_concrete_toolset() -> None:
    """When wrapped capability returns a concrete toolset, result is FilteredToolset."""
    tool_a = _make_test_tool("tool_a")
    tool_b = _make_test_tool("tool_b")

    from wolfharness.capabilities.function_toolset import FunctionToolsetCapability

    inner = FunctionToolsetCapability([tool_a, tool_b], name="test")

    async def allow_all(_ctx: Any, _tool_def: Any) -> bool:
        return True

    cap = FilteredToolsetCapability(inner, allow_all)
    toolset = cap.get_toolset()
    assert toolset is not None
    assert isinstance(toolset, FilteredToolset)

    names = await _get_tool_names(toolset)
    assert set(names) == {"tool_a", "tool_b"}


@pytest.mark.asyncio
async def test_concrete_toolset_filter_excludes_tools() -> None:
    """Concrete toolset path also applies filtering correctly."""
    tool_a = _make_test_tool("tool_a")
    tool_b = _make_test_tool("tool_b")
    tool_c = _make_test_tool("tool_c")

    from wolfharness.capabilities.function_toolset import FunctionToolsetCapability

    inner = FunctionToolsetCapability([tool_a, tool_b, tool_c], name="test")

    async def filter_func(_ctx: Any, tool_def: Any) -> bool:
        return tool_def.name != "tool_b"

    cap = FilteredToolsetCapability(inner, filter_func)
    toolset = cap.get_toolset()
    assert toolset is not None
    assert isinstance(toolset, FilteredToolset)

    names = await _get_tool_names(toolset)
    assert "tool_a" in names
    assert "tool_c" in names
    assert "tool_b" not in names


@pytest.mark.asyncio
async def test_concrete_toolset_returns_none_when_inner_returns_none() -> None:
    """Concrete toolset path returns None when inner returns None."""
    from wolfharness.capabilities.function_toolset import FunctionToolsetCapability

    inner = FunctionToolsetCapability(name="empty")

    async def allow_all(_ctx: Any, _tool_def: Any) -> bool:
        return True

    cap = FilteredToolsetCapability(inner, allow_all)
    toolset = cap.get_toolset()
    assert toolset is None


# ---- on_change delegation ----


@pytest.mark.asyncio
async def test_on_change_delegates_to_wrapped() -> None:
    """on_change() delegates to the wrapped capability when it implements on_change."""
    tool = _make_test_tool("tool_a")
    inner = FunctionToolsetCapability([tool], name="inner")
    cap = FilteredToolsetCapability(inner, lambda _ctx, _td: True)

    # FunctionToolsetCapability.on_change() returns None (static tools)
    result = cap.on_change()
    assert result is None


@pytest.mark.asyncio
async def test_on_change_returns_none_for_non_wolfharness_capability() -> None:
    """on_change() returns None when wrapped capability doesn't implement on_change."""

    class _BareCapability:
        """A minimal AbstractCapability stub without on_change."""

        def get_toolset(self) -> Any:
            return None

        def get_instructions(self) -> Any:
            return None

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

    from pydantic_ai.capabilities import AbstractCapability

    # Create a real AbstractCapability subclass without on_change
    class _NoOnChange(AbstractCapability[Any]):
        def get_toolset(self) -> Any:
            return None

    bare = _NoOnChange()
    cap = FilteredToolsetCapability(bare, lambda _ctx, _td: True)
    result = cap.on_change()
    assert result is None


# ---- get_instructions delegation ----


@pytest.mark.asyncio
async def test_get_instructions_delegates() -> None:
    """get_instructions() delegates to the wrapped capability."""
    inner = FunctionToolsetCapability(
        [_make_test_tool("tool_a")],
        name="inner",
        instructions="You are a test assistant.",
    )
    cap = FilteredToolsetCapability(inner, lambda _ctx, _td: True)
    instructions = cap.get_instructions()
    assert instructions == "You are a test assistant."


@pytest.mark.asyncio
async def test_get_instructions_none_when_inner_has_none() -> None:
    """get_instructions() returns None when wrapped has no instructions."""
    inner = FunctionToolsetCapability([_make_test_tool("tool_a")], name="inner")
    cap = FilteredToolsetCapability(inner, lambda _ctx, _td: True)
    assert cap.get_instructions() is None


# ---- Lifecycle delegation ----


@pytest.mark.asyncio
async def test_aenter_delegates_to_wrapped() -> None:
    """__aenter__ delegates to the wrapped capability."""
    tool = _make_test_tool("tool_a")
    inner = FunctionToolsetCapability([tool], name="inner")
    cap = FilteredToolsetCapability(inner, lambda _ctx, _td: True)

    entered = await cap.__aenter__()
    assert entered is cap


@pytest.mark.asyncio
async def test_aexit_delegates_to_wrapped() -> None:
    """__aexit__ delegates to the wrapped capability."""
    tool = _make_test_tool("tool_a")
    inner = FunctionToolsetCapability([tool], name="inner")
    cap = FilteredToolsetCapability(inner, lambda _ctx, _td: True)

    await cap.__aenter__()
    # Should not raise
    await cap.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_lifecycle_context_manager() -> None:
    """FilteredToolsetCapability works as an async context manager."""
    tool = _make_test_tool("tool_a")
    inner = FunctionToolsetCapability([tool], name="inner")
    cap = FilteredToolsetCapability(inner, lambda _ctx, _td: True)

    async with cap as ctx:
        assert ctx is cap


# ---- Properties ----


def test_name_property_default() -> None:
    """Name defaults to the wrapped capability's name."""
    inner = FunctionToolsetCapability([_make_test_tool("t")], name="my_cap")
    cap = FilteredToolsetCapability(inner, lambda _ctx, _td: True)
    assert cap.name == "my_cap"


def test_name_property_override() -> None:
    """Name can be overridden via constructor."""
    inner = FunctionToolsetCapability([_make_test_tool("t")], name="my_cap")
    cap = FilteredToolsetCapability(inner, lambda _ctx, _td: True, name="custom")
    assert cap.name == "custom"


def test_wrapped_property() -> None:
    """Wrapped property returns the wrapped capability."""
    inner = FunctionToolsetCapability([_make_test_tool("t")], name="inner")
    cap = FilteredToolsetCapability(inner, lambda _ctx, _td: True)
    assert cap.wrapped is inner


def test_filter_func_property() -> None:
    """filter_func property returns the filter function."""

    async def my_filter(_ctx: Any, _td: Any) -> bool:
        return True

    inner = FunctionToolsetCapability([_make_test_tool("t")], name="inner")
    cap = FilteredToolsetCapability(inner, my_filter)
    assert cap.filter_func is my_filter


# ---- Name derivation fallback ----


def test_name_derived_from_class_name_when_no_name_property() -> None:
    """When wrapped capability has no 'name' property, class name is used."""
    from pydantic_ai.capabilities import AbstractCapability

    class _NamelessCap(AbstractCapability[Any]):
        def get_toolset(self) -> Any:
            return None

    bare = _NamelessCap()
    cap = FilteredToolsetCapability(bare, lambda _ctx, _td: True)
    assert cap.name == "_NamelessCap"


# ---- Filter with deny list pattern ----


@pytest.mark.asyncio
async def test_deny_list_pattern() -> None:
    """Common deny-list pattern: filter excludes specific tool names."""
    tool_a = _make_test_tool("tool_a")
    tool_b = _make_test_tool("tool_b")
    tool_c = _make_test_tool("tool_c")
    inner = FunctionToolsetCapability([tool_a, tool_b, tool_c], name="inner")

    deny = {"tool_b"}

    async def deny_filter(_ctx: Any, tool_def: Any) -> bool:
        return tool_def.name not in deny

    cap = FilteredToolsetCapability(inner, deny_filter)
    toolset = cap.get_toolset()
    assert toolset is not None

    names = await _get_tool_names(toolset)
    assert set(names) == {"tool_a", "tool_c"}


# ---- Filter with allow list pattern ----


@pytest.mark.asyncio
async def test_allow_list_pattern() -> None:
    """Common allow-list pattern: filter includes only specific tool names."""
    tool_a = _make_test_tool("tool_a")
    tool_b = _make_test_tool("tool_b")
    tool_c = _make_test_tool("tool_c")
    inner = FunctionToolsetCapability([tool_a, tool_b, tool_c], name="inner")

    allow = {"tool_a", "tool_c"}

    async def allow_filter(_ctx: Any, tool_def: Any) -> bool:
        return tool_def.name in allow

    cap = FilteredToolsetCapability(inner, allow_filter)
    toolset = cap.get_toolset()
    assert toolset is not None

    names = await _get_tool_names(toolset)
    assert set(names) == {"tool_a", "tool_c"}


# ---- Single tool filtering ----


@pytest.mark.asyncio
async def test_single_tool_kept() -> None:
    """Filtering with one tool that passes through."""
    tool_a = _make_test_tool("tool_a")
    inner = FunctionToolsetCapability([tool_a], name="inner")

    async def allow_all(_ctx: Any, _tool_def: Any) -> bool:
        return True

    cap = FilteredToolsetCapability(inner, allow_all)
    toolset = cap.get_toolset()
    assert toolset is not None

    names = await _get_tool_names(toolset)
    assert names == ["tool_a"]


@pytest.mark.asyncio
async def test_single_tool_filtered_out() -> None:
    """Filtering with one tool that gets blocked."""
    tool_a = _make_test_tool("tool_a")
    inner = FunctionToolsetCapability([tool_a], name="inner")

    async def block_all(_ctx: Any, _tool_def: Any) -> bool:
        return False

    cap = FilteredToolsetCapability(inner, block_all)
    toolset = cap.get_toolset()
    assert toolset is not None

    names = await _get_tool_names(toolset)
    assert names == []
