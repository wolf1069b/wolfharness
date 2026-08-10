"""Tests for FunctionToolsetCapability."""

from __future__ import annotations

from typing import Any

from pydantic_ai.toolsets import FunctionToolset
import pytest

from wolfharness.capabilities.function_toolset import FunctionToolsetCapability
from wolfharness.tools.base import Tool


pytestmark = pytest.mark.unit


def _make_test_tool(name: str = "test_tool") -> Tool[Any]:
    """Create a minimal Tool for testing."""

    def dummy_fn(x: int) -> int:
        """Return x doubled.

        Args:
            x: Input value
        """
        return x * 2

    return Tool.from_callable(dummy_fn, name_override=name)


def _make_test_tool_async(name: str = "async_tool") -> Tool[Any]:
    """Create an async Tool for testing."""

    async def async_fn(x: int) -> int:
        """Return x tripled.

        Args:
            x: Input value
        """
        return x * 3

    return Tool.from_callable(async_fn, name_override=name)


# ---- Construction tests ----


def test_capability_construction_with_tools() -> None:
    """FunctionToolsetCapability can be constructed with a list of tools."""
    tool1 = _make_test_tool("tool_a")
    tool2 = _make_test_tool("tool_b")
    cap = FunctionToolsetCapability([tool1, tool2], name="my_cap")

    assert cap.name == "my_cap"
    assert len(cap.tools) == 2
    assert cap.tools[0].name == "tool_a"
    assert cap.tools[1].name == "tool_b"


def test_capability_construction_default_name() -> None:
    """FunctionToolsetCapability uses 'function_tools' as default name."""
    tool = _make_test_tool()
    cap = FunctionToolsetCapability([tool])

    assert cap.name == "function_tools"


def test_capability_construction_empty_tools() -> None:
    """FunctionToolsetCapability can be constructed with an empty tool list."""
    cap = FunctionToolsetCapability([], name="empty_cap")

    assert cap.name == "empty_cap"
    assert cap.tools == []


def test_capability_construction_with_instructions() -> None:
    """FunctionToolsetCapability stores instructions when provided."""
    tool = _make_test_tool()
    cap = FunctionToolsetCapability([tool], instructions="Use these tools wisely.")

    assert cap.get_instructions() == "Use these tools wisely."


def test_capability_construction_without_instructions() -> None:
    """FunctionToolsetCapability returns None for instructions when not provided."""
    tool = _make_test_tool()
    cap = FunctionToolsetCapability([tool])

    assert cap.get_instructions() is None


def test_capability_tools_property_returns_copy() -> None:
    """Tools property returns a copy — mutating it does not affect the capability."""
    tool = _make_test_tool("original")
    cap = FunctionToolsetCapability([tool])

    tools_copy = cap.tools
    tools_copy.clear()
    assert len(cap.tools) == 1


# ---- get_toolset tests ----


def test_get_toolset_returns_function_toolset() -> None:
    """get_toolset() returns a FunctionToolset when tools are present."""
    tool = _make_test_tool("my_tool")
    cap = FunctionToolsetCapability([tool], name="test_cap")

    toolset = cap.get_toolset()
    assert toolset is not None
    assert isinstance(toolset, FunctionToolset)


def test_get_toolset_none_when_empty() -> None:
    """get_toolset() returns None when no tools are configured."""
    cap = FunctionToolsetCapability([], name="empty_cap")

    assert cap.get_toolset() is None


def test_get_toolset_contains_correct_tools() -> None:
    """get_toolset() contains all wrapped tools by name."""
    tool1 = _make_test_tool("alpha")
    tool2 = _make_test_tool("beta")
    cap = FunctionToolsetCapability([tool1, tool2])

    toolset = cap.get_toolset()
    assert toolset is not None

    # FunctionToolset stores tools in a dict keyed by name
    tool_names = list(toolset.tools.keys())  # type: ignore[attr-defined]
    assert "alpha" in tool_names
    assert "beta" in tool_names
    assert len(tool_names) == 2


def test_get_toolset_single_tool() -> None:
    """get_toolset() works correctly with a single tool."""
    tool = _make_test_tool("solo")
    cap = FunctionToolsetCapability([tool])

    toolset = cap.get_toolset()
    assert toolset is not None
    tool_names = list(toolset.tools.keys())  # type: ignore[attr-defined]
    assert tool_names == ["solo"]


def test_get_toolset_with_async_tool() -> None:
    """get_toolset() handles async tools correctly."""
    tool = _make_test_tool_async("async_tool")
    cap = FunctionToolsetCapability([tool])

    toolset = cap.get_toolset()
    assert toolset is not None
    tool_names = list(toolset.tools.keys())  # type: ignore[attr-defined]
    assert "async_tool" in tool_names


def test_get_toolset_id_matches_capability_name() -> None:
    """FunctionToolset id is set to the capability name."""
    tool = _make_test_tool()
    cap = FunctionToolsetCapability([tool], name="custom_id_cap")

    toolset = cap.get_toolset()
    assert toolset is not None
    # FunctionToolset stores id as an attribute
    assert toolset.id == "custom_id_cap"  # type: ignore[attr-defined]


def test_get_toolset_returns_new_instance_each_call() -> None:
    """get_toolset() returns a new FunctionToolset on each call."""
    tool = _make_test_tool("fresh")
    cap = FunctionToolsetCapability([tool])

    ts1 = cap.get_toolset()
    ts2 = cap.get_toolset()
    assert ts1 is not None
    assert ts2 is not None
    assert ts1 is not ts2


# ---- get_instructions tests ----


def test_get_instructions_returns_string() -> None:
    """get_instructions() returns the provided instructions string."""
    tool = _make_test_tool()
    cap = FunctionToolsetCapability([tool], instructions="Be careful with these tools.")

    result = cap.get_instructions()
    assert result == "Be careful with these tools."


def test_get_instructions_returns_none_by_default() -> None:
    """get_instructions() returns None when no instructions are set."""
    tool = _make_test_tool()
    cap = FunctionToolsetCapability([tool])

    assert cap.get_instructions() is None


def test_get_instructions_empty_string() -> None:
    """get_instructions() returns empty string when explicitly set to empty."""
    tool = _make_test_tool()
    cap = FunctionToolsetCapability([tool], instructions="")

    assert cap.get_instructions() == ""


# ---- on_change tests ----


def test_on_change_returns_none() -> None:
    """on_change() returns None for static tools."""
    tool = _make_test_tool()
    cap = FunctionToolsetCapability([tool])

    assert cap.on_change() is None


def test_on_change_returns_none_when_empty() -> None:
    """on_change() returns None even when no tools are configured."""
    cap = FunctionToolsetCapability([])

    assert cap.on_change() is None


# ---- Lifecycle tests ----


@pytest.mark.asyncio
async def test_aenter_returns_self() -> None:
    """__aenter__ returns the capability instance (no-op)."""
    tool = _make_test_tool()
    cap = FunctionToolsetCapability([tool])

    result = await cap.__aenter__()
    assert result is cap


@pytest.mark.asyncio
async def test_aexit_is_noop() -> None:
    """__aexit__ does not raise and returns None."""
    tool = _make_test_tool()
    cap = FunctionToolsetCapability([tool])

    result = await cap.__aexit__(None, None, None)
    assert result is None


@pytest.mark.asyncio
async def test_aexit_with_exception_is_noop() -> None:
    """__aexit__ does not suppress exceptions (returns None)."""
    tool = _make_test_tool()
    cap = FunctionToolsetCapability([tool])

    result = await cap.__aexit__(ValueError, ValueError("test"), None)
    assert result is None


@pytest.mark.asyncio
async def test_context_manager_protocol() -> None:
    """FunctionToolsetCapability works as an async context manager."""
    tool = _make_test_tool()
    cap = FunctionToolsetCapability([tool], name="ctx_cap")

    async with cap as ctx:
        assert ctx is cap
        assert ctx.name == "ctx_cap"


# ---- Integration: tools are wrapped correctly ----


@pytest.mark.asyncio
async def test_toolset_tools_are_callable() -> None:
    """Tools in the FunctionToolset are wrapped and callable."""
    tool = _make_test_tool("double_it")
    cap = FunctionToolsetCapability([tool])

    toolset = cap.get_toolset()
    assert toolset is not None
    # The FunctionToolset should have the tool registered
    tools_dict = toolset.tools  # type: ignore[attr-defined]
    assert "double_it" in tools_dict


def test_multiple_capabilities_are_independent() -> None:
    """Multiple FunctionToolsetCapability instances do not share state."""
    tool1 = _make_test_tool("cap1_tool")
    tool2 = _make_test_tool("cap2_tool")

    cap1 = FunctionToolsetCapability([tool1], name="cap1")
    cap2 = FunctionToolsetCapability([tool2], name="cap2")

    ts1 = cap1.get_toolset()
    ts2 = cap2.get_toolset()
    assert ts1 is not None
    assert ts2 is not None

    names1 = list(ts1.tools.keys())  # type: ignore[attr-defined]
    names2 = list(ts2.tools.keys())  # type: ignore[attr-defined]

    assert "cap1_tool" in names1
    assert "cap2_tool" in names2
    assert "cap1_tool" not in names2
    assert "cap2_tool" not in names1


def test_capability_with_mixed_sync_async_tools() -> None:
    """FunctionToolsetCapability handles a mix of sync and async tools."""
    sync_tool = _make_test_tool("sync")
    async_tool = _make_test_tool_async("async")
    cap = FunctionToolsetCapability([sync_tool, async_tool])

    toolset = cap.get_toolset()
    assert toolset is not None
    names = list(toolset.tools.keys())  # type: ignore[attr-defined]
    assert set(names) == {"sync", "async"}
