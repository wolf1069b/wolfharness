"""Tests for CombinedToolsetCapability."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import CombinedToolset, FunctionToolset
import pytest

from wolfharness.capabilities.change_event import ChangeEvent
from wolfharness.capabilities.combined_toolset import CombinedToolsetCapability
from wolfharness.capabilities.function_toolset import FunctionToolsetCapability
from wolfharness.tools.base import Tool


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _make_test_tool(name: str = "test_tool") -> Tool[Any]:
    """Create a minimal Tool for testing."""

    def dummy_fn(x: int) -> int:
        """Return x doubled.

        Args:
            x: Input value
        """
        return x * 2

    return Tool.from_callable(dummy_fn, name_override=name)


class _FakeCap(AbstractCapability[AgentDepsT]):
    """Minimal fake capability for testing composition.

    Provides configurable toolset, instructions, on_change, and lifecycle.
    """

    def __init__(
        self,
        *,
        name: str = "fake_cap",
        tools: list[Tool[Any]] | None = None,
        instructions: str | None = None,
        change_events: list[ChangeEvent] | None = None,
        enter_called: bool = False,
    ) -> None:
        self._name = name
        self._tools = tools or []
        self._instructions = instructions
        self._change_events = change_events
        self.enter_count = 0
        self.exit_count = 0
        self._enter_called = enter_called

    @property
    def name(self) -> str:
        return self._name

    def get_toolset(self) -> Any:
        if not self._tools:
            return None
        from wolfharness.tools.tool_wrapping import wrap_tool_for_pydantic_ai

        pa_tools = [wrap_tool_for_pydantic_ai(tool) for tool in self._tools]
        return FunctionToolset(pa_tools, id=self._name)

    def get_instructions(self) -> str | None:
        return self._instructions

    def on_change(self) -> AsyncIterator[ChangeEvent] | None:
        if self._change_events is None:
            return None

        events = list(self._change_events)

        async def _gen() -> AsyncIterator[ChangeEvent]:
            for event in events:
                await asyncio.sleep(0)
                yield event

        return _gen()

    async def __aenter__(self) -> _FakeCap[AgentDepsT]:
        self.enter_count += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.exit_count += 1


class _NoLifecycleCap(AbstractCapability[AgentDepsT]):
    """Capability that does NOT implement __aenter__/__aexit__."""

    def __init__(self, *, name: str = "no_lifecycle") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def get_toolset(self) -> None:
        return None

    def get_instructions(self) -> None:
        return None

    def on_change(self) -> None:
        return None


class _NoOnChangeCap(AbstractCapability[AgentDepsT]):
    """Capability that does NOT implement on_change()."""

    def __init__(
        self,
        *,
        name: str = "no_on_change",
        instructions: str | None = None,
    ) -> None:
        self._name = name
        self._instructions = instructions

    @property
    def name(self) -> str:
        return self._name

    def get_toolset(self) -> None:
        return None

    def get_instructions(self) -> str | None:
        return self._instructions


async def _get_tool_names(toolset: Any) -> list[str]:
    """Extract tool names from an AbstractToolset (recursive for CombinedToolset)."""
    if hasattr(toolset, "tools") and isinstance(toolset.tools, dict):
        return list(toolset.tools.keys())
    if hasattr(toolset, "toolsets"):
        names: list[str] = []
        for ts in toolset.toolsets:
            names.extend(await _get_tool_names(ts))
        return names
    return []


# ---- Construction tests ----


def test_construction_with_capabilities() -> None:
    """CombinedToolsetCapability can be constructed with a list of capabilities."""
    cap1 = _FakeCap(name="cap1")
    cap2 = _FakeCap(name="cap2")
    combined = CombinedToolsetCapability([cap1, cap2])

    assert len(combined.capabilities) == 2
    assert combined.capabilities[0] is cap1
    assert combined.capabilities[1] is cap2


def test_construction_with_name_override() -> None:
    """CombinedToolsetCapability uses the provided name."""
    cap = _FakeCap(name="inner")
    combined = CombinedToolsetCapability([cap], name="my_combined")

    assert combined.name == "my_combined"


def test_construction_default_name_derived_from_children() -> None:
    """Default name is 'combined:' + child names joined with ','."""
    cap1 = _FakeCap(name="alpha")
    cap2 = _FakeCap(name="beta")
    combined = CombinedToolsetCapability([cap1, cap2])

    assert combined.name == "combined:alpha,beta"


def test_construction_default_name_with_no_name_children() -> None:
    """Default name uses class name for children without a name property."""
    cap = _NoOnChangeCap(name="gamma")
    combined = CombinedToolsetCapability([cap])

    assert "gamma" in combined.name


def test_construction_empty_list() -> None:
    """CombinedToolsetCapability can be constructed with an empty list."""
    combined = CombinedToolsetCapability([])

    assert combined.capabilities == []
    assert combined.name == "combined:empty"


def test_capabilities_property_returns_copy() -> None:
    """Capabilities property returns a copy — mutating it does not affect the capability."""
    cap1 = _FakeCap(name="c1")
    cap2 = _FakeCap(name="c2")
    combined = CombinedToolsetCapability([cap1, cap2])

    caps_copy = combined.capabilities
    caps_copy.clear()
    assert len(combined.capabilities) == 2


def test_name_property() -> None:
    """Name property returns the capability name."""
    combined = CombinedToolsetCapability([], name="test_name")
    assert combined.name == "test_name"


# ---- get_toolset tests ----


def test_get_toolset_returns_combined_toolset() -> None:
    """get_toolset() returns a CombinedToolset when children have toolsets."""
    tool1 = _make_test_tool("tool_a")
    tool2 = _make_test_tool("tool_b")
    cap1 = _FakeCap(name="cap1", tools=[tool1])
    cap2 = _FakeCap(name="cap2", tools=[tool2])
    combined = CombinedToolsetCapability([cap1, cap2])

    toolset = combined.get_toolset()
    assert toolset is not None
    assert isinstance(toolset, CombinedToolset)


def test_get_toolset_none_when_no_children_have_tools() -> None:
    """get_toolset() returns None when no child provides a toolset."""
    cap1 = _FakeCap(name="cap1", tools=[])
    cap2 = _FakeCap(name="cap2", tools=[])
    combined = CombinedToolsetCapability([cap1, cap2])

    assert combined.get_toolset() is None


def test_get_toolset_none_when_empty() -> None:
    """get_toolset() returns None when no children are configured."""
    combined = CombinedToolsetCapability([])

    assert combined.get_toolset() is None


def test_get_toolset_merges_tools_from_all_children() -> None:
    """get_toolset() merges tools from all children."""
    tool1 = _make_test_tool("alpha")
    tool2 = _make_test_tool("beta")
    tool3 = _make_test_tool("gamma")
    cap1 = _FakeCap(name="cap1", tools=[tool1, tool2])
    cap2 = _FakeCap(name="cap2", tools=[tool3])
    combined = CombinedToolsetCapability([cap1, cap2])

    toolset = combined.get_toolset()
    assert toolset is not None
    assert isinstance(toolset, CombinedToolset)
    assert len(toolset.toolsets) == 2


def test_get_toolset_skips_children_without_toolset() -> None:
    """get_toolset() skips children that return None for toolset."""
    tool1 = _make_test_tool("real_tool")
    cap_with_tools = _FakeCap(name="has_tools", tools=[tool1])
    cap_without_tools = _FakeCap(name="no_tools", tools=[])
    combined = CombinedToolsetCapability([cap_with_tools, cap_without_tools])

    toolset = combined.get_toolset()
    assert toolset is not None
    assert isinstance(toolset, CombinedToolset)
    assert len(toolset.toolsets) == 1


def test_get_toolset_single_child() -> None:
    """get_toolset() works correctly with a single child."""
    tool = _make_test_tool("solo")
    cap = _FakeCap(name="only", tools=[tool])
    combined = CombinedToolsetCapability([cap])

    toolset = combined.get_toolset()
    assert toolset is not None
    assert isinstance(toolset, CombinedToolset)
    assert len(toolset.toolsets) == 1


def test_get_toolset_returns_new_instance_each_call() -> None:
    """get_toolset() returns a new CombinedToolset on each call."""
    tool = _make_test_tool("fresh")
    cap = _FakeCap(name="cap", tools=[tool])
    combined = CombinedToolsetCapability([cap])

    ts1 = combined.get_toolset()
    ts2 = combined.get_toolset()
    assert ts1 is not None
    assert ts2 is not None
    assert ts1 is not ts2


# ---- get_instructions tests ----


def test_get_instructions_concatenates_from_all_children() -> None:
    """get_instructions() concatenates instructions from all children."""
    cap1 = _FakeCap(name="cap1", instructions="First instruction.")
    cap2 = _FakeCap(name="cap2", instructions="Second instruction.")
    combined = CombinedToolsetCapability([cap1, cap2])

    result = combined.get_instructions()
    assert result is not None
    assert "First instruction." in result
    assert "Second instruction." in result
    assert "\n\n" in result


def test_get_instructions_returns_none_when_no_instructions() -> None:
    """get_instructions() returns None when no child provides instructions."""
    cap1 = _FakeCap(name="cap1", instructions=None)
    cap2 = _FakeCap(name="cap2", instructions=None)
    combined = CombinedToolsetCapability([cap1, cap2])

    assert combined.get_instructions() is None


def test_get_instructions_skips_none_children() -> None:
    """get_instructions() skips children that return None."""
    cap1 = _FakeCap(name="cap1", instructions="Only instruction.")
    cap2 = _FakeCap(name="cap2", instructions=None)
    combined = CombinedToolsetCapability([cap1, cap2])

    result = combined.get_instructions()
    assert result == "Only instruction."


def test_get_instructions_empty_list() -> None:
    """get_instructions() returns None when no children are configured."""
    combined = CombinedToolsetCapability([])

    assert combined.get_instructions() is None


def test_get_instructions_single_child() -> None:
    """get_instructions() works correctly with a single child."""
    cap = _FakeCap(name="cap", instructions="Solo instruction.")
    combined = CombinedToolsetCapability([cap])

    assert combined.get_instructions() == "Solo instruction."


def test_get_instructions_joined_with_double_newline() -> None:
    r"""get_instructions() joins multiple instructions with '\n\n'."""
    cap1 = _FakeCap(name="cap1", instructions="A")
    cap2 = _FakeCap(name="cap2", instructions="B")
    cap3 = _FakeCap(name="cap3", instructions="C")
    combined = CombinedToolsetCapability([cap1, cap2, cap3])

    result = combined.get_instructions()
    assert result == "A\n\nB\n\nC"


# ---- on_change tests ----


def test_on_change_returns_none_when_no_children_have_on_change() -> None:
    """on_change() returns None when no child provides on_change()."""
    cap1 = _NoOnChangeCap(name="cap1")
    cap2 = _NoOnChangeCap(name="cap2")
    combined = CombinedToolsetCapability([cap1, cap2])

    assert combined.on_change() is None


def test_on_change_returns_none_when_empty() -> None:
    """on_change() returns None when no children are configured."""
    combined = CombinedToolsetCapability([])

    assert combined.on_change() is None


def test_on_change_returns_none_when_children_return_none() -> None:
    """on_change() returns None when all children's on_change() returns None."""
    cap1 = _FakeCap(name="cap1", change_events=None)
    cap2 = _FakeCap(name="cap2", change_events=None)
    combined = CombinedToolsetCapability([cap1, cap2])

    assert combined.on_change() is None


@pytest.mark.asyncio
async def test_on_change_merges_events_from_multiple_children() -> None:
    """on_change() merges events from multiple children."""
    events1 = [ChangeEvent(capability_name="cap1", kind="tools_changed")]
    events2 = [ChangeEvent(capability_name="cap2", kind="prompts_changed")]
    cap1 = _FakeCap(name="cap1", change_events=events1)
    cap2 = _FakeCap(name="cap2", change_events=events2)
    combined = CombinedToolsetCapability([cap1, cap2])

    gen = combined.on_change()
    assert gen is not None

    collected = [event async for event in gen]

    assert len(collected) == 2
    names = {e.capability_name for e in collected}
    assert names == {"cap1", "cap2"}


@pytest.mark.asyncio
async def test_on_change_yields_events_from_single_child() -> None:
    """on_change() yields events from a single child with on_change()."""
    events = [
        ChangeEvent(capability_name="only_cap", kind="tools_changed"),
        ChangeEvent(capability_name="only_cap", kind="resource_list_changed"),
    ]
    cap = _FakeCap(name="only_cap", change_events=events)
    combined = CombinedToolsetCapability([cap])

    gen = combined.on_change()
    assert gen is not None

    collected = [event async for event in gen]

    assert len(collected) == 2
    assert collected[0].kind == "tools_changed"
    assert collected[1].kind == "resource_list_changed"


@pytest.mark.asyncio
async def test_on_change_mixed_children_some_with_some_without() -> None:
    """on_change() merges only from children that have on_change()."""
    events = [ChangeEvent(capability_name="dynamic_cap", kind="skills_changed")]
    cap_with = _FakeCap(name="dynamic_cap", change_events=events)
    cap_without = _NoOnChangeCap(name="static_cap")
    combined = CombinedToolsetCapability([cap_with, cap_without])

    gen = combined.on_change()
    assert gen is not None

    collected = [event async for event in gen]

    assert len(collected) == 1
    assert collected[0].capability_name == "dynamic_cap"
    assert collected[0].kind == "skills_changed"


@pytest.mark.asyncio
async def test_on_change_yields_in_order_from_single_child() -> None:
    """on_change() preserves event ordering within a single child."""
    events = [
        ChangeEvent(capability_name="cap", kind="tools_changed"),
        ChangeEvent(capability_name="cap", kind="prompts_changed"),
        ChangeEvent(capability_name="cap", kind="resource_list_changed"),
        ChangeEvent(capability_name="cap", kind="skills_changed"),
    ]
    cap = _FakeCap(name="cap", change_events=events)
    combined = CombinedToolsetCapability([cap])

    gen = combined.on_change()
    assert gen is not None

    collected = [event async for event in gen]

    assert len(collected) == 4
    assert collected[0].kind == "tools_changed"
    assert collected[1].kind == "prompts_changed"
    assert collected[2].kind == "resource_list_changed"
    assert collected[3].kind == "skills_changed"


@pytest.mark.asyncio
async def test_on_change_handles_empty_generator() -> None:
    """on_change() handles a child whose on_change() yields no events."""
    cap = _FakeCap(name="cap", change_events=[])
    combined = CombinedToolsetCapability([cap])

    gen = combined.on_change()
    assert gen is not None

    collected = [event async for event in gen]

    assert collected == []


# ---- Lifecycle tests ----


@pytest.mark.asyncio
async def test_aenter_enters_all_children() -> None:
    """__aenter__ enters all children that implement lifecycle."""
    cap1 = _FakeCap(name="cap1")
    cap2 = _FakeCap(name="cap2")
    combined = CombinedToolsetCapability([cap1, cap2])

    result = await combined.__aenter__()
    assert result is combined
    assert cap1.enter_count == 1
    assert cap2.enter_count == 1

    await combined.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_aexit_exits_all_children() -> None:
    """__aexit__ exits all children that implement lifecycle."""
    cap1 = _FakeCap(name="cap1")
    cap2 = _FakeCap(name="cap2")
    combined = CombinedToolsetCapability([cap1, cap2])

    await combined.__aenter__()
    await combined.__aexit__(None, None, None)

    assert cap1.exit_count == 1
    assert cap2.exit_count == 1


@pytest.mark.asyncio
async def test_aexit_exits_in_reverse_order() -> None:
    """__aexit__ exits children in reverse order (LIFO via AsyncExitStack)."""
    exit_order: list[str] = []

    class _OrderedCap(AbstractCapability[AgentDepsT]):
        def __init__(self, name: str) -> None:
            self._name = name

        @property
        def name(self) -> str:
            return self._name

        def get_toolset(self) -> None:
            return None

        def get_instructions(self) -> None:
            return None

        async def __aenter__(self) -> _OrderedCap[AgentDepsT]:
            return self

        async def __aexit__(self, *args: object) -> None:
            exit_order.append(self._name)

    cap1 = _OrderedCap("first")
    cap2 = _OrderedCap("second")
    cap3 = _OrderedCap("third")
    combined = CombinedToolsetCapability([cap1, cap2, cap3])

    await combined.__aenter__()
    await combined.__aexit__(None, None, None)

    assert exit_order == ["third", "second", "first"]


@pytest.mark.asyncio
async def test_lifecycle_with_children_without_lifecycle() -> None:
    """Children without lifecycle are skipped during enter/exit."""
    cap_with_lifecycle = _FakeCap(name="has_lifecycle")
    cap_without_lifecycle = _NoLifecycleCap(name="no_lifecycle")
    combined = CombinedToolsetCapability([cap_with_lifecycle, cap_without_lifecycle])

    result = await combined.__aenter__()
    assert result is combined
    assert cap_with_lifecycle.enter_count == 1

    await combined.__aexit__(None, None, None)
    assert cap_with_lifecycle.exit_count == 1


@pytest.mark.asyncio
async def test_context_manager_protocol() -> None:
    """CombinedToolsetCapability works as an async context manager."""
    cap1 = _FakeCap(name="cap1")
    cap2 = _FakeCap(name="cap2")
    combined = CombinedToolsetCapability([cap1, cap2], name="ctx_combined")

    async with combined as ctx:
        assert ctx is combined
        assert ctx.name == "ctx_combined"

    assert cap1.exit_count == 1
    assert cap2.exit_count == 1


@pytest.mark.asyncio
async def test_lifecycle_empty_list() -> None:
    """__aenter__/__aexit__ work with an empty children list."""
    combined = CombinedToolsetCapability([])

    result = await combined.__aenter__()
    assert result is combined

    await combined.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_aexit_propagates_exception() -> None:
    """__aexit__ does not suppress exceptions (returns None)."""
    cap = _FakeCap(name="cap")
    combined = CombinedToolsetCapability([cap])

    await combined.__aenter__()
    result = await combined.__aexit__(ValueError, ValueError("test"), None)
    assert result is None


# ---- Integration tests ----


@pytest.mark.asyncio
async def test_full_composition_with_function_toolset_capabilities() -> None:
    """CombinedToolsetCapability composes real FunctionToolsetCapability instances."""
    tool1 = _make_test_tool("tool_one")
    tool2 = _make_test_tool("tool_two")
    cap1 = FunctionToolsetCapability([tool1], name="cap1", instructions="First.")
    cap2 = FunctionToolsetCapability([tool2], name="cap2", instructions="Second.")
    combined = CombinedToolsetCapability([cap1, cap2])

    # Toolset
    toolset = combined.get_toolset()
    assert toolset is not None
    assert isinstance(toolset, CombinedToolset)
    assert len(toolset.toolsets) == 2

    # Instructions
    instructions = combined.get_instructions()
    assert instructions is not None
    assert "First." in instructions
    assert "Second." in instructions

    # on_change — FunctionToolsetCapability returns None
    assert combined.on_change() is None

    # Lifecycle
    async with combined:
        pass


def test_composition_preserves_order() -> None:
    """Capabilities are composed in the order they are provided."""
    cap1 = _FakeCap(name="first", instructions="1")
    cap2 = _FakeCap(name="second", instructions="2")
    cap3 = _FakeCap(name="third", instructions="3")
    combined = CombinedToolsetCapability([cap1, cap2, cap3])

    result = combined.get_instructions()
    assert result == "1\n\n2\n\n3"

    caps = combined.capabilities
    assert caps[0].name == "first"  # type: ignore[attr-defined]
    assert caps[1].name == "second"  # type: ignore[attr-defined]
    assert caps[2].name == "third"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_mixed_capabilities_some_with_tools_some_without() -> None:
    """Mixed children: some provide tools, some provide instructions, some neither."""
    tool = _make_test_tool("only_tool")
    cap_tools = _FakeCap(name="tools_cap", tools=[tool])
    cap_instr = _FakeCap(name="instr_cap", instructions="Important note.")
    cap_empty = _FakeCap(name="empty_cap")
    combined = CombinedToolsetCapability([cap_tools, cap_instr, cap_empty])

    # Toolset should have only 1 toolset (from cap_tools)
    toolset = combined.get_toolset()
    assert toolset is not None
    assert isinstance(toolset, CombinedToolset)
    assert len(toolset.toolsets) == 1

    # Instructions should have 1 (from cap_instr)
    instructions = combined.get_instructions()
    assert instructions == "Important note."

    # on_change should be None (no child provides it)
    assert combined.on_change() is None


# ---- get_instructions with callables (prefix cache fix) ----


class _CallableInstrCap(AbstractCapability[AgentDepsT]):
    """Capability that returns a callable from get_instructions."""

    def __init__(self, *, name: str = "callable_cap") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def get_toolset(self) -> None:
        return None

    def get_instructions(self) -> Any:
        def _dynamic(ctx: Any) -> str:
            return "dynamic content"

        return _dynamic


class _ListInstrCap(AbstractCapability[AgentDepsT]):
    """Capability that returns [str, callable] from get_instructions."""

    def __init__(self, *, name: str = "list_cap") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def get_toolset(self) -> None:
        return None

    def get_instructions(self) -> Any:
        def _dynamic(ctx: Any) -> str:
            return "dynamic"

        return ["static", _dynamic]


def test_get_instructions_handles_callable_from_child() -> None:
    """get_instructions() preserves callables from children (not silently dropped)."""
    cap = _CallableInstrCap(name="dyn_cap")
    combined = CombinedToolsetCapability([cap])

    result = combined.get_instructions()
    assert result is not None
    # Should be a list containing the callable
    assert isinstance(result, list)
    assert len(result) == 1
    assert callable(result[0])


def test_get_instructions_handles_mixed_str_and_callable() -> None:
    """get_instructions() returns list when children return mixed str and callable."""
    cap_str = _FakeCap(name="str_cap", instructions="static text")
    cap_callable = _CallableInstrCap(name="dyn_cap")
    combined = CombinedToolsetCapability([cap_str, cap_callable])

    result = combined.get_instructions()
    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == "static text"
    assert callable(result[1])


def test_get_instructions_handles_list_from_child() -> None:
    """get_instructions() flattens list returns from children."""
    cap_list = _ListInstrCap(name="list_cap")
    cap_str = _FakeCap(name="str_cap", instructions="extra")
    combined = CombinedToolsetCapability([cap_list, cap_str])

    result = combined.get_instructions()
    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 3  # "static", callable, "extra"
    assert result[0] == "static"
    assert callable(result[1])
    assert result[2] == "extra"


def test_get_instructions_all_str_returns_joined_string() -> None:
    """get_instructions() returns joined string when all children return str (backward compat)."""
    cap1 = _FakeCap(name="cap1", instructions="A")
    cap2 = _FakeCap(name="cap2", instructions="B")
    combined = CombinedToolsetCapability([cap1, cap2])

    result = combined.get_instructions()
    assert result == "A\n\nB"
