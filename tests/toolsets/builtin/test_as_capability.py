"""Tests for builtin toolset get_toolset() and get_tools() methods."""

from __future__ import annotations

from collections.abc import Awaitable
import contextlib
from pathlib import PurePosixPath
from typing import Any, cast
from unittest.mock import MagicMock

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AbstractToolset
import pytest

from wolfharness.capabilities.function_toolset import FunctionToolsetCapability
from wolfharness_toolsets.builtin import (
    CodeTools,
    DebugTools,
    ProcessManagementTools,
    SubagentTools,
    WorkersTools,
)


pytestmark = pytest.mark.unit


def _make_run_context() -> RunContext[Any]:
    """Create a minimal RunContext for testing toolset resolution."""
    return RunContext(
        deps=MagicMock(),
        model=MagicMock(),
        usage=MagicMock(),
        messages=[],
        tracer=MagicMock(),
        retries={},
    )


async def _resolve_toolset(cap: AbstractCapability[Any]) -> AbstractToolset[Any] | None:
    """Resolve a capability to its underlying AbstractToolset.

    Since get_toolset() may return a callable for lazy evaluation,
    we need to invoke the callable with a mock RunContext to get the
    actual toolset. Also ensures tools are initialized by calling
    get_tools() first for lazy-initialization providers.
    """
    from collections.abc import Callable

    # Ensure tools are initialized for lazy providers (e.g. CodeTools)
    get_tools_fn = getattr(cap, "get_tools", None)
    if get_tools_fn is not None:
        with contextlib.suppress(Exception):
            await get_tools_fn()

    toolset_or_callable = cap.get_toolset()
    if toolset_or_callable is None:
        return None
    if isinstance(toolset_or_callable, AbstractToolset):
        return toolset_or_callable
    mock_ctx = _make_run_context()
    callable_ts = cast(Callable[[Any], Any], toolset_or_callable)
    result = callable_ts(mock_ctx)
    if isinstance(result, Awaitable):
        return await result
    return result


@pytest.mark.unit
class TestCapabilityIsAbstractCapability:
    """Tests that builtin toolsets are AbstractCapability instances."""

    async def test_returns_toolset_capability(self) -> None:
        """FunctionToolsetCapability is an AbstractCapability."""

        class SimpleProvider(FunctionToolsetCapability):
            def __init__(self) -> None:
                super().__init__(name="simple")
                self._tools = [self.create_tool(lambda x: x, name_override="identity")]

        provider = SimpleProvider()
        assert isinstance(provider, AbstractCapability)

    async def test_empty_tools_returns_toolset(self) -> None:
        """Provider with no tools is still an AbstractCapability."""

        class EmptyProvider(FunctionToolsetCapability):
            pass

        provider = EmptyProvider(name="empty")
        assert isinstance(provider, AbstractCapability)


@pytest.mark.unit
class TestDebugToolsAsCapability:
    """Tests for DebugTools as a capability."""

    async def test_is_abstract_capability(self) -> None:
        """DebugTools is an AbstractCapability."""
        provider = DebugTools()
        assert isinstance(provider, AbstractCapability)

    async def test_toolset_contains_expected_tools(self) -> None:
        """Capability toolset includes introspection and platform_paths tools."""
        provider = DebugTools()
        toolset = await _resolve_toolset(provider)

        assert toolset is not None
        tools = await toolset.get_tools(_make_run_context())

        assert "execute_introspection" in tools
        assert "get_platform_paths" in tools


@pytest.mark.unit
class TestSubagentToolsAsCapability:
    """Tests for SubagentTools as a capability."""

    async def test_is_abstract_capability(self) -> None:
        """SubagentTools is an AbstractCapability."""
        provider = SubagentTools()
        assert isinstance(provider, AbstractCapability)

    async def test_toolset_contains_expected_tools(self) -> None:
        """Capability toolset includes task tool."""
        provider = SubagentTools()
        toolset = await _resolve_toolset(provider)

        assert toolset is not None
        tools = await toolset.get_tools(_make_run_context())

        assert "list_available_nodes" not in tools
        assert "task" in tools


@pytest.mark.unit
class TestSkillsToolsAsCapability:
    """Tests for skill loading tools as a capability.

    SkillsTools has been consolidated into SkillManagerCap (unify-skill-loading).
    These tests verify that load_skill and list_skills are exposed via
    SkillManagerCap.get_toolset().
    """

    async def test_skill_manager_cap_exposes_load_and_list_tools(self) -> None:
        """SkillManagerCap toolset includes load_skill and list_skills tools."""
        from wolfharness.capabilities.skill_manager_cap import SkillManagerCap
        from wolfharness.skills.skill import Skill

        local_skill = Skill(
            name="test-skill",
            description="A test skill",
            skill_path=PurePosixPath("skill://test-skill"),
            instructions="Test instructions.",
        )
        cap = SkillManagerCap(local_skills={"test-skill": local_skill})
        toolset = await _resolve_toolset(cap)

        assert toolset is not None
        tools = await toolset.get_tools(_make_run_context())

        assert "load_skill" in tools
        assert "list_skills" in tools


@pytest.mark.unit
class TestCodeToolsAsCapability:
    """Tests for CodeTools as a capability."""

    async def test_is_abstract_capability(self) -> None:
        """CodeTools is an AbstractCapability."""
        provider = CodeTools()
        assert isinstance(provider, AbstractCapability)

    async def test_toolset_contains_format_code_tool(self) -> None:
        """Capability toolset always includes format_code tool."""
        provider = CodeTools()
        toolset = await _resolve_toolset(provider)

        assert toolset is not None
        tools = await toolset.get_tools(_make_run_context())

        assert "format_code" in tools


@pytest.mark.unit
class TestProcessManagementToolsAsCapability:
    """Tests for ProcessManagementTools as a capability."""

    async def test_is_abstract_capability(self) -> None:
        """ProcessManagementTools is an AbstractCapability."""
        provider = ProcessManagementTools()
        assert isinstance(provider, AbstractCapability)

    async def test_toolset_contains_expected_tools(self) -> None:
        """Capability toolset includes process management tools."""
        provider = ProcessManagementTools()
        toolset = await _resolve_toolset(provider)

        assert toolset is not None
        tools = await toolset.get_tools(_make_run_context())

        assert "start_process" in tools
        assert "get_process_output" in tools
        assert "wait_for_process" in tools
        assert "kill_process" in tools
        assert "release_process" in tools
        assert "list_processes" in tools


@pytest.mark.unit
class TestWorkersToolsAsCapability:
    """Tests for WorkersTools as a capability."""

    async def test_is_abstract_capability_with_no_workers(self) -> None:
        """WorkersTools with no workers is an AbstractCapability."""
        provider = WorkersTools(workers=[])
        assert isinstance(provider, AbstractCapability)

    async def test_empty_workers_yields_no_tools(self) -> None:
        """WorkersTools with empty workers list produces empty toolset."""
        provider = WorkersTools(workers=[])
        toolset = await _resolve_toolset(provider)

        assert toolset is None
