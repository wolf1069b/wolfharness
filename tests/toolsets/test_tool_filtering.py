"""Tests for tool filtering in toolset configurations."""

from __future__ import annotations

import pytest

from wolfharness.capabilities.filtered_toolset import FilteredToolsetCapability
from wolfharness_config.toolsets import (
    CodeToolsetConfig,
    SkillsToolsetConfig,
    SubagentToolsetConfig,
)


pytestmark = pytest.mark.unit


async def test_subagent_tool_filtering():
    """Test filtering tools in subagent toolset."""
    # Unfiltered provider has all tools
    config_all = SubagentToolsetConfig()
    provider_all = config_all.get_provider()
    tools = await provider_all.get_tools()
    tool_names = {t.name for t in tools}

    assert "task" in tool_names
    assert "list_available_nodes" not in tool_names

    # Filtered config wraps in FilteredToolsetCapability
    config = SubagentToolsetConfig(tools={"task": True})
    provider = config.get_provider()
    assert isinstance(provider, FilteredToolsetCapability)

    # Inner capability still has all tools
    inner_tools = await provider.wrapped.get_tools()
    inner_names = {t.name for t in inner_tools}
    assert "task" in inner_names
    assert "list_available_nodes" not in inner_names


async def test_skills_tool_filtering():
    """Test filtering in skills toolset (now empty — tools owned by SkillManagerCap)."""
    # SkillsToolsetConfig.get_provider() now returns an empty
    # FunctionToolsetCapability since load_skill/list_skills are owned
    # by SkillManagerCap (unify-skill-loading change).
    config_all = SkillsToolsetConfig()
    provider_all = config_all.get_provider()
    tools = await provider_all.get_tools()
    tool_names = {t.name for t in tools}

    # No tools from SkillsToolsetConfig anymore.
    assert "load_skill" not in tool_names
    assert "list_skills" not in tool_names

    # Filtered config still wraps in FilteredToolsetCapability
    config = SkillsToolsetConfig(tools={"load_skill": False})
    provider = config.get_provider()
    assert isinstance(provider, FilteredToolsetCapability)


async def test_code_toolset_filtering():
    """Test filtering tools in code toolset."""
    # Unfiltered provider has all tools
    config_all = CodeToolsetConfig()
    provider_all = config_all.get_provider()
    tools = await provider_all.get_tools()
    tool_names = {t.name for t in tools}

    assert "format_code" in tool_names
    assert "run_diagnostics" in tool_names

    # Filtered config wraps in FilteredToolsetCapability
    config = CodeToolsetConfig(tools={"format_code": True, "ast_grep": False})
    provider = config.get_provider()
    assert isinstance(provider, FilteredToolsetCapability)

    # Inner capability still has all tools
    inner_tools = await provider.wrapped.get_tools()
    inner_names = {t.name for t in inner_tools}
    assert "format_code" in inner_names
    assert "run_diagnostics" in inner_names


async def test_filtering_provider_delegates_attributes():
    """Test that FilteredToolsetCapability delegates attributes correctly."""
    config = SubagentToolsetConfig(tools={"task": True})
    provider = config.get_provider()

    assert isinstance(provider, FilteredToolsetCapability)
    # Name delegates to wrapped capability
    assert provider.name == "subagent_tools"
    # Wrapped capability is accessible
    assert provider.wrapped.name == "subagent_tools"
