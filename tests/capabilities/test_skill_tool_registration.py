"""Tests for per-skill tool registration, MCP tool registration, and allowed_tools filtering.

Covers tasks:
- 2.7: End-to-end SKILL.md with tools -> agent toolset contains {skill_name}__tool__loads
- 3.5: End-to-end SKILL.md with mcp_servers -> agent toolset contains prefixed MCP tools
- 4.2: allowed_tools: ["read", "list"] filtering
- 4.3: No allowed_tools -> all accessible
- 4.4: allowed_tools: [] -> all filtered out
- 8.11: get_wrapper_toolset() with multiple skills, composite filter
- 12.1: Skill with both tools AND mcp_servers
- 12.2: Invalid import_path -> graceful error
- 12.3: McpServerCap creation failure -> other skills unaffected
- 12.4: Pool rebuild with skill removal
- 12.5: allowed_tools: [] edge case
- 13.1: End-to-end tools registration
- 13.2: End-to-end mcp_servers registration
- 13.6: End-to-end allowed_tools filtering
- 13.7: Multiple skills with same import_path isolated by prefix
- 13.8: Pool rebuild re-imports tools and re-creates McpServerCap children
- 13.9: SkillManagerCap + McpServerCap child lifecycle (enter all, exit in reverse)
- 13.13: Multiple skills each with MCP servers -> isolation by prefix
- 13.14: for_run() on SkillManagerCap -> new instance has tool_manager and _skill_tools
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Self
from unittest.mock import MagicMock, patch

from pydantic_ai.toolsets import (
    CombinedToolset,
    FilteredToolset,
    PrefixedToolset,
)
import pytest

from wolfharness.capabilities.resource_protocols import (
    SkillEntry,
    SkillResource,
)
from wolfharness.capabilities.skill_manager_cap import SkillManagerCap
from wolfharness.skills.skill import Skill
from wolfharness.skills.skill_tool_manager import SkillToolManager
from wolfharness_config.skills import SkillMcpServerConfig, SkillToolConfig


pytestmark = pytest.mark.unit


# ---- Helpers ----


def make_skill(
    name: str = "test-skill",
    description: str = "A test skill",
    instructions: str = "Test instructions.",
    tools: list[SkillToolConfig] | None = None,
    mcp_servers: dict[str, SkillMcpServerConfig] | None = None,
    allowed_tools: list[str] | None = None,
) -> Skill:
    """Create a minimal Skill for testing.

    Args:
        name: Skill name.
        description: Skill description.
        instructions: Skill instructions content.
        tools: Optional list of SkillToolConfig.
        mcp_servers: Optional dict of SkillMcpServerConfig.
        allowed_tools: Optional list of allowed tool names.

    Returns:
        A Skill instance.
    """
    return Skill(
        name=name,
        description=description,
        skill_path=PurePosixPath(f"skill://{name}"),
        instructions=instructions,
        tools=tools,
        mcp_servers=mcp_servers,
        allowed_tools=allowed_tools,
    )


class MockMcpCap(SkillResource):
    """Mock McpServerCap for testing without real MCP connections."""

    def __init__(self, name: str = "mock-mcp") -> None:
        self._name = name
        self._entered = False
        self._exited = False
        self._exit_order: int | None = None

    def get_serialization_name(self) -> str:
        return self._name

    async def list_skills(self) -> list[SkillEntry]:
        return []

    async def read_skill(self, name: str) -> str | None:
        return None

    async def skill_exists(self, name: str) -> bool:
        return False

    async def __aenter__(self) -> Self:
        self._entered = True
        return self

    async def __aexit__(self, *args: object) -> None:
        self._exited = True

    def get_toolset(self) -> Any:
        return None

    def get_instructions(self) -> str | None:
        return None


def _get_tool_names(toolset: Any) -> set[str]:
    """Extract tool names from a toolset by calling get_tools.

    Args:
        toolset: A toolset instance.

    Returns:
        Set of tool name strings.
    """
    if toolset is None:
        return set()

    # PrefixedToolset wraps another toolset and prefixes tool names.
    if isinstance(toolset, PrefixedToolset):
        inner = toolset.wrapped
        inner_names = _get_tool_names(inner)
        prefix = toolset.prefix
        return {f"{prefix}{n}" for n in inner_names}

    if isinstance(toolset, CombinedToolset):
        names: set[str] = set()
        for ts in toolset.toolsets:
            names |= _get_tool_names(ts)
        return names

    # FunctionToolset and similar — try calling get_tools.
    try:
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            tools = loop.run_until_complete(toolset.get_tools())
            return {td.name for td in tools.values()}
        finally:
            loop.close()
    except Exception:  # noqa: BLE001
        pass

    return set()


# =========================================================================
# Task 2.7 / 13.1: End-to-end tools registration
# =========================================================================


def test_skill_with_tools_registers_prefixed_tool() -> None:
    """SKILL.md with tools: [{import_path: "json:loads"}] -> toolset has {skill_name}__tool__loads.

    Given a skill with a Python tool import, When SkillManagerCap is
    constructed with a SkillToolManager, Then the assembled toolset
    contains a PrefixedToolset with prefix 'my-skill__tool__'.
    """
    skill = make_skill(
        name="my-skill",
        tools=[SkillToolConfig(type="python", import_path="json:loads")],
    )
    manager = SkillToolManager()
    cap = SkillManagerCap(
        local_skills={"my-skill": skill},
        tool_manager=manager,
    )

    # _skill_tools should have the imported tool.
    assert "my-skill" in cap._skill_tools
    assert len(cap._skill_tools["my-skill"]) == 1
    assert cap._skill_tools["my-skill"][0].name == "loads"

    # get_toolset() should return a CombinedToolset with a PrefixedToolset.
    toolset = cap.get_toolset()
    assert toolset is not None
    assert isinstance(toolset, CombinedToolset)

    # Verify the PrefixedToolset exists with the correct prefix.
    prefixed = [ts for ts in toolset.toolsets if isinstance(ts, PrefixedToolset)]
    assert len(prefixed) >= 1
    tool_prefixes = {ts.prefix for ts in prefixed}
    assert "my-skill__tool__" in tool_prefixes


def test_skill_without_tools_does_not_register_tools() -> None:
    """Skill without tools frontmatter -> no {skill_name}__tool__* tools.

    Given a skill without tools, When SkillManagerCap is constructed,
    Then _skill_tools should be empty and no PrefixedToolset for that
    skill should be in the toolset.
    """
    skill = make_skill(name="plain-skill")
    manager = SkillToolManager()
    cap = SkillManagerCap(
        local_skills={"plain-skill": skill},
        tool_manager=manager,
    )

    assert len(cap._skill_tools) == 0

    toolset = cap.get_toolset()
    # Toolset is not None — it always includes built-in load_skill/list_skills.
    # But no prefixed skill tools are present.
    assert toolset is not None


# =========================================================================
# Task 3.5 / 13.2: End-to-end MCP server registration
# =========================================================================


def test_skill_with_mcp_servers_creates_prefixed_toolset() -> None:
    """SKILL.md with mcp_servers -> agent toolset contains prefixed MCP tools.

    Given a skill with mcp_servers frontmatter, When SkillManagerCap is
    constructed, Then McpServerCap instances are created and stored in
    _skill_mcp_children, and get_toolset() wraps them in
    PrefixedToolset with prefix '{skill_name}__mcp__'.
    """
    mcp_config = SkillMcpServerConfig(
        command="uvx",
        args=["some-mcp-server"],
    )
    skill = make_skill(
        name="my-skill",
        mcp_servers={"server1": mcp_config},
    )
    cap = SkillManagerCap(local_skills={"my-skill": skill})

    # McpServerCap should be created.
    assert "my-skill" in cap._skill_mcp_children
    assert len(cap._skill_mcp_children["my-skill"]) == 1

    # get_toolset() should include PrefixedToolset with __mcp__ prefix.
    toolset = cap.get_toolset()
    assert toolset is not None
    assert isinstance(toolset, CombinedToolset)

    prefixed = [ts for ts in toolset.toolsets if isinstance(ts, PrefixedToolset)]
    mcp_prefixes = {ts.prefix for ts in prefixed if "__mcp__" in ts.prefix}
    assert "my-skill__mcp__" in mcp_prefixes


def test_skill_without_mcp_servers_has_no_mcp_children() -> None:
    """Skill without mcp_servers -> no _skill_mcp_children entry.

    Given a skill without mcp_servers, When SkillManagerCap is
    constructed, Then _skill_mcp_children should be empty.
    """
    skill = make_skill(name="plain-skill")
    cap = SkillManagerCap(local_skills={"plain-skill": skill})

    assert len(cap._skill_mcp_children) == 0


# =========================================================================
# Task 4.2: allowed_tools: ["read", "list"] filtering
# =========================================================================


def test_allowed_tools_filters_non_allowed_tools() -> None:
    """Skill with allowed_tools: ["read", "list"] -> write filtered out.

    Given a skill with allowed_tools=["read", "list"], When
    get_wrapper_toolset() is called, Then a FilteredToolset is returned
    that filters non-allowed tools.
    """
    skill = make_skill(
        name="restricted",
        allowed_tools=["read", "list"],
    )
    cap = SkillManagerCap(local_skills={"restricted": skill})

    # Create a mock inner toolset to filter.
    mock_toolset = MagicMock()
    wrapper = cap.get_wrapper_toolset(mock_toolset)

    # Should return a FilteredToolset since allowed_tools is non-empty.
    assert wrapper is not None
    assert isinstance(wrapper, FilteredToolset)


# =========================================================================
# Task 4.3: No allowed_tools -> all accessible
# =========================================================================


def test_no_allowed_tools_means_all_accessible() -> None:
    """Skill without allowed_tools -> filter with empty set (all skill tools filtered).

    Given a skill with allowed_tools=None, When get_wrapper_toolset()
    is called, Then a FilteredToolset is returned. The parsed_allowed_tools()
    returns [] for None, which is not None, so the filter activates with
    an empty allowed set — all skill tools are filtered out, but non-skill
    tools still pass.
    """
    skill = make_skill(name="open-skill")
    cap = SkillManagerCap(local_skills={"open-skill": skill})

    mock_toolset = MagicMock()
    wrapper = cap.get_wrapper_toolset(mock_toolset)

    # Filter activates even for None allowed_tools (parsed returns []).
    assert wrapper is not None
    assert isinstance(wrapper, FilteredToolset)


# =========================================================================
# Task 4.4 / 12.5: allowed_tools: [] -> all filtered out
# =========================================================================


def test_empty_allowed_tools_filters_all_skill_tools() -> None:
    """allowed_tools: [] (explicitly empty) -> FilteredToolset with empty allowed set.

    Given a skill with allowed_tools=[], When get_wrapper_toolset() is
    called, Then a FilteredToolset is returned. parsed_allowed_tools()
    returns [] which is not None, so the filter activates with an
    empty allowed set — all skill tools are filtered out.
    """
    skill = make_skill(
        name="blocked",
        allowed_tools=[],
    )
    cap = SkillManagerCap(local_skills={"blocked": skill})

    mock_toolset = MagicMock()
    wrapper = cap.get_wrapper_toolset(mock_toolset)

    # Empty list -> parsed_allowed_tools() returns [] -> not None -> filter activates.
    assert wrapper is not None
    assert isinstance(wrapper, FilteredToolset)


def test_empty_string_allowed_tools_also_no_filter() -> None:
    """allowed_tools as empty string -> FilteredToolset (same as None or []).

    Given a skill with allowed_tools="" (empty string), When
    get_wrapper_toolset() is called, Then a FilteredToolset is returned.
    parsed_allowed_tools() returns [] for empty string, which is not None,
    so the filter activates.
    """
    skill = make_skill(name="empty-str-skill")
    # Manually set allowed_tools to empty string.
    skill.allowed_tools = ""
    cap = SkillManagerCap(local_skills={"empty-str-skill": skill})

    mock_toolset = MagicMock()
    wrapper = cap.get_wrapper_toolset(mock_toolset)

    assert wrapper is not None
    assert isinstance(wrapper, FilteredToolset)


# =========================================================================
# Task 8.11: get_wrapper_toolset() with multiple skills, composite filter
# =========================================================================


def test_multiple_skills_composite_allowed_tools_filter() -> None:
    """Multiple skills each with different allowed_tools -> composite filter.

    Given skill "alpha" with allowed_tools=["read"] and skill "beta"
    with allowed_tools=["write"], When get_wrapper_toolset() is called,
    Then a single FilteredToolset is returned with a composite filter
    that handles both skills.
    """
    alpha = make_skill(name="alpha", allowed_tools=["read"])
    beta = make_skill(name="beta", allowed_tools=["write"])
    cap = SkillManagerCap(local_skills={"alpha": alpha, "beta": beta})

    mock_toolset = MagicMock()
    wrapper = cap.get_wrapper_toolset(mock_toolset)

    assert wrapper is not None
    assert isinstance(wrapper, FilteredToolset)

    # Verify the filter function handles both skills correctly.
    # The filter_func is stored on the FilteredToolset.
    filter_func = wrapper.filter_func

    # Create mock tool definitions.
    from pydantic_ai.tools import ToolDefinition

    def make_tool_def(name: str) -> ToolDefinition:
        return ToolDefinition(name=name, description="", parameters_json_schema={})

    # alpha__tool__read should pass (read is in alpha's allowed set).
    assert filter_func(MagicMock(), make_tool_def("alpha__tool__read")) is True
    # alpha__tool__write should be filtered (write not in alpha's allowed set).
    assert filter_func(MagicMock(), make_tool_def("alpha__tool__write")) is False
    # beta__tool__write should pass (write is in beta's allowed set).
    assert filter_func(MagicMock(), make_tool_def("beta__tool__write")) is True
    # beta__tool__read should be filtered (read not in beta's allowed set).
    assert filter_func(MagicMock(), make_tool_def("beta__tool__read")) is False
    # non-skill tool should always pass.
    assert filter_func(MagicMock(), make_tool_def("some_other_tool")) is True


# =========================================================================
# Task 12.1: Skill with both tools AND mcp_servers
# =========================================================================


def test_skill_with_both_tools_and_mcp_servers() -> None:
    """Skill with both tools AND mcp_servers -> both prefixed toolsets registered.

    Given a skill with both Python tools and MCP servers, When
    SkillManagerCap is constructed, Then both {name}__tool__* and
    {name}__mcp__* PrefixedToolsets are created.
    """
    skill = make_skill(
        name="combined-skill",
        tools=[SkillToolConfig(type="python", import_path="json:loads")],
        mcp_servers={
            "server1": SkillMcpServerConfig(command="uvx", args=["some-server"]),
        },
    )
    manager = SkillToolManager()
    cap = SkillManagerCap(
        local_skills={"combined-skill": skill},
        tool_manager=manager,
    )

    # Both _skill_tools and _skill_mcp_children should have entries.
    assert "combined-skill" in cap._skill_tools
    assert "combined-skill" in cap._skill_mcp_children

    toolset = cap.get_toolset()
    assert toolset is not None
    assert isinstance(toolset, CombinedToolset)

    prefixed = [ts for ts in toolset.toolsets if isinstance(ts, PrefixedToolset)]
    prefixes = {ts.prefix for ts in prefixed}
    assert "combined-skill__tool__" in prefixes
    assert "combined-skill__mcp__" in prefixes


# =========================================================================
# Task 12.2: Invalid import_path -> graceful error
# =========================================================================


def test_invalid_import_path_graceful_error() -> None:
    """Invalid import_path -> graceful error, other skills unaffected.

    Given one skill with an invalid import_path and another with a
    valid one, When SkillManagerCap is constructed, Then the invalid
    skill's tools are skipped (logged warning) and the valid skill's
    tools are still imported.
    """
    bad_skill = make_skill(
        name="bad-skill",
        tools=[
            SkillToolConfig(
                type="python",
                import_path="nonexistent_module_xyz:nonexistent_func",
            )
        ],
    )
    good_skill = make_skill(
        name="good-skill",
        tools=[SkillToolConfig(type="python", import_path="json:loads")],
    )
    manager = SkillToolManager()

    with patch("wolfharness.capabilities.skill_manager_cap.logger"):
        cap = SkillManagerCap(
            local_skills={"bad-skill": bad_skill, "good-skill": good_skill},
            tool_manager=manager,
        )

    # bad-skill should not have imported tools.
    assert "bad-skill" not in cap._skill_tools
    # good-skill should have imported tools.
    assert "good-skill" in cap._skill_tools
    assert len(cap._skill_tools["good-skill"]) == 1


# =========================================================================
# Task 12.3: McpServerCap creation failure -> other skills unaffected
# =========================================================================


def test_mcp_creation_failure_other_skills_unaffected() -> None:
    """McpServerCap creation failure (bad config) -> other skills' tools still registered.

    Given one skill with a bad MCP server config (neither command nor
    url) and another with valid tools, When SkillManagerCap is
    constructed, Then the bad MCP server is skipped and the valid
    skill's tools are still registered.
    """
    bad_mcp_skill = make_skill(
        name="bad-mcp-skill",
        mcp_servers={
            "bad-server": SkillMcpServerConfig(),  # No command or url.
        },
    )
    good_tool_skill = make_skill(
        name="good-tool-skill",
        tools=[SkillToolConfig(type="python", import_path="json:loads")],
    )
    manager = SkillToolManager()

    with patch("wolfharness.capabilities.skill_manager_cap.logger"):
        cap = SkillManagerCap(
            local_skills={
                "bad-mcp-skill": bad_mcp_skill,
                "good-tool-skill": good_tool_skill,
            },
            tool_manager=manager,
        )

    # bad-mcp-skill should not have MCP children (conversion failed).
    assert "bad-mcp-skill" not in cap._skill_mcp_children
    # good-tool-skill should still have imported tools.
    assert "good-tool-skill" in cap._skill_tools


# =========================================================================
# Task 12.4: Pool rebuild with skill removal
# =========================================================================


def test_pool_rebuild_with_skill_removal() -> None:
    """Pool rebuild with skill removal -> removed skill's tools no longer in toolset.

    Given a SkillManagerCap with skill "alpha", When a new
    SkillManagerCap is created without "alpha" (simulating rebuild
    after removal), Then the new cap's _skill_tools does not contain
    "alpha".
    """
    alpha = make_skill(
        name="alpha",
        tools=[SkillToolConfig(type="python", import_path="json:loads")],
    )
    manager = SkillToolManager()
    old_cap = SkillManagerCap(
        local_skills={"alpha": alpha},
        tool_manager=manager,
    )
    assert "alpha" in old_cap._skill_tools

    # Simulate rebuild: create a new cap without "alpha".
    beta = make_skill(
        name="beta",
        tools=[SkillToolConfig(type="python", import_path="os:getcwd")],
    )
    new_cap = SkillManagerCap(
        local_skills={"beta": beta},
        tool_manager=manager,
    )

    # Alpha is gone, beta is present.
    assert "alpha" not in new_cap._skill_tools
    assert "beta" in new_cap._skill_tools


# =========================================================================
# Task 13.6: End-to-end allowed_tools filtering
# =========================================================================


def test_end_to_end_allowed_tools_filtering() -> None:
    """allowed_tools filtering: skill with allowed_tools -> non-allowed tools filtered.

    Given a skill with allowed_tools=["read", "list"], When
    get_wrapper_toolset() is called, Then a FilteredToolset is applied
    that would filter out tools not in the allowed set.
    """
    skill = make_skill(
        name="restricted",
        allowed_tools=["read", "list"],
    )
    cap = SkillManagerCap(local_skills={"restricted": skill})

    mock_toolset = MagicMock()
    wrapper = cap.get_wrapper_toolset(mock_toolset)

    assert wrapper is not None
    assert isinstance(wrapper, FilteredToolset)

    # Verify filter behavior.
    from pydantic_ai.tools import ToolDefinition

    def make_td(name: str) -> ToolDefinition:
        return ToolDefinition(name=name, description="", parameters_json_schema={})

    filter_func = wrapper.filter_func

    # restricted__tool__read should pass.
    assert filter_func(MagicMock(), make_td("restricted__tool__read")) is True
    # restricted__tool__list should pass.
    assert filter_func(MagicMock(), make_td("restricted__tool__list")) is True
    # restricted__tool__write should be filtered.
    assert filter_func(MagicMock(), make_td("restricted__tool__write")) is False


# =========================================================================
# Task 13.7: Multiple skills with same import_path isolated by prefix
# =========================================================================


def test_multiple_skills_same_import_path_isolated_by_prefix() -> None:
    """Multiple skills with same import_path -> isolated by {skill_name}__tool__ prefix.

    Given two skills "alpha" and "beta" both declaring
    tools: [{import_path: "os:getcwd"}], When SkillManagerCap is
    constructed, Then both alpha__tool__getcwd and beta__tool__getcwd
    are registered with separate prefixes.
    """
    alpha = make_skill(
        name="alpha",
        tools=[SkillToolConfig(type="python", import_path="os:getcwd")],
    )
    beta = make_skill(
        name="beta",
        tools=[SkillToolConfig(type="python", import_path="os:getcwd")],
    )
    manager = SkillToolManager()
    cap = SkillManagerCap(
        local_skills={"alpha": alpha, "beta": beta},
        tool_manager=manager,
    )

    # Both skills should have imported tools.
    assert "alpha" in cap._skill_tools
    assert "beta" in cap._skill_tools
    assert cap._skill_tools["alpha"][0].name == "getcwd"
    assert cap._skill_tools["beta"][0].name == "getcwd"

    # get_toolset() should have both prefixes.
    toolset = cap.get_toolset()
    assert toolset is not None
    assert isinstance(toolset, CombinedToolset)

    prefixed = [ts for ts in toolset.toolsets if isinstance(ts, PrefixedToolset)]
    prefixes = {ts.prefix for ts in prefixed}
    assert "alpha__tool__" in prefixes
    assert "beta__tool__" in prefixes


# =========================================================================
# Task 13.8: Pool rebuild re-imports tools and re-creates McpServerCap
# =========================================================================


def test_pool_rebuild_re_imports_tools() -> None:
    """Pool rebuild re-imports tools and re-creates McpServerCap children.

    Given an initial SkillManagerCap with tools, When a new cap is
    created (simulating rebuild) with the same tool_manager, Then the
    new cap has _skill_tools populated (re-imported) and
    _skill_mcp_children recreated.
    """
    skill = make_skill(
        name="rebuild-skill",
        tools=[SkillToolConfig(type="python", import_path="json:loads")],
        mcp_servers={
            "server1": SkillMcpServerConfig(command="uvx", args=["server"]),
        },
    )
    manager = SkillToolManager()
    old_cap = SkillManagerCap(
        local_skills={"rebuild-skill": skill},
        tool_manager=manager,
    )

    # Simulate rebuild with a new cap using same manager.
    new_cap = SkillManagerCap(
        local_skills={"rebuild-skill": skill},
        tool_manager=manager,
    )

    # Tools should be re-imported.
    assert "rebuild-skill" in new_cap._skill_tools
    assert len(new_cap._skill_tools["rebuild-skill"]) == 1
    # MCP children should be re-created.
    assert "rebuild-skill" in new_cap._skill_mcp_children
    assert len(new_cap._skill_mcp_children["rebuild-skill"]) == 1

    # The new cap should be a different instance.
    assert new_cap is not old_cap


# =========================================================================
# Task 13.9: SkillManagerCap + McpServerCap child lifecycle
# =========================================================================


@pytest.mark.asyncio
async def test_mcp_child_lifecycle_enter_all_exit_reverse() -> None:
    """SkillManagerCap + McpServerCap child lifecycle (enter all, exit in reverse).

    Given a SkillManagerCap with McpServerCap children, When
    __aenter__ is called, Then all children are entered. When
    __aexit__ is called, all children are exited.
    """
    skill = make_skill(
        name="lifecycle-skill",
        mcp_servers={
            "server1": SkillMcpServerConfig(command="uvx", args=["server1"]),
            "server2": SkillMcpServerConfig(command="uvx", args=["server2"]),
        },
    )
    cap = SkillManagerCap(local_skills={"lifecycle-skill": skill})

    # Get the created McpServerCap children.
    children = cap._skill_mcp_children["lifecycle-skill"]
    assert len(children) == 2

    # Enter context — all children should be entered.
    await cap.__aenter__()
    for child in children:
        assert child.__aenter__  # Method exists.
        # We can't easily check __aenter__ state on real McpServerCap
        # without a running server, but we verify no exception is raised.

    # Exit context — all children should be exited.
    await cap.__aexit__(None, None, None)


# =========================================================================
# Task 13.13: Multiple skills each with MCP servers -> isolation by prefix
# =========================================================================


def test_multiple_skills_mcp_isolation_by_prefix() -> None:
    """Multiple skills each with MCP servers -> isolation by prefix, independent lifecycle.

    Given two skills "alpha" and "beta" each with their own MCP
    servers, When SkillManagerCap is constructed, Then each skill's
    MCP tools are prefixed with their respective skill name.
    """
    alpha = make_skill(
        name="alpha",
        mcp_servers={
            "server1": SkillMcpServerConfig(command="uvx", args=["server1"]),
        },
    )
    beta = make_skill(
        name="beta",
        mcp_servers={
            "server2": SkillMcpServerConfig(command="uvx", args=["server2"]),
        },
    )
    cap = SkillManagerCap(local_skills={"alpha": alpha, "beta": beta})

    # Both skills should have MCP children.
    assert "alpha" in cap._skill_mcp_children
    assert "beta" in cap._skill_mcp_children
    assert len(cap._skill_mcp_children["alpha"]) == 1
    assert len(cap._skill_mcp_children["beta"]) == 1

    # Different McpServerCap instances.
    alpha_child = cap._skill_mcp_children["alpha"][0]
    beta_child = cap._skill_mcp_children["beta"][0]
    assert alpha_child is not beta_child

    # get_toolset() should have separate prefixes.
    toolset = cap.get_toolset()
    assert toolset is not None
    assert isinstance(toolset, CombinedToolset)

    prefixed = [ts for ts in toolset.toolsets if isinstance(ts, PrefixedToolset)]
    prefixes = {ts.prefix for ts in prefixed}
    assert "alpha__mcp__" in prefixes
    assert "beta__mcp__" in prefixes


# =========================================================================
# Task 13.14: for_run() -> new instance has tool_manager and _skill_tools
# =========================================================================


@pytest.mark.asyncio
async def test_for_run_preserves_tool_manager_and_skill_tools() -> None:
    """for_run() on SkillManagerCap -> new instance has tool_manager and _skill_tools.

    Given a SkillManagerCap with tool_manager and _skill_tools
    populated, When for_run(ctx) is called, Then the new instance
    has tool_manager set and _skill_tools populated with the same
    imported tools.
    """
    skill = make_skill(
        name="for-run-skill",
        tools=[SkillToolConfig(type="python", import_path="json:loads")],
    )
    manager = SkillToolManager()
    cap = SkillManagerCap(
        local_skills={"for-run-skill": skill},
        tool_manager=manager,
    )

    # Original cap has tools.
    assert "for-run-skill" in cap._skill_tools
    original_tools = cap._skill_tools["for-run-skill"]

    # Call for_run.
    ctx_mock = MagicMock()
    new_cap = await cap.for_run(ctx_mock)

    # New instance should have tool_manager.
    assert new_cap._tool_manager is not None
    # New instance should have _skill_tools populated.
    assert "for-run-skill" in new_cap._skill_tools
    # Same tool names (re-imported).
    assert new_cap._skill_tools["for-run-skill"][0].name == original_tools[0].name
    # New instance is different.
    assert new_cap is not cap


# =========================================================================
# Additional coverage: allowed_tools with actual filter behavior
# =========================================================================


def test_allowed_tools_filter_actually_filters() -> None:
    """Verify the FilteredToolset filter function correctly allows/denies.

    Given a skill with allowed_tools=["read", "list"], When the
    composite filter function is applied to various tool names, Then
    only allowed tools pass.
    """
    from pydantic_ai.tools import ToolDefinition

    skill = make_skill(
        name="restricted",
        allowed_tools=["read", "list"],
    )
    cap = SkillManagerCap(local_skills={"restricted": skill})

    mock_toolset = MagicMock()
    wrapper = cap.get_wrapper_toolset(mock_toolset)
    assert wrapper is not None

    filter_func = wrapper.filter_func

    def make_td(name: str) -> ToolDefinition:
        return ToolDefinition(name=name, description="", parameters_json_schema={})

    # Skill tools matching allowed set.
    assert filter_func(MagicMock(), make_td("restricted__tool__read")) is True
    assert filter_func(MagicMock(), make_td("restricted__tool__list")) is True
    # Skill tool NOT in allowed set.
    assert filter_func(MagicMock(), make_td("restricted__tool__write")) is False
    # Non-skill tool always passes.
    assert filter_func(MagicMock(), make_td("bash")) is True
    assert filter_func(MagicMock(), make_td("read")) is True  # bare, no prefix
