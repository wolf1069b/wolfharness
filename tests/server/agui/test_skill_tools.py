"""Integration tests for AG-UI skill tools bridge.

Tests the AGUISkillToolAdapter and AGUISkillBridge classes which convert
SkillCommand instances to AG-UI Tool format for protocol interoperability.
"""

from __future__ import annotations

import pytest
from upathtools import UPath

from wolfharness.skills.command import SkillCommand
from wolfharness.skills.skill import Skill
from wolfharness_server.agui_server.skill_tools import (
    AGUISkillBridge,
    AGUISkillToolAdapter,
)


pytestmark = pytest.mark.integration


# Skip all tests in this module if ag_ui is not available
try:
    import ag_ui  # noqa: F401
except ImportError:
    pytest.skip("ag_ui module not available", allow_module_level=True)


@pytest.fixture
def sample_skill() -> Skill:
    """Create a sample Skill for testing."""
    return Skill(
        name="test-skill",
        description="A test skill for integration testing",
        skill_path=UPath("/tmp/test-skill"),
        metadata={"key": "value"},
    )


@pytest.fixture
def sample_command(sample_skill: Skill) -> SkillCommand:
    """Create a sample SkillCommand for testing."""
    return SkillCommand(
        name="test-skill",
        description="Execute test skill operations",
        skill=sample_skill,
        input_hint="Provide test arguments",
        category="test",
    )


@pytest.fixture
def sample_adapter(sample_command: SkillCommand) -> AGUISkillToolAdapter:
    """Create a sample AGUISkillToolAdapter for testing."""
    return AGUISkillToolAdapter(skill_cmd=sample_command)


@pytest.fixture
def empty_bridge() -> AGUISkillBridge:
    """Create an empty AGUISkillBridge for testing."""
    return AGUISkillBridge()


class TestAGUISkillToolAdapter:
    """Tests for AGUISkillToolAdapter class."""

    def test_to_agui_tool_creates_tool(self, sample_adapter: AGUISkillToolAdapter) -> None:
        """Verify adapter creates AG-UI Tool correctly."""
        tool = sample_adapter.to_agui_tool()

        assert tool is not None
        assert isinstance(tool.name, str)
        assert isinstance(tool.description, str)
        assert isinstance(tool.parameters, dict)

    def test_tool_name_format(self, sample_adapter: AGUISkillToolAdapter) -> None:
        """Verify skill__{name} prefix format is applied correctly."""
        tool = sample_adapter.to_agui_tool()

        assert tool.name.startswith("skill__")
        assert tool.name == "skill__test-skill"
        assert tool.name.removeprefix("skill__") == "test-skill"

    def test_tool_parameters_schema(self, sample_adapter: AGUISkillToolAdapter) -> None:
        """Validate OpenAI function schema structure in tool parameters."""
        tool = sample_adapter.to_agui_tool()

        # Verify schema follows OpenAI function calling format
        assert tool.parameters["type"] == "object"
        assert "properties" in tool.parameters
        assert "required" in tool.parameters
        assert isinstance(tool.parameters["properties"], dict)
        assert isinstance(tool.parameters["required"], list)

    def test_tool_has_string_arguments_parameter(
        self, sample_adapter: AGUISkillToolAdapter
    ) -> None:
        """Verify tool has single 'arguments' string parameter as expected."""
        tool = sample_adapter.to_agui_tool()

        assert "arguments" in tool.parameters["properties"]
        arguments_schema = tool.parameters["properties"]["arguments"]
        assert arguments_schema["type"] == "string"
        assert "description" in arguments_schema

    def test_tool_description_matches_skill(self, sample_command: SkillCommand) -> None:
        """Verify tool description is taken from skill command."""
        adapter = AGUISkillToolAdapter(skill_cmd=sample_command)
        tool = adapter.to_agui_tool()

        assert tool.description == sample_command.description
        assert tool.description == "Execute test skill operations"

    def test_tool_arguments_description_matches_input_hint(
        self, sample_command: SkillCommand
    ) -> None:
        """Verify arguments parameter description uses input_hint."""
        adapter = AGUISkillToolAdapter(skill_cmd=sample_command)
        tool = adapter.to_agui_tool()

        args_desc = tool.parameters["properties"]["arguments"]["description"]
        assert args_desc == sample_command.input_hint
        assert args_desc == "Provide test arguments"

    def test_multiple_skills_different_tools(self) -> None:
        """Verify different skills create different tools with unique names."""
        skill1 = Skill(
            name="skill-one",
            description="First skill",
            skill_path=UPath("/tmp/skill1"),
        )
        skill2 = Skill(
            name="skill-two",
            description="Second skill",
            skill_path=UPath("/tmp/skill2"),
        )

        cmd1 = SkillCommand(
            name="skill-one",
            description="First command",
            skill=skill1,
        )
        cmd2 = SkillCommand(
            name="skill-two",
            description="Second command",
            skill=skill2,
        )

        adapter1 = AGUISkillToolAdapter(skill_cmd=cmd1)
        adapter2 = AGUISkillToolAdapter(skill_cmd=cmd2)

        tool1 = adapter1.to_agui_tool()
        tool2 = adapter2.to_agui_tool()

        assert tool1.name == "skill__skill-one"
        assert tool2.name == "skill__skill-two"
        assert tool1.name != tool2.name
        assert tool1.description == "First command"
        assert tool2.description == "Second command"

    def test_adapter_preserves_skill_reference(self, sample_command: SkillCommand) -> None:
        """Verify adapter maintains reference to original skill command."""
        adapter = AGUISkillToolAdapter(skill_cmd=sample_command)

        assert adapter.skill_cmd is sample_command
        assert adapter.skill_cmd.name == "test-skill"
        assert adapter.skill_cmd.skill.name == "test-skill"


class TestAGUISkillBridge:
    """Tests for AGUISkillBridge class."""

    def test_handle_change_adds_tool(
        self, empty_bridge: AGUISkillBridge, sample_command: SkillCommand
    ) -> None:
        """Verify handle_change adds a new tool when command is provided."""
        bridge = empty_bridge

        # Initially empty
        assert len(bridge.get_tools()) == 0

        # Add a skill
        bridge.handle_change("test-skill", sample_command)

        # Should have one tool
        tools = bridge.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "skill__test-skill"

    def test_handle_change_removes_tool(
        self, empty_bridge: AGUISkillBridge, sample_command: SkillCommand
    ) -> None:
        """Verify handle_change removes tool when command is None."""
        bridge = empty_bridge

        # Add a skill first
        bridge.handle_change("test-skill", sample_command)
        assert len(bridge.get_tools()) == 1

        # Remove it
        bridge.handle_change("test-skill", None)

        # Should be empty
        assert len(bridge.get_tools()) == 0

    def test_handle_change_updates_tool(
        self, empty_bridge: AGUISkillBridge, sample_skill: Skill
    ) -> None:
        """Verify handle_change updates existing tool when re-registered."""
        bridge = empty_bridge

        # Add initial command
        command_v1 = SkillCommand(
            name="test-skill",
            description="Version 1",
            skill=sample_skill,
        )
        bridge.handle_change("test-skill", command_v1)

        tools = bridge.get_tools()
        assert len(tools) == 1
        assert tools[0].description == "Version 1"

        # Update with new description
        command_v2 = SkillCommand(
            name="test-skill",
            description="Version 2",
            skill=sample_skill,
        )
        bridge.handle_change("test-skill", command_v2)

        tools = bridge.get_tools()
        assert len(tools) == 1
        assert tools[0].description == "Version 2"

    def test_get_tools_returns_list(
        self, empty_bridge: AGUISkillBridge, sample_command: SkillCommand
    ) -> None:
        """Verify get_tools returns a list of Tool instances."""
        bridge = empty_bridge
        bridge.handle_change("test-skill", sample_command)

        tools = bridge.get_tools()

        assert isinstance(tools, list)
        assert len(tools) >= 0
        for tool in tools:
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "parameters")

    def test_get_tools_returns_empty_initially(self, empty_bridge: AGUISkillBridge) -> None:
        """Verify get_tools returns empty list when no skills registered."""
        bridge = empty_bridge

        tools = bridge.get_tools()

        assert tools == []
        assert len(tools) == 0

    def test_get_tools_multiple_tools(self, empty_bridge: AGUISkillBridge) -> None:
        """Verify get_tools returns multiple tools when registered."""
        bridge = empty_bridge

        # Create multiple skills
        for i in range(3):
            skill = Skill(
                name=f"skill-{i}",
                description=f"Skill number {i}",
                skill_path=UPath(f"/tmp/skill{i}"),
            )
            command = SkillCommand(
                name=f"skill-{i}",
                description=f"Command {i}",
                skill=skill,
            )
            bridge.handle_change(f"skill-{i}", command)

        tools = bridge.get_tools()

        assert len(tools) == 3
        tool_names = {t.name for t in tools}
        assert tool_names == {"skill__skill-0", "skill__skill-1", "skill__skill-2"}

    def test_get_handler_returns_adapter(
        self, empty_bridge: AGUISkillBridge, sample_command: SkillCommand
    ) -> None:
        """Verify get_handler returns adapter for valid tool name."""
        bridge = empty_bridge
        bridge.handle_change("test-skill", sample_command)

        adapter = bridge.get_handler("skill__test-skill")

        assert adapter is not None
        assert isinstance(adapter, AGUISkillToolAdapter)
        assert adapter.skill_cmd.name == "test-skill"

    def test_get_handler_with_prefix(self, empty_bridge: AGUISkillBridge) -> None:
        """Verify skill__ prefix handling in get_handler."""
        bridge = empty_bridge

        skill = Skill(
            name="my-skill",
            description="My skill",
            skill_path=UPath("/tmp/my-skill"),
        )
        command = SkillCommand(
            name="my-skill",
            description="My command",
            skill=skill,
        )
        bridge.handle_change("my-skill", command)

        # Without prefix - should not work
        adapter_no_prefix = bridge.get_handler("my-skill")
        assert adapter_no_prefix is None

        # With prefix - should work
        adapter_with_prefix = bridge.get_handler("skill__my-skill")
        assert adapter_with_prefix is not None
        assert isinstance(adapter_with_prefix, AGUISkillToolAdapter)

    def test_get_handler_returns_none_for_missing(self, empty_bridge: AGUISkillBridge) -> None:
        """Verify get_handler returns None for non-existent tools."""
        bridge = empty_bridge

        result = bridge.get_handler("skill__non-existent")

        assert result is None

    def test_get_handler_returns_none_without_prefix(
        self, empty_bridge: AGUISkillBridge, sample_command: SkillCommand
    ) -> None:
        """Verify get_handler returns None when tool name lacks skill__ prefix."""
        bridge = empty_bridge
        bridge.handle_change("test-skill", sample_command)

        # Try without prefix
        result = bridge.get_handler("test-skill")
        assert result is None

        # Try with wrong prefix
        result2 = bridge.get_handler("cmd__test-skill")
        assert result2 is None

    def test_handle_change_nonexistent_remove_is_noop(self, empty_bridge: AGUISkillBridge) -> None:
        """Verify removing non-existent skill is a no-op."""
        bridge = empty_bridge

        # Should not raise
        bridge.handle_change("non-existent", None)

        # Still empty
        assert len(bridge.get_tools()) == 0

    def test_multiple_skills_isolated(self, empty_bridge: AGUISkillBridge) -> None:
        """Verify multiple skills are isolated and don't interfere."""
        bridge = empty_bridge

        skill1 = Skill(
            name="skill-one",
            description="First",
            skill_path=UPath("/tmp/s1"),
        )
        skill2 = Skill(
            name="skill-two",
            description="Second",
            skill_path=UPath("/tmp/s2"),
        )

        bridge.handle_change(
            "skill-one", SkillCommand(name="skill-one", description="Cmd1", skill=skill1)
        )
        bridge.handle_change(
            "skill-two", SkillCommand(name="skill-two", description="Cmd2", skill=skill2)
        )

        # Remove one
        bridge.handle_change("skill-one", None)

        tools = bridge.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "skill__skill-two"

        # Verify correct adapter is returned
        adapter = bridge.get_handler("skill__skill-two")
        assert adapter is not None
        assert adapter.skill_cmd.name == "skill-two"


class TestToolExecutionFlow:
    """Integration tests for complete tool execution flows."""

    def test_handler_provides_adapter_for_tool(self) -> None:
        """Verify handler flow provides correct adapter for tool execution."""
        bridge = AGUISkillBridge()

        skill = Skill(
            name="exec-skill",
            description="Execution skill",
            skill_path=UPath("/tmp/exec"),
        )
        command = SkillCommand(
            name="exec-skill",
            description="Execute something",
            skill=skill,
            input_hint="Execution arguments",
        )

        bridge.handle_change("exec-skill", command)

        # Simulate tool lookup during execution
        adapter = bridge.get_handler("skill__exec-skill")

        assert adapter is not None
        tool = adapter.to_agui_tool()
        assert tool.name == "skill__exec-skill"
        assert "arguments" in tool.parameters["properties"]

    def test_adapter_has_correct_skill(self) -> None:
        """Verify adapter maintains correct skill reference through bridge."""
        bridge = AGUISkillBridge()

        skill = Skill(
            name="ref-skill",
            description="Reference skill",
            skill_path=UPath("/tmp/ref"),
            metadata={"test": "data"},
        )
        command = SkillCommand(
            name="ref-skill",
            description="Reference command",
            skill=skill,
        )

        bridge.handle_change("ref-skill", command)
        adapter = bridge.get_handler("skill__ref-skill")

        assert adapter is not None
        assert adapter.skill_cmd.skill is skill
        assert adapter.skill_cmd.skill.name == "ref-skill"
        assert adapter.skill_cmd.skill.metadata == {"test": "data"}

    def test_end_to_end_tool_creation_and_lookup(self) -> None:
        """Complete end-to-end test of tool creation and lookup flow."""
        bridge = AGUISkillBridge()

        # Create and register multiple skills
        skills_data = [
            ("code-review", "Review code", "Provide code to review"),
            ("refactor", "Refactor code", "Code to refactor"),
            ("test-gen", "Generate tests", "Function to test"),
        ]

        for name, desc, hint in skills_data:
            skill = Skill(
                name=name,
                description=f"{desc} skill",
                skill_path=UPath(f"/tmp/{name}"),
            )
            command = SkillCommand(
                name=name,
                description=desc,
                skill=skill,
                input_hint=hint,
            )
            bridge.handle_change(name, command)

        # Verify all tools are available
        tools = bridge.get_tools()
        assert len(tools) == 3

        # Verify each tool can be looked up
        for name, desc, _hint in skills_data:
            adapter = bridge.get_handler(f"skill__{name}")
            assert adapter is not None, f"Adapter for {name} should exist"

            tool = adapter.to_agui_tool()
            assert tool.name == f"skill__{name}"
            assert tool.description == desc

    def test_skill_command_frozen_integrity(self, sample_command: SkillCommand) -> None:
        """Verify frozen SkillCommand maintains integrity through adapter."""
        adapter = AGUISkillToolAdapter(skill_cmd=sample_command)

        # Verify we can access all fields
        assert adapter.skill_cmd.name == "test-skill"
        assert adapter.skill_cmd.description == "Execute test skill operations"
        assert adapter.skill_cmd.input_hint == "Provide test arguments"
        assert adapter.skill_cmd.category == "test"

        tool = adapter.to_agui_tool()
        assert tool.name.endswith(adapter.skill_cmd.name)

    def test_bridge_state_isolation(self) -> None:
        """Verify separate bridge instances have isolated state."""
        bridge1 = AGUISkillBridge()
        bridge2 = AGUISkillBridge()

        skill = Skill(
            name="isolate-skill",
            description="Isolation test",
            skill_path=UPath("/tmp/isolate"),
        )
        command = SkillCommand(
            name="isolate-skill",
            description="Isolation cmd",
            skill=skill,
        )

        # Add to bridge1 only
        bridge1.handle_change("isolate-skill", command)

        # bridge1 has it
        assert len(bridge1.get_tools()) == 1
        assert bridge1.get_handler("skill__isolate-skill") is not None

        # bridge2 does not
        assert len(bridge2.get_tools()) == 0
        assert bridge2.get_handler("skill__isolate-skill") is None


class TestEdgeCases:
    """Edge case tests for AG-UI skill tools bridge."""

    def test_skill_with_special_characters_in_description(self) -> None:
        """Verify skills with special characters in description work correctly."""
        skill = Skill(
            name="special-skill",
            description="Description with \"quotes\" and 'apostrophes' and <brackets>",
            skill_path=UPath("/tmp/special"),
        )
        command = SkillCommand(
            name="special-skill",
            description=skill.description,
            skill=skill,
        )
        adapter = AGUISkillToolAdapter(command)

        tool = adapter.to_agui_tool()
        assert tool.description == "Description with \"quotes\" and 'apostrophes' and <brackets>"

    def test_empty_metadata_skill(self) -> None:
        """Verify skills with empty metadata work correctly."""
        skill = Skill(
            name="minimal-skill",
            description="Minimal skill",
            skill_path=UPath("/tmp/minimal"),
            metadata={},
        )
        command = SkillCommand(
            name="minimal-skill",
            description="Minimal command",
            skill=skill,
        )
        adapter = AGUISkillToolAdapter(command)

        assert adapter.skill_cmd.skill.metadata == {}
        tool = adapter.to_agui_tool()
        assert tool.name == "skill__minimal-skill"

    def test_skill_with_license_and_compatibility(self) -> None:
        """Verify skills with optional fields work correctly."""
        skill = Skill(
            name="licensed-skill",
            description="Licensed skill",
            skill_path=UPath("/tmp/licensed"),
            license="MIT",
            compatibility="python>=3.10",
            allowed_tools="read,bash",
        )
        command = SkillCommand(
            name="licensed-skill",
            description="Licensed command",
            skill=skill,
        )
        adapter = AGUISkillToolAdapter(command)

        assert adapter.skill_cmd.skill.license == "MIT"
        assert adapter.skill_cmd.skill.compatibility == "python>=3.10"
        assert adapter.skill_cmd.skill.allowed_tools == "read,bash"

    def test_concurrent_add_remove_operations(self, empty_bridge: AGUISkillBridge) -> None:
        """Verify bridge handles multiple add/remove operations correctly."""
        bridge = empty_bridge

        # Add and remove same skill multiple times
        for i in range(5):
            skill = Skill(
                name="volatile-skill",
                description=f"Version {i}",
                skill_path=UPath("/tmp/volatile"),
            )
            command = SkillCommand(
                name="volatile-skill",
                description=f"Command {i}",
                skill=skill,
            )
            bridge.handle_change("volatile-skill", command)
            assert len(bridge.get_tools()) == 1
            assert bridge.get_tools()[0].description == f"Command {i}"

            bridge.handle_change("volatile-skill", None)
            assert len(bridge.get_tools()) == 0

    def test_hyphenated_skill_names(self) -> None:
        """Verify hyphenated skill names are handled correctly."""
        skill = Skill(
            name="my-awesome-skill",
            description="A skill with hyphens",
            skill_path=UPath("/tmp/my-awesome-skill"),
        )
        command = SkillCommand(
            name="my-awesome-skill",
            description="Awesome command",
            skill=skill,
        )
        adapter = AGUISkillToolAdapter(command)

        tool = adapter.to_agui_tool()
        assert tool.name == "skill__my-awesome-skill"

        bridge = AGUISkillBridge()
        bridge.handle_change("my-awesome-skill", command)

        adapter_from_bridge = bridge.get_handler("skill__my-awesome-skill")
        assert adapter_from_bridge is not None
        assert adapter_from_bridge.skill_cmd.name == "my-awesome-skill"
