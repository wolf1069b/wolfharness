"""Tests for SkillCommand dataclass."""

from __future__ import annotations

import pytest
from upathtools import UPath

from wolfharness.skills.command import SkillCommand
from wolfharness.skills.skill import Skill


pytestmark = pytest.mark.unit


@pytest.fixture
def sample_skill() -> Skill:
    """Create a sample Skill for testing."""
    return Skill(
        name="test-skill",
        description="A test skill for unit testing",
        skill_path=UPath("/tmp/test-skill"),
        metadata={"key": "value"},
    )


@pytest.fixture
def skill_command(sample_skill: Skill) -> SkillCommand:
    """Create a sample SkillCommand for testing."""
    return SkillCommand(
        name="test-skill",
        description="A test skill command",
        skill=sample_skill,
    )


def test_dataclass_instantiation_with_all_fields(sample_skill: Skill) -> None:
    """Test that SkillCommand can be instantiated with all fields."""
    command = SkillCommand(
        name="my-skill",
        description="My custom skill command",
        skill=sample_skill,
        input_hint="Provide file path",
        category="custom",
    )

    assert command.name == "my-skill"
    assert command.description == "My custom skill command"
    assert command.skill == sample_skill
    assert command.input_hint == "Provide file path"
    assert command.category == "custom"


def test_dataclass_instantiation_with_defaults(sample_skill: Skill) -> None:
    """Test that SkillCommand uses default values when not provided."""
    command = SkillCommand(
        name="test-skill",
        description="A test skill command",
        skill=sample_skill,
    )

    assert command.name == "test-skill"
    assert command.description == "A test skill command"
    assert command.skill == sample_skill
    assert command.input_hint == "Arguments for skill"
    assert command.category == "skill"


def test_frozen_immutability(sample_skill: Skill) -> None:
    """Test that frozen dataclass cannot be modified after creation."""
    command = SkillCommand(
        name="test-skill",
        description="A test skill command",
        skill=sample_skill,
    )

    with pytest.raises(AttributeError):
        command.name = "new-name"  # type: ignore[misc]

    with pytest.raises(AttributeError):
        command.description = "new description"  # type: ignore[misc]

    with pytest.raises(AttributeError):
        command.input_hint = "new hint"  # type: ignore[misc]

    with pytest.raises(AttributeError):
        command.category = "new category"  # type: ignore[misc]


def test_is_valid_input_with_valid_text(skill_command: SkillCommand) -> None:
    """Test that is_valid_input returns True for non-empty input."""
    is_valid, error = skill_command.is_valid_input("some input text")
    assert is_valid is True
    assert error is None


def test_is_valid_input_with_whitespace_only(skill_command: SkillCommand) -> None:
    """Test that is_valid_input returns False for whitespace-only input."""
    is_valid, error = skill_command.is_valid_input("   ")
    assert is_valid is False
    assert error == "Input cannot be empty"


def test_is_valid_input_with_empty_string(skill_command: SkillCommand) -> None:
    """Test that is_valid_input returns False for empty string."""
    is_valid, error = skill_command.is_valid_input("")
    assert is_valid is False
    assert error == "Input cannot be empty"


def test_accessing_nested_skill_attributes(sample_skill: Skill) -> None:
    """Test that nested Skill attributes are accessible through SkillCommand."""
    command = SkillCommand(
        name="test-skill",
        description="A test skill command",
        skill=sample_skill,
    )

    assert command.skill.name == "test-skill"
    assert command.skill.description == "A test skill for unit testing"
    assert command.skill.metadata == {"key": "value"}
    assert command.skill.license is None
    assert command.skill.compatibility is None


def test_skill_command_equality(sample_skill: Skill) -> None:
    """Test that SkillCommand instances can be compared for equality."""
    command1 = SkillCommand(
        name="test-skill",
        description="A test skill command",
        skill=sample_skill,
    )
    command2 = SkillCommand(
        name="test-skill",
        description="A test skill command",
        skill=sample_skill,
    )
    command3 = SkillCommand(
        name="other-skill",
        description="A different skill command",
        skill=sample_skill,
    )

    assert command1 == command2
    assert command1 != command3


def test_skill_command_unhashable_with_pydantic_skill(sample_skill: Skill) -> None:
    """Test that SkillCommand cannot be hashed due to unhashable Skill field.

    The underlying Skill is a Pydantic model without frozen=True,
    which makes it unhashable. This is expected behavior.
    """
    command = SkillCommand(
        name="test-skill",
        description="A test skill command",
        skill=sample_skill,
    )

    # Should raise TypeError because Skill is not hashable
    with pytest.raises(TypeError, match="unhashable type"):
        hash(command)


def test_skill_command_repr(sample_skill: Skill) -> None:
    """Test that SkillCommand has a useful repr."""
    command = SkillCommand(
        name="test-skill",
        description="A test skill command",
        skill=sample_skill,
    )

    repr_str = repr(command)
    assert "SkillCommand" in repr_str
    assert "test-skill" in repr_str
