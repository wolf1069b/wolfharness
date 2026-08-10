"""Tests for SkillsRegistry event emission."""

from __future__ import annotations

import pytest
from upathtools import UPath

from wolfharness.skills.registry import SkillsRegistry
from wolfharness.skills.skill import Skill


pytestmark = pytest.mark.unit


def create_test_skill(name: str = "test-skill", description: str = "A test skill") -> Skill:
    """Create a minimal test skill."""
    return Skill(
        name=name,
        description=description,
        skill_path=UPath("/tmp/test-skill"),
    )


class TestSkillAddedEvents:
    """Tests for skill addition events."""

    def test_callback_fires_when_skill_added(self) -> None:
        """Test that registered callback is called when skill is added."""
        registry = SkillsRegistry()
        called_with: list[tuple[str, Skill]] = []

        def on_added(name: str, _skill: Skill) -> None:
            called_with.append((name, _skill))

        registry.on_skill_added(on_added)
        skill = create_test_skill("test-skill")
        registry.register("test-skill", skill)

        assert len(called_with) == 1
        assert called_with[0][0] == "test-skill"
        assert called_with[0][1] is skill

    def test_multiple_callbacks_supported(self) -> None:
        """Test that multiple callbacks can be registered and all are called."""
        registry = SkillsRegistry()
        calls_1: list[tuple[str, Skill]] = []
        calls_2: list[tuple[str, Skill]] = []

        def callback_1(name: str, skill: Skill) -> None:
            calls_1.append((name, skill))

        def callback_2(name: str, skill: Skill) -> None:
            calls_2.append((name, skill))

        registry.on_skill_added(callback_1)
        registry.on_skill_added(callback_2)

        skill = create_test_skill("multi-callback-skill")
        registry.register("multi-callback-skill", skill)

        assert len(calls_1) == 1
        assert len(calls_2) == 1
        assert calls_1[0][0] == "multi-callback-skill"
        assert calls_2[0][0] == "multi-callback-skill"

    def test_no_errors_when_no_callbacks_registered(self) -> None:
        """Test that registration works without any callbacks (backward compat)."""
        registry = SkillsRegistry()
        skill = create_test_skill("no-callback-skill")

        # Should not raise any errors
        registry.register("no-callback-skill", skill)

        assert "no-callback-skill" in registry

    def test_callback_receives_correct_skill(self) -> None:
        """Test that callback receives the exact skill that was registered."""
        registry = SkillsRegistry()
        received_skill: Skill | None = None

        def on_added(name: str, _skill: Skill) -> None:
            nonlocal received_skill
            received_skill = _skill

        registry.on_skill_added(on_added)
        skill = create_test_skill("specific-skill", "A specific test skill")
        registry.register("specific-skill", skill)

        assert received_skill is skill
        assert received_skill is not None
        assert received_skill.name == "specific-skill"
        assert received_skill.description == "A specific test skill"


class TestSkillRemovedEvents:
    """Tests for skill removal events."""

    def test_callback_fires_when_skill_removed(self) -> None:
        """Test that registered callback is called when skill is removed."""
        registry = SkillsRegistry()
        called_with: list[tuple[str, None]] = []

        def on_removed(name: str, _skill: None) -> None:
            called_with.append((name, _skill))

        registry.on_skill_removed(on_removed)
        skill = create_test_skill("remove-skill")
        registry.register("remove-skill", skill)
        del registry["remove-skill"]

        assert len(called_with) == 1
        assert called_with[0][0] == "remove-skill"
        assert called_with[0][1] is None

    def test_removal_callbacks_multiple_skills(self) -> None:
        """Test removal callbacks fire correctly for multiple skills."""
        registry = SkillsRegistry()
        removed_names: list[str] = []

        def on_removed(name: str, _skill: None) -> None:
            removed_names.append(name)

        registry.on_skill_removed(on_removed)

        skill1 = create_test_skill("skill-1")
        skill2 = create_test_skill("skill-2")
        registry.register("skill-1", skill1)
        registry.register("skill-2", skill2)

        del registry["skill-1"]
        del registry["skill-2"]

        assert len(removed_names) == 2
        assert "skill-1" in removed_names
        assert "skill-2" in removed_names

    def test_no_errors_when_no_removal_callbacks(self) -> None:
        """Test that removal works without any callbacks (backward compat)."""
        registry = SkillsRegistry()
        skill = create_test_skill("remove-no-callback")
        registry.register("remove-no-callback", skill)

        # Should not raise any errors
        del registry["remove-no-callback"]

        assert "remove-no-callback" not in registry

    def test_removing_nonexistent_skill_raises_error(self) -> None:
        """Test that removing a non-existent skill raises an error."""
        registry = SkillsRegistry()

        with pytest.raises(Exception, match="nonexistent"):
            del registry["nonexistent-skill"]


class TestCombinedEvents:
    """Tests for combined addition and removal events."""

    def test_both_callbacks_work_together(self) -> None:
        """Test that both add and remove callbacks work when registered together."""
        registry = SkillsRegistry()
        added: list[str] = []
        removed: list[str] = []

        def on_added(name: str, _skill: Skill) -> None:
            added.append(name)

        def on_removed(name: str, _skill: None) -> None:
            removed.append(name)

        registry.on_skill_added(on_added)
        registry.on_skill_removed(on_removed)

        skill = create_test_skill("lifecycle-skill")
        registry.register("lifecycle-skill", skill)
        del registry["lifecycle-skill"]

        assert added == ["lifecycle-skill"]
        assert removed == ["lifecycle-skill"]

    def test_replace_skill_triggers_add_callback(self) -> None:
        """Test that replacing a skill triggers the add callback."""
        registry = SkillsRegistry()
        added_skills: list[str] = []

        def on_added(name: str, _skill: Skill) -> None:
            added_skills.append(name)

        registry.on_skill_added(on_added)

        skill1 = create_test_skill("replaceable-skill")
        skill2 = create_test_skill("replaceable-skill")

        registry.register("replaceable-skill", skill1)
        # Replace with replace=True
        registry.register("replaceable-skill", skill2, replace=True)

        # Should be called twice (once for initial, once for replacement)
        assert added_skills.count("replaceable-skill") == 2


class TestBatchInitialization:
    """Tests for batch initialization scenarios."""

    def test_batch_registration_emits_individual_events(self) -> None:
        """Test that batch registration emits individual events for each skill."""
        registry = SkillsRegistry()
        added_names: list[str] = []

        def on_added(name: str, _skill: Skill) -> None:
            added_names.append(name)

        registry.on_skill_added(on_added)

        skills = [
            create_test_skill("batch-1"),
            create_test_skill("batch-2"),
            create_test_skill("batch-3"),
        ]

        # Simulate batch registration
        for skill in skills:
            registry.register(skill.name, skill)

        assert len(added_names) == 3
        assert "batch-1" in added_names
        assert "batch-2" in added_names
        assert "batch-3" in added_names

    def test_empty_registry_has_no_callbacks(self) -> None:
        """Test that a fresh registry starts with empty callback lists."""
        registry = SkillsRegistry()

        assert registry._skill_added_handlers == []
        assert registry._skill_removed_handlers == []
