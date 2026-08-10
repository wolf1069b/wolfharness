"""Integration tests for end-to-end skill resolution.

Tests cover real skill loading workflows through AgentPool,
including URI resolution, bare name resolution, reference content
loading, argument substitution, and multiple provider resolution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from upathtools import UPath

from wolfharness import AgentPool
from wolfharness.skills.exceptions import SecurityError
from wolfharness.skills.uri_resolver import ResolvedSkillURI
from wolfharness.tools.exceptions import ToolError


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from pathlib import Path


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_skill_content() -> str:
    """Create sample skill content for testing."""
    return """---
name: test-skill
description: A test skill for integration testing
---

# Test Skill Instructions

This is a test skill for integration testing.

## Arguments

- Arg 1: $1
- Arg 2: $2
- All args: $@

## Usage

Use this skill with: load_skill(ctx, "test-skill", "arg1 arg2")
"""


@pytest.fixture
def test_skill_with_reference(tmp_path: Path, test_skill_content: str) -> UPath:
    """Create a test skill directory with SKILL.md and reference file."""
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()

    # Create main SKILL.md
    skills_md = skill_dir / "SKILL.md"
    skills_md.write_text(test_skill_content)

    # Create references directory with reference file
    ref_dir = skill_dir / "references"
    ref_dir.mkdir()
    ref_file = ref_dir / "guide.md"
    ref_file.write_text("# Guide\n\nThis is a reference guide for test-skill.")

    return UPath(skill_dir)


@pytest.fixture
def python_expert_skill(tmp_path: Path) -> UPath:
    """Create a python-expert skill directory."""
    skill_dir = tmp_path / "python-expert"
    skill_dir.mkdir()

    content = """---
name: python-expert
description: Python programming expert skill
---

# Python Expert

You are a Python programming expert.

Use these best practices:
- Follow PEP 8
- Use type hints
- Write docstrings
"""
    skills_md = skill_dir / "SKILL.md"
    skills_md.write_text(content)

    return UPath(skill_dir)


@pytest.fixture
def docker_expert_skill(tmp_path: Path) -> UPath:
    """Create a docker-expert skill directory."""
    skill_dir = tmp_path / "docker-expert"
    skill_dir.mkdir()

    content = """---
name: docker-expert
description: Docker and containerization expert
---

# Docker Expert

You are a Docker expert.

Key concepts:
- Images vs Containers
- Dockerfile best practices
- Multi-stage builds
"""
    skills_md = skill_dir / "SKILL.md"
    skills_md.write_text(content)

    return UPath(skill_dir)


# =============================================================================
# Test Class: EndToEndSkillLoading
# =============================================================================


@pytest.mark.integration
class TestEndToEndSkillLoading:
    """Test end-to-end skill loading through AgentPool."""

    async def test_skill_loading_by_bare_name(
        self,
        tmp_path: Path,
        python_expert_skill: UPath,
    ) -> None:
        """Test loading skill using bare name through SkillsManager."""
        import yaml

        config_path = tmp_path / "config.yml"
        config_content = {
            "agents": {
                "test_agent": {
                    "type": "native",
                    "model": "openai:gpt-5-nano",
                    "system_prompt": "You are a test agent",
                }
            },
            "skills": {
                "paths": [str(tmp_path)],
                "include_default": False,
            },
        }
        config_path.write_text(yaml.dump(config_content))

        async with AgentPool(str(config_path)) as pool:
            # Get skill via SkillsManager (bare name lookup)
            skill_names = [s.name for s in pool.skills.list_skills()]
            assert "python-expert" in skill_names

            # Get instructions via SkillsManager
            instructions = pool.skills.get_skill_instructions("python-expert")
            assert "Python Expert" in instructions

    async def test_bare_name_falls_back_to_skills_manager(
        self,
        tmp_path: Path,
        docker_expert_skill: UPath,
    ) -> None:
        """Test that bare skill name uses SkillsManager."""
        import yaml

        config_path = tmp_path / "config.yml"
        config_content = {
            "agents": {
                "test_agent": {
                    "type": "native",
                    "model": "openai:gpt-5-nano",
                    "system_prompt": "You are a test agent",
                }
            },
            "skills": {
                "paths": [str(tmp_path)],
                "include_default": False,
            },
        }
        config_path.write_text(yaml.dump(config_content))

        async with AgentPool(str(config_path)) as pool:
            # Skills should be available through SkillsManager
            skills = pool.skills.list_skills()
            skill_map = {s.name: s for s in skills}

            assert "docker-expert" in skill_map
            assert "Docker and containerization" in skill_map["docker-expert"].description


# =============================================================================
# Test Class: ReferenceContentLoading
# =============================================================================


@pytest.mark.integration
class TestReferenceContentLoading:
    """Test loading skill reference content."""

    async def test_reference_path_resolution(
        self,
        tmp_path: Path,
        test_skill_with_reference: UPath,
    ) -> None:
        """Test resolving URI with reference path."""
        # Parse URI with reference (D9 flat format: skill://skill-name/ref/path)
        resolved = ResolvedSkillURI.parse("skill://test-skill/references/guide.md")

        assert resolved.skill_name == "test-skill"
        assert resolved.reference_path == "references/guide.md"

    async def test_reference_file_loading(
        self,
        tmp_path: Path,
        test_skill_with_reference: UPath,
    ) -> None:
        """Test loading reference file content from skill."""
        import yaml

        config_path = tmp_path / "config.yml"
        config_content = {
            "agents": {
                "test_agent": {
                    "type": "native",
                    "model": "openai:gpt-5-nano",
                    "system_prompt": "You are a test agent",
                }
            },
            "skills": {
                "paths": [str(tmp_path)],
                "include_default": False,
            },
        }
        config_path.write_text(yaml.dump(config_content))

        async with AgentPool(str(config_path)) as pool:
            # Get skill via SkillsManager
            skill = pool.skills.get_skill("test-skill")

            # Check reference file exists
            ref_file = skill.skill_path / "references" / "guide.md"
            assert ref_file.exists()

            content = ref_file.read_text()
            assert "This is a reference guide" in content


# =============================================================================
# Test Class: ArgumentSubstitution
# =============================================================================


@pytest.mark.integration
class TestArgumentSubstitution:
    """Test argument substitution in skill instructions."""

    async def test_positional_argument_substitution(
        self,
        tmp_path: Path,
        test_skill_with_reference: UPath,
    ) -> None:
        """Test $1, $2 argument substitution."""
        from wolfharness_toolsets.builtin.skills import _substitute_arguments

        instructions = "First: $1, Second: $2"
        result = _substitute_arguments(instructions, "hello world")

        assert result == "First: hello, Second: world"

    async def test_all_arguments_substitution(
        self,
        tmp_path: Path,
        test_skill_with_reference: UPath,
    ) -> None:
        """Test $@ and $ARGUMENTS substitution."""
        from wolfharness_toolsets.builtin.skills import _substitute_arguments

        instructions = "All: $@, Also: $ARGUMENTS"
        result = _substitute_arguments(instructions, "arg1 arg2 arg3")

        assert result == "All: arg1 arg2 arg3, Also: arg1 arg2 arg3"

    async def test_mixed_argument_substitution(
        self,
        tmp_path: Path,
        test_skill_with_reference: UPath,
    ) -> None:
        """Test mixed positional and all-arguments substitution."""
        from wolfharness_toolsets.builtin.skills import _substitute_arguments

        instructions = "First: $1, Rest: $@"
        result = _substitute_arguments(instructions, "alpha beta gamma")

        assert result == "First: alpha, Rest: alpha beta gamma"

    async def test_no_arguments_substitution(
        self,
        tmp_path: Path,
        test_skill_with_reference: UPath,
    ) -> None:
        """Test behavior when no arguments provided."""
        from wolfharness_toolsets.builtin.skills import _substitute_arguments

        instructions = "Args: $1, All: $@"
        result = _substitute_arguments(instructions, None)

        # Should remain unchanged when no args provided
        assert result == "Args: $1, All: $@"


# =============================================================================
# Test Class: MultipleSkillsResolution
# =============================================================================


@pytest.mark.integration
class TestMultipleSkillsResolution:
    """Test skill resolution with multiple skills."""

    async def test_multiple_skills_loading(
        self,
        tmp_path: Path,
        python_expert_skill: UPath,
        docker_expert_skill: UPath,
    ) -> None:
        """Test loading multiple skills from same directory."""
        import yaml

        config_path = tmp_path / "config.yml"
        config_content = {
            "agents": {
                "test_agent": {
                    "type": "native",
                    "model": "openai:gpt-5-nano",
                    "system_prompt": "You are a test agent",
                }
            },
            "skills": {
                "paths": [str(tmp_path)],
                "include_default": False,
            },
        }
        config_path.write_text(yaml.dump(config_content))

        async with AgentPool(str(config_path)) as pool:
            skills = pool.skills.list_skills()
            skill_names = {s.name for s in skills}

            assert "python-expert" in skill_names
            assert "docker-expert" in skill_names


# =============================================================================
# Test Class: ErrorHandlingAndSecurity
# =============================================================================


@pytest.mark.integration
class TestErrorHandlingAndSecurity:
    """Test error handling and security in skill resolution."""

    async def test_skill_not_found_error(
        self,
        tmp_path: Path,
        python_expert_skill: UPath,
    ) -> None:
        """Test SkillNotFoundError for non-existent skill."""
        import yaml

        config_path = tmp_path / "config.yml"
        config_content = {
            "agents": {
                "test_agent": {
                    "type": "native",
                    "model": "openai:gpt-5-nano",
                    "system_prompt": "You are a test agent",
                }
            },
            "skills": {
                "paths": [str(tmp_path)],
                "include_default": False,
            },
        }
        config_path.write_text(yaml.dump(config_content))

        async with AgentPool(str(config_path)) as pool:
            # Test via SkillsManager
            with pytest.raises(ToolError, match="non-existent-skill"):
                pool.skills.get_skill("non-existent-skill")

    async def test_path_traversal_detection(
        self,
        tmp_path: Path,
        python_expert_skill: UPath,
    ) -> None:
        """Test path traversal detection in skill URIs."""
        with pytest.raises(SecurityError, match="Path traversal detected"):
            ResolvedSkillURI.parse("skill://skill-name/../other")

    async def test_null_byte_detection(
        self,
        tmp_path: Path,
        python_expert_skill: UPath,
    ) -> None:
        """Test null byte detection in skill URIs."""
        with pytest.raises(SecurityError, match="null bytes"):
            ResolvedSkillURI.parse("skill://skill\x00name")


# =============================================================================
# Test Class: BackwardCompatibility
# =============================================================================


@pytest.mark.integration
class TestBackwardCompatibility:
    """Test backward compatibility with existing skill loading."""

    async def test_skills_manager_still_works(
        self,
        tmp_path: Path,
        python_expert_skill: UPath,
    ) -> None:
        """Test that pool.skills still works as before."""
        import yaml

        config_path = tmp_path / "config.yml"
        config_content = {
            "agents": {
                "test_agent": {
                    "type": "native",
                    "model": "openai:gpt-5-nano",
                    "system_prompt": "You are a test agent",
                }
            },
            "skills": {
                "paths": [str(tmp_path)],
                "include_default": False,
            },
        }
        config_path.write_text(yaml.dump(config_content))

        async with AgentPool(str(config_path)) as pool:
            # SkillsManager should still be functional
            all_skills = pool.skills.list_skills()
            assert len(all_skills) > 0

            # Can still get instructions via SkillsManager
            instructions = pool.skills.get_skill_instructions("python-expert")
            assert "Python Expert" in instructions

    async def test_bare_skill_name_parsing(
        self,
        tmp_path: Path,
        python_expert_skill: UPath,
    ) -> None:
        """Test parsing bare skill names without URI scheme."""
        # Bare skill name should parse successfully
        resolved = ResolvedSkillURI.parse("python-expert")

        assert resolved.skill_name == "python-expert"
        assert resolved.reference_path is None

    async def test_both_resolution_methods_available(
        self,
        tmp_path: Path,
        python_expert_skill: UPath,
        docker_expert_skill: UPath,
    ) -> None:
        """Test that both old and new resolution methods work."""
        import yaml

        config_path = tmp_path / "config.yml"
        config_content = {
            "agents": {
                "test_agent": {
                    "type": "native",
                    "model": "openai:gpt-5-nano",
                    "system_prompt": "You are a test agent",
                }
            },
            "skills": {
                "paths": [str(tmp_path)],
                "include_default": False,
            },
        }
        config_path.write_text(yaml.dump(config_content))

        async with AgentPool(str(config_path)) as pool:
            # Old SkillsManager-based resolution
            all_skills = pool.skills.list_skills()
            skill_via_manager = next(s for s in all_skills if s.name == "docker-expert")
            assert skill_via_manager.name == "docker-expert"
