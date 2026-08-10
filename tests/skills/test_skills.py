"""Tests for skills functionality."""

from __future__ import annotations

from pathlib import Path
import tempfile
from textwrap import dedent
from typing import TYPE_CHECKING

import pytest
from upathtools import UPath

from wolfharness.skills.registry import SkillsRegistry
from wolfharness.skills.skill import Skill


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def temp_skills_dir() -> Generator[Path]:
    """Create a temporary directory with test skills."""
    with tempfile.TemporaryDirectory() as temp_dir:
        skills_dir = Path(temp_dir) / "test_skills"
        skills_dir.mkdir()

        # Create a test skill
        test_skill_dir = skills_dir / "test_skill"
        test_skill_dir.mkdir()

        skill_content = dedent("""
        ---
        name: test_skill
        description: A test skill for unit testing
        ---

        # Test Skill Instructions

        This is a test skill that demonstrates the skills system.

        ## Usage

        Use this skill when testing the skills functionality.
        """).strip()

        (test_skill_dir / "SKILL.md").write_text(skill_content)

        yield skills_dir


@pytest.fixture
def isolated_registry(temp_skills_dir):
    """Create a registry that only searches the test directory."""
    # Override the DEFAULT_SKILL_PATHS to prevent discovery of global skills
    original_paths = SkillsRegistry.DEFAULT_SKILL_PATHS
    SkillsRegistry.DEFAULT_SKILL_PATHS = []
    try:
        registry = SkillsRegistry(skills_dirs=[temp_skills_dir])
        yield registry
    finally:
        SkillsRegistry.DEFAULT_SKILL_PATHS = original_paths


def test_skill_load_instructions_preserves_domain_placeholders(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    fta_review_dir = skills_dir / "fta-review"
    fta_review_dir.mkdir()

    (fta_review_dir / "SKILL.md").write_text(
        dedent("""
        ---
        name: fta-review
        description: Review an FTA tree
        ---

        # FTA Review

        {{ reviewer_skill_catalog }}
        """).strip(),
        encoding="utf-8",
    )

    instructions = Skill.from_skill_dir(UPath(fta_review_dir)).load_instructions()

    assert "{{ reviewer_skill_catalog }}" in instructions
    assert "fta-causal-path-review" not in instructions
