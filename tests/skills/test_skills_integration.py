"""Integration tests for configurable skill loading paths in AgentPool."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from textwrap import dedent
from typing import Any

import pytest
from upathtools import UPath
import yaml

from wolfharness.delegation.pool import AgentPool


pytestmark = pytest.mark.integration


def create_skill(path: Path, name: str, description: str, instructions: str):
    """Create a skill in the specified directory."""
    skill_dir = path / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        dedent(f"""
        ---
        name: {name}
        description: {description}
        ---
        {instructions}
        """).strip()
    )


@pytest.fixture
def temp_skills():
    """Create temporary skill directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base = Path(temp_dir)
        dir_a = base / "dir_a"
        dir_b = base / "dir_b"
        dir_a.mkdir()
        dir_b.mkdir()

        create_skill(dir_a, "skill-a", "Description A", "Instructions A")
        create_skill(dir_b, "skill-b", "Description B", "Instructions B")
        create_skill(dir_b, "conflict-skill", "Conflict from B", "Instructions Conflict B")
        create_skill(dir_a, "conflict-skill", "Conflict from A", "Instructions Conflict A")

        yield dir_a, dir_b


@pytest.mark.asyncio
async def test_skills_backward_compatibility():
    """Init pool with a manifest having NO skills section.

    Assert that default paths are searched.
    """
    # Create a manifest without skills section
    manifest_dict: dict[str, Any] = {
        "agents": {
            "test_agent": {
                "type": "native",
                "model": "openai:gpt-4o",
            }
        }
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yml"
        config_path.write_text(yaml.dump(manifest_dict))

        async with AgentPool(config_path) as pool:
            # We can't easily assert default paths existence since they depend on the environment,
            # but we can verify that the SkillsManager is initialized with default config.
            assert pool.skills._config is not None  # type: ignore
            assert pool.skills._config.include_default is True  # type: ignore
            assert pool.skills._config.paths == []  # type: ignore


@pytest.mark.asyncio
async def test_skills_custom_path(temp_skills: tuple[Path, Path]):
    """Init pool with a manifest having a custom skills.paths.

    Assert that skills from that path are loaded.
    """
    dir_a, _ = temp_skills

    manifest_dict: dict[str, Any] = {
        "skills": {"paths": [str(dir_a)], "include_default": False},
        "agents": {
            "test_agent": {
                "type": "native",
                "model": "openai:gpt-4o",
            }
        },
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yml"
        config_path.write_text(yaml.dump(manifest_dict))

        async with AgentPool(config_path) as pool:
            skills = pool.skills.list_skills()
            skill_names = [s.name for s in skills]
            assert "skill-a" in skill_names
            assert "skill-b" not in skill_names

            skill = pool.skills.get_skill("skill-a")
            assert skill.description == "Description A"


@pytest.mark.asyncio
async def test_skills_disable_defaults(monkeypatch: pytest.MonkeyPatch):
    """Init pool with skills.include_default: false.

    Assert that default paths are NOT searched.
    """
    from wolfharness_config import skills

    # Mock default paths to something we can control
    with tempfile.TemporaryDirectory() as temp_dir:
        default_dir = Path(temp_dir) / "default_skills"
        default_dir.mkdir()
        create_skill(default_dir, "default-skill", "Default", "Instructions")

        monkeypatch.setattr(skills, "DEFAULT_SKILLS_PATHS", [UPath(default_dir)])

        manifest_dict: dict[str, Any] = {
            "skills": {"include_default": False},
            "agents": {
                "test_agent": {
                    "type": "native",
                    "model": "openai:gpt-4o",
                }
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir_2:
            config_path = Path(temp_dir_2) / "config.yml"
            config_path.write_text(yaml.dump(manifest_dict))

            async with AgentPool(config_path) as pool:
                skills_list = pool.skills.list_skills()
                skill_names = [s.name for s in skills_list]
                assert "default-skill" not in skill_names


@pytest.mark.asyncio
async def test_skills_conflict_resolution(temp_skills: tuple[Path, Path]):
    """Init pool with two paths containing the same skill name.

    Assert that the version from the EARLIER path in the list is the one loaded.
    """
    dir_a, dir_b = temp_skills

    # [dir_a, dir_b] -> dir_a should win
    manifest_dict: dict[str, Any] = {
        "skills": {"paths": [str(dir_a), str(dir_b)], "include_default": False},
        "agents": {
            "test_agent": {
                "type": "native",
                "model": "openai:gpt-4o",
            }
        },
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yml"
        config_path.write_text(yaml.dump(manifest_dict))

        async with AgentPool(config_path) as pool:
            skill = pool.skills.get_skill("conflict-skill")
            assert skill.description == "Conflict from A"

    # [dir_b, dir_a] -> dir_b should win
    manifest_dict["skills"]["paths"] = [str(dir_b), str(dir_a)]

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yml"
        config_path.write_text(yaml.dump(manifest_dict))

        async with AgentPool(config_path) as pool:
            skill = pool.skills.get_skill("conflict-skill")
            assert skill.description == "Conflict from B"


@pytest.mark.asyncio
async def test_skills_relative_paths():
    """Init pool from a YAML file that specifies relative skill paths.

    Assert that they are resolved correctly relative to the YAML file.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir)
        # Create a relative path from config_dir to dir_a
        # In this test, we can just move dir_a inside config_dir/skills
        skills_dir = config_dir / "my_skills"
        skills_dir.mkdir()
        create_skill(skills_dir, "rel-skill", "Relative Description", "Instructions")

        manifest_dict: dict[str, Any] = {
            "skills": {"paths": ["./my_skills"], "include_default": False},
            "agents": {
                "test_agent": {
                    "type": "native",
                    "model": "openai:gpt-4o",
                }
            },
        }

        config_path = config_dir / "config.yml"
        config_path.write_text(yaml.dump(manifest_dict))

        # Run from a different CWD to ensure relative path is resolved against config file
        old_cwd = Path.cwd()
        os.chdir(tempfile.gettempdir())
        try:
            async with AgentPool(config_path) as pool:
                skill = pool.skills.get_skill("rel-skill")
                assert skill is not None
                assert skill.description == "Relative Description"
        finally:
            os.chdir(old_cwd)
