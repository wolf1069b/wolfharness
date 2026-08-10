from __future__ import annotations

from pathlib import PurePosixPath
from types import SimpleNamespace
from typing import Any

import pytest
from upathtools import UPath

from wolfharness.capabilities.resource_protocols import SkillEntry, SkillResource
from wolfharness.capabilities.skill_manager_cap import SkillManagerCap
from wolfharness.skills.skill import Skill
from wolfharness_toolsets.builtin.skills import list_skills, load_skill, load_skill_for_node


def _write_skill(root, name: str, description: str) -> Skill:
    skill_dir = root / name
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{name} instructions\n",
        encoding="utf-8",
    )
    return Skill.from_skill_dir(UPath(skill_dir))


class _FakeSkillResource(SkillResource):
    """In-memory remote skill provider implementing SkillResource."""

    def __init__(self, skills: list[Skill]) -> None:
        self._skills = skills

    def get_serialization_name(self) -> str:
        return "scratchpad"

    async def list_skills(self) -> list[SkillEntry]:
        return [
            SkillEntry(
                name=skill.name,
                description=skill.description,
                uri=f"skill://scratchpad/{skill.name}",
                source="provider",
            )
            for skill in self._skills
        ]

    async def read_skill(self, skill_name: str) -> str | None:
        skill = next((s for s in self._skills if s.name == skill_name), None)
        if skill is None:
            return None
        return f"{skill_name} provider instructions"

    async def skill_exists(self, skill_name: str) -> bool:
        return any(s.name == skill_name for s in self._skills)


class _FakePool:
    """Minimal pool exposing scope functions + a real SkillManagerCap."""

    skill_resolver = None

    def __init__(self, skills: list[Skill], provider_skills: list[Skill] | None = None) -> None:
        self._provider_skills = provider_skills or []
        children = [_FakeSkillResource(self._provider_skills)] if self._provider_skills else []
        self.skill_capabilities = [
            SkillManagerCap(
                local_skills={s.name: s for s in skills},
                children=children,
                name="pool-skills",
            )
        ]

    def is_skill_visible_to_node(self, skill: Skill, node_name: str | None) -> bool:
        if node_name == "rebuttal_agent":
            return True
        return skill.metadata.get("scope") != "rebuttal_agent"


def _ctx(pool: _FakePool, node_name: str) -> Any:
    """Build a RuntimeAgentContext (AgentContext) deps with pool + node name."""
    from wolfharness.agents.context import AgentContext as RuntimeAgentContext

    node = SimpleNamespace(name=node_name)
    return RuntimeAgentContext(node=node, pool=pool)


@pytest.mark.unit
async def test_load_skill_filters_by_current_node_package_scope(tmp_path):
    host_skill = _write_skill(tmp_path, "diagnosis-planning", "Diagnosis planning")
    package_skill = _write_skill(tmp_path, "fta-review", "FTA review")
    package_skill.metadata["scope"] = "rebuttal_agent"
    pool = _FakePool([host_skill, package_skill])

    result = await load_skill(_ctx(pool, "librarian"), "fta-review")

    assert "Skill 'fta-review' not found" in result
    assert "diagnosis-planning" in result
    assert "fta-review instructions" not in result


@pytest.mark.unit
async def test_load_skill_for_node_uses_target_node_package_scope(tmp_path):
    host_skill = _write_skill(tmp_path, "diagnosis-planning", "Diagnosis planning")
    package_skill = _write_skill(tmp_path, "fta-review", "FTA review")
    package_skill.metadata["scope"] = "rebuttal_agent"
    pool = _FakePool([host_skill, package_skill])

    result = await load_skill_for_node(_ctx(pool, "engineer"), "fta-review", "rebuttal_agent")

    assert "# fta-review" in result
    assert "fta-review instructions" in result


@pytest.mark.unit
async def test_list_skills_filters_by_current_node_package_scope(tmp_path):
    host_skill = _write_skill(tmp_path, "diagnosis-planning", "Diagnosis planning")
    package_skill = _write_skill(tmp_path, "fta-review", "FTA review")
    package_skill.metadata["scope"] = "rebuttal_agent"
    pool = _FakePool([host_skill, package_skill])

    result = await list_skills(_ctx(pool, "librarian"))

    assert "diagnosis-planning" in result
    assert "fta-review" not in result


@pytest.mark.unit
async def test_list_skills_filters_provider_skills_by_current_node_package_scope(tmp_path):
    """Provider skills from SkillResource are always visible (no scope metadata).

    SkillEntry doesn't carry metadata, so scope-based filtering doesn't apply
    to provider skills — only local filesystem skills can be scoped.
    """
    host_skill = _write_skill(tmp_path, "diagnosis-planning", "Diagnosis planning")
    provider_skill = Skill(
        name="fta-review",
        description="FTA review",
        skill_path=PurePosixPath("skill://scratchpad/fta-review"),
    )
    pool = _FakePool([host_skill], [provider_skill])

    result = await list_skills(_ctx(pool, "librarian"))

    assert "diagnosis-planning" in result
    assert "fta-review" in result  # Provider skills are always visible


@pytest.mark.unit
async def test_hidden_package_skill_does_not_shadow_visible_provider_skill(tmp_path):
    package_skill = _write_skill(tmp_path, "fta-review", "FTA review")
    package_skill.metadata["scope"] = "rebuttal_agent"
    provider_skill = Skill(
        name="fta-review",
        description="Host provider skill",
        skill_path=PurePosixPath("skill://scratchpad/fta-review"),
        metadata={"scope": "host"},
    )
    pool = _FakePool([package_skill], [provider_skill])

    result = await load_skill(_ctx(pool, "librarian"), "fta-review")

    assert "fta-review provider instructions" in result
    assert "fta-review instructions" not in result
