from __future__ import annotations

import asyncio

import pytest

from wolfharness import BaseTeam
from wolfharness.skills.exceptions import SkillNotFoundError


pytestmark = pytest.mark.unit


class _Pool:
    skill_resolver = object()

    async def get_skill_instructions_for_node(self, skill_name: str, node_name: str) -> str:
        return f"# {skill_name}\nUse this skill for {node_name}."


def test_team_loads_member_skills_from_pool_provider() -> None:
    team = BaseTeam([], mode="parallel", name="review_team")
    team.agent_pool = _Pool()

    result = asyncio.run(
        team._load_member_skill_instructions({
            "root_cause_reviewer": ["fta-causal-path-review"],
        })
    )

    assert '<skill-instruction name="fta-causal-path-review">' in result["root_cause_reviewer"]
    assert "Use this skill for root_cause_reviewer." in result["root_cause_reviewer"]


def test_team_requires_pool_skill_resolver_for_member_skills() -> None:
    team = BaseTeam([], mode="parallel", name="review_team")
    team.agent_pool = None

    with pytest.raises(SkillNotFoundError):
        asyncio.run(team._load_skill_instructions("fta-causal-path-review", "root_cause_reviewer"))


def test_team_injects_member_skills_into_member_prompt() -> None:
    prompt = BaseTeam._inject_member_skill_instructions(
        "root_cause_reviewer",
        ["Review the FTA."],
        {"root_cause_reviewer": "<skill-instruction>Skill body</skill-instruction>"},
    )

    assert prompt == ["<skill-instruction>Skill body</skill-instruction>\n\nReview the FTA."]
