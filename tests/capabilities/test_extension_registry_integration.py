"""Integration tests for ExtensionRegistry — session-level scoping.

Tests:
- Session 1 has skill A, session 2 has skill B, neither sees the other's skills.
"""

from __future__ import annotations

import pytest

from wolfharness.capabilities.extension_registry import (
    ExtensionRegistry,
    Scope,
    ScopeLevel,
)


pytestmark = pytest.mark.integration


class FakeSkillCap:
    """Fake SkillResource capability."""

    def __init__(self, name: str, skills: dict[str, str]) -> None:
        self._name = name
        self._skills = skills

    @property
    def name(self) -> str:
        return self._name

    async def list_skills(self) -> list:
        from wolfharness.capabilities.resource_protocols import SkillEntry

        return [
            SkillEntry(name=n, description=d, uri=f"skill://{n}") for n, d in self._skills.items()
        ]

    async def read_skill(self, name: str) -> str | None:
        return self._skills.get(name)

    async def skill_exists(self, name: str) -> bool:
        return name in self._skills


class TestSessionLevelSkillScoping:
    """Test session-level skill scoping (task 4.37)."""

    @pytest.mark.asyncio
    async def test_session_isolation(self) -> None:
        """Session 1 has skill A, session 2 has skill B."""
        reg = ExtensionRegistry()

        skill_a = FakeSkillCap("skill-a", {"skill-a": "Content A"})
        skill_b = FakeSkillCap("skill-b", {"skill-b": "Content B"})

        reg.register(
            skill_a,
            Scope(level=ScopeLevel.SESSION, session_id="ses1"),
        )
        reg.register(
            skill_b,
            Scope(level=ScopeLevel.SESSION, session_id="ses2"),
        )

        ses1_skills = reg.get_skill_resources(Scope(level=ScopeLevel.SESSION, session_id="ses1"))
        ses2_skills = reg.get_skill_resources(Scope(level=ScopeLevel.SESSION, session_id="ses2"))

        assert skill_a in ses1_skills
        assert skill_b not in ses1_skills

        assert skill_b in ses2_skills
        assert skill_a not in ses2_skills

    @pytest.mark.asyncio
    async def test_pool_visible_to_all_sessions(self) -> None:
        """Pool-level skills visible to all sessions."""
        reg = ExtensionRegistry()

        pool_skill = FakeSkillCap("pool-skill", {"pool-skill": "Pool content"})
        reg.register(pool_skill, Scope(level=ScopeLevel.POOL))

        ses1_skills = reg.get_skill_resources(Scope(level=ScopeLevel.SESSION, session_id="ses1"))
        ses2_skills = reg.get_skill_resources(Scope(level=ScopeLevel.SESSION, session_id="ses2"))

        assert pool_skill in ses1_skills
        assert pool_skill in ses2_skills

    @pytest.mark.asyncio
    async def test_resolve_uri_session_scoped(self) -> None:
        """URI resolution respects session scope."""
        reg = ExtensionRegistry()

        skill_a = FakeSkillCap("skill-a", {"my-skill": "Session 1 content"})
        skill_b = FakeSkillCap("skill-b", {"my-skill": "Session 2 content"})

        reg.register(
            skill_a,
            Scope(level=ScopeLevel.SESSION, session_id="ses1"),
        )
        reg.register(
            skill_b,
            Scope(level=ScopeLevel.SESSION, session_id="ses2"),
        )

        result1 = await reg.resolve_uri(
            "skill://my-skill",
            Scope(level=ScopeLevel.SESSION, session_id="ses1"),
        )
        result2 = await reg.resolve_uri(
            "skill://my-skill",
            Scope(level=ScopeLevel.SESSION, session_id="ses2"),
        )

        from wolfharness.skills.skill import Skill

        assert isinstance(result1, Skill)
        assert result1.instructions == "Session 1 content"
        assert isinstance(result2, Skill)
        assert result2.instructions == "Session 2 content"
