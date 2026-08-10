"""Tests for flat skill:// URI format (no provider segment).

Verifies that ``skill://{name}`` URIs (without ``/local/`` provider segment)
resolve correctly through:
1. ``ResolvedSkillURI.parse()`` — parsing extracts skill_name from netloc.
2. ``ExtensionRegistry.resolve_uri()`` — flat URI routes to all SkillResource caps.
3. ``SkillURIResolver.resolve()`` — end-to-end resolution via ExtensionRegistry.
4. ``SkillManagerCap.list_skills()`` — returns URIs in ``skill://{name}`` format.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Self

import pytest

from wolfharness.capabilities.extension_registry import (
    ExtensionRegistry,
    Scope,
    ScopeLevel,
)
from wolfharness.capabilities.resource_protocols import SkillEntry, SkillResource
from wolfharness.capabilities.skill_manager_cap import SkillManagerCap
from wolfharness.skills.skill import Skill
from wolfharness.skills.uri_resolver import ResolvedSkillURI, SkillURIResolver


pytestmark = pytest.mark.unit


# ---- Test helpers ----


class FakeSkillResource(SkillResource):
    """Minimal SkillResource implementation for testing."""

    def __init__(
        self,
        name: str = "skill_cap",
        skills: dict[str, str] | None = None,
    ) -> None:
        self._name = name
        self._skills = skills or {}

    def get_serialization_name(self) -> str:
        return self._name

    async def list_skills(self) -> list[SkillEntry]:
        return [
            SkillEntry(
                name=n,
                description=f"Description for {n}",
                uri=f"skill://{n}",
                source="local",
            )
            for n in self._skills
        ]

    async def read_skill(self, name: str) -> str | None:
        return self._skills.get(name)

    async def skill_exists(self, name: str) -> bool:
        return name in self._skills

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    def get_toolset(self) -> Any:
        return None

    def get_instructions(self) -> str | None:
        return None


# ---- 17.5: ResolvedSkillURI.parse() handles flat URIs ----


def test_parse_flat_uri_no_provider() -> None:
    """Parse flat URI without provider segment.

    Given skill://my-skill, When parsed, Then skill_name='my-skill', provider=None.
    """
    result = ResolvedSkillURI.parse("skill://my-skill")
    assert result.skill_name == "my-skill"
    assert result.reference_path is None


def test_parse_flat_uri_with_hyphenated_name() -> None:
    """Parse flat URI with hyphenated skill name.

    Given skill://python-expert, When parsed, Then skill_name='python-expert'.
    """
    result = ResolvedSkillURI.parse("skill://python-expert")
    assert result.skill_name == "python-expert"


def test_parse_flat_uri_with_reference_path() -> None:
    """Parse flat URI with reference path.

    Given skill://my-skill/references/guide.md, When parsed, Then
    reference_path extracted correctly.
    """
    result = ResolvedSkillURI.parse("skill://my-skill/references/guide.md")
    assert result.skill_name == "my-skill"
    assert result.reference_path == "references/guide.md"


# ---- 17.5: ExtensionRegistry.resolve_uri() with flat URIs ----


@pytest.mark.asyncio
async def test_flat_uri_resolves_via_extension_registry() -> None:
    """Flat URI resolves through ExtensionRegistry.

    Given a SkillResource registered in ExtensionRegistry, When
    resolve_uri('skill://my-skill') is called, Then the skill content is
    returned.
    """
    reg = ExtensionRegistry()
    skill_cap = FakeSkillResource(skills={"my-skill": "skill content here"})
    reg.register(skill_cap, Scope(level=ScopeLevel.POOL))

    result = await reg.resolve_uri("skill://my-skill", Scope(level=ScopeLevel.POOL))
    assert result is not None
    assert isinstance(result, Skill)
    assert result.name == "my-skill"
    assert result.instructions == "skill content here"


@pytest.mark.asyncio
async def test_flat_uri_nonexistent_skill_returns_none() -> None:
    """Nonexistent flat URI returns None.

    Given a SkillResource without the requested skill, When
    resolve_uri('skill://nonexistent') is called, Then None is returned.
    """
    reg = ExtensionRegistry()
    skill_cap = FakeSkillResource(skills={"existing": "content"})
    reg.register(skill_cap, Scope(level=ScopeLevel.POOL))

    result = await reg.resolve_uri("skill://nonexistent", Scope(level=ScopeLevel.POOL))
    assert result is None


# ---- 17.6: SkillURIResolver.resolve() with flat URIs ----


@pytest.mark.asyncio
async def test_flat_uri_resolves_via_skill_uri_resolver() -> None:
    """Flat URI resolves via SkillURIResolver.

    Given a SkillURIResolver with ExtensionRegistry, When
    resolve('skill://my-skill') is called, Then a Skill instance with
    correct name and instructions is returned.
    """
    reg = ExtensionRegistry()
    skill_cap = FakeSkillResource(skills={"my-skill": "# My Skill\nInstructions here."})
    reg.register(skill_cap, Scope(level=ScopeLevel.POOL))

    resolver = SkillURIResolver(extension_registry=reg)
    skill = await resolver.resolve("skill://my-skill")

    assert skill.name == "my-skill"
    assert skill.instructions == "# My Skill\nInstructions here."


@pytest.mark.asyncio
async def test_flat_uri_resolver_raises_for_missing_skill() -> None:
    """Missing flat URI raises SkillNotFoundError.

    Given a SkillURIResolver with ExtensionRegistry, When
    resolve('skill://nonexistent') is called, Then SkillNotFoundError is
    raised.
    """
    from wolfharness.skills.exceptions import SkillNotFoundError

    reg = ExtensionRegistry()
    skill_cap = FakeSkillResource(skills={"existing": "content"})
    reg.register(skill_cap, Scope(level=ScopeLevel.POOL))

    resolver = SkillURIResolver(extension_registry=reg)
    with pytest.raises(SkillNotFoundError):
        await resolver.resolve("skill://nonexistent")


# ---- 17.7: list_skills() returns flat URI format ----


@pytest.mark.asyncio
async def test_list_skills_returns_flat_uri_format() -> None:
    """list_skills() returns flat URI format.

    Given a SkillManagerCap with local skills, When list_skills() is called,
    Then each entry's URI is in 'skill://{name}' format (no '/local/'
    segment).
    """
    skill = Skill(
        name="my-test-skill",
        description="A test skill",
        skill_path=PurePosixPath("/tmp/my-test-skill"),
        instructions="Test instructions",
    )
    cap = SkillManagerCap(local_skills={"my-test-skill": skill})

    entries = await cap.list_skills()

    assert len(entries) == 1
    assert entries[0].uri == "skill://my-test-skill"
    assert "/local/" not in entries[0].uri


@pytest.mark.asyncio
async def test_list_skills_flat_uri_multiple_skills() -> None:
    """list_skills() returns flat URIs for multiple skills.

    Given a SkillManagerCap with multiple local skills, When list_skills()
    is called, Then all URIs use flat format without '/local/'.
    """
    skills = {
        "alpha": Skill(
            name="alpha",
            description="Alpha skill",
            skill_path=PurePosixPath("/tmp/alpha"),
            instructions="Alpha",
        ),
        "beta": Skill(
            name="beta",
            description="Beta skill",
            skill_path=PurePosixPath("/tmp/beta"),
            instructions="Beta",
        ),
    }
    cap = SkillManagerCap(local_skills=skills)

    entries = await cap.list_skills()

    assert len(entries) == 2
    uris = {e.uri for e in entries}
    assert uris == {"skill://alpha", "skill://beta"}
    for entry in entries:
        assert "/local/" not in entry.uri


@pytest.mark.asyncio
async def test_list_commands_returns_flat_uri_format() -> None:
    """list_commands() returns flat URI format.

    Given a SkillManagerCap with user-invocable skills, When list_commands()
    is called, Then each command's skill_uri is in 'skill://{name}' format.
    """
    skill = Skill(
        name="my-command-skill",
        description="A command skill",
        skill_path=PurePosixPath("/tmp/my-command-skill"),
        instructions="Command instructions",
    )
    # Ensure user_invocable is True (default)
    assert skill.user_invocable is True

    cap = SkillManagerCap(local_skills={"my-command-skill": skill})

    commands = await cap.list_commands()

    assert len(commands) == 1
    assert commands[0].skill_uri == "skill://my-command-skill"
    assert "/local/" not in commands[0].skill_uri
