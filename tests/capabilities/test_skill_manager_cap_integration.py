"""Integration tests for SkillManagerCap with embedded MCP server.

Tests cover:
  1. Skill with embedded MCP server — tools from MCP available alongside skill instructions
  2. MCP child lifecycle (enter/exit)
  3. Partial failure (MCP server fails, skill still works)
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Self
from unittest.mock import MagicMock

from pydantic_ai.capabilities import AbstractCapability
import pytest

from wolfharness.capabilities.resource_protocols import (
    SkillEntry,
    SkillResource,
)
from wolfharness.capabilities.skill_manager_cap import SkillManagerCap
from wolfharness.skills.skill import Skill


pytestmark = pytest.mark.unit


# ---- Mock MCP server capability ----


class MockMcpServerCap(SkillResource, AbstractCapability[Any]):
    """Mock McpServerCap that implements SkillResource for integration testing."""

    def __init__(
        self,
        name: str = "mock-mcp",
        skills: list[SkillEntry] | None = None,
        content_map: dict[str, str] | None = None,
        fail_on_connect: bool = False,
    ) -> None:
        self.name = name
        self._skills = skills or []
        self._content_map = content_map or {}
        self._fail = fail_on_connect
        self._entered = False

    async def for_run(self, ctx: Any) -> MockMcpServerCap:
        """Return a fresh per-run copy (mock)."""
        return MockMcpServerCap(
            name=self.name,
            skills=list(self._skills),
            content_map=dict(self._content_map),
            fail_on_connect=self._fail,
        )

    def get_serialization_name(self) -> str:
        return self.name

    async def list_skills(self) -> Any:
        if self._fail:
            raise RuntimeError("MCP server connection failed")
        return list(self._skills)

    async def read_skill(self, name: str) -> str | None:
        if self._fail:
            raise RuntimeError("MCP server connection failed")
        return self._content_map.get(name)

    async def skill_exists(self, name: str) -> bool:
        if self._fail:
            raise RuntimeError("MCP server connection failed")
        return name in self._content_map

    async def __aenter__(self) -> Self:
        self._entered = True
        return self

    async def __aexit__(self, *args: object) -> None:
        self._entered = False

    def get_toolset(self) -> Any:
        return None

    def get_instructions(self) -> str | None:
        return None


# ---- Test 1: Skill with embedded MCP server ----


async def test_skill_with_mcp_instructions_and_remote_skills() -> None:
    """Skill instructions available alongside MCP-provided remote skills."""
    local_skill = Skill(
        name="my-skill",
        description="A local skill",
        skill_path=PurePosixPath("skill://local/my-skill"),
        instructions="Use this skill for awesome things.",
    )

    remote_skills = [
        SkillEntry(
            name="mcp-skill",
            description="Skill from MCP server",
            uri="skill://mock-mcp/mcp-skill",
            source="remote",
        ),
    ]
    mcp_child = MockMcpServerCap(
        name="mock-mcp",
        skills=remote_skills,
        content_map={"mcp-skill": "Remote skill content"},
    )

    cap = SkillManagerCap(
        local_skills={"my-skill": local_skill},
        children=[mcp_child],
    )

    # get_instructions returns [metadata_str, dynamic_callable]
    instructions = cap.get_instructions()
    assert instructions is not None
    assert isinstance(instructions, list)
    # First element is static metadata string
    metadata = instructions[0]
    assert isinstance(metadata, str)
    assert 'name="my-skill"' in metadata
    # Second element is the dynamic callable
    assert callable(instructions[1])

    # list_skills returns both local and remote
    all_skills = await cap.list_skills()
    names = [s.name for s in all_skills]
    assert "my-skill" in names
    assert "mcp-skill" in names

    # read_skill works for both
    local_content = await cap.read_skill("my-skill")
    assert local_content == "Use this skill for awesome things."

    remote_content = await cap.read_skill("mcp-skill")
    assert remote_content == "Remote skill content"


# ---- Test 2: MCP child lifecycle (enter/exit) ----


async def test_mcp_child_lifecycle_enter_exit() -> None:
    """MCP child is entered on __aenter__ and exited on __aexit__."""
    mcp_child = MockMcpServerCap(name="lifecycle-mcp")
    cap = SkillManagerCap(
        local_skills={},
        children=[mcp_child],
    )

    assert not mcp_child._entered
    await cap.__aenter__()
    assert mcp_child._entered
    await cap.__aexit__(None, None, None)
    assert not mcp_child._entered


# ---- Test 3: Partial failure (MCP server fails, skill still works) ----


async def test_partial_failure_mcp_fails_skill_still_works() -> None:
    """When MCP server fails, local skills still work."""
    local_skill = Skill(
        name="resilient-skill",
        description="Works even when MCP fails",
        skill_path=PurePosixPath("skill://local/resilient-skill"),
        instructions="Local instructions still available.",
    )

    failing_mcp = MockMcpServerCap(
        name="failing-mcp",
        fail_on_connect=True,
    )

    cap = SkillManagerCap(
        local_skills={"resilient-skill": local_skill},
        children=[failing_mcp],
    )

    # Local skill instructions still available
    instructions = cap.get_instructions()
    assert instructions is not None
    assert isinstance(instructions, list)
    metadata = instructions[0]
    assert isinstance(metadata, str)
    assert 'name="resilient-skill"' in metadata
    assert callable(instructions[1])

    # list_skills doesn't crash — returns local only
    skills = await cap.list_skills()
    names = [s.name for s in skills]
    assert "resilient-skill" in names

    # read_skill for local skill works
    content = await cap.read_skill("resilient-skill")
    assert content == "Local instructions still available."

    # skill_exists for local skill works
    assert await cap.skill_exists("resilient-skill")


# ---- Test 4: get_instructions returns [metadata, callable] with dynamic content ----


async def test_get_instructions_dynamic_callable_produces_skill_content() -> None:
    """get_instructions returns [metadata, callable]; all mode produces skill content."""
    local_skill = Skill(
        name="injected-skill",
        description="Skill to inject",
        skill_path=PurePosixPath("skill://local/injected-skill"),
        instructions="Injected instructions content.",
    )

    mcp_child = MockMcpServerCap(name="present-mcp")
    cap = SkillManagerCap(
        local_skills={"injected-skill": local_skill},
        children=[mcp_child],
        inject_mode="all",
    )

    result = cap.get_instructions()
    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 2

    # Static metadata
    metadata = result[0]
    assert isinstance(metadata, str)
    assert "<available-skills>" in metadata
    assert 'name="injected-skill"' in metadata

    # Dynamic callable — no matcher, so all skills are injected (backward compat)
    dynamic_fn = result[1]
    assert callable(dynamic_fn)

    # Build a mock RunContext with messages
    ctx = MagicMock()
    ctx.messages = []

    content = await dynamic_fn(ctx)
    assert content is not None
    assert "Injected instructions content." in content
    assert '<skill_content name="injected-skill">' in content


async def test_get_instructions_with_matcher_fn() -> None:
    """get_instructions callable respects matcher_fn for skill selection."""
    skill_a = Skill(
        name="skill-a",
        description="Skill A",
        skill_path=PurePosixPath("skill://local/skill-a"),
        instructions="Content A.",
    )
    skill_b = Skill(
        name="skill-b",
        description="Skill B",
        skill_path=PurePosixPath("skill://local/skill-b"),
        instructions="Content B.",
    )

    def matcher(messages: list[object]) -> list[str]:
        return ["skill-a"]  # Only match skill-a

    cap = SkillManagerCap(
        local_skills={"skill-a": skill_a, "skill-b": skill_b},
        matcher_fn=matcher,
        inject_mode="matcher",
    )

    result = cap.get_instructions()
    assert result is not None
    assert isinstance(result, list)

    ctx = MagicMock()
    ctx.messages = []

    content = await result[1](ctx)
    assert content is not None
    assert "Content A." in content
    assert "Content B." not in content


async def test_get_instructions_with_always_active() -> None:
    """get_instructions callable includes always_active skills even with matcher."""
    skill_a = Skill(
        name="skill-a",
        description="Skill A",
        skill_path=PurePosixPath("skill://local/skill-a"),
        instructions="Content A.",
    )
    skill_b = Skill(
        name="skill-b",
        description="Skill B",
        skill_path=PurePosixPath("skill://local/skill-b"),
        instructions="Content B.",
    )

    def matcher(messages: list[object]) -> list[str]:
        return ["skill-a"]

    cap = SkillManagerCap(
        local_skills={"skill-a": skill_a, "skill-b": skill_b},
        matcher_fn=matcher,
        always_active={"skill-b"},
        inject_mode="matcher",
    )

    result = cap.get_instructions()
    assert result is not None

    ctx = MagicMock()
    ctx.messages = []

    content = await result[1](ctx)
    assert content is not None
    assert "Content A." in content
    assert "Content B." in content  # always_active bypasses matcher


# ---- Test 5: inject_mode=description (default) emits catalog only ----


async def test_inject_mode_description_emits_catalog_only() -> None:
    """inject_mode='description' emits <available-skills> but no <skill_content>."""
    local_skill = Skill(
        name="desc-skill",
        description="A description-mode skill",
        skill_path=PurePosixPath("skill://local/desc-skill"),
        instructions="Full instructions that should NOT be injected.",
    )

    cap = SkillManagerCap(
        local_skills={"desc-skill": local_skill},
        inject_mode="description",  # explicit default
    )

    result = cap.get_instructions()
    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 2

    # Static metadata
    metadata = result[0]
    assert isinstance(metadata, str)
    assert "<available-skills>" in metadata
    assert 'name="desc-skill"' in metadata

    # Dynamic callable — description mode returns None (no <skill_content>)
    ctx = MagicMock()
    ctx.messages = []
    content = await result[1](ctx)
    assert content is None


async def test_inject_mode_description_is_default() -> None:
    """The default inject_mode is 'description' (catalog only)."""
    local_skill = Skill(
        name="default-skill",
        description="Default mode skill",
        skill_path=PurePosixPath("skill://local/default-skill"),
        instructions="Instructions.",
    )

    cap = SkillManagerCap(local_skills={"default-skill": local_skill})
    assert cap._inject_mode == "description"

    ctx = MagicMock()
    ctx.messages = []
    result = cap.get_instructions()
    assert result is not None
    content = await result[1](ctx)
    assert content is None


# ---- Test 6: inject_mode=matcher without matcher_fn warns and falls back ----


async def test_inject_mode_matcher_without_fn_falls_back_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """inject_mode='matcher' with no matcher_fn falls back to description + warning."""
    local_skill = Skill(
        name="no-matcher-skill",
        description="Skill without matcher",
        skill_path=PurePosixPath("skill://local/no-matcher-skill"),
        instructions="Should not be injected.",
    )

    cap = SkillManagerCap(
        local_skills={"no-matcher-skill": local_skill},
        inject_mode="matcher",
        # No matcher_fn provided
    )

    result = cap.get_instructions()
    assert result is not None

    ctx = MagicMock()
    ctx.messages = []
    content = await result[1](ctx)
    assert content is None  # Fell back to description mode


# ---- Test 7: inject_mode=all injects every skill ----


async def test_inject_mode_all_injects_every_skill() -> None:
    """inject_mode='all' injects full instructions for every local skill."""
    skill_a = Skill(
        name="all-a",
        description="All mode A",
        skill_path=PurePosixPath("skill://local/all-a"),
        instructions="Content A.",
    )
    skill_b = Skill(
        name="all-b",
        description="All mode B",
        skill_path=PurePosixPath("skill://local/all-b"),
        instructions="Content B.",
    )

    cap = SkillManagerCap(
        local_skills={"all-a": skill_a, "all-b": skill_b},
        inject_mode="all",
    )

    result = cap.get_instructions()
    assert result is not None

    ctx = MagicMock()
    ctx.messages = []
    content = await result[1](ctx)
    assert content is not None
    assert "Content A." in content
    assert "Content B." in content


# ---- Test 8: get_toolset exposes load_skill and list_skills ----


async def test_get_toolset_exposes_load_and_list_skills() -> None:
    """SkillManagerCap.get_toolset() exposes load_skill and list_skills tools."""
    from pydantic_ai.toolsets import AbstractToolset

    local_skill = Skill(
        name="tool-skill",
        description="A skill for tool testing",
        skill_path=PurePosixPath("skill://local/tool-skill"),
        instructions="Test instructions.",
    )

    cap = SkillManagerCap(local_skills={"tool-skill": local_skill})
    toolset = cap.get_toolset()
    assert toolset is not None
    assert isinstance(toolset, AbstractToolset)


# ---- Test 9: safe_uri returns flat skill://{name} ----


async def test_safe_uri_flat_form() -> None:
    """Skill.safe_uri returns flat skill://{name} for local skills."""
    from upathtools import UPath

    from wolfharness.skills.skill import Skill

    # Local filesystem skill
    local_skill = Skill(
        name="ponytail",
        description="A local skill",
        skill_path=UPath("/tmp/skills/ponytail"),
        instructions="Instructions.",
    )
    assert local_skill.safe_uri == "skill://ponytail"
    assert local_skill.safe_uri != "skill://local/ponytail"


# ---- Test 10: remove_child works for unregister ----


async def test_remove_child_removes_from_children() -> None:
    """SkillManagerCap.remove_child() removes a child capability."""
    mcp_child = MockMcpServerCap(name="removable-mcp")
    cap = SkillManagerCap(
        local_skills={},
        children=[mcp_child],
    )

    assert len(cap.children) == 1
    assert cap.remove_child(mcp_child)
    assert len(cap.children) == 0
    assert not cap.remove_child(mcp_child)  # already removed


# ---- Test 11: remote skills not auto-injected ----


async def test_remote_skills_not_auto_injected() -> None:
    """Remote skills are NOT auto-injected into <available-skills> or <skill_content>."""
    local_skill = Skill(
        name="local-only",
        description="A local skill",
        skill_path=PurePosixPath("skill://local/local-only"),
        instructions="Local instructions.",
    )

    remote_skills = [
        SkillEntry(
            name="remote-skill",
            description="Remote skill from MCP",
            uri="skill://remote-skill",
            source="remote",
        ),
    ]
    mcp_child = MockMcpServerCap(
        name="remote-mcp",
        skills=remote_skills,
        content_map={"remote-skill": "Remote content"},
    )

    cap = SkillManagerCap(
        local_skills={"local-only": local_skill},
        children=[mcp_child],
        inject_mode="all",
    )

    result = cap.get_instructions()
    assert result is not None

    # Static metadata should only include local skills
    metadata = result[0]
    assert isinstance(metadata, str)
    assert 'name="local-only"' in metadata
    assert "remote-skill" not in metadata

    # Dynamic content should only include local skills
    ctx = MagicMock()
    ctx.messages = []
    content = await result[1](ctx)
    assert content is not None
    assert "Local instructions." in content
    assert "Remote content" not in content
