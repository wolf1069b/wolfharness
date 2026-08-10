"""Tests for MCP-based skills integration.

This module tests that MCP-based skills are properly exposed through:
- GET /command endpoint (for OpenCode)
- load_skill tool
- list_skills tool


# TODO: L2 migration — test requires complex mock pool dependencies that
# cannot be easily replaced with a real pool. Needs investigation.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from unittest.mock import AsyncMock, MagicMock

import pytest

from wolfharness.capabilities.resource_protocols import SkillEntry, SkillResource
from wolfharness.skills.skill import Skill
from wolfharness_toolsets.builtin.skills import list_skills, load_skill


pytestmark = pytest.mark.integration


@pytest.fixture
def mock_agent_context():
    """Create a mock agent context with pool that has MCP-based skills."""
    ctx_pool = MagicMock()

    # Mock local skills (empty - simulating no local skills)
    ctx_pool.skills.list_skills.return_value = []
    ctx_pool.skills.get_skill_instructions.return_value = ""

    # Mock MCP-based skills as SkillEntry objects (returned by SkillResource.list_skills())
    mcp_skill_hyphen = Skill(
        name="systematic-troubleshooting",
        description="Systematic troubleshooting guide",
        skill_path=PurePosixPath("skill://mcp_provider/systematic-troubleshooting"),
        instructions="# Troubleshooting Guide\n\nFollow these steps...",
        metadata={"skill_type": "resource", "provider": "mcp_provider"},
    )
    mcp_skill_from_underscore = Skill(
        name="equipment-operation-assistant",
        description="Equipment operation assistant guide",
        skill_path=PurePosixPath("skill://mcp_provider/equipment-operation-assistant"),
        instructions="# Equipment Operation\n\nFollow these procedures...",
        metadata={"skill_type": "resource", "provider": "mcp_provider"},
    )

    # SkillEntry objects returned by SkillResource.list_skills()
    mcp_entries = [
        SkillEntry(
            name="systematic-troubleshooting",
            description="Systematic troubleshooting guide",
            uri="skill://mcp_provider/systematic-troubleshooting",
            source="remote",
        ),
        SkillEntry(
            name="equipment-operation-assistant",
            description="Equipment operation assistant guide",
            uri="skill://mcp_provider/equipment-operation-assistant",
            source="remote",
        ),
    ]

    # Mock skill_capabilities with a fake SkillManagerCap
    mock_child_provider = MagicMock()
    mock_child_provider.list_skills = AsyncMock(return_value=mcp_entries)
    mock_child_provider.read_skill = AsyncMock(
        return_value="# Troubleshooting Guide\n\nFollow these steps..."
    )
    mock_child_provider.skill_exists = AsyncMock(return_value=True)

    # Make isinstance(mock_child_provider, SkillResource) return True
    # and give it a real get_serialization_name used by SkillManagerCap.list_skills.
    class _FakeSkillResource(SkillResource):
        def get_serialization_name(self) -> str:
            return "mcp_provider"

        async def list_skills(self):
            return mcp_entries

        async def read_skill(self, skill_name: str) -> str | None:
            return "# Troubleshooting Guide\n\nFollow these steps..."

        async def skill_exists(self, skill_name: str) -> bool:
            return True

    fake_child = _FakeSkillResource()
    from wolfharness.capabilities.skill_manager_cap import SkillManagerCap

    mock_cap = SkillManagerCap(local_skills={}, children=[fake_child], name="pool-skills")
    ctx_pool.skill_capabilities = [mock_cap]

    # Mock skill_resolver
    mock_resolver = MagicMock()
    mock_resolver.list_providers.return_value = ["mcp_provider"]
    mock_provider_from_resolver = MagicMock()
    mock_provider_from_resolver.list_skills = AsyncMock(return_value=mcp_entries)
    mock_provider_from_resolver.read_skill = AsyncMock(
        return_value="# Troubleshooting Guide\n\nFollow these steps..."
    )
    mock_provider_from_resolver.skill_exists = AsyncMock(return_value=True)
    mock_resolver.get_provider.return_value = mock_provider_from_resolver

    # Mock resolve method to return appropriate skill based on name
    async def mock_resolve(uri: str):
        if "systematic-troubleshooting" in uri:
            return mcp_skill_hyphen
        if "equipment-operation-assistant" in uri:
            return mcp_skill_from_underscore
        raise ValueError(f"Skill not found: {uri}")

    mock_resolver.resolve = mock_resolve
    ctx_pool.skill_resolver = mock_resolver

    from types import SimpleNamespace

    from wolfharness.agents.context import AgentContext as RuntimeAgentContext

    ctx = RuntimeAgentContext(node=SimpleNamespace(name="test"), pool=ctx_pool)

    return ctx, mcp_skill_hyphen, mcp_skill_from_underscore


@pytest.mark.asyncio
async def test_list_skills_includes_mcp_skills(mock_agent_context):
    """Test that list_skills includes MCP-based skills."""
    ctx, _mcp_skill_hyphen, _mcp_skill_from_underscore = mock_agent_context

    result = await list_skills(ctx)

    # Should include MCP-based skills (all normalized to kebab-case)
    assert "systematic-troubleshooting" in result
    assert "equipment-operation-assistant" in result
    print(f"list_skills output:\n{result}")


@pytest.mark.asyncio
async def test_load_skill_finds_mcp_skills_with_hyphen(mock_agent_context):
    """Test that load_skill can find MCP-based skills with hyphen names."""
    ctx, _mcp_skill_hyphen, _ = mock_agent_context

    result = await load_skill(ctx, "systematic-troubleshooting")

    # Should successfully load the skill
    assert "systematic-troubleshooting" in result
    assert "Troubleshooting Guide" in result
    print(f"load_skill (hyphen) output:\n{result}")


@pytest.mark.asyncio
async def test_load_skill_normalizes_underscore_to_hyphen(mock_agent_context):
    """Test that load_skill normalizes underscore names to hyphens per spec."""
    ctx, _, _mcp_skill_from_underscore = mock_agent_context

    # Calling with underscores should work because normalization converts to hyphens
    result = await load_skill(ctx, "equipment_operation_assistant")

    # The skill name is normalized to kebab-case
    assert "equipment-operation-assistant" in result
    assert "Equipment operation assistant guide" in result
    print(f"load_skill (underscore normalized) output:\n{result}")


@pytest.mark.asyncio
async def test_load_skill_returns_error_for_missing_skill(mock_agent_context):
    """Test that load_skill returns error for non-existent skill."""
    ctx, _, _ = mock_agent_context

    result = await load_skill(ctx, "nonexistent-skill")

    # Should return error message
    assert "not found" in result.lower() or "No skills available" in result
    print(f"load_skill error output:\n{result}")


@pytest.mark.asyncio
async def test_list_skills_shows_empty_when_no_skills():
    """Test that list_skills shows 'No skills available' when pool has no skills."""
    ctx = MagicMock()
    ctx.pool = MagicMock()
    ctx.pool.skills.list_skills.return_value = []
    ctx.pool.skill_provider = None
    ctx.pool.skill_resolver = None

    result = await list_skills(ctx)

    assert "No skills available" in result


@pytest.mark.asyncio
async def test_load_skill_with_uri(mock_agent_context):
    """Test that load_skill works with skill:// URI."""
    ctx, mcp_skill_hyphen, _ = mock_agent_context

    # Mock the resolver to return the skill
    ctx.pool.skill_resolver.resolve = AsyncMock(return_value=mcp_skill_hyphen)

    result = await load_skill(ctx, "skill://mcp_provider/systematic-troubleshooting")

    # Should successfully load via URI
    assert "systematic-troubleshooting" in result
    print(f"load_skill with URI output:\n{result}")
