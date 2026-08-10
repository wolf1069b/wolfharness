"""Red flag test: skill reference loading via MCP scratchpad (HTTP) provider.

This test reproduces the bug where loading skill references via skill:// URIs
fails with "Provider 'systematic-troubleshooting' not registered" when the
provider name in the URI is actually the skill name (provider-less URI pattern).

Setup mimics ng's scratchpad connection (streamable-http MCP server):
- MCP provider name: "pool_mcp_scratchpad" (NOT "systematic-troubleshooting")
- Skill name: "systematic-troubleshooting"
- Reference path: "references/expert_knowledge/excavator/excavator-hard-starting.md"

Expected behavior (after fix):
- load_skill("systematic-troubleshooting") -> success (bare name search)
- load_skill("skill://systematic-troubleshooting/references/...") -> success
  (fallback: treat netloc as skill name, search all providers)

Bug behavior (before fix):
- load_skill("systematic-troubleshooting") -> success (bare name search)
- load_skill("skill://systematic-troubleshooting/references/...") -> fails with
  "Provider 'systematic-troubleshooting' not registered"


# TODO: L2 migration — test uses mock_pool as skill data container
# (skill_resolver, skills), not for SessionController/SessionPool creation.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wolfharness.skills.exceptions import SkillNotFoundError
from wolfharness.skills.skill import Skill
from wolfharness.skills.uri_resolver import SkillURIResolver
from wolfharness_toolsets.builtin.skills import load_skill


pytestmark = pytest.mark.integration


# =============================================================================
# Test helpers
# =============================================================================


class _FakeSkillResource:
    """Fake provider implementing SkillResource for testing URI resolution.

    Returns pre-built Skill objects, simulating a provider that has
    already loaded skills from MCP or another source.
    """

    def __init__(self, skills: list[Skill]) -> None:
        self._skills = skills

    async def list_skills(self):
        from wolfharness.capabilities.resource_protocols import SkillEntry

        return [
            SkillEntry(
                name=skill.name,
                description=skill.description,
                uri=f"skill://provider/{skill.name}",
                source="remote",
            )
            for skill in self._skills
        ]

    async def read_skill(self, name: str) -> str | None:
        for skill in self._skills:
            if skill.name == name:
                if skill.instructions:
                    return skill.load_instructions()
                return f"Instructions for {name}"
        return None

    async def skill_exists(self, name: str) -> bool:
        return any(skill.name == name for skill in self._skills)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_mcp_client_with_scratchpad_skills():
    """Create a mock MCPClient that simulates ng's scratchpad HTTP server.

    Returns skills with:
    - skill://systematic-troubleshooting/SKILL.md
    - skill://systematic-troubleshooting/_manifest
    """
    client = MagicMock()
    client.connected = True
    client.list_prompts = AsyncMock(return_value=[])

    # Simulate scratchpad's list_resources() output
    mock_resource_skillmd = MagicMock()
    mock_resource_skillmd.name = "systematic-troubleshooting-skillmd"
    mock_resource_skillmd.uri = "skill://systematic-troubleshooting/SKILL.md"
    mock_resource_skillmd.description = "Systematic troubleshooting skill"

    mock_resource_manifest = MagicMock()
    mock_resource_manifest.name = "systematic-troubleshooting-manifest"
    mock_resource_manifest.uri = "skill://systematic-troubleshooting/_manifest"
    mock_resource_manifest.description = "Skill manifest"

    client.list_resources = AsyncMock(return_value=[mock_resource_skillmd, mock_resource_manifest])

    # Simulate reading reference files
    async def mock_read_resource(uri: str):
        if "excavator-hard-starting.md" in uri:
            return ["# Excavator Hard Starting\n\nDiagnostic procedure for hard starting."]
        if "SKILL.md" in uri:
            return ["# Systematic Troubleshooting\n\nFollow the procedure."]
        if "_manifest" in uri:
            return [
                (
                    '{"name": "systematic-troubleshooting",'
                    ' "description": "Systematic troubleshooting"}'
                )
            ]
        return []

    client.read_resource = AsyncMock(side_effect=mock_read_resource)
    return client


@pytest.fixture
def scratchpad_provider(mock_mcp_client_with_scratchpad_skills):
    """Create a fake SkillResource provider that simulates ng's scratchpad connection.

    Provider name is "pool_mcp_scratchpad" (as registered by MCPManager),
    NOT "systematic-troubleshooting" (which is the skill name).

    Uses a fake implementing the SkillResource protocol directly, returning
    a pre-built Skill object. This tests the resolver's fallback behavior,
    not McpServerCap's skill loading from MCP resources.
    """
    skill = Skill(
        name="systematic-troubleshooting",
        description="Systematic troubleshooting",
        skill_path=PurePosixPath("skill://pool_mcp_scratchpad/systematic-troubleshooting"),
        metadata={"original_name": "systematic_troubleshooting"},
        instructions="Systematic troubleshooting instructions.",
    )
    return _FakeSkillResource([skill])


@pytest.fixture
def mock_agent_context_with_resolver(scratchpad_provider):
    """Create a mock AgentContext with a SkillURIResolver registered."""
    resolver = SkillURIResolver()
    # Register the scratchpad provider with its actual name
    resolver.register_provider("pool_mcp_scratchpad", scratchpad_provider)

    # Create mock pool
    mock_pool = MagicMock()
    mock_pool.skill_resolver = resolver
    mock_pool.skill_provider = None  # Not needed for URI resolution tests
    mock_pool.skills = MagicMock()
    mock_pool.skills.list_skills = MagicMock(return_value=[])

    # Create mock agent context
    mock_ctx = MagicMock(spec="AgentContext")
    mock_ctx.pool = mock_pool
    return mock_ctx


# =============================================================================
# RED FLAG TESTS
# =============================================================================


class TestScratchpadSkillReferenceLoading:
    """Red flag tests for skill reference loading via MCP scratchpad (HTTP)."""

    @pytest.mark.asyncio
    async def test_bare_skill_name_loads_successfully(
        self,
        scratchpad_provider: _FakeSkillResource,
    ) -> None:
        """Test 1: Bare skill name works (this already works in production)."""
        resolver = SkillURIResolver()
        resolver.register_provider("pool_mcp_scratchpad", scratchpad_provider)

        skill = await resolver.resolve("systematic-troubleshooting")

        assert skill is not None
        assert skill.name == "systematic-troubleshooting"

    @pytest.mark.asyncio
    async def test_uri_with_skill_name_as_netloc_loads_successfully(
        self,
        scratchpad_provider: _FakeSkillResource,
    ) -> None:
        """Test 2: URI with skill name as netloc should work via fallback.

        This is the RED FLAG test - it reproduces the bug.

        URI: skill://systematic-troubleshooting/references/expert_knowledge/...

        The netloc "systematic-troubleshooting" is NOT a registered provider
        (the actual provider is "pool_mcp_scratchpad"). The resolver should:
        1. Detect that "systematic-troubleshooting" is not a provider
        2. Fall back to treating it as a skill name
        3. Search all providers for a skill with that name
        4. Return the skill with reference path attached

        Bug (before fix): Raises ValueError("Provider 'systematic-troubleshooting' not registered")
        Expected (after fix): Returns the skill with reference_path set
        """
        resolver = SkillURIResolver()
        resolver.register_provider("pool_mcp_scratchpad", scratchpad_provider)

        uri = "skill://systematic-troubleshooting/references/expert_knowledge/excavator/excavator-hard-starting.md"

        # This should NOT raise ValueError
        skill = await resolver.resolve(uri)

        # Verify skill was found (resolver didn't raise ValueError)
        assert skill is not None
        assert skill.name == "systematic-troubleshooting"

    @pytest.mark.skip(reason="L2 migration: requires mock internals — remains L1 unit test")
    @pytest.mark.asyncio
    async def test_load_skill_tool_with_uri_and_reference(
        self,
        mock_agent_context_with_resolver,
    ) -> None:
        """Test 3: Full load_skill tool call with URI + reference path.

        This tests the complete chain:
        1. load_skill(ctx, "skill://systematic-troubleshooting/references/...")
        2. -> ResolvedSkillURI.parse()
        3. -> resolver.resolve()
        4. -> Should return skill instructions + reference content

        Note: This test mocks the reference content loading since we're not
        testing the MCP read_resource chain here, just the URI resolution.
        """
        ctx = mock_agent_context_with_resolver

        # We need to mock _load_reference_content to avoid actual MCP calls
        with patch("wolfharness_toolsets.builtin.skills._load_reference_content") as mock_load_ref:
            mock_load_ref.return_value = (
                "\n\n## Reference: expert_knowledge/excavator/excavator-hard-starting.md\n\n"
                "# Excavator Hard Starting\n\nDiagnostic procedure."
            )

            result = await load_skill(
                ctx,
                "skill://systematic-troubleshooting/references/expert_knowledge/excavator/excavator-hard-starting.md",
            )

        # Should NOT contain error messages
        assert "Failed to resolve" not in result
        assert "not registered" not in result

        # Should contain the reference content
        assert "Excavator Hard Starting" in result or "Reference:" in result

    @pytest.mark.asyncio
    async def test_uri_resolver_fallback_searches_all_providers(
        self,
    ) -> None:
        """Test 4: Fallback searches ALL providers, not just the first one.

        Setup:
        - Provider "local" has skill "local-skill"
        - Provider "pool_mcp_scratchpad" has skill "systematic-troubleshooting"

        URI: skill://systematic-troubleshooting/references/file.md

        The fallback should search provider "local" first (no match),
        then search provider "pool_mcp_scratchpad" (match found).
        """
        resolver = SkillURIResolver()

        # Provider 1: local filesystem skills
        local_skill = Skill(
            name="local-skill",
            description="Local skill",
            skill_path=PurePosixPath("/tmp/local-skill"),
            instructions="Local skill instructions.",
        )
        local_provider = _FakeSkillResource([local_skill])

        # Provider 2: scratchpad (HTTP MCP server)
        scratchpad_skill = Skill(
            name="systematic-troubleshooting",
            description="Systematic troubleshooting",
            skill_path=PurePosixPath("skill://pool_mcp_scratchpad/systematic-troubleshooting"),
            metadata={"original_name": "systematic_troubleshooting"},
            instructions="Systematic troubleshooting instructions.",
        )
        scratchpad_provider = _FakeSkillResource([scratchpad_skill])

        resolver.register_provider("local", local_provider)
        resolver.register_provider("pool_mcp_scratchpad", scratchpad_provider)

        uri = "skill://systematic-troubleshooting/references/guide.md"
        skill = await resolver.resolve(uri)

        assert skill is not None
        assert skill.name == "systematic-troubleshooting"

    @pytest.mark.asyncio
    async def test_uri_resolver_fallback_with_name_alternatives(
        self,
    ) -> None:
        """Test 5: Fallback tries name alternatives (-/_ swap).

        Setup:
        - Provider has skill with name "systematic_troubleshooting" (underscored)
        - URI netloc is "systematic-troubleshooting" (kebab-case)

        The fallback should try both "systematic-troubleshooting" and
        "systematic_troubleshooting" when searching providers.
        """
        resolver = SkillURIResolver()

        # Skill with underscored name (as stored in MCP server metadata)
        underscore_skill = Skill(
            name="systematic_troubleshooting",  # Will be normalized to kebab-case by validator
            description="Systematic troubleshooting",
            skill_path=PurePosixPath("skill://pool_mcp_scratchpad/systematic_troubleshooting"),
            metadata={"original_name": "systematic_troubleshooting"},
            instructions="Systematic troubleshooting instructions.",
        )

        provider = _FakeSkillResource([underscore_skill])
        resolver.register_provider("pool_mcp_scratchpad", provider)

        # URI uses kebab-case
        uri = "skill://systematic-troubleshooting/references/guide.md"
        skill = await resolver.resolve(uri)

        assert skill is not None
        assert skill.name == "systematic-troubleshooting"  # Normalized to kebab-case

    @pytest.mark.asyncio
    async def test_provider_not_registered_raises_when_no_fallback_match(
        self,
    ) -> None:
        """Test 6: When fallback search finds nothing, raise appropriate error.

        This ensures we don't accidentally swallow real errors.
        """
        resolver = SkillURIResolver()

        # Provider has a completely different skill
        other_skill = Skill(
            name="other-skill",
            description="Other skill",
            skill_path=PurePosixPath("skill://pool_mcp_scratchpad/other-skill"),
            instructions="Other instructions.",
        )
        provider = _FakeSkillResource([other_skill])
        resolver.register_provider("pool_mcp_scratchpad", provider)

        uri = "skill://nonexistent-skill/references/guide.md"

        with pytest.raises(SkillNotFoundError, match="not found"):
            await resolver.resolve(uri)
