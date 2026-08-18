"""Integration tests for AgentPool skill integration.

Tests cover skill_resolver property, skill_capabilities property,
skill resolution through pool, and provider registration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import pytest
from upathtools import UPath

from wolfharness import AgentPool, AgentsManifest, NativeAgentConfig
from wolfharness.capabilities.resource_protocols import SkillResource
from wolfharness.capabilities.skill_manager_cap import SkillManagerCap
from wolfharness.skills.uri_resolver import SkillURIResolver
from wolfharness_config.skills import SkillsConfig


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from pathlib import Path


# =============================================================================
# Test helpers
# =============================================================================


class _FakeSkillResourceProvider(SkillResource):
    """Fake provider implementing SkillResource for testing registration."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def list_skills(self):
        return []

    async def read_skill(self, name: str) -> str | None:
        return None

    async def skill_exists(self, name: str) -> bool:
        return False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_skill(tmp_path: Path) -> UPath:
    """Create a test skill directory."""
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()

    content = """---
name: test-skill
description: A test skill for pool integration
---

# Test Skill

This is a test skill.
"""
    skills_md = skill_dir / "SKILL.md"
    skills_md.write_text(content)

    return UPath(skill_dir)


@pytest.fixture
def another_skill(tmp_path: Path) -> UPath:
    """Create another test skill directory."""
    skill_dir = tmp_path / "another-skill"
    skill_dir.mkdir()

    content = """---
name: another-skill
description: Another test skill
---

# Another Skill

Another test skill content.
"""
    skills_md = skill_dir / "SKILL.md"
    skills_md.write_text(content)

    return UPath(skill_dir)


@pytest.fixture
def manifest_with_skills(tmp_path: Path, test_skill: UPath) -> AgentsManifest:
    """Create a manifest with skills configured."""
    agent_config = NativeAgentConfig(
        name="test_agent",
        model="test",
        system_prompt="You are a test agent",
    )

    return AgentsManifest(
        agents={"test_agent": agent_config},
        skills=SkillsConfig(
            paths=[UPath(tmp_path)],
            include_default=False,
        ),
    )


# =============================================================================
# Test Class: SkillResolverProperty
# =============================================================================


@pytest.mark.integration
class TestSkillResolverProperty:
    """Test AgentPool.skill_resolver property."""

    async def test_skill_resolver_available_when_skills_configured(
        self,
        manifest_with_skills: AgentsManifest,
    ) -> None:
        """Test that skill_resolver is available when skills are configured."""
        async with AgentPool(manifest_with_skills) as pool:
            assert pool.skill_resolver is not None

    async def test_skill_resolver_is_uri_resolver(
        self,
        manifest_with_skills: AgentsManifest,
    ) -> None:
        """Test that skill_resolver is a SkillURIResolver instance."""
        async with AgentPool(manifest_with_skills) as pool:
            assert isinstance(pool.skill_resolver, SkillURIResolver)

    async def test_skill_resolver_has_providers(
        self,
        manifest_with_skills: AgentsManifest,
    ) -> None:
        """Test that skill_resolver has providers registered."""
        async with AgentPool(manifest_with_skills) as pool:
            resolver = pool.skill_resolver
            assert resolver is not None

            providers = resolver.list_providers()
            assert isinstance(providers, list)

    async def test_skill_resolver_exists_without_skills(
        self,
    ) -> None:
        """Test skill_resolver behavior without explicit skills config."""
        agent_config = NativeAgentConfig(
            name="test_agent",
            model="test",
            system_prompt="You are a test agent",
        )
        manifest = AgentsManifest(agents={"test_agent": agent_config})

        async with AgentPool(manifest) as pool:
            resolver = pool.skill_resolver
            assert resolver is not None
            assert len(resolver.list_providers()) >= 0


# =============================================================================
# Test Class: SkillCapabilitiesProperty
# =============================================================================


@pytest.mark.integration
class TestSkillCapabilitiesProperty:
    """Test AgentPool.skill_capabilities property (replaces skill_provider)."""

    async def test_skill_capabilities_available_when_skills_configured(
        self,
        manifest_with_skills: AgentsManifest,
    ) -> None:
        """Test that skill_capabilities is available when skills are configured."""
        async with AgentPool(manifest_with_skills) as pool:
            assert len(pool.skill_capabilities) > 0

    async def test_skill_capabilities_contains_skill_manager_cap(
        self,
        manifest_with_skills: AgentsManifest,
    ) -> None:
        """Test that skill_capabilities contains a SkillManagerCap instance."""
        async with AgentPool(manifest_with_skills) as pool:
            caps = pool.skill_capabilities
            assert len(caps) > 0
            assert isinstance(caps[0], SkillManagerCap)

    async def test_skill_capabilities_has_local_skills(
        self,
        manifest_with_skills: AgentsManifest,
    ) -> None:
        """Test that the SkillManagerCap has local skills loaded."""
        async with AgentPool(manifest_with_skills) as pool:
            cap = pool.skill_capabilities[0]
            assert isinstance(cap, SkillManagerCap)
            assert "test-skill" in cap.local_skills


# =============================================================================
# Test Class: SkillResolutionThroughPool
# =============================================================================


@pytest.mark.integration
class TestSkillResolutionThroughPool:
    """Test skill resolution through AgentPool."""

    async def test_resolve_via_skills_manager(
        self,
        manifest_with_skills: AgentsManifest,
    ) -> None:
        """Test resolving skills through pool's SkillsManager."""
        async with AgentPool(manifest_with_skills) as pool:
            skill = pool.skills.get_skill("test-skill")
            assert skill.name == "test-skill"

    async def test_list_skills_via_manager(
        self,
        manifest_with_skills: AgentsManifest,
    ) -> None:
        """Test listing skills through pool's SkillsManager."""
        async with AgentPool(manifest_with_skills) as pool:
            skills = pool.skills.list_skills()
            skill_names = {s.name for s in skills}
            assert "test-skill" in skill_names

    async def test_multiple_skills_resolution(
        self,
        tmp_path: Path,
        test_skill: UPath,
        another_skill: UPath,
    ) -> None:
        """Test resolution of multiple skills."""
        agent_config = NativeAgentConfig(
            name="test_agent",
            model="test",
            system_prompt="You are a test agent",
        )
        manifest = AgentsManifest(
            agents={"test_agent": agent_config},
            skills=SkillsConfig(
                paths=[UPath(tmp_path)],
                include_default=False,
            ),
        )

        async with AgentPool(manifest) as pool:
            skill1 = pool.skills.get_skill("test-skill")
            skill2 = pool.skills.get_skill("another-skill")

            assert skill1.name == "test-skill"
            assert skill2.name == "another-skill"


# =============================================================================
# Test Class: PoolLifecycle
# =============================================================================


@pytest.mark.integration
class TestPoolLifecycle:
    """Test skill integration during pool lifecycle."""

    async def test_resolver_initialized_on_enter(
        self,
        manifest_with_skills: AgentsManifest,
    ) -> None:
        """Test that resolver is initialized when pool enters context."""
        pool = AgentPool(manifest_with_skills)

        async with pool:
            assert pool.skill_resolver is not None

    async def test_skill_capabilities_initialized_on_enter(
        self,
        manifest_with_skills: AgentsManifest,
    ) -> None:
        """Test that skill_capabilities is initialized when pool enters context."""
        pool = AgentPool(manifest_with_skills)

        async with pool:
            assert len(pool.skill_capabilities) > 0

    async def test_skills_work_via_manager(
        self,
        manifest_with_skills: AgentsManifest,
    ) -> None:
        """Test that pool.skills works throughout pool lifecycle."""
        async with AgentPool(manifest_with_skills) as pool:
            legacy_skills = pool.skills.list_skills()

            legacy_names = {s.name for s in legacy_skills}

            assert "test-skill" in legacy_names


# =============================================================================
# Test Class: ProviderRegistration
# =============================================================================


@pytest.mark.integration
class TestRegisterUnregisterSkillProvider:
    """Test AgentPool.register_skill_provider() and unregister_skill_provider()."""

    async def test_register_skill_provider_adds_to_resolver(
        self,
        manifest_with_skills: AgentsManifest,
    ) -> None:
        """Test that register_skill_provider() adds provider to URI resolver."""
        async with AgentPool(manifest_with_skills) as pool:
            mock_provider = _FakeSkillResourceProvider("resolver_provider")

            pool.register_skill_provider(mock_provider)

            assert pool._skill_resolver is not None
            assert "resolver_provider" in pool._skill_resolver.list_providers()

    async def test_unregister_skill_provider_removes_from_resolver(
        self,
        manifest_with_skills: AgentsManifest,
    ) -> None:
        """Test that unregister_skill_provider() removes from URI resolver."""
        async with AgentPool(manifest_with_skills) as pool:
            mock_provider = _FakeSkillResourceProvider("rm_provider")

            pool.register_skill_provider(mock_provider)
            assert pool._skill_resolver is not None
            assert "rm_provider" in pool._skill_resolver.list_providers()

            pool.unregister_skill_provider(mock_provider)
            assert "rm_provider" not in pool._skill_resolver.list_providers()

    async def test_register_before_setup_buffers_and_drains(
        self,
        manifest_with_skills: AgentsManifest,
    ) -> None:
        """Test that register_skill_provider() buffers when called before setup."""
        async with AgentPool(manifest_with_skills) as pool:
            pending = getattr(pool, "_pending_skill_providers", [])
            assert len(pending) == 0


# =============================================================================
# Test Class: TopLevelMcpRegistration (RFC-0058)
# =============================================================================


@pytest.mark.integration
class TestTopLevelMcpPoolRegistration:
    """Top-level McpServerCap instances are independently registered at POOL scope."""

    @pytest.fixture
    def manifest_with_mcp(self) -> AgentsManifest:
        """Create a manifest with a top-level MCP server configured."""
        from wolfharness_config.mcp_server import StreamableHTTPMCPServerConfig

        agent_config = NativeAgentConfig(
            name="test_agent",
            model="test",
            system_prompt="You are a test agent",
        )
        return AgentsManifest(
            agents={"test_agent": agent_config},
            mcp_servers=[
                StreamableHTTPMCPServerConfig(
                    url="http://127.0.0.1:1/mcp",
                    name="kb",
                )
            ],
        )

    @staticmethod
    def _attach_cap(pool: Any) -> None:
        """Inject a provider directly for registration testing.

        This avoids requiring a live MCP server connection.
        """
        from wolfharness.capabilities.mcp_server_cap import McpServerCap

        cap = McpServerCap(
            pool.mcp.servers[0],
            name="pool_mcp_kb",
            client=object(),
        )
        pool.mcp.providers.append(cap)

    async def test_mcp_servers_registered_at_pool_scope(
        self,
        manifest_with_mcp: AgentsManifest,
    ) -> None:
        """The top-level McpServerCap is discoverable via get_resource_access()."""
        from wolfharness.capabilities.extension_registry import Scope, ScopeLevel
        from wolfharness.capabilities.mcp_server_cap import McpServerCap

        async with AgentPool(manifest_with_mcp) as pool:
            self._attach_cap(pool)
            await pool._rebuild_skill_capabilities()
            pool_scope = Scope(level=ScopeLevel.POOL)

            visible = pool.extension_registry.get_resource_access(pool_scope)
            mcp_caps = [c for c in visible if isinstance(c, McpServerCap)]

            assert mcp_caps, "McpServerCap should be registered at POOL scope"

    async def test_mcp_server_tool_prefix_de_duplicated(
        self,
        manifest_with_mcp: AgentsManifest,
        monkeypatch: Any,
    ) -> None:
        """Two servers sharing a display_name get unique tool_prefix values."""
        from wolfharness.mcp_server.manager import MCPManager
        from wolfharness_config.mcp_server import StreamableHTTPMCPServerConfig

        # Patch MCPClient so setup_server() proceeds without a live server.
        class _FakeClient:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            async def __aenter__(self) -> Self:
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

        monkeypatch.setattr(
            "wolfharness.mcp_server.client.MCPClient",
            _FakeClient,
        )

        manager = MCPManager(
            name="pool_mcp",
            servers=[
                StreamableHTTPMCPServerConfig(
                    url="http://127.0.0.1:1/mcp",
                    name="kb",
                ),
                StreamableHTTPMCPServerConfig(
                    url="http://127.0.0.1:2/mcp",
                    name="kb",
                ),
            ],
        )
        async with manager:
            prefixes = [p.tool_prefix for p in manager.providers]
            assert len(prefixes) == 2
            assert prefixes[0] == "kb"
            assert prefixes[1] == "kb_2"
