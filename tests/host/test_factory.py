"""Unit tests for AgentFactory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wolfharness.host.factory import AgentFactory


if TYPE_CHECKING:
    from wolfharness import AgentPool


pytestmark = pytest.mark.unit


_DEFAULT_TOOLS_PROVIDER = MagicMock()
_DEFAULT_MCP = MagicMock()


def _make_host_context(
    *,
    pool: Any | None = None,
    config_file_path: str | None = None,
    mcp: Any | None = _DEFAULT_MCP,
) -> Any:
    """Build a mock HostContext with common defaults."""
    ctx = MagicMock()
    ctx.config_file_path = config_file_path
    ctx.mcp = mcp if mcp is not None else MagicMock()
    return ctx


def _make_native_cfg(
    *,
    name: str | None = "test_agent",
    agent: Any | None = None,
) -> Any:
    """Build a mock NativeAgentConfig instance.

    The returned mock passes ``isinstance(cfg, NativeAgentConfig)``
    because we patch the isinstance check in the factory.
    """
    cfg = MagicMock()
    cfg.name = name
    cfg.get_agent = MagicMock(return_value=agent or MagicMock())
    cfg.get_mcp_servers = MagicMock(return_value=[])
    return cfg


def _make_agent_mock() -> Any:
    """Build a mock agent with all attributes the factory touches."""
    agent = MagicMock()
    agent.__aenter__ = AsyncMock(return_value=agent)
    agent.__aexit__ = AsyncMock(return_value=None)
    agent.load_session = AsyncMock(return_value=None)
    agent.env = None
    agent._internal_fs = MagicMock()
    agent._build_pool_configs = MagicMock(return_value=())
    agent._build_agent_configs = MagicMock(return_value=())
    agent.mcp = MagicMock()
    agent.mcp.get_or_create_session = MagicMock(return_value=MagicMock())
    agent.mcp.update_session_snapshot = MagicMock(return_value=None)
    agent.mcp._session_contexts = {}
    agent.mcp._acp_mcp_manager = None
    agent.tools = MagicMock()
    agent.tools.add_provider = MagicMock()
    return agent


def _make_session(*, parent_session_id: str | None = None) -> Any:
    """Build a mock SessionState."""
    session = MagicMock()
    session.parent_session_id = parent_session_id
    return session


# ---------------------------------------------------------------------------
# compile()
# ---------------------------------------------------------------------------


def test_compile_returns_empty_registry(minimal_pool: AgentPool) -> None:
    """Given a manifest and host_context, compile() returns empty AgentRegistry."""
    factory = AgentFactory(pool=minimal_pool)
    registry = factory.compile(
        manifest=MagicMock(),
        host_context=_make_host_context(),
    )
    assert len(registry) == 0
    assert registry.list_names() == []


def test_compile_does_not_register_skills_tools_provider_at_agent_scope(
    minimal_pool: AgentPool,
) -> None:
    """skills_tools_provider is registered at POOL scope by _rebuild_skill_capabilities().

    _compile_agent_capabilities() must NOT include it in the returned list,
    otherwise it gets registered at AGENT scope too, causing duplicate
    capability entries in get_visible_capabilities() and tool name conflicts.
    """
    from wolfharness.capabilities.extension_registry import Scope, ScopeLevel

    skills_provider = MagicMock()
    skills_provider.__class__ = type("FakeSkillsProvider", (), {})

    factory = AgentFactory(pool=minimal_pool)
    # Simulate skills_tools_provider already at POOL scope
    minimal_pool.extension_registry.register(skills_provider, Scope(level=ScopeLevel.POOL))

    # Build a manifest with one agent
    manifest = MagicMock()
    manifest.agents = {"test_agent": MagicMock()}
    manifest.team_mode = None

    factory.compile(
        manifest=manifest,
        host_context=_make_host_context(),
    )

    # Check AGENT scope does NOT have skills_provider
    agent_scope = Scope(level=ScopeLevel.AGENT, agent_name="test_agent")
    visible = minimal_pool.extension_registry.get_visible_capabilities(agent_scope)
    # skills_provider should appear exactly once (from POOL scope), not twice
    count = sum(1 for c in visible if c is skills_provider)
    assert count == 1, f"skills_tools_provider should appear once, not {count}"


def test_compile_registers_config_capabilities_at_agent_scope(
    minimal_pool: AgentPool,
) -> None:
    """Config-defined capabilities (e.g. Viking) are registered at AGENT scope."""
    # Use our test ResourceAccess capability
    from tests.fixtures.test_resource_cap import TestResourceAccessCap
    from wolfharness.capabilities.extension_registry import Scope, ScopeLevel

    test_cap = TestResourceAccessCap(read_text="test", read_uri="test://x")

    # Build a mock NativeAgentConfig with capabilities
    cfg = MagicMock()
    cfg.name = "test_agent"
    cfg.capabilities = [test_cap]
    cfg.get_tool_providers = MagicMock(return_value=[])
    cfg.team_mode = None

    # Make isinstance(cfg, NativeAgentConfig) return True
    manifest = MagicMock()
    manifest.agents = {"test_agent": cfg}
    manifest.team_mode = None

    factory = AgentFactory(pool=minimal_pool)

    with patch("wolfharness.models.agents.NativeAgentConfig", (type(cfg),)):
        factory.compile(
            manifest=manifest,
            host_context=_make_host_context(),
        )

    agent_scope = Scope(level=ScopeLevel.AGENT, agent_name="test_agent")
    visible = minimal_pool.extension_registry.get_visible_capabilities(agent_scope)
    # TestResourceAccessCap should be at AGENT scope.
    from wolfharness.capabilities.resource_protocols import ResourceAccess

    ra_caps = [c for c in visible if isinstance(c, ResourceAccess)]
    # SkillManagerCap now also implements ResourceAccess (RFC-0058); filter to
    # the config-defined capability under test.
    test_ra_caps = [c for c in ra_caps if isinstance(c, TestResourceAccessCap)]
    assert len(test_ra_caps) == 1
    assert isinstance(test_ra_caps[0], TestResourceAccessCap)


def test_get_visible_capabilities_no_duplicates_across_scopes(
    minimal_pool: AgentPool,
) -> None:
    """Same capability at POOL + AGENT scope appears twice — document this behavior.

    ExtensionRegistry.get_visible_capabilities() does NOT deduplicate.
    This test documents that behavior so callers know to handle duplicates
    or avoid cross-scope registration of the same instance.
    """
    from wolfharness.capabilities.extension_registry import Scope, ScopeLevel

    cap = MagicMock()
    reg = minimal_pool.extension_registry
    reg.register(cap, Scope(level=ScopeLevel.POOL))
    reg.register(cap, Scope(level=ScopeLevel.AGENT, agent_name="a1"))

    agent_scope = Scope(level=ScopeLevel.AGENT, agent_name="a1")
    visible = reg.get_visible_capabilities(agent_scope)
    count = sum(1 for c in visible if c is cap)
    assert count == 2, "Same cap at POOL + AGENT appears twice (no dedup)"


# ---------------------------------------------------------------------------
# create_session_agent — native main path
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="L2 migration: requires mock internals — remains L1 unit test")
@pytest.mark.asyncio
async def test_create_session_agent_native_main_calls_get_agent_with_pool(
    minimal_pool: AgentPool,
) -> None:
    """When cfg is NativeAgentConfig and no parent, get_agent is called with pool."""
    agent = _make_agent_mock()
    cfg = _make_native_cfg(agent=agent)
    pool = MagicMock()
    host_context = _make_host_context(pool=pool)

    factory = AgentFactory(pool=pool)

    with patch("wolfharness.models.agents.NativeAgentConfig", (type(cfg),)):
        result = await factory.create_session_agent(
            agent_name="test_agent",
            session_id="sess-1",
            host_context=host_context,
            session=_make_session(),
            cfg=cfg,
        )

    assert result is agent
    cfg.get_agent.assert_called_once_with(
        input_provider=None,
        pool=pool,
    )


@pytest.mark.asyncio
async def test_create_session_agent_native_main_calls_aenter(minimal_pool: AgentPool) -> None:
    """When creating a native main agent, __aenter__ is called."""
    agent = _make_agent_mock()
    cfg = _make_native_cfg(agent=agent)
    host_context = _make_host_context()

    factory = AgentFactory(pool=minimal_pool)

    with patch("wolfharness.models.agents.NativeAgentConfig", (type(cfg),)):
        await factory.create_session_agent(
            agent_name="test_agent",
            session_id="sess-1",
            host_context=host_context,
            session=_make_session(),
            cfg=cfg,
        )

    agent.__aenter__.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_session_agent_native_main_no_pool_providers(minimal_pool: AgentPool) -> None:
    """When creating a native main agent, no pool providers are added.

    (include_aggregating=False)
    """
    agent = _make_agent_mock()
    cfg = _make_native_cfg(agent=agent)
    host_context = _make_host_context()

    factory = AgentFactory(pool=minimal_pool)

    with patch("wolfharness.models.agents.NativeAgentConfig", (type(cfg),)):
        await factory.create_session_agent(
            agent_name="test_agent",
            session_id="sess-1",
            host_context=host_context,
            session=_make_session(),
            cfg=cfg,
        )

    # _inject_pool_providers with include_aggregating=False adds no providers.
    agent.tools.add_provider.assert_not_called()


# ---------------------------------------------------------------------------
# create_session_agent — non-native path
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="L2 migration: requires mock internals — remains L1 unit test")
@pytest.mark.asyncio
async def test_create_session_agent_non_native_builds_snapshot_manually(
    minimal_pool: AgentPool,
) -> None:
    """When cfg is NOT NativeAgentConfig, MCP snapshot is built from pool."""
    agent = _make_agent_mock()
    # Non-native cfg: not an instance of NativeAgentConfig
    cfg = MagicMock()
    cfg.name = "acp_agent"
    cfg.get_agent = MagicMock(return_value=agent)
    cfg.get_mcp_servers = MagicMock(return_value=[])

    # Mock MCP manager with one enabled server
    mock_server = MagicMock()
    mock_server.enabled = True
    mcp = MagicMock()
    mcp.servers = [mock_server]
    mcp.get_aggregating_provider = MagicMock(return_value=MagicMock())

    pool = MagicMock()
    host_context = _make_host_context(mcp=mcp, pool=pool)

    factory = AgentFactory(pool=pool)

    # empty tuple → isinstance always False
    with patch("wolfharness.models.agents.NativeAgentConfig", ()):
        result = await factory.create_session_agent(
            agent_name="acp_agent",
            session_id="sess-1",
            host_context=host_context,
            session=_make_session(),
            cfg=cfg,
        )

    assert result is agent
    cfg.get_agent.assert_called_once_with(
        input_provider=None,
        pool=pool,
    )
    agent.__aenter__.assert_awaited_once()
    agent.mcp.get_or_create_session.assert_called_once_with("sess-1")
    agent.mcp.update_session_snapshot.assert_called_once()


# ---------------------------------------------------------------------------
# create_session_agent — name fix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_session_agent_fixes_missing_name(minimal_pool: AgentPool) -> None:
    """When cfg.name is None, model_copy is called to set the name."""
    agent = _make_agent_mock()
    cfg = _make_native_cfg(name=None, agent=agent)
    # model_copy returns a new mock that also passes isinstance
    new_cfg = MagicMock()
    new_cfg.name = "fixed_name"
    new_cfg.get_agent = MagicMock(return_value=agent)
    new_cfg.get_mcp_servers = MagicMock(return_value=[])
    cfg.model_copy = MagicMock(return_value=new_cfg)

    host_context = _make_host_context()
    factory = AgentFactory(pool=minimal_pool)

    with patch("wolfharness.models.agents.NativeAgentConfig", (type(new_cfg),)):
        await factory.create_session_agent(
            agent_name="fixed_name",
            session_id="sess-1",
            host_context=host_context,
            session=_make_session(),
            cfg=cfg,
        )

    cfg.model_copy.assert_called_once_with(update={"name": "fixed_name"})
    new_cfg.get_agent.assert_called_once()


# ---------------------------------------------------------------------------
# create_session_agent — load_session called on native main
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_session_agent_native_main_loads_session(minimal_pool: AgentPool) -> None:
    """When creating a native main agent, load_session is called."""
    agent = _make_agent_mock()
    cfg = _make_native_cfg(agent=agent)
    host_context = _make_host_context()

    factory = AgentFactory(pool=minimal_pool)

    with patch("wolfharness.models.agents.NativeAgentConfig", (type(cfg),)):
        await factory.create_session_agent(
            agent_name="test_agent",
            session_id="sess-42",
            host_context=host_context,
            session=_make_session(),
            cfg=cfg,
        )

    agent.load_session.assert_awaited_once_with("sess-42")
