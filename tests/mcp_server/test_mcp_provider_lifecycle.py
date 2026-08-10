"""Integration tests for the MCP provider lifecycle.

Tests the full flow from config snapshot inheritance through to
toolset materialization, covering:

1. Snapshot inheritance (parent → child agent)
2. Pool-level configs visible to child agents
3. Agent-level configs NOT inherited by children
4. ``get_capabilities(session_id=...)`` with snapshot stored on session context
5. ``get_capabilities(session_id=...)`` for session-scoped configs via session context
6. Legacy fallback: ``get_capabilities()`` with no session_id
7. Skill configs in snapshot via ``with_skill_configs()``
8. Full lifecycle: snapshot → ``get_capabilities(session_id=...)`` → MCPToolset created
9. Real tool calls through the session_id path (in-process FastMCP)
"""

from __future__ import annotations

from typing import Any

from pydantic_ai._run_context import RunContext
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.usage import RunUsage
import pytest

from wolfharness.mcp_server.config_snapshot import McpConfigEntry, McpConfigSnapshot
from wolfharness.mcp_server.global_pool import GlobalConnectionPool
from wolfharness.mcp_server.manager import MCPManager
from wolfharness.mcp_server.session_pool import SessionConnectionPool
from wolfharness_config.mcp_server import (
    AcpMCPServerConfig,
    StdioMCPServerConfig,
    StreamableHTTPMCPServerConfig,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stdio_cfg(
    name: str, command: str = "python", args: list[str] | None = None
) -> StdioMCPServerConfig:
    """Create a stdio MCP server config for testing."""
    return StdioMCPServerConfig(name=name, command=command, args=args or [])


def _http_cfg(name: str, url: str = "http://localhost:9999/mcp") -> StreamableHTTPMCPServerConfig:
    """Create a streamable HTTP MCP server config for testing."""
    return StreamableHTTPMCPServerConfig(name=name, url=url)


def _acp_cfg(name: str, acp_id: str = "test-acp") -> AcpMCPServerConfig:
    """Create an ACP MCP server config for testing."""
    return AcpMCPServerConfig(name=name, acp_id=acp_id)


@pytest.fixture
def run_context() -> RunContext[Any]:
    """Minimal RunContext for toolset calls."""
    from pydantic_ai.models.test import TestModel

    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )


@pytest.fixture
def fastmcp_server() -> Any:
    """In-process FastMCP server with tools for integration testing."""
    from fastmcp.server import FastMCP

    server: FastMCP[None] = FastMCP("lifecycle_test_server")

    @server.tool()
    async def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello, {name}!"

    @server.tool()
    async def calculate(a: int, b: int) -> dict[str, int]:
        """Add two numbers."""
        return {"sum": a + b}

    return server


# ---------------------------------------------------------------------------
# 1. Snapshot inheritance: parent → child agent
# ---------------------------------------------------------------------------


class TestSnapshotInheritance:
    """Verify that child agents inherit the right config partitions from parent."""

    def test_child_inherits_pool_configs_from_parent(self):
        """Pool-level configs in parent's snapshot appear in child's pool_configs."""
        pool_entry = McpConfigEntry(server_config=_stdio_cfg("pool_srv"), source="pool")
        agent_entry = McpConfigEntry(server_config=_stdio_cfg("parent_agent_srv"), source="agent")
        session_entry = McpConfigEntry(server_config=_stdio_cfg("session_srv"), source="session")

        parent_snapshot = McpConfigSnapshot(
            pool_configs=(pool_entry,),
            agent_configs=(agent_entry,),
            session_configs=(session_entry,),
        )

        # Child agent builds its snapshot from parent's pool + session configs
        # but uses its OWN agent_configs.
        child_agent_entry = McpConfigEntry(
            server_config=_stdio_cfg("child_agent_srv"), source="agent"
        )
        child_snapshot = McpConfigSnapshot(
            pool_configs=parent_snapshot.pool_configs,
            agent_configs=(child_agent_entry,),
            session_configs=parent_snapshot.session_configs,
            skill_configs=(),
        )

        assert child_snapshot.pool_configs == (pool_entry,)
        assert child_snapshot.session_configs == (session_entry,)

    def test_child_does_not_inherit_parent_agent_configs(self):
        """Agent-level configs are NOT inherited — child uses its own."""
        parent_agent_entry = McpConfigEntry(server_config=_stdio_cfg("parent_only"), source="agent")
        parent_snapshot = McpConfigSnapshot(
            agent_configs=(parent_agent_entry,),
        )

        child_agent_entry = McpConfigEntry(server_config=_stdio_cfg("child_only"), source="agent")
        child_snapshot = McpConfigSnapshot(
            pool_configs=parent_snapshot.pool_configs,
            agent_configs=(child_agent_entry,),
            session_configs=parent_snapshot.session_configs,
        )

        # Parent's agent config must NOT appear in child
        assert parent_agent_entry not in child_snapshot.agent_configs
        assert child_agent_entry in child_snapshot.agent_configs

    def test_child_inherits_session_configs_from_parent(self):
        """Session-scoped configs are inherited from parent."""
        session_entry = McpConfigEntry(server_config=_http_cfg("session_srv"), source="session")
        parent_snapshot = McpConfigSnapshot(
            session_configs=(session_entry,),
        )
        child_snapshot = McpConfigSnapshot(
            pool_configs=parent_snapshot.pool_configs,
            agent_configs=(),
            session_configs=parent_snapshot.session_configs,
        )
        assert child_snapshot.session_configs == (session_entry,)


# ---------------------------------------------------------------------------
# 2. Pool-level configs visible to child via snapshot.global_configs
# ---------------------------------------------------------------------------


class TestPoolConfigsVisibleToChild:
    """Verify pool-level configs appear in child's snapshot.global_configs."""

    def test_global_configs_contains_pool_and_agent(self):
        """``global_configs`` property returns pool + agent configs."""
        pool_entry = McpConfigEntry(server_config=_stdio_cfg("p1"), source="pool")
        agent_entry = McpConfigEntry(server_config=_stdio_cfg("a1"), source="agent")
        session_entry = McpConfigEntry(server_config=_stdio_cfg("s1"), source="session")
        skill_entry = McpConfigEntry(
            server_config=_stdio_cfg("sk1"), source="skill", skill_name="my_skill"
        )

        snapshot = McpConfigSnapshot(
            pool_configs=(pool_entry,),
            agent_configs=(agent_entry,),
            session_configs=(session_entry,),
            skill_configs=(skill_entry,),
        )

        assert pool_entry in snapshot.global_configs
        assert agent_entry in snapshot.global_configs
        assert session_entry not in snapshot.global_configs
        assert skill_entry not in snapshot.global_configs

    def test_child_global_configs_has_pool_from_parent(self):
        """Child's global_configs includes pool configs inherited from parent."""
        pool_entry = McpConfigEntry(server_config=_stdio_cfg("shared"), source="pool")
        parent_snapshot = McpConfigSnapshot(pool_configs=(pool_entry,))

        child_snapshot = McpConfigSnapshot(
            pool_configs=parent_snapshot.pool_configs,
            agent_configs=(),
        )
        assert pool_entry in child_snapshot.global_configs


# ---------------------------------------------------------------------------
# 3. Agent-level configs NOT inherited
# ---------------------------------------------------------------------------


class TestAgentConfigsNotInherited:
    """Verify agent-level configs stay private to the declaring agent."""

    def test_parent_agent_configs_absent_from_child_global(self):
        """Parent's agent_configs are NOT in child's global_configs."""
        parent_agent_entry = McpConfigEntry(
            server_config=_stdio_cfg("parent_private"), source="agent"
        )
        parent_snapshot = McpConfigSnapshot(
            agent_configs=(parent_agent_entry,),
        )
        child_snapshot = McpConfigSnapshot(
            pool_configs=parent_snapshot.pool_configs,
            agent_configs=(),  # Child has no agent configs
        )
        assert parent_agent_entry not in child_snapshot.global_configs
        assert len(child_snapshot.global_configs) == 0


# ---------------------------------------------------------------------------
# 4. get_capabilities() with snapshot parameter
# ---------------------------------------------------------------------------


class TestAsCapabilityWithSnapshot:
    """Verify MCPManager.get_capabilities(session_id=...) works with a snapshot."""

    async def test_snapshot_global_configs_produce_capabilities(self):
        """Global configs in snapshot produce MCP capabilities."""
        manager = MCPManager(name="test")
        pool_entry = McpConfigEntry(
            server_config=_stdio_cfg("pool_srv", command="python", args=["s.py"]),
            source="pool",
        )
        agent_entry = McpConfigEntry(
            server_config=_stdio_cfg("agent_srv", command="python", args=["s2.py"]),
            source="agent",
        )
        snapshot = McpConfigSnapshot(
            pool_configs=(pool_entry,),
            agent_configs=(agent_entry,),
        )
        session_id = "test-snapshot-global"
        manager.get_or_create_session(session_id)
        manager.update_session_snapshot(session_id, snapshot)
        caps = await manager.get_capabilities(session_id=session_id)
        assert len(caps) == 2
        ids = {c.id for c in caps}
        assert "pool_srv" in ids
        assert "agent_srv" in ids
        await manager.cleanup_session(session_id)
        await manager.cleanup()

    async def test_snapshot_skips_disabled_servers(self):
        """Disabled servers in snapshot are skipped."""
        manager = MCPManager(name="test")
        enabled_entry = McpConfigEntry(
            server_config=_stdio_cfg("enabled", command="python", args=["s.py"]),
            source="pool",
        )
        disabled_cfg = _stdio_cfg("disabled", command="python", args=["s2.py"])
        disabled_cfg.enabled = False
        disabled_entry = McpConfigEntry(server_config=disabled_cfg, source="pool")
        snapshot = McpConfigSnapshot(
            pool_configs=(enabled_entry, disabled_entry),
        )
        session_id = "test-snapshot-disabled"
        manager.get_or_create_session(session_id)
        manager.update_session_snapshot(session_id, snapshot)
        caps = await manager.get_capabilities(session_id=session_id)
        assert len(caps) == 1
        assert caps[0].id == "enabled"
        await manager.cleanup_session(session_id)
        await manager.cleanup()

    async def test_snapshot_skips_acp_in_global_configs(self):
        """ACP configs in global_configs are skipped by get_capabilities()."""
        manager = MCPManager(name="test")
        stdio_entry = McpConfigEntry(
            server_config=_stdio_cfg("native", command="python", args=["s.py"]),
            source="pool",
        )
        acp_entry = McpConfigEntry(
            server_config=_acp_cfg("acp_srv"),
            source="pool",
        )
        snapshot = McpConfigSnapshot(
            pool_configs=(stdio_entry, acp_entry),
        )
        session_id = "test-snapshot-acp"
        manager.get_or_create_session(session_id)
        manager.update_session_snapshot(session_id, snapshot)
        caps = await manager.get_capabilities(session_id=session_id)
        assert len(caps) == 1
        assert caps[0].id == "native"
        await manager.cleanup_session(session_id)
        await manager.cleanup()

    async def test_snapshot_empty_global_configs(self):
        """Empty snapshot produces no global capabilities."""
        manager = MCPManager(name="test")
        snapshot = McpConfigSnapshot()
        session_id = "test-snapshot-empty"
        manager.get_or_create_session(session_id)
        manager.update_session_snapshot(session_id, snapshot)
        caps = await manager.get_capabilities(session_id=session_id)
        assert len(caps) == 0
        await manager.cleanup_session(session_id)
        await manager.cleanup()


# ---------------------------------------------------------------------------
# 5. get_capabilities() with session_pool for session-scoped configs
# ---------------------------------------------------------------------------


class TestAsCapabilityWithSessionPool:
    """Verify get_capabilities(session_id=...) uses session context for session-scoped configs."""

    async def test_session_scoped_configs_use_session_pool(self):
        """Session-scoped configs get transports from the session's connection pool."""
        manager = MCPManager(name="test")
        session_id = "test-session-scoped"

        session_entry = McpConfigEntry(
            server_config=_http_cfg("session_srv", "http://localhost:8888/mcp"),
            source="session",
        )
        skill_entry = McpConfigEntry(
            server_config=_http_cfg("skill_srv", "http://localhost:8889/mcp"),
            source="skill",
            skill_name="my_skill",
        )
        snapshot = McpConfigSnapshot(
            session_configs=(session_entry,),
            skill_configs=(skill_entry,),
        )
        manager.get_or_create_session(session_id)
        manager.update_session_snapshot(session_id, snapshot)
        caps = await manager.get_capabilities(session_id=session_id)
        assert len(caps) == 2
        ids = {c.id for c in caps}
        assert "session_srv" in ids
        assert "skill_srv" in ids
        await manager.cleanup_session(session_id)
        await manager.cleanup()

    async def test_session_pool_caches_transports(self):
        """SessionConnectionPool caches transports by (client_id, skill_name)."""
        session_pool = SessionConnectionPool("test-session")
        cfg = _http_cfg("srv", "http://localhost:7777/mcp")

        t1 = await session_pool.get_transport(cfg)
        t2 = await session_pool.get_transport(cfg)
        assert t1 is t2  # Same cached transport
        await session_pool.cleanup()

    async def test_session_pool_isolates_by_skill_name(self):
        """Different skill_name produces different transports."""
        session_pool = SessionConnectionPool("test-session")
        cfg = _http_cfg("srv", "http://localhost:6666/mcp")

        t1 = await session_pool.get_transport(cfg, skill_name="skill_a")
        t2 = await session_pool.get_transport(cfg, skill_name="skill_b")
        assert t1 is not t2
        await session_pool.cleanup()

    async def test_session_scoped_configs_require_session_id(self):
        """Session-scoped configs in a snapshot are not processed without session_id.

        Without a session_id, get_capabilities() uses the legacy path
        (self.servers only) and does not consult any snapshot.
        """
        manager = MCPManager(name="test")
        session_entry = McpConfigEntry(
            server_config=_http_cfg("session_srv"),
            source="session",
        )
        snapshot = McpConfigSnapshot(
            session_configs=(session_entry,),
        )
        # Store snapshot on a session context, but call get_capabilities()
        # without session_id — the snapshot should not be consulted.
        session_id = "test-no-session-id"
        manager.get_or_create_session(session_id)
        manager.update_session_snapshot(session_id, snapshot)

        # Without session_id, only self.servers is processed (empty here)
        caps = await manager.get_capabilities()
        assert len(caps) == 0
        await manager.cleanup_session(session_id)
        await manager.cleanup()

    async def test_add_transport_stores_pre_created(self):
        """add_transport() stores pre-created transport for reuse."""
        session_pool = SessionConnectionPool("test-session")
        cfg = _http_cfg("pre_created", "http://localhost:5555/mcp")

        # Create a mock transport
        from unittest.mock import MagicMock

        mock_transport = MagicMock()
        await session_pool.add_transport(cfg.client_id, mock_transport)

        # get_transport should return the pre-created one
        t = await session_pool.get_transport(cfg)
        assert t is mock_transport
        await session_pool.cleanup()


# ---------------------------------------------------------------------------
# 6. Legacy fallback: get_capabilities(snapshot=None)
# ---------------------------------------------------------------------------


class TestLegacyFallback:
    """Verify get_capabilities(snapshot=None) still works for backward compat."""

    async def test_legacy_path_uses_self_servers(self):
        """Legacy path uses manager.servers list directly."""
        manager = MCPManager(
            name="test",
            servers=[
                _stdio_cfg("srv1", command="python", args=["s1.py"]),
                _stdio_cfg("srv2", command="python", args=["s2.py"]),
            ],
        )
        caps = await manager.get_capabilities()
        assert len(caps) == 2
        ids = {c.id for c in caps}
        assert "srv1" in ids
        assert "srv2" in ids
        await manager.cleanup()

    async def test_legacy_path_skips_acp(self):
        """Legacy path skips ACP configs."""
        manager = MCPManager(
            name="test",
            servers=[
                _stdio_cfg("native", command="python", args=["s.py"]),
                _acp_cfg("acp_srv"),
            ],
        )
        caps = await manager.get_capabilities()
        assert len(caps) == 1
        assert caps[0].id == "native"
        await manager.cleanup()

    async def test_legacy_path_skips_disabled(self):
        """Legacy path skips disabled configs."""
        disabled_cfg = _stdio_cfg("disabled", command="python", args=["s.py"])
        disabled_cfg.enabled = False
        manager = MCPManager(
            name="test",
            servers=[
                _stdio_cfg("enabled", command="python", args=["s2.py"]),
                disabled_cfg,
            ],
        )
        caps = await manager.get_capabilities()
        assert len(caps) == 1
        assert caps[0].id == "enabled"
        await manager.cleanup()

    async def test_legacy_path_empty_servers(self):
        """Legacy path with no servers returns empty list."""
        manager = MCPManager(name="test")
        caps = await manager.get_capabilities()
        assert len(caps) == 0
        await manager.cleanup()


# ---------------------------------------------------------------------------
# 7. Skill configs in snapshot via with_skill_configs()
# ---------------------------------------------------------------------------


class TestSkillConfigsInSnapshot:
    """Verify with_skill_configs() adds skill MCP configs to snapshot."""

    def test_with_skill_configs_replaces_skill_configs(self):
        """with_skill_configs() replaces (not appends) skill configs."""
        original_skill = McpConfigEntry(
            server_config=_stdio_cfg("old_skill_srv"), source="skill", skill_name="old"
        )
        snapshot = McpConfigSnapshot(skill_configs=(original_skill,))

        new_skill = McpConfigEntry(
            server_config=_stdio_cfg("new_skill_srv"), source="skill", skill_name="new"
        )
        updated = snapshot.with_skill_configs((new_skill,))

        assert original_skill not in updated.skill_configs
        assert new_skill in updated.skill_configs
        assert len(updated.skill_configs) == 1

    def test_with_skill_configs_preserves_other_partitions(self):
        """with_skill_configs() preserves pool/agent/session configs."""
        pool_entry = McpConfigEntry(server_config=_stdio_cfg("p"), source="pool")
        agent_entry = McpConfigEntry(server_config=_stdio_cfg("a"), source="agent")
        session_entry = McpConfigEntry(server_config=_stdio_cfg("s"), source="session")

        snapshot = McpConfigSnapshot(
            pool_configs=(pool_entry,),
            agent_configs=(agent_entry,),
            session_configs=(session_entry,),
        )
        skill_entry = McpConfigEntry(
            server_config=_stdio_cfg("sk"), source="skill", skill_name="sk1"
        )
        updated = snapshot.with_skill_configs((skill_entry,))

        assert updated.pool_configs == (pool_entry,)
        assert updated.agent_configs == (agent_entry,)
        assert updated.session_configs == (session_entry,)
        assert updated.skill_configs == (skill_entry,)

    def test_with_skill_configs_on_empty_snapshot(self):
        """with_skill_configs() on empty snapshot works."""
        snapshot = McpConfigSnapshot()
        skill_entry = McpConfigEntry(
            server_config=_stdio_cfg("sk"), source="skill", skill_name="sk1"
        )
        updated = snapshot.with_skill_configs((skill_entry,))
        assert updated.skill_configs == (skill_entry,)
        assert updated.pool_configs == ()

    def test_skill_configs_in_session_scoped(self):
        """Skill configs appear in session_scoped_configs property."""
        skill_entry = McpConfigEntry(
            server_config=_stdio_cfg("sk"), source="skill", skill_name="sk1"
        )
        snapshot = McpConfigSnapshot(skill_configs=(skill_entry,))
        assert skill_entry in snapshot.session_scoped_configs

    def test_with_session_configs_replaces_session_configs(self):
        """with_session_configs() replaces session configs."""
        old_session = McpConfigEntry(server_config=_stdio_cfg("old_sess"), source="session")
        snapshot = McpConfigSnapshot(session_configs=(old_session,))

        new_session = McpConfigEntry(server_config=_stdio_cfg("new_sess"), source="session")
        updated = snapshot.with_session_configs((new_session,))

        assert old_session not in updated.session_configs
        assert new_session in updated.session_configs


# ---------------------------------------------------------------------------
# 8. Full lifecycle: snapshot → get_capabilities() → MCPToolset created
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    """End-to-end: create snapshot → get_capabilities(session_id=...) → MCPToolset materialized."""

    async def test_full_lifecycle_global_configs(self):
        """Pool + agent configs flow through snapshot to MCPToolset."""
        manager = MCPManager(name="test")
        pool_entry = McpConfigEntry(
            server_config=_stdio_cfg("pool_srv", command="python", args=["s.py"]),
            source="pool",
        )
        agent_entry = McpConfigEntry(
            server_config=_stdio_cfg("agent_srv", command="python", args=["s2.py"]),
            source="agent",
        )
        snapshot = McpConfigSnapshot(
            pool_configs=(pool_entry,),
            agent_configs=(agent_entry,),
        )
        session_id = "test-lifecycle-global"
        manager.get_or_create_session(session_id)
        manager.update_session_snapshot(session_id, snapshot)
        caps = await manager.get_capabilities(session_id=session_id)

        assert len(caps) == 2
        for cap in caps:
            assert isinstance(cap.local, MCPToolset)
            assert cap.id is not None

        await manager.cleanup_session(session_id)
        await manager.cleanup()

    async def test_full_lifecycle_with_session_pool(self):
        """Full lifecycle with session-scoped configs via session context."""
        manager = MCPManager(name="test")
        session_id = "test-lifecycle-session"

        pool_entry = McpConfigEntry(
            server_config=_stdio_cfg("pool_srv", command="python", args=["s.py"]),
            source="pool",
        )
        session_entry = McpConfigEntry(
            server_config=_http_cfg("session_srv", "http://localhost:4444/mcp"),
            source="session",
        )
        snapshot = McpConfigSnapshot(
            pool_configs=(pool_entry,),
            session_configs=(session_entry,),
        )
        manager.get_or_create_session(session_id)
        manager.update_session_snapshot(session_id, snapshot)
        caps = await manager.get_capabilities(session_id=session_id)

        # 1 global + 1 session-scoped
        assert len(caps) == 2
        for cap in caps:
            assert isinstance(cap.local, MCPToolset)

        await manager.cleanup_session(session_id)
        await manager.cleanup()

    async def test_full_lifecycle_snapshot_then_skill_update(self):
        """Snapshot can be updated with skill configs after initial creation."""
        manager = MCPManager(name="test")
        session_id = "test-lifecycle-skill-update"

        pool_entry = McpConfigEntry(
            server_config=_stdio_cfg("pool_srv", command="python", args=["s.py"]),
            source="pool",
        )
        snapshot = McpConfigSnapshot(pool_configs=(pool_entry,))

        manager.get_or_create_session(session_id)
        manager.update_session_snapshot(session_id, snapshot)

        # First call without skill configs
        caps1 = await manager.get_capabilities(session_id=session_id)
        assert len(caps1) == 1

        # Update snapshot with skill configs
        skill_entry = McpConfigEntry(
            server_config=_http_cfg("skill_srv", "http://localhost:3333/mcp"),
            source="skill",
            skill_name="my_skill",
        )
        updated_snapshot = snapshot.with_skill_configs((skill_entry,))
        manager.update_session_snapshot(session_id, updated_snapshot)

        # Second call with skill configs
        caps2 = await manager.get_capabilities(session_id=session_id)
        assert len(caps2) == 2  # 1 global + 1 session-scoped

        await manager.cleanup_session(session_id)
        await manager.cleanup()

    async def test_full_lifecycle_mixed_sources(self):
        """Snapshot with all four source types produces correct capabilities."""
        manager = MCPManager(name="test")
        session_id = "test-lifecycle-mixed"

        pool_entry = McpConfigEntry(
            server_config=_stdio_cfg("pool_srv", command="python", args=["s.py"]),
            source="pool",
        )
        agent_entry = McpConfigEntry(
            server_config=_stdio_cfg("agent_srv", command="python", args=["s2.py"]),
            source="agent",
        )
        session_entry = McpConfigEntry(
            server_config=_http_cfg("session_srv", "http://localhost:2222/mcp"),
            source="session",
        )
        skill_entry = McpConfigEntry(
            server_config=_http_cfg("skill_srv", "http://localhost:1111/mcp"),
            source="skill",
            skill_name="sk1",
        )
        snapshot = McpConfigSnapshot(
            pool_configs=(pool_entry,),
            agent_configs=(agent_entry,),
            session_configs=(session_entry,),
            skill_configs=(skill_entry,),
        )
        manager.get_or_create_session(session_id)
        manager.update_session_snapshot(session_id, snapshot)
        caps = await manager.get_capabilities(session_id=session_id)

        # 2 global (pool + agent) + 2 session-scoped (session + skill) = 4
        assert len(caps) == 4
        ids = {c.id for c in caps}
        assert {"pool_srv", "agent_srv", "session_srv", "skill_srv"} == ids

        await manager.cleanup_session(session_id)
        await manager.cleanup()

    async def test_capability_has_correct_url(self):
        """Each MCP capability has the correct URL based on server type."""
        manager = MCPManager(name="test")
        stdio_entry = McpConfigEntry(
            server_config=_stdio_cfg("stdio_srv", command="python", args=["s.py"]),
            source="pool",
        )
        http_entry = McpConfigEntry(
            server_config=_http_cfg("http_srv", "http://localhost:8080/mcp"),
            source="pool",
        )
        snapshot = McpConfigSnapshot(
            pool_configs=(stdio_entry, http_entry),
        )
        session_id = "test-url"
        manager.get_or_create_session(session_id)
        manager.update_session_snapshot(session_id, snapshot)
        caps = await manager.get_capabilities(session_id=session_id)

        cap_by_id = {c.id: c for c in caps}
        # stdio produces mcp://stdio/... URL
        assert cap_by_id["stdio_srv"].url.startswith("mcp://stdio/")
        # HTTP produces the actual URL
        assert "localhost:8080" in cap_by_id["http_srv"].url

        await manager.cleanup_session(session_id)
        await manager.cleanup()


# ---------------------------------------------------------------------------
# 9. Real tool calls through the snapshot path (in-process FastMCP)
# ---------------------------------------------------------------------------


@pytest.mark.real_mcp
class TestRealToolCallsViaSnapshot:
    """Exercise real tool calls through the session_id-based capability path.

    Uses in-process FastMCP servers (no external processes) to verify
    that MCPToolset instances created via get_capabilities(session_id=...)
    can actually list and call tools.
    """

    async def test_snapshot_capability_can_call_tools(
        self, fastmcp_server: Any, run_context: RunContext[Any]
    ):
        """MCPToolset from session_id path can list and call tools."""
        # Build a snapshot with a single global config
        # We use the in-process server directly as the "transport"
        manager = MCPManager(name="test")

        # Create a StreamableHTTPMCPServerConfig — but we'll patch
        # GlobalConnectionPool.get_transport to return the in-process server
        cfg = _http_cfg("test_server", "http://localhost:0/mcp")
        entry = McpConfigEntry(server_config=cfg, source="pool")
        snapshot = McpConfigSnapshot(pool_configs=(entry,))

        session_id = "test-real-tools"
        manager.get_or_create_session(session_id)
        manager.update_session_snapshot(session_id, snapshot)

        # Patch the global pool to return our in-process server
        original_get_transport = GlobalConnectionPool.get_transport

        async def _mock_get_transport(self_pool: Any, config: Any) -> Any:
            return fastmcp_server

        GlobalConnectionPool.get_transport = _mock_get_transport  # type: ignore[method-assign]
        try:
            caps = await manager.get_capabilities(session_id=session_id)
            assert len(caps) == 1
            cap = caps[0]
            assert isinstance(cap.local, MCPToolset)

            async with cap.local as toolset:
                tools = await toolset.get_tools(run_context)
                assert "greet" in tools
                assert "calculate" in tools

                result = await toolset.call_tool(
                    "greet", {"name": "World"}, run_context, tools["greet"]
                )
                assert result == "Hello, World!"

                result = await toolset.call_tool(
                    "calculate", {"a": 10, "b": 20}, run_context, tools["calculate"]
                )
                assert result == {"sum": 30}
        finally:
            GlobalConnectionPool.get_transport = original_get_transport  # type: ignore[method-assign]
            await manager.cleanup_session(session_id)
            await manager.cleanup()

    async def test_two_snapshots_produce_independent_toolsets(
        self, fastmcp_server: Any, run_context: RunContext[Any]
    ):
        """Two get_capabilities() calls with the same session share a cached MCPToolset."""
        manager = MCPManager(name="test")
        cfg = _http_cfg("shared_server", "http://localhost:0/mcp")
        entry = McpConfigEntry(server_config=cfg, source="pool")
        snapshot = McpConfigSnapshot(pool_configs=(entry,))

        session_id = "test-two-snapshots"
        manager.get_or_create_session(session_id)
        manager.update_session_snapshot(session_id, snapshot)

        original_get_transport = GlobalConnectionPool.get_transport

        async def _mock_get_transport(self_pool: Any, config: Any) -> Any:
            return fastmcp_server

        GlobalConnectionPool.get_transport = _mock_get_transport  # type: ignore[method-assign]
        try:
            caps1 = await manager.get_capabilities(session_id=session_id)
            caps2 = await manager.get_capabilities(session_id=session_id)

            # MCP wrappers are distinct but share the cached MCPToolset
            assert caps1[0] is not caps2[0]
            assert caps1[0].local is caps2[0].local

            async with caps1[0].local as ts1:
                tools1 = await ts1.get_tools(run_context)
                r1 = await ts1.call_tool("greet", {"name": "first"}, run_context, tools1["greet"])
                assert r1 == "Hello, first!"
        finally:
            GlobalConnectionPool.get_transport = original_get_transport  # type: ignore[method-assign]
            await manager.cleanup_session(session_id)
            await manager.cleanup()

    async def test_session_scoped_toolset_works(
        self, fastmcp_server: Any, run_context: RunContext[Any]
    ):
        """Session-scoped config via session context produces working toolset."""
        manager = MCPManager(name="test")
        session_id = "test-session-scoped-tools"

        cfg = _http_cfg("session_srv", "http://localhost:0/mcp")
        entry = McpConfigEntry(server_config=cfg, source="session")
        snapshot = McpConfigSnapshot(session_configs=(entry,))

        ctx = manager.get_or_create_session(session_id)
        manager.update_session_snapshot(session_id, snapshot)

        # Add the in-process server as a pre-created transport in the
        # session's connection pool
        if ctx.connection_pool is not None:
            await ctx.connection_pool.add_transport(cfg.client_id, fastmcp_server)

        caps = await manager.get_capabilities(session_id=session_id)
        assert len(caps) == 1
        cap = caps[0]
        assert isinstance(cap.local, MCPToolset)

        async with cap.local as toolset:
            tools = await toolset.get_tools(run_context)
            assert "greet" in tools

            result = await toolset.call_tool(
                "greet", {"name": "Session"}, run_context, tools["greet"]
            )
            assert result == "Hello, Session!"

        await manager.cleanup_session(session_id)
        await manager.cleanup()


# ---------------------------------------------------------------------------
# 10. Snapshot immutability and composition
# ---------------------------------------------------------------------------


class TestSnapshotImmutability:
    """Verify McpConfigSnapshot is immutable and composable."""

    def test_snapshot_is_frozen(self):
        """McpConfigSnapshot is a frozen dataclass."""
        entry = McpConfigEntry(server_config=_stdio_cfg("srv"), source="pool")
        snapshot = McpConfigSnapshot(pool_configs=(entry,))
        with pytest.raises(AttributeError):
            snapshot.pool_configs = ()  # type: ignore[misc]

    def test_entry_is_frozen(self):
        """McpConfigEntry is a frozen dataclass."""
        entry = McpConfigEntry(server_config=_stdio_cfg("srv"), source="pool")
        with pytest.raises(AttributeError):
            entry.source = "agent"  # type: ignore[misc]

    def test_all_configs_property(self):
        """all_configs returns all entries in canonical order."""
        pool_e = McpConfigEntry(server_config=_stdio_cfg("p"), source="pool")
        agent_e = McpConfigEntry(server_config=_stdio_cfg("a"), source="agent")
        session_e = McpConfigEntry(server_config=_stdio_cfg("s"), source="session")
        skill_e = McpConfigEntry(server_config=_stdio_cfg("sk"), source="skill", skill_name="sk1")
        snapshot = McpConfigSnapshot(
            pool_configs=(pool_e,),
            agent_configs=(agent_e,),
            session_configs=(session_e,),
            skill_configs=(skill_e,),
        )
        assert snapshot.all_configs == (pool_e, agent_e, session_e, skill_e)

    def test_empty_snapshot_all_configs(self):
        """Empty snapshot has empty all_configs."""
        snapshot = McpConfigSnapshot()
        assert snapshot.all_configs == ()
        assert snapshot.global_configs == ()
        assert snapshot.session_scoped_configs == ()


# ---------------------------------------------------------------------------
# 11. Simulated parent → child snapshot building (orchestrator pattern)
# ---------------------------------------------------------------------------


class TestOrchestratorSnapshotPattern:
    """Simulate the snapshot building logic from orchestrator/core.py.

    Verifies the pattern used in ``get_or_create_session_agent()``:
    - Child inherits parent's pool_configs and session_configs
    - Child gets its own agent_configs
    - skill_configs start empty
    """

    def test_child_snapshot_from_parent(self):
        """Child snapshot is built from parent's pool+session + child's agent."""
        # Parent has pool, agent, and session configs
        pool_e = McpConfigEntry(server_config=_stdio_cfg("pool_srv"), source="pool")
        parent_agent_e = McpConfigEntry(
            server_config=_stdio_cfg("parent_agent_srv"), source="agent"
        )
        session_e = McpConfigEntry(server_config=_http_cfg("session_srv"), source="session")
        parent_snapshot = McpConfigSnapshot(
            pool_configs=(pool_e,),
            agent_configs=(parent_agent_e,),
            session_configs=(session_e,),
        )

        # Child builds its own agent_configs
        child_agent_e = McpConfigEntry(server_config=_stdio_cfg("child_agent_srv"), source="agent")

        # This is the pattern from orchestrator/core.py L1026-1039
        child_snapshot = McpConfigSnapshot(
            pool_configs=parent_snapshot.pool_configs,
            agent_configs=(child_agent_e,),
            session_configs=parent_snapshot.session_configs,
            skill_configs=(),
        )

        # Child inherits pool and session from parent
        assert child_snapshot.pool_configs == (pool_e,)
        assert child_snapshot.session_configs == (session_e,)

        # Child does NOT inherit parent's agent configs
        assert parent_agent_e not in child_snapshot.agent_configs
        assert child_agent_e in child_snapshot.agent_configs

        # Skill configs start empty
        assert child_snapshot.skill_configs == ()

    def test_main_session_snapshot_pattern(self):
        """Main session snapshot pattern from orchestrator/core.py L1099-1104."""
        # Simulate _build_pool_configs() and _build_agent_configs()
        pool_e = McpConfigEntry(server_config=_stdio_cfg("pool_srv"), source="pool")
        agent_e = McpConfigEntry(server_config=_stdio_cfg("agent_srv"), source="agent")

        # Pattern from L1099-1104
        snapshot = McpConfigSnapshot(
            pool_configs=(pool_e,),
            agent_configs=(agent_e,),
            session_configs=(),
            skill_configs=(),
        )
        assert snapshot.global_configs == (pool_e, agent_e)
        assert snapshot.session_scoped_configs == ()

    def test_non_native_snapshot_pattern(self):
        """Non-native agent snapshot pattern from orchestrator/core.py L1148-1165."""
        pool_e = McpConfigEntry(server_config=_stdio_cfg("pool_srv"), source="pool")
        agent_e = McpConfigEntry(server_config=_stdio_cfg("agent_srv"), source="agent")

        # Pattern from L1160-1165
        snapshot = McpConfigSnapshot(
            pool_configs=(pool_e,),
            agent_configs=(agent_e,),
            session_configs=(),
            skill_configs=(),
        )
        assert snapshot.all_configs == (pool_e, agent_e)

    def test_acp_session_merges_session_configs(self):
        """ACP session.merge pattern from session.py L519-531.

        When ``initialize_mcp_servers()`` runs, it merges new session
        entries into the existing snapshot, deduplicating by client_id.
        """
        existing_session_e = McpConfigEntry(
            server_config=_http_cfg("old_session", "http://localhost:1/mcp"),
            source="session",
        )
        existing_snapshot = McpConfigSnapshot(
            session_configs=(existing_session_e,),
        )

        # New entries from initialize_mcp_servers()
        new_session_e = McpConfigEntry(
            server_config=_http_cfg("new_session", "http://localhost:2/mcp"),
            source="session",
        )
        # Duplicate (same client_id as existing)
        dup_cfg = _http_cfg("old_session", "http://localhost:1/mcp")

        # Deduplication logic from session.py L522-527
        existing_session = existing_snapshot.session_configs
        seen_ids: set[str] = {e.server_config.client_id for e in existing_session}
        merged: list[McpConfigEntry] = list(existing_session)
        for cfg in (new_session_e.server_config, dup_cfg):
            entry = McpConfigEntry(server_config=cfg, source="session")
            if cfg.client_id not in seen_ids:
                merged.append(entry)
                seen_ids.add(cfg.client_id)

        updated = existing_snapshot.with_session_configs(tuple(merged))

        # Should have 2 entries (original + new, no duplicate)
        assert len(updated.session_configs) == 2
        client_ids = [e.server_config.client_id for e in updated.session_configs]
        # client_id for StreamableHTTPMCPServerConfig is based on URL, not name
        assert "streamable_http_http://localhost:1/mcp" in client_ids
        assert "streamable_http_http://localhost:2/mcp" in client_ids
        # No duplicate client_ids
        assert len(client_ids) == len(set(client_ids))


# ---------------------------------------------------------------------------
# 12. GlobalConnectionPool integration with get_capabilities()
# ---------------------------------------------------------------------------


class TestGlobalPoolIntegration:
    """Verify GlobalConnectionPool works correctly with get_capabilities(session_id=...)."""

    async def test_global_pool_caches_stdio_transports(self):
        """GlobalConnectionPool caches stdio transports across get_capabilities() calls."""
        manager = MCPManager(name="test")
        cfg = _stdio_cfg("cached_srv", command="python", args=["s.py"])
        entry = McpConfigEntry(server_config=cfg, source="pool")
        snapshot = McpConfigSnapshot(pool_configs=(entry,))

        session_id = "test-global-cache"
        manager.get_or_create_session(session_id)
        manager.update_session_snapshot(session_id, snapshot)

        caps1 = await manager.get_capabilities(session_id=session_id)
        caps2 = await manager.get_capabilities(session_id=session_id)

        # MCP wrappers are distinct
        assert caps1[0] is not caps2[0]
        # MCPToolset is cached by client_id (shared)
        assert caps1[0].local is caps2[0].local
        await manager.cleanup_session(session_id)
        await manager.cleanup()

    async def test_global_pool_skips_acp(self):
        """GlobalConnectionPool raises NotImplementedError for ACP configs."""
        pool = GlobalConnectionPool()
        acp_cfg = _acp_cfg("acp_srv")
        with pytest.raises(NotImplementedError, match="ACP"):
            await pool.get_transport(acp_cfg)
        await pool.shutdown_all()
