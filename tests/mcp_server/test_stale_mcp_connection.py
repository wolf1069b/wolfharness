"""Fix-verification tests for stale MCP connection bug on session resume.

Previously, when an agent shared the pool's MCPManager (``_mcp_shared =
True``), the ``_toolset_cache`` on that shared manager was never
invalidated between sessions.  On session resume, a new
``SessionConnectionPool`` was created with fresh transports, but
``get_capabilities()`` returned the OLD cached ``MCPToolset`` (which held
the dead transport from the previous WebSocket session).

After T10-T12, ``get_capabilities()`` accepts ``session_id`` and routes
session-scoped configs through per-session ``McpSessionContext`` objects
with their own ``toolset_cache``.  ``cleanup_session()`` clears the
per-session cache, ensuring the next session gets a fresh toolset.

These tests verify the fix at the MCPManager level without requiring
real ACP connections or WebSocket infrastructure.
"""

from __future__ import annotations

from typing import Any, Self, cast
from unittest.mock import patch

import pytest

from wolfharness.mcp_server.config_snapshot import McpConfigEntry, McpConfigSnapshot
from wolfharness.mcp_server.manager import MCPManager
from wolfharness_config.mcp_server import AcpMCPServerConfig


# ---------------------------------------------------------------------------
# Fakes (matching test_mcpmanager_caching.py patterns)
# ---------------------------------------------------------------------------


class _FakeToolset:
    """Fake MCPToolset that captures the transport for inspection."""

    def __init__(self, **kwargs: Any) -> None:
        self.client: Any = kwargs.get("client")
        self.id = kwargs.get("id")
        self.is_running = False

    async def __aenter__(self) -> Self:
        self.is_running = True
        return self

    async def __aexit__(self, *args: object) -> None:
        self.is_running = False


class _FakeMCP:
    """Fake MCP capability that exposes the underlying toolset."""

    def __init__(
        self,
        local: Any = None,
        allowed_tools: list[str] | None = None,
        id: str | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> None:
        self.local = local
        self.allowed_tools = allowed_tools
        self.id = id


class _FakeTransport:
    """Fake transport with a label to distinguish session 1 vs session 2."""

    def __init__(self, label: str) -> None:
        self.label = label


# ---------------------------------------------------------------------------
# Test: fresh toolset returned after session resume (was: stale toolset)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_session_resume_returns_fresh_toolset() -> None:
    """get_capabilities() returns a fresh toolset after session cleanup+recreate.

    After T10-T12, session-scoped toolsets are cached on the per-session
    ``McpSessionContext.toolset_cache`` rather than the global
    ``MCPManager._toolset_cache``.  When ``cleanup_session()`` is called
    between sessions, the per-session context (including its toolset
    cache) is removed, so the next session gets a brand-new toolset with
    the correct fresh transport.

    Steps:
    1. Create a shared MCPManager (simulating pool-level manager).
    2. Session 1: register session, add transport A, build snapshot,
       call ``get_capabilities(session_id="s1")`` -> toolset cached with
       transport A on the session context.
    3. Call ``cleanup_session("s1")`` to tear down session 1.
    4. Session 2 (resume): register a new session, add transport B,
       build a new snapshot, call ``get_capabilities(session_id="s2")``.
    5. Assert: the returned toolset holds transport B (fresh), NOT
       transport A (stale), and is a different object than toolset1.
    """
    manager = MCPManager(name="pool_mcp")

    acp_config = AcpMCPServerConfig(name="scratchpad", acp_id="acp-server-1")
    client_id = acp_config.client_id  # "acp_acp-server-1"

    try:
        # --- Session 1 ---
        ctx1 = manager.get_or_create_session("s1")
        assert ctx1.connection_pool is not None
        transport_a = _FakeTransport("session-1-transport")
        await ctx1.connection_pool.add_transport(client_id, cast(Any, transport_a))

        snapshot1 = McpConfigSnapshot(
            session_configs=(McpConfigEntry(server_config=acp_config, source="session"),),
        )
        manager.update_session_snapshot("s1", snapshot1)

        with (
            patch("pydantic_ai.mcp.MCPToolset", _FakeToolset),
            patch("pydantic_ai.capabilities.MCP", _FakeMCP),
        ):
            caps1 = await manager.get_capabilities(session_id="s1")

        assert len(caps1) == 1
        toolset1 = cast(_FakeToolset, caps1[0].local)
        assert toolset1.client is transport_a  # toolset holds session 1 transport
        assert client_id in ctx1.toolset_cache  # cached on session context

        # --- Clean up session 1 ---
        await manager.cleanup_session("s1")
        assert manager.get_session_context("s1") is None

        # --- Session 2 (resume) ---
        ctx2 = manager.get_or_create_session("s2")
        assert ctx2.connection_pool is not None
        transport_b = _FakeTransport("session-2-transport")
        await ctx2.connection_pool.add_transport(client_id, cast(Any, transport_b))

        snapshot2 = McpConfigSnapshot(
            session_configs=(McpConfigEntry(server_config=acp_config, source="session"),),
        )
        manager.update_session_snapshot("s2", snapshot2)

        with (
            patch("pydantic_ai.mcp.MCPToolset", _FakeToolset),
            patch("pydantic_ai.capabilities.MCP", _FakeMCP),
        ):
            caps2 = await manager.get_capabilities(session_id="s2")

        assert len(caps2) == 1
        toolset2 = cast(_FakeToolset, caps2[0].local)

        # FIX: toolset2 is a DIFFERENT object than toolset1 (not stale)
        assert toolset2 is not toolset1

        # FIX: toolset2 holds transport_b (fresh), not transport_a (stale)
        assert toolset2.client is transport_b
        assert toolset2.client is not transport_a

        # Session 2's toolset cache has the fresh toolset
        assert client_id in ctx2.toolset_cache
    finally:
        await manager.cleanup()
        if manager.get_session_context("s2") is not None:
            await manager.cleanup_session("s2")


# ---------------------------------------------------------------------------
# Test: cache key is deterministic across sessions (explains why bug occurred)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_acp_client_id_is_deterministic_across_sessions() -> None:
    """AcpMCPServerConfig.client_id is deterministic.

    The ``client_id`` for ACP configs is ``f"acp_{acp_id}"``, which means
    the same ACP server always maps to the same cache key.  This is why
    per-session isolation is necessary — without it, the cache key
    doesn't change even though the transport does.
    """
    config1 = AcpMCPServerConfig(name="server", acp_id="my-acp-1")
    config2 = AcpMCPServerConfig(name="server", acp_id="my-acp-1")

    assert config1.client_id == config2.client_id
    assert config1.client_id == "acp_my-acp-1"


# ---------------------------------------------------------------------------
# Test: SessionConnectionPool correctly provides fresh transport
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_session_pool_provides_fresh_transport_on_resume() -> None:
    """SessionConnectionPool correctly returns the new transport.

    This test confirms that SessionConnectionPool properly stores and
    returns the fresh transport.  With per-session contexts, each
    session's pool is isolated from the others.
    """
    from wolfharness.mcp_server.session_pool import SessionConnectionPool

    acp_config = AcpMCPServerConfig(name="scratchpad", acp_id="acp-server-1")
    client_id = acp_config.client_id

    # Session 1
    pool1 = SessionConnectionPool(session_id="s1")
    t1 = _FakeTransport("old")
    await pool1.add_transport(client_id, cast(Any, t1))
    result1 = await pool1.get_transport(acp_config)
    assert result1 is cast(Any, t1)

    # Session 2 (resume) — new pool, new transport
    pool2 = SessionConnectionPool(session_id="s2")
    t2 = _FakeTransport("new")
    await pool2.add_transport(client_id, cast(Any, t2))
    result2 = await pool2.get_transport(acp_config)
    assert result2 is cast(Any, t2)
    assert result2 is not cast(Any, t1)  # Fresh transport, not stale

    await pool1.cleanup()
    await pool2.cleanup()


# ---------------------------------------------------------------------------
# Test: multiple ACP servers all get fresh toolsets (was: all go stale)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_multiple_acp_servers_get_fresh_toolsets() -> None:
    """When multiple ACP servers are configured, all get fresh toolsets.

    After the fix, each session has its own ``toolset_cache``.  After
    ``cleanup_session()`` removes the session context, the next session
    creates fresh toolsets for every ACP MCP server, each holding the
    correct fresh transport.
    """
    manager = MCPManager(name="pool_mcp")

    config_a = AcpMCPServerConfig(name="server_a", acp_id="acp-a")
    config_b = AcpMCPServerConfig(name="server_b", acp_id="acp-b")

    try:
        # --- Session 1 ---
        ctx1 = manager.get_or_create_session("s1")
        assert ctx1.connection_pool is not None
        t_a1 = _FakeTransport("a-s1")
        t_b1 = _FakeTransport("b-s1")
        await ctx1.connection_pool.add_transport(config_a.client_id, cast(Any, t_a1))
        await ctx1.connection_pool.add_transport(config_b.client_id, cast(Any, t_b1))

        snapshot1 = McpConfigSnapshot(
            session_configs=(
                McpConfigEntry(server_config=config_a, source="session"),
                McpConfigEntry(server_config=config_b, source="session"),
            ),
        )
        manager.update_session_snapshot("s1", snapshot1)

        with (
            patch("pydantic_ai.mcp.MCPToolset", _FakeToolset),
            patch("pydantic_ai.capabilities.MCP", _FakeMCP),
        ):
            caps1 = await manager.get_capabilities(session_id="s1")

        assert len(caps1) == 2

        # --- Clean up session 1 ---
        await manager.cleanup_session("s1")
        assert manager.get_session_context("s1") is None

        # --- Session 2 (resume) ---
        ctx2 = manager.get_or_create_session("s2")
        assert ctx2.connection_pool is not None
        t_a2 = _FakeTransport("a-s2")
        t_b2 = _FakeTransport("b-s2")
        await ctx2.connection_pool.add_transport(config_a.client_id, cast(Any, t_a2))
        await ctx2.connection_pool.add_transport(config_b.client_id, cast(Any, t_b2))

        snapshot2 = McpConfigSnapshot(
            session_configs=(
                McpConfigEntry(server_config=config_a, source="session"),
                McpConfigEntry(server_config=config_b, source="session"),
            ),
        )
        manager.update_session_snapshot("s2", snapshot2)

        with (
            patch("pydantic_ai.mcp.MCPToolset", _FakeToolset),
            patch("pydantic_ai.capabilities.MCP", _FakeMCP),
        ):
            caps2 = await manager.get_capabilities(session_id="s2")

        assert len(caps2) == 2

        fresh_transports = {cast(Any, t_a2), cast(Any, t_b2)}
        stale_transports = {cast(Any, t_a1), cast(Any, t_b1)}

        # All toolsets hold session-2 transports (fresh, not stale)
        for cap in caps2:
            toolset = cast(_FakeToolset, cap.local)
            toolset_client = cast(Any, toolset.client)
            assert toolset_client in fresh_transports, (
                f"Toolset holds unexpected transport {toolset_client.label}, "
                f"expected one of session-2 transports"
            )
            assert toolset_client not in stale_transports, (
                f"Toolset should NOT hold session-1 transport {toolset_client.label}"
            )
    finally:
        await manager.cleanup()
        if manager.get_session_context("s2") is not None:
            await manager.cleanup_session("s2")


# ---------------------------------------------------------------------------
# Test: cleanup_session clears per-session cache (was: disconnect_all)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_cleanup_session_clears_per_session_cache() -> None:
    """cleanup_session() clears the per-session toolset cache.

    After T10-T12, ``cleanup_session()`` clears ``ctx.toolset_cache``,
    cleans the session connection pool, and removes the session context
    from the session registry.  This ensures stale toolsets are not
    reused when the session is resumed.
    """
    manager = MCPManager(name="pool_mcp")

    acp_config = AcpMCPServerConfig(name="scratchpad", acp_id="acp-1")
    client_id = acp_config.client_id

    try:
        # Session 1 — populate per-session cache
        ctx = manager.get_or_create_session("s1")
        assert ctx.connection_pool is not None
        t1 = _FakeTransport("s1")
        await ctx.connection_pool.add_transport(client_id, cast(Any, t1))

        snapshot = McpConfigSnapshot(
            session_configs=(McpConfigEntry(server_config=acp_config, source="session"),),
        )
        manager.update_session_snapshot("s1", snapshot)

        with (
            patch("pydantic_ai.mcp.MCPToolset", _FakeToolset),
            patch("pydantic_ai.capabilities.MCP", _FakeMCP),
        ):
            await manager.get_capabilities(session_id="s1")

        # Per-session cache is populated
        assert len(ctx.toolset_cache) == 1
        assert client_id in ctx.toolset_cache

        # cleanup_session clears the per-session cache and removes the context
        await manager.cleanup_session("s1")

        # Session context is removed from the registry
        assert manager.get_session_context("s1") is None
    finally:
        await manager.cleanup()
        if manager.get_session_context("s1") is not None:
            await manager.cleanup_session("s1")
