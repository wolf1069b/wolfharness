"""Tests for MCP connection persistence across turns (issue #175).

These tests verify that MCPToolset instances are eagerly entered on cache miss,
so that pydantic-ai's per-turn ``__aenter__``/``__aexit__`` goes 1→2→1 instead
of 0→1→0. This prevents connection teardown and re-initialization every turn.

The fix: ``MCPManager.get_capabilities()`` calls ``await toolset.__aenter__()``
on cache miss, holding one reference open for the session/pool lifetime.
``cleanup_session()`` and ``disconnect_all()`` already call ``__aexit__()``
which brings the count back to 0 and closes the connection.
"""

from __future__ import annotations

from unittest.mock import patch

from pydantic_ai.mcp import MCPToolset
import pytest

from wolfharness.mcp_server.config_snapshot import McpConfigEntry, McpConfigSnapshot
from wolfharness.mcp_server.manager import MCPManager
from wolfharness_config.mcp_server import StdioMCPServerConfig


pytestmark = pytest.mark.unit


# =============================================================================
# Tests: MCPToolset eager enter on cache miss
# =============================================================================


async def test_eager_enter_sets_running_count() -> None:
    """After get_capabilities(), the cached toolset should have _running_count == 1.

    Before fix: _running_count is 0 (no eager enter).
    After fix: _running_count is 1 (eagerly entered on cache miss).
    """
    config = StdioMCPServerConfig(command="python", args=["server.py"])
    manager = MCPManager(servers=[config])

    caps = await manager.get_capabilities()

    assert len(caps) == 1
    toolset = caps[0].local
    assert isinstance(toolset, MCPToolset)
    assert toolset._running_count == 1, (
        f"Expected _running_count==1 (eagerly entered), got {toolset._running_count}"
    )


async def test_eager_enter_persists_across_calls() -> None:
    """Multiple get_capabilities() calls should NOT re-enter the toolset.

    The toolset is cached by client_id. Only the first call (cache miss)
    should call __aenter__. Subsequent calls reuse the cached, already-entered
    toolset.
    """
    config = StdioMCPServerConfig(command="python", args=["server.py"])
    manager = MCPManager(servers=[config])

    caps1 = await manager.get_capabilities()
    caps2 = await manager.get_capabilities()

    assert caps1[0].local is caps2[0].local
    toolset = caps1[0].local
    assert toolset._running_count == 1, (
        f"Expected _running_count==1 after 2 calls, got {toolset._running_count}"
    )


async def test_disconnect_all_closes_eagerly_entered_toolset() -> None:
    """disconnect_all() should bring _running_count back to 0."""
    config = StdioMCPServerConfig(command="python", args=["server.py"])
    manager = MCPManager(servers=[config])

    caps = await manager.get_capabilities()
    toolset = caps[0].local
    assert toolset._running_count == 1

    await manager.disconnect_all()

    assert toolset._running_count == 0
    assert len(manager._toolset_cache) == 0


async def test_cleanup_session_closes_session_scoped_toolset() -> None:
    """cleanup_session() should close session-scoped toolsets.

    When a session_id is provided and the session context has a snapshot,
    session-scoped configs use ``ctx.toolset_cache``. cleanup_session() should
    call __aexit__ on those toolsets, bringing _running_count to 0.
    """
    config = StdioMCPServerConfig(command="python", args=["server.py"])
    manager = MCPManager(servers=[config])

    # Set up a session context with a session-scoped config
    session_id = "test-session"
    ctx = manager.get_or_create_session(session_id)
    # Build a snapshot with the config as a session-scoped config
    entry = McpConfigEntry(server_config=config, source="session")
    snapshot = McpConfigSnapshot(session_configs=(entry,))
    ctx.snapshot = snapshot

    caps = await manager.get_capabilities(session_id=session_id)
    assert len(caps) == 1
    toolset = caps[0].local
    assert toolset._running_count == 1

    await manager.cleanup_session(session_id)

    assert toolset._running_count == 0


async def test_eager_enter_failure_does_not_cache() -> None:
    """If __aenter__ fails, the toolset should NOT be cached.

    A failed connection should not leave a stale entry in the cache.
    Subsequent calls can retry.
    """
    config = StdioMCPServerConfig(command="python", args=["server.py"])
    manager = MCPManager(servers=[config])

    call_count = 0

    async def failing_aenter(self: MCPToolset) -> MCPToolset:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("MCP server unreachable")
        self._running_count += 1
        return self

    with (
        patch.object(MCPToolset, "__aenter__", failing_aenter),
        pytest.raises(ConnectionError),
    ):
        await manager.get_capabilities()

    # Cache should be empty (failed enter should not cache)
    assert len(manager._toolset_cache) == 0

    # Second call should succeed (retry)
    caps = await manager.get_capabilities()
    assert len(caps) == 1
    assert caps[0].local._running_count == 1


async def test_disconnect_all_also_cleans_session_scoped_toolsets() -> None:
    """disconnect_all() should also close session-scoped toolsets.

    Edge case found by edge-case-analyzer: disconnect_all() only cleaned
    global _toolset_cache but missed per-session toolset caches.
    """
    config = StdioMCPServerConfig(command="python", args=["server.py"])
    manager = MCPManager(servers=[config])

    # Set up a session with its own toolset cache
    session_id = "test-session"
    ctx = manager.get_or_create_session(session_id)
    entry = McpConfigEntry(server_config=config, source="pool")
    snapshot = McpConfigSnapshot(pool_configs=(entry,))
    ctx.snapshot = snapshot

    caps = await manager.get_capabilities(session_id=session_id)
    toolset = caps[0].local
    assert toolset._running_count == 1

    # disconnect_all should clean both global and session-scoped caches
    await manager.disconnect_all()

    assert toolset._running_count == 0
    assert len(manager._toolset_cache) == 0
    # Session context should also be cleaned up
    for sctx in manager._session_contexts.values():
        assert len(sctx.toolset_cache) == 0


# =============================================================================
# Tests: cleanup robustness — timeout and exception isolation
# =============================================================================


async def test_disconnect_all_isolates_failing_toolset() -> None:
    """A toolset __aexit__ that raises must not block cleanup of other toolsets.

    Verifies the fix for Gemini Code Assist review comment: each __aexit__
    call in disconnect_all() is wrapped in try-except so one failure
    doesn't abort the entire shutdown.
    """
    config_a = StdioMCPServerConfig(command="python", args=["server_a.py"])
    config_b = StdioMCPServerConfig(command="python", args=["server_b.py"])
    manager = MCPManager(servers=[config_a, config_b])

    caps = await manager.get_capabilities()
    assert len(caps) == 2
    toolset_a = caps[0].local
    toolset_b = caps[1].local

    # Patch toolset_a.__aexit__ to raise an unexpected exception
    original_aexit_a = toolset_a.__aexit__

    async def failing_aexit(exc_type, exc_val, exc_tb):
        raise OSError("connection reset")

    toolset_a.__aexit__ = failing_aexit  # type: ignore[assignment]

    # disconnect_all should not raise — the OSError is caught and logged
    await manager.disconnect_all()

    # toolset_b should still be closed despite toolset_a's failure
    assert toolset_b._running_count == 0
    assert len(manager._toolset_cache) == 0

    # Restore for cleanup
    toolset_a.__aexit__ = original_aexit_a  # type: ignore[assignment]


async def test_disconnect_all_timeout_does_not_block() -> None:
    """A toolset __aexit__ that hangs must not block cleanup indefinitely.

    Verifies that _MCP_CLEANUP_TIMEOUT prevents deadlocks during shutdown
    when an HTTP proxy doesn't promptly close TCP connections.
    """
    import asyncio

    config = StdioMCPServerConfig(command="python", args=["server.py"])
    manager = MCPManager(servers=[config])

    caps = await manager.get_capabilities()
    toolset = caps[0].local

    # Patch __aexit__ to hang forever
    hang_event = asyncio.Event()

    async def hanging_aexit(exc_type, exc_val, exc_tb):
        await hang_event.wait()  # Never set — will hang
        if toolset._running_count > 0:
            toolset._running_count -= 1

    toolset.__aexit__ = hanging_aexit  # type: ignore[assignment]

    # Use a short timeout for the test to avoid waiting 30s
    import unittest.mock

    with unittest.mock.patch("wolfharness.mcp_server.manager._MCP_CLEANUP_TIMEOUT", 0.1):
        # disconnect_all should complete within the timeout, not hang
        await asyncio.wait_for(manager.disconnect_all(), timeout=5.0)

    # Cache should be cleared even though __aexit__ timed out
    assert len(manager._toolset_cache) == 0


async def test_cleanup_session_isolates_failing_toolset() -> None:
    """cleanup_session() must not abort if one toolset __aexit__ raises.

    Verifies the catch-all except Exception added to cleanup_session()'s
    toolset cleanup loop.
    """
    config = StdioMCPServerConfig(command="python", args=["server.py"])
    manager = MCPManager(servers=[config])

    # Set up a session with a session-scoped config
    session_id = "test-session"
    ctx = manager.get_or_create_session(session_id)
    entry = McpConfigEntry(server_config=config, source="session")
    snapshot = McpConfigSnapshot(session_configs=(entry,))
    ctx.snapshot = snapshot

    caps = await manager.get_capabilities(session_id=session_id)
    toolset = caps[0].local
    assert toolset._running_count == 1

    # Patch __aexit__ to raise an unexpected exception
    async def failing_aexit(exc_type, exc_val, exc_tb):
        raise OSError("connection reset")

    toolset.__aexit__ = failing_aexit  # type: ignore[assignment]

    # cleanup_session should not raise — the OSError is caught and logged
    await manager.cleanup_session(session_id)

    # Session context should be removed
    assert manager.get_session_context(session_id) is None
