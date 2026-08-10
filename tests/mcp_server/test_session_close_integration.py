"""Integration test: create session, run turn, close, verify empty contexts.

Verifies the full lifecycle: ``get_or_create_session`` →
``update_session_snapshot`` → ``get_capabilities`` → ``cleanup_session``
leaves ``_session_contexts`` empty.
"""

from __future__ import annotations

import pytest

from wolfharness.mcp_server.config_snapshot import McpConfigSnapshot
from wolfharness.mcp_server.manager import MCPManager


@pytest.mark.integration
async def test_integration_create_run_close() -> None:
    """After create → snapshot → capability → cleanup, session context is gone.

    Exercises the full session lifecycle on MCPManager without real model
    calls.  Uses an empty ``McpConfigSnapshot`` so ``get_capabilities`` returns
    an empty list (no MCP servers to connect).
    """
    manager = MCPManager(name="test")
    session_id = "test-integration-1"

    try:
        # 1. Create session context
        manager.get_or_create_session(session_id)
        assert session_id in manager._session_contexts

        # 2. Store a snapshot (empty — no real servers needed)
        snapshot = McpConfigSnapshot()
        manager.update_session_snapshot(session_id, snapshot)

        ctx = manager.get_or_create_session(session_id)
        assert ctx.snapshot is snapshot

        # 3. Simulate running a turn — get_capabilities builds capabilities
        caps = await manager.get_capabilities(session_id=session_id)
        assert caps == []

        # Session context must still exist during the turn
        assert session_id in manager._session_contexts

        # 4. Close the session
        await manager.cleanup_session(session_id)

        # 5. Verify the session context has been removed
        assert session_id not in manager._session_contexts
    finally:
        # Ensure cleanup even on assertion failure
        await manager.cleanup_session(session_id)
        await manager.cleanup()


@pytest.mark.integration
async def test_integration_close_recreate_fresh() -> None:
    """After closing a session and recreating with the same ID, the new context is fresh.

    Verifies that ``cleanup_session`` fully removes the old ``McpSessionContext``
    and a subsequent ``get_or_create_session`` with the same session ID creates
    a brand-new context with fresh resource objects (different toolset_cache,
    different connection_pool, and a ``None`` snapshot).
    """
    manager = MCPManager(name="test")
    session_id = "test-recreate-1"

    try:
        # 1. Create initial session context
        original_ctx = manager.get_or_create_session(session_id)
        assert session_id in manager._session_contexts

        # 2. Store original resource references
        original_toolset_cache = original_ctx.toolset_cache
        original_connection_pool = original_ctx.connection_pool
        assert original_connection_pool is not None

        # 3. Set a snapshot so we can verify the new context does not carry it
        snapshot = McpConfigSnapshot()
        manager.update_session_snapshot(session_id, snapshot)
        assert original_ctx.snapshot is snapshot

        # 4. Close the session
        await manager.cleanup_session(session_id)
        assert session_id not in manager._session_contexts

        # 5. Recreate with the same session ID
        new_ctx = manager.get_or_create_session(session_id)
        assert session_id in manager._session_contexts

        # 6. The new McpSessionContext is a different object
        assert new_ctx is not original_ctx

        # 7. Fresh toolset_cache — different object, empty
        assert new_ctx.toolset_cache is not original_toolset_cache
        assert new_ctx.toolset_cache == {}

        # 8. Fresh connection_pool — different object
        assert new_ctx.connection_pool is not original_connection_pool
        assert new_ctx.connection_pool is not None

        # 9. Snapshot is None (fresh, not carrying old snapshot)
        assert new_ctx.snapshot is None
    finally:
        await manager.cleanup_session(session_id)
        await manager.cleanup()
