"""Unit tests for MCPManager session context lifecycle.

Covers ``McpSessionContext`` creation, snapshot storage, ACP transport
registration, and cleanup (including idempotency and concurrency safety).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import pytest

from wolfharness.mcp_server.config_snapshot import McpConfigSnapshot
from wolfharness.mcp_server.manager import MCPManager, McpSessionContext


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeTransport:
    """Fake ``ClientTransport`` for testing ``add_acp_transport``."""

    def __init__(self, label: str = "fake-acp") -> None:
        self.label = label

    @asynccontextmanager
    async def connect_session(self):
        """Fake ``connect_session`` that yields immediately."""
        yield


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager() -> MCPManager:
    """Provide a fresh ``MCPManager`` instance."""
    return MCPManager()


# ---------------------------------------------------------------------------
# 1. get_or_create_session — idempotent for same session_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_or_create_session_creates_and_returns_same(
    manager: MCPManager,
) -> None:
    """Two calls with the same session_id return the same McpSessionContext."""
    ctx1 = manager.get_or_create_session("sess-1")
    ctx2 = manager.get_or_create_session("sess-1")

    assert ctx1 is ctx2
    assert isinstance(ctx1, McpSessionContext)
    assert manager.get_session_context("sess-1") is not None


# ---------------------------------------------------------------------------
# 2. get_or_create_session — different IDs get different contexts
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_or_create_session_creates_fresh_for_different_ids(
    manager: MCPManager,
) -> None:
    """Different session_ids produce distinct McpSessionContext objects."""
    ctx_a = manager.get_or_create_session("sess-a")
    ctx_b = manager.get_or_create_session("sess-b")

    assert ctx_a is not ctx_b
    assert ctx_a.connection_pool is not ctx_b.connection_pool
    assert ctx_a.toolset_cache is not ctx_b.toolset_cache
    assert manager.get_session_context("sess-a") is not None
    assert manager.get_session_context("sess-b") is not None


# ---------------------------------------------------------------------------
# 3. update_session_snapshot — stores snapshot on the context
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_update_session_snapshot_stores_snapshot(
    manager: MCPManager,
) -> None:
    """update_session_snapshot stores the snapshot on the session context."""
    snapshot = McpConfigSnapshot()
    manager.update_session_snapshot("sess-snap", snapshot)

    ctx = manager.get_or_create_session("sess-snap")
    assert ctx.snapshot is snapshot


# ---------------------------------------------------------------------------
# 4. add_acp_transport — stores transport and (connection_id, session_key)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_add_acp_transport_stores_transport_and_ids(
    manager: MCPManager,
) -> None:
    """add_acp_transport adds transport to connection_pool and records IDs."""
    transport: Any = _FakeTransport("acp-1")

    await manager.add_acp_transport(
        session_id="sess-acp",
        client_id="acp-client-1",
        transport=transport,
        connection_id="conn-1",
        session_key=42,
    )

    ctx = manager.get_or_create_session("sess-acp")

    # Transport should be in the connection pool
    assert ctx.connection_pool is not None
    key = ("acp-client-1", None)
    assert key in ctx.connection_pool._connections
    assert ctx.connection_pool._connections[key].transport is transport

    # (connection_id, session_key) should be recorded
    assert ("conn-1", 42) in ctx.acp_connection_ids


# ---------------------------------------------------------------------------
# 5. cleanup_session — clears all resources
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_cleanup_session_clears_all_resources(
    manager: MCPManager,
) -> None:
    """After cleanup, session_id is NOT in the session context registry."""
    manager.get_or_create_session("sess-clean")
    assert manager.get_session_context("sess-clean") is not None

    await manager.cleanup_session("sess-clean")

    assert manager.get_session_context("sess-clean") is None


# ---------------------------------------------------------------------------
# 6. cleanup_session — idempotent (double-call is no-op)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_cleanup_session_is_idempotent(
    manager: MCPManager,
) -> None:
    """Double-call to cleanup_session does not raise."""
    manager.get_or_create_session("sess-idem")

    await manager.cleanup_session("sess-idem")
    # Second call should be a no-op, not raise
    await manager.cleanup_session("sess-idem")

    assert manager.get_session_context("sess-idem") is None


# ---------------------------------------------------------------------------
# 7. concurrent cleanup_session — no error with asyncio.gather
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_concurrent_cleanup_session_no_error(
    manager: MCPManager,
) -> None:
    """asyncio.gather of two concurrent cleanup calls completes without error."""
    manager.get_or_create_session("sess-concurrent")

    results = await asyncio.gather(
        manager.cleanup_session("sess-concurrent"),
        manager.cleanup_session("sess-concurrent"),
        return_exceptions=True,
    )

    for result in results:
        assert not isinstance(result, Exception), f"Concurrent cleanup raised: {result!r}"

    assert manager.get_session_context("sess-concurrent") is None


# ---------------------------------------------------------------------------
# 8. concurrent cleanup from two paths (WebSocket disconnect + SessionController)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_concurrent_cleanup_from_two_paths(
    manager: MCPManager,
) -> None:
    """Two concurrent cleanup_session calls with active resources complete safely.

    Simulates WebSocket disconnect and SessionController.close_session firing
    cleanup_session() simultaneously on a session that has a snapshot, a
    transport in the connection pool, and recorded ACP connection IDs.
    """
    session_id = "test-two-paths"
    manager.get_or_create_session(session_id)

    # Populate session context with resources as if a real session ran
    snapshot = McpConfigSnapshot()
    manager.update_session_snapshot(session_id, snapshot)

    transport: Any = _FakeTransport("acp-two-paths")
    await manager.add_acp_transport(
        session_id=session_id,
        client_id="client-two-paths",
        transport=transport,
        connection_id="conn-two-paths",
        session_key=1,
    )

    ctx = manager.get_or_create_session(session_id)
    assert ctx.snapshot is snapshot
    assert len(ctx.acp_connection_ids) == 1

    # Simulate both WebSocket disconnect and SessionController firing
    # cleanup_session() at the same time.
    results = await asyncio.gather(
        manager.cleanup_session(session_id),
        manager.cleanup_session(session_id),
        return_exceptions=True,
    )

    for result in results:
        assert not isinstance(result, Exception), (
            f"Concurrent cleanup from two paths raised: {result!r}"
        )

    assert manager.get_session_context(session_id) is None
