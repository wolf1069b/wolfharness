"""Unit tests for SessionConnectionPool."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

from pydantic import HttpUrl
import pytest

from wolfharness.mcp_server.session_pool import (
    SessionConnectionPool,
    _create_transport,
    _stdio_owner_task,
)
from wolfharness_config.mcp_server import (
    AcpMCPServerConfig,
    SSEMCPServerConfig,
    StdioMCPServerConfig,
    StreamableHTTPMCPServerConfig,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeTransport:
    """Fake ClientTransport for testing."""

    def __init__(self, label: str = "fake") -> None:
        self.label = label

    @asynccontextmanager
    async def connect_session(self):
        """Fake connect_session that yields immediately."""
        yield


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stdio_config() -> StdioMCPServerConfig:
    return StdioMCPServerConfig(name="stdio-srv", command="python", args=["-m", "srv"])


@pytest.fixture
def sse_config() -> SSEMCPServerConfig:
    return SSEMCPServerConfig(name="sse-srv", url=HttpUrl("http://localhost:8080/sse"))


@pytest.fixture
def http_config() -> StreamableHTTPMCPServerConfig:
    return StreamableHTTPMCPServerConfig(
        name="http-srv", url=HttpUrl("https://api.example.com/mcp")
    )


@pytest.fixture
def acp_config() -> AcpMCPServerConfig:
    return AcpMCPServerConfig(name="acp-srv", acp_id="server-123")


@pytest.fixture
def session_pool() -> SessionConnectionPool:
    return SessionConnectionPool(session_id="test-session-001")


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_session_pool_instantiation(
    session_pool: SessionConnectionPool,
) -> None:
    """SessionConnectionPool stores session_id and starts empty."""
    assert session_pool._session_id == "test-session-001"
    assert len(session_pool._connections) == 0


# ---------------------------------------------------------------------------
# get_transport — caching by (client_id, skill_name)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_get_transport_sse_caches(
    session_pool: SessionConnectionPool,
    sse_config: SSEMCPServerConfig,
) -> None:
    """get_transport creates and caches transport for SSE config."""
    fake = _FakeTransport("sse")
    with patch("wolfharness.mcp_server.session_pool._create_transport", return_value=fake):
        t1 = await session_pool.get_transport(sse_config)
        t2 = await session_pool.get_transport(sse_config)

    assert t1 is t2 is fake
    key = (sse_config.client_id, None)
    assert key in session_pool._connections


@pytest.mark.unit
async def test_get_transport_http_caches(
    session_pool: SessionConnectionPool,
    http_config: StreamableHTTPMCPServerConfig,
) -> None:
    """get_transport creates and caches transport for HTTP config."""
    fake = _FakeTransport("http")
    with patch("wolfharness.mcp_server.session_pool._create_transport", return_value=fake):
        t1 = await session_pool.get_transport(http_config)
        t2 = await session_pool.get_transport(http_config)

    assert t1 is t2 is fake


@pytest.mark.unit
async def test_get_transport_stdio(
    session_pool: SessionConnectionPool,
    stdio_config: StdioMCPServerConfig,
) -> None:
    """get_transport for stdio spawns an owner task and waits for ready."""
    fake = _FakeTransport("stdio")
    with patch("wolfharness.mcp_server.session_pool._create_transport", return_value=fake):
        transport = await session_pool.get_transport(stdio_config)

    assert transport is fake
    key = (stdio_config.client_id, None)
    conn = session_pool._connections[key]
    assert conn.is_stdio is True
    assert conn.owner_task is not None
    assert conn.ready_event is not None
    assert conn.ready_event.is_set()
    assert conn.close_event is not None
    assert conn.done_event is not None

    # Clean up owner task
    await session_pool.cleanup()


# ---------------------------------------------------------------------------
# get_transport — skill_name isolation
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_different_skill_names_get_different_transports(
    session_pool: SessionConnectionPool,
    sse_config: SSEMCPServerConfig,
) -> None:
    """Different skill_name values produce different cache entries for same client_id."""
    fake1 = _FakeTransport("skill-a")
    fake2 = _FakeTransport("skill-b")

    call_count = 0

    def _create_fake(config: Any) -> _FakeTransport:
        nonlocal call_count
        call_count += 1
        return fake1 if call_count == 1 else fake2

    with patch(
        "wolfharness.mcp_server.session_pool._create_transport",
        side_effect=_create_fake,
    ):
        t1 = await session_pool.get_transport(sse_config, skill_name="skill-a")
        t2 = await session_pool.get_transport(sse_config, skill_name="skill-b")

    assert t1 is fake1
    assert t2 is fake2
    assert t1 is not t2

    # Two separate cache entries
    assert (sse_config.client_id, "skill-a") in session_pool._connections
    assert (sse_config.client_id, "skill-b") in session_pool._connections
    assert len(session_pool._connections) == 2


@pytest.mark.unit
async def test_same_skill_name_shares_transport(
    session_pool: SessionConnectionPool,
    sse_config: SSEMCPServerConfig,
) -> None:
    """Same (client_id, skill_name) pair returns cached transport."""
    fake = _FakeTransport("shared")
    with patch("wolfharness.mcp_server.session_pool._create_transport", return_value=fake):
        t1 = await session_pool.get_transport(sse_config, skill_name="my-skill")
        t2 = await session_pool.get_transport(sse_config, skill_name="my-skill")

    assert t1 is t2 is fake
    assert len(session_pool._connections) == 1


# ---------------------------------------------------------------------------
# add_transport()
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_add_transport_stores_pre_created(
    session_pool: SessionConnectionPool,
) -> None:
    """add_transport stores a pre-created transport without owner task."""
    fake = _FakeTransport("pre-created")

    await session_pool.add_transport(
        client_id="custom-client",
        transport=fake,
        skill_name="my-skill",
    )

    key = ("custom-client", "my-skill")
    conn = session_pool._connections[key]
    assert conn.transport is fake
    assert conn.is_pre_created is True
    assert conn.is_stdio is False
    assert conn.owner_task is None


@pytest.mark.unit
async def test_add_transport_without_skill_name(
    session_pool: SessionConnectionPool,
) -> None:
    """add_transport works with skill_name=None."""
    fake = _FakeTransport("no-skill")

    await session_pool.add_transport(
        client_id="custom-client",
        transport=fake,
    )

    key = ("custom-client", None)
    assert key in session_pool._connections
    assert session_pool._connections[key].transport is fake


@pytest.mark.unit
async def test_add_transport_overwrites_existing(
    session_pool: SessionConnectionPool,
) -> None:
    """add_transport overwrites an existing entry with the same key."""
    fake1 = _FakeTransport("first")
    fake2 = _FakeTransport("second")

    await session_pool.add_transport(client_id="c1", transport=fake1)
    await session_pool.add_transport(client_id="c1", transport=fake2)

    key = ("c1", None)
    assert session_pool._connections[key].transport is fake2


# ---------------------------------------------------------------------------
# cleanup()
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_cleanup_empty_pool(session_pool: SessionConnectionPool) -> None:
    """Cleanup on empty pool is a no-op."""
    await session_pool.cleanup()
    assert len(session_pool._connections) == 0


@pytest.mark.unit
async def test_cleanup_clears_all_connections(
    session_pool: SessionConnectionPool,
    sse_config: SSEMCPServerConfig,
    http_config: StreamableHTTPMCPServerConfig,
) -> None:
    """Cleanup removes all connections from the pool."""
    fake1 = _FakeTransport("sse")
    fake2 = _FakeTransport("http")

    call_count = 0

    def _create_fake(config: Any) -> _FakeTransport:
        nonlocal call_count
        call_count += 1
        return fake1 if call_count == 1 else fake2

    with patch(
        "wolfharness.mcp_server.session_pool._create_transport",
        side_effect=_create_fake,
    ):
        await session_pool.get_transport(sse_config)
        await session_pool.get_transport(http_config)

    assert len(session_pool._connections) == 2

    await session_pool.cleanup()

    assert len(session_pool._connections) == 0


@pytest.mark.unit
async def test_cleanup_signals_stdio_owner_tasks(
    session_pool: SessionConnectionPool,
    stdio_config: StdioMCPServerConfig,
) -> None:
    """Cleanup signals close_event for stdio owner tasks and waits for completion."""
    fake = _FakeTransport("stdio")
    with patch("wolfharness.mcp_server.session_pool._create_transport", return_value=fake):
        await session_pool.get_transport(stdio_config)

    key = (stdio_config.client_id, None)
    conn = session_pool._connections[key]
    owner_task = conn.owner_task
    assert owner_task is not None

    await session_pool.cleanup()

    assert owner_task.done()
    assert len(session_pool._connections) == 0


@pytest.mark.unit
async def test_cleanup_idempotent(
    session_pool: SessionConnectionPool,
    sse_config: SSEMCPServerConfig,
) -> None:
    """Cleanup can be called multiple times safely."""
    fake = _FakeTransport("sse")
    with patch("wolfharness.mcp_server.session_pool._create_transport", return_value=fake):
        await session_pool.get_transport(sse_config)

    await session_pool.cleanup()
    await session_pool.cleanup()  # should not raise

    assert len(session_pool._connections) == 0


@pytest.mark.unit
async def test_cleanup_with_mixed_transports(
    session_pool: SessionConnectionPool,
    stdio_config: StdioMCPServerConfig,
    sse_config: SSEMCPServerConfig,
) -> None:
    """Cleanup handles mixed stdio + SSE connections."""
    fake1 = _FakeTransport("stdio")
    fake2 = _FakeTransport("sse")

    call_count = 0

    def _create_fake(config: Any) -> _FakeTransport:
        nonlocal call_count
        call_count += 1
        return fake1 if call_count == 1 else fake2

    with patch(
        "wolfharness.mcp_server.session_pool._create_transport",
        side_effect=_create_fake,
    ):
        await session_pool.get_transport(stdio_config)
        await session_pool.get_transport(sse_config)

    assert len(session_pool._connections) == 2

    await session_pool.cleanup()

    assert len(session_pool._connections) == 0


# ---------------------------------------------------------------------------
# _create_transport helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_create_transport_acp_raises(acp_config: AcpMCPServerConfig) -> None:
    """_create_transport raises NotImplementedError for ACP configs."""
    with pytest.raises(NotImplementedError, match="add_transport"):
        _create_transport(acp_config)


# ---------------------------------------------------------------------------
# _stdio_owner_task
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_stdio_owner_task_sets_ready_and_waits() -> None:
    """_stdio_owner_task sets ready_event immediately and waits for close_event."""
    import asyncio

    ready_event = asyncio.Event()
    close_event = asyncio.Event()
    done_event = asyncio.Event()
    fake_transport = _FakeTransport("stdio")

    task = asyncio.create_task(
        _stdio_owner_task(
            transport=fake_transport,
            ready_event=ready_event,
            close_event=close_event,
            done_event=done_event,
        )
    )

    # Ready should be set immediately
    await asyncio.sleep(0.01)
    assert ready_event.is_set()
    assert not done_event.is_set()

    # Signal close
    close_event.set()

    await task

    assert done_event.is_set()


# ---------------------------------------------------------------------------
# copy_pre_created_transports
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_copy_pre_created_transports_copies_acp_only(
    session_pool: SessionConnectionPool,
) -> None:
    """copy_pre_created_transports copies only pre-created (ACP) transports."""
    source = SessionConnectionPool("source-session")
    acp_transport = _FakeTransport("acp")
    http_transport = _FakeTransport("http")

    await source.add_transport("acp-1", acp_transport)
    # Simulate a non-pre-created transport by inserting directly
    from wolfharness.mcp_server.session_pool import _SessionConnection

    source._connections[("http-1", None)] = _SessionConnection(
        transport=http_transport,
        is_stdio=False,
        is_pre_created=False,
    )

    await session_pool.copy_pre_created_transports(source)

    assert ("acp-1", None) in session_pool._connections
    assert ("http-1", None) not in session_pool._connections
    assert session_pool._connections[("acp-1", None)].transport is acp_transport
    assert session_pool._connections[("acp-1", None)].is_pre_created is True


@pytest.mark.unit
async def test_copy_pre_created_transports_does_not_overwrite(
    session_pool: SessionConnectionPool,
) -> None:
    """copy_pre_created_transports does not overwrite existing transports."""
    source = SessionConnectionPool("source")
    child_transport = _FakeTransport("child-acp")
    parent_transport = _FakeTransport("parent-acp")

    await session_pool.add_transport("acp-1", child_transport)
    await source.add_transport("acp-1", parent_transport)

    await session_pool.copy_pre_created_transports(source)

    # Child's transport should be preserved
    assert session_pool._connections[("acp-1", None)].transport is child_transport


@pytest.mark.unit
async def test_copy_pre_created_transports_empty_source(
    session_pool: SessionConnectionPool,
) -> None:
    """copy_pre_created_transports with empty source is a no-op."""
    source = SessionConnectionPool("empty-source")

    await session_pool.copy_pre_created_transports(source)

    assert len(session_pool._connections) == 0


@pytest.mark.unit
async def test_copy_pre_created_transports_preserves_skill_name(
    session_pool: SessionConnectionPool,
) -> None:
    """copy_pre_created_transports preserves skill_name in the key."""
    source = SessionConnectionPool("source")
    acp_transport = _FakeTransport("skill-acp")

    await source.add_transport("acp-1", acp_transport, skill_name="my-skill")

    await session_pool.copy_pre_created_transports(source)

    assert ("acp-1", "my-skill") in session_pool._connections
