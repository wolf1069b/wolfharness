"""Unit tests for GlobalConnectionPool."""

from __future__ import annotations

from typing import Self
from unittest.mock import patch

from pydantic import HttpUrl
import pytest

from wolfharness.mcp_server.global_pool import GlobalConnectionPool
from wolfharness_config.mcp_server import (
    AcpMCPServerConfig,
    SSEMCPServerConfig,
    StdioMCPServerConfig,
    StreamableHTTPMCPServerConfig,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeConnectSession:
    """Fake async context manager for transport.connect_session()."""

    def __init__(self) -> None:
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> Self:
        self.entered = True
        return self

    async def __aexit__(self, *args: object) -> None:
        self.exited = True


class _FakeTransport:
    """Fake ClientTransport that does not start real servers."""

    def __init__(self) -> None:
        self._session = _FakeConnectSession()

    def connect_session(self) -> _FakeConnectSession:
        return self._session


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
def fake_transport() -> _FakeTransport:
    return _FakeTransport()


@pytest.fixture
def pool() -> GlobalConnectionPool:
    return GlobalConnectionPool()


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pool_instantiation(pool: GlobalConnectionPool) -> None:
    """GlobalConnectionPool can be instantiated with no args."""
    assert len(pool._connections) == 0


# ---------------------------------------------------------------------------
# get_transport — HTTP/SSE (fresh transport per call, no caching)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_get_transport_sse(
    pool: GlobalConnectionPool, sse_config: SSEMCPServerConfig, fake_transport: _FakeTransport
) -> None:
    """SSE config creates a fresh transport per call (not cached)."""
    with patch.object(SSEMCPServerConfig, "to_transport", return_value=fake_transport):
        transport = await pool.get_transport(sse_config)

    assert transport is fake_transport
    # HTTP/SSE connections are NOT cached in _connections
    assert len(pool._connections) == 0


@pytest.mark.unit
async def test_get_transport_http(
    pool: GlobalConnectionPool,
    http_config: StreamableHTTPMCPServerConfig,
    fake_transport: _FakeTransport,
) -> None:
    """StreamableHTTP config creates a fresh transport per call (not cached)."""
    with patch.object(StreamableHTTPMCPServerConfig, "to_transport", return_value=fake_transport):
        transport = await pool.get_transport(http_config)

    assert transport is fake_transport
    assert len(pool._connections) == 0


@pytest.mark.unit
async def test_get_transport_sse_creates_fresh_each_call(
    pool: GlobalConnectionPool,
    sse_config: SSEMCPServerConfig,
) -> None:
    """Multiple get_transport() calls for SSE create independent transports."""
    fake1 = _FakeTransport()
    fake2 = _FakeTransport()
    with patch.object(SSEMCPServerConfig, "to_transport", side_effect=[fake1, fake2]):
        t1 = await pool.get_transport(sse_config)
        t2 = await pool.get_transport(sse_config)

    assert t1 is fake1
    assert t2 is fake2
    assert len(pool._connections) == 0


# ---------------------------------------------------------------------------
# get_transport — stdio (owner task, cached for pool lifetime)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_get_transport_stdio(
    pool: GlobalConnectionPool,
    stdio_config: StdioMCPServerConfig,
    fake_transport: _FakeTransport,
) -> None:
    """Stdio config spawns an owner task and waits for ready_event."""
    with patch.object(StdioMCPServerConfig, "to_transport", return_value=fake_transport):
        transport = await pool.get_transport(stdio_config)

    assert transport is not fake_transport  # Returns _SharedSessionTransport wrapper
    client_id = stdio_config.client_id
    conn = pool._connections[client_id]
    assert conn.owner_task is not None
    assert conn.ready_event.is_set()
    assert conn.shared_session_transport is not None

    # Clean up owner task
    await pool.shutdown_all()


@pytest.mark.unit
async def test_get_transport_stdio_share_connection(
    pool: GlobalConnectionPool,
    stdio_config: StdioMCPServerConfig,
    fake_transport: _FakeTransport,
) -> None:
    """Multiple get_transport() calls for same client_id share connection."""
    with patch.object(StdioMCPServerConfig, "to_transport", return_value=fake_transport):
        t1 = await pool.get_transport(stdio_config)
        t2 = await pool.get_transport(stdio_config)

    assert t1 is t2
    # Only one connection in the pool
    assert len(pool._connections) == 1

    # Clean up
    await pool.shutdown_all()


# ---------------------------------------------------------------------------
# get_transport — ACP raises NotImplementedError
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_get_transport_acp_raises(
    pool: GlobalConnectionPool, acp_config: AcpMCPServerConfig
) -> None:
    """ACP config raises NotImplementedError in GlobalConnectionPool."""
    with pytest.raises(NotImplementedError, match="ACP transport"):
        await pool.get_transport(acp_config)


# ---------------------------------------------------------------------------
# shutdown_all()
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_shutdown_all_clears_connections(
    pool: GlobalConnectionPool,
    stdio_config: StdioMCPServerConfig,
    fake_transport: _FakeTransport,
) -> None:
    """shutdown_all() removes all cached stdio connections."""
    with patch.object(StdioMCPServerConfig, "to_transport", return_value=fake_transport):
        await pool.get_transport(stdio_config)

    assert len(pool._connections) == 1

    await pool.shutdown_all()

    assert len(pool._connections) == 0


@pytest.mark.unit
async def test_shutdown_all_with_stdio(
    pool: GlobalConnectionPool,
    stdio_config: StdioMCPServerConfig,
    fake_transport: _FakeTransport,
) -> None:
    """shutdown_all() signals and waits for stdio owner tasks."""
    with patch.object(StdioMCPServerConfig, "to_transport", return_value=fake_transport):
        await pool.get_transport(stdio_config)

    client_id = stdio_config.client_id
    owner_task = pool._connections[client_id].owner_task
    assert owner_task is not None

    await pool.shutdown_all()

    assert owner_task.done()
    assert len(pool._connections) == 0


@pytest.mark.unit
async def test_shutdown_all_empty_pool(pool: GlobalConnectionPool) -> None:
    """shutdown_all() on empty pool is a no-op."""
    await pool.shutdown_all()
    assert len(pool._connections) == 0


@pytest.mark.unit
async def test_shutdown_all_multiple_stdio(
    pool: GlobalConnectionPool,
    fake_transport: _FakeTransport,
) -> None:
    """shutdown_all() handles multiple stdio connections."""
    configs = [
        StdioMCPServerConfig(name=f"stdio-{i}", command="python", args=["-m", f"srv{i}"])
        for i in range(3)
    ]
    for cfg in configs:
        with patch.object(StdioMCPServerConfig, "to_transport", return_value=fake_transport):
            await pool.get_transport(cfg)

    assert len(pool._connections) == 3

    await pool.shutdown_all()

    assert len(pool._connections) == 0
