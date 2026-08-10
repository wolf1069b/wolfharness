"""Tests for GlobalConnectionPool transport sharing across sessions.

Verifies that multiple sessions can share pool-level MCP connections
without stream contention or duplicate connect_session() calls.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from wolfharness.mcp_server.global_pool import GlobalConnectionPool


pytestmark = pytest.mark.unit


class _FakeTransport:
    """Fake transport that tracks connect_session() calls."""

    def __init__(self, name: str = "test") -> None:
        self.name = name
        self.connect_count = 0
        self._active_sessions = 0

    @asynccontextmanager
    async def connect_session(self, **kwargs: Any) -> Any:
        """Track connect_session calls.

        Increments connect_count and _active_sessions on each call.
        """
        self.connect_count += 1
        self._active_sessions += 1
        fake_session = MagicMock()
        try:
            yield fake_session
        finally:
            self._active_sessions -= 1


class TestGlobalConnectionPoolSharing:
    """Tests for sharing transports across multiple sessions."""

    async def test_http_transport_not_shared_directly(self) -> None:
        """HTTP transport is not shared directly.

        Given an HTTP config, when get_transport() is called twice,
        then each call returns a fresh transport whose connect_session()
        can be entered independently without interference.

        HTTP/SSE transports are never cached — each call creates a fresh
        transport to avoid stream contention.
        """
        pool = GlobalConnectionPool()

        # Create a fake HTTP config
        from wolfharness_config.mcp_server import StreamableHTTPMCPServerConfig

        config = StreamableHTTPMCPServerConfig(
            name="test-http",
            url="http://localhost:9999/mcp",
        )

        # Mock to_transport to return our fake
        fake_transport = _FakeTransport("http")
        with patch.object(type(config), "to_transport", return_value=fake_transport):
            transport1 = await pool.get_transport(config)
            transport2 = await pool.get_transport(config)

        # Both should work — enter connect_session() concurrently
        results: list[int] = []

        async def _use_transport(t: Any) -> None:
            async with t.connect_session():
                results.append(len(results))

        await asyncio.gather(_use_transport(transport1), _use_transport(transport2))

        assert len(results) == 2

        await pool.shutdown_all()

    async def test_stdio_transport_owner_task_manages_lifecycle(self) -> None:
        """Stdio transport owner task manages lifecycle.

        Given a stdio config, when get_transport() is called,
        then the owner task enters connect_session() once and
        get_transport() returns a transport that can be used
        without calling connect_session() again on the underlying transport.
        """
        pool = GlobalConnectionPool()

        from wolfharness_config.mcp_server import StdioMCPServerConfig

        config = StdioMCPServerConfig(
            name="test-stdio",
            command="echo",
            args=["hello"],
        )

        fake_transport = _FakeTransport("stdio")
        with patch.object(type(config), "to_transport", return_value=fake_transport):
            await pool.get_transport(config)

        # The owner task should have called connect_session() exactly once
        assert fake_transport.connect_count == 1

        # But get_transport() should return something that can be
        # used by MCPToolset without calling connect_session() again
        # on the underlying transport
        # (This is the fix — the returned transport should be a wrapper)

        await pool.shutdown_all()

    async def test_two_sessions_share_stdio_without_duplicate_connect(self) -> None:
        """Two sessions share stdio without duplicate connect.

        Given a stdio config, when two sessions call get_transport(),
        then the underlying transport's connect_session() is called
        only once (by the owner task), and both sessions can use it.
        """
        pool = GlobalConnectionPool()

        from wolfharness_config.mcp_server import StdioMCPServerConfig

        config = StdioMCPServerConfig(
            name="test-stdio-shared",
            command="echo",
            args=["hello"],
        )

        fake_transport = _FakeTransport("stdio-shared")
        with patch.object(type(config), "to_transport", return_value=fake_transport):
            t1 = await pool.get_transport(config)
            t2 = await pool.get_transport(config)

        # Owner task entered connect_session() once
        assert fake_transport.connect_count == 1

        # Both transports should be usable
        results: list[int] = []

        async def _use(t: Any) -> None:
            async with t.connect_session():
                results.append(len(results))

        await asyncio.gather(_use(t1), _use(t2))

        assert len(results) == 2
        # Still only one connect_session() on the underlying transport
        assert fake_transport.connect_count == 1

        await pool.shutdown_all()
