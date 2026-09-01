"""Unit tests for ACP server startup error visibility (Fix A).

Covers ``ACPServer._start_async`` serve-loop exception handling:
- With ``raise_exceptions=True`` the exception is re-raised after logging
  (consistent with ``BaseServer.start`` semantics).
- With ``raise_exceptions=False`` the exception is logged and swallowed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from wolfharness import AgentPool
from wolfharness.models.agents import NativeAgentConfig
from wolfharness.models.manifest import AgentsManifest
from wolfharness_server.acp_server import ACPServer


pytestmark = pytest.mark.unit


@pytest.fixture
def acp_server() -> ACPServer:
    """Build an ACPServer with a minimal pool (default ``raise_exceptions=True``)."""
    manifest = AgentsManifest(agents={"test_agent": NativeAgentConfig(model="test")})
    pool = AgentPool(manifest)
    return ACPServer(pool=pool)


async def _resolved_default_agent() -> AsyncMock:
    """Return a mock default agent with an async context manager."""
    agent = AsyncMock()
    agent.name = "test_agent"
    agent.__aenter__ = AsyncMock(return_value=agent)
    agent.__aexit__ = AsyncMock(return_value=None)
    return agent


async def _raise_serve(*args: object, **kwargs: object) -> None:
    """Stand-in for ``acp.serve`` that always fails."""
    raise RuntimeError("serve boom")


async def test_start_async_reraises_when_raise_exceptions(
    acp_server: ACPServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Serve-loop errors propagate when ``raise_exceptions`` is True."""
    monkeypatch.setattr(acp_server, "_resolve_default_agent", _resolved_default_agent)
    monkeypatch.setattr("wolfharness_server.acp_server.server.serve", _raise_serve)

    acp_server.raise_exceptions = True
    with pytest.raises(RuntimeError, match="serve boom"):
        await acp_server._start_async()


async def test_start_async_swallows_when_not_raise_exceptions(
    acp_server: ACPServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Serve-loop errors are logged but not re-raised when ``raise_exceptions`` is False."""
    monkeypatch.setattr(acp_server, "_resolve_default_agent", _resolved_default_agent)
    monkeypatch.setattr("wolfharness_server.acp_server.server.serve", _raise_serve)

    acp_server.raise_exceptions = False
    await acp_server._start_async()
