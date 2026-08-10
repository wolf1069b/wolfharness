"""Tests for _inject_pool_providers behavior.

Verifies that:
- _inject_pool_providers() no longer injects skills_tools_provider
  (consolidated into SkillManagerCap, unify-skill-loading).
- _inject_pool_providers() with include_aggregating=True injects the MCP
  aggregating provider.
- _inject_pool_providers() with pool=None returns early without error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from wolfharness.host.factory import _inject_pool_providers


if TYPE_CHECKING:
    from wolfharness import AgentPool


pytestmark = pytest.mark.unit


class FakeAgent:
    """Minimal agent stub for testing _inject_pool_providers."""

    def __init__(self) -> None:
        self._external_capabilities: list[Any] = []


class FakeHostContext:
    """Minimal host context stub for testing."""

    def __init__(
        self,
        mcp_aggregating_provider: Any | None = None,
    ) -> None:
        self._mcp_aggregating = mcp_aggregating_provider

        class FakeMcp:
            def get_aggregating_provider(self) -> Any:
                return mcp_aggregating_provider

        self.mcp = FakeMcp()


def test_inject_pool_providers_no_skills_injection(minimal_pool: AgentPool) -> None:
    """Skills tools are NOT injected (owned by SkillManagerCap now).

    Given a host context, When _inject_pool_providers is called with
    include_aggregating=False, Then no providers are injected.
    """
    agent = FakeAgent()
    host_context = FakeHostContext()
    pool = minimal_pool

    _inject_pool_providers(agent, host_context, pool, include_aggregating=False)

    assert len(agent._external_capabilities) == 0


def test_inject_pool_providers_pool_none_returns_early() -> None:
    """No injection when pool is None."""
    agent = FakeAgent()
    host_context = FakeHostContext()

    _inject_pool_providers(agent, host_context, None, include_aggregating=False)

    assert len(agent._external_capabilities) == 0


def test_inject_pool_providers_includes_mcp_aggregating(minimal_pool: AgentPool) -> None:
    """MCP aggregating provider injected when include_aggregating=True."""
    mcp_provider = MagicMock(name="mcp_aggregating_provider")
    agent = FakeAgent()
    host_context = FakeHostContext(mcp_aggregating_provider=mcp_provider)
    pool = minimal_pool

    _inject_pool_providers(agent, host_context, pool, include_aggregating=True)

    assert mcp_provider in agent._external_capabilities
