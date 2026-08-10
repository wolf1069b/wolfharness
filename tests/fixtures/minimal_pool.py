"""Minimal AgentPool fixture for L2 integration tests.

Provides a real AgentPool with TestModel — replaces MagicMock(pool) usage
in L2 tests to surface real integration bugs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yamling

from wolfharness import AgentPool, AgentsManifest


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

MINIMAL_CONFIG = """\
agents:
  test_agent:
    type: native
    model: test
    system_prompt: "You are a test agent."
"""


@pytest.fixture
async def minimal_pool() -> AsyncIterator[AgentPool]:
    """Real AgentPool with TestModel — no MCP, no storage, no external deps."""
    manifest = yamling.load_yaml(MINIMAL_CONFIG, verify_type=dict)
    manifest_obj = AgentsManifest.model_validate(manifest)
    async with AgentPool(manifest_obj) as pool:
        yield pool
