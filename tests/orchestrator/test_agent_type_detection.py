"""Tests for agent type detection using AGENT_TYPE ClassVar instead of metadata.

The agent_type on RunHandle should come from ``agent.AGENT_TYPE`` (the ClassVar)
rather than ``session.metadata["agent_type"]`` which may be missing or stale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from wolfharness import AgentPool
    from wolfharness.orchestrator.core import SessionController


pytestmark = pytest.mark.unit


@pytest.fixture
def controller(minimal_pool: AgentPool) -> SessionController:
    """Return a SessionController backed by the real pool."""
    assert minimal_pool.session_pool is not None
    return minimal_pool.session_pool.sessions
