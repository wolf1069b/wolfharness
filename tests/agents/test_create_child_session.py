"""Tests for AgentContext.create_child_session() convenience API.

# TODO: L2 migration — test uses mock_pool as both agent_pool and
# host_context on mock_node, requires significant rework for real pool.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from wolfharness.agents.context import AgentContext
from wolfharness.sessions import SessionData
from wolfharness_storage.memory_provider.provider import MemoryStorageProvider


pytestmark = pytest.mark.integration


@pytest.fixture
def mock_node() -> MagicMock:
    """Create a mock MessageNode with session_id."""
    node = MagicMock()
    node.name = "parent_agent"
    node.session_id = "ses_parent_abc123"
    # Real nodes store session_id on _events (EventManager); mocks
    # must mirror this so create_child_session can read the parent.
    node._events.session_id = "ses_parent_abc123"
    node.agent_pool = None  # Will be set per-test
    return node


def _make_mock_session_pool(store: MemoryStorageProvider) -> MagicMock:
    """Create a mock session_pool that persists via MemoryStorageProvider."""
    session_pool = MagicMock()

    async def mock_create_session(
        *,
        session_id: str,
        agent_name: str,
        parent_session_id: str | None = None,
        agent_type: str = "native",
        **kwargs: object,
    ) -> MagicMock:
        parent_data = None
        if parent_session_id:
            parent_data = await store.load_session(parent_session_id)
        session_data = SessionData(
            session_id=session_id,
            agent_name=agent_name,
            parent_id=parent_session_id,
            agent_type=agent_type,
            project_id=parent_data.project_id if parent_data else None,
            cwd=parent_data.cwd if parent_data else None,
        )
        await store.save_session(session_data)
        return MagicMock(session_id=session_id)

    async def mock_create_child_session(
        parent_session_id: str,
        agent_name: str,
        agent_type: str = "native",
        *,
        session_id: str | None = None,
        **kwargs: object,
    ) -> MagicMock:
        from wolfharness.utils.identifiers import generate_session_id

        child_sid = session_id or generate_session_id()
        parent_data = None
        if parent_session_id:
            parent_data = await store.load_session(parent_session_id)
        session_data = SessionData(
            session_id=child_sid,
            agent_name=agent_name,
            parent_id=parent_session_id,
            agent_type=agent_type,
            project_id=parent_data.project_id if parent_data else None,
            cwd=parent_data.cwd if parent_data else None,
        )
        await store.save_session(session_data)
        return MagicMock(session_id=child_sid)

    session_pool.create_session = mock_create_session
    session_pool.create_child_session = mock_create_child_session
    # get_or_create_session_agent is async; must use AsyncMock so await works
    session_pool.sessions.get_or_create_session_agent = AsyncMock()
    return session_pool


async def test_create_child_session_with_pool(mock_node: MagicMock) -> None:
    """When pool is available, create_child_session delegates to session_pool."""
    store = MemoryStorageProvider()
    mock_pool = MagicMock()
    mock_pool.manifest.name = "test_pool"
    mock_pool.session_pool = _make_mock_session_pool(store)

    mock_node.agent_pool = mock_pool
    # host_context is accessed instead of agent_pool after M2 migration
    mock_node.host_context = mock_pool

    # Persist the parent session so the manager can inherit project_id/cwd
    parent = SessionData(
        session_id="ses_parent_abc123",
        agent_name="coordinator",
        project_id="proj_42",
        cwd="/home/user/project",
    )

    ctx = AgentContext(node=mock_node)

    await store.save_session(parent)
    child_id = await ctx.create_child_session(
        agent_name="coder",
        agent_type="native",
    )

    # Verify child was persisted with correct fields
    child = await store.load_session(child_id)
    assert child is not None
    assert child.parent_id == "ses_parent_abc123"
    assert child.agent_name == "coder"
    assert child.agent_type == "native"
    assert child.project_id == "proj_42"
    assert child.cwd == "/home/user/project"


async def test_create_child_session_with_explicit_parent(mock_node: MagicMock) -> None:
    """When parent_session_id is provided explicitly, it overrides node.session_id."""
    store = MemoryStorageProvider()
    mock_pool = MagicMock()
    mock_pool.manifest.name = "test_pool"
    mock_pool.session_pool = _make_mock_session_pool(store)

    mock_node.agent_pool = mock_pool
    # host_context is accessed instead of agent_pool after M2 migration
    mock_node.host_context = mock_pool

    # Persist a different parent
    other_parent = SessionData(
        session_id="ses_other_parent",
        agent_name="router",
        project_id="proj_99",
        cwd="/tmp/workspace",
    )

    ctx = AgentContext(node=mock_node)

    await store.save_session(other_parent)
    child_id = await ctx.create_child_session(
        agent_name="analyst",
        agent_type="acp",
        parent_session_id="ses_other_parent",
    )

    child = await store.load_session(child_id)
    assert child is not None
    assert child.parent_id == "ses_other_parent"
    assert child.agent_name == "analyst"
    assert child.agent_type == "acp"
    assert child.project_id == "proj_99"
    assert child.cwd == "/tmp/workspace"


async def test_create_child_session_no_pool(mock_node: MagicMock) -> None:
    """When no pool is available, create_child_session falls back to generate_session_id."""
    mock_node.agent_pool = None
    # host_context is accessed instead of agent_pool after M2 migration
    mock_node.host_context = None

    ctx = AgentContext(node=mock_node)
    child_id = await ctx.create_child_session(
        agent_name="coder",
        agent_type="native",
    )

    # Should return a non-empty generated ID without persistence
    assert child_id is not None
    assert len(child_id) > 0
    assert child_id.startswith("ses_")


async def test_create_child_session_no_node_session_id(mock_node: MagicMock) -> None:
    """When node has no session_id and no explicit parent, fallback to generate_session_id."""
    mock_node.session_id = None
    mock_node._events.session_id = None
    store = MemoryStorageProvider()
    mock_pool = MagicMock()
    mock_pool.manifest.name = "test_pool"
    mock_pool.session_pool = _make_mock_session_pool(store)

    mock_node.agent_pool = mock_pool
    # host_context is accessed instead of agent_pool after M2 migration
    mock_node.host_context = mock_pool

    ctx = AgentContext(node=mock_node)
    child_id = await ctx.create_child_session(
        agent_name="coder",
        agent_type="native",
    )

    # With no effective parent (node.session_id is None), the method
    # falls back to generate_session_id() since create_session
    # requires a non-None parent.
    assert child_id is not None
    assert len(child_id) > 0
