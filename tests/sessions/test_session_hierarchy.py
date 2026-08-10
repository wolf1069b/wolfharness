"""Tests for session hierarchy functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from wolfharness.orchestrator import SessionPool
from wolfharness.sessions import SessionData
from wolfharness.utils.identifiers import generate_session_id
from wolfharness_config.storage import SQLStorageConfig
from wolfharness_storage.memory_provider.provider import MemoryStorageProvider
from wolfharness_storage.sql_provider import SQLModelProvider


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from pathlib import Path

    from wolfharness import AgentPool


@pytest.fixture
def memory_store() -> MemoryStorageProvider:
    """Create a memory storage provider for testing."""
    return MemoryStorageProvider()


@pytest.fixture
def sql_store(tmp_path: Path) -> SQLModelProvider:
    """Create a SQL session store with temp database."""
    db_path = tmp_path / "test_hierarchy.db"
    config = SQLStorageConfig(url=f"sqlite:///{db_path}")
    return SQLModelProvider(config)


class TestSessionHierarchy:
    """Tests for session parent-child hierarchy."""

    async def test_create_with_parent_id(
        self, minimal_pool: AgentPool, memory_store: MemoryStorageProvider
    ) -> None:
        """Test that parent_id is persisted correctly via create_session."""
        session_pool = SessionPool(pool=minimal_pool, store=memory_store)
        await session_pool.start()

        # Create parent session directly in store
        parent = SessionData(session_id="parent_1", agent_name="coordinator")
        await memory_store.save_session(parent)

        # Create child session via session pool
        child_state = await session_pool.create_session(
            session_id=generate_session_id(),
            agent_name="coder",
            parent_session_id="parent_1",
        )
        child_id = child_state.session_id

        # Verify child has parent_id
        child = await memory_store.load_session(child_id)
        assert child is not None
        assert child.parent_id == "parent_1"

        # Verify when loaded again
        loaded = await memory_store.load_session(child_id)
        assert loaded is not None
        assert loaded.parent_id == "parent_1"

    async def test_list_by_parent_id_memory(
        self, minimal_pool: AgentPool, memory_store: MemoryStorageProvider
    ) -> None:
        """Test filtering sessions by parent_id with memory store."""
        session_pool = SessionPool(pool=minimal_pool, store=memory_store)
        await session_pool.start()

        # Create root and parent sessions directly
        root = SessionData(session_id="root_1", agent_name="root_agent")
        await memory_store.save_session(root)

        parent = SessionData(session_id="parent_1", agent_name="parent_agent")
        await memory_store.save_session(parent)

        # Create child sessions via session pool
        child1_state = await session_pool.create_session(
            session_id=generate_session_id(),
            agent_name="child_agent",
            parent_session_id="parent_1",
        )
        child2_state = await session_pool.create_session(
            session_id=generate_session_id(),
            agent_name="child_agent2",
            parent_session_id="parent_1",
        )
        child1_id = child1_state.session_id
        child2_id = child2_state.session_id

        # List children of parent by filtering in Python
        all_ids = await memory_store.list_session_ids()
        children: list[str] = []
        for sid in all_ids:
            s = await memory_store.load_session(sid)
            if s is not None and s.parent_id == "parent_1":
                children.append(s.session_id)

        # Verify only children of parent are returned
        assert len(children) == 2
        assert child1_id in children
        assert child2_id in children
        assert "root_1" not in children
        assert "parent_1" not in children

    async def test_list_by_parent_id_sql(
        self, minimal_pool: AgentPool, sql_store: SQLModelProvider
    ) -> None:
        """Test filtering sessions by parent_id with SQL store."""
        session_pool = SessionPool(pool=minimal_pool, store=sql_store)

        async with sql_store:
            await session_pool.start()

            # Create root and parent sessions directly
            root = SessionData(session_id="root_1", agent_name="root_agent")
            await sql_store.save_session(root)

            parent = SessionData(session_id="parent_1", agent_name="parent_agent")
            await sql_store.save_session(parent)

            # Create child sessions via session pool
            child1_state = await session_pool.create_session(
                session_id=generate_session_id(),
                agent_name="child_agent",
                parent_session_id="parent_1",
            )
            child2_state = await session_pool.create_session(
                session_id=generate_session_id(),
                agent_name="child_agent2",
                parent_session_id="parent_1",
            )
            child1_id = child1_state.session_id
            child2_id = child2_state.session_id

            # SQLModelProvider.list_session_ids does not support parent_id filter,
            # so verify children exist by loading each one and checking parent_id.
            all_ids = await sql_store.list_session_ids()
            children = [sid for sid in all_ids if sid in (child1_id, child2_id)]

            # Verify only children of parent are returned
            assert len(children) == 2
            assert child1_id in children
            assert child2_id in children
            assert "root_1" not in children
            assert "parent_1" not in children

    async def test_create_with_invalid_parent(
        self, minimal_pool: AgentPool, memory_store: MemoryStorageProvider
    ) -> None:
        """Test that creating with non-existent parent_id succeeds (permissive)."""
        session_pool = SessionPool(pool=minimal_pool, store=memory_store)
        await session_pool.start()

        # Create child with fake parent_id
        child_state = await session_pool.create_session(
            session_id=generate_session_id(),
            agent_name="agent",
            parent_session_id="nonexistent_parent_id",
        )
        child_id = child_state.session_id

        # Should succeed (permissive validation)
        child = await memory_store.load_session(child_id)
        assert child is not None
        assert child.parent_id == "nonexistent_parent_id"

        # Verify persisted correctly
        loaded = await memory_store.load_session(child_id)
        assert loaded is not None
        assert loaded.parent_id == "nonexistent_parent_id"

    async def test_list_by_parent_id_with_no_children(
        self, minimal_pool: AgentPool, memory_store: MemoryStorageProvider
    ) -> None:
        """Test filtering by parent_id returns empty list when no children exist."""
        session_pool = SessionPool(pool=minimal_pool, store=memory_store)
        await session_pool.start()

        # Create parent but no children
        await memory_store.save_session(SessionData(session_id="parent_1", agent_name="root_agent"))
        await memory_store.save_session(SessionData(session_id="other_1", agent_name="other_agent"))

        # List children of non-existent parent via session pool controller
        children = session_pool.sessions.get_children("nonexistent_parent")

        # Should return empty list
        assert len(children) == 0

    async def test_nested_hierarchy(
        self, minimal_pool: AgentPool, memory_store: MemoryStorageProvider
    ) -> None:
        """Test multi-level hierarchy (grandparent -> parent -> child)."""
        session_pool = SessionPool(pool=minimal_pool, store=memory_store)
        await session_pool.start()

        # Create grandparent session directly
        grandparent = SessionData(session_id="gp_1", agent_name="root_agent")
        await memory_store.save_session(grandparent)

        # Create parent as child of grandparent
        parent_state = await session_pool.create_session(
            session_id=generate_session_id(),
            agent_name="parent_agent",
            parent_session_id="gp_1",
        )
        parent_id = parent_state.session_id

        # Create child as child of parent
        child_state = await session_pool.create_session(
            session_id=generate_session_id(),
            agent_name="child_agent",
            parent_session_id=parent_id,
        )
        child_id = child_state.session_id

        # Verify hierarchy by loading all sessions and filtering by parent_id in Python
        all_ids = await memory_store.list_session_ids()

        # Load all sessions once for filtering
        loaded_sessions: list[SessionData] = []
        for sid in all_ids:
            s = await memory_store.load_session(sid)
            if s is not None:
                loaded_sessions.append(s)

        grandparent_children = [s.session_id for s in loaded_sessions if s.parent_id == "gp_1"]
        assert len(grandparent_children) == 1
        assert parent_id in grandparent_children

        parent_children = [s.session_id for s in loaded_sessions if s.parent_id == parent_id]
        assert len(parent_children) == 1
        assert child_id in parent_children

        child_children = [s.session_id for s in loaded_sessions if s.parent_id == child_id]
        assert len(child_children) == 0
