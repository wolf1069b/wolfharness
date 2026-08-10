import asyncio
from typing import Any

from pydantic_ai.models.test import TestModel
import pytest
from sqlalchemy import select

from wolfharness import Agent, AgentPool, AgentsManifest, NativeAgentConfig
from wolfharness.agents.events import RunStartedEvent
from wolfharness_config.storage import SQLStorageConfig, StorageConfig
from wolfharness_storage.sql_provider import SQLModelProvider
from wolfharness_storage.sql_provider.models import Conversation
from wolfharness_toolsets.builtin.subagent_tools import SubagentTools


pytestmark = pytest.mark.integration


def _stream_empty(queue: asyncio.Queue[Any]) -> bool:
    """Check if a subscriber queue has no buffered items."""
    return queue.empty()


@pytest.fixture
async def sql_provider(tmp_path):
    """Create SQLModelProvider instance with file-based SQLite."""
    db_path = tmp_path / "test.db"
    # Use auto_migration=False to ensure create_all uses current models
    config = SQLStorageConfig(url=f"sqlite+aiosqlite:///{db_path}", auto_migration=False)
    async with SQLModelProvider(config) as p:
        yield p


@pytest.fixture
async def test_pool(sql_provider):
    """Create a pool with two agents and SQL storage."""
    # We pass the provider config to the manifest
    manifest = AgentsManifest(
        agents={
            "parent": NativeAgentConfig(name="parent", model="test"),
            "child": NativeAgentConfig(name="child", model="test"),
        },
        storage=StorageConfig(providers=[sql_provider.config]),
    )
    async with AgentPool(manifest) as pool:
        # Register subagent tools on parent
        parent: Any = pool.manifest.agents["parent"].get_agent(pool=pool)
        assert isinstance(parent, Agent)
        parent._add_capability(SubagentTools())

        # Mock models for both
        await parent.set_model(TestModel())
        child: Any = pool.manifest.agents["child"].get_agent(pool=pool)
        assert isinstance(child, Agent)
        await child.set_model(TestModel(custom_output_text="Child response"))

        yield pool


@pytest.mark.asyncio
async def test_run_started_event_lineage(test_pool):
    """Test that RunStartedEvent contains parent_session_id."""
    child = test_pool.manifest.agents["child"].get_agent(pool=test_pool)
    parent_session_id = "parent-123"

    events = [
        event async for event in child.run_stream("hello", parent_session_id=parent_session_id)
    ]

    run_started = next(e for e in events if isinstance(e, RunStartedEvent))
    assert run_started.parent_session_id == parent_session_id
    assert run_started.session_id  # should be a non-empty string


@pytest.mark.asyncio
async def test_subagent_event_lineage(test_pool):
    """Test that child session events are receivable via scope=descendants."""
    pool = test_pool
    parent = pool.manifest.agents["parent"].get_agent(pool=pool)

    parent_session_id = "parent-456"
    assert pool.session_pool is not None

    # Subscribe to parent with descendants scope to catch child events
    queue = await pool.session_pool.event_bus.subscribe(parent_session_id, scope="descendants")

    # Run parent which will delegate to child via task tool
    ctx = parent.get_context()
    tools = SubagentTools()
    parent.session_id = parent_session_id

    await tools.task(ctx, agent_or_team="child", prompt="Do something", description="test lineage")

    # Collect events from the queue — raw events flow through EventBus
    # with scope=descendants (no SubAgentEvent wrapping in session path).
    child_events: list[Any] = []
    await asyncio.sleep(0.1)  # Give events time to propagate

    while True:
        try:
            envelope = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        except asyncio.QueueShutDown:
            break
        if envelope is None:
            break
        child_events.append(envelope)

    await pool.session_pool.event_bus.unsubscribe(parent_session_id, queue)

    # Verify that events from the child session were received
    # (SpawnSessionStart, RunStartedEvent, etc.)
    assert len(child_events) > 0, "Expected events from child session via descendants scope"
    # Verify the parent-child lineage in SessionController
    children = pool.session_pool.sessions.get_children(parent_session_id)
    assert len(children) > 0, "Expected child sessions to be registered"
    for child_id in children:
        child_session = pool.session_pool.sessions.get_session(child_id)
        assert child_session is not None
        assert child_session.parent_session_id == parent_session_id


def test_conversation_model_defines_parent_id() -> None:
    """Regression: ORM must expose parent_id so log_session INSERT and DB schema stay aligned."""
    from wolfharness_storage.sql_provider.models import Conversation

    assert "parent_id" in Conversation.model_fields


@pytest.mark.asyncio
async def test_sql_storage_parent_id(test_pool):
    """Test that SQL storage shows correct parent_id for child session."""
    storage_manager = test_pool.storage
    sql_provider = storage_manager.providers[0]
    assert isinstance(sql_provider, SQLModelProvider)

    parent_id = "parent-session-db"
    child_id = "child-session-db"

    # Log parent session
    await sql_provider.log_session(session_id=parent_id, node_name="parent")

    # Log child session with parent_id
    await sql_provider.log_session(
        session_id=child_id, node_name="child", parent_session_id=parent_id
    )

    # Verify in DB
    from sqlalchemy.ext.asyncio import AsyncSession

    async with AsyncSession(sql_provider.engine) as session:
        result = await session.execute(select(Conversation).where(Conversation.id == child_id))  # type: ignore[arg-type]
        convo = result.scalar_one_or_none()
        assert convo is not None
        assert convo.parent_id == parent_id


@pytest.mark.asyncio
async def test_storage_soft_validation(test_pool):
    """Test that soft validation works (no crash if parent missing)."""
    storage_manager = test_pool.storage
    sql_provider = storage_manager.providers[0]
    assert isinstance(sql_provider, SQLModelProvider)

    child_id = "child-with-ghost-parent"
    ghost_parent_id = "non-existent-parent"

    # This should not raise an exception despite missing parent
    await sql_provider.log_session(
        session_id=child_id, node_name="child", parent_session_id=ghost_parent_id
    )

    # Verify child still saved
    from sqlalchemy.ext.asyncio import AsyncSession

    async with AsyncSession(sql_provider.engine) as session:
        result = await session.execute(select(Conversation).where(Conversation.id == child_id))  # type: ignore[arg-type]
        convo = result.scalar_one()
        assert convo.id == child_id
        assert convo.parent_id == ghost_parent_id
