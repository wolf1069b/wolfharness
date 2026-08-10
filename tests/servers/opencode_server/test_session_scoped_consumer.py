"""Tests for session-scoped EventBus consumer in OpenCodeSessionPoolIntegration."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from wolfharness.agents.events import RunStartedEvent, StreamCompleteEvent
from wolfharness.agents.events.events import UserMessageInsertedEvent
from wolfharness.messaging.messages import ChatMessage
from wolfharness.orchestrator.core import SessionPool
from wolfharness_server.opencode_server.models import (
    MessagePath,
    MessageTime,
    MessageWithParts,
)
from wolfharness_server.opencode_server.session_pool_integration import (
    OpenCodeSessionPoolIntegration,
    get_messages_for_session,
)
from wolfharness_server.opencode_server.state import ServerState


pytestmark = pytest.mark.integration


@pytest.fixture
def mock_agent_pool() -> Mock:
    """Create a mock AgentPool for SessionPool construction."""
    from wolfharness.messaging.messages import ChatMessage

    pool = Mock()
    pool.main_agent = Mock()
    pool.main_agent.name = "test-agent"
    pool.manifest = Mock()
    pool.manifest.agents = {}
    pool.manifest.config_file_path = "test-pool-config"
    pool._config_file_path = None
    pool.storage = Mock()
    pool.storage.load_session = AsyncMock(return_value=None)
    pool.storage.save_session = AsyncMock(return_value=None)

    async def _mock_stream_events(*args: Any, **kwargs: Any) -> Any:
        """Yield a minimal run event sequence for testing."""
        session_id = kwargs.get("session_id", "unknown")
        run_id = "run-mock-001"
        yield RunStartedEvent(session_id=session_id, run_id=run_id)
        yield StreamCompleteEvent(
            message=ChatMessage(content="test response", role="assistant"),
        )

    mock_agent = Mock()
    mock_agent._stream_events = _mock_stream_events
    mock_agent._input_provider = None
    mock_agent.conversation = Mock()
    mock_agent.conversation.add_chat_messages = Mock()
    pool.get_agent = Mock(return_value=mock_agent)

    return pool


@pytest.fixture
def mock_session_store() -> Mock:
    """Create a mock SessionStore."""
    store = Mock()
    store.save_session = AsyncMock(return_value=None)
    store.delete_session = AsyncMock(return_value=None)
    store.load_session = AsyncMock(return_value=None)
    store.list_session_ids = AsyncMock(return_value=[])
    return store


@pytest.fixture
async def session_pool(mock_agent_pool: Mock, mock_session_store: Mock) -> SessionPool:
    """Create a real SessionPool with mocked dependencies."""
    sp = SessionPool(
        pool=mock_agent_pool,
        store=mock_session_store,
        enable_auto_resume=False,
        enable_event_bus=True,
    )
    mock_agent_pool.session_pool = sp
    await sp.start()
    yield sp
    await sp.shutdown()


@pytest.fixture
def server_state(tmp_path: Any, mock_agent_pool: Mock) -> ServerState:
    """Create a minimal ServerState for testing."""
    agent = Mock()
    agent.model_name = None  # resolve_default_model_info() fallback
    agent.name = "test-agent"
    agent.storage = Mock()
    agent.host_context = mock_agent_pool
    agent._agent_pool = mock_agent_pool  # state.py resolves _pool via agent._agent_pool
    agent.env = Mock()
    return ServerState(working_dir=str(tmp_path), agent=agent)


@pytest.mark.asyncio
async def test_get_messages_prefers_live_opencode_messages(
    server_state: ServerState,
) -> None:
    """Live OpenCode messages should be authoritative for TUI rendering."""
    session_id = "live-session"
    assistant_msg = MessageWithParts.assistant(
        message_id="assistant-message",
        session_id=session_id,
        time=MessageTime(created=1000),
        agent_name="test-agent",
        model_id="test-model",
        parent_id="user-message",
        provider_id="test-provider",
        path=MessagePath(cwd="/tmp", root="/tmp"),
    )
    assistant_msg.add_text_part("Live streamed content")
    server_state.messages[session_id] = [assistant_msg]

    # Mark as a subagent session so get_messages_for_session() takes
    # the fast path and returns live messages without consulting
    # the SessionPool.
    mock_cached_session = Mock()
    mock_cached_session.parent_id = "parent-session"
    server_state.sessions[session_id] = mock_cached_session

    stale_session_pool = Mock()
    stale_session_pool.get_messages = AsyncMock(
        return_value=[ChatMessage(content="", role="assistant")]
    )
    server_state.pool.session_pool = stale_session_pool

    messages = await get_messages_for_session(server_state, session_id)

    assert messages == [assistant_msg]
    stale_session_pool.get_messages.assert_not_awaited()


@pytest.mark.asyncio
async def test_event_consumer_started_on_session_creation(
    session_pool: SessionPool,
    server_state: ServerState,
) -> None:
    """Creating a session should start a session-scoped event consumer."""
    integration = OpenCodeSessionPoolIntegration(
        session_pool=session_pool,
        server_state=server_state,
    )

    await integration.create_session(
        session_id="test-consumer-session",
        agent_name="test-agent",
    )

    assert "test-consumer-session" in integration._session_groups
    assert "test-consumer-session" in integration._consumer_streams

    # Clean up
    await integration._stop_event_consumer("test-consumer-session")


@pytest.mark.asyncio
async def test_event_consumer_stopped_on_shutdown(
    session_pool: SessionPool,
    server_state: ServerState,
) -> None:
    """Shutdown should stop all event consumers."""
    integration = OpenCodeSessionPoolIntegration(
        session_pool=session_pool,
        server_state=server_state,
    )

    await integration.create_session(
        session_id="test-shutdown-session",
        agent_name="test-agent",
    )

    assert "test-shutdown-session" in integration._session_groups

    await integration.shutdown()

    assert "test-shutdown-session" not in integration._session_groups


@pytest.mark.asyncio
async def test_session_scoped_consumer_receives_events(
    session_pool: SessionPool,
    server_state: ServerState,
) -> None:
    """Session-scoped consumer should receive and broadcast events."""
    integration = OpenCodeSessionPoolIntegration(
        session_pool=session_pool,
        server_state=server_state,
    )

    broadcast_events: list[Any] = []
    original_broadcast = server_state.broadcast_event

    async def capture_broadcast(event: Any) -> None:
        broadcast_events.append(event)
        await original_broadcast(event)

    server_state.broadcast_event = capture_broadcast  # type: ignore[method-assign]

    await integration.create_session(
        session_id="test-receive-session",
        agent_name="test-agent",
    )

    # Give consumer time to start
    await asyncio.sleep(0.05)

    # Publish an event
    await session_pool.event_bus.publish(
        "test-receive-session",
        RunStartedEvent(session_id="test-receive-session", run_id="run-001"),
    )

    # Wait for consumer to process
    await asyncio.sleep(0.1)

    # The event should have been broadcast by the session-scoped consumer
    assert len(broadcast_events) >= 1

    # Clean up
    await integration._stop_event_consumer("test-receive-session")


@pytest.mark.asyncio
async def test_multiple_requests_share_one_consumer(
    session_pool: SessionPool,
    server_state: ServerState,
) -> None:
    """Multiple create_session calls for the same session should not create duplicate consumers."""
    integration = OpenCodeSessionPoolIntegration(
        session_pool=session_pool,
        server_state=server_state,
    )

    await integration.create_session(
        session_id="test-dedup-session",
        agent_name="test-agent",
    )

    first_task = integration._session_groups["test-dedup-session"]

    # Second create_session should be idempotent
    await integration.create_session(
        session_id="test-dedup-session",
        agent_name="test-agent",
    )

    second_task = integration._session_groups["test-dedup-session"]

    assert first_task is second_task

    # Clean up
    await integration._stop_event_consumer("test-dedup-session")


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="D1: start_event_consumer() is idempotent — _before_consumer_loop() "
    "only runs once. _message_registered stays True after turn 1, so "
    "turn 2 reuses turn 1's assistant_msg_id instead of generating a new one. "
    "This is a protocol-server-side issue independent of the per-prompt RunHandle "
    "migration. Fix: reset _message_registered on RunStartedEvent in the "
    "OpenCode event bridge.",
    strict=False,
    raises=AssertionError,
)
@pytest.mark.known_bug
async def test_consumer_resets_message_registered_for_new_turn(
    session_pool: SessionPool,
    server_state: ServerState,
) -> None:
    """D1: Consumer should reset _message_registered for each new turn.

    The session-scoped consumer's _before_consumer_loop() creates a fresh
    EventProcessorContext with _message_registered=False. But since
    start_event_consumer() is idempotent, _before_consumer_loop() only
    runs ONCE. On turn 2, _message_registered is still True from turn 1,
    so _handle_event skips assistant message registration — turn 2 reuses
    turn 1's assistant_msg_id.

    This test verifies that after publishing events for two turns through
    the same consumer, _message_registered is reset between turns.
    """
    integration = OpenCodeSessionPoolIntegration(
        session_pool=session_pool,
        server_state=server_state,
    )

    session_id = "test-d1-reset"
    await integration.create_session(session_id=session_id, agent_name="test-agent")
    await integration.start_event_consumer(session_id)

    # Simulate turn 1: publish RunStartedEvent + StreamCompleteEvent
    await session_pool.event_bus.publish(
        session_id,
        RunStartedEvent(
            run_id="run-1",
            session_id=session_id,
            agent_name="test-agent",
        ),
    )
    await asyncio.sleep(0.1)

    # After turn 1's first event, _message_registered should be True
    assert integration._message_registered.get(session_id, False), (
        "Turn 1 should have registered the assistant message"
    )

    # Simulate turn 1 completion
    await session_pool.event_bus.publish(
        session_id,
        StreamCompleteEvent(
            message=ChatMessage(content="turn 1 response", role="assistant"),
        ),
    )
    await asyncio.sleep(0.1)

    # Simulate turn 2: publish another RunStartedEvent
    await session_pool.event_bus.publish(
        session_id,
        RunStartedEvent(
            run_id="run-2",
            session_id=session_id,
            agent_name="test-agent",
        ),
    )
    await asyncio.sleep(0.1)

    # D1 bug: _message_registered is still True from turn 1.
    # After a fix, it should be reset to False on RunStartedEvent
    # so that _handle_event creates a new assistant message for turn 2.
    assert not integration._message_registered.get(session_id, False), (
        "_message_registered should be reset to False for turn 2 so "
        "that a new assistant message is registered. Currently it stays "
        "True from turn 1 (issue D1: consumer loop idempotency)."
    )

    # Clean up
    await integration._stop_event_consumer(session_id)


@pytest.mark.asyncio
async def test_consumer_handles_spawn_session_start(
    session_pool: SessionPool,
    server_state: ServerState,
) -> None:
    """Consumer should handle SpawnSessionStart by creating child consumers."""
    from wolfharness.agents.events import SpawnSessionStart

    integration = OpenCodeSessionPoolIntegration(
        session_pool=session_pool,
        server_state=server_state,
    )

    await integration.create_session(
        session_id="test-parent-session",
        agent_name="test-agent",
    )

    await session_pool.create_session(
        "test-child-session",
        parent_session_id="test-parent-session",
    )

    # Give consumer time to start
    await asyncio.sleep(0.05)

    # Publish SpawnSessionStart
    await session_pool.event_bus.publish(
        "test-parent-session",
        SpawnSessionStart(
            parent_session_id="test-parent-session",
            child_session_id="test-child-session",
            spawn_mechanism="task",
            source_name="test-tool",
            source_type="tool",
            description="Test subagent spawn",
            metadata={"prompt": "Inspect child task"},
        ),
    )

    # Wait for child consumer to be created
    await asyncio.sleep(0.1)

    # The child consumer should be running (it's tracked in the parent consumer's child_tasks)
    # We can't directly access child_tasks, but we can verify no exceptions occurred
    assert "test-parent-session" in integration._session_groups
    assert "test-child-session" in server_state.sessions
    assert server_state.sessions["test-child-session"].parent_id == "test-parent-session"

    # Publish UserMessageInsertedEvent for child session (simulates what
    # _route_message() does in production when the child agent's run starts).
    await session_pool.event_bus.publish(
        "test-child-session",
        UserMessageInsertedEvent(
            session_id="test-child-session",
            message_id="msg_child_prompt",
            content="Inspect child task",
            delivery="initial",
            source="protocol",
        ),
    )
    await asyncio.sleep(0.1)

    child_messages = await get_messages_for_session(server_state, "test-child-session")
    assert child_messages
    child_text = str(child_messages[0].parts)
    assert "Inspect child task" in child_text

    await session_pool.event_bus.publish(
        "test-child-session",
        StreamCompleteEvent(
            message=ChatMessage(role="assistant", content="Child task finished"),
            session_id="test-child-session",
        ),
    )
    await asyncio.sleep(0.1)

    completed_child_messages = await get_messages_for_session(server_state, "test-child-session")
    completed_child_text = " ".join(str(message.parts) for message in completed_child_messages)
    assert "Child task finished" in completed_child_text

    # Clean up
    await integration._stop_event_consumer("test-parent-session")
    await integration._stop_event_consumer("test-child-session")
