"""Integration tests for ACP subagent event handling.

Tests that ACPProtocolHandler correctly converts agent stream events to ACP
session update notifications via ProtocolEventConsumerMixin and ACPEventConverter.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock

import anyio
from pydantic_ai import RequestUsage, TextPartDelta
import pytest

from acp.schema import ClientCapabilities
from acp.schema.notifications import SessionNotification
from wolfharness.agents.events.events import (
    PartDeltaEvent,
    RunErrorEvent,
    SpawnSessionStart,
    StreamCompleteEvent,
    ToolCallStartEvent,
)
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.core import EventBus, EventEnvelope
from wolfharness_server.acp_server.event_converter import ACPEventConverter
from wolfharness_server.acp_server.handler import ACPProtocolHandler


pytestmark = [pytest.mark.unit, pytest.mark.anyio]


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """Return a mock EventBus with async subscribe/unsubscribe."""
    bus = AsyncMock(spec=EventBus)
    from tests._helpers.mock_stream import EmptyReceiveStream

    bus.subscribe = AsyncMock(return_value=EmptyReceiveStream())
    bus.unsubscribe = AsyncMock(return_value=None)
    return bus


@pytest.fixture
def mock_agent_pool(mock_event_bus: AsyncMock) -> Mock:
    """Return a mock AgentPool with session_pool and main_agent."""
    pool = Mock()
    pool.session_pool = Mock()
    pool.session_pool.event_bus = mock_event_bus
    pool.main_agent = Mock()
    pool.main_agent.metadata = {"use_session_pool": True}
    return pool


@pytest.fixture
def mock_client() -> AsyncMock:
    """Return a mock ACP client."""
    client = AsyncMock()
    client.session_update = AsyncMock(return_value=None)
    return client


@pytest.fixture
def acp_handler(
    mock_agent_pool: Mock,
    mock_client: AsyncMock,
) -> ACPProtocolHandler:
    """Return an ACPProtocolHandler with mocked dependencies."""
    session_manager = AsyncMock()
    event_converter = ACPEventConverter()
    handler = ACPProtocolHandler(
        host_context=mock_agent_pool,
        session_manager=session_manager,
        event_converter=event_converter,
        client=mock_client,
        client_capabilities=None,
    )
    # Prevent child consumer creation from SpawnSessionStart events
    # (existing tests were written before child consumer support was added)
    handler._on_spawn_session_start = AsyncMock()  # type: ignore[assignment]
    return handler


async def test_acp_handler_converts_spawn_session_start(
    acp_handler: ACPProtocolHandler,
    mock_event_bus: AsyncMock,
    mock_client: AsyncMock,
) -> None:
    """SpawnSessionStart produces session/update with AgentMessageChunk containing subagent name."""
    _queue: asyncio.Queue[EventEnvelope] = asyncio.Queue(maxsize=100)
    mock_event_bus.subscribe = AsyncMock(return_value=_queue)

    event = SpawnSessionStart(
        child_session_id="child-1",
        parent_session_id="sess-1",
        source_name="subagent-agent",
        source_type="agent",
        depth=1,
        description="test spawn",
        spawn_mechanism="spawn",
    )
    await _queue.put(EventEnvelope(source_session_id="sess-1", event=event))

    await acp_handler.start_event_consumer("sess-1")
    await asyncio.sleep(0.1)
    _queue.shutdown()
    for _ in range(100):
        if len(acp_handler._converters) == 0 or "sess-1" not in acp_handler._consumer_streams:
            break
        await asyncio.sleep(0.01)

    mock_client.session_update.assert_awaited()
    notification: SessionNotification[Any] = mock_client.session_update.await_args.args[0]
    assert isinstance(notification, SessionNotification)
    assert notification.session_id == "sess-1"
    assert notification.update.session_update == "agent_message_chunk"
    assert "subagent-agent" in notification.update.content.text


async def test_acp_handler_converts_part_delta(
    acp_handler: ACPProtocolHandler,
    mock_event_bus: AsyncMock,
    mock_client: AsyncMock,
) -> None:
    """PartDeltaEvent from subagent is converted to AgentMessageChunk."""
    _queue: asyncio.Queue[EventEnvelope] = asyncio.Queue(maxsize=100)
    mock_event_bus.subscribe = AsyncMock(return_value=_queue)

    event = PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="hello"))
    await _queue.put(EventEnvelope(source_session_id="sess-1", event=event))

    await acp_handler.start_event_consumer("sess-1")
    await asyncio.sleep(0.1)
    _queue.shutdown()
    for _ in range(100):
        if len(acp_handler._converters) == 0 or "sess-1" not in acp_handler._consumer_streams:
            break
        await asyncio.sleep(0.01)

    mock_client.session_update.assert_awaited()
    notification: SessionNotification[Any] = mock_client.session_update.await_args.args[0]
    assert isinstance(notification, SessionNotification)
    assert notification.update.session_update == "agent_message_chunk"
    assert notification.update.content.text == "hello"


async def test_acp_handler_converts_tool_call(
    acp_handler: ACPProtocolHandler,
    mock_event_bus: AsyncMock,
    mock_client: AsyncMock,
) -> None:
    """ToolCallStartEvent from subagent produces ToolCallStart notification."""
    _queue: asyncio.Queue[EventEnvelope] = asyncio.Queue(maxsize=100)
    mock_event_bus.subscribe = AsyncMock(return_value=_queue)

    event = ToolCallStartEvent(
        tool_call_id="tc-1",
        tool_name="bash",
        title="Run bash command",
        kind="execute",
    )
    await _queue.put(EventEnvelope(source_session_id="sess-1", event=event))

    await acp_handler.start_event_consumer("sess-1")
    await asyncio.sleep(0.1)
    _queue.shutdown()
    for _ in range(100):
        if len(acp_handler._converters) == 0 or "sess-1" not in acp_handler._consumer_streams:
            break
        await asyncio.sleep(0.01)

    mock_client.session_update.assert_awaited()
    notification: SessionNotification[Any] = mock_client.session_update.await_args.args[0]
    assert isinstance(notification, SessionNotification)
    assert notification.update.session_update == "tool_call"
    assert notification.update.tool_call_id == "tc-1"
    assert notification.update.title == "Run bash command"
    assert notification.update.kind == "execute"


async def test_acp_handler_converts_stream_complete(
    mock_agent_pool: Mock,
    mock_event_bus: AsyncMock,
    mock_client: AsyncMock,
) -> None:
    """StreamCompleteEvent produces UsageUpdate (+ TurnCompleteUpdate if client supports it)."""
    _queue: asyncio.Queue[EventEnvelope] = asyncio.Queue(maxsize=100)
    mock_event_bus.subscribe = AsyncMock(return_value=_queue)

    handler = ACPProtocolHandler(
        host_context=mock_agent_pool,
        session_manager=AsyncMock(),
        event_converter=ACPEventConverter(),
        client=mock_client,
        client_capabilities=ClientCapabilities(turn_complete=True),
    )

    message = ChatMessage(
        content="done",
        role="assistant",
        usage=RequestUsage(input_tokens=5, output_tokens=5),
    )
    event = StreamCompleteEvent(message=message, session_id="sess-1")
    await _queue.put(EventEnvelope(source_session_id="sess-1", event=event))

    await handler.start_event_consumer("sess-1")
    await asyncio.sleep(0.1)
    _queue.shutdown()
    for _ in range(100):
        if len(handler._converters) == 0 or "sess-1" not in handler._consumer_streams:
            break
        await asyncio.sleep(0.01)

    assert mock_client.session_update.await_count == 2
    calls = mock_client.session_update.await_args_list

    assert calls[0].args[0].update.session_update == "usage_update"
    assert calls[0].args[0].update.used == 10

    assert calls[1].args[0].update.session_update == "turn_complete"
    assert calls[1].args[0].update.stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# 9.5: Event + closure completion notification (mock done_event)
# ---------------------------------------------------------------------------


async def test_notify_completed_called_when_done_event_set(
    acp_handler: ACPProtocolHandler,
    mock_client: AsyncMock,
) -> None:
    """_notify_completed is called when done_event is set.

    Given: An ACPProtocolHandler with a parent converter and _parent_of entry.
    When: _await_child_and_notify's done_event is set.
    Then: _notify_completed sends a ToolCallProgress completion notification.
    """
    # Set up parent converter in zed mode so build_subagent_completed yields
    from wolfharness_server.acp_server.event_converter import ACPEventConverter

    zed_converter = ACPEventConverter(subagent_display_mode="zed")
    zed_converter._current_message_id = "test-msg"
    acp_handler._converters["parent-ses"] = zed_converter
    # Seed the converter's _subagent_tool_call_ids map
    from wolfharness.agents.events import SpawnSessionStart

    spawn = SpawnSessionStart(
        child_session_id="child-ses",
        parent_session_id="parent-ses",
        tool_call_id="tc-905",
        spawn_mechanism="spawn",
        source_name="coder",
        source_type="agent",
        depth=1,
        description="Test",
    )
    async for _ in zed_converter.convert(spawn):
        pass

    done_event = anyio.Event()
    acp_handler._parent_of["child-ses"] = "parent-ses"

    task = asyncio.ensure_future(
        acp_handler._await_child_and_notify(
            parent_sid="parent-ses",
            child_sid="child-ses",
            done_event=done_event,
        )
    )

    done_event.set()
    await task

    mock_client.session_update.assert_awaited()
    notification = mock_client.session_update.await_args.args[0]
    assert notification.session_id == "parent-ses"
    assert notification.update.status == "completed"
    assert notification.update.tool_call_id == "tc-905"


# ---------------------------------------------------------------------------
# 9.6: done_event is None race — immediate notification fired
# ---------------------------------------------------------------------------


async def test_done_event_none_race_immediate_notification(
    acp_handler: ACPProtocolHandler,
    mock_client: AsyncMock,
) -> None:
    """When _consumer_done_events.get returns None, _notify_completed fires immediately.

    Given: An ACPProtocolHandler where _consumer_done_events.get(child_sid) returns None.
    When: _on_spawn_session_start processes a SpawnSessionStart.
    Then: _notify_completed is called immediately (no closure spawned).
    """
    from wolfharness_server.acp_server.event_converter import ACPEventConverter

    zed_converter = ACPEventConverter(subagent_display_mode="zed")
    zed_converter._current_message_id = "test-msg"
    acp_handler._converters["parent-ses"] = zed_converter
    from wolfharness.agents.events import SpawnSessionStart

    spawn = SpawnSessionStart(
        child_session_id="child-race",
        parent_session_id="parent-ses",
        tool_call_id="tc-906",
        spawn_mechanism="spawn",
        source_name="coder",
        source_type="agent",
        depth=1,
        description="Race test",
    )
    async for _ in zed_converter.convert(spawn):
        pass

    # Ensure _consumer_done_events is empty (simulating race)
    acp_handler._consumer_done_events.clear()
    # Restore the real _on_spawn_session_start (fixture overrides it with AsyncMock)
    import types

    from wolfharness_server.acp_server.handler import ACPProtocolHandler as _HandlerCls

    acp_handler._on_spawn_session_start = types.MethodType(  # type: ignore[assignment]
        _HandlerCls._on_spawn_session_start, acp_handler
    )

    # Mock start_event_consumer to not actually start a consumer
    async def _noop_start(sid: str) -> None:
        pass

    acp_handler.start_event_consumer = _noop_start  # type: ignore[assignment]

    envelope = EventEnvelope(source_session_id="parent-ses", event=spawn)
    await acp_handler._on_spawn_session_start("parent-ses", envelope)

    # _notify_completed should have been called immediately
    mock_client.session_update.assert_awaited()
    notification = mock_client.session_update.await_args.args[0]
    assert notification.session_id == "parent-ses"
    assert notification.update.status == "completed"


# ---------------------------------------------------------------------------
# 9.7: Concurrent child sessions — each gets correct tool_call_id completion
# ---------------------------------------------------------------------------


async def test_concurrent_children_each_get_completion_notification(
    acp_handler: ACPProtocolHandler,
    mock_client: AsyncMock,
) -> None:
    """Multiple concurrent child sessions each receive their own completion notification.

    Given: Two child sessions spawned from the same parent.
    When: Both done_events are set.
    Then: _notify_completed is called for each child with the correct tool_call_id.
    """
    from wolfharness_server.acp_server.event_converter import ACPEventConverter

    zed_converter = ACPEventConverter(subagent_display_mode="zed")
    zed_converter._current_message_id = "test-msg"
    acp_handler._converters["parent-ses"] = zed_converter
    from wolfharness.agents.events import SpawnSessionStart

    # Seed converter with two spawn events
    for i, child_sid in enumerate(["child-a", "child-b"]):
        spawn = SpawnSessionStart(
            child_session_id=child_sid,
            parent_session_id="parent-ses",
            tool_call_id=f"tc-concurrent-{i}",
            spawn_mechanism="spawn",
            source_name="coder",
            source_type="agent",
            depth=1,
            description=f"Child {i}",
        )
        async for _ in zed_converter.convert(spawn):
            pass

    done_a = anyio.Event()
    done_b = anyio.Event()
    acp_handler._parent_of["child-a"] = "parent-ses"
    acp_handler._parent_of["child-b"] = "parent-ses"

    task_a = asyncio.ensure_future(
        acp_handler._await_child_and_notify("parent-ses", "child-a", done_a)
    )
    task_b = asyncio.ensure_future(
        acp_handler._await_child_and_notify("parent-ses", "child-b", done_b)
    )

    done_a.set()
    await task_a
    done_b.set()
    await task_b

    assert mock_client.session_update.await_count == 2
    tool_call_ids = {
        call.args[0].update.tool_call_id for call in mock_client.session_update.await_args_list
    }
    assert "tc-concurrent-0" in tool_call_ids
    assert "tc-concurrent-1" in tool_call_ids


# ---------------------------------------------------------------------------
# 9.8: Closure error handling — session_update raises, exception logged not swallowed
# ---------------------------------------------------------------------------


async def test_closure_error_logged_not_swallowed(
    acp_handler: ACPProtocolHandler,
    mock_client: AsyncMock,
) -> None:
    """When session_update raises a non-connection error, exception is logged but not re-raised.

    Given: An ACPProtocolHandler where client.session_update raises ValueError.
    When: _await_child_and_notify completes (done_event set).
    Then: The closure does NOT re-raise the ValueError (caught by generic except).
    """
    from wolfharness_server.acp_server.event_converter import ACPEventConverter

    zed_converter = ACPEventConverter(subagent_display_mode="zed")
    zed_converter._current_message_id = "test-msg"
    acp_handler._converters["parent-ses"] = zed_converter
    from wolfharness.agents.events import SpawnSessionStart

    spawn = SpawnSessionStart(
        child_session_id="child-err",
        parent_session_id="parent-ses",
        tool_call_id="tc-908",
        spawn_mechanism="spawn",
        source_name="coder",
        source_type="agent",
        depth=1,
        description="Error test",
    )
    async for _ in zed_converter.convert(spawn):
        pass

    mock_client.session_update = AsyncMock(side_effect=ValueError("unexpected error"))

    done_event = anyio.Event()
    acp_handler._parent_of["child-err"] = "parent-ses"

    task = asyncio.ensure_future(
        acp_handler._await_child_and_notify("parent-ses", "child-err", done_event)
    )

    done_event.set()
    # Should not raise — exception is caught by generic except in _await_child_and_notify
    await task

    mock_client.session_update.assert_awaited()


# ---------------------------------------------------------------------------
# 9.9: _consumer_task_refs cleanup after task completion
# ---------------------------------------------------------------------------


async def test_consumer_task_refs_cleanup_after_closure(
    acp_handler: ACPProtocolHandler,
    mock_client: AsyncMock,
) -> None:
    """_consumer_task_refs is cleaned up after closure task completes.

    Given: An ACPProtocolHandler with a closure task in _consumer_task_refs.
    When: The closure task completes (done_event set).
    Then: The task is removed from _consumer_task_refs.
    """
    from wolfharness_server.acp_server.event_converter import ACPEventConverter

    zed_converter = ACPEventConverter(subagent_display_mode="zed")
    zed_converter._current_message_id = "test-msg"
    acp_handler._converters["parent-ses"] = zed_converter
    from wolfharness.agents.events import SpawnSessionStart

    spawn = SpawnSessionStart(
        child_session_id="child-ref",
        parent_session_id="parent-ses",
        tool_call_id="tc-909",
        spawn_mechanism="spawn",
        source_name="coder",
        source_type="agent",
        depth=1,
        description="Ref cleanup test",
    )
    async for _ in zed_converter.convert(spawn):
        pass

    done_event = anyio.Event()
    acp_handler._parent_of["child-ref"] = "parent-ses"

    task = asyncio.ensure_future(
        acp_handler._await_child_and_notify("parent-ses", "child-ref", done_event)
    )
    acp_handler._consumer_task_refs.append(task)

    assert task in acp_handler._consumer_task_refs

    done_event.set()
    await task

    assert task not in acp_handler._consumer_task_refs


# ---------------------------------------------------------------------------
# 9.10: _parent_of cleanup on normal child exit
# ---------------------------------------------------------------------------


async def test_parent_of_cleanup_on_child_exit(
    acp_handler: ACPProtocolHandler,
    mock_client: AsyncMock,
) -> None:
    """_parent_of entry is popped when child consumer loop ends.

    Given: An ACPProtocolHandler with _parent_of[child_sid] = parent_sid.
    When: _await_child_and_notify's done_event is set (simulating child exit).
    Then: _parent_of[child_sid] is removed.
    """
    from wolfharness_server.acp_server.event_converter import ACPEventConverter

    zed_converter = ACPEventConverter(subagent_display_mode="zed")
    zed_converter._current_message_id = "test-msg"
    acp_handler._converters["parent-ses"] = zed_converter
    from wolfharness.agents.events import SpawnSessionStart

    spawn = SpawnSessionStart(
        child_session_id="child-cleanup",
        parent_session_id="parent-ses",
        tool_call_id="tc-910",
        spawn_mechanism="spawn",
        source_name="coder",
        source_type="agent",
        depth=1,
        description="Cleanup test",
    )
    async for _ in zed_converter.convert(spawn):
        pass

    done_event = anyio.Event()
    acp_handler._parent_of["child-cleanup"] = "parent-ses"

    assert "child-cleanup" in acp_handler._parent_of

    task = asyncio.ensure_future(
        acp_handler._await_child_and_notify("parent-ses", "child-cleanup", done_event)
    )

    done_event.set()
    await task

    assert "child-cleanup" not in acp_handler._parent_of


# ---------------------------------------------------------------------------
# 9.12: Recursive cancellation — parent stop cascades to children and grandchildren
# ---------------------------------------------------------------------------


async def test_recursive_cancellation_cascades_to_grandchildren(
    acp_handler: ACPProtocolHandler,
) -> None:
    """_cancel_subagents walks _parent_of tree and stops all descendants.

    Given: A 3-level hierarchy in _parent_of: parent → child → grandchild.
    When: _cancel_subagents is called on the parent.
    Then: stop_event_consumer is called for child AND grandchild.
    """
    # Set up a 3-level hierarchy
    acp_handler._parent_of["child-1"] = "parent-1"
    acp_handler._parent_of["grandchild-1"] = "child-1"

    stopped_sessions: list[str] = []

    async def _mock_stop(sid: str) -> None:
        stopped_sessions.append(sid)

    acp_handler.stop_event_consumer = _mock_stop  # type: ignore[assignment]

    await acp_handler._cancel_subagents("parent-1")

    # Both child-1 and grandchild-1 should be stopped
    assert "child-1" in stopped_sessions
    assert "grandchild-1" in stopped_sessions
    # _parent_of should be empty after cleanup
    assert "child-1" not in acp_handler._parent_of
    assert "grandchild-1" not in acp_handler._parent_of


async def test_acp_handler_converts_run_error(
    acp_handler: ACPProtocolHandler,
    mock_event_bus: AsyncMock,
    mock_client: AsyncMock,
) -> None:
    """RunErrorEvent produces error-formatted AgentMessageChunk."""
    _queue: asyncio.Queue[EventEnvelope] = asyncio.Queue(maxsize=100)
    mock_event_bus.subscribe = AsyncMock(return_value=_queue)

    event = RunErrorEvent(message="something broke", agent_name="test-agent")
    await _queue.put(EventEnvelope(source_session_id="sess-1", event=event))

    await acp_handler.start_event_consumer("sess-1")
    await asyncio.sleep(0.1)
    _queue.shutdown()
    for _ in range(100):
        if len(acp_handler._converters) == 0 or "sess-1" not in acp_handler._consumer_streams:
            break
        await asyncio.sleep(0.01)

    mock_client.session_update.assert_awaited()
    notification: SessionNotification[Any] = mock_client.session_update.await_args.args[0]
    assert isinstance(notification, SessionNotification)
    assert notification.update.session_update == "agent_message_chunk"
    assert "something broke" in notification.update.content.text
    assert "test-agent" in notification.update.content.text


async def test_acp_handler_connection_error_stops_consumer(
    acp_handler: ACPProtocolHandler,
    mock_event_bus: AsyncMock,
    mock_client: AsyncMock,
) -> None:
    """ConnectionResetError during session_update triggers ConsumerShutdown."""
    _queue: asyncio.Queue[EventEnvelope] = asyncio.Queue(maxsize=100)
    mock_event_bus.subscribe = AsyncMock(return_value=_queue)

    mock_client.session_update = AsyncMock(side_effect=ConnectionResetError("connection lost"))

    event = PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="hello"))
    await _queue.put(EventEnvelope(source_session_id="sess-1", event=event))

    await acp_handler.start_event_consumer("sess-1")
    await asyncio.sleep(0.1)
    _queue.shutdown()
    for _ in range(100):
        if len(acp_handler._converters) == 0 or "sess-1" not in acp_handler._consumer_streams:
            break
        await asyncio.sleep(0.01)

    assert "sess-1" not in acp_handler._session_groups
    mock_event_bus.unsubscribe.assert_awaited()


async def test_acp_handler_converter_isolated_per_session(
    acp_handler: ACPProtocolHandler,
    mock_event_bus: AsyncMock,
    mock_client: AsyncMock,
) -> None:
    """Two sessions have separate converters, events don't cross."""
    _queue1: asyncio.Queue[EventEnvelope] = asyncio.Queue(maxsize=100)
    _queue2: asyncio.Queue[EventEnvelope] = asyncio.Queue(maxsize=100)
    mock_event_bus.subscribe = AsyncMock(side_effect=[_queue1, _queue2])

    # Capture converter instances before the loop cleans them up
    captured_converters: dict[str, ACPEventConverter] = {}
    original_before = acp_handler._before_consumer_loop

    async def _patched_before(session_id: str) -> None:
        await original_before(session_id)
        captured_converters[session_id] = acp_handler._converters[session_id]

    acp_handler._before_consumer_loop = _patched_before  # type: ignore[assignment]

    event1 = SpawnSessionStart(
        child_session_id="child-1",
        parent_session_id="sess-1",
        source_name="agent-a",
        source_type="agent",
        depth=1,
        description="spawn a",
        spawn_mechanism="spawn",
    )
    event2 = SpawnSessionStart(
        child_session_id="child-2",
        parent_session_id="sess-2",
        source_name="agent-b",
        source_type="agent",
        depth=1,
        description="spawn b",
        spawn_mechanism="spawn",
    )
    await _queue1.put(EventEnvelope(source_session_id="sess-1", event=event1))
    await _queue2.put(EventEnvelope(source_session_id="sess-2", event=event2))

    await acp_handler.start_event_consumer("sess-1")
    await acp_handler.start_event_consumer("sess-2")
    await asyncio.sleep(0.2)
    _queue1.shutdown()
    _queue2.shutdown()

    # Verify separate converter instances were created
    assert "sess-1" in captured_converters
    assert "sess-2" in captured_converters
    assert captured_converters["sess-1"] is not captured_converters["sess-2"]

    assert mock_client.session_update.await_count == 2
    calls = mock_client.session_update.await_args_list

    sess1_notifications = [c.args[0] for c in calls if c.args[0].session_id == "sess-1"]
    sess2_notifications = [c.args[0] for c in calls if c.args[0].session_id == "sess-2"]

    assert len(sess1_notifications) == 1
    assert len(sess2_notifications) == 1
    assert "agent-a" in sess1_notifications[0].update.content.text
    assert "agent-b" in sess2_notifications[0].update.content.text


async def test_acp_handler_no_child_consumers_created(
    acp_handler: ACPProtocolHandler,
    mock_event_bus: AsyncMock,
) -> None:
    """Verify _session_groups only has parent session, no child consumers."""
    _queue: asyncio.Queue[EventEnvelope] = asyncio.Queue(maxsize=100)
    mock_event_bus.subscribe = AsyncMock(return_value=_queue)

    event = SpawnSessionStart(
        child_session_id="child-1",
        parent_session_id="sess-1",
        source_name="subagent-agent",
        source_type="agent",
        depth=1,
        description="test spawn",
        spawn_mechanism="spawn",
    )
    await _queue.put(EventEnvelope(source_session_id="sess-1", event=event))

    await acp_handler.start_event_consumer("sess-1")

    await asyncio.sleep(0.05)

    assert len(acp_handler._session_groups) == 1
    assert "sess-1" in acp_handler._session_groups
    assert "child-1" not in acp_handler._session_groups

    _queue.shutdown()
    for _ in range(100):
        if len(acp_handler._converters) == 0 or "sess-1" not in acp_handler._consumer_streams:
            break
        await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# Handler integration: subagent context, field_meta, nesting
# ---------------------------------------------------------------------------


async def test_on_spawn_creates_child_converter_with_subagent_context(
    acp_handler: ACPProtocolHandler,
    mock_client: AsyncMock,
) -> None:
    """_on_spawn_session_start creates child converter with SubagentContext.

    Given: An ACPProtocolHandler with restored _on_spawn_session_start.
    When: A SpawnSessionStart event is processed.
    Then: A child converter is created with the correct subagent_context.
    """
    import types

    from wolfharness_server.acp_server.handler import ACPProtocolHandler as _HandlerCls

    # Restore real method (fixture overrides it with AsyncMock)
    acp_handler._on_spawn_session_start = types.MethodType(  # type: ignore[assignment]
        _HandlerCls._on_spawn_session_start, acp_handler
    )

    # Mock start_event_consumer to not actually start a consumer
    async def _noop_start(sid: str) -> None:
        pass

    acp_handler.start_event_consumer = _noop_start  # type: ignore[assignment]

    spawn = SpawnSessionStart(
        child_session_id="child-ctx",
        parent_session_id="sess-1",
        tool_call_id="tc-ctx-1",
        spawn_mechanism="spawn",
        source_name="coder",
        source_type="agent",
        depth=1,
        description="Context test",
    )
    envelope = EventEnvelope(source_session_id="sess-1", event=spawn)
    await acp_handler._on_spawn_session_start("sess-1", envelope)

    child_converter = acp_handler._converters.get("child-ctx")
    assert child_converter is not None
    assert child_converter.subagent_context is not None
    assert child_converter.subagent_context.parent_tool_call_id == "tc-ctx-1"
    assert child_converter.subagent_context.subagent_type == "coder"


async def test_before_consumer_loop_skips_when_converter_exists(
    acp_handler: ACPProtocolHandler,
) -> None:
    """_before_consumer_loop returns early when converter already exists.

    Given: An ACPProtocolHandler.
    When: _before_consumer_loop is called twice.
    Then: Only one converter exists for the session (second call returns early).
    """
    # First call creates a converter
    await acp_handler._before_consumer_loop("sess-before")
    assert "sess-before" in acp_handler._converters
    first_converter = acp_handler._converters["sess-before"]

    # Second call should return early (converter already exists)
    await acp_handler._before_consumer_loop("sess-before")
    assert "sess-before" in acp_handler._converters
    assert acp_handler._converters["sess-before"] is first_converter


async def test_handle_event_stamps_field_meta_on_child_notification(
    acp_handler: ACPProtocolHandler,
    mock_client: AsyncMock,
) -> None:
    """_handle_event stamps field_meta on SessionNotification for child sessions.

    Given: A child converter with SubagentContext is in _converters.
    When: A PartDeltaEvent is handled for that child session.
    Then: The resulting SessionNotification has the correct field_meta dict.
    """
    from wolfharness_server.acp_server.event_converter import SubagentContext

    child_converter = ACPEventConverter(
        subagent_context=SubagentContext(parent_tool_call_id="tc-123", subagent_type="coder"),
    )
    acp_handler._converters["child-ses"] = child_converter

    event = PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="hello from child"))
    envelope = EventEnvelope(source_session_id="child-ses", event=event)
    await acp_handler._handle_event("child-ses", envelope)

    mock_client.session_update.assert_awaited_once()
    notification: SessionNotification[Any] = mock_client.session_update.await_args.args[0]
    assert notification.field_meta == {
        "parentToolCallId": "tc-123",
        "subagentType": "coder",
        "provenance": "subagent",
    }


async def test_root_session_notification_has_field_meta_none(
    acp_handler: ACPProtocolHandler,
    mock_client: AsyncMock,
) -> None:
    """Root session notifications have field_meta=None.

    Given: A root session converter (no subagent_context).
    When: A PartDeltaEvent is handled for that session.
    Then: The resulting SessionNotification has field_meta=None.
    """
    # Root converter is created by _before_consumer_loop without subagent_context
    await acp_handler._before_consumer_loop("sess-root")
    assert "sess-root" in acp_handler._converters
    assert acp_handler._converters["sess-root"].subagent_context is None

    event = PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="root message"))
    envelope = EventEnvelope(source_session_id="sess-root", event=event)
    await acp_handler._handle_event("sess-root", envelope)

    mock_client.session_update.assert_awaited_once()
    notification: SessionNotification[Any] = mock_client.session_update.await_args.args[0]
    assert notification.field_meta is None


async def test_nested_subagents_each_have_own_context(
    acp_handler: ACPProtocolHandler,
) -> None:
    """Nested subagent converters each have their own SubagentContext.

    Given: Two levels of child converters in _converters.
    When: Inspecting their subagent_context values.
    Then: Each converter has its own context pointing to its parent_tool_call_id.
    """
    from wolfharness_server.acp_server.event_converter import SubagentContext

    child_converter = ACPEventConverter(
        subagent_context=SubagentContext(
            parent_tool_call_id="tc-child", subagent_type="child-agent"
        ),
    )
    grandchild_converter = ACPEventConverter(
        subagent_context=SubagentContext(
            parent_tool_call_id="tc-grandchild", subagent_type="grandchild-agent"
        ),
    )
    acp_handler._converters["child-ses"] = child_converter
    acp_handler._converters["grandchild-ses"] = grandchild_converter

    child_ctx = acp_handler._converters["child-ses"].subagent_context
    grandchild_ctx = acp_handler._converters["grandchild-ses"].subagent_context

    assert child_ctx is not None
    assert grandchild_ctx is not None
    assert child_ctx.parent_tool_call_id == "tc-child"
    assert grandchild_ctx.parent_tool_call_id == "tc-grandchild"
    assert grandchild_ctx.parent_tool_call_id != child_ctx.parent_tool_call_id
