"""Tests for async BackgroundTaskProvider EventBus alignment.

These tests verify that _run_and_stream() NO LONGER emits SubAgentEvents
via ctx.events.emit_event(). Events are broadcast via the EventBus by
TurnRunner; consumers subscribe with scope="descendants" to receive them.

This is critical for:
- Avoiding duplicate event delivery (EventBus + manual emit)
- Proper separation of concerns: provider writes to filesystem only
- Protocol consumers (OpenCode/ACP) receiving events via EventBus
"""

# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
# Mock-heavy test code: assigning to spec'd attributes, accessing mock call_args_list,
# and accessing event.message.content through union types are all expected.

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic_ai import RunContext
from pydantic_ai.messages import TextPartDelta
import pytest

from wolfharness import ChatMessage
from wolfharness.agents.base_agent import BaseAgent
from wolfharness.agents.context import AgentContext
from wolfharness.agents.events import (
    PartDeltaEvent,
    RunFailedEvent,
    StreamCompleteEvent,
    SubAgentEvent,
)
from wolfharness.capabilities.background_task.capability import (
    BackgroundTaskCapability,
)
from wolfharness.delegation import AgentPool


pytestmark = pytest.mark.anyio


async def _wait_until_called(mock_obj: Any, timeout: float = 3.0, interval: float = 0.05) -> None:
    elapsed = 0.0
    while mock_obj.call_count == 0 and elapsed < timeout:
        await asyncio.sleep(interval)
        elapsed += interval
    assert mock_obj.call_count > 0, f"Mock was not called within {timeout}s"


def _wrap_in_run_context(agent_ctx):
    """Wrap an AgentContext in a mock RunContext for capability tool methods."""
    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = agent_ctx
    run_ctx.tool_call_id = agent_ctx.tool_call_id
    return run_ctx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_node(*, name: str = "worker", description: str = "A worker agent") -> MagicMock:
    """Create a mock node that looks like a BaseAgent with run_stream support."""
    node = MagicMock(spec=BaseAgent)
    node.type = "agent"
    node.name = name
    node.description = description
    node.model_name = "test:model"
    node.session_id = "ses_parent_123"
    return node


def _make_mock_pool(nodes: dict[str, MagicMock] | None = None) -> AgentPool:
    """Create a mock AgentPool with the given nodes."""
    pool = MagicMock(spec=AgentPool)

    if nodes is None:
        mock_node = _make_mock_node()
        nodes = {"worker": mock_node}

    pool.agent_configs = nodes
    pool.nodes = nodes
    pool.all_agents = list(nodes.items())
    pool.teams = {}
    pool.sessions = None

    mock_session_pool = MagicMock()

    def _run_stream_proxy(child_session_id: str, formatted_prompt: str, **kwargs: Any):
        for node in nodes.values():
            if hasattr(node, "run_stream"):
                return node.run_stream(
                    formatted_prompt,
                    deps=kwargs.get("deps", MagicMock()),
                    session_id=child_session_id,
                    parent_session_id="ses_parent_123",
                    depth=1,
                    message_history=MagicMock(),
                )

        async def _empty():
            return
            yield

        return _empty()

    mock_session_pool.run_stream = MagicMock(side_effect=_run_stream_proxy)

    _queues: dict[str, asyncio.Queue] = {}

    async def _subscribe(session_id: str, scope: str = "session"):
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        _queues[session_id] = queue
        return queue

    async def _send_message(session_id: str, prompt: str, input_provider=None, **kwargs: Any):
        queue = _queues.get(session_id)
        if queue is None:
            queue = asyncio.Queue(maxsize=100)
            _queues[session_id] = queue
        for node in nodes.values():
            if hasattr(node, "run_stream"):
                stream = node.run_stream(
                    prompt,
                    deps=kwargs.get("deps", MagicMock()),
                    session_id=session_id,
                    parent_session_id="ses_parent_123",
                    depth=1,
                    message_history=MagicMock(),
                )

                async def _feed(stream=stream, queue=queue):
                    try:
                        async for event in stream:
                            await queue.put(event)
                    except (ValueError, RuntimeError, TypeError) as exc:
                        await queue.put(
                            RunFailedEvent(run_id="test", session_id=session_id, exception=exc)
                        )
                    finally:
                        queue.shutdown()

                feed_task = asyncio.create_task(_feed())
                assert feed_task is not None
                break
        return MagicMock()

    mock_session_pool.send_message = AsyncMock(side_effect=_send_message)
    mock_session_pool.inject_prompt = AsyncMock()
    mock_session_pool.steer = AsyncMock()
    mock_session_pool.followup = AsyncMock(return_value=True)
    mock_session_pool.event_bus = MagicMock()
    mock_session_pool.event_bus.subscribe = AsyncMock(side_effect=_subscribe)
    mock_session_pool.event_bus.unsubscribe = AsyncMock()
    mock_session_pool.sessions = MagicMock()
    mock_session_pool.sessions.get_or_create_session_agent = AsyncMock(
        return_value=_make_mock_node()
    )
    pool.session_pool = mock_session_pool
    pool.manifest = MagicMock()
    pool.manifest.agents = nodes
    return pool


def _make_agent_context(
    pool: AgentPool | None = None,
    data: dict[str, Any] | None = None,
) -> AgentContext:
    """Create a minimal AgentContext for testing."""
    agent = MagicMock(spec=BaseAgent)
    agent.type = "agent"
    agent.name = "coordinator"
    agent.session_id = "ses_parent_123"
    agent.agent_pool = pool

    ctx = MagicMock(spec=AgentContext)
    ctx.node = agent
    ctx.pool = pool
    ctx.data = data if data is not None else {}
    ctx.tool_call_id = "tc_001"

    # Mock AgentRunContext for session state resolution
    mock_run_ctx = MagicMock()
    mock_run_ctx.session_id = "ses_parent_123"
    mock_run_ctx._run_handle = None
    mock_run_ctx.child_done_events = {}
    ctx.run_ctx = mock_run_ctx

    ctx.events = MagicMock()
    ctx.events.emit_event = AsyncMock()
    ctx.create_child_session = AsyncMock(return_value="ses_child_456")
    ctx.internal_fs = MagicMock()
    ctx.internal_fs.mkdirs = MagicMock()
    ctx.internal_fs.pipe = MagicMock()

    return ctx


async def _collect_stream_events(events: list[Any]):
    """Create an async generator that yields the given events."""
    for event in events:
        yield event


# ---------------------------------------------------------------------------
# Test: async task does NOT emit SubAgentEvents to parent stream
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_async_task_does_not_emit_subagent_events():
    """Verify that async task does NOT emit SubAgentEvent via ctx.events.emit_event()."""
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    mock_node = pool.nodes["worker"]
    generic_event = MagicMock()
    generic_event.__class__ = type("GenericEvent", (), {})
    complete_event = StreamCompleteEvent(
        message=ChatMessage(content="Final result", role="assistant"),
    )

    mock_node.run_stream = MagicMock(
        return_value=_collect_stream_events([generic_event, complete_event]),
    )

    with patch(
        "wolfharness.capabilities.background_task.capability._generate_task_id",
        return_value="bg_asevnt01",
    ):
        await capability._task(
            _wrap_in_run_context(ctx),
            agent="worker",
            message="test task",
            async_mode=True,
        )

    # Wait for the background task to complete
    await asyncio.sleep(0.2)

    # Check emitted events — no SubAgentEvents should be emitted by the provider
    emitted_events = [call.args[0] for call in ctx.events.emit_event.call_args_list]
    sub_events = [e for e in emitted_events if isinstance(e, SubAgentEvent)]
    assert len(sub_events) == 0, f"Expected 0 SubAgentEvents, got {len(sub_events)}"

    # SpawnSessionStart is now auto-emitted by create_child_session(),
    # not manually by the provider. Verify create_child_session was called
    # with spawn_mechanism="task" so the auto-emission happens in production.
    ctx.create_child_session.assert_awaited_once()
    call_kwargs = ctx.create_child_session.call_args.kwargs
    assert call_kwargs.get("spawn_mechanism") == "task"
    assert call_kwargs.get("description") == ""  # title is None → ""


# ---------------------------------------------------------------------------
# Test: async task does NOT emit StreamCompleteEvent on normal finish
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_async_task_does_not_emit_stream_complete_on_finish():
    """Verify that async task does NOT emit SubAgentEvent(StreamCompleteEvent)."""
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    mock_node = pool.nodes["worker"]
    complete_event = StreamCompleteEvent(
        message=ChatMessage(content="Analysis complete", role="assistant"),
    )
    mock_node.run_stream = MagicMock(
        return_value=_collect_stream_events([complete_event]),
    )

    with patch(
        "wolfharness.capabilities.background_task.capability._generate_task_id",
        return_value="bg_asynct01",
    ):
        await capability._task(
            _wrap_in_run_context(ctx),
            agent="worker",
            message="test task",
            async_mode=True,
        )

    # Wait for background task to complete
    await asyncio.sleep(0.2)

    emitted_events = [call.args[0] for call in ctx.events.emit_event.call_args_list]
    sub_events = [e for e in emitted_events if isinstance(e, SubAgentEvent)]
    stream_complete_wrapped = [e for e in sub_events if isinstance(e.event, StreamCompleteEvent)]

    assert len(stream_complete_wrapped) == 0, (
        f"Expected 0 SubAgentEvent(StreamCompleteEvent), got {len(stream_complete_wrapped)}"
    )


# ---------------------------------------------------------------------------
# Test: async task does NOT emit StreamCompleteEvent on error
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_async_task_does_not_emit_stream_complete_on_error():
    """Verify that async task does NOT emit SubAgentEvent even when task raises an error."""
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    mock_node = pool.nodes["worker"]

    # Create a stream that raises an error
    async def _error_stream(**kwargs):
        yield PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="Starting..."))
        msg = "Simulated error"
        raise ValueError(msg)

    mock_node.run_stream = MagicMock(return_value=_error_stream())

    with patch(
        "wolfharness.capabilities.background_task.capability._generate_task_id",
        return_value="bg_asynct01",
    ):
        await capability._task(
            _wrap_in_run_context(ctx),
            agent="worker",
            message="test task",
            async_mode=True,
        )

    # Wait for background task to complete
    await asyncio.sleep(0.3)

    emitted_events = [call.args[0] for call in ctx.events.emit_event.call_args_list]
    sub_events = [e for e in emitted_events if isinstance(e, SubAgentEvent)]

    assert len(sub_events) == 0, f"Expected 0 SubAgentEvents on error, got {len(sub_events)}"


# ---------------------------------------------------------------------------
# Test: async task preserves filesystem writing
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_async_task_preserves_filesystem_writing():
    """Verify that filesystem writing still happens without event emission."""
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    mock_node = pool.nodes["worker"]
    complete_event = StreamCompleteEvent(
        message=ChatMessage(content="Written to file", role="assistant"),
    )
    mock_node.run_stream = MagicMock(
        return_value=_collect_stream_events([complete_event]),
    )

    with patch(
        "wolfharness.capabilities.background_task.capability._generate_task_id",
        return_value="bg_fswrt001",
    ):
        await capability._task(
            _wrap_in_run_context(ctx),
            agent="worker",
            message="test task",
            async_mode=True,
        )

    # Wait for background task to complete
    await asyncio.sleep(0.2)

    # Verify filesystem writing still happened
    fs_pipe_calls = ctx.internal_fs.pipe.call_args_list
    assert len(fs_pipe_calls) > 0, "Expected filesystem pipe calls to be made"

    # Verify NO SubAgentEvent emission
    emitted_events = [call.args[0] for call in ctx.events.emit_event.call_args_list]
    sub_events = [e for e in emitted_events if isinstance(e, SubAgentEvent)]
    assert len(sub_events) == 0, "Expected 0 SubAgentEvent emissions"


# ---------------------------------------------------------------------------
# Test: async task writes PartDeltaEvent content to filesystem
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_async_task_writes_part_delta_to_filesystem():
    """Verify that PartDeltaEvent content is written to filesystem."""
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    mock_node = pool.nodes["worker"]
    delta_event = PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="Hello "))
    complete_event = StreamCompleteEvent(
        message=ChatMessage(content="Hello world", role="assistant"),
    )
    mock_node.run_stream = MagicMock(
        return_value=_collect_stream_events([delta_event, complete_event]),
    )

    with patch(
        "wolfharness.capabilities.background_task.capability._generate_task_id",
        return_value="bg_delta001",
    ):
        await capability._task(
            _wrap_in_run_context(ctx),
            agent="worker",
            message="test task",
            async_mode=True,
        )

    # Wait for background task to complete
    await asyncio.sleep(0.2)

    # Verify filesystem writes happened
    fs_pipe_calls = ctx.internal_fs.pipe.call_args_list
    assert len(fs_pipe_calls) >= 1, "Expected filesystem pipe calls for delta + complete"

    # Verify NO SubAgentEvent emission
    emitted_events = [call.args[0] for call in ctx.events.emit_event.call_args_list]
    sub_events = [e for e in emitted_events if isinstance(e, SubAgentEvent)]
    assert len(sub_events) == 0, "Expected 0 SubAgentEvent emissions"


# ---------------------------------------------------------------------------
# Test: async task returns immediately (non-blocking)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_async_task_returns_immediately():
    """Verify that async task still returns immediately."""
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    mock_node = pool.nodes["worker"]
    complete_event = StreamCompleteEvent(
        message=ChatMessage(content="Done", role="assistant"),
    )
    mock_node.run_stream = MagicMock(
        return_value=_collect_stream_events([complete_event]),
    )

    with patch(
        "wolfharness.capabilities.background_task.capability._generate_task_id",
        return_value="bg_quick0001",
    ):
        result = await capability._task(
            _wrap_in_run_context(ctx),
            agent="worker",
            message="test task",
            async_mode=True,
        )

    # Should return immediately with formatted text
    assert "Status: running" in result
    assert "Task ID: bg_quick0001" in result


# ---------------------------------------------------------------------------
# Test: background task completes when parent session ends
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_async_task_completes_when_parent_session_ends():
    """Regression test: background task should complete even when
    the parent session's event emitter raises CancelledError.

    Since we no longer emit events via ctx.events.emit_event(),
    this test verifies the task completes via EventBus path.
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    # Make emit_event raise CancelledError (simulating parent session ending)
    ctx.events.emit_event = AsyncMock(side_effect=asyncio.CancelledError("Parent session ended"))

    mock_node = pool.nodes["worker"]

    # Create a stream that takes 0.2s to complete
    async def _slow_stream(**kwargs):
        yield PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="Partial..."))
        await asyncio.sleep(0.2)
        yield StreamCompleteEvent(
            message=ChatMessage(content="Task completed successfully", role="assistant")
        )

    mock_node.run_stream = MagicMock(return_value=_slow_stream())

    with patch(
        "wolfharness.capabilities.background_task.capability._generate_task_id",
        return_value="bg_parend01",
    ):
        await capability._task(
            _wrap_in_run_context(ctx),
            agent="worker",
            message="test task",
            async_mode=True,
        )

    # Wait for the task to complete
    await asyncio.sleep(0.5)

    # Task should have completed successfully, NOT been cancelled
    task_model = capability._get_session_state(_wrap_in_run_context(ctx)).task_manager.get_task(
        "bg_parend01"
    )
    assert task_model is not None
    assert task_model.status == "completed", f"Expected completed, got {task_model.status}"

    # Output file should have the content
    fs_pipe_calls = ctx.internal_fs.pipe.call_args_list
    assert len(fs_pipe_calls) > 0, "Expected filesystem pipe calls to be made"
    # The last pipe call should have the final content
    last_pipe_content = fs_pipe_calls[-1][0][1]
    if isinstance(last_pipe_content, bytes):
        last_pipe_content = last_pipe_content.decode("utf-8")
    assert "Task completed successfully" in last_pipe_content


# ---------------------------------------------------------------------------
# Test: SpawnSessionStart is still emitted
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_async_task_emits_spawn_session_start():
    """Verify that create_child_session() is called with spawn_mechanism='task'
    so that SpawnSessionStart is auto-emitted by the wolfharness framework.
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    mock_node = pool.nodes["worker"]
    complete_event = StreamCompleteEvent(
        message=ChatMessage(content="Done", role="assistant"),
    )
    mock_node.run_stream = MagicMock(
        return_value=_collect_stream_events([complete_event]),
    )

    with patch(
        "wolfharness.capabilities.background_task.capability._generate_task_id",
        return_value="bg_spawn01",
    ):
        await capability._task(
            _wrap_in_run_context(ctx),
            agent="worker",
            message="test task",
            async_mode=True,
        )

    await asyncio.sleep(0.2)

    # SpawnSessionStart is now auto-emitted by create_child_session().
    # Verify the call was made with the correct parameters that trigger
    # the auto-emission in the real (non-mocked) implementation.
    ctx.create_child_session.assert_awaited_once()
    call_kwargs = ctx.create_child_session.call_args.kwargs
    assert call_kwargs.get("spawn_mechanism") == "task"
    assert call_kwargs.get("agent_name") == "worker"
    assert call_kwargs.get("tool_call_id") == "tc_001"


# ---------------------------------------------------------------------------
# Test: async task completion callback calls inject_prompt
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_async_task_completion_callback_injects_prompt():
    """Verify that task completion still calls session_pool.inject_prompt()."""
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    mock_node = pool.nodes["worker"]
    complete_event = StreamCompleteEvent(
        message=ChatMessage(content="Done", role="assistant"),
    )
    mock_node.run_stream = MagicMock(
        return_value=_collect_stream_events([complete_event]),
    )

    with patch(
        "wolfharness.capabilities.background_task.capability._generate_task_id",
        return_value="bg_inject01",
    ):
        await capability._task(
            _wrap_in_run_context(ctx),
            agent="worker",
            message="test task",
            async_mode=True,
        )

    # Wait for background task and callback to complete + 500ms debounce
    await _wait_until_called(pool.session_pool.followup)

    # Verify followup was called (not steer)
    pool.session_pool.followup.assert_awaited()

    call_args = pool.session_pool.followup.call_args
    assert call_args is not None
    assert call_args[0][0] == "ses_parent_123"
