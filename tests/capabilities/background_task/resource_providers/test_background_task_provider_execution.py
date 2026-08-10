"""Tests for BackgroundTaskCapability task() execution paths (sync and async)."""

# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
# Mock-heavy test code: assigning to spec'd attributes, accessing mock call_args_list,
# and accessing event.message.content through union types are all expected.

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic_ai import RunContext
import pytest

from wolfharness import ChatMessage
from wolfharness.agents.base_agent import BaseAgent
from wolfharness.agents.context import AgentContext
from wolfharness.agents.events import (
    RunErrorEvent,
    RunFailedEvent,
    StreamCompleteEvent,
    SubAgentEvent,
    ToolCallCompleteEvent,
    ToolCallStartEvent,
)
from wolfharness.capabilities.background_task.capability import (
    MAX_DELEGATION_DEPTH,
    BackgroundTaskCapability,
    _generate_task_id,
)
from wolfharness.capabilities.background_task.types import BackgroundTask
from wolfharness.delegation import AgentPool, BaseTeam
from wolfharness.tools.exceptions import ToolError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrap_in_run_context(agent_ctx):
    """Wrap an AgentContext in a mock RunContext for capability tool methods."""
    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = agent_ctx
    run_ctx.tool_call_id = agent_ctx.tool_call_id
    return run_ctx


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

    pool.manifest = MagicMock()
    pool.manifest.agents = nodes
    pool.nodes = nodes
    pool.agent_configs = nodes
    pool.all_agents = list(nodes.items())
    pool.teams = {}
    pool.sessions = None

    mock_session_pool = MagicMock()

    async def _get_or_create_session_agent(agent_name: str, agent_type: str):
        return nodes.get(agent_name, next(iter(nodes.values())))

    mock_session_pool.sessions = MagicMock()
    mock_session_pool.sessions.get_or_create_session_agent = AsyncMock(
        side_effect=_get_or_create_session_agent,
    )
    mock_session_pool.steer = AsyncMock()
    mock_session_pool.followup = AsyncMock(return_value=True)

    def _run_stream_proxy(child_session_id: str, formatted_prompt: str, **kwargs: Any):
        for node in nodes.values():
            if hasattr(node, "run_stream"):
                return node.run_stream(
                    formatted_prompt,
                    deps=MagicMock(),
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
        queue = asyncio.Queue()
        _queues[session_id] = queue
        return queue

    async def _send_message(session_id: str, prompt: str, input_provider=None, **kwargs: Any):
        queue = _queues.get(session_id)
        if queue is None:
            queue = asyncio.Queue()
            _queues[session_id] = queue
        for node in nodes.values():
            if hasattr(node, "run_stream"):
                stream = node.run_stream(
                    prompt,
                    deps=MagicMock(),
                    session_id=session_id,
                    parent_session_id="ses_parent_123",
                    depth=1,
                    message_history=MagicMock(),
                )

                async def _feed(stream=stream, queue=queue):
                    async for event in stream:
                        await queue.put(event)
                    await queue.put(None)

                feed_task = asyncio.create_task(_feed())
                assert feed_task is not None
                break
        return MagicMock()

    mock_session_pool.send_message = AsyncMock(side_effect=_send_message)
    mock_session_pool.inject_prompt = AsyncMock()
    mock_session_pool.event_bus = MagicMock()
    mock_session_pool.event_bus.subscribe = AsyncMock(side_effect=_subscribe)
    mock_session_pool.event_bus.unsubscribe = AsyncMock()
    pool.session_pool = mock_session_pool
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
    ctx.input_provider = None

    return ctx


async def _collect_stream_events(events: list[Any]):
    """Create an async generator that yields the given events."""
    for event in events:
        yield event


async def _make_event_queue(events: list[Any]) -> asyncio.Queue[Any]:
    """Create an asyncio.Queue pre-filled with events and a None sentinel."""
    q: asyncio.Queue[Any] = asyncio.Queue()
    for event in events:
        await q.put(event)
    await q.put(None)  # sentinel
    return q


# ---------------------------------------------------------------------------
# _generate_task_id tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_generate_task_id_format():
    """Test that _generate_task_id produces bg_ + 8 hex chars format."""
    task_id = _generate_task_id("analyze equipment")
    assert task_id.startswith("bg_")
    # After "bg_", should be exactly 8 hex characters
    hex_part = task_id[3:]
    assert len(hex_part) == 12
    assert all(c in "0123456789abcdef" for c in hex_part)


@pytest.mark.unit
def test_generate_task_id_uniqueness():
    """Test that _generate_task_id produces unique IDs."""
    ids = {_generate_task_id("test") for _ in range(100)}
    assert len(ids) == 100, "Each call should produce a unique ID"


@pytest.mark.unit
def test_generate_task_id_ignores_description():
    """Test that _generate_task_id ignores the description parameter."""
    id1 = _generate_task_id("task A")
    id2 = _generate_task_id("task B")
    # Both should be valid bg_ format, regardless of description
    assert id1.startswith("bg_")
    assert id2.startswith("bg_")
    # They should be different (random)
    assert id1 != id2


# ---------------------------------------------------------------------------
# Sync path: validation errors
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_task_no_pool_returns_error():
    """Test that task raises ToolError when no pool is available."""
    capability = BackgroundTaskCapability(schemas=None)
    ctx = _make_agent_context(pool=None)

    with pytest.raises(ToolError, match="No agent pool available"):
        await capability._task(_wrap_in_run_context(ctx), agent="worker", message="test task")


@pytest.mark.unit
async def test_task_agent_not_found_returns_error():
    """Test that task returns an error string when the requested agent doesn't exist."""
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    result = await capability._task(
        _wrap_in_run_context(ctx), agent="nonexistent", message="test task"
    )
    assert "Agent 'nonexistent' not found" in result


# ---------------------------------------------------------------------------
# Sync path: delegation depth
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_task_delegation_depth_limit():
    """Test that task returns an error when max delegation depth is reached."""
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool, data={"delegation_depth": MAX_DELEGATION_DEPTH})

    result = await capability._task(_wrap_in_run_context(ctx), agent="worker", message="test task")
    assert f"Max delegation depth ({MAX_DELEGATION_DEPTH}) reached" in result


# ---------------------------------------------------------------------------
# Sync path: SpawnSessionStart emission
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_task_sync_emits_spawn_session_start():
    """Test that create_child_session is called with spawn_mechanism='task'
    so SpawnSessionStart is auto-emitted by the wolfharness framework.
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    # Mock session_pool.run_stream to yield a StreamCompleteEvent
    complete_event = StreamCompleteEvent(
        message=ChatMessage(content="Task done", role="assistant"),
    )
    pool.session_pool.run_stream = MagicMock(
        return_value=_collect_stream_events([complete_event]),
    )

    await capability._task(_wrap_in_run_context(ctx), agent="worker", message="test task")

    # SpawnSessionStart is now auto-emitted by create_child_session().
    # Verify the call was made with the correct parameters.
    ctx.create_child_session.assert_awaited_once()
    call_kwargs = ctx.create_child_session.call_args.kwargs
    assert call_kwargs.get("spawn_mechanism") == "task"
    assert call_kwargs.get("agent_name") == "worker"
    assert call_kwargs.get("tool_call_id") == "tc_001"


# ---------------------------------------------------------------------------
# Sync path: no SubAgentEvent wrapping (events via EventBus)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_task_sync_does_not_emit_subagent_events():
    """Test that sync path does NOT wrap stream events as SubAgentEvent.

    Events are broadcast via EventBus by TurnRunner; the sync path only
    drains the stream to extract the final result.
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    # Create a generic event and a StreamCompleteEvent
    generic_event = MagicMock()
    generic_event.__class__ = type("GenericEvent", (), {})
    complete_event = StreamCompleteEvent(
        message=ChatMessage(content="Final result", role="assistant"),
    )

    pool.session_pool.run_stream = MagicMock(
        return_value=_collect_stream_events([generic_event, complete_event]),
    )

    result = await capability._task(_wrap_in_run_context(ctx), agent="worker", message="test task")

    # Check that NO SubAgentEvents were emitted
    emitted_events = [call.args[0] for call in ctx.events.emit_event.call_args_list]
    sub_events = [e for e in emitted_events if isinstance(e, SubAgentEvent)]
    assert len(sub_events) == 0, f"Expected 0 SubAgentEvents, got {len(sub_events)}"
    # SpawnSessionStart is now auto-emitted by create_child_session(), not by the capability
    # Result should still be extracted correctly
    assert result == "Final result"


# ---------------------------------------------------------------------------
# Sync path: returns final text result
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_task_sync_returns_final_text_from_stream_complete():
    """Test that sync path returns final text from StreamCompleteEvent."""
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    complete_event = StreamCompleteEvent(
        message=ChatMessage(content="The analysis result", role="assistant"),
    )
    pool.session_pool.run_stream = MagicMock(
        return_value=_collect_stream_events([complete_event]),
    )

    result = await capability._task(_wrap_in_run_context(ctx), agent="worker", message="analyze")

    assert result == "The analysis result"


@pytest.mark.unit
async def test_task_sync_captures_attempt_completion_start():
    """Test that sync path captures result from attempt_completion ToolCallStartEvent."""
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    mock_node = pool.nodes["worker"]
    attempt_start = ToolCallStartEvent(
        tool_call_id="tc_attempt_001",
        tool_name="attempt_completion",
        title="Completing task",
        raw_input={"result": "Completion result from attempt"},
    )
    mock_node.run_stream = MagicMock(
        return_value=_collect_stream_events([attempt_start]),
    )

    result = await capability._task(_wrap_in_run_context(ctx), agent="worker", message="analyze")

    assert result == "Completion result from attempt"


@pytest.mark.unit
async def test_task_sync_captures_attempt_completion_complete():
    """Test that sync path captures result from attempt_completion ToolCallCompleteEvent."""
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    mock_node = pool.nodes["worker"]
    attempt_complete = ToolCallCompleteEvent(
        tool_name="attempt_completion",
        tool_call_id="tc_attempt_002",
        tool_input={"result": "Done from complete"},
        tool_result="Done from complete",
        agent_name="worker",
        message_id="msg_001",
    )
    mock_node.run_stream = MagicMock(
        return_value=_collect_stream_events([attempt_complete]),
    )

    result = await capability._task(_wrap_in_run_context(ctx), agent="worker", message="analyze")

    assert result == "Done from complete"


# ---------------------------------------------------------------------------
# Sync path: child session creation
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_task_creates_child_session():
    """Test that task creates a child session via ctx.create_child_session."""
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

    await capability._task(_wrap_in_run_context(ctx), agent="worker", message="test")

    ctx.create_child_session.assert_awaited_once_with(
        agent_name="worker",
        agent_type="native",
        parent_session_id="ses_parent_123",
        spawn_mechanism="task",
        description="",
        tool_call_id="tc_001",
    )


# ---------------------------------------------------------------------------
# Sync path: skill injection
# ---------------------------------------------------------------------------


@pytest.mark.unit
@patch(
    "wolfharness.capabilities.background_task.capability.load_skill_for_node",
    new_callable=AsyncMock,
)
async def test_task_injects_skills(mock_load_skill):
    """Test that task injects skills into the formatted prompt."""
    mock_load_skill.return_value = "Skill instructions content"

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

    await capability._task(
        _wrap_in_run_context(ctx),
        agent="worker",
        message="test task",
        load_skills=["diagnostics"],
    )

    # Verify load_skill_for_node was called with target agent name
    mock_load_skill.assert_awaited_once_with(ctx, "diagnostics", node_name="worker")

    # Verify the prompt passed to run_stream includes skill XML
    call_args = mock_node.run_stream.call_args
    prompt = call_args[0][0]  # First positional arg
    assert '<skill-instruction name="diagnostics">' in prompt
    assert "Skill instructions content" in prompt


# ---------------------------------------------------------------------------
# Sync path: prompt formatting with expected_output
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_task_includes_expected_output_in_prompt():
    """Test that task includes expected_output in the formatted prompt."""
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

    await capability._task(
        _wrap_in_run_context(ctx),
        agent="worker",
        message="Analyze motor vibration",
        expected_output="Vibration analysis report with RMS values",
    )

    call_args = mock_node.run_stream.call_args
    prompt = call_args[0][0]
    assert "<task>" in prompt
    assert "Analyze motor vibration" in prompt
    assert "<expected_output>" in prompt
    assert "Vibration analysis report with RMS values" in prompt


# ---------------------------------------------------------------------------
# Async path: returns JSON string
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_task_async_returns_formatted_text():
    """Test that async path returns formatted text with task_id, session_id, status."""
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
        return_value="bg_testtask",
    ):
        result = await capability._task(
            _wrap_in_run_context(ctx),
            agent="worker",
            message="test task",
            async_mode=True,
        )

    # Parse the formatted text result
    assert "Task ID: bg_testtask" in result
    assert "Session ID: ses_child_456" in result
    assert "Status: running" in result
    assert "Description: worker" in result


# ---------------------------------------------------------------------------
# Async path: BackgroundTask registered in manager
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_task_async_registers_background_task():
    """Test that async path registers a BackgroundTask in the manager."""
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
        return_value="bg_regtest1",
    ):
        await capability._task(
            _wrap_in_run_context(ctx),
            agent="worker",
            message="test task",
            async_mode=True,
        )

    # Verify the task was registered in the manager
    task_model = capability._get_session_state(_wrap_in_run_context(ctx)).task_manager.get_task(
        "bg_regtest1"
    )
    assert task_model is not None
    assert isinstance(task_model, BackgroundTask)
    assert task_model.agent_or_team == "worker"
    assert task_model.child_session_id == "ses_child_456"
    assert task_model.output_file == "/tasks/bg_regtest1/output.md"


# ---------------------------------------------------------------------------
# Async path: task directory created on internal filesystem
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_task_async_creates_task_directory():
    """Test that async path creates the task directory on internal filesystem."""
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
        return_value="bg_dirtst01",
    ):
        await capability._task(
            _wrap_in_run_context(ctx),
            agent="worker",
            message="test task",
            async_mode=True,
        )

    ctx.internal_fs.mkdirs.assert_called_with("/tasks/bg_dirtst01", exist_ok=True)


# ---------------------------------------------------------------------------
# Async path: SpawnSessionStart still emitted
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_task_async_emits_spawn_session_start():
    """Test that async path calls create_child_session with spawn_mechanism='task'
    so SpawnSessionStart is auto-emitted by the wolfharness framework.
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
        return_value="bg_spawnt01",
    ):
        await capability._task(
            _wrap_in_run_context(ctx),
            agent="worker",
            message="test task",
            async_mode=True,
        )

    # SpawnSessionStart is now auto-emitted by create_child_session()
    ctx.create_child_session.assert_awaited_once()
    call_kwargs = ctx.create_child_session.call_args.kwargs
    assert call_kwargs.get("spawn_mechanism") == "task"
    assert call_kwargs.get("agent_name") == "worker"


# Source type detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_task_detects_team_parallel_source_type():
    """Test that task delegates to Team nodes correctly."""
    capability = BackgroundTaskCapability(schemas=None)
    mock_team = MagicMock(spec=BaseTeam)
    mock_team.type = "team"
    mock_team.description = "A parallel team"
    pool = _make_mock_pool(nodes={"review_team": mock_team})
    ctx = _make_agent_context(pool=pool)

    complete_event = StreamCompleteEvent(
        message=ChatMessage(content="Done", role="assistant"),
    )
    mock_team.run_stream = MagicMock(
        return_value=_collect_stream_events([complete_event]),
    )

    await capability._task(_wrap_in_run_context(ctx), agent="review_team", message="review code")

    # SpawnSessionStart source_type is now handled inside create_child_session()
    ctx.create_child_session.assert_awaited_once()
    assert ctx.create_child_session.call_args.kwargs.get("agent_name") == "review_team"


@pytest.mark.unit
async def test_task_detects_team_sequential_source_type():
    """Test that task delegates to TeamRun nodes correctly."""
    capability = BackgroundTaskCapability(schemas=None)
    mock_team_run = MagicMock(spec=BaseTeam)
    mock_team_run.type = "team"
    mock_team_run.description = "A sequential team"
    pool = _make_mock_pool(nodes={"pipeline": mock_team_run})
    ctx = _make_agent_context(pool=pool)

    complete_event = StreamCompleteEvent(
        message=ChatMessage(content="Done", role="assistant"),
    )
    mock_team_run.run_stream = MagicMock(
        return_value=_collect_stream_events([complete_event]),
    )

    await capability._task(_wrap_in_run_context(ctx), agent="pipeline", message="process data")

    # SpawnSessionStart source_type is now handled inside create_child_session()
    ctx.create_child_session.assert_awaited_once()
    assert ctx.create_child_session.call_args.kwargs.get("agent_name") == "pipeline"


# ---------------------------------------------------------------------------
# Sync path: nested SubAgentEvent is ignored (events via EventBus)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_task_sync_ignores_nested_subagent_events():
    """Test that sync path ignores nested SubAgentEvents.

    Nested events are broadcast via their respective child sessions'
    EventBus; the sync path only drains its own stream for the final result.
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    mock_node = pool.nodes["worker"]
    # Create a nested SubAgentEvent (from a further-delegated subagent)
    inner_event = MagicMock()
    nested_sub_event = SubAgentEvent(
        source_name="inner_agent",
        source_type="agent",
        event=inner_event,
        depth=1,
        child_session_id="ses_inner_789",
        model_id="test:inner-model",
        mode="code",
    )
    complete_event = StreamCompleteEvent(
        message=ChatMessage(content="Final", role="assistant"),
    )
    mock_node.run_stream = MagicMock(
        return_value=_collect_stream_events([nested_sub_event, complete_event]),
    )

    result = await capability._task(_wrap_in_run_context(ctx), agent="worker", message="test")

    assert result == "Final"

    # No SubAgentEvents should be emitted
    emitted_events = [call.args[0] for call in ctx.events.emit_event.call_args_list]
    sub_events = [e for e in emitted_events if isinstance(e, SubAgentEvent)]
    assert len(sub_events) == 0, f"Expected 0 SubAgentEvents, got {len(sub_events)}"


# ---------------------------------------------------------------------------
# Sync path: deps propagation with incremented depth
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_task_sync_passes_incremented_depth_in_deps():
    """Test that sync path passes deps with incremented delegation_depth."""
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool, data={"delegation_depth": 2, "custom_key": "value"})

    mock_node = pool.nodes["worker"]
    complete_event = StreamCompleteEvent(
        message=ChatMessage(content="Done", role="assistant"),
    )
    mock_node.run_stream = MagicMock(
        return_value=_collect_stream_events([complete_event]),
    )

    await capability._task(_wrap_in_run_context(ctx), agent="worker", message="test")

    call_args = pool.session_pool.run_stream.call_args
    deps = call_args.kwargs.get("deps") or call_args[1].get("deps")
    assert deps["delegation_depth"] == 3
    assert deps["custom_key"] == "value"


# ---------------------------------------------------------------------------
# Sync path: node model_id extraction
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_task_extracts_model_id_from_base_agent():
    """Test that task calls create_child_session correctly for BaseAgent nodes.

    Model ID extraction was removed — SpawnSessionStart (including model_id)
    is now auto-emitted by create_child_session() in the wolfharness framework.
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

    await capability._task(_wrap_in_run_context(ctx), agent="worker", message="test")

    # create_child_session is called with the right agent_name
    ctx.create_child_session.assert_awaited_once()
    assert ctx.create_child_session.call_args.kwargs.get("agent_name") == "worker"


# ---------------------------------------------------------------------------
# Sync path: error event handling (RunErrorEvent / RunFailedEvent)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_task_sync_returns_error_on_run_error_event():
    """_task_sync should return an error result when stream yields RunErrorEvent."""
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    error_event = RunErrorEvent(message="Model overloaded", agent_name="worker")
    pool.session_pool.run_stream = MagicMock(
        return_value=_collect_stream_events([error_event]),
    )

    result = await capability._task(_wrap_in_run_context(ctx), agent="worker", message="test task")
    assert result.startswith("Error:")
    assert "Model overloaded" in result


@pytest.mark.unit
async def test_task_sync_returns_error_on_run_failed_event():
    """_task_sync should return an error result when stream yields RunFailedEvent."""
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    failed_event = RunFailedEvent(
        run_id="run_001",
        session_id="ses_child",
        exception=RuntimeError("Subagent crashed"),
    )
    pool.session_pool.run_stream = MagicMock(
        return_value=_collect_stream_events([failed_event]),
    )

    result = await capability._task(_wrap_in_run_context(ctx), agent="worker", message="test task")
    assert result.startswith("Task failed:")
    assert "Subagent crashed" in result


@pytest.mark.unit
async def test_task_sync_catches_stream_exception():
    """_task_sync should catch exceptions raised by the stream iterator."""
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    async def _failing_stream():
        raise ValueError("Stream iterator crashed")
        yield  # unreachable

    pool.session_pool.run_stream = MagicMock(return_value=_failing_stream())

    result = await capability._task(_wrap_in_run_context(ctx), agent="worker", message="test task")
    assert result.startswith("Task failed:")
    assert "Stream iterator crashed" in result


@pytest.mark.unit
async def test_task_sync_returns_fallback_when_no_result():
    """_task_sync should return fallback error when stream produces no result."""
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool)

    # Empty stream — no terminal event at all
    pool.session_pool.run_stream = MagicMock(
        return_value=_collect_stream_events([]),
    )

    result = await capability._task(_wrap_in_run_context(ctx), agent="worker", message="test task")
    assert result == "Error: No result produced"
