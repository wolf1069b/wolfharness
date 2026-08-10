"""Integration tests for BackgroundTaskProvider end-to-end behavior.

Tests cover the full lifecycle of the BackgroundTaskProvider:
- Provider instantiation from config-style kwargs
- Synchronous task delegation via task(async_mode=False)
- Async task creation and output retrieval via background_output
- Cancel flow via background_cancel
- Cleanup after shutdown via BackgroundTaskManager.shutdown()
"""

from __future__ import annotations

import asyncio
import re
from unittest.mock import AsyncMock, MagicMock

# Imports for force_retrieval graph integration test
from pydantic_ai import Agent as PydanticAgent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
import pytest

from wolfharness import Agent, AgentContext, ChatMessage
from wolfharness.agents.context import AgentRunContext
from wolfharness.agents.events import StreamCompleteEvent
from wolfharness.capabilities.background_task import (
    BackgroundTaskCapability as BackgroundTaskProvider,
)
from wolfharness.capabilities.background_task.capability import (
    BackgroundTaskCapability,
)
from wolfharness.capabilities.background_task.manager import BackgroundTaskManager
from wolfharness.capabilities.background_task.types import BackgroundTask
from wolfharness.delegation import AgentPool
from wolfharness.tools.exceptions import ToolError


def _wrap_in_run_context(agent_ctx):
    """Wrap an AgentContext in a mock RunContext for capability tool methods.

    Uses ``spec=RunContext`` so that ``isinstance(ctx, RunContext)`` returns
    ``True`` in ``_get_session_state``, ensuring the same session state is
    shared between ``_task`` and ``_background_output``/``_background_cancel``.
    """
    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = agent_ctx
    run_ctx.tool_call_id = agent_ctx.tool_call_id
    return run_ctx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pool_with_agent() -> AgentPool:
    """Create a mock AgentPool with a streaming-capable agent.

    Uses MagicMock (not AsyncMock) for the pool itself so that
    ``bool(pool)`` returns True, matching real AgentPool behavior.
    """
    pool = MagicMock(spec=AgentPool)

    mock_agent = MagicMock(spec=Agent)
    mock_agent.type = "agent"
    mock_agent.description = "A test agent"
    mock_agent.name = "test_agent"

    pool.manifest = MagicMock()
    pool.manifest.agents = {"test_agent": mock_agent}
    pool.nodes = {"test_agent": mock_agent}
    pool.agent_configs = {"test_agent": mock_agent}
    pool.file_ops = MagicMock()

    # Set up a mock session_pool for the SessionPool-only code path
    mock_session_pool = MagicMock()

    async def _empty_stream(*args, **kwargs):
        return
        yield

    mock_session_pool.run_stream = MagicMock(side_effect=_empty_stream)
    mock_session_pool.send_message = AsyncMock(return_value=MagicMock())
    mock_session_pool.inject_prompt = AsyncMock()
    mock_session_pool.event_bus = MagicMock()
    mock_session_pool.event_bus.subscribe = AsyncMock(return_value=asyncio.Queue())
    mock_session_pool.event_bus.unsubscribe = AsyncMock()

    # Set up sessions.get_or_create_session_agent for task() method
    mock_session_pool.sessions = MagicMock()
    mock_session_pool.sessions.get_or_create_session_agent = AsyncMock(return_value=mock_agent)

    pool.session_pool = mock_session_pool

    return pool


@pytest.fixture
def mock_internal_fs() -> MagicMock:
    """Create a mock internal filesystem for async task output."""
    fs = MagicMock()
    fs.mkdirs = MagicMock()
    fs.pipe = MagicMock()
    return fs


# ---------------------------------------------------------------------------
# force_retrieval graph integration test helpers
# ---------------------------------------------------------------------------


def _make_force_retrieval_model_fn(seen_tool_choices: list):
    """Create a FunctionModel callback that verifies tool_choice enforcement.

    The model function inspects ``info.model_settings['tool_choice']`` — this is
    the value injected by ``BackgroundTaskCapability.get_model_settings()`` when
    ``force_retrieval=True`` and ``_pending_retrievals`` is non-empty.

    Flow:
    1. First call: pending_retrievals is non-empty → tool_choice=['background_output']
       → model MUST call background_output (tool_choice forces it)
    2. Second call: pending_retrievals is empty (background_output cleared it)
       → tool_choice is absent → model produces final text
    """

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tc = (info.model_settings or {}).get("tool_choice")
        seen_tool_choices.append(tc)

        if tc == ["background_output"]:
            # tool_choice forces us to call background_output
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="background_output",
                        args={"task_id": "bg_test"},
                    ),
                ],
                model_name="test",
            )
        # No tool_choice restriction → produce final text
        return ModelResponse(
            parts=[TextPart(content="All tasks retrieved")],
            model_name="test",
        )

    return respond


# ---------------------------------------------------------------------------
# Provider instantiation from config
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_provider_instantiation_from_config_kwargs():
    """Test that BackgroundTaskProvider can be instantiated with config-style kwargs."""
    capability = BackgroundTaskCapability(
        schemas=None,
        enabled_tools=None,
    )

    assert capability is not None
    # The new API uses per-session state; verify a session state can be created
    # and contains a BackgroundTaskManager.
    mock_ctx = MagicMock()
    state = capability._get_session_state(mock_ctx)
    assert isinstance(state.task_manager, BackgroundTaskManager)

    toolset = capability.get_toolset()
    tool_names = set(toolset.tools.keys())
    assert tool_names == {"task", "background_output", "background_cancel", "steer_task"}


@pytest.mark.integration
async def test_provider_partial_tool_enablement():
    """Test that enabled_tools correctly filters tools for specific agent roles."""
    # Coordinating agent needs all three tools
    full_capability = BackgroundTaskCapability(
        schemas=None,
        enabled_tools=None,
    )
    full_toolset = full_capability.get_toolset()
    assert len(full_toolset.tools) == 4

    # Subagent only needs task (sync delegation back)
    task_only_capability = BackgroundTaskCapability(
        schemas=None,
        enabled_tools=["task"],
    )
    task_only_toolset = task_only_capability.get_toolset()
    assert len(task_only_toolset.tools) == 1
    assert "task" in task_only_toolset.tools


# ---------------------------------------------------------------------------
# Synchronous task delegation
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_sync_task_delegation(mock_pool_with_agent: AgentPool):
    """Test that task(async_mode=False) delegates synchronously and returns result."""
    capability = BackgroundTaskCapability(schemas=None)
    agent = Agent(name="parent_agent", model="test")
    agent_ctx = AgentContext(node=agent, pool=mock_pool_with_agent, input_provider=MagicMock())

    result = await capability._task(
        _wrap_in_run_context(agent_ctx),
        agent="test_agent",
        message="Diagnose hydraulic pressure loss",
        expected_output="Root cause and recommended fix",
        async_mode=False,
    )

    assert "No result produced" in result  # Mock agent produces empty final_result


@pytest.mark.integration
async def test_sync_task_delegation_with_streaming_result(mock_pool_with_agent: AgentPool):
    """Test sync task delegation captures result from attempt_completion."""
    capability = BackgroundTaskCapability(schemas=None)

    # Set up session_pool mock so the SessionPool path is used
    async def _session_stream(*args, **kwargs):
        yield StreamCompleteEvent(
            message=ChatMessage(content="Hydraulic pump failure detected", role="assistant")
        )

    mock_pool_with_agent.session_pool = MagicMock()
    mock_pool_with_agent.session_pool.run_stream = _session_stream
    mock_pool_with_agent.session_pool.sessions = MagicMock()
    mock_pool_with_agent.session_pool.sessions.get_or_create_session_agent = AsyncMock(
        return_value=mock_pool_with_agent.nodes["test_agent"],
    )

    agent = Agent(name="parent_agent", model="test")
    agent_ctx = AgentContext(node=agent, pool=mock_pool_with_agent, input_provider=MagicMock())

    result = await capability._task(
        _wrap_in_run_context(agent_ctx),
        agent="test_agent",
        message="Diagnose hydraulic pressure loss",
        expected_output="Root cause",
        async_mode=False,
    )

    assert "Hydraulic pump failure detected" in result


@pytest.mark.integration
async def test_sync_task_no_pool_raises_tool_error():
    """Test that sync task raises ToolError when pool is None."""
    capability = BackgroundTaskCapability(schemas=None)
    agent = Agent(name="parent_agent", model="test")
    agent_ctx = AgentContext(node=agent, pool=None)

    with pytest.raises(ToolError, match="No agent pool available"):
        await capability._task(
            _wrap_in_run_context(agent_ctx),
            agent="test_agent",
            message="Test",
            expected_output="Result",
        )


@pytest.mark.integration
async def test_sync_task_unknown_agent_raises_tool_error(mock_pool_with_agent: AgentPool):
    """Test that sync task returns an error string when mode is not in pool."""
    capability = BackgroundTaskCapability(schemas=None)
    agent = Agent(name="parent_agent", model="test")
    agent_ctx = AgentContext(node=agent, pool=mock_pool_with_agent, input_provider=MagicMock())

    result = await capability._task(
        _wrap_in_run_context(agent_ctx),
        agent="nonexistent",
        message="Test",
        expected_output="Result",
    )
    assert "Agent 'nonexistent' not found" in result


# ---------------------------------------------------------------------------
# Async task creation + output query
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_async_task_creation_returns_metadata(
    mock_pool_with_agent: AgentPool, mock_internal_fs: MagicMock
):
    """Test that task(async_mode=True) returns task_id and metadata immediately."""
    capability = BackgroundTaskCapability(schemas=None)
    agent = Agent(name="parent_agent", model="test")

    # Set up agent context with internal filesystem.
    # ctx.internal_fs is a property that delegates to ctx.node.internal_fs,
    # so we must set it on the agent (node), not the context.
    agent._internal_fs = mock_internal_fs

    agent_ctx = AgentContext(node=agent, pool=mock_pool_with_agent, input_provider=MagicMock())

    # Override run_stream to produce a quick completion
    mock_node = mock_pool_with_agent.nodes["test_agent"]

    async def _quick_stream(prompt=None, deps=None, **kwargs):
        yield StreamCompleteEvent(
            message=ChatMessage(content="Background analysis complete", role="assistant")
        )

    mock_node.run_stream = _quick_stream
    mock_node.model_name = "test-model"

    result = await capability._task(
        _wrap_in_run_context(agent_ctx),
        agent="test_agent",
        message="Run long diagnostic in background",
        expected_output="Full diagnostic report",
        async_mode=True,
        title="Background diagnostic",
    )

    task_id_match = re.search(r"Task ID: (\S+)", result)
    assert task_id_match is not None
    assert task_id_match.group(1)

    session_match = re.search(r"Session ID: (\S+)", result)
    assert session_match is not None

    assert "Status: running" in result

    # Verify task directory was created on internal filesystem
    mock_internal_fs.mkdirs.assert_called_once()


@pytest.mark.integration
async def test_background_output_for_running_task(
    mock_pool_with_agent: AgentPool, mock_internal_fs: MagicMock
):
    """Test that background_output returns status for a running task without blocking."""
    capability = BackgroundTaskCapability(schemas=None)
    agent = Agent(name="parent_agent", model="test")

    # ctx.internal_fs delegates to ctx.node.internal_fs
    agent._internal_fs = mock_internal_fs
    agent_ctx = AgentContext(node=agent, pool=mock_pool_with_agent, input_provider=MagicMock())

    # Create a task that stays running (doesn't complete immediately)
    mock_node = mock_pool_with_agent.nodes["test_agent"]

    async def _slow_stream(prompt=None, deps=None, **kwargs):
        await asyncio.sleep(10)  # Long-running task
        yield StreamCompleteEvent(message=ChatMessage(content="Finally done", role="assistant"))

    mock_node.run_stream = _slow_stream
    mock_node.model_name = "test-model"

    result = await capability._task(
        _wrap_in_run_context(agent_ctx),
        agent="test_agent",
        message="Long running task",
        expected_output="Result",
        async_mode=True,
    )

    task_id_match = re.search(r"Task ID: (\S+)", result)
    assert task_id_match is not None
    task_id = task_id_match.group(1)

    # Query output without blocking
    output = await capability._background_output(
        _wrap_in_run_context(agent_ctx), task_id=task_id, block=False
    )

    # Should indicate task is running, pending, or completed (mock may complete quickly)
    assert "running" in output or "pending" in output or "completed" in output


@pytest.mark.integration
async def test_background_output_for_completed_task():
    """Test that background_output returns result for a completed task."""
    capability = BackgroundTaskCapability(schemas=None)

    agent = Agent(name="parent_agent", model="test")
    agent_ctx = AgentContext(node=agent, pool=None)

    # Get session state and register a completed task in the manager
    state = capability._get_session_state(agent_ctx)
    task_model = BackgroundTask(
        id="test-completed-task",
        description="Test completed task",
        agent_or_team="test_agent",
        prompt="Test prompt",
        parent_session_id=None,
        child_session_id=None,
        status="completed",
        result="Diagnosis: bearing wear detected",
    )
    state.task_manager.register_task(task_model)

    output = await capability._background_output(
        _wrap_in_run_context(agent_ctx), task_id="test-completed-task"
    )

    assert "completed" in output
    assert "Diagnosis: bearing wear detected" in output


@pytest.mark.integration
async def test_background_output_for_nonexistent_task():
    """Test that background_output returns not-found for unknown task_id."""
    capability = BackgroundTaskCapability(schemas=None)
    agent = Agent(name="parent_agent", model="test")
    agent_ctx = AgentContext(node=agent, pool=None)

    output = await capability._background_output(
        _wrap_in_run_context(agent_ctx), task_id="nonexistent-id"
    )

    assert "not found" in output


# ---------------------------------------------------------------------------
# Cancel flow
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_cancel_pending_task():
    """Test cancelling a task that is still pending."""
    capability = BackgroundTaskCapability(schemas=None)

    agent = Agent(name="parent_agent", model="test")
    agent_ctx = AgentContext(node=agent, pool=None)

    state = capability._get_session_state(agent_ctx)
    task_model = BackgroundTask(
        id="test-pending-task",
        description="Pending task",
        agent_or_team="test_agent",
        prompt="Test prompt",
        parent_session_id=None,
        child_session_id=None,
        status="pending",
    )
    state.task_manager.register_task(task_model)

    result = await capability._background_cancel(
        _wrap_in_run_context(agent_ctx), task_id="test-pending-task"
    )

    assert "cancelled" in result.lower()
    assert task_model.status == "cancelled"


@pytest.mark.integration
async def test_cancel_all_tasks():
    """Test cancelling all running tasks via cancel_all=True."""
    capability = BackgroundTaskCapability(schemas=None)

    agent = Agent(name="parent_agent", model="test")
    agent_ctx = AgentContext(node=agent, pool=None)

    state = capability._get_session_state(agent_ctx)

    # Register two pending tasks
    for i in range(2):
        task_model = BackgroundTask(
            id=f"test-batch-task-{i}",
            description=f"Batch task {i}",
            agent_or_team="test_agent",
            prompt="Test prompt",
            parent_session_id=None,
            child_session_id=None,
            status="pending",
        )
        state.task_manager.register_task(task_model)

    result = await capability._background_cancel(_wrap_in_run_context(agent_ctx), cancel_all=True)

    assert "Cancelled 2 background task(s)" in result


@pytest.mark.integration
async def test_cancel_no_args_raises_tool_error():
    """Test that background_cancel raises ToolError without task_id or cancel_all."""
    capability = BackgroundTaskCapability(schemas=None)
    agent = Agent(name="parent_agent", model="test")
    agent_ctx = AgentContext(node=agent, pool=None)

    with pytest.raises(ToolError, match="Either task_id or cancel_all"):
        await capability._background_cancel(_wrap_in_run_context(agent_ctx))


@pytest.mark.integration
async def test_cancel_both_args_raises_tool_error():
    """Test that background_cancel raises ToolError when both task_id and cancel_all are provided."""
    capability = BackgroundTaskCapability(schemas=None)
    agent = Agent(name="parent_agent", model="test")
    agent_ctx = AgentContext(node=agent, pool=None)

    with pytest.raises(ToolError, match="Cannot specify both"):
        await capability._background_cancel(
            _wrap_in_run_context(agent_ctx), task_id="some-task", cancel_all=True
        )


# ---------------------------------------------------------------------------
# Cleanup after shutdown
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_shutdown_cancels_running_tasks():
    """Test that BackgroundTaskManager.shutdown() cancels all running tasks."""
    manager = BackgroundTaskManager()

    # Register a pending task
    task_model = BackgroundTask(
        id="test-shutdown-task",
        description="Task to be cleaned up",
        agent_or_team="test_agent",
        prompt="Test prompt",
        parent_session_id=None,
        child_session_id=None,
        status="pending",
    )
    manager.register_task(task_model)

    # Start the task with a long-running coroutine
    async def _long_running():
        await asyncio.sleep(60)

    manager.start_task("test-shutdown-task", _long_running())

    # Give the task a moment to start
    await asyncio.sleep(0.1)

    # Shutdown should cancel all tasks
    await manager.shutdown()

    # Verify registries are cleared
    assert manager.get_task("test-shutdown-task") is None
    assert manager.get_all_tasks() == []


@pytest.mark.integration
async def test_shutdown_clears_empty_manager():
    """Test that shutdown works on an empty manager without errors."""
    manager = BackgroundTaskManager()
    await manager.shutdown()

    assert manager.get_all_tasks() == []


# ---------------------------------------------------------------------------
# Export verification
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_background_task_provider_importable_from_package():
    """Test that BackgroundTaskCapability is importable from the capabilities package."""
    from wolfharness.capabilities.background_task import (
        BackgroundTaskCapability as Imported,
    )

    # BackgroundTaskProvider is already imported at module level; verify identity
    assert BackgroundTaskProvider is not None
    assert BackgroundTaskProvider is Imported


# ---------------------------------------------------------------------------
# Session pool integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_async_task_with_session_pool_inject_prompt(
    mock_pool_with_agent: AgentPool,
    mock_internal_fs: MagicMock,
):
    """Test that session_pool.followup is called when a background task completes."""
    capability = BackgroundTaskCapability(schemas=None, notification_debounce_ms=10)
    agent = Agent(name="parent_agent", model="test")
    agent.session_id = "test-parent-session-123"
    agent._internal_fs = mock_internal_fs
    agent_ctx = AgentContext(node=agent, pool=mock_pool_with_agent, input_provider=MagicMock())

    # Set up a mock AgentRunContext so the deliver callback uses session_pool.followup
    mock_run_ctx = MagicMock(spec=AgentRunContext)
    mock_run_ctx.run_id = "test-run-id-123"
    mock_run_ctx.session_id = "test-parent-session-123"
    mock_run_ctx._run_handle = None
    mock_run_ctx.child_done_events = {}
    mock_run_ctx.depth = 0
    agent_ctx.run_ctx = mock_run_ctx  # type: ignore[attr-defined]

    # Set up session_pool with followup mock
    mock_session_pool = MagicMock()
    mock_session_pool.inject_prompt = AsyncMock()
    mock_session_pool.steer = AsyncMock()
    mock_session_pool.followup = AsyncMock(return_value=True)
    mock_session_pool.sessions = MagicMock()
    mock_session_pool.sessions.get_or_create_session_agent = AsyncMock(
        return_value=mock_pool_with_agent.nodes["test_agent"],
    )
    mock_session_pool.event_bus = MagicMock()

    async def _subscribe_with_event(session_id, scope="session"):
        queue = asyncio.Queue()
        await queue.put(
            StreamCompleteEvent(message=ChatMessage(content="Quick result", role="assistant"))
        )
        return queue

    mock_session_pool.event_bus.subscribe = AsyncMock(side_effect=_subscribe_with_event)
    mock_session_pool.event_bus.unsubscribe = AsyncMock()
    mock_session_pool.send_message = AsyncMock(return_value=MagicMock())
    mock_pool_with_agent.session_pool = mock_session_pool  # type: ignore[attr-defined]

    # Override run_stream to produce a quick completion
    mock_node = mock_pool_with_agent.nodes["test_agent"]

    async def _quick_stream(prompt=None, deps=None, **kwargs):
        yield StreamCompleteEvent(message=ChatMessage(content="Quick result", role="assistant"))

    mock_node.run_stream = _quick_stream
    mock_node.model_name = "test-model"

    result = await capability._task(
        _wrap_in_run_context(agent_ctx),
        agent="test_agent",
        message="Quick task",
        expected_output="Result",
        async_mode=True,
    )

    task_id_match = re.search(r"Task ID: (\S+)", result)
    assert task_id_match is not None
    task_id = task_id_match.group(1)

    # Wait for the background task to complete and notification to be sent
    state = capability._get_session_state(agent_ctx)
    for _ in range(50):
        task = state.task_manager.get_task(task_id)
        if task is not None and task.status in ("completed", "error", "cancelled", "timed_out"):
            break
        await asyncio.sleep(0.1)
    else:
        pytest.fail("Task did not complete in time")

    # Allow notification task to execute (debounce + delivery)
    await asyncio.sleep(0.5)

    # Verify followup was called with parent session ID (followup replaces steer)
    mock_session_pool.followup.assert_awaited_once()
    call_args = mock_session_pool.followup.await_args
    assert call_args is not None
    parent_session_id_arg = call_args.args[0]
    notice = call_args.args[1]
    assert parent_session_id_arg, "parent_session_id should not be empty"
    assert "BACKGROUND TASK" in notice


@pytest.mark.integration
async def test_task_with_session_pool_run_stream(mock_pool_with_agent: AgentPool):
    """Test that sync task uses session_pool.run_stream when available."""
    capability = BackgroundTaskCapability(schemas=None)

    async def _session_stream(*args, **kwargs):
        yield StreamCompleteEvent(
            message=ChatMessage(content="Session pool result", role="assistant")
        )

    mock_session_pool = MagicMock()
    mock_session_pool.run_stream = MagicMock(return_value=_session_stream())
    mock_session_pool.sessions = MagicMock()
    mock_session_pool.sessions.get_or_create_session_agent = AsyncMock(
        return_value=mock_pool_with_agent.nodes["test_agent"],
    )
    mock_pool_with_agent.session_pool = mock_session_pool  # type: ignore[attr-defined]

    agent = Agent(name="parent_agent", model="test")
    agent_ctx = AgentContext(node=agent, pool=mock_pool_with_agent, input_provider=MagicMock())

    result = await capability._task(
        _wrap_in_run_context(agent_ctx),
        agent="test_agent",
        message="Test message",
        expected_output="Test output",
        async_mode=False,
    )

    assert "Session pool result" in result
    mock_session_pool.run_stream.assert_called_once()
    call_args = mock_session_pool.run_stream.call_args
    assert call_args is not None
    child_session_id = call_args.args[0]
    prompt = call_args.args[1]
    assert child_session_id
    assert "Test message" in prompt


@pytest.mark.integration
async def test_background_task_survives_parent_session_end(
    mock_pool_with_agent: AgentPool,
    mock_internal_fs: MagicMock,
):
    """Test that async task continues running even if parent session context changes."""
    capability = BackgroundTaskCapability(schemas=None)
    agent = Agent(name="parent_agent", model="test")
    agent._internal_fs = mock_internal_fs
    agent_ctx = AgentContext(node=agent, pool=mock_pool_with_agent, input_provider=MagicMock())

    mock_node = mock_pool_with_agent.nodes["test_agent"]

    async def _slow_stream(prompt=None, deps=None, **kwargs):
        await asyncio.sleep(0.3)
        yield StreamCompleteEvent(message=ChatMessage(content="Eventually done", role="assistant"))

    mock_node.run_stream = _slow_stream
    mock_node.model_name = "test-model"

    result = await capability._task(
        _wrap_in_run_context(agent_ctx),
        agent="test_agent",
        message="Long task",
        expected_output="Result",
        async_mode=True,
    )

    task_id_match = re.search(r"Task ID: (\S+)", result)
    assert task_id_match is not None
    task_id = task_id_match.group(1)

    # Verify task is running or pending (mock may complete quickly)
    state = capability._get_session_state(agent_ctx)
    task = state.task_manager.get_task(task_id)
    assert task is not None
    assert task.status in ("pending", "running", "completed")

    # Simulate parent session end by changing node state
    agent.session_id = "new_session_id"

    # Give the task a moment to continue
    await asyncio.sleep(0.1)

    # Verify task is still tracked (may have completed due to mock event queue)
    task = state.task_manager.get_task(task_id)
    assert task is not None

    # Cancel the task to clean up (if not already completed)
    if task.status not in ("completed", "cancelled", "error", "timed_out"):
        await capability._background_cancel(_wrap_in_run_context(agent_ctx), task_id=task_id)
        # Allow cancellation to propagate and task to be cleaned up
        await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# force_retrieval graph integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_force_retrieval_redirects_end_through_agent_graph():
    """Integration test: force_retrieval intercepts End via after_node_run.

    Tests the full agent graph integration that unit tests cannot cover:
    1. Agent produces text -> End -> after_node_run intercepts (pending retrievals)
    2. Agent is redirected to ModelRequestNode (not End)
    3. get_model_settings forces tool_choice=['background_output']
    4. Agent calls background_output -> _pending_retrievals cleared
    5. Agent produces text -> End -> _pending_retrievals empty -> End passes

    Uses FunctionModel (not TestModel) because FunctionModel can inspect
    ``info.model_settings['tool_choice']`` — the value injected by
    ``get_model_settings()``. TestModel ignores tool_choice entirely.

    This follows the pattern from pydantic-ai's own test suite:
    ``test_capability_can_inject_forcing_tool_choice_per_step`` and
    ``test_after_node_run_end_to_node_override`` in tests/test_capabilities.py.
    """
    capability = BackgroundTaskCapability(schemas=None, force_retrieval="tool_choice")

    # Patch before_run to set pending_retrievals on the session state
    # (normally before_run clears stale state, but we set it to simulate
    # a background task launched during the run).
    async def _setup_before_run(ctx: object) -> None:
        state = capability._get_session_state(ctx)  # type: ignore[arg-type]
        state.pending_retrievals = {"bg_test"}
        await state.batcher.start()

    capability.before_run = _setup_before_run  # type: ignore[method-assign]

    seen_tool_choices: list = []
    model = FunctionModel(_make_force_retrieval_model_fn(seen_tool_choices))

    agent = PydanticAgent(
        model=model,
        capabilities=[capability],
        deps_type=AgentContext,
    )

    # Mock deps: pool=None so get_instructions returns the default string
    # without trying to iterate pool.manifest.agents
    mock_deps = MagicMock()
    mock_deps.pool = None

    result = await agent.run("Test prompt", deps=mock_deps)

    # Verify the run completed (not stuck in infinite redirect loop)
    assert result is not None
    assert result.output is not None

    # Verify background_output was called: pending_retrievals should be empty
    # (the _background_output tool discards task_id at the start)
    all_states = list(capability._session_states.values()) + list(
        capability._ephemeral_states.values()
    )
    assert all(len(s.pending_retrievals) == 0 for s in all_states), (
        "background_output was not called — after_node_run redirect may have failed"
    )

    # Verify tool_choice enforcement:
    # Step 1: after_node_run redirected End→ModelRequestNode, get_model_settings
    #          forced tool_choice=['background_output'] → model called the tool
    # Step 2: _pending_retrievals empty → tool_choice absent → model produced text
    assert seen_tool_choices == [["background_output"], None], (
        f"Expected tool_choice sequence [['background_output'], None], got {seen_tool_choices}"
    )
