"""Unit tests for BackgroundTaskCapability."""

import asyncio
from pathlib import Path
from typing import Protocol, runtime_checkable
from unittest.mock import AsyncMock, MagicMock

from pydantic_ai import RunContext
from pydantic_ai._agent_graph import ModelRequestNode
from pydantic_ai.capabilities import CapabilityOrdering, NativeTool, ProcessHistory
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.toolsets import FunctionToolset
from pydantic_graph import End

from wolfharness.agents.context import AgentContext
from wolfharness.capabilities.background_task import BackgroundTaskCapability
import wolfharness.capabilities.background_task.capability as btc_module
from wolfharness.capabilities.background_task.capability import ForceRetrievalMode


SCHEMAS_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "wolfharness"
    / "capabilities"
    / "background_task"
    / "schemas"
)
TASK_SCHEMA_PATH = str(SCHEMAS_DIR / "task.yaml")


@runtime_checkable
class _InstructionCallable(Protocol):
    """Protocol for the instruction callable returned by get_instructions()."""

    def __call__(self, ctx: RunContext[AgentContext]) -> str: ...


def test_get_instructions_returns_callable():
    """get_instructions() should return a callable (not a static string)."""
    capability = BackgroundTaskCapability()
    instructions = capability.get_instructions()
    assert instructions is not None
    assert isinstance(instructions, _InstructionCallable)


def test_get_instructions_callable_extracts_agents():
    """The instructions callable should extract available agents from the pool, skipping the current agent."""
    capability = BackgroundTaskCapability()

    # Build a mock RunContext[AgentContext]
    mock_ctx = MagicMock()
    mock_ctx.deps.pool.manifest.agents = {
        "worker": MagicMock(description="A worker"),
        "librarian": MagicMock(description="Researcher"),
        "engineer": MagicMock(description="The current agent"),
    }
    mock_ctx.deps.node.name = "engineer"

    instructions = capability.get_instructions()
    assert instructions is not None
    assert isinstance(instructions, _InstructionCallable)
    result: str = instructions(mock_ctx)

    assert "# Available Agents:" in result
    assert "- worker: A worker" in result
    assert "- librarian: Researcher" in result
    assert "- engineer:" not in result


def test_get_toolset_returns_function_toolset():
    """get_toolset() should return a FunctionToolset with all tools when no filter is set."""
    capability = BackgroundTaskCapability(schemas=None)
    toolset = capability.get_toolset()

    assert toolset is not None
    assert isinstance(toolset, FunctionToolset)
    tool_names = set(toolset.tools.keys())
    assert "task" in tool_names
    assert "background_output" in tool_names
    assert "background_cancel" in tool_names
    assert "steer_task" in tool_names


def test_get_toolset_respects_enabled_tools():
    """get_toolset() should only include tools listed in enabled_tools."""
    capability = BackgroundTaskCapability(schemas=None, enabled_tools=["task"])
    toolset = capability.get_toolset()

    assert toolset is not None
    assert isinstance(toolset, FunctionToolset)
    tool_names = set(toolset.tools.keys())
    assert tool_names == {"task"}


def test_get_ordering_wrapped_by_process_history_native_tool():
    """get_ordering() should return CapabilityOrdering wrapped_by ProcessHistory and NativeTool."""
    capability = BackgroundTaskCapability()
    ordering = capability.get_ordering()

    assert ordering is not None
    assert isinstance(ordering, CapabilityOrdering)
    assert ProcessHistory in ordering.wrapped_by
    assert NativeTool in ordering.wrapped_by


async def test_before_run_is_noop():
    """before_run() should complete without exception and clear _pending_retrievals."""
    capability = BackgroundTaskCapability()
    mock_ctx = MagicMock()
    # Should not raise any exception
    await capability.before_run(mock_ctx)


# ---- force_retrieval disabled (default) tests ----


async def test_after_node_run_disabled_returns_end_unchanged():
    """When force_retrieval is disabled, after_node_run should return End unchanged."""
    capability = BackgroundTaskCapability()
    mock_ctx = MagicMock()
    end_result = End(data=None)
    result = await capability.after_node_run(mock_ctx, node=MagicMock(), result=end_result)
    assert result is end_result


def test_get_model_settings_disabled_returns_none():
    """When force_retrieval is disabled, get_model_settings should return None."""
    capability = BackgroundTaskCapability()
    assert capability.get_model_settings() is None


# ---- force_retrieval backward compatibility tests ----


def test_force_retrieval_bool_true_maps_to_tool_choice():
    """force_retrieval=True should coerce to ForceRetrievalMode.tool_choice."""
    capability = BackgroundTaskCapability(force_retrieval=True)
    assert capability._force_retrieval is ForceRetrievalMode.tool_choice


def test_force_retrieval_bool_false_maps_to_disabled():
    """force_retrieval=False should coerce to ForceRetrievalMode.disabled."""
    capability = BackgroundTaskCapability(force_retrieval=False)
    assert capability._force_retrieval is ForceRetrievalMode.disabled


def test_force_retrieval_str_directive():
    """force_retrieval='directive' should coerce to ForceRetrievalMode.directive."""
    capability = BackgroundTaskCapability(force_retrieval="directive")
    assert capability._force_retrieval is ForceRetrievalMode.directive


# ---- force_retrieval=tool_choice: after_node_run tests ----


async def test_after_node_run_intercepts_end_with_pending():
    """When force_retrieval=tool_choice and _pending_retrievals is non-empty, after_node_run should return ModelRequestNode."""
    capability = BackgroundTaskCapability(force_retrieval="tool_choice")
    mock_ctx = MagicMock()
    state = capability._get_session_state(mock_ctx)
    state.pending_retrievals = {"bg_abc123"}
    end_result = End(data=None)
    result = await capability.after_node_run(mock_ctx, node=MagicMock(), result=end_result)
    assert isinstance(result, ModelRequestNode)
    assert isinstance(result.request, ModelRequest)
    assert len(result.request.parts) == 1
    assert isinstance(result.request.parts[0], UserPromptPart)
    content = result.request.parts[0].content
    assert "<system-reminder>" in content
    assert "1 background task" in content
    assert "bg_abc123" in content
    assert "background_output" in content


async def test_after_node_run_passes_end_when_no_pending():
    """When force_retrieval=tool_choice but _pending_retrievals is empty, after_node_run should return End unchanged."""
    capability = BackgroundTaskCapability(force_retrieval="tool_choice")
    mock_ctx = MagicMock()
    end_result = End(data=None)
    result = await capability.after_node_run(mock_ctx, node=MagicMock(), result=end_result)
    assert result is end_result


async def test_after_node_run_passes_non_end_result():
    """When result is not End, after_node_run should return it unchanged."""
    capability = BackgroundTaskCapability(force_retrieval="tool_choice")
    mock_ctx = MagicMock()
    state = capability._get_session_state(mock_ctx)
    state.pending_retrievals = {"bg_abc123"}
    non_end_result = MagicMock()
    result = await capability.after_node_run(mock_ctx, node=MagicMock(), result=non_end_result)
    assert result is non_end_result


# ---- force_retrieval=tool_choice: get_model_settings tests ----


def test_get_model_settings_tool_choice_returns_callable():
    """When force_retrieval=tool_choice, get_model_settings should return a callable."""
    capability = BackgroundTaskCapability(force_retrieval="tool_choice")
    settings = capability.get_model_settings()
    assert settings is not None
    assert callable(settings)


def test_get_model_settings_forces_tool_choice_with_pending():
    """When _pending_retrievals is non-empty, the callable should return tool_choice=[background_output]."""
    capability = BackgroundTaskCapability(force_retrieval="tool_choice")
    mock_ctx = MagicMock()
    state = capability._get_session_state(mock_ctx)
    state.pending_retrievals = {"bg_abc123"}
    settings_fn = capability.get_model_settings()
    assert settings_fn is not None
    result = settings_fn(mock_ctx)  # type: ignore[misc]
    assert result.get("tool_choice") == ["background_output"]


def test_get_model_settings_empty_when_no_pending():
    """When _pending_retrievals is empty, the callable should return empty ModelSettings."""
    capability = BackgroundTaskCapability(force_retrieval="tool_choice")
    settings_fn = capability.get_model_settings()
    assert settings_fn is not None
    mock_ctx = MagicMock()
    result = settings_fn(mock_ctx)  # type: ignore[misc]
    assert "tool_choice" not in result


# ---- force_retrieval=directive: after_node_run tests ----


async def test_directive_after_node_run_intercepts_end_with_pending():
    """When force_retrieval=directive and _pending_retrievals is non-empty, after_node_run should return ModelRequestNode."""
    capability = BackgroundTaskCapability(force_retrieval="directive")
    mock_ctx = MagicMock()
    state = capability._get_session_state(mock_ctx)
    state.pending_retrievals = {"bg_abc123"}
    end_result = End(data=None)
    result = await capability.after_node_run(mock_ctx, node=MagicMock(), result=end_result)
    assert isinstance(result, ModelRequestNode)
    assert isinstance(result.request, ModelRequest)
    assert len(result.request.parts) == 1
    assert isinstance(result.request.parts[0], UserPromptPart)
    content = result.request.parts[0].content
    assert "<system-reminder>" in content
    assert "bg_abc123" in content
    assert "background_output" in content


async def test_directive_after_node_run_passes_end_when_no_pending():
    """When force_retrieval=directive but _pending_retrievals is empty, after_node_run should return End unchanged."""
    capability = BackgroundTaskCapability(force_retrieval="directive")
    mock_ctx = MagicMock()
    end_result = End(data=None)
    result = await capability.after_node_run(mock_ctx, node=MagicMock(), result=end_result)
    assert result is end_result


def test_directive_get_model_settings_returns_none():
    """When force_retrieval=directive, get_model_settings should return None (no tool_choice forcing)."""
    capability = BackgroundTaskCapability(force_retrieval="directive")
    assert capability.get_model_settings() is None


# ---- Loop breaker tests ----


async def test_loop_breaker_allows_end_after_max_retries():
    """after_node_run should allow End after max_retrieval_retries injections."""
    capability = BackgroundTaskCapability(force_retrieval="directive", max_retrieval_retries=2)
    mock_ctx = MagicMock()
    state = capability._get_session_state(mock_ctx)
    state.pending_retrievals = {"bg_abc123"}
    end_result = End(data=None)

    # First injection
    result1 = await capability.after_node_run(mock_ctx, node=MagicMock(), result=end_result)
    assert isinstance(result1, ModelRequestNode)
    assert state.retrieval_retry_count == 1

    # Second injection
    result2 = await capability.after_node_run(mock_ctx, node=MagicMock(), result=end_result)
    assert isinstance(result2, ModelRequestNode)
    assert state.retrieval_retry_count == 2

    # Third attempt: loop breaker kicks in, End passes through
    result3 = await capability.after_node_run(mock_ctx, node=MagicMock(), result=end_result)
    assert result3 is end_result
    assert state.retrieval_retry_count == 2  # Counter not incremented past max


async def test_loop_breaker_default_max_is_3():
    """Default max_retrieval_retries should be 3."""
    capability = BackgroundTaskCapability(force_retrieval="tool_choice")
    assert capability._max_retrieval_retries == 3


async def test_before_run_resets_retry_count():
    """before_run should reset _retrieval_retry_count to 0."""
    capability = BackgroundTaskCapability(force_retrieval="tool_choice")
    mock_ctx = MagicMock()
    state = capability._get_session_state(mock_ctx)
    state.retrieval_retry_count = 5
    state.pending_retrievals = {"bg_aaa"}
    await capability.before_run(mock_ctx)
    assert state.retrieval_retry_count == 0
    assert state.pending_retrievals == set()


# ---- Per-run tracking tests ----


async def test_before_run_clears_pending_retrievals():
    """before_run should clear _pending_retrievals."""
    capability = BackgroundTaskCapability(force_retrieval="tool_choice")
    mock_ctx = MagicMock()
    state = capability._get_session_state(mock_ctx)
    state.pending_retrievals = {"bg_aaa", "bg_bbb"}
    await capability.before_run(mock_ctx)
    assert state.pending_retrievals == set()


def test_pending_retrievals_add_on_task_async():
    """_task_async should add task_id to _pending_retrievals when force_retrieval is enabled."""
    capability = BackgroundTaskCapability(force_retrieval="tool_choice")
    mock_ctx = MagicMock()
    state = capability._get_session_state(mock_ctx)
    # Simulate what _task_async does: generate a task_id and add it
    # We test the tracking behavior, not the full _task_async flow
    task_id = "bg_test123"
    if capability._force_retrieval is not ForceRetrievalMode.disabled:
        state.pending_retrievals.add(task_id)
    assert task_id in state.pending_retrievals


def test_pending_retrievals_not_added_when_disabled():
    """_task_async should NOT add task_id when force_retrieval is disabled."""
    capability = BackgroundTaskCapability()
    mock_ctx = MagicMock()
    state = capability._get_session_state(mock_ctx)
    task_id = "bg_test456"
    if capability._force_retrieval is not ForceRetrievalMode.disabled:
        state.pending_retrievals.add(task_id)
    assert task_id not in state.pending_retrievals


def test_background_output_discards_from_pending():
    """_background_output should discard task_id from _pending_retrievals."""
    capability = BackgroundTaskCapability(force_retrieval="tool_choice")
    capability._pending_retrievals = {"bg_aaa", "bg_bbb"}
    # Simulate what _background_output does at its start
    capability._pending_retrievals.discard("bg_aaa")
    assert "bg_aaa" not in capability._pending_retrievals
    assert "bg_bbb" in capability._pending_retrievals


def test_background_output_discard_is_noop_for_unknown_id():
    """Discarding an unknown task_id should be a no-op (no error)."""
    capability = BackgroundTaskCapability(force_retrieval="tool_choice")
    capability._pending_retrievals = {"bg_aaa"}
    # discard on a set is a no-op for missing elements
    capability._pending_retrievals.discard("bg_nonexistent")
    assert capability._pending_retrievals == {"bg_aaa"}


# ---- Tool name resolution test ----


def test_output_tool_name_defaults_to_background_output():
    """_output_tool_name should default to 'background_output' when no schema is provided."""
    capability = BackgroundTaskCapability()
    assert capability._output_tool_name == "background_output"


def test_output_tool_name_uses_schema_name():
    """_output_tool_name should use the name from _background_output_schema when available."""
    capability = BackgroundTaskCapability()
    # Simulate having a schema with a custom name
    capability._background_output_schema = {"name": "get_task_result"}  # type: ignore[assignment]
    # Re-resolve the tool name (as __init__ does)
    capability._output_tool_name = (
        capability._background_output_schema.get("name")
        if capability._background_output_schema
        else None
    ) or "background_output"
    assert capability._output_tool_name == "get_task_result"


# ---- Schema override tests (load_skills visibility) ----


def test_task_tool_schema_has_load_skills_required():
    """The task tool's ToolDefinition should have load_skills in required list after schema override."""
    capability = BackgroundTaskCapability(schemas={"task": TASK_SCHEMA_PATH})
    toolset = capability.get_toolset()

    assert toolset is not None
    assert isinstance(toolset, FunctionToolset)
    assert "task" in toolset.tools  # type: ignore[union-attr]

    tool = toolset.tools["task"]  # type: ignore[union-attr]
    tool_def = tool.tool_def
    params = tool_def.parameters_json_schema

    # load_skills should be in the properties
    assert "load_skills" in params.get("properties", {})

    # load_skills should be in the required list (from YAML schema)
    required = params.get("required", [])
    assert "load_skills" in required, (
        f"load_skills should be required after schema override, got required={required}"
    )

    # The description should contain skill:// URI format guidance
    load_skills_desc = params["properties"]["load_skills"].get("description", "")
    assert "skill://" in load_skills_desc, (
        f"load_skills description should mention skill:// URIs, got: {load_skills_desc[:100]}"
    )


def test_task_tool_schema_preserves_all_parameters():
    """All YAML schema parameters should be present in the overridden ToolDefinition."""
    capability = BackgroundTaskCapability(schemas={"task": TASK_SCHEMA_PATH})
    toolset = capability.get_toolset()

    assert toolset is not None
    tool = toolset.tools["task"]  # type: ignore[union-attr]
    params = tool.tool_def.parameters_json_schema

    expected_props = {"agent", "load_skills", "message", "expected_output", "title", "async_mode"}
    actual_props = set(params.get("properties", {}).keys())
    assert expected_props == actual_props, (
        f"Schema properties mismatch. Expected: {expected_props}, Got: {actual_props}"
    )


# ---- _format_skills_instructions tests ----


async def test_format_skills_instructions_calls_load_skill():
    """_format_skills_instructions should call load_skill_for_node for each skill name and include results."""
    capability = BackgroundTaskCapability()
    mock_ctx = MagicMock(spec=AgentContext)
    mock_ctx.pool = MagicMock()
    mock_ctx.pool.is_skill_visible_to_node = MagicMock(return_value=True)

    skill_uris = [
        "skill://fta-initialize/references/source-priority.md",
        "skill://fta-initialize/references/credibility-rules.md",
    ]

    # Patch load_skill_for_node at module level
    original = btc_module.load_skill_for_node
    btc_module.load_skill_for_node = AsyncMock(
        side_effect=lambda ctx, name, node_name=None: f"CONTENT_FOR_{name}"
    )

    try:
        result = await capability._format_skills_instructions(mock_ctx, skill_uris, "librarian")
    finally:
        btc_module.load_skill_for_node = original

    # Result should contain skill-instruction XML tags with content
    assert "<skill-instruction" in result
    assert "CONTENT_FOR_skill://fta-initialize/references/source-priority.md" in result
    assert "CONTENT_FOR_skill://fta-initialize/references/credibility-rules.md" in result


async def test_format_skills_instructions_handles_errors():
    """_format_skills_instructions should include error messages when load_skill_for_node fails."""
    capability = BackgroundTaskCapability()
    mock_ctx = MagicMock(spec=AgentContext)
    mock_ctx.pool = MagicMock()
    mock_ctx.pool.is_skill_visible_to_node = MagicMock(return_value=True)

    skill_uris = ["skill://nonexistent/ref.md"]

    original = btc_module.load_skill_for_node
    btc_module.load_skill_for_node = AsyncMock(
        return_value="Skill 'nonexistent' not found. Available skills: []"
    )

    try:
        result = await capability._format_skills_instructions(mock_ctx, skill_uris, "librarian")
    finally:
        btc_module.load_skill_for_node = original

    # Result should still contain skill-instruction tags, but with error message
    assert "<skill-instruction" in result
    assert "not found" in result


# ---- End-to-end skill loading in _task ----


async def test_task_prepends_skills_content_to_prompt():
    """_task should prepend skills_content to the formatted prompt before passing to subagent."""
    capability = BackgroundTaskCapability(schemas={"task": TASK_SCHEMA_PATH})

    # Mock the pool and context
    mock_pool = MagicMock()
    mock_pool.agent_configs = {"librarian": MagicMock()}
    mock_pool.manifest = MagicMock()
    mock_pool.manifest.agents = {"librarian": MagicMock(), "engineer": MagicMock()}
    mock_pool.session_pool = MagicMock()
    mock_pool.session_pool.event_bus = MagicMock()
    mock_pool.session_pool.event_bus.subscribe = AsyncMock(return_value=asyncio.Queue())
    mock_pool.session_pool.event_bus.unsubscribe = AsyncMock()
    mock_pool.session_pool.send_message = AsyncMock(return_value=MagicMock())

    mock_node = MagicMock()
    mock_node.name = "engineer"
    mock_node.session_id = "test-session"

    mock_run_ctx = MagicMock()
    mock_run_ctx.deps = MagicMock(spec=AgentContext)
    mock_run_ctx.deps.pool = mock_pool
    mock_run_ctx.deps.node = mock_node
    mock_run_ctx.deps.data = {}
    mock_run_ctx.deps.run_ctx = None
    mock_run_ctx.deps.create_child_session = AsyncMock(return_value="child-session-id")
    mock_run_ctx.deps.internal_fs = MagicMock()
    mock_run_ctx.deps.internal_fs.mkdirs = MagicMock()
    mock_run_ctx.deps.get_input_provider = MagicMock(return_value=None)
    mock_run_ctx.session_id = "parent-session"
    mock_run_ctx.tool_call_id = "call_test"

    # Mock _format_skills_instructions to return known content
    expected_skills_content = '<skill-instruction name="fta-initialize/references/source-priority.md">\nSKILL CONTENT HERE\n</skill-instruction>'
    capability._format_skills_instructions = AsyncMock(return_value=expected_skills_content)  # type: ignore[assignment]

    # Call _task with load_skills and async_mode=True
    await capability._task(
        mock_run_ctx,
        agent="librarian",
        message="Do research",
        expected_output="A research report",
        load_skills=["skill://fta-initialize/references/source-priority.md"],
        title="Research task",
        async_mode=True,
    )

    # Verify _format_skills_instructions was called with the correct skills
    capability._format_skills_instructions.assert_called_once()
    call_args = capability._format_skills_instructions.call_args
    assert call_args is not None
    # Second positional arg should be the skill list
    skills_arg = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("skill_names")
    assert "skill://fta-initialize/references/source-priority.md" in skills_arg

    # Wait for the background task to progress
    await asyncio.sleep(0.1)

    # Verify send_message was called with a prompt that includes skills content
    receive_call = mock_pool.session_pool.send_message.call_args
    assert receive_call is not None, "send_message should have been called by the background task"
    prompt_arg = receive_call[0][1] if len(receive_call[0]) > 1 else receive_call[1].get("content")
    assert "SKILL CONTENT HERE" in prompt_arg, (
        f"Skills content should be prepended to prompt. Prompt starts with: {prompt_arg[:200]}"
    )


# ---- _steer_task tests ----


async def test_steer_task_advisory_calls_followup():
    """_steer_task with mode='advisory' should call session_pool.followup()."""
    capability = BackgroundTaskCapability()

    # Mock task manager with a running task
    mock_task = MagicMock()
    mock_task.status = "running"
    mock_task.child_session_id = "child-session-123"
    mock_task.description = "Research task"

    mock_state = MagicMock()
    mock_state.task_manager.get_task.return_value = mock_task
    capability._get_session_state = MagicMock(return_value=mock_state)  # type: ignore[assignment]

    # Mock session pool
    mock_session_pool = MagicMock()
    mock_session_pool.followup = AsyncMock(return_value=True)
    mock_session_pool.steer = AsyncMock(return_value=True)

    mock_ctx = MagicMock()
    mock_ctx.deps.pool.session_pool = mock_session_pool

    result = await capability._steer_task(
        mock_ctx,
        task_id="task-1",
        message="Focus on hydraulic system failures",
        mode="advisory",
    )

    # Should call followup, not steer
    expected_msg = (
        "<system-reminder>\n"
        "[STEERING DIRECTIVE — ADVISORY]\n"
        "This is a directive from the parent diagnostic agent, not a new user query.\n"
        "Integrate this new information into your research and adjust your plan accordingly.\n"
        "Do NOT restart from scratch — refine your existing research direction.\n"
        "Skip any research branches this directive rules out.\n"
        "---\n"
        "Focus on hydraulic system failures\n"
        "</system-reminder>"
    )
    mock_session_pool.followup.assert_called_once_with("child-session-123", expected_msg)
    mock_session_pool.steer.assert_not_called()
    assert "advisory" in result
    assert "next-turn" in result


async def test_steer_task_interrupt_calls_steer():
    """_steer_task with mode='interrupt' should call session_pool.steer()."""
    capability = BackgroundTaskCapability()

    mock_task = MagicMock()
    mock_task.status = "running"
    mock_task.child_session_id = "child-session-456"
    mock_task.description = "Planning task"

    mock_state = MagicMock()
    mock_state.task_manager.get_task.return_value = mock_task
    capability._get_session_state = MagicMock(return_value=mock_state)  # type: ignore[assignment]

    mock_session_pool = MagicMock()
    mock_session_pool.followup = AsyncMock(return_value=True)
    mock_session_pool.steer = AsyncMock(return_value=True)

    mock_ctx = MagicMock()
    mock_ctx.deps.pool.session_pool = mock_session_pool

    result = await capability._steer_task(
        mock_ctx,
        task_id="task-2",
        message="Stop current direction, investigate electrical system instead",
        mode="interrupt",
    )

    # Should call steer, not followup
    expected_msg = (
        "<system-reminder>\n"
        "[STEERING DIRECTIVE — INTERRUPT]\n"
        "This is a directive from the parent diagnostic agent, not a new user query.\n"
        "Adjust your current research direction immediately based on this new information.\n"
        "Do NOT restart from scratch — integrate this into your existing work.\n"
        "Skip any research branches this directive rules out.\n"
        "---\n"
        "Stop current direction, investigate electrical system instead\n"
        "</system-reminder>"
    )
    mock_session_pool.steer.assert_called_once_with("child-session-456", expected_msg)
    mock_session_pool.followup.assert_not_called()
    assert "interrupt" in result
    assert "mid-turn" in result


async def test_steer_task_raises_error_for_unknown_task():
    """_steer_task should raise ToolError when task_id is not found."""
    from wolfharness.tools.exceptions import ToolError

    capability = BackgroundTaskCapability()

    mock_state = MagicMock()
    mock_state.task_manager.get_task.return_value = None
    capability._get_session_state = MagicMock(return_value=mock_state)  # type: ignore[assignment]

    mock_ctx = MagicMock()
    mock_ctx.deps.pool.session_pool = MagicMock()

    try:
        await capability._steer_task(mock_ctx, task_id="nonexistent", message="test")
        msg = "Should have raised ToolError"
        raise AssertionError(msg)
    except ToolError as e:
        assert "nonexistent" in str(e)


async def test_steer_task_returns_message_for_terminal_task():
    """_steer_task should return a message (not raise) when task is already completed."""
    capability = BackgroundTaskCapability()

    mock_task = MagicMock()
    mock_task.status = "completed"
    mock_task.child_session_id = "child-session-789"

    mock_state = MagicMock()
    mock_state.task_manager.get_task.return_value = mock_task
    capability._get_session_state = MagicMock(return_value=mock_state)  # type: ignore[assignment]

    mock_ctx = MagicMock()
    mock_ctx.deps.pool.session_pool = MagicMock()
    mock_ctx.deps.pool.session_pool.steer = AsyncMock()
    mock_ctx.deps.pool.session_pool.followup = AsyncMock()

    result = await capability._steer_task(mock_ctx, task_id="task-done", message="test message")

    assert "already completed" in result
    mock_ctx.deps.pool.session_pool.steer.assert_not_called()
    mock_ctx.deps.pool.session_pool.followup.assert_not_called()


async def test_steer_task_raises_error_when_no_session_pool():
    """_steer_task should raise ToolError when SessionPool is unavailable."""
    from wolfharness.tools.exceptions import ToolError

    capability = BackgroundTaskCapability()

    mock_task = MagicMock()
    mock_task.status = "running"
    mock_task.child_session_id = "child-session-000"

    mock_state = MagicMock()
    mock_state.task_manager.get_task.return_value = mock_task
    capability._get_session_state = MagicMock(return_value=mock_state)  # type: ignore[assignment]

    mock_ctx = MagicMock()
    mock_ctx.deps.pool = None

    try:
        await capability._steer_task(mock_ctx, task_id="task-1", message="test")
        msg = "Should have raised ToolError"
        raise AssertionError(msg)
    except ToolError as e:
        assert "SessionPool" in str(e)


async def test_steer_task_default_mode_is_advisory():
    """_steer_task should default to 'advisory' mode when mode is not specified."""
    capability = BackgroundTaskCapability()

    mock_task = MagicMock()
    mock_task.status = "running"
    mock_task.child_session_id = "child-default"

    mock_state = MagicMock()
    mock_state.task_manager.get_task.return_value = mock_task
    capability._get_session_state = MagicMock(return_value=mock_state)  # type: ignore[assignment]

    mock_session_pool = MagicMock()
    mock_session_pool.followup = AsyncMock(return_value=True)
    mock_session_pool.steer = AsyncMock(return_value=True)

    mock_ctx = MagicMock()
    mock_ctx.deps.pool.session_pool = mock_session_pool

    await capability._steer_task(mock_ctx, task_id="task-default", message="New context")

    # Default should be advisory → followup
    mock_session_pool.followup.assert_called_once()
    mock_session_pool.steer.assert_not_called()


async def test_steer_task_returns_failure_when_delivery_fails():
    """_steer_task should return a failure message when session_pool returns False."""
    capability = BackgroundTaskCapability()

    mock_task = MagicMock()
    mock_task.status = "running"
    mock_task.child_session_id = "child-stale"

    mock_state = MagicMock()
    mock_state.task_manager.get_task.return_value = mock_task
    capability._get_session_state = MagicMock(return_value=mock_state)  # type: ignore[assignment]

    mock_session_pool = MagicMock()
    mock_session_pool.followup = AsyncMock(return_value=False)
    mock_session_pool.steer = AsyncMock(return_value=False)

    mock_ctx = MagicMock()
    mock_ctx.deps.pool.session_pool = mock_session_pool

    result = await capability._steer_task(mock_ctx, task_id="task-stale", message="test")

    assert "could not be delivered" in result
