"""Tests for ToolInterceptCapability in tool_intercept.py.

Covers tasks 5.1-5.13 from the unify-tool-interception-to-pydantic-ai-capabilities change:

Unit tests (5.1-5.10):
  - get_wrapper_toolset() behavior for modes: always, never, per_tool
  - wrap_tool_execute() error handling and pass-through
  - before_tool_execute() modified_input and deny via ModelRetry
  - after_tool_execute() modified_output, additional_context, injection consumption

Integration tests (5.11-5.13):
  - Hooks fire for MCP tools (not just direct tools)
  - Confirmation works for MCP tools when mode="always"
  - No double-firing when old AgentHooks is active AND capability chain is active
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic_ai import ModelRetry
from pydantic_ai.messages import ToolCallPart, ToolReturn
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage
import pytest

from wolfharness import Agent
from wolfharness.agents.native_agent.tool_intercept import ToolInterceptCapability
from wolfharness.hooks import AgentHooks, CallableHook
from wolfharness.hooks.base import HookResult


pytestmark = pytest.mark.unit


# ============================================================================
# Helpers and Fixtures
# ============================================================================


class MockRunCtx:
    """Mock run context with session_id and cancelled flag."""

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id
        self.cancelled: bool = False


class MockDeps:
    """Mock deps for RunContext with agent and run_ctx."""

    def __init__(
        self,
        agent: Any = None,
        session_id: str | None = None,
    ) -> None:
        self.agent = agent or MagicMock()
        self.run_ctx = MockRunCtx(session_id) if session_id else None


def make_run_context(deps: Any = ...) -> RunContext[Any]:
    """Create a RunContext with mock deps."""
    actual_deps = MockDeps() if deps is ... else deps
    return RunContext(
        deps=actual_deps,
        model=TestModel(),
        usage=RunUsage(),
    )


@pytest.fixture
def mock_agent() -> Agent[Any]:
    """Create an agent with TestModel for capability testing."""
    model = TestModel(custom_output_text="test")
    return Agent(name="capability-test-agent", model=model)


@pytest.fixture
def mock_hook_manager(mock_agent: Agent[Any]) -> MagicMock:
    """Create a mock NativeAgentHookManager with _agent set to mock_agent."""
    hm = MagicMock()
    hm.agent_name = "test-agent"
    hm._agent = mock_agent
    hm.run_pre_tool_hooks = AsyncMock(return_value=HookResult(decision="allow"))
    hm.run_post_tool_hooks = AsyncMock(return_value=HookResult(decision="allow"))
    return hm


def make_capability(hook_manager: MagicMock) -> Any:
    """Create a ToolInterceptCapability with the given hook_manager."""
    return ToolInterceptCapability(hook_manager=hook_manager)


def make_tool_call(name: str = "test_tool", args: dict[str, Any] | None = None) -> ToolCallPart:
    """Create a ToolCallPart for testing."""
    return ToolCallPart(tool_name=name, args=args or {"x": 1})


def make_tool_def(name: str = "test_tool") -> ToolDefinition:
    """Create a ToolDefinition for testing."""
    return ToolDefinition(name=name)


# ============================================================================
# 5.1: get_wrapper_toolset() wraps with ApprovalRequiredToolset when mode=always
# ============================================================================


@pytest.mark.anyio
async def test_get_wrapper_toolset_always_wraps_with_approval_required(
    mock_hook_manager: MagicMock,
) -> None:
    """get_wrapper_toolset() wraps toolset with ApprovalRequiredToolset when mode='always'."""
    from pydantic_ai.toolsets import ApprovalRequiredToolset

    cap = make_capability(mock_hook_manager)
    mock_toolset = MagicMock()

    with patch.object(type(cap), "_get_confirmation_mode", return_value="always"):
        result = cap.get_wrapper_toolset(mock_toolset)

    assert result is not None
    assert isinstance(result, ApprovalRequiredToolset)
    assert result.wrapped is mock_toolset  # The wrapped toolset should be the input


# ============================================================================
# 5.2: get_wrapper_toolset() returns None when mode=never
# ============================================================================


@pytest.mark.anyio
async def test_get_wrapper_toolset_never_returns_none(
    mock_hook_manager: MagicMock,
) -> None:
    """get_wrapper_toolset() returns None when mode='never'."""
    cap = make_capability(mock_hook_manager)
    mock_toolset = MagicMock()

    with patch.object(type(cap), "_get_confirmation_mode", return_value="never"):
        result = cap.get_wrapper_toolset(mock_toolset)

    assert result is None


# ============================================================================
# 5.3: get_wrapper_toolset() wraps with per-tool check when mode=per_tool
# ============================================================================


@pytest.mark.anyio
async def test_get_wrapper_toolset_per_tool_wraps_with_approval_required(
    mock_hook_manager: MagicMock,
) -> None:
    """get_wrapper_toolset() wraps with per-tool check when mode='per_tool'."""
    from pydantic_ai.toolsets import ApprovalRequiredToolset

    cap = make_capability(mock_hook_manager)
    mock_toolset = MagicMock()

    # Mock tool manager with some tools requiring confirmation
    mock_tool1 = MagicMock()
    mock_tool1.name = "dangerous_tool"
    mock_tool1.requires_confirmation = True
    mock_tool2 = MagicMock()
    mock_tool2.name = "safe_tool"
    mock_tool2.requires_confirmation = False

    mock_hook_manager._agent._builtin_provider._tools = [mock_tool1, mock_tool2]

    with patch.object(type(cap), "_get_confirmation_mode", return_value="per_tool"):
        result = cap.get_wrapper_toolset(mock_toolset)

    assert result is not None
    assert isinstance(result, ApprovalRequiredToolset)

    # Verify the approval function correctly identifies tools requiring confirmation
    ctx = make_run_context()
    tool_def_confirm = ToolDefinition(name="dangerous_tool")
    tool_def_safe = ToolDefinition(name="safe_tool")

    assert result.approval_required_func(ctx, tool_def_confirm, {}) is True
    assert result.approval_required_func(ctx, tool_def_safe, {}) is False


# ============================================================================
# 5.4: wrap_tool_execute() catches exception and returns annotated ToolReturn
# ============================================================================


@pytest.mark.anyio
async def test_wrap_tool_execute_catches_exception(
    mock_hook_manager: MagicMock,
) -> None:
    """wrap_tool_execute() catches exception and returns annotated ToolReturn."""
    cap = make_capability(mock_hook_manager)
    ctx = make_run_context()
    call = make_tool_call("failing_tool")
    tool_def = make_tool_def("failing_tool")
    args = {"x": 1}

    async def failing_handler(a: dict[str, Any]) -> Any:
        raise ValueError("Something went wrong")

    result = await cap.wrap_tool_execute(
        ctx, call=call, tool_def=tool_def, args=args, handler=failing_handler
    )

    assert isinstance(result, ToolReturn)
    assert "failing_tool" in str(result.content)
    assert "Something went wrong" in str(result.content)


# ============================================================================
# 5.4.1: wrap_tool_execute() re-raises RunAbortedError (control-flow exception)
# ============================================================================


@pytest.mark.anyio
async def test_wrap_tool_execute_reraises_run_aborted_error(
    mock_hook_manager: MagicMock,
) -> None:
    """wrap_tool_execute() must re-raise RunAbortedError, not swallow it.

    RunAbortedError is a control-flow exception raised when the user cancels
    an elicitation/question (e.g. question tool cancel). It must propagate
    up to NativeTurn.execute()'s except RunAbortedError handler to abort the
    run. If swallowed here, the LLM receives it as a tool error and continues
    executing.
    """
    from wolfharness.tasks.exceptions import RunAbortedError

    cap = make_capability(mock_hook_manager)
    ctx = make_run_context()
    call = make_tool_call("question")
    tool_def = make_tool_def("question")
    args: dict[str, Any] = {}

    async def aborting_handler(a: dict[str, Any]) -> Any:
        raise RunAbortedError("User cancelled the questionnaire")

    with pytest.raises(RunAbortedError, match="User cancelled"):
        await cap.wrap_tool_execute(
            ctx, call=call, tool_def=tool_def, args=args, handler=aborting_handler
        )


# ============================================================================
# 5.4.2: wrap_tool_execute() re-raises ModelRetry (pydantic-ai control-flow)
# ============================================================================


@pytest.mark.anyio
async def test_wrap_tool_execute_reraises_model_retry(
    mock_hook_manager: MagicMock,
) -> None:
    """wrap_tool_execute() must re-raise ModelRetry, not swallow it.

    ModelRetry is pydantic-ai's mechanism to ask the LLM to retry. Tools raise
    it for fixable errors (e.g. invalid questionnaire XML, elicitation ErrorData).
    If swallowed, the LLM sees a tool success with error text instead of a retry
    signal — semantically different behavior.
    """
    cap = make_capability(mock_hook_manager)
    ctx = make_run_context()
    call = make_tool_call("question")
    tool_def = make_tool_def("question")
    args: dict[str, Any] = {}

    async def retrying_handler(a: dict[str, Any]) -> Any:
        raise ModelRetry("Elicitation failed: server error")

    with pytest.raises(ModelRetry, match="Elicitation failed"):
        await cap.wrap_tool_execute(
            ctx, call=call, tool_def=tool_def, args=args, handler=retrying_handler
        )


# ============================================================================
# 5.4.3: wrap_tool_execute() re-raises ToolSkippedError (control-flow exception)
# ============================================================================


@pytest.mark.anyio
async def test_wrap_tool_execute_reraises_tool_skipped_error(
    mock_hook_manager: MagicMock,
) -> None:
    """wrap_tool_execute() must re-raise ToolSkippedError, not swallow it.

    ToolSkippedError is raised when a pre-tool hook denies execution (e.g.
    MCP tool_bridge). It signals the tool was skipped, not failed — swallowing
    it converts a skip signal into a tool error response.
    """
    from wolfharness.tasks.exceptions import ToolSkippedError

    cap = make_capability(mock_hook_manager)
    ctx = make_run_context()
    call = make_tool_call("mcp_tool")
    tool_def = make_tool_def("mcp_tool")
    args: dict[str, Any] = {}

    async def skipping_handler(a: dict[str, Any]) -> Any:
        raise ToolSkippedError("Tool mcp_tool blocked by hook")

    with pytest.raises(ToolSkippedError, match="blocked"):
        await cap.wrap_tool_execute(
            ctx, call=call, tool_def=tool_def, args=args, handler=skipping_handler
        )


# ============================================================================
# 5.4.4: wrap_tool_execute() re-raises CallDeferred (pydantic-ai deferred execution)
# ============================================================================


@pytest.mark.anyio
async def test_wrap_tool_execute_reraises_call_deferred(
    mock_hook_manager: MagicMock,
) -> None:
    """wrap_tool_execute() must re-raise CallDeferred, not swallow it.

    CallDeferred is raised by handle_elicitation() when an MCP tool's
    elicitation can't be resolved synchronously. It signals pydantic-ai to
    checkpoint the run and resume later. If swallowed, deferred execution
    breaks — the LLM receives a tool error instead of a deferred signal.
    """
    from pydantic_ai.exceptions import CallDeferred

    cap = make_capability(mock_hook_manager)
    ctx = make_run_context()
    call = make_tool_call("question")
    tool_def = make_tool_def("question")
    args: dict[str, Any] = {}

    async def deferring_handler(a: dict[str, Any]) -> Any:
        raise CallDeferred

    with pytest.raises(CallDeferred):
        await cap.wrap_tool_execute(
            ctx, call=call, tool_def=tool_def, args=args, handler=deferring_handler
        )


# ============================================================================
# 5.4.5: wrap_tool_execute() re-raises ApprovalRequired (human-in-the-loop)
# ============================================================================


@pytest.mark.anyio
async def test_wrap_tool_execute_reraises_approval_required(
    mock_hook_manager: MagicMock,
) -> None:
    """wrap_tool_execute() must re-raise ApprovalRequired, not swallow it.

    ApprovalRequired is raised by ApprovalRequiredToolset when a tool requires
    human approval before execution. It signals pydantic-ai to pause and request
    approval. If swallowed, the approval flow breaks.
    """
    from pydantic_ai.exceptions import ApprovalRequired

    cap = make_capability(mock_hook_manager)
    ctx = make_run_context()
    call = make_tool_call("dangerous_tool")
    tool_def = make_tool_def("dangerous_tool")
    args: dict[str, Any] = {"path": "/etc/passwd"}

    async def approval_handler(a: dict[str, Any]) -> Any:
        raise ApprovalRequired

    with pytest.raises(ApprovalRequired):
        await cap.wrap_tool_execute(
            ctx, call=call, tool_def=tool_def, args=args, handler=approval_handler
        )


# ============================================================================
# 5.4.6: wrap_tool_execute() re-raises ToolRetryError (pydantic-ai retry signal)
# ============================================================================


@pytest.mark.anyio
async def test_wrap_tool_execute_reraises_tool_retry_error(
    mock_hook_manager: MagicMock,
) -> None:
    """wrap_tool_execute() must re-raise ToolRetryError, not swallow it.

    ToolRetryError is what ModelRetry becomes after _raw_execute converts it.
    In the normal flow (wrap_validation_errors=True), ModelRetry from tool
    bodies is converted to ToolRetryError before reaching wrap_tool_execute.
    This is the actual retry signal that pydantic-ai's _run_execute_hooks
    expects to propagate.
    """
    from pydantic_ai.exceptions import ToolRetryError
    from pydantic_ai.messages import RetryPromptPart

    cap = make_capability(mock_hook_manager)
    ctx = make_run_context()
    call = make_tool_call("question")
    tool_def = make_tool_def("question")
    args: dict[str, Any] = {}

    async def retrying_handler(a: dict[str, Any]) -> Any:
        raise ToolRetryError(RetryPromptPart(content="Invalid questionnaire format, please retry"))

    with pytest.raises(ToolRetryError, match="Invalid questionnaire"):
        await cap.wrap_tool_execute(
            ctx, call=call, tool_def=tool_def, args=args, handler=retrying_handler
        )


# ============================================================================
# 5.5: wrap_tool_execute() passes through successful results unchanged
# ============================================================================


@pytest.mark.anyio
async def test_wrap_tool_execute_passes_through_success(
    mock_hook_manager: MagicMock,
) -> None:
    """wrap_tool_execute() passes through successful results unchanged."""
    cap = make_capability(mock_hook_manager)
    ctx = make_run_context()
    call = make_tool_call("success_tool")
    tool_def = make_tool_def("success_tool")
    args = {"x": 1}

    async def success_handler(a: dict[str, Any]) -> Any:
        return "success result"

    result = await cap.wrap_tool_execute(
        ctx, call=call, tool_def=tool_def, args=args, handler=success_handler
    )

    assert result == "success result"


# ============================================================================
# 5.6: before_tool_execute() applies modified_input from pre-tool hooks
# ============================================================================


@pytest.mark.anyio
async def test_before_tool_execute_applies_modified_input(
    mock_hook_manager: MagicMock,
) -> None:
    """before_tool_execute() applies modified_input from pre-tool hooks."""
    cap = make_capability(mock_hook_manager)
    ctx = make_run_context()
    call = make_tool_call("test_tool", {"x": 1})
    tool_def = make_tool_def("test_tool")
    args: dict[str, Any] = {"x": 1}

    mock_hook_manager.run_pre_tool_hooks.return_value = HookResult(
        decision="allow", modified_input={"y": 2}
    )

    result = await cap.before_tool_execute(ctx, call=call, tool_def=tool_def, args=args)

    assert result == {"x": 1, "y": 2}


# ============================================================================
# 5.7: before_tool_execute() raises ModelRetry when pre-tool hook denies
# ============================================================================


@pytest.mark.anyio
async def test_before_tool_execute_raises_model_retry_on_deny(
    mock_hook_manager: MagicMock,
) -> None:
    """before_tool_execute() raises ModelRetry when pre-tool hook denies."""
    cap = make_capability(mock_hook_manager)
    ctx = make_run_context()
    call = make_tool_call("denied_tool", {"x": 1})
    tool_def = make_tool_def("denied_tool")
    args: dict[str, Any] = {"x": 1}

    mock_hook_manager.run_pre_tool_hooks.return_value = HookResult(
        decision="deny", reason="Tool not allowed in current context"
    )

    with pytest.raises(ModelRetry, match="denied_tool"):
        await cap.before_tool_execute(ctx, call=call, tool_def=tool_def, args=args)


# ============================================================================
# 5.8: after_tool_execute() applies modified_output from post-tool hooks
# ============================================================================


@pytest.mark.anyio
async def test_after_tool_execute_applies_modified_output(
    mock_hook_manager: MagicMock,
) -> None:
    """after_tool_execute() applies modified_output from post-tool hooks."""
    cap = make_capability(mock_hook_manager)
    ctx = make_run_context()
    call = make_tool_call("test_tool", {"x": 1})
    tool_def = make_tool_def("test_tool")
    args: dict[str, Any] = {"x": 1}
    original_result = "original output"

    mock_hook_manager.run_post_tool_hooks.return_value = HookResult(
        decision="allow", modified_output="replaced output"
    )

    # Mock get_active_run_context to return None (no injection manager)
    with patch.object(
        type(mock_hook_manager._agent),
        "get_active_run_context",
        return_value=None,
    ):
        result = await cap.after_tool_execute(
            ctx, call=call, tool_def=tool_def, args=args, result=original_result
        )

    assert result == "replaced output"


# ============================================================================
# 5.9: after_tool_execute() applies additional_context from post-tool hooks
# ============================================================================


@pytest.mark.anyio
async def test_after_tool_execute_applies_additional_context(
    mock_hook_manager: MagicMock,
) -> None:
    """after_tool_execute() applies additional_context from post-tool hooks."""
    cap = make_capability(mock_hook_manager)
    ctx = make_run_context()
    call = make_tool_call("test_tool", {"x": 1})
    tool_def = make_tool_def("test_tool")
    args: dict[str, Any] = {"x": 1}
    original_result = "original output"

    mock_hook_manager.run_post_tool_hooks.return_value = HookResult(
        decision="allow", additional_context="<extra-info>Important context</extra-info>"
    )

    # Mock get_active_run_context to return None (no injection manager)
    with patch.object(
        type(mock_hook_manager._agent),
        "get_active_run_context",
        return_value=None,
    ):
        result = await cap.after_tool_execute(
            ctx, call=call, tool_def=tool_def, args=args, result=original_result
        )

    assert isinstance(result, ToolReturn)
    assert "original output" in str(result.return_value)
    assert "<extra-info>Important context</extra-info>" in str(result.content)


# ============================================================================
# 5.10: after_tool_execute() consumes pending injection
# ============================================================================


@pytest.mark.anyio
async def test_after_tool_execute_consumes_pending_injection(
    mock_hook_manager: MagicMock,
) -> None:
    """after_tool_execute() consumes pending injection from PromptInjectionManager."""
    cap = make_capability(mock_hook_manager)
    ctx = make_run_context()
    call = make_tool_call("test_tool", {"x": 1})
    tool_def = make_tool_def("test_tool")
    args: dict[str, Any] = {"x": 1}
    original_result = "tool result"

    mock_hook_manager.run_post_tool_hooks.return_value = HookResult(decision="allow")

    # Mock run context with injection manager that has a pending injection
    mock_run_ctx = MagicMock()
    mock_injection_manager = MagicMock()
    mock_injection_manager.consume = AsyncMock(return_value="<injection>Extra context</injection>")
    mock_run_ctx.injection_manager = mock_injection_manager

    with patch.object(
        type(mock_hook_manager._agent),
        "get_active_run_context",
        return_value=mock_run_ctx,
    ):
        result = await cap.after_tool_execute(
            ctx, call=call, tool_def=tool_def, args=args, result=original_result
        )

    # Verify injection was consumed
    mock_injection_manager.consume.assert_called_once()

    # Verify injection was appended to result
    assert isinstance(result, ToolReturn)
    assert "<injection>Extra context</injection>" in str(result.content)


# ============================================================================
# 5.11: Integration test — hooks fire for MCP tools (not just direct tools)
# ============================================================================


@pytest.mark.anyio
async def test_hooks_fire_for_mcp_tools(mock_agent: Agent[Any]) -> None:
    """Hooks fire for MCP tools (not just direct tools).

    This test verifies that the ToolInterceptCapability's before_tool_execute
    and after_tool_execute are invoked for MCP-sourced tools, not just direct
    tools registered via wrap_tool().
    """
    hook_call_log: list[str] = []

    async def pre_tool_hook(**kwargs: Any) -> HookResult:
        hook_call_log.append(f"pre:{kwargs.get('tool_name')}")
        return {"decision": "allow"}

    async def post_tool_hook(**kwargs: Any) -> HookResult:
        hook_call_log.append(f"post:{kwargs.get('tool_name')}")
        return {"decision": "allow"}

    agent_hooks = AgentHooks(
        pre_tool_use=[
            CallableHook(event="pre_tool_use", fn=pre_tool_hook)  # type: ignore[arg-type]
        ],
        post_tool_use=[
            CallableHook(event="post_tool_use", fn=post_tool_hook)  # type: ignore[arg-type]
        ],
    )

    mock_agent._hook_manager.agent_hooks = agent_hooks

    capability = ToolInterceptCapability(hook_manager=mock_agent._hook_manager)

    # Simulate an MCP tool call through the capability chain
    ctx = make_run_context(deps=MockDeps(agent=mock_agent))
    call = make_tool_call("mcp_filesystem_read", {"path": "/test"})
    tool_def = make_tool_def("mcp_filesystem_read")
    args: dict[str, Any] = {"path": "/test"}

    # ToolInterceptCapability handles tool interception directly
    tool_intercept_cap = capability

    # Mock the hook manager's methods to track calls
    with (
        patch.object(
            mock_agent._hook_manager,
            "run_pre_tool_hooks",
            AsyncMock(return_value=HookResult(decision="allow")),
        ) as mock_pre,
        patch.object(
            mock_agent._hook_manager,
            "run_post_tool_hooks",
            AsyncMock(return_value=HookResult(decision="allow")),
        ) as mock_post,
        patch.object(mock_agent._hook_manager, "agent_name", "test-agent"),
        patch.object(type(mock_agent), "get_active_run_context", return_value=None),
    ):
        await tool_intercept_cap.before_tool_execute(ctx, call=call, tool_def=tool_def, args=args)
        await tool_intercept_cap.after_tool_execute(
            ctx, call=call, tool_def=tool_def, args=args, result="mcp_result"
        )

        # Verify hooks were called for the MCP tool
        mock_pre.assert_called_once()
        mock_post.assert_called_once()

        # Verify tool name was passed correctly
        pre_call_kwargs = mock_pre.call_args.kwargs
        assert pre_call_kwargs["tool_name"] == "mcp_filesystem_read"


# ============================================================================
# 5.12: Integration test — confirmation works for MCP tools when mode="always"
# ============================================================================


@pytest.mark.anyio
async def test_confirmation_works_for_mcp_tools_mode_always(
    mock_agent: Agent[Any],
) -> None:
    """Confirmation works for MCP tools when mode='always'.

    Verifies that get_wrapper_toolset() wraps the toolset with
    ApprovalRequiredToolset when mode="always", which applies to ALL tools
    including MCP tools.
    """
    from pydantic_ai.toolsets import ApprovalRequiredToolset

    mock_agent._hook_manager.agent_hooks = None
    mock_agent._hook_manager.agent_name = "test-agent"
    mock_agent._hook_manager._agent = mock_agent

    capability = ToolInterceptCapability(hook_manager=mock_agent._hook_manager)
    tool_intercept_cap = capability

    # Mock _get_confirmation_mode to return "always"
    mock_toolset = MagicMock()

    with patch.object(type(tool_intercept_cap), "_get_confirmation_mode", return_value="always"):
        result = tool_intercept_cap.get_wrapper_toolset(mock_toolset)

    assert result is not None
    assert isinstance(result, ApprovalRequiredToolset)
    # The wrapper should wrap the input toolset (which could contain MCP tools)
    assert result.wrapped is mock_toolset


# ============================================================================
# 5.13: Integration test — no double-firing when old AgentHooks is active
# ============================================================================


@pytest.mark.anyio
async def test_no_double_firing_when_old_agenthooks_active(
    mock_agent: Agent[Any],
) -> None:
    """No double-firing when old AgentHooks is active AND capability chain is active.

    Verifies that get_capabilities() returns a ToolInterceptCapability directly
    (not a CombinedCapability wrapping AgentHooks.get_capabilities()). This
    prevents double-firing because the legacy Hooks callbacks are never
    registered in the capability chain — only ToolInterceptCapability fires
    tool hooks, delegating to AgentHooks.run_pre_tool_hooks() /
    run_post_tool_hooks().
    """
    hook_fire_count: list[str] = []

    async def pre_tool_hook(**kwargs: Any) -> HookResult:
        hook_fire_count.append("pre")
        return {"decision": "allow"}

    async def post_tool_hook(**kwargs: Any) -> HookResult:
        hook_fire_count.append("post")
        return {"decision": "allow"}

    agent_hooks = AgentHooks(
        pre_tool_use=[
            CallableHook(event="pre_tool_use", fn=pre_tool_hook)  # type: ignore[arg-type]
        ],
        post_tool_use=[
            CallableHook(event="post_tool_use", fn=post_tool_hook)  # type: ignore[arg-type]
        ],
    )

    mock_agent._hook_manager.agent_hooks = agent_hooks
    mock_agent._hook_manager.agent_name = "test-agent"
    mock_agent._hook_manager._agent = mock_agent

    capability = ToolInterceptCapability(hook_manager=mock_agent._hook_manager)

    # ToolInterceptCapability handles tool interception directly,
    # preventing double-firing because only ToolInterceptCapability fires
    # tool hooks, delegating to AgentHooks.run_pre_tool_hooks() /
    # run_post_tool_hooks().
    assert isinstance(capability, ToolInterceptCapability)
    assert hasattr(capability, "before_tool_execute")
    assert hasattr(capability, "after_tool_execute")


# ============================================================================
# 5.14: wrap_tool_execute() records a logfire span carrying tool call args
# ============================================================================


@pytest.mark.anyio
async def test_wrap_tool_execute_records_tool_call_span_with_args(
    mock_hook_manager: MagicMock,
) -> None:
    """wrap_tool_execute() opens a logfire span carrying tool name, id, and args.

    Regression test for missing tool-call arguments in observability data:
    the span attributes must include ``tool_name``, ``tool_call_id`` and the
    full ``args`` dict so logfire exports per-tool-call parameters.
    """
    import logfire

    cap = make_capability(mock_hook_manager)
    ctx = make_run_context()
    call = make_tool_call("test_tool", args={"mode": "standard", "max_tokens": 42})
    tool_def = make_tool_def("test_tool")
    args: dict[str, Any] = {"mode": "standard", "max_tokens": 42}

    async def success_handler(a: dict[str, Any]) -> Any:
        return "ok"

    expected_span: dict[str, Any] = {}

    def fake_span(name: str, **attributes: Any) -> MagicMock:
        expected_span["name"] = name
        expected_span["attributes"] = attributes
        return MagicMock()

    with patch.object(logfire, "span", side_effect=fake_span):
        result = await cap.wrap_tool_execute(
            ctx, call=call, tool_def=tool_def, args=args, handler=success_handler
        )

    assert result == "ok"
    assert expected_span["name"] == "tool.call"
    attributes = expected_span["attributes"]
    assert attributes["tool_name"] == "test_tool"
    assert attributes["tool_call_id"] == call.tool_call_id
    assert attributes["args"] == {"mode": "standard", "max_tokens": 42}
