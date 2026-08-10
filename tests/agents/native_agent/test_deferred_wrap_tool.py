"""Tests for deferred-aware wrap_tool() in tool_wrapping.py.

Verifies that wrap_tool() catches CallDeferred / ApprovalRequired exceptions
raised during resume re-execution and routes to DeferredToolBridge, while
preserving identical code path for deferred=False tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic_ai.messages import ToolReturn
from pydantic_ai.tools import RunContext
import pytest

from wolfharness.agents.context import AgentContext, AgentRunContext
from wolfharness.tools import ApprovalRequired, CallDeferred, Tool


if TYPE_CHECKING:
    from collections.abc import Callable, Generator


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def agent_ctx() -> Generator[AgentContext]:
    """Create a minimal AgentContext for wrap_tool usage.

    Patches AgentContext.handle_confirmation at class level so that
    dataclasses.replace()-created copies also use the mock.
    """
    from unittest.mock import MagicMock

    node = MagicMock()
    node.name = "test-agent"
    agent_run_ctx = AgentRunContext(session_id="test-session")
    ctx = AgentContext(node=node, run_ctx=agent_run_ctx)

    with patch.object(AgentContext, "handle_confirmation", AsyncMock(return_value="allow")):
        yield ctx


@pytest.fixture
def run_ctx(agent_ctx: AgentContext) -> RunContext[Any]:
    """Create a mock RunContext with AgentContext deps."""
    model = MagicMock()
    model.system = "test"
    model.model_name = "test-model"

    return RunContext(
        deps=agent_ctx,
        model=model,
        usage=MagicMock(),
    )


# ============================================================================
# Helper: create a tool with deferred flag
# ============================================================================


def _create_deferred_tool(
    body: Callable[..., Any],
    *,
    deferred: bool = True,
    deferred_kind: str = "external",
    deferred_strategy: str = "block",
    requires_confirmation: bool = False,
) -> Tool[Any]:
    """Create a Tool with specific deferred settings."""
    tool = Tool.from_callable(body, requires_confirmation=requires_confirmation)
    tool.deferred = deferred
    tool.deferred_kind = deferred_kind  # type: ignore[assignment]
    tool.deferred_strategy = deferred_strategy  # type: ignore[assignment]
    return tool


# ============================================================================
# Tests: deferred=False preserves identical code path
# ============================================================================


@pytest.mark.unit
async def test_non_deferred_tool_normal_return(agent_ctx: AgentContext) -> None:
    """deferred=False tool returns normally — identical to current behavior."""
    from wolfharness.agents.native_agent.tool_wrapping import wrap_tool

    def normal_tool(text: str) -> str:
        return f"result: {text}"

    tool = _create_deferred_tool(normal_tool, deferred=False)
    wrapped = wrap_tool(tool, agent_ctx)

    result = await wrapped(text="hello")
    assert result == "result: hello"


@pytest.mark.unit
async def test_non_deferred_tool_with_context(
    agent_ctx: AgentContext, run_ctx: RunContext[Any]
) -> None:
    """deferred=False tool with RunContext returns normally."""
    from wolfharness.agents.native_agent.tool_wrapping import wrap_tool

    def ctx_tool(ctx: RunContext[Any], text: str) -> str:
        assert ctx.deps is not None
        return f"ctx: {text}"

    tool = _create_deferred_tool(ctx_tool, deferred=False)
    wrapped = wrap_tool(tool, agent_ctx)

    result = await wrapped(run_ctx, text="world")
    assert result == "ctx: world"


@pytest.mark.unit
async def test_non_deferred_tool_tool_result_conversion(agent_ctx: AgentContext) -> None:
    """deferred=False tool that returns ToolResult converts to ToolReturn."""
    from wolfharness.agents.native_agent.tool_wrapping import wrap_tool
    from wolfharness.tools.base import ToolResult

    def result_tool(text: str) -> ToolResult:
        return ToolResult(content=f"content: {text}", metadata={"key": "val"})

    tool = _create_deferred_tool(result_tool, deferred=False)
    wrapped = wrap_tool(tool, agent_ctx)

    result = await wrapped(text="test")
    assert isinstance(result, ToolReturn)
    assert result.content == "content: test"
    assert result.metadata == {"key": "val"}


# ============================================================================
# Tests: deferred=True catch CallDeferred during resume re-execution
# ============================================================================


@pytest.mark.unit
async def test_deferred_tool_catches_call_deferred_on_resume(
    agent_ctx: AgentContext,
) -> None:
    """Tool body raises CallDeferred → wrap_tool() catches and routes to bridge."""
    from wolfharness.agents.native_agent import tool_wrapping as tw_mod

    def deferred_body(text: str) -> str:
        raise CallDeferred(metadata={"call_id": "tc-1"})

    tool = _create_deferred_tool(deferred_body, deferred=True, deferred_kind="external")
    wrapped = tw_mod.wrap_tool(tool, agent_ctx)

    with patch.object(tw_mod, "_handle_deferred_exception", new_callable=AsyncMock) as mock_handler:
        mock_handler.return_value = ToolReturn(return_value="deferred_placeholder")
        result = await wrapped(text="hello")

    # _handle_deferred_exception should be called with the exception and tool info
    mock_handler.assert_awaited_once()
    call_args = mock_handler.call_args
    assert isinstance(call_args[0][0], CallDeferred)
    assert call_args[0][1] is tool
    assert isinstance(result, ToolReturn)
    assert result.return_value == "deferred_placeholder"


@pytest.mark.unit
async def test_deferred_tool_catches_call_deferred_with_context(
    agent_ctx: AgentContext,
    run_ctx: RunContext[Any],
) -> None:
    """Tool body with RunContext raises CallDeferred → caught properly."""
    from wolfharness.agents.native_agent import tool_wrapping as tw_mod

    def deferred_ctx_body(ctx: RunContext[Any], text: str) -> str:
        raise CallDeferred(metadata={"from": "ctx_tool"})

    tool = _create_deferred_tool(deferred_ctx_body, deferred=True, deferred_kind="external")
    wrapped = tw_mod.wrap_tool(tool, agent_ctx)

    with patch.object(tw_mod, "_handle_deferred_exception", new_callable=AsyncMock) as mock_handler:
        mock_handler.return_value = ToolReturn(return_value="placeholder")
        result = await wrapped(run_ctx, text="ctx")

    mock_handler.assert_awaited_once()
    assert isinstance(mock_handler.call_args[0][0], CallDeferred)
    assert isinstance(result, ToolReturn)
    assert result.return_value == "placeholder"


@pytest.mark.unit
async def test_deferred_tool_catches_approval_required_on_resume(
    agent_ctx: AgentContext,
) -> None:
    """Tool body raises ApprovalRequired → wrap_tool() catches and routes to bridge."""
    from wolfharness.agents.native_agent import tool_wrapping as tw_mod

    def approval_body(text: str) -> str:
        raise ApprovalRequired(metadata={"needs": "human"})

    tool = _create_deferred_tool(
        approval_body, deferred=True, deferred_kind="unapproved", deferred_strategy="block"
    )
    wrapped = tw_mod.wrap_tool(tool, agent_ctx)

    with patch.object(tw_mod, "_handle_deferred_exception", new_callable=AsyncMock) as mock_handler:
        mock_handler.return_value = ToolReturn(return_value="blocked")
        result = await wrapped(text="approve me")

    mock_handler.assert_awaited_once()
    assert isinstance(mock_handler.call_args[0][0], ApprovalRequired)
    assert isinstance(result, ToolReturn)
    assert result.return_value == "blocked"


# ============================================================================
# Tests: deferred=True normal returns pass through unchanged
# ============================================================================


@pytest.mark.unit
async def test_deferred_tool_normal_return_on_resume_passes_through(
    agent_ctx: AgentContext,
) -> None:
    """Tool body returns normally on resume — no deferred lifecycle triggered."""
    from wolfharness.agents.native_agent.tool_wrapping import wrap_tool

    def normal_body(text: str) -> str:
        return f"done: {text}"

    tool = _create_deferred_tool(normal_body, deferred=True, deferred_kind="external")
    wrapped = wrap_tool(tool, agent_ctx)

    result = await wrapped(text="resume")
    assert result == "done: resume"


@pytest.mark.unit
async def test_deferred_tool_normal_return_with_tool_result_passes_through(
    agent_ctx: AgentContext,
) -> None:
    """Tool body returns ToolResult on resume — converts to ToolReturn normally."""
    from wolfharness.agents.native_agent.tool_wrapping import wrap_tool
    from wolfharness.tools.base import ToolResult

    def result_body(text: str) -> ToolResult:
        return ToolResult(content=f"content: {text}")

    tool = _create_deferred_tool(result_body, deferred=True, deferred_kind="external")
    wrapped = wrap_tool(tool, agent_ctx)

    result = await wrapped(text="resume")
    assert isinstance(result, ToolReturn)
    assert result.content == "content: resume"


# ============================================================================
# Tests: _handle_deferred_exception unit tests
# ============================================================================


@pytest.mark.unit
async def test_handle_deferred_exception_re_raises_call_deferred(
    agent_ctx: AgentContext,
) -> None:
    """_handle_deferred_exception re-raises CallDeferred (bridge not yet integrated)."""
    from wolfharness.agents.native_agent.tool_wrapping import _handle_deferred_exception

    exc = CallDeferred(metadata={"key": "value"})
    tool = _create_deferred_tool(lambda: "unused", deferred=True)

    # Currently re-raises since DeferredToolBridge (Task 12) is not integrated.
    # When the bridge is ready, this will return a ToolReturn instead.
    with pytest.raises(CallDeferred):
        await _handle_deferred_exception(exc, tool)


@pytest.mark.unit
async def test_handle_deferred_exception_re_raises_approval_required(
    agent_ctx: AgentContext,
) -> None:
    """_handle_deferred_exception re-raises ApprovalRequired (bridge not yet integrated)."""
    from wolfharness.agents.native_agent.tool_wrapping import _handle_deferred_exception

    exc = ApprovalRequired(metadata={"reason": "confirm"})
    tool = _create_deferred_tool(
        lambda: "unused", deferred=True, deferred_kind="unapproved", deferred_strategy="block"
    )

    with pytest.raises(ApprovalRequired):
        await _handle_deferred_exception(exc, tool)


# ============================================================================
# Tests: deferred=False does NOT catch exceptions (identical behavior)
# ============================================================================


@pytest.mark.unit
async def test_non_deferred_tool_does_not_catch_call_deferred(
    agent_ctx: AgentContext,
) -> None:
    """deferred=False tool: CallDeferred propagates as normal exception."""
    from wolfharness.agents.native_agent.tool_wrapping import wrap_tool

    def raises_deferred(text: str) -> str:
        raise CallDeferred(metadata={"bad": "not deferred"})

    tool = _create_deferred_tool(raises_deferred, deferred=False)
    wrapped = wrap_tool(tool, agent_ctx)

    with pytest.raises(CallDeferred):
        await wrapped(text="test")


@pytest.mark.unit
async def test_non_deferred_tool_does_not_catch_approval_required(
    agent_ctx: AgentContext,
) -> None:
    """deferred=False tool: ApprovalRequired propagates as normal exception."""
    from wolfharness.agents.native_agent.tool_wrapping import wrap_tool

    def raises_approval(text: str) -> str:
        raise ApprovalRequired(metadata={"bad": "not deferred"})

    tool = _create_deferred_tool(raises_approval, deferred=False)
    wrapped = wrap_tool(tool, agent_ctx)

    with pytest.raises(ApprovalRequired):
        await wrapped(text="test")


# ============================================================================
# Tests: deferred=True no-context function catches exceptions
# ============================================================================


@pytest.mark.unit
async def test_deferred_no_context_function_catches_call_deferred(
    agent_ctx: AgentContext,
) -> None:
    """Tool without RunContext/AgentContext raising CallDeferred → caught."""
    from wolfharness.agents.native_agent import tool_wrapping as tw_mod

    def no_ctx_body(text: str) -> str:
        raise CallDeferred(metadata={"no_context": True})

    # Use requires_confirmation=False so handle_confirmation returns "allow" without mock
    tool = _create_deferred_tool(no_ctx_body, deferred=True, requires_confirmation=False)
    wrapped = tw_mod.wrap_tool(tool, agent_ctx)

    with patch.object(tw_mod, "_handle_deferred_exception", new_callable=AsyncMock) as mock_handler:
        mock_handler.return_value = ToolReturn(return_value="ok")
        result = await wrapped(text="test")

    mock_handler.assert_awaited_once()
    assert isinstance(result, ToolReturn)
    assert isinstance(mock_handler.call_args[0][0], CallDeferred)
