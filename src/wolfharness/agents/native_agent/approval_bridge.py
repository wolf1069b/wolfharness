"""Bridge pydantic-ai ApprovalRequiredToolset signals to AgentPool InputProvider.

When pydantic-ai defers tool calls for approval (via `requires_approval=True`),
the `HandleDeferredToolCalls` capability intercepts the deferred requests and
routes them through AgentPool's `InputProvider` for UI confirmation.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.tools import (
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolApproved,
    ToolDenied,
)

from wolfharness.log import get_logger


if TYPE_CHECKING:
    from wolfharness import Agent
    from wolfharness.agents.context import AgentContext, ConfirmationResult


logger = get_logger(__name__)


def _map_confirmation_result(
    result: ConfirmationResult,
    tool_name: str,
) -> ToolApproved | ToolDenied:
    """Map AgentPool ConfirmationResult to pydantic-ai approval result.

    Args:
        result: AgentPool confirmation result ("allow", "skip", "abort_run", "abort_chain")
        tool_name: Name of the tool being confirmed

    Returns:
        ToolApproved if allowed, ToolDenied otherwise
    """
    match result:
        case "allow":
            return ToolApproved()
        case "skip":
            return ToolDenied(message=f"Tool {tool_name} execution skipped")
        case "abort_run":
            return ToolDenied(message=f"Tool {tool_name} denied: run aborted by user")
        case "abort_chain":
            return ToolDenied(message=f"Tool {tool_name} denied: agent chain aborted by user")


async def _resolve_deferred_approvals(
    ctx: RunContext[AgentContext],
    requests: DeferredToolRequests,
    input_provider: Any | None = None,
) -> DeferredToolResults | None:
    """Resolve deferred approval requests via InputProvider.

    For each approval request in the deferred tool requests, calls
    `InputProvider.get_tool_confirmation()` and maps the result to
    pydantic-ai's `ToolApproved` or `ToolDenied`.

    Args:
        ctx: pydantic-ai RunContext with AgentContext as deps
        requests: Deferred tool requests from pydantic-ai
        input_provider: Optional InputProvider to use directly instead of
            resolving via ctx.deps.get_input_provider()

    Returns:
        DeferredToolResults with approval/denial for each request,
        or None if there are no approval requests to handle
    """
    if not requests.approvals:
        return None

    agent_ctx = ctx.deps
    provider = input_provider
    if provider is None:
        provider = agent_ctx.get_input_provider()

    approvals: dict[str, bool | ToolApproved | ToolDenied] = {}

    for call in requests.approvals:
        tool_name = call.tool_name

        # Build confirmation context with tool execution details
        confirm_ctx = replace(
            agent_ctx,
            tool_name=tool_name,
            tool_call_id=call.tool_call_id,
            tool_input=call.args if isinstance(call.args, dict) else {},
        )

        try:
            result = await provider.get_tool_confirmation(confirm_ctx, "")
        except Exception:
            logger.exception(
                "InputProvider.get_tool_confirmation failed",
                tool_name=tool_name,
                tool_call_id=call.tool_call_id,
            )
            # Default to denial on provider error
            result = "skip"

        approvals[call.tool_call_id] = _map_confirmation_result(result, tool_name)

    return DeferredToolResults(approvals=approvals)


def create_approval_bridge_capability(
    agent: Agent[Any, Any],
    input_provider: Any | None = None,
) -> HandleDeferredToolCalls[AgentContext[Any]]:
    """Create a HandleDeferredToolCalls capability bridged to InputProvider.

    This capability intercepts pydantic-ai deferred tool approval requests
    and routes them through AgentPool's `InputProvider` for user confirmation.

    Args:
        agent: The Agent instance (used to access tool_confirmation_mode)
        input_provider: Optional InputProvider to use directly. If None,
            resolves via ctx.deps.get_input_provider() at runtime.

    Returns:
        HandleDeferredToolCalls capability configured with the bridge handler
    """

    async def handler(
        ctx: RunContext[AgentContext],
        requests: DeferredToolRequests,
    ) -> DeferredToolResults | None:
        # Only handle approval requests (not external execution calls)
        if requests.approvals:
            return await _resolve_deferred_approvals(ctx, requests, input_provider)
        return None

    return HandleDeferredToolCalls(handler=handler)
