"""Bridge pydantic-ai deferred tool execution signals to AgentPool's durable execution layer.

When pydantic-ai encounters tools with ``deferred=True``, the
`HandleDeferredToolCalls` capability intercepts the deferred requests and
classifies them by strategy:

- **block**: Emit ``ToolCallDeferredEvent`` and exclude from returned results.
  The tool call remains unresolved in pydantic-ai's ``FinalResult``, enabling
  the CheckpointManager (Task 13) to persist state for later resumption.
- **continue**: Resolve inline with a placeholder ``ToolReturn``, allowing
  the agent to continue processing.
- **non-deferred**: Return ``None`` to pass through to the next capability
  (typically ``ApprovalBridge``).

This capability MUST be registered BEFORE ``approval_bridge`` in the
pydantic-ai capability chain so that block-strategy tools are intercepted
before approval_bridge resolves them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.messages import ToolReturn

from wolfharness.agents.events.events import ToolCallDeferredEvent
from wolfharness.log import get_logger


if TYPE_CHECKING:
    from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, RunContext

    from wolfharness.agents.context import AgentContext


logger = get_logger(__name__)

DEFAULT_PLACEHOLDER = "This tool is processing in the background."


async def _emit_deferred_event(
    ctx: RunContext[AgentContext[Any]],
    event: ToolCallDeferredEvent,
) -> None:
    """Publish a ``ToolCallDeferredEvent`` to the event bus.

    Args:
        ctx: pydantic-ai RunContext with ``AgentContext`` as deps.
        event: The deferred event to emit.
    """
    run_ctx = ctx.deps.run_ctx
    if run_ctx is None:
        logger.debug("No run_ctx available — event dropped", tool_name=event.tool_name)
        return

    if run_ctx.event_bus is not None:
        await run_ctx.event_bus.publish(run_ctx.session_id, event)
    else:
        logger.warning("No event_bus available — deferred event dropped", tool_name=event.tool_name)


async def _resolve_deferred_calls(
    ctx: RunContext[AgentContext[Any]],
    requests: DeferredToolRequests,
    deferred_tools: dict[str, str],
) -> DeferredToolResults | None:
    """Classify deferred tool requests by strategy and resolve or defer.

    Args:
        ctx: pydantic-ai RunContext with AgentContext as deps.
        requests: Deferred tool requests from pydantic-ai.
        deferred_tools: Mapping of tool_name → deferred_strategy for tools
            with ``deferred=True`` in AgentPool configuration.

    Returns:
        ``DeferredToolResults`` with continue-strategy calls resolved and
        block-strategy calls excluded, or ``None`` if no deferred tools
        were found in the requests.
    """
    run_ctx = ctx.deps.run_ctx
    session_id = run_ctx.session_id if run_ctx is not None else ""

    continue_results: dict[str, Any] = {}
    has_any_deferred = False

    # Only handle `calls` (external execution). `approvals` (human-in-the-loop)
    # are handled by the next capability (approval_bridge).
    for call in requests.calls:
        strategy = deferred_tools.get(call.tool_name)
        if strategy is None:
            continue  # Not a deferred tool — let next capability handle it

        has_any_deferred = True

        match strategy:
            case "block":
                # Emit event for block-strategy tools.
                # CheckpointManager (Task 13) will persist state.
                event = ToolCallDeferredEvent(
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                    deferred_strategy="block",
                    deferred_handle="",
                    status="pending",
                    session_id=session_id,
                )
                await _emit_deferred_event(ctx, event)
                # Mark the run as checkpointed so that the orchestrator
                # can transition the RunHandle to checkpointed status.
                run_ctx = ctx.deps.run_ctx
                if run_ctx is not None:
                    run_ctx.checkpointed = True
                logger.debug(
                    "Deferred tool call (block strategy)",
                    tool_name=call.tool_name,
                    tool_call_id=call.tool_call_id,
                )
                # Block calls are NOT included in results → remain unresolved

            case "continue":
                # Resolve inline with placeholder so agent can continue.
                placeholder = ToolReturn(return_value=DEFAULT_PLACEHOLDER)
                continue_results[call.tool_call_id] = placeholder
                logger.debug(
                    "Deferred tool call (continue strategy) — resolved with placeholder",
                    tool_name=call.tool_name,
                    tool_call_id=call.tool_call_id,
                )

    if not has_any_deferred:
        # No deferred tools found — pass through to next capability (approval_bridge)
        return None

    # Build results with continue-strategy calls resolved.
    # Block-strategy calls are excluded → they remain unresolved.
    # Non-deferred calls are also excluded → they flow to the next capability
    # via the CombinedCapability.handle_deferred_tool_calls pipeline.
    return requests.build_results(calls=continue_results)


def create_deferred_bridge_capability(
    deferred_tools: dict[str, str],
) -> HandleDeferredToolCalls[AgentContext[Any]]:
    """Create a ``HandleDeferredToolCalls`` capability for deferred tool bridging.

    This capability MUST be registered BEFORE ``approval_bridge`` in the
    pydantic-ai capability chain (see Decision 9 in design.md).

    Args:
        deferred_tools: Mapping of ``tool_name`` → ``deferred_strategy`` for
            tools configured with ``deferred=True`` in AgentPool. Strategy
            must be ``"block"`` or ``"continue"``.

    Returns:
        ``HandleDeferredToolCalls`` capability configured with the bridge handler.
    """

    async def handler(
        ctx: RunContext[AgentContext[Any]],
        requests: DeferredToolRequests,
    ) -> DeferredToolResults | None:
        return await _resolve_deferred_calls(ctx, requests, deferred_tools)

    return HandleDeferredToolCalls(handler=handler)
