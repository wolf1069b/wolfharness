"""Tool wrapping utilities for pydantic-ai integration.

Simplified after unified tool interception: hooks, confirmation, and injection
are now handled by ``_ToolInterceptCapability`` in the capability chain.
This module retains only ``AgentContext`` injection and deferred execution support.
"""

from __future__ import annotations

from functools import wraps
import inspect
from typing import TYPE_CHECKING, Any, cast

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred
from pydantic_ai.messages import ToolReturn

from wolfharness.agents.context import AgentContext
from wolfharness.log import get_logger
from wolfharness.tools.base import ToolResult
from wolfharness.utils.inspection import execute, get_argument_key
from wolfharness.utils.signatures import create_modified_signature, update_signature


logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from wolfharness.tools.base import Tool


def _inject_additional_context(
    result: Any,
    additional: str,
) -> ToolReturn:
    """Inject additional context into a tool result.

    Wraps or modifies the result to include additional context that will
    be visible to the model after the tool execution.

    Args:
        result: Original tool result (any type or ToolReturn)
        additional: Additional context to inject (already XML-wrapped)

    Returns:
        ToolReturn with the additional context appended to content
    """
    from dataclasses import replace

    if isinstance(result, ToolReturn):
        existing = result.content
        if existing is None:
            return replace(result, content=additional)
        if isinstance(existing, str):
            return replace(result, content=f"{existing}\n\n{additional}")
        return replace(result, content=[*existing, f"\n\n{additional}"])
    return ToolReturn(return_value=result, content=additional)


def _convert_result(result: Any) -> Any:
    """Convert AgentPool ToolResult to pydantic-ai ToolReturn."""
    if isinstance(result, ToolResult):
        val = result.structured_content or result.content
        return ToolReturn(return_value=val, content=result.content, metadata=result.metadata)
    return result


def wrap_tool[TReturn](
    tool: Tool[TReturn],
    agent_ctx: AgentContext,
) -> Callable[..., Awaitable[TReturn | ToolReturn | None]]:
    """Wrap tool with AgentContext injection.

    Confirmation, hooks, and injection are handled by the capability chain's
    ``_ToolInterceptCapability`` (via ``get_wrapper_toolset``, ``before_tool_execute``,
    ``wrap_tool_execute``, ``after_tool_execute``). This function only handles
    AgentContext parameter injection and deferred execution support.

    Args:
        tool: The tool to wrap.
        agent_ctx: Agent context for dependency injection.
    """
    fn = tool.get_callable()
    run_ctx_key = get_argument_key(fn, RunContext)
    agent_ctx_key = get_argument_key(fn, AgentContext)

    if run_ctx_key:
        param_names = list(inspect.signature(fn).parameters.keys())
        run_ctx_index = param_names.index(run_ctx_key)
        if run_ctx_index != 0:
            msg = f"Tool {tool.name!r}: RunContext param {run_ctx_key!r} must come first."
            raise ValueError(msg)

    if run_ctx_key or agent_ctx_key:

        async def wrapped(  # pyright: ignore[reportRedeclaration]
            ctx: RunContext, *args: Any, **kwargs: Any
        ) -> TReturn | ToolReturn | None:  # pyright: ignore
            if agent_ctx.data is None:
                agent_ctx.data = ctx.deps

            # Always update tool_call_id and tool_name on the AgentContext,
            # even when agent_ctx_key is None (e.g., when the tool function
            # takes RunContext[AgentContext] instead of AgentContext directly).
            # Without this, handle_elicitation() falls back to run_ctx.run_id
            # as the elicitation handle, causing a mismatch between
            # PendingDeferredCall.tool_call_id and ToolCallPart.tool_call_id.
            if ctx.deps is not None:
                # ctx.deps is typed as object, but at runtime it's an AgentContext.
                # These assignments are safe because the deps type is always AgentContext
                # for AgentPool agents, and the fields are set here for tool identification.
                ctx.deps.tool_call_id = ctx.tool_call_id or ""  # type: ignore[attr-defined]
                ctx.deps.tool_name = ctx.tool_name or ""  # type: ignore[attr-defined]

            if agent_ctx_key:
                model_name = f"{ctx.model.system}:{ctx.model.model_name}" if ctx.model else None
                call_ctx = _replace_agent_ctx(
                    agent_ctx,
                    tool_name=ctx.tool_name or "",
                    tool_call_id=ctx.tool_call_id or "",
                    tool_input=kwargs.copy(),
                    model_name=model_name,
                    run_ctx=ctx.deps.run_ctx if ctx.deps else None,  # type: ignore[attr-defined]
                )
                kwargs[agent_ctx_key] = call_ctx

            if run_ctx_key:

                def _exec(*a: Any, **kw: Any) -> Any:
                    return execute(fn, ctx, *a, **kw)

            else:

                def _exec(*a: Any, **kw: Any) -> Any:
                    return execute(fn, *a, **kw)

            if tool.deferred:
                try:
                    return cast(
                        TReturn | ToolReturn | None, _convert_result(await _exec(*args, **kwargs))
                    )
                except (CallDeferred, ApprovalRequired) as exc:
                    return await _handle_deferred_exception(exc, tool)
            return cast(TReturn | ToolReturn | None, _convert_result(await _exec(*args, **kwargs)))

    else:

        async def wrapped(*args: Any, **kwargs: Any) -> TReturn | ToolReturn | None:  # type: ignore[misc]
            if tool.deferred:
                try:
                    return cast(
                        TReturn | ToolReturn | None,
                        _convert_result(await execute(fn, *args, **kwargs)),
                    )
                except (CallDeferred, ApprovalRequired) as exc:
                    return await _handle_deferred_exception(exc, tool)
            return cast(
                TReturn | ToolReturn | None, _convert_result(await execute(fn, *args, **kwargs))
            )

    wraps(fn)(wrapped)  # pyright: ignore
    wrapped.__annotations__ = fn.__annotations__
    wrapped.__doc__ = tool.description
    wrapped.__name__ = tool.name
    if agent_ctx_key and not run_ctx_key:
        new_sig = create_modified_signature(fn, remove=agent_ctx_key, inject={"ctx": RunContext})
        update_signature(wrapped, new_sig)
    elif agent_ctx_key and run_ctx_key:
        new_sig = create_modified_signature(fn, remove=agent_ctx_key)
        update_signature(wrapped, new_sig)
    return wrapped


def _replace_agent_ctx(
    agent_ctx: AgentContext,
    *,
    tool_name: str,
    tool_call_id: str,
    tool_input: dict[str, Any],
    model_name: str | None,
    run_ctx: Any | None,
) -> AgentContext:
    """Create a per-call copy of AgentContext with tool execution fields."""
    from dataclasses import replace

    return replace(
        agent_ctx,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        tool_input=tool_input,
        model_name=model_name,
        run_ctx=run_ctx,
    )


async def _handle_deferred_exception(
    exc: CallDeferred | ApprovalRequired,
    tool: Tool[Any],
) -> ToolReturn:
    """Handle a deferred execution exception raised during resume re-execution.

    When a tool body raises ``CallDeferred`` or ``ApprovalRequired`` during
    resume re-execution (after ``ToolApproved`` triggers the body), this
    function routes the exception to ``DeferredToolBridge`` based on
    ``tool.deferred_strategy``.

    For ``block`` strategy, the bridge checkpoints and emits deferral events.
    For ``continue`` strategy, the bridge resolves inline with a placeholder.

    .. note::
        ``DeferredToolBridge`` integration is deferred to Task 12.
        Until the bridge is available, this function re-raises the original
        exception so that pydantic-ai's native handling applies as fallback.

    Args:
        exc: The deferred execution exception raised by the tool body.
        tool: The Tool instance that was executing.

    Returns:
        A ``ToolReturn`` placeholder representing the deferred result.
    """
    raise exc
