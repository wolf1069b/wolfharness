"""Shared utility for wrapping AgentPool tools for pydantic-ai.

Extracted from ``AbstractCapability._wrap_for_pydantic_ai`` so that
``ToolsetFactory`` implementations can wrap tools without depending on the
deprecated ``AbstractCapability`` hierarchy.

The long-term migration path is:

1. ``StaticToolsetFactory`` and friends call ``wrap_tool_for_pydantic_ai``
   directly (no ``AbstractCapability`` import).
2. ``AbstractCapability._wrap_for_pydantic_ai`` delegates here for backwards
   compatibility.
3. Once all callers migrate, ``AbstractCapability`` is removed.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from pydantic_ai import ModelRetry, RunContext


if TYPE_CHECKING:
    from wolfharness.agents.context import AgentContext
    from wolfharness.tools.base import Tool


def wrap_tool_for_pydantic_ai(tool: Tool[Any]) -> Any:
    """Wrap an AgentPool tool so pydantic-ai can schema-generate it.

    AgentPool tools take ``AgentContext`` as a parameter, but pydantic-ai
    only recognizes ``RunContext``. This wrapper creates a function that
    accepts ``RunContext`` (which carries ``AgentContext`` in ``deps``) and
    injects the ``AgentContext`` into the original tool call.
    """
    from wolfharness.agents.context import AgentContext

    original_fn = tool.get_callable()
    sig = inspect.signature(original_fn)

    # Find the AgentContext parameter (handle string annotations from __future__)
    agent_ctx_param: str | None = None
    for name, param in sig.parameters.items():
        ann = param.annotation
        if ann is AgentContext or (isinstance(ann, type) and ann is AgentContext):
            agent_ctx_param = name
            break
        # Handle string annotations (from __future__ import annotations)
        if isinstance(ann, str) and "AgentContext" in ann:
            agent_ctx_param = name
            break

    if agent_ctx_param is None:
        # No AgentContext - pass through directly
        return tool.to_pydantic_ai()

    # Build a wrapper that accepts RunContext and injects AgentContext/RunContext
    other_params: list[inspect.Parameter] = []
    run_ctx_param: str | None = None
    for n, p in sig.parameters.items():
        if n == agent_ctx_param:
            continue
        ann = p.annotation
        # Detect RunContext parameter in original function
        if ann is RunContext or (isinstance(ann, str) and "RunContext" in ann):
            run_ctx_param = n
            continue
        other_params.append(p)

    async def wrapper(ctx: RunContext[AgentContext], *args: Any, **kwargs: Any) -> Any:
        from dataclasses import replace

        agent_ctx = replace(
            ctx.deps,
            tool_name=ctx.tool_name,
            tool_call_id=ctx.tool_call_id,
            tool_input=kwargs.copy(),
        )
        try:
            sig_bound = sig.bind_partial(*args, **kwargs)
        except TypeError as e:
            valid_params = [
                name
                for name, p in sig.parameters.items()
                if name not in (agent_ctx_param, run_ctx_param)
            ]
            msg = str(e)
            raise ModelRetry(
                f"Tool '{tool.name}' called with invalid arguments: {msg}. "
                f"Accepted parameters: {valid_params}"
            ) from e
        sig_bound.arguments[agent_ctx_param] = agent_ctx
        if run_ctx_param is not None:
            sig_bound.arguments[run_ctx_param] = ctx
        return await tool.execute(*sig_bound.args, **sig_bound.kwargs)

    # Copy metadata
    wrapper.__name__ = tool.name
    wrapper.__doc__ = tool.description
    wrapper.__wrapped__ = original_fn  # type: ignore[attr-defined]

    # Build signature: RunContext + other params (without AgentContext/RunContext)
    new_params = [
        inspect.Parameter("ctx", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=RunContext)
    ]
    new_params.extend(other_params)
    wrapper.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        new_params, return_annotation=sig.return_annotation
    )
    wrapper.__annotations__ = {"ctx": RunContext}
    for n, p in sig.parameters.items():
        if n in (agent_ctx_param, run_ctx_param):
            continue
        if p.annotation is not inspect.Parameter.empty:
            wrapper.__annotations__[n] = p.annotation
    if sig.return_annotation is not inspect.Signature.empty:
        wrapper.__annotations__["return"] = sig.return_annotation

    return tool.to_pydantic_ai(function_override=wrapper)
