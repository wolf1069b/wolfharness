"""Adapter for converting AgentPool history processors to pydantic-ai ProcessHistory capabilities.

Replaces manual history processor resolution in
:class:`~wolfharness.agents.native_agent.agent.Agent`
with a unified capability-based approach. AgentPool history processors may have
untyped ``RunContext`` parameters; pydantic-ai's ``takes_run_context()`` requires
the first parameter to be explicitly typed as ``RunContext``. This adapter wraps
such processors so pydantic-ai correctly passes the run context.
"""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any, get_type_hints

from pydantic_ai.capabilities import ProcessHistory


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.tools import RunContext


class ProcessHistoryAdapter:
    """Adapts AgentPool history processors to pydantic-ai ProcessHistory capabilities.

    AgentPool history processors support these signatures (the context parameter
    may be untyped):

    - ``def processor(messages) -> list[ModelMessage]``
    - ``def processor(ctx, messages) -> list[ModelMessage]``
    - ``async def processor(messages) -> list[ModelMessage]``
    - ``async def processor(ctx, messages) -> list[ModelMessage]``

    pydantic-ai's :class:`~pydantic_ai.capabilities.ProcessHistory` uses
    :func:`~pydantic_ai._utils.takes_run_context` to detect context-aware
    processors, which requires the first parameter to carry a ``RunContext``
    type annotation. When an AgentPool processor has two parameters but the
    first one is untyped, pydantic-ai treats it as a single-parameter
    processor and passes ``messages`` as the first argument, which breaks
    the processor at runtime.

    This adapter detects such cases and wraps the processor with an explicit
    ``RunContext`` annotation so pydantic-ai routes arguments correctly.
    """

    @staticmethod
    def wrap_processor(processor: Callable[..., Any]) -> Callable[..., Any]:
        """Adapt a single AgentPool history processor for pydantic-ai compatibility.

        Args:
            processor: An AgentPool history processor callable.

        Returns:
            A processor callable compatible with
            :class:`~pydantic_ai.capabilities.ProcessHistory`.

        Raises:
            ValueError: If the processor does not have 1 or 2 parameters.
        """
        sig = inspect.signature(processor)
        params = list(sig.parameters.values())
        n_params = len(params)

        if n_params not in (1, 2):
            msg = f"History processor must take 1 or 2 arguments, got {n_params}"
            raise ValueError(msg)

        if n_params == 1:
            # Single-parameter processor (messages only) — already compatible.
            return processor

        # Two-parameter processor: AgentPool convention is ``(ctx, messages)``.
        first_param_name = params[0].name

        # Resolve annotations with get_type_hints to handle string annotations
        # from ``from __future__ import annotations``.
        try:
            hints = get_type_hints(processor)
        except (TypeError, ValueError, NameError):
            hints = {}

        first_param_annotation = hints.get(first_param_name)

        if first_param_annotation is not None and _is_run_context_annotation(
            first_param_annotation
        ):
            return processor

        # The first parameter is untyped or typed as something other than
        # ``RunContext``. Wrap the processor so pydantic-ai sees a properly
        # annotated signature and passes the RunContext as the first arg.
        if inspect.iscoroutinefunction(processor):

            @functools.wraps(processor)
            async def _wrapped_async(
                ctx: RunContext[Any], messages: list[ModelMessage]
            ) -> list[ModelMessage]:
                result: list[ModelMessage] = await processor(ctx, messages)
                return result

            return _wrapped_async

        @functools.wraps(processor)
        def _wrapped_sync(ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
            result: list[ModelMessage] = processor(ctx, messages)
            return result

        return _wrapped_sync

    @staticmethod
    def from_processors(
        processors: Sequence[Callable[..., Any]],
    ) -> list[ProcessHistory[Any]]:
        """Convert AgentPool history processors to pydantic-ai ProcessHistory capabilities.

        Args:
            processors: Sequence of AgentPool history processor callables.

        Returns:
            A list of :class:`~pydantic_ai.capabilities.ProcessHistory` instances,
            one per input processor, in the same order.
        """
        return [ProcessHistory(ProcessHistoryAdapter.wrap_processor(p)) for p in processors]


def _is_run_context_annotation(annotation: Any) -> bool:
    """Check whether *annotation* is ``RunContext`` or ``RunContext[...]``.

    Args:
        annotation: A Python type annotation to inspect.

    Returns:
        ``True`` when the annotation denotes pydantic-ai's ``RunContext``.
    """
    from pydantic_ai.tools import RunContext

    if annotation is RunContext:
        return True

    # typing.RunContext[SomeDeps] -> origin is RunContext
    origin = getattr(annotation, "__origin__", None)
    if origin is RunContext:
        return True

    # Handle typing.Annotated[RunContext[...], ...]
    args = getattr(annotation, "__args__", None)
    if args:
        for arg in args:
            if arg is RunContext:
                return True
            if getattr(arg, "__origin__", None) is RunContext:
                return True

    return False
