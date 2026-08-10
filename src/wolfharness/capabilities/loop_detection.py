"""Loop detection capability — prevents infinite agent delegation loops.

Tracks delegation depth via ``wrap_node_run``. When depth exceeds
``max_depth``, raises ``LoopDetectionError`` to abort the run.

Uses ``contextvars.ContextVar`` so depth propagates across nested
agent runs (each delegation starts a new run via ``for_run()`` which
would otherwise reset a plain instance counter).
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability


if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.capabilities import AgentNode, NodeResult, WrapNodeRunHandler


# Module-level ContextVar — propagated across nested async tasks and
# agent runs, so delegation depth survives the fresh copies that
# ``for_run()`` creates for child agents.
_delegation_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "wolfharness_delegation_depth",
    default=0,
)


class LoopDetectionError(Exception):
    """Raised when delegation depth exceeds the configured maximum."""

    def __init__(self, depth: int, max_depth: int) -> None:
        self.depth = depth
        self.max_depth = max_depth
        super().__init__(
            f"Loop detection: delegation depth {depth} exceeds maximum {max_depth}. "
            f"This likely indicates an infinite agent delegation loop."
        )


@dataclass
class LoopDetectionCapability(AbstractCapability[Any]):
    """Prevent infinite agent loops via depth tracking.

    Wraps ``node_run`` and increments a ``ContextVar`` depth counter on
    each nested call. When depth exceeds ``max_depth``, raises
    ``LoopDetectionError``.

    The depth is tracked via ``contextvars`` rather than an instance
    variable so that it propagates across nested agent delegation runs
    (where ``for_run()`` creates a fresh capability copy that would
    otherwise reset the counter to zero).
    """

    max_depth: int = 10

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            msg = f"max_depth must be >= 1, got {self.max_depth}"
            raise ValueError(msg)

    @property
    def has_wrap_node_run(self) -> bool:
        return True

    async def wrap_node_run(
        self,
        ctx: RunContext[Any],
        *,
        node: AgentNode[Any],
        handler: WrapNodeRunHandler[Any],
    ) -> NodeResult[Any]:
        depth = _delegation_depth.get()
        token = _delegation_depth.set(depth + 1)
        try:
            if depth + 1 > self.max_depth:
                raise LoopDetectionError(depth + 1, self.max_depth)
            return await handler(node)
        finally:
            _delegation_depth.reset(token)

    async def for_run(self, ctx: RunContext[Any]) -> LoopDetectionCapability:
        return LoopDetectionCapability(max_depth=self.max_depth)
