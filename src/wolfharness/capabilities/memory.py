"""Memory capability — persistent memory across turns.

Stores and retrieves key-value memories via ``after_node_run`` (persist)
and injects them via ``get_instructions`` (dynamic callable). Memories
are scoped per session.

The injection uses a callable returned from ``get_instructions`` so that
the memory content is marked ``dynamic=True`` by pydantic-ai — this
avoids mutating the system prompt in-place (which would invalidate the
prefix cache on every turn).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability


if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.capabilities import AgentNode, NodeResult
    from pydantic_ai.capabilities.abstract import AgentInstructions  # type: ignore[attr-defined]


@dataclass
class MemoryCapability(AbstractCapability[Any]):
    """Persist and retrieve memory across conversation turns.

    After each node run, extracts memories from the conversation result
    and stores them. Before each model request, injects relevant memories
    into the system prompt so the model has context from prior turns.

    Memory extraction and injection are delegated to callables so
    different strategies (LLM-based extraction, keyword matching,
    vector search) can be plugged in.

    The store is **shared** across all per-run copies (``for_run()`` does
    not copy the dict) so that memories extracted during a run persist
    into subsequent runs.
    """

    _store: dict[str, str] = field(default_factory=dict, repr=False)
    _extract_fn: Any = field(default=None, repr=False)
    _inject_fn: Any = field(default=None, repr=False)

    @property
    def has_wrap_node_run(self) -> bool:
        return False

    def set_extract_fn(self, fn: Any) -> None:
        self._extract_fn = fn

    def set_inject_fn(self, fn: Any) -> None:
        self._inject_fn = fn

    async def after_node_run(
        self,
        ctx: RunContext[Any],
        *,
        node: AgentNode[Any],
        result: NodeResult[Any],
    ) -> NodeResult[Any]:
        if self._extract_fn is None:
            return result
        new_memories: dict[str, str] = await self._extract_fn(result)
        if new_memories:
            self._store.update(new_memories)
        return result

    def get_instructions(self) -> AgentInstructions[Any] | None:
        """Return a dynamic callable that injects memory content.

        The callable receives ``RunContext`` (with ``.messages``) at run
        time and calls ``_inject_fn(store, messages)`` to produce the
        memory text. This text is marked ``dynamic=True`` by pydantic-ai,
        which is correct — memory content may change between turns.

        Returns ``None`` when there is no store or no inject function.
        """
        if not self._store or self._inject_fn is None:
            return None

        async def _inject_memory(ctx: RunContext[Any]) -> str | None:
            result = self._inject_fn(self._store, ctx.messages)
            if inspect.isawaitable(result):
                result = await result
            return result if result else None

        return _inject_memory

    async def for_run(self, ctx: RunContext[Any]) -> MemoryCapability:
        cap = MemoryCapability()
        # Share the same dict reference so memories persist across runs.
        cap._store = self._store
        cap._extract_fn = self._extract_fn
        cap._inject_fn = self._inject_fn
        return cap
