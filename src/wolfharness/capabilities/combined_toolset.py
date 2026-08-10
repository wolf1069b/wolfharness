r"""CombinedToolsetCapability — composes multiple capabilities into one.

Part of the capability-native migration (M3). This capability composes
a list of :class:`~pydantic_ai.capabilities.AbstractCapability` instances
and presents them as a single capability. It is the capability-layer
replacement for
:class:`~CombinedToolsetCapability`.

Composition rules:

- **``get_toolset()``**: collects non-``None`` toolsets from all children.
  If the list is non-empty, returns a
  :class:`~pydantic_ai.toolsets.CombinedToolset` wrapping them.
  Returns ``None`` if no child provides a toolset.
- **``get_instructions()``**: collects non-``None`` instructions from all
  children, joins them with ``"\n\n"``, returns the result (or ``None``).
- **``on_change()``**: merges the ``on_change()`` async generators from
  all children that provide one, using an ``asyncio.Queue`` bridge.
  Yields :class:`~wolfharness.capabilities.change_event.ChangeEvent`
  from any child. Returns ``None`` if no child provides ``on_change()``.
- **``__aenter__``/``__aexit__``**: enters/exits all children, using
  ``AsyncExitStack`` for safe cleanup even if some children fail.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, AgentToolset, CombinedToolset


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType

    from pydantic_ai.capabilities.abstract import AgentInstructions  # type: ignore[attr-defined]

    from wolfharness.capabilities.change_event import ChangeEvent
    from wolfharness.tools.base import Tool


# ---- Protocols for optional capability methods ----
# AbstractCapability from pydantic-ai does not define ``name``, ``on_change()``,
# or ``__aenter__``/``__aexit__``. These Protocols provide type-safe
# ``isinstance`` checks without ``getattr``/``hasattr``.


@runtime_checkable
class _NamedCapability(Protocol):
    """Protocol for capabilities that expose a ``name`` property."""

    @property
    def name(self) -> str: ...


@runtime_checkable
class _OnChangeCapable(Protocol):
    """Protocol for capabilities that implement ``on_change()``."""

    def on_change(self) -> AsyncIterator[ChangeEvent] | None: ...


@runtime_checkable
class _LifecycleCapable(Protocol):
    """Protocol for capabilities that implement async context manager."""

    async def __aenter__(self) -> Any: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None: ...


@runtime_checkable
class _ToolCollecting(Protocol):
    """Protocol for capabilities that expose ``get_tools()`` (backward compat)."""

    async def get_tools(self) -> Sequence[Any]: ...


class CombinedToolsetCapability(AbstractCapability[AgentDepsT]):
    """Composes multiple capabilities into a single capability.

    This is the capability-layer equivalent of
    :class:`~CombinedToolsetCapability`.
    It collects toolsets, instructions, change events, and lifecycle
    management from all child capabilities and presents them as one.

    Attributes:
        capabilities: The list of child capabilities being composed.
    """

    def __init__(
        self,
        capabilities: list[AbstractCapability[AgentDepsT]],
        *,
        name: str | None = None,
    ) -> None:
        """Initialize the combined capability.

        Args:
            capabilities: List of capabilities to compose. The order
                determines toolset and instruction ordering.
            name: Optional name for this capability. Defaults to
                ``"combined:"`` followed by the names of all children
                joined with ``,``.
        """
        self._capabilities: list[AbstractCapability[AgentDepsT]] = list(capabilities)
        self._name = name or self._derive_name()
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._background_tasks: set[asyncio.Task[None]] = set()

    def _derive_name(self) -> str:
        """Derive a name from child capability names."""
        parts: list[str] = []
        for cap in self._capabilities:
            if isinstance(cap, _NamedCapability):
                parts.append(cap.name)
            else:
                parts.append(type(cap).__name__)
        return f"combined:{','.join(parts)}" if parts else "combined:empty"

    # ---- Properties ----

    @property
    def name(self) -> str:
        """Return the capability name."""
        return self._name

    @property
    def capabilities(self) -> list[AbstractCapability[AgentDepsT]]:
        """Return a shallow copy of the child capabilities list."""
        return list(self._capabilities)

    # ---- AbstractCapability overrides ----

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Return a unified toolset combining all children's toolsets.

        Collects non-``None`` toolsets from all children. If any are
        found, wraps them in a
        :class:`~pydantic_ai.toolsets.CombinedToolset`. Returns ``None``
        if no child provides a toolset.
        """
        toolsets: list[AbstractToolset[AgentDepsT]] = []
        for cap in self._capabilities:
            toolset = cap.get_toolset()
            if toolset is None:
                continue
            if isinstance(toolset, AbstractToolset):
                toolsets.append(toolset)
            else:
                # ToolsetFunc (async callable) — AgentToolset accepts
                # callable toolsets directly in pydantic-ai v2.
                toolsets.append(toolset)  # type: ignore[arg-type]
        if not toolsets:
            return None
        return CombinedToolset(toolsets=toolsets)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        r"""Return concatenated instructions from all children.

        Collects non-``None`` instructions from all children. Handles
        ``str``, callables, and sequences (lists/tuples) — previous
        implementation silently dropped non-``str`` returns, which
        prevented children from returning dynamic instruction callables.

        When all parts are strings, joins them with ``"\n\n"`` and returns
        a single ``str`` (backward compat). When any part is a callable,
        returns the full list so pydantic-ai can mark callables as
        ``dynamic=True``.
        """
        parts: list[str | Any] = []
        for cap in self._capabilities:
            instr = cap.get_instructions()
            if instr is None:
                continue
            if isinstance(instr, str) or callable(instr):
                parts.append(instr)
            elif isinstance(instr, Sequence):
                parts.extend(instr)
        if not parts:
            return None
        # If all parts are strings, join for backward compat.
        if all(isinstance(p, str) for p in parts):
            return "\n\n".join(parts)
        return parts

    # ---- Backward compat: tool collection ----

    async def get_tools(self) -> Sequence[Tool[object]]:
        """Collect tools from all children that have ``get_tools()``.

        This is a backward-compat method for code that still calls
        ``get_tools()`` on the combined capability. It iterates
        children that satisfy the :class:`_ToolCollecting` Protocol and
        collects their tools into a flat list.
        """
        import asyncio

        all_tools: list[Tool[object]] = []
        coros: list[Any] = []
        caps_with_tools: list[AbstractCapability[AgentDepsT]] = []
        for cap in self._capabilities:
            if isinstance(cap, _ToolCollecting):
                coros.append(cap.get_tools())
                caps_with_tools.append(cap)
        if not coros:
            return all_tools
        results = await asyncio.gather(*coros, return_exceptions=True)
        for _cap, result in zip(caps_with_tools, results, strict=False):
            if isinstance(result, BaseException):
                continue
            all_tools.extend(result)
        return all_tools

    # ---- Change signal bridging ----

    def on_change(self) -> AsyncIterator[ChangeEvent] | None:
        """Merge ``on_change()`` streams from all children.

        Collects ``on_change()`` async generators from all children that
        provide one (via the :class:`_OnChangeCapable` Protocol). If no
        child provides ``on_change()``, returns ``None``.

        The merged generator uses an ``asyncio.Queue`` to bridge multiple
        async generators: each child's generator is consumed by a
        background task that puts events into the queue. The merged
        generator yields events from the queue as they arrive from any
        child.
        """
        generators: list[AsyncIterator[ChangeEvent]] = []
        for cap in self._capabilities:
            if isinstance(cap, _OnChangeCapable):
                gen = cap.on_change()
                if gen is not None:
                    generators.append(gen)
        if not generators:
            return None
        return self._merge_generators(generators)

    async def _merge_generators(
        self,
        generators: list[AsyncIterator[ChangeEvent]],
    ) -> AsyncIterator[ChangeEvent]:
        """Merge multiple async generators into one via asyncio.Queue.

        Spawns a consumer task per generator. Each consumer iterates its
        generator and puts events into a shared queue. The merged
        generator yields from the queue. All tasks are cancelled and
        cleaned up when the consumer stops.
        """
        queue: asyncio.Queue[ChangeEvent | None] = asyncio.Queue()
        tasks: list[asyncio.Task[None]] = []
        num_generators = len(generators)

        async def _consume(gen: AsyncIterator[ChangeEvent]) -> None:
            try:
                async for event in gen:
                    await queue.put(event)
            finally:
                await queue.put(None)

        for gen in generators:
            task = asyncio.create_task(_consume(gen))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            tasks.append(task)

        finished_count = 0
        try:
            while finished_count < num_generators:
                event = await queue.get()
                if event is None:
                    finished_count += 1
                    continue
                yield event
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    # ---- Lifecycle delegation ----

    async def __aenter__(self) -> CombinedToolsetCapability[AgentDepsT]:
        """Enter async context for all children.

        Uses :class:`contextlib.AsyncExitStack` to ensure that if a later
        child fails to enter, already-entered children are properly
        exited.
        """
        self._exit_stack = AsyncExitStack()
        for cap in self._capabilities:
            if isinstance(cap, _LifecycleCapable):
                await self._exit_stack.enter_async_context(cap)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context for all children.

        Delegates to the :class:`AsyncExitStack` which exits children in
        reverse order (LIFO), propagating the first exception.
        """
        stack = self._exit_stack
        self._exit_stack = AsyncExitStack()
        await stack.aclose()
