"""FilteredToolsetCapability — wraps another capability with a tool filter.

Part of the capability-native migration (M3). This capability wraps another
:class:`~pydantic_ai.capabilities.AbstractCapability` and applies a
:class:`~pydantic_ai.toolsets.FilteredToolset` to the toolset it produces,
allowing tool-level access control without modifying the wrapped capability.

This replaces :class:`~FilteredToolsetCapability`
in the capability-native architecture.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
import inspect
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, AgentToolset, FilteredToolset


if TYPE_CHECKING:
    from types import TracebackType

    from wolfharness.capabilities.change_event import ChangeEvent


ToolFilterFunc = Callable[[RunContext[AgentDepsT], ToolDefinition], bool | Awaitable[bool]]
"""Filter function signature for tool acceptance.

Return ``True`` to include the tool, ``False`` to exclude it.
Both sync and async variants are accepted (``FilteredToolset`` handles both).
"""


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
        exc_tb: TracebackType | None,
    ) -> None: ...


class FilteredToolsetCapability(AbstractCapability[AgentDepsT]):
    """Wraps another :class:`AbstractCapability` with a tool filter.

    Delegates ``get_instructions()``, ``on_change()``, and lifecycle
    methods (``__aenter__``/``__aexit__``) to the wrapped capability.
    The ``get_toolset()`` method wraps the wrapped capability's toolset
    in a :class:`FilteredToolset` that applies the provided filter function.

    If the wrapped capability does not implement ``on_change()`` (i.e.
    it is not an wolfharness capability), ``on_change()`` returns ``None``.

    Attributes:
        wrapped: The inner ``AbstractCapability`` being filtered.
        filter_func: Async or sync function deciding tool inclusion.
    """

    def __init__(
        self,
        wrapped: AbstractCapability[AgentDepsT],
        filter_func: ToolFilterFunc[AgentDepsT],
        *,
        name: str | None = None,
    ) -> None:
        """Initialize the filtered toolset capability.

        Args:
            wrapped: The capability to wrap. Its toolset will be filtered.
            filter_func: Function called per tool to decide inclusion.
                Receives the run context and tool definition; returns
                ``True`` to include, ``False`` to exclude.
            name: Optional name override. Defaults to the wrapped
                capability's name (if available) or class name.
        """
        self._wrapped = wrapped
        self._filter_func = filter_func
        self._name = name or _derive_name(wrapped)

    # ---- Properties ----

    @property
    def name(self) -> str:
        """Return the capability name."""
        return self._name

    @property
    def wrapped(self) -> AbstractCapability[AgentDepsT]:
        """Return the wrapped capability."""
        return self._wrapped

    @property
    def filter_func(self) -> ToolFilterFunc[AgentDepsT]:
        """Return the tool filter function."""
        return self._filter_func

    # ---- AbstractCapability overrides ----

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Return a filtered version of the wrapped capability's toolset.

        If the wrapped capability returns ``None`` (no toolset), this
        also returns ``None``. Otherwise, the toolset is wrapped in a
        :class:`FilteredToolset` with the configured filter function.

        When the wrapped capability returns a ``ToolsetFunc`` (async
        callable), this method returns a new ``ToolsetFunc`` that:
        1. Calls the inner function to get the toolset.
        2. Wraps the result in ``FilteredToolset``.
        3. Returns ``None`` if the inner function returned ``None``.
        """
        inner = self._wrapped.get_toolset()
        if inner is None:
            return None

        # If it's a callable (ToolsetFunc), wrap lazily.
        if callable(inner) and not isinstance(inner, AbstractToolset):
            filter_func = self._filter_func

            async def _filtered_build(
                ctx: RunContext[AgentDepsT],
            ) -> AbstractToolset[AgentDepsT] | None:
                result: Any = inner(ctx)
                if inspect.isawaitable(result):
                    result = await result
                if result is None:
                    return None
                return FilteredToolset(wrapped=result, filter_func=filter_func)

            return _filtered_build

        # Concrete AbstractToolset — wrap directly.
        return FilteredToolset(wrapped=inner, filter_func=self._filter_func)

    def get_instructions(self) -> Any:
        """Delegate instructions to the wrapped capability."""
        return self._wrapped.get_instructions()

    # ---- Change signal bridging ----

    def on_change(self) -> AsyncIterator[ChangeEvent] | None:
        """Delegate ``on_change()`` to the wrapped capability.

        If the wrapped capability implements ``on_change()`` (i.e. it
        is an wolfharness capability), its return value is forwarded.
        Otherwise, returns ``None`` — static capabilities never change.
        """
        if isinstance(self._wrapped, _OnChangeCapable):
            return self._wrapped.on_change()
        return None

    # ---- Lifecycle delegation ----

    async def __aenter__(self) -> FilteredToolsetCapability[AgentDepsT]:
        """Enter async context, delegating to the wrapped capability.

        If the wrapped capability is an async context manager, its
        ``__aenter__`` is called. Otherwise, this is a no-op.
        """
        if isinstance(self._wrapped, _LifecycleCapable):
            await self._wrapped.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context, delegating to the wrapped capability.

        If the wrapped capability is an async context manager, its
        ``__aexit__`` is called. Otherwise, this is a no-op.
        """
        if isinstance(self._wrapped, _LifecycleCapable):
            await self._wrapped.__aexit__(exc_type, exc_val, exc_tb)


def _derive_name(capability: AbstractCapability[AgentDepsT]) -> str:
    """Derive a name string from a capability for naming purposes.

    Uses the capability's ``name`` property if it conforms to
    :class:`_NamedCapability`, otherwise falls back to the class name.
    """
    if isinstance(capability, _NamedCapability):
        return capability.name
    return type(capability).__name__
