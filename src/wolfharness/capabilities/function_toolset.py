"""FunctionToolsetCapability — wraps a static list of Tool instances as a capability.

This capability holds tools directly and returns a concrete
``FunctionToolset`` eagerly — no async ``ToolsetFunc`` needed.
It also provides ``add_tool()``, ``register_tool()``, ``register_worker()``,
and ``tool()`` decorator methods for backward compatibility with code that
previously used the old static provider pattern.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, overload

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset, FunctionToolset


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from types import TracebackType

    from wolfharness import Agent, MessageNode
    from wolfharness.capabilities.change_event import ChangeEvent
    from wolfharness.common_types import ToolSource, ToolType
    from wolfharness.prompts.prompts import BasePrompt
    from wolfharness.tools.base import Tool


class FunctionToolsetCapability(AbstractCapability[AgentDepsT]):
    """Wraps a static list of :class:`~wolfharness.tools.base.Tool` instances.

    Provides:
    - ``get_toolset()``: eagerly builds a ``FunctionToolset`` from the
      wrapped tools via :func:`~wolfharness.tools.tool_wrapping.wrap_tool_for_pydantic_ai`.
      Returns ``None`` if the tool list is empty.
    - ``get_instructions()``: returns the optional instructions string, or
      ``None`` if none were provided.
    - ``on_change()``: returns ``None`` — static tools never change.
    - ``__aenter__``/``__aexit__``: no-ops (no lifecycle to manage).

    Attributes:
        name: Human-readable name for this capability.
    """

    def __init__(
        self,
        tools: Sequence[Tool[Any]] | None = None,
        *,
        name: str = "function_tools",
        instructions: str | None = None,
        prompts: Sequence[BasePrompt] | None = None,
        owner: str | None = None,
    ) -> None:
        """Initialize the capability.

        Args:
            tools: The list of ``Tool`` instances to wrap.
            name: Human-readable name for this capability.
            instructions: Optional instructions string for the system prompt.
            prompts: Optional list of prompts (backward compat).
            owner: Optional owner string (backward compat).
        """
        self._tools: list[Tool[Any]] = list(tools) if tools else []
        self._name = name
        self._instructions = instructions
        self._prompts: list[BasePrompt] = list(prompts) if prompts else []
        self.owner: str | None = owner

    # ---- Backward compat properties ----

    @property
    def kind(self) -> str:
        """Return the provider kind (backward compat)."""
        return "tools"

    async def get_tools(self) -> Sequence[Tool[Any]]:
        """Return the wrapped tool list (async, for backward compat)."""
        return self._tools

    async def get_prompts(self) -> list[BasePrompt]:
        """Return the prompts list (backward compat)."""
        return self._prompts

    def add_tool(self, tool: ToolType) -> None:
        """Add a tool to this capability."""
        from wolfharness.tools.base import Tool as ToolClass

        match tool:
            case ToolClass():
                self._tools.append(tool)
            case Callable() | str():
                self._tools.append(ToolClass.from_callable(tool))

    def create_tool(
        self,
        fn: Callable[..., Any],
        read_only: bool | None = None,
        destructive: bool | None = None,
        idempotent: bool | None = None,
        open_world: bool | None = None,
        requires_confirmation: bool = False,
        metadata: dict[str, Any] | None = None,
        category: Any | None = None,
        name_override: str | None = None,
        description_override: str | None = None,
        schema_override: Any | None = None,
        prepare: Any | None = None,
    ) -> Tool[Any]:
        """Create a tool from a function and add it to this capability."""
        from wolfharness.tools.base import Tool as ToolClass
        from wolfharness_config.tools import ToolHints

        tool = ToolClass.from_callable(
            fn=fn,
            category=category,
            source=self._name,
            requires_confirmation=requires_confirmation,
            metadata=metadata,
            name_override=name_override,
            description_override=description_override,
            schema_override=schema_override,
            prepare=prepare,
            hints=ToolHints(
                read_only=read_only,
                destructive=destructive,
                idempotent=idempotent,
                open_world=open_world,
            ),
        )
        self._tools.append(tool)
        return tool

    def remove_tool(self, name: str) -> bool:
        """Remove a tool by name."""
        for i, tool in enumerate(self._tools):
            if tool.name == name:
                self._tools.pop(i)
                return True
        return False

    def register_tool(
        self,
        tool: ToolType,
        *,
        name_override: str | None = None,
        description_override: str | None = None,
        enabled: bool = True,
        source: ToolSource = "dynamic",
        requires_confirmation: bool = False,
        metadata: dict[str, str] | None = None,
    ) -> Tool[Any]:
        """Register a new tool with custom settings."""
        from wolfharness.tools.base import Tool as ToolClass

        match tool:
            case ToolClass():
                tool.description = description_override or tool.description
                tool.name = name_override or tool.name
                tool.source = source
                tool.metadata = tool.metadata | (metadata or {})
                tool.enabled = enabled
            case _:
                tool = ToolClass.from_callable(
                    tool,
                    enabled=enabled,
                    source=source,
                    name_override=name_override,
                    description_override=description_override,
                    requires_confirmation=requires_confirmation,
                    metadata=metadata or {},
                )
        self.add_tool(tool)
        return tool

    def register_worker(
        self,
        worker: MessageNode[Any, Any],
        *,
        name: str | None = None,
        reset_history_on_run: bool = True,
        pass_message_history: bool = False,
        parent: Agent[Any, Any] | None = None,
    ) -> Tool[Any]:
        """Register an agent as a worker tool."""
        from wolfharness import Agent, BaseTeam
        from wolfharness.agents.base_agent import BaseAgent

        match worker:
            case Agent():
                tool = worker.to_tool(
                    parent=parent,
                    name=name,
                    reset_history_on_run=reset_history_on_run,
                    pass_message_history=pass_message_history,
                )
            case BaseTeam() | BaseAgent():
                tool = worker.to_tool(name=name, description=worker.description)
            case _:
                raise ValueError(f"Unsupported worker type: {type(worker)}")
        self.add_tool(tool)
        return tool

    @overload
    def tool(self, func: Callable[..., Any]) -> Callable[..., Any]: ...

    @overload
    def tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        enabled: bool = True,
        requires_confirmation: bool = False,
        metadata: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

    def tool(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        enabled: bool = True,
        requires_confirmation: bool = False,
        metadata: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a function as a tool."""
        from wolfharness.tools.base import Tool

        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            tool_obj = Tool.from_callable(
                f,
                name_override=name,
                description_override=description,
                enabled=enabled,
                requires_confirmation=requires_confirmation,
                metadata=metadata or {},
                **kwargs,
            )
            self.add_tool(tool_obj)
            return f

        return decorator if func is None else decorator(func)

    # ---- Properties ----

    @property
    def name(self) -> str:
        """Return the capability name."""
        return self._name

    @property
    def tools(self) -> list[Tool[Any]]:
        """Return the wrapped tool list."""
        return list(self._tools)

    # ---- AbstractCapability overrides ----

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Return a ``FunctionToolset`` built from the wrapped tools.

        Each tool is converted to a pydantic-ai compatible callable via
        :func:`~wolfharness.tools.tool_wrapping.wrap_tool_for_pydantic_ai`,
        then assembled into a ``FunctionToolset``.

        Returns ``None`` if the tool list is empty.
        """
        if not self._tools:
            return None

        from wolfharness.tools.tool_wrapping import wrap_tool_for_pydantic_ai

        pa_tools = [wrap_tool_for_pydantic_ai(tool) for tool in self._tools]
        return FunctionToolset(pa_tools, id=self._name)

    def get_instructions(self) -> str | None:
        """Return the instructions string, or ``None`` if none were provided."""
        return self._instructions

    # ---- Change signal ----

    def on_change(self) -> AsyncIterator[ChangeEvent] | None:
        """Return ``None`` — static tools never change.

        Callers that expect an async iterator should check for ``None``
        before iterating.
        """
        return None

    # ---- Lifecycle (no-ops) ----

    async def __aenter__(self) -> FunctionToolsetCapability[AgentDepsT]:
        """Enter async context. No-op — static tools need no lifecycle."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context. No-op — static tools need no lifecycle."""
