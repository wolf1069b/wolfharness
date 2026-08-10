"""Base class for all agent types."""

from __future__ import annotations

from abc import abstractmethod
import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, ClassVar, Literal, assert_never, overload
import warnings

from anyenv import MultiEventHandler, method_spawner
from anyenv.signals import Signal
import anyio
from upathtools.filesystems import IsolatedMemoryFileSystem

from wolfharness.agents.context import AgentContext, AgentRunContext
from wolfharness.agents.events import (
    RunErrorEvent,
    StreamCompleteEvent,
    resolve_event_handlers,
)
from wolfharness.agents.modes import ModeInfo
from wolfharness.capabilities.function_toolset import FunctionToolsetCapability
from wolfharness.common_types import IndividualEventHandler
from wolfharness.log import get_logger
from wolfharness.messaging import ChatMessage, MessageHistory, MessageNode
from wolfharness.observability.spans import safe_span
from wolfharness.prompts.convert import convert_prompts
from wolfharness.utils.inspection import call_with_context
from wolfharness.utils.time_utils import get_now


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from contextvars import Token
    from datetime import datetime

    from evented_config import EventConfig
    from exxec import ExecutionEnvironment
    from fsspec import AbstractFileSystem
    from pydantic_ai import UserContent
    from pydantic_ai.messages import ModelMessage
    from slashed import BaseCommand, CommandStore
    from tokonomics.model_discovery.model_info import ModelInfo
    from upathtools.filesystems import OverlayFileSystem

    from acp.schema import AvailableCommandsUpdate
    from wolfharness.agents.events import (
        CommandCompleteEvent,
        CommandOutputEvent,
        RichAgentStreamEvent,
        StreamWithCommandsEvent,
        ToastInfo,
    )
    from wolfharness.agents.modes import ConfigOptionChanged, ModeCategory, ModeCategoryId
    from wolfharness.agents.native_agent import Agent
    from wolfharness.common_types import (
        AgentName,
        AnyEventHandlerType,
        MCPServerStatus,
        ProcessorCallback,
        PromptCompatible,
        StrPath,
    )
    from wolfharness.delegation import AgentPool, BaseTeam
    from wolfharness.hooks import AgentHooks
    from wolfharness.messaging import ChatMessage
    from wolfharness.orchestrator.core import EventBus, SessionPool, SessionState
    from wolfharness.orchestrator.run import RunHandle
    from wolfharness.orchestrator.turn import Turn
    from wolfharness.sessions import SessionData
    from wolfharness.talk.stats import MessageStats
    from wolfharness.ui.base import InputProvider
    from wolfharness_config.lifecycle import LifecycleConfig
    from wolfharness_config.mcp_server import MCPServerConfig

    # Union type for state updates emitted via state_updated signal
    type StateUpdate = (
        ModeInfo | ModelInfo | AvailableCommandsUpdate | ConfigOptionChanged | ToastInfo
    )


# ContextVar for per-execution isolation of _current_run_ctx (RFC-0021 compliance)
_current_run_ctx_var: ContextVar[AgentRunContext | None] = ContextVar(
    "_current_run_ctx_var",
    default=None,
)

_in_turn_context: ContextVar[bool] = ContextVar(
    "_in_turn_context",
    default=False,
)

logger = get_logger(__name__)

# Literal type for all agent types
type AgentTypeLiteral = Literal["native", "acp"]


_SLASH_PATTERN: re.Pattern[str] = re.compile(r"^/([\w-]+)(?:\s+(.*))?$")


def _parse_slash_command(command_text: str) -> tuple[str, str] | None:
    """Parse slash command into name and args.

    Args:
        command_text: Full command text

    Returns:
        Tuple of (cmd_name, args) or None if invalid
    """
    if match := _SLASH_PATTERN.match(command_text.strip()):
        cmd_name = match.group(1)
        args = match.group(2) or ""
        return cmd_name, args.strip()
    return None


def _is_slash_command(text: str) -> bool:
    """Check if text starts with a slash command."""
    return bool(_SLASH_PATTERN.match(text.strip()))


class BaseAgent[TDeps = None, TResult = str](MessageNode[TDeps, TResult]):
    """Base class for Agent and ACPAgent.

    Provides shared infrastructure:
    - _builtin_provider: FunctionToolsetCapability for built-in tools
    - _worker_provider: FunctionToolsetCapability for worker tools
    - _external_capabilities: list of external capabilities
    - conversation: MessageHistory for conversation state
    - event_handler: MultiEventHandler for event distribution
    - _event_queue: Queue for streaming events
    - _input_provider: Provider for user input/confirmations
      (deprecated: use SessionState.input_provider)
    - env: ExecutionEnvironment for running code/commands
    - context property: Returns NodeContext for the agent

    Signals:
    """

    # Abstract class variable - subclasses must define this
    AGENT_TYPE: ClassVar[AgentTypeLiteral]

    @dataclass(frozen=True)
    class AgentReset:
        """Emitted when agent is reset."""

        agent_name: AgentName
        timestamp: datetime = field(default_factory=get_now)

    @dataclass(frozen=True)
    class InterruptEvent:
        """Emitted when agent is interrupted."""

        agent_name: AgentName
        timestamp: datetime = field(default_factory=get_now)

    agent_reset = Signal[AgentReset]()
    state_updated: Signal[StateUpdate] = Signal()
    # Signal emitted when agent is interrupted
    interrupted: Signal[InterruptEvent] = Signal()

    def _session_initial_prompt_for_title(
        self,
        session_id: str,
        initial_prompt: str | None,
    ) -> str | None:
        """Return the prompt used for title generation for this session."""
        if initial_prompt is None:
            return None
        ctx = self.host_context
        if ctx is None or ctx.session_pool is None:
            return initial_prompt

        session_controller = ctx.session_pool.sessions
        get_session = getattr(session_controller, "get_session", None)
        if not callable(get_session):
            return initial_prompt

        session_state = get_session(session_id)
        metadata = getattr(session_state, "metadata", None)
        if isinstance(metadata, dict) and metadata.get("generate_title") is False:
            return None
        return initial_prompt

    def __init__(
        self,
        *,
        name: str = "agent",
        deps_type: type[TDeps] | None = None,
        description: str | None = None,
        display_name: str | None = None,
        mcp_servers: Sequence[str | MCPServerConfig] | None = None,
        agent_pool: AgentPool[Any] | None = None,
        enable_logging: bool = True,
        event_configs: Sequence[EventConfig] | None = None,
        # New shared parameters
        env: ExecutionEnvironment | StrPath | None = None,
        input_provider: InputProvider | None = None,
        output_type: type[TResult] = str,  # type: ignore[assignment]  # ty: ignore[invalid-parameter-default]
        event_handlers: Sequence[AnyEventHandlerType] | None = None,
        commands: Sequence[BaseCommand] | None = None,
        hooks: AgentHooks | None = None,
        lifecycle_config: LifecycleConfig | None = None,
    ) -> None:
        """Initialize base agent with shared infrastructure.

        Args:
            name: Agent name
            deps_type: Type of dependencies to use
            description: Agent description
            display_name: Human-readable display name
            mcp_servers: MCP server configurations
            agent_pool: Agent pool for coordination
            enable_logging: Whether to enable database logging
            event_configs: Event trigger configurations
            env: Execution environment, or a path (str/PathLike) to use as cwd
                for a LocalExecutionEnvironment
            input_provider: Provider for user input and confirmations
            output_type: Output type for this agent
            event_handlers: Event handlers for this agent
            commands: Slash commands to register with this agent
            hooks: Agent hooks for intercepting agent behavior at run and tool events
            lifecycle_config: Optional lifecycle configuration for the RunLoop.
                When None, all defaults (in-memory) are used.
        """
        from exxec import ExecutionEnvironment, LocalExecutionEnvironment
        from slashed import CommandStore

        from wolfharness.agents.staged_content import StagedContent
        from wolfharness_commands import get_commands

        super().__init__(
            name=name,
            description=description,
            display_name=display_name,
            mcp_servers=mcp_servers,
            agent_pool=agent_pool,
            enable_logging=enable_logging,
            event_configs=event_configs,
        )
        self._infinite = False
        self.deps_type = deps_type  # or type(None)
        self._background_task: asyncio.Task[ChatMessage[Any]] | None = None
        storage = agent_pool.storage if agent_pool else None
        self.conversation = MessageHistory(storage=storage)
        match env:
            case ExecutionEnvironment():
                self.env = env
            case str() | os.PathLike() | None:
                self.env = LocalExecutionEnvironment(cwd=str(env) if env is not None else None)
            case _ as unreachable:
                assert_never(unreachable)
        self._input_provider = input_provider
        if input_provider is not None:
            warnings.warn(
                "BaseAgent._input_provider is deprecated. Use SessionState.input_provider instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._output_type: type[TResult] = output_type
        # Direct capability management (replaces deprecated ToolManager)
        self._builtin_provider = FunctionToolsetCapability(name="builtin")
        self._worker_provider = FunctionToolsetCapability(name="workers")
        self._external_capabilities: list[Any] = []
        self._session_capabilities: list[Any] = []
        self._tool_mode: Any | None = None
        handlers = resolve_event_handlers(event_handlers)
        self.event_handler: MultiEventHandler[IndividualEventHandler] = MultiEventHandler(handlers)  # ty: ignore[invalid-assignment]
        self.hooks = hooks
        self._cancelled = False
        # _background_run_ctx is used only for the background task's internal state.
        # It is intentionally NOT used as a fallback in get_active_run_context().
        self._background_run_ctx: AgentRunContext | None = None
        # _run_context is set during standalone (pool-less) runs so that
        # get_active_run_context() and is_turn_active() work cross-task.
        self._run_context: AgentRunContext | None = None
        # Deferred initialization support - subclasses set True in __aenter__,
        # override ensure_initialized() to do actual connection
        self._connect_pending: bool = False
        self._command_store = CommandStore(commands=[*get_commands(), *(commands or [])])
        # Initialize store (registers builtin help/exit commands)
        self._command_store._initialize_sync()
        # Internal filesystem for tool/session state (can get written to via AgentContext)
        self._internal_fs = IsolatedMemoryFileSystem()
        self.staged_content = StagedContent()
        self.metadata: dict[str, Any] = {}
        self._lifecycle_config: LifecycleConfig | None = lifecycle_config
        self._lifecycle_dimensions: tuple[Any, Any, Any, Any, Any] | None = None

    @property
    def _current_run_ctx(self) -> AgentRunContext | None:
        """Get current run context (using ContextVar for concurrency safety)."""
        return _current_run_ctx_var.get()

    @property
    def lifecycle_config(self) -> LifecycleConfig | None:
        """Lifecycle configuration for this agent's RunLoop.

        Returns:
            The LifecycleConfig if set, or ``None`` for all-defaults.
        """
        return self._lifecycle_config

    @property
    def lifecycle_dimensions(
        self,
    ) -> tuple[Any, Any, Any, Any, Any] | None:
        """Pre-created lifecycle dimensions for the current run, if any.

        Returns:
            Tuple of ``(trigger_source, journal, snapshot_store,
            comm_channel, event_transport)`` if dimensions were created
            from ``lifecycle_config``, or ``None`` if defaults should be
            used.
        """
        return self._lifecycle_dimensions

    @property
    def session_id(self) -> str | None:
        """Current conversation session bound to this agent, if any."""
        return getattr(self._events, "session_id", None)

    @session_id.setter
    def session_id(self, value: str | None) -> None:
        self._events.session_id = value

    def __repr__(self) -> str:
        typ = self.__class__.__name__
        desc = f", {self.description!r}" if self.description else ""
        return f"{typ}({self.name!r}, model={self.model_name!r}{desc})"

    def set_session_context(self, session_id: str, parent_session_id: str | None = None) -> None:
        """Set session context for the agent and its event manager.

        Args:
            session_id: The session ID to set
            parent_session_id: Optional parent session ID
        """
        self._events.session_id = session_id
        self._events.parent_session_id = parent_session_id

    async def __prompt__(self) -> str:
        typ = self.__class__.__name__
        model = self.model_name or "default"
        parts = [f"Agent: {self.name}", f"Type: {typ}", f"Model: {model}"]
        if self.description:
            parts.append(f"Description: {self.description}")
        parts.extend([await self._tools_prompt(), self.conversation.__prompt__()])
        return "\n".join(parts)

    @overload
    def __and__(  # if other doesnt define deps, we take the agents one
        self, other: ProcessorCallback[Any] | BaseTeam[TDeps, Any] | Agent[TDeps, Any]
    ) -> BaseTeam[TDeps, Any]: ...

    @overload
    def __and__(  # otherwise, we dont know and deps is Any
        self, other: ProcessorCallback[Any] | BaseTeam[Any, Any] | Agent[Any, Any]
    ) -> BaseTeam[Any, Any]: ...

    def __and__(self, other: MessageNode[Any, Any] | ProcessorCallback[Any]) -> BaseTeam[Any, Any]:
        """Create parallel team using & operator.

        Example:
            group = analyzer & planner & executor  # Create group of 3
            group = analyzer & existing_group  # Add to existing group
        """
        from wolfharness.agents.native_agent import Agent
        from wolfharness.delegation.base_team import BaseTeam

        match other:
            case BaseTeam():
                return BaseTeam([self, *other.nodes], mode="parallel")
            case Callable():
                agent_2 = Agent.from_callback(other, agent_pool=self._agent_pool)  # ty: ignore[no-matching-overload]
                return BaseTeam([self, agent_2], mode="parallel")
            case MessageNode():
                return BaseTeam([self, other], mode="parallel")
            case _:
                raise ValueError(f"Invalid agent type: {type(other)}")

    @overload
    def __or__(self, other: MessageNode[TDeps, Any]) -> BaseTeam[TDeps, Any]: ...

    @overload
    def __or__[TOtherDeps](self, other: MessageNode[TOtherDeps, Any]) -> BaseTeam[Any, Any]: ...

    @overload
    def __or__(self, other: ProcessorCallback[Any]) -> BaseTeam[Any, Any]: ...

    def __or__(self, other: MessageNode[Any, Any] | ProcessorCallback[Any]) -> BaseTeam[Any, Any]:
        # Create new execution with sequential mode (for piping)
        from wolfharness.agents.native_agent import Agent
        from wolfharness.delegation.base_team import BaseTeam

        if callable(other):
            other = Agent.from_callback(other, agent_pool=self._agent_pool)

        return BaseTeam([self, other], mode="sequential")

    async def update_state(self, config_id: str, value_id: str) -> None:
        from wolfharness.agents.modes import ConfigOptionChanged

        self.log.info("Config option changed", config_id=config_id, mode=value_id)
        change = ConfigOptionChanged(config_id=config_id, value_id=value_id)
        await self.state_updated.emit(change)

    @property
    def command_store(self) -> CommandStore:
        """Get the command store for slash commands."""
        return self._command_store

    @property
    def internal_fs(self) -> IsolatedMemoryFileSystem:
        """Get the internal filesystem for tool/session state.

        Tools can use this to store logs, history, temporary files, etc.
        Access via AgentContext.fs in tool implementations.

        Returns:
            In-memory filesystem scoped to this agent
        """
        return self._internal_fs

    @property
    def overlay_fs(self) -> OverlayFileSystem:
        """Get unified filesystem view combining agent storage and VFS resources.

        Provides a layered filesystem where:
        - Writes go to the agent's internal filesystem (upper layer)
        - Reads fall through to VFS resources if not found locally

        Returns:
            OverlayFileSystem combining internal_fs and pool's VFS registry
        """
        from upathtools.filesystems import OverlayFileSystem

        # Build layers: internal_fs on top, VFS resources below
        layers: list[AbstractFileSystem] = [self._internal_fs]
        ctx = self.host_context
        if ctx is not None and not ctx.vfs_registry.is_empty:
            layers.append(ctx.vfs_registry.get_fs())
        return OverlayFileSystem(filesystems=layers)

    async def reset(self) -> None:
        """Reset agent state (conversation history and tool states)."""
        await self.conversation.clear()
        await self._reset_tool_states()
        event = self.AgentReset(agent_name=self.name)
        await self.agent_reset.emit(event)

    # ---- Capability management (replaces deprecated ToolManager) ----

    @property
    def _all_capabilities(self) -> list[Any]:
        """Return all capabilities: external + session + worker + builtin.

        When ``_tool_mode == "codemode"``, callers should handle wrapping
        in :class:`CodeModeCapability` in an async context (e.g.,
        ``get_agentlet()``) since tool collection is async.
        """
        return [
            *self._external_capabilities,
            *self._session_capabilities,
            self._worker_provider,
            self._builtin_provider,
        ]

    def _add_capability(self, capability: Any, owner: str | None = None) -> None:
        """Add an external capability.

        Args:
            capability: AbstractCapability instance to add
            owner: Optional owner string (set on capability if supported)
        """
        if owner is not None:
            # FunctionToolsetCapability supports .owner
            with suppress(AttributeError):
                capability.owner = owner
        self._external_capabilities.append(capability)

    async def _get_all_tools(self) -> list[Any]:
        """Collect tools from all capabilities concurrently."""
        from typing import Protocol, runtime_checkable

        @runtime_checkable
        class _ToolProviding(Protocol):
            async def get_tools(self) -> Sequence[Any]: ...

        import asyncio

        caps = self._all_capabilities
        coros: list[Any] = []
        caps_with_tools: list[Any] = []
        for cap in caps:
            if isinstance(cap, _ToolProviding):
                coros.append(cap.get_tools())
                caps_with_tools.append(cap)
        if not coros:
            return []
        results = await asyncio.gather(*coros, return_exceptions=True)
        tools_map: dict[str, Any] = {}
        for cap, result in zip(caps_with_tools, results, strict=False):
            if isinstance(result, BaseException):
                logger.warning(
                    "Failed to get tools from capability",
                    capability=type(cap).__name__,
                    error=str(result),
                )
                continue
            for tool in result:
                tools_map[tool.name] = tool
        return list(tools_map.values())

    async def _tools_prompt(self) -> str:
        """Generate a summary of available tools for prompts."""
        all_tools = await self._get_all_tools()
        enabled_tools = [t.name for t in all_tools if t.enabled]
        if not enabled_tools:
            return "No tools available"
        return f"Available tools: {', '.join(enabled_tools)}"

    async def _reset_tool_states(self) -> None:
        """Reset all tools to their default enabled states."""
        for tool in await self._get_all_tools():
            tool.enabled = True

    async def _get_mcp_server_info(self) -> dict[str, MCPServerStatus]:
        """Get MCP server status by merging agent-scoped and pool-level managers.

        ``self.mcp`` is always set on ``MessageNode`` (either shared with
        the pool or a dedicated agent-scoped ``MCPManager``). When
        ``self.host_context`` is available AND ``self.mcp is not
        self.host_context.mcp`` (identity check — agent has its own
        servers), pool-level servers from ``host_context.mcp`` are merged
        in, with agent-scoped keys taking precedence on collision.

        Returns:
            Dict mapping ``client_id`` to ``MCPServerStatus``.
        """
        result = await self.mcp.get_server_status()
        # Merge pool-level servers when the agent has its own MCPManager
        # (not shared with the pool). Agent-scoped takes precedence on
        # key collision.
        host_ctx = self.host_context
        if host_ctx is not None and self.mcp is not host_ctx.mcp:
            pool_status = await host_ctx.mcp.get_server_status()
            pool_status.update(result)
            result = pool_status
        return result

    async def get_resource(self, name: str) -> Any:
        """Get a specific MCP resource by name or URI.

        Uses the ``ExtensionRegistry`` to query ``ResourceAccess`` providers
        at SESSION scope (POOL + AGENT + SESSION). Falls back to iterating
        ``_all_capabilities`` when the registry is unavailable.

        Args:
            name: Name or URI of the resource to find

        Raises:
            ToolError: If resource not found
        """
        import asyncio

        from wolfharness.capabilities.extension_registry import Scope, ScopeLevel
        from wolfharness.capabilities.resource_protocols import ResourceAccess
        from wolfharness.tools.exceptions import ToolError

        all_resources: list[Any] = []

        host_ctx = self.host_context
        registry = host_ctx.extension_registry if host_ctx is not None else None
        if registry is not None:
            session_id = self.session_id or ""
            if session_id:
                scope = Scope(
                    level=ScopeLevel.SESSION,
                    agent_name=self.name,
                    session_id=session_id,
                )
            else:
                scope = Scope(level=ScopeLevel.AGENT, agent_name=self.name)
            resource_caps = registry.get_resource_access(scope)
        else:
            caps = self._all_capabilities
            resource_caps = [cap for cap in caps if isinstance(cap, ResourceAccess)]

        if resource_caps:
            results = await asyncio.gather(
                *(cap.list_resources() for cap in resource_caps),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    continue
                all_resources.extend(result)
        resource = next(
            (
                r
                for r in all_resources
                if getattr(r, "name", None) == name or getattr(r, "uri", None) == name
            ),
            None,
        )
        if not resource:
            raise ToolError(f"Resource not found: {name}")
        return resource

    async def list_prompts(self) -> list[Any]:
        """Get all prompts from external capabilities.

        Queries the ``ExtensionRegistry`` at SESSION scope (falling back to
        AGENT scope when no session is available) to find
        ``FunctionToolsetCapability`` instances and collects their MCP prompts.

        Returns:
            List of prompt objects from capabilities that provide prompts
        """
        from wolfharness.capabilities.extension_registry import Scope, ScopeLevel
        from wolfharness.capabilities.function_toolset import FunctionToolsetCapability
        from wolfharness.prompts.prompts import MCPClientPrompt as MCPPrompt

        all_prompts: list[Any] = []

        host_ctx = self.host_context
        registry = host_ctx.extension_registry if host_ctx is not None else None
        if registry is not None:
            session_id = self.session_id or ""
            if session_id:
                scope = Scope(
                    level=ScopeLevel.SESSION,
                    agent_name=self.name,
                    session_id=session_id,
                )
            else:
                scope = Scope(level=ScopeLevel.AGENT, agent_name=self.name)
            caps = registry.get_visible_capabilities(scope)
        else:
            caps = self._external_capabilities

        for provider in caps:
            if isinstance(provider, FunctionToolsetCapability):
                try:
                    prompts = await provider.get_prompts()
                    mcp_prompts = [p for p in prompts if isinstance(p, MCPPrompt)]
                    all_prompts.extend(mcp_prompts)
                except Exception:
                    logger.exception(
                        "Failed to get prompts from provider",
                        provider=provider,
                    )
        return all_prompts

    @asynccontextmanager
    async def _with_session_capabilities(
        self,
        capabilities: Sequence[Any],
    ) -> AsyncIterator[None]:
        """Temporarily inject session-level capabilities for the duration of a run.

        Args:
            capabilities: Session-level capabilities to inject
        """
        self._session_capabilities[:] = list(capabilities)
        try:
            yield
        finally:
            self._session_capabilities.clear()

    @asynccontextmanager
    async def _temporary_tools(
        self,
        tools: Any,
        *,
        exclusive: bool = False,
    ) -> AsyncIterator[list[Any]]:
        """Temporarily register tools on the builtin provider.

        Args:
            tools: Tool(s) to register (Tool, callable, str, or sequence thereof)
            exclusive: Whether to temporarily disable all other tools

        Yields:
            List of registered tool infos
        """
        from wolfharness.tools.base import Tool as ToolClass

        match tools:
            case str() | Callable() | ToolClass():
                tools_list = [tools]
            case _:
                tools_list = list(tools)
        # Store original tool states if exclusive
        all_tools = await self._get_all_tools()
        original_states: dict[str, bool] = {}
        if exclusive:
            original_states = {t.name: t.enabled for t in all_tools}
            for t in all_tools:
                t.enabled = False
        registered_tools: list[Any] = []
        try:
            for tool in tools_list:
                tool_info = self._builtin_provider.register_tool(tool)
                registered_tools.append(tool_info)
            yield registered_tools
        finally:
            for tool_info in registered_tools:
                self._builtin_provider.remove_tool(tool_info.name)
            if exclusive:
                for name_, was_enabled in original_states.items():
                    all_tools_local = await self._get_all_tools()
                    for t_ in all_tools_local:
                        if t_.name == name_:
                            t_.enabled = was_enabled

    def get_context(
        self,
        data: Any = None,
        input_provider: InputProvider | None = None,
        tool_call_id: str | None = None,
        tool_input: dict[str, Any] | None = None,
        tool_name: str | None = None,
        run_ctx: AgentRunContext | None = None,
    ) -> AgentContext[Any]:
        """Create a new context for this agent.

        Args:
            data: Optional custom data to attach to the context
            input_provider: Optional input provider override
            tool_call_id: Optional tool call ID
            tool_input: Optional tool input
            tool_name: Optional tool name
            run_ctx: Optional per-run context for accessing run-isolated state

        Returns:
            A new AgentContext instance
        """
        return AgentContext(
            node=self,
            pool=self._agent_pool,
            input_provider=input_provider or self._input_provider,
            data=data,
            model_name=self.model_name,
            tool_call_id=tool_call_id,
            tool_input=tool_input or {},
            tool_name=tool_name,
            run_ctx=run_ctx,
        )

    @property
    @abstractmethod
    def model_name(self) -> str | None:
        """Get the model name used by this agent."""
        ...

    @abstractmethod
    async def set_model(self, model: str) -> None:
        """Set the model for this agent.

        Args:
            model: New model identifier to use
        """
        ...

    @abstractmethod
    def create_turn(
        self,
        prompts: list[UserContent],
        run_ctx: AgentRunContext,
        message_history: list[ModelMessage],
    ) -> Turn:
        """Create a Turn for single-cycle execution.

        Args:
            prompts: Pre-converted prompt strings for this turn.
            run_ctx: Per-run isolated context.
            message_history: Incoming message history.

        Returns:
            A Turn instance that can be executed via execute().
        """
        ...

    def create_run(
        self,
        prompt: str,
        run_ctx: AgentRunContext,
        message_history: list[ModelMessage],
        event_bus: EventBus,
        session: SessionState,
    ) -> RunHandle:
        """Construct a RunHandle for v2 session-level execution.

        This is the v2 entry point that replaces the legacy ``run()``
        method for session-managed runs. It only constructs the
        RunHandle — no execution happens here. The caller is
        responsible for calling ``run_handle.start(prompt)`` to begin
        the idle/wake/turn loop, or for using :meth:`create_run_stream`
        which wraps that pattern.

        Args:
            prompt: Initial user prompt for the first turn. Not used
                during construction; pass it to ``start()`` when ready.
            run_ctx: Per-run isolated context.
            message_history: Incoming message history for the first turn.
            event_bus: Event bus for publishing stream events.
            session: Per-session state containing the turn lock.

        Returns:
            A RunHandle wired with agent, event_bus, session, and
            run_ctx, ready to be started via ``start(prompt)``.
        """
        from wolfharness.orchestrator.run import RunHandle

        return RunHandle(
            run_id=run_ctx.run_id,
            session_id=run_ctx.session_id,
            agent_type=self.AGENT_TYPE,
            agent=self,
            event_bus=event_bus,
            session=session,
            run_ctx=run_ctx,
            _message_history=list(message_history),
        )

    async def create_run_stream(
        self,
        prompt: str,
        run_ctx: AgentRunContext,
        message_history: list[ModelMessage],
        event_bus: EventBus,
        session: SessionState,
    ) -> AsyncGenerator[RichAgentStreamEvent[TResult]]:
        """Run agent with streaming output via the v2 RunHandle lifecycle.

        This is the v2 streaming wrapper around :meth:`create_run` and
        ``RunHandle.start()``. It constructs a RunHandle, starts the
        idle/wake/turn loop, yields all stream events, and closes the
        handle when the stream completes.

        Args:
            prompt: Initial user prompt for the first turn.
            run_ctx: Per-run isolated context.
            message_history: Incoming message history for the first turn.
            event_bus: Event bus for publishing stream events.
            session: Per-session state containing the turn lock.

        Yields:
            Stream events from the turn execution.
        """
        async with self.create_run(
            prompt=prompt,
            run_ctx=run_ctx,
            message_history=message_history,
            event_bus=event_bus,
            session=session,
        ) as run_handle:
            async for event in run_handle.start(prompt):
                yield event
                if isinstance(event, StreamCompleteEvent | RunErrorEvent):
                    run_handle.close()
                    break

    async def run_iter(
        self,
        *prompt_groups: Sequence[PromptCompatible],
        store_history: bool = True,
        wait_for_connections: bool | None = None,
    ) -> AsyncIterator[ChatMessage[TResult]]:
        """Run agent sequentially on multiple prompt groups.

        Args:
            prompt_groups: Groups of prompts to process sequentially
            store_history: Whether to store in conversation history
            wait_for_connections: Whether to wait for connected agents

        Yields:
            Response messages in sequence

        Example:
            questions = [
                ["What is your name?"],
                ["How old are you?", image1],
                ["Describe this image", image2],
            ]
            async for response in agent.run_iter(*questions):
                print(response.content)
        """
        for prompts in prompt_groups:
            response = await self.run(
                *prompts,
                store_history=store_history,
                wait_for_connections=wait_for_connections,
            )
            yield response  # pyright: ignore

    async def run_in_background(
        self,
        *prompt: PromptCompatible,
        max_count: int | None = None,
        interval: float = 1.0,
        **kwargs: Any,
    ) -> asyncio.Task[ChatMessage[TResult] | None]:
        """Run agent continuously in background with prompt or dynamic prompt function.

        Args:
            prompt: Static prompt or function that generates prompts
            max_count: Maximum number of runs (None = infinite)
            interval: Seconds between runs
            **kwargs: Arguments passed to run()
        """
        self._infinite = max_count is None

        # Create run context for background task
        self._background_run_ctx = AgentRunContext()

        async def _continuous() -> ChatMessage[Any]:
            count = 0
            self.log.debug("Starting continuous run", max_count=max_count, interval=interval)
            latest = None
            # _background_run_ctx is always set before starting the task
            run_ctx = self._background_run_ctx
            assert run_ctx is not None
            while (max_count is None or count < max_count) and not run_ctx.cancelled:
                try:
                    agent_ctx = self.get_context(run_ctx=run_ctx)
                    current_prompts = [
                        call_with_context(p, agent_ctx, **kwargs) if callable(p) else p
                        for p in prompt
                    ]
                    self.log.debug("Generated prompt", iteration=count)
                    latest = await self.run(current_prompts, **kwargs)
                    self.log.debug("Run continuous result", iteration=count)

                    count += 1
                    await anyio.sleep(interval)
                except asyncio.CancelledError:
                    self.log.debug("Continuous run cancelled")
                    break
                except Exception:
                    # Check if we were cancelled (may surface as other exceptions)
                    if run_ctx.cancelled:
                        self.log.debug("Continuous run cancelled via flag")
                        break
                    count += 1
                    self.log.exception("Background run failed")
                    await anyio.sleep(interval)
            self.log.debug("Continuous run completed", iterations=count)
            return latest  # type: ignore[return-value]  # ty: ignore[invalid-return-type]

        await self.stop()  # Cancel any existing background task
        self._cancelled = False  # Reset cancellation flag for backward compat
        self._background_run_ctx.cancelled = False
        task = asyncio.create_task(_continuous(), name=f"background_{self.name}")
        self.log.debug("Started background task", task_name=task.get_name())
        self._background_task = task
        return task

    async def stop(self) -> None:
        """Stop continuous execution if running."""
        self._cancelled = True  # Signal cancellation via flag for backward compat
        if self._background_run_ctx:
            self._background_run_ctx.cancelled = True
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            with suppress(asyncio.CancelledError):  # Expected when we cancel the task
                await self._background_task
            self._background_task = None

    def is_busy(self) -> bool:
        """Check if agent is currently processing tasks."""
        return bool(self._pending_tasks or self._background_task)

    async def wait(self) -> ChatMessage[TResult]:
        """Wait for background execution to complete."""
        if not self._background_task:
            raise RuntimeError("No background task running")
        if self._infinite:
            raise RuntimeError("Cannot wait on infinite execution")
        try:
            return await self._background_task
        finally:
            self._background_task = None

    def _get_session_run_ctx(self, session_id: str | None = None) -> AgentRunContext | None:
        """Get active run context from SessionPool for cross-task access.

        Args:
            session_id: Optional session ID to look up. Falls back to
                self._events.session_id if not provided.

        Returns:
            The session's active run context, or None if not found.
        """
        if self.host_context is not None:
            session_pool = self.host_context.session_pool
            if session_pool is None:
                return None
            effective_session_id = session_id or getattr(self._events, "session_id", None)
            if effective_session_id is not None:
                session = session_pool.sessions.get_session(effective_session_id)
                if session is not None and session.current_run_id is not None:
                    run_handle = session_pool.get_run(session.current_run_id)
                    if run_handle is not None:
                        run_ctx = run_handle.run_ctx
                        if run_ctx is not None and not run_ctx.completed:
                            return run_ctx
        return None

    def get_active_run_context(self, session_id: str | None = None) -> AgentRunContext | None:
        """Get the currently active run context.

        Public API for external callers (e.g., SessionPool) to check if a
        turn is active and access the run context without relying on
        private attributes.

        Uses three-level fallback:
        1. ContextVar (_current_run_ctx_var) for the current task
        2. SessionPool lookup when pooled (via session_id or agent_pool)
        3. _background_run_ctx for background task state

        Args:
            session_id: Optional session ID for SessionPool lookup.
                When provided, used for the SessionPool fallback instead of
                instance state.

        Returns:
            The active run context, or None if no turn is running.
        """
        # Level 1: ContextVar for the current task (highest precedence)
        run_ctx = _current_run_ctx_var.get()
        if run_ctx is not None and not run_ctx.completed:
            return run_ctx

        # Level 2: SessionPool lookup when pooled and session has active run
        ctx = self.host_context
        if ctx is not None:
            session_pool = ctx.session_pool
            if session_pool is not None:
                effective_session_id = session_id or getattr(self._events, "session_id", None)
                if effective_session_id is not None:
                    session = session_pool.sessions.get_session(effective_session_id)
                    if session is not None and session.current_run_id is not None:
                        run_handle = session_pool.get_run(session.current_run_id)
                        if run_handle is not None and not run_handle.run_ctx.completed:
                            return run_handle.run_ctx

        # Level 2.5: Instance-level _run_context for standalone (pool-less) runs.
        # This enables cross-task is_turn_active() checks when no SessionPool exists.
        if self._run_context is not None and not self._run_context.completed:
            return self._run_context

        # Level 3: Background run context (lowest precedence)
        if self._background_run_ctx is not None and not self._background_run_ctx.completed:
            return self._background_run_ctx

        return None

    def is_turn_active(self) -> bool:
        """Check if a turn is currently running.

        Returns:
            True if there is an active run context, False otherwise.
        """
        return self.get_active_run_context() is not None

    def queue_prompt(self, *prompts: PromptCompatible, session_id: str | None = None) -> None:
        """Queue a prompt to be processed after the current run completes.

        When called during an active run_stream, the queued prompt will be
        processed in a continuation loop without exiting the stream. This
        allows tools or external code to schedule follow-up work.

        !!! warning "Deprecated for pooled agents"
            Use ``host_context.session_pool.followup()`` instead.

        For standalone agents (no session pool), the injection_manager-based
        path is used as a fallback.

        Args:
            *prompts: Prompts to queue (same format as run/run_stream)
            session_id: Optional session ID for SessionPool fallback lookup.

        Example:
            # In a tool implementation:
            async def my_tool(ctx: AgentContext) -> str:
                ctx.agent.queue_prompt("Now analyze the results")
                return "Initial work done"
        """
        run_ctx = self.get_active_run_context(session_id=session_id)

        # Pooled agents: delegate to session_pool.followup().
        ctx = self.host_context
        if ctx is not None and ctx.session_pool is not None:
            warnings.warn(
                "queue_prompt() is deprecated for pooled agents. "
                "Use host_context.session_pool.followup() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            session_pool = ctx.session_pool
            effective_session_id = session_id or (
                run_ctx.session_id if run_ctx else self._events.session_id
            )
            if effective_session_id is not None:
                combined = "\n".join(str(p) for p in prompts)
                self.fire_and_forget(session_pool.followup(effective_session_id, combined))
                return

        # Standalone agents: use injection_manager
        if run_ctx is not None and run_ctx.injection_manager is not None:
            combined = "\n".join(str(p) for p in prompts)
            run_ctx.injection_manager.inject(combined)

    def inject_prompt(self, message: str, session_id: str | None = None) -> None:
        """Inject a message into the conversation mid-run.

        The message will be injected after the next tool completes (if the
        agent supports tool hooks). If no tool executes before the run
        iteration completes, the message is automatically queued for the
        next iteration.

        !!! warning "Deprecated for pooled agents"
            Use ``host_context.session_pool.steer()`` instead.

        For standalone agents (no session pool), the injection_manager-based
        path is used as a fallback.

        Args:
            message: Message to inject
            session_id: Optional session ID. Falls back to the active run
                context's session_id if available.

        Example:
            # In a tool implementation:
            async def my_tool(ctx: AgentContext) -> str:
                ctx.agent.inject_prompt("Also check the test coverage")
                return "Changes made"
        """
        run_ctx = self.get_active_run_context(session_id=session_id)
        ctx = self.host_context

        # Pooled agents: delegate to session_pool.steer().
        if ctx is not None and ctx.session_pool is not None:
            warnings.warn(
                "inject_prompt() is deprecated for pooled agents. "
                "Use host_context.session_pool.steer() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            session_pool = ctx.session_pool
            effective_session_id = session_id or (
                run_ctx.session_id if run_ctx else self._events.session_id
            )
            if effective_session_id is not None:
                self.fire_and_forget(session_pool.steer(effective_session_id, message))
                return
            # FALLBACK: effective_session_id is None but session_pool exists.
            # This happens when BackgroundTaskProvider calls inject_prompt
            # after the lead agent's run has ended (no active run context
            # and agent's _events.session_id is None for shared agents).
            # Try to find the most recently active session for this agent.
            sessions = session_pool.sessions.find_sessions_by_agent_name(self.name)
            if sessions:
                most_recent = max(sessions, key=lambda s: s.last_active_at)
                self.fire_and_forget(session_pool.steer(most_recent.session_id, message))
                return

        # Standalone agents: use injection_manager
        if run_ctx is not None and not run_ctx.completed and run_ctx.injection_manager is not None:
            run_ctx.injection_manager.inject(message)

    def has_pending_injections(self, session_id: str | None = None) -> bool:
        """Check if there are pending injections.

        Args:
            session_id: Optional session ID for SessionPool fallback lookup.
        """
        run_ctx = self.get_active_run_context(session_id=session_id)
        if run_ctx is not None and run_ctx.injection_manager is not None:
            return run_ctx.injection_manager.has_pending()
        return False

    def _maybe_pool_stream(
        self,
        *,
        prompts: tuple[PromptCompatible, ...],
        session_id: str | None,
        parent_session_id: str | None,
        input_provider: InputProvider | None,
        depth: int,
        wait_for_connections: bool | None,
        _skip_pool: bool,
    ) -> AsyncIterator[RichAgentStreamEvent[TResult]] | None:
        """Return async iterator for pool-based streaming, or None if not applicable."""
        from wolfharness.utils.identifiers import generate_session_id

        if (
            not _skip_pool
            and self.host_context is not None
            and self.host_context.session_pool is not None
            and not _in_turn_context.get()
        ):
            session_pool = self.host_context.session_pool
            # Fall back to the agent's home session (set via set_session_context
            # during AgentFactory.create_session_agent) before generating a new
            # session_id. This ensures that instance-level state mutations
            # (_temporary_tools, /register-tool, override(tools=...)) are
            # preserved when run_stream() is called without an explicit
            # session_id — see issue #204.
            effective_session_id = (
                session_id or getattr(self._events, "session_id", None) or generate_session_id()
            )
            existing_session = session_pool.sessions.get_session(effective_session_id)
            if existing_session is None or existing_session.agent_name == self.name:
                return self._pool_stream_iter(
                    session_pool=session_pool,
                    effective_session_id=effective_session_id,
                    existing_session=existing_session,
                    prompts=prompts,
                    parent_session_id=parent_session_id,
                    input_provider=input_provider,
                    depth=depth,
                    wait_for_connections=wait_for_connections,
                )
        return None

    @method_spawner
    async def _pool_stream_iter(
        self,
        *,
        session_pool: SessionPool,
        effective_session_id: str,
        existing_session: SessionState | None,
        prompts: tuple[PromptCompatible, ...],
        parent_session_id: str | None,
        input_provider: InputProvider | None,
        depth: int,
        wait_for_connections: bool | None,
    ) -> AsyncIterator[RichAgentStreamEvent[TResult]]:
        """Path A: Stream events via SessionPool."""
        if existing_session is None:
            user_prompts = [str(p) for p in prompts if isinstance(p, str)]
            initial_prompt = user_prompts[-1] if user_prompts else None
            title_initial_prompt = self._session_initial_prompt_for_title(
                effective_session_id,
                initial_prompt,
            )
            await self.log_session(
                session_id=effective_session_id,
                initial_prompt=title_initial_prompt,
                model=self.model_name,
                parent_session_id=parent_session_id,
            )
            await session_pool.create_session(
                effective_session_id,
                agent_name=self.name,
                parent_session_id=parent_session_id,
            )

        final_message: ChatMessage[TResult] | None = None
        async for event in session_pool.run_stream(
            effective_session_id,
            *prompts,
            input_provider=input_provider,
            parent_session_id=parent_session_id,
            depth=depth,
        ):
            yield event

            if isinstance(event, StreamCompleteEvent):
                final_message = event.message
            if isinstance(event, StreamCompleteEvent | RunErrorEvent):
                break
        if final_message is not None:
            await self.message_sent.emit(final_message)
            session = session_pool.sessions.get_session(effective_session_id)
            if session is not None and getattr(session, "is_per_session_agent", False):
                await self.connections.route_message(final_message, wait=wait_for_connections)

    async def _prepare_standalone_context(
        self,
        *,
        prompts: tuple[PromptCompatible, ...],
        session_id: str | None,
        parent_session_id: str | None,
        deps: TDeps | None,
        depth: int,
        _run_ctx: AgentRunContext | None,
    ) -> tuple[AgentRunContext, str, EventBus, Any, bool]:
        """Prepare run context for standalone streaming execution.

        Returns:
            (run_ctx, effective_session_id, local_bus, stream, created_local_bus)
        """
        from wolfharness.orchestrator.core import EventBus
        from wolfharness.utils.identifiers import generate_session_id

        if _run_ctx is not None:
            run_ctx = _run_ctx
            effective_session_id = session_id or run_ctx.session_id
        else:
            effective_session_id = session_id or generate_session_id()

            user_prompts = [str(p) for p in prompts if isinstance(p, str)]
            initial_prompt = user_prompts[-1] if user_prompts else None

            await self.log_session(
                session_id=effective_session_id,
                initial_prompt=initial_prompt,
                model=self.model_name,
                parent_session_id=parent_session_id,
            )

            run_ctx = AgentRunContext(deps=deps, depth=depth, session_id=effective_session_id)

        _created_local_bus = run_ctx.event_bus is None
        if _created_local_bus:
            local_bus: EventBus = EventBus()
            run_ctx.event_bus = local_bus
        else:
            assert run_ctx.event_bus is not None
            local_bus = run_ctx.event_bus

        stream = await local_bus.subscribe(effective_session_id, scope="session")
        return run_ctx, effective_session_id, local_bus, stream, _created_local_bus

    def _get_consumer_handler(
        self, event_handlers: Sequence[AnyEventHandlerType] | None
    ) -> MultiEventHandler[IndividualEventHandler]:
        """Return the appropriate event handler for streaming consumption."""
        if event_handlers is not None:
            return MultiEventHandler[IndividualEventHandler](resolve_event_handlers(event_handlers))
        return self.event_handler

    def _cleanup_after_stream(
        self, run_ctx: AgentRunContext, token: Token[AgentRunContext | None] | None
    ) -> None:
        """Clean up after streaming: reset ContextVar, clear injections, clear run context."""
        if token is not None:
            with suppress(ValueError):
                _current_run_ctx_var.reset(token)
        if run_ctx.injection_manager is not None:
            run_ctx.injection_manager.clear()
        if self._run_context is run_ctx:
            self._run_context = None

    @method_spawner  # type: ignore[arg-type]
    async def run_stream(  # type: ignore[override]  # noqa: PLR0915
        self,
        *prompts: PromptCompatible,
        store_history: bool = True,
        message_id: str | None = None,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        parent_id: str | None = None,
        message_history: MessageHistory | None = None,
        input_provider: InputProvider | None = None,
        wait_for_connections: bool | None = None,
        deps: TDeps | None = None,
        event_handlers: Sequence[AnyEventHandlerType] | None = None,
        depth: int = 0,
        _run_ctx: AgentRunContext | None = None,
        _skip_pool: bool = False,
        **pydantic_ai_kwargs: Any,
    ) -> AsyncIterator[RichAgentStreamEvent[TResult]]:
        """Run agent with streaming output (the react loop).

        This is the self-contained react loop: it creates a per-run context,
        logs the session, and processes prompts through _run_stream_once().
        For native agents, PydanticAI's PendingMessageDrainCapability handles
        follow-up prompts; for non-native agents, a manual follow-up loop
        drains the injection queue.

        This method can be used standalone (no SessionPool required) or
        called indirectly via SessionPool-managed agents. Protocol servers
        that need session lifecycle management should use
        ``SessionPool.run_stream()`` instead.

        If prompts are queued via queue_prompt() during execution, they will be
        processed in sequence without exiting the stream.

        Args:
            *prompts: Input prompts (various formats supported)
            store_history: Whether to store in history
            message_id: Optional message ID
            session_id: Optional conversation ID
            parent_session_id: Optional parent conversation ID
            parent_id: Optional parent message ID
            message_history: Optional message history
            input_provider: Optional input provider
            wait_for_connections: Whether to wait for connected agents
            deps: Optional dependencies
            event_handlers: Optional event handlers
            depth: Current delegation depth (0 = top-level run)
            **pydantic_ai_kwargs: Extra kwargs forwarded to PydanticAI.

        Yields:
            Stream events during execution
        """
        from wolfharness.orchestrator.core import drain_and_merge

        # --- Path A: SessionPool available & outside a turn context ---
        pool_stream = self._maybe_pool_stream(
            prompts=prompts,
            session_id=session_id,
            parent_session_id=parent_session_id,
            input_provider=input_provider,
            depth=depth,
            wait_for_connections=wait_for_connections,
            _skip_pool=_skip_pool,
        )
        if pool_stream is not None:
            async for event in pool_stream:
                yield event
            return

        # --- Path B: Standalone / in-turn react loop ---
        # Producer/consumer pattern: _run_stream_once() publishes events to
        # local_bus, consumer drains via drain_and_merge(). This captures
        # events that bypass _stream_events() (e.g. ToolCallProgressEvent
        # from report_progress, SpawnSessionStart from create_child_session).
        (
            run_ctx,
            effective_session_id,
            local_bus,
            queue,
            _created_local_bus,
        ) = await self._prepare_standalone_context(
            prompts=prompts,
            session_id=session_id,
            parent_session_id=parent_session_id,
            deps=deps,
            depth=depth,
            _run_ctx=_run_ctx,
        )
        run_ctx.cancelled = False
        self._cancelled = False
        run_ctx.current_task = asyncio.current_task()
        self._run_context = run_ctx

        # When lifecycle_config is set, pre-create dimensions so they're
        # available for any RunHandle that wraps this run (e.g. via
        # SessionController). When None or all-defaults, RunHandle's
        # __post_init__ creates in-memory defaults automatically.
        if self._lifecycle_config is not None and not self._lifecycle_config.is_all_defaults():
            from wolfharness.lifecycle.factory import create_dimensions

            self._lifecycle_dimensions = create_dimensions(
                self._lifecycle_config,
                effective_session_id,
            )

        token: Token[AgentRunContext | None] | None = None
        with safe_span("agent.run_stream"):
            try:
                token = _current_run_ctx_var.set(run_ctx)

                producer_error: BaseException | None = None

                async def _producer() -> None:
                    nonlocal producer_error
                    try:
                        async for event in self._run_stream_once(
                            run_ctx,
                            *prompts,
                            store_history=store_history,
                            message_id=message_id,
                            session_id=effective_session_id,
                            parent_session_id=parent_session_id,
                            parent_id=parent_id,
                            message_history=message_history,
                            input_provider=input_provider,
                            wait_for_connections=wait_for_connections,
                            deps=deps,
                            event_handlers=event_handlers,
                            _owns_event_bus=_created_local_bus,
                            **pydantic_ai_kwargs,
                        ):
                            await local_bus.publish(effective_session_id, event)
                    except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
                        producer_error = exc
                    finally:
                        with anyio.CancelScope(shield=True):
                            if _created_local_bus:
                                await local_bus.close_session(effective_session_id)
                            else:
                                await local_bus.unsubscribe(effective_session_id, queue)

                producer_task = asyncio.ensure_future(_producer())
                try:
                    consumer_handler = self._get_consumer_handler(event_handlers)
                    consumer_context = self.get_context(
                        input_provider=input_provider, run_ctx=run_ctx
                    )

                    async for envelope in drain_and_merge(queue):
                        event = envelope.event
                        with suppress(
                            ValueError, TypeError, RuntimeError, KeyError, AttributeError
                        ):
                            await consumer_handler(consumer_context, event)
                        yield event
                        if isinstance(event, StreamCompleteEvent):
                            break
                        if isinstance(event, RunErrorEvent):
                            break
                finally:
                    with suppress(asyncio.CancelledError):
                        await producer_task

                if producer_error is not None:
                    raise producer_error
            finally:
                self._cleanup_after_stream(run_ctx, token)

    async def _run_stream_once(
        self,
        run_ctx: AgentRunContext,
        *prompts: PromptCompatible,
        store_history: bool = True,
        message_id: str | None = None,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        parent_id: str | None = None,
        message_history: MessageHistory | None = None,
        input_provider: InputProvider | None = None,
        wait_for_connections: bool | None = None,
        deps: TDeps | None = None,
        event_handlers: Sequence[AnyEventHandlerType] | None = None,
        _owns_event_bus: bool = False,
        **pydantic_ai_kwargs: Any,
    ) -> AsyncIterator[RichAgentStreamEvent[TResult]]:
        """Process a single prompt group with streaming output.

        This is the internal implementation called by run_stream().
        Session initialization is handled by the caller.

        Args:
            run_ctx: Per-run context for state isolation
            *prompts: Input prompts (various formats supported)
            store_history: Whether to store in history
            message_id: Optional message ID
            session_id: Optional conversation ID
            parent_session_id: Optional parent conversation ID
            parent_id: Optional parent message ID
            message_history: Optional message history
            input_provider: Optional input provider
            wait_for_connections: Whether to wait for connected agents
            deps: Optional dependencies
            event_handlers: Optional event handlers
            _owns_event_bus: Whether the caller created a local EventBus
                (standalone mode). When True, ``message_sent`` is emitted
                here. When False, the caller (Path A) handles emission.
            **pydantic_ai_kwargs: Extra kwargs forwarded to PydanticAI.

        Yields:
            Stream events during execution
        """
        from wolfharness.messaging import ChatMessage

        # Convert prompts to standard UserContent format
        converted_prompts = await convert_prompts(prompts)
        # Prepend any staged content
        if staged := await self.staged_content.consume_as_text():
            converted_prompts = [*converted_prompts, staged]
        # Get message history (either passed or agent's own)
        conversation = message_history if message_history is not None else self.conversation
        # Determine effective parent_id (from param or last message in history)
        pending_parts = conversation.get_pending_parts()
        effective_parent_id = parent_id if parent_id else conversation.get_last_message_id()

        user_msg = ChatMessage.user_prompt(
            message=converted_prompts,
            parent_id=effective_parent_id,
            session_id=session_id,
        )

        # Event handler dispatch is now performed in the consumer loop
        # of run_stream(), which sees all events including those published
        # directly to the EventBus (e.g. ToolCallProgressEvent from
        # report_progress).  This avoids missing events that bypass
        # _stream_events().

        # Stream events from implementation
        final_message = None
        conversation = message_history if message_history is not None else self.conversation

        # run_ctx is created by run_stream() (or future callers); do not replace it here or
        # per-run isolation (event queue, injections, cancellation) breaks.

        await self.message_received.emit(user_msg)

        # Save user message to conversation history immediately
        # This ensures user messages are preserved even if the run is cancelled
        # or no assistant response is generated (e.g., elicitation cancelled)
        if store_history:
            conversation.add_chat_messages([user_msg])

        try:
            async for event in self._stream_events(
                run_ctx,
                [*pending_parts, *converted_prompts],
                user_msg=user_msg,
                effective_parent_id=effective_parent_id,
                store_history=store_history,
                message_id=message_id,
                session_id=session_id,
                parent_session_id=parent_session_id,
                parent_id=parent_id,
                message_history=conversation,
                input_provider=input_provider,
                wait_for_connections=wait_for_connections,
                deps=deps,
                **pydantic_ai_kwargs,
            ):
                yield event
                # Capture final message from StreamCompleteEvent
                if isinstance(event, StreamCompleteEvent):
                    final_message = event.message
                    break
                # On RunErrorEvent, don't break — let _stream_events()
                # raise the error on the next __anext__() call so that
                # producer_error is set in _native_runner().
        except Exception:
            self.log.exception("Agent stream failed")
            raise

        # Pick up result from side channel when _stream_events yielded nothing
        # (native agents: events went directly to EventBus via executor.execute)
        if final_message is None and run_ctx.terminal_tool_result is not None:
            final_message = run_ctx.terminal_tool_result

        # Post-processing after stream completes — shielded to prevent
        # TaskGroup cancellation from interrupting hooks/routing/persistence
        if final_message is not None:
            with anyio.CancelScope(shield=True):
                # Emit signal (always - for event handlers).
                # Skip when run_ctx was provided by SessionPool (Path A);
                # the Path A wrapper in run_stream() handles emission in that case.
                # When we created a local bus ourselves (_owns_event_bus),
                # we are in standalone mode and must emit here.
                if _owns_event_bus:
                    await self.message_sent.emit(final_message)
                # Route to connected agents (always - they decide what to do with it)
                await self.connections.route_message(final_message, wait=wait_for_connections)
                # Conditional persistence based on store_history
                # TODO: Verify store_history semantics across all use cases:
                #   - Should subagent tool calls set store_history=False?
                #   - Should forked/ephemeral runs always skip persistence?
                #   - Should signals still fire when store_history=False?
                #   Current behavior: store_history controls both DB logging
                #   AND conversation context
                if store_history:
                    # Log to persistent storage and add to conversation context
                    # Note: user_msg was already added at the start of the run
                    # Use extend_last=True to include both user_msg and
                    # final_message in _last_messages
                    await self.log_message(final_message)
                    conversation.add_chat_messages([final_message], extend_last=True)

    async def _execute_slash_command_streaming(
        self, command_text: str
    ) -> AsyncIterator[CommandOutputEvent | CommandCompleteEvent]:
        """Execute a single slash command and yield events as they happen.

        Args:
            command_text: Full command text including slash

        Yields:
            Command output and completion events
        """
        from slashed.events import (
            CommandExecutedEvent,
            CommandOutputEvent as SlashedCommandOutputEvent,
        )

        from wolfharness.agents.events import CommandCompleteEvent, CommandOutputEvent

        parsed = _parse_slash_command(command_text)
        if not parsed:
            self.log.warning("Invalid slash command", command=command_text)
            yield CommandCompleteEvent(command="unknown", success=False)
            return

        cmd_name, args = parsed
        # Set up event queue for this command execution
        event_queue: asyncio.Queue[Any] = asyncio.Queue()
        # Temporarily set event handler on command store
        old_handler = self._command_store.event_handler
        self._command_store.event_handler = event_queue.put
        # Use active run context (ContextVar + SessionPool fallback)
        run_ctx = self.get_active_run_context()
        cmd_ctx = self._command_store.create_context(data=self.get_context(run_ctx=run_ctx))
        command_str = f"{cmd_name} {args}".strip()
        try:
            execute_task = asyncio.create_task(
                self._command_store.execute_command(command_str, cmd_ctx)
            )
            success = True
            # Yield events from queue as command runs
            while not execute_task.done():
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                    match event:
                        case SlashedCommandOutputEvent(output=output):
                            yield CommandOutputEvent(command=cmd_name, output=output)
                        case CommandExecutedEvent(success=False, error=error) if error:
                            yield CommandOutputEvent(
                                command=cmd_name, output=f"Command error: {error}"
                            )
                            success = False
                except TimeoutError:
                    continue

            # Ensure command task completes and handle any remaining events
            try:
                await execute_task
            except Exception as e:
                self.log.exception("Command execution failed", command=cmd_name)
                success = False
                yield CommandOutputEvent(command=cmd_name, output=f"Command error: {e}")

            # Drain any remaining events from queue
            while not event_queue.empty():
                try:
                    event = event_queue.get_nowait()
                    if isinstance(event, SlashedCommandOutputEvent):
                        yield CommandOutputEvent(command=cmd_name, output=event.output)
                except asyncio.QueueEmpty:
                    break

            yield CommandCompleteEvent(command=cmd_name, success=success)

        finally:
            self._command_store.event_handler = old_handler

    async def run_stream_with_commands(
        self,
        *prompts: PromptCompatible,
        **kwargs: Any,
    ) -> AsyncIterator[StreamWithCommandsEvent[TResult]]:
        """Run agent with slash command support.

        Separates slash commands from regular prompts, executes commands first,
        then processes remaining content through the agent.

        Args:
            *prompts: Input prompts (may include slash commands)
            **kwargs: Additional arguments passed to run_stream

        Yields:
            Command events from slash command execution, then stream events from agent
        """
        # Separate slash commands from regular content
        commands: list[str] = []
        regular_prompts: list[Any] = []

        for prompt in prompts:
            if isinstance(prompt, str) and _is_slash_command(prompt):
                self.log.debug("Found slash command", command=prompt)
                commands.append(prompt.strip())
            else:
                regular_prompts.append(prompt)

        # Execute all commands first with streaming
        for command in commands:
            self.log.info("Processing slash command", command=command)
            async for cmd_event in self._execute_slash_command_streaming(command):
                yield cmd_event

        # If we have regular content, process it through the agent
        if regular_prompts:
            self.log.debug("Processing prompts through agent", num_prompts=len(regular_prompts))
            async for event in self.run_stream(*regular_prompts, **kwargs):  # type: ignore[attr-defined]
                yield event

    @abstractmethod
    def _stream_events(
        self,
        run_ctx: AgentRunContext,
        prompts: list[UserContent],
        *,
        user_msg: ChatMessage[Any],
        message_history: MessageHistory,
        effective_parent_id: str | None,
        message_id: str | None = None,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        parent_id: str | None = None,
        input_provider: InputProvider | None = None,
        deps: TDeps | None = None,
        wait_for_connections: bool | None = None,
        store_history: bool = True,
        **pydantic_ai_kwargs: Any,
    ) -> AsyncIterator[RichAgentStreamEvent[TResult]]:
        """Agent-specific streaming implementation.

        Subclasses must implement this to provide their streaming logic.
        Prompts are pre-converted to UserContent format by run_stream().

        Args:
            run_ctx: Per-run context for state isolation
            prompts: Converted prompts in UserContent format
            user_msg: Pre-created user ChatMessage (from base class)
            effective_parent_id: Resolved parent message ID for threading
            message_id: Optional message ID
            session_id: Optional conversation ID
            parent_session_id: Optional parent conversation ID
            parent_id: Optional parent message ID
            input_provider: Optional input provider
            message_history: Optional message history
            deps: Optional dependencies
            wait_for_connections: Whether to wait for connected agents
            store_history: Whether to store in history
            **pydantic_ai_kwargs: Extra kwargs forwarded to PydanticAI.

        Yields:
            Stream events during execution
        """
        ...

    async def set_tool_confirmation_mode(self, mode: str) -> None:
        """Set tool confirmation mode (agent-specific implementation).

        Each agent type handles permission modes differently:
        - NativeAgent: tool_confirmation_mode ("always", "never", "per_tool")
        - ACPAgent: auto_approve (bool)

        Subclasses should override this method if they support permission modes.
        The default implementation delegates to _set_mode(mode, "mode").

        Args:
            mode: Mode value in the agent's native format
        """
        await self._set_mode(mode, "mode")

    def is_initializing(self) -> bool:
        """Check if agent is still initializing.

        Returns:
            True if deferred initialization is pending
        """
        return self._connect_pending

    async def ensure_initialized(self) -> None:
        """Wait for deferred initialization to complete.

        Subclasses that use deferred init should:
        1. Set `self._connect_pending = True` in `__aenter__`
        2. Override this method to do actual connection work
        3. Set `self._connect_pending = False` when done

        The base implementation is a no-op for agents without deferred init.
        """

    def is_cancelled(self) -> bool:
        """Check if agent has been cancelled.

        Returns:
            True if cancellation was requested
        """
        # Check both instance flag (for backward compat) and background run context
        background_cancelled = (
            self._background_run_ctx.cancelled if self._background_run_ctx else False
        )
        return self._cancelled or background_cancelled

    async def interrupt(
        self, run_ctx: AgentRunContext | None = None, session_id: str | None = None
    ) -> None:
        """Interrupt the currently running stream.

        Sets the cancelled flag, calls subclass-specific _interrupt(),
        and emits the interrupted signal.

        When pooled, delegates to SessionPool to cancel the active run via
        RunHandle.  When run_ctx is not provided (e.g., from OpenCode
        abort_session), tries _current_run_ctx_var (ContextVar) first, then
        falls back to SessionPool's session.current_run_id + get_run() for
        cross-task access.

        Args:
            run_ctx: Optional per-run context for the stream to interrupt
            session_id: Optional session ID for SessionPool fallback lookup.
        """
        self._cancelled = True
        # When no run_ctx is provided, try ContextVar first (same task),
        # then fall back to SessionPool for cross-task access.
        effective_run_ctx = run_ctx
        if effective_run_ctx is None:
            effective_run_ctx = _current_run_ctx_var.get()
        if effective_run_ctx is None:
            effective_run_ctx = self._get_session_run_ctx(session_id=session_id)
        if effective_run_ctx:
            effective_run_ctx.cancelled = True
        if self._background_run_ctx:
            self._background_run_ctx.cancelled = True

        # When pooled, delegate to SessionPool for proper run cancellation.
        ctx = self.host_context
        if ctx is not None and ctx.session_pool is not None:
            effective_session_id = session_id or (
                effective_run_ctx.session_id if effective_run_ctx else self._events.session_id
            )
            if effective_session_id is not None:
                session_pool = ctx.session_pool
                session_pool.sessions.cancel_run_for_session(effective_session_id)

        await self._interrupt(effective_run_ctx)
        await self.interrupted.emit(self.InterruptEvent(agent_name=self.name))
        logger.info("Agent interrupted", agent=self.name)

    @abstractmethod
    async def _interrupt(self, run_ctx: AgentRunContext | None = None) -> None:
        """Subclass-specific interrupt implementation.

        Args:
            run_ctx: Optional per-run context for the stream to interrupt
        """

    async def get_stats(self) -> MessageStats:
        """Get message statistics."""
        from wolfharness.talk.stats import MessageStats

        return MessageStats(messages=list(self.conversation.chat_messages))

    async def get_mcp_server_info(self) -> dict[str, MCPServerStatus]:
        """Get information about configured MCP servers.

        Returns a dict mapping server names to their status info. Used by
        the OpenCode /mcp endpoint to display MCP servers in the UI.

        The default implementation checks external capabilities for MCP servers.
        Subclasses may override to provide agent-specific MCP server info.

        Returns:
            Dict mapping server name to MCPServerStatus
        """
        return await self._get_mcp_server_info()

    @method_spawner
    async def run(
        self,
        *prompts: PromptCompatible | ChatMessage[Any],
        store_history: bool = True,
        message_id: str | None = None,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        parent_id: str | None = None,
        message_history: MessageHistory | None = None,
        deps: TDeps | None = None,
        input_provider: InputProvider | None = None,
        event_handlers: Sequence[AnyEventHandlerType] | None = None,
        wait_for_connections: bool | None = None,
        depth: int = 0,
    ) -> ChatMessage[TResult]:
        """Run agent with prompt and get response.

        This is the standard synchronous run method shared by all agent types.
        It collects all streaming events from run_stream() and returns the final message.

        Args:
            prompts: User query or instruction
            store_history: Whether the message exchange should be added to the
                            context window
            message_id: Optional message id for the returned message.
                        Automatically generated if not provided.
            session_id: Optional conversation id for the returned message.
            parent_session_id: Optional parent conversation id.
            parent_id: Parent message id
            message_history: Optional MessageHistory object to
                             use instead of agent's own conversation
            deps: Optional dependencies for the agent
            input_provider: Optional input provider for the agent
            event_handlers: Optional event handlers for this run (overrides agent's handlers)
            wait_for_connections: Whether to wait for connected agents to complete
            depth: Current delegation depth (0 = top-level run)

        Returns:
            ChatMessage containing response and run information

        Raises:
            RuntimeError: If no final message received from stream
            UnexpectedModelBehavior: If the model fails or behaves unexpectedly
        """
        # Delegate to run_stream() for all execution.
        # SessionPool-managed agents (protocol servers) use SessionPool.run_stream()
        # directly, not this method.  run() is a convenience wrapper that collects
        # streaming events and returns the final message.
        final_message = None
        async for event in self.run_stream(  # type: ignore[attr-defined]
            *prompts,
            store_history=store_history,
            message_id=message_id,
            session_id=session_id,
            parent_session_id=parent_session_id,
            parent_id=parent_id,
            message_history=message_history,
            deps=deps,
            input_provider=input_provider,
            event_handlers=event_handlers,
            wait_for_connections=wait_for_connections,
            depth=depth,
        ):
            if isinstance(event, StreamCompleteEvent):
                final_message = event.message

        if final_message is None:
            raise RuntimeError("No final message received from stream")

        return final_message

    @abstractmethod
    async def get_available_models(self) -> list[ModelInfo] | None:
        """Get available models for this agent.

        Returns a list of models that can be used with this agent, or None
        if model discovery is not supported for this agent type.

        Uses tokonomics.ModelInfo which includes pricing, capabilities,
        and limits. Can be converted to protocol-specific formats (OpenCode, ACP).

        Returns:
            List of tokonomics ModelInfo, or None if not supported
        """
        ...

    @abstractmethod
    async def get_modes(self) -> list[ModeCategory]:
        """Get available mode categories for this agent.

        Returns a list of mode categories that can be switched. Each category
        represents a group of mutually exclusive modes (e.g., permissions,
        models, behavior presets).

        Different agent types expose different modes:
        - Native Agent: permissions + model selection
        - ACPAgent: Passthrough from remote server

        Returns:
            List of ModeCategory, empty list if no modes supported
        """
        ...

    @overload
    async def set_mode(self, mode: ModeInfo) -> None: ...

    @overload
    async def set_mode(self, mode: str, category_id: ModeCategoryId | str) -> None: ...

    async def set_mode(
        self, mode: ModeInfo | str, category_id: ModeCategoryId | str | None = None
    ) -> None:
        """Set a mode within a category.

        Args:
            mode: The mode to activate - either a ModeInfo object or mode ID string.
            category_id: Category ID. Required if mode is a string, optional if ModeInfo.
        """
        if isinstance(mode, ModeInfo):
            mode_id = mode.id
            resolved_category = category_id or mode.category_id
        else:
            mode_id = mode
            if not category_id:
                raise ValueError("category_id is required when mode is a string")
            resolved_category = category_id

        if not resolved_category:
            raise ValueError("category_id could not be determined from ModeInfo")

        await self._set_mode(mode_id, resolved_category)

    @abstractmethod
    async def _set_mode(self, mode_id: str, category_id: str) -> None:
        """Agent-specific mode switching implementation."""
        ...

    @abstractmethod
    async def list_sessions(
        self,
        *,
        cwd: str | None = None,
        limit: int | None = None,
    ) -> list[SessionData]:
        """List available sessions for this agent.

        Returns session information including session IDs, working directories,
        titles, and last update timestamps.

        Args:
            cwd: Filter sessions by working directory (optional)
            limit: Maximum number of sessions to return (optional)

        Returns:
            List of SessionData objects
        """
        ...

    @abstractmethod
    async def load_session(self, session_id: str) -> SessionData | None:
        """Load and restore a session by ID.

        Loads session data and restores the conversation history for the specified session.
        For agents that support session persistence, this switches the active session.

        Args:
            session_id: Unique identifier for the session to load

        Returns:
            SessionData if session was found and loaded, None otherwise
        """
        ...

    async def resume_session(self, session_id: str) -> SessionData | None:
        """Resume a session by ID without loading conversation history.

        Unlike load_session, this does NOT populate conversation.chat_messages.
        It restores the agent's internal state so the conversation can continue,
        but assumes the client already has the history (or doesn't need it).

        This is useful for:
        - Reconnecting after a disconnect
        - Automated workflows that don't need UI history
        - Faster session switching when history display isn't needed

        Default implementation calls load_session (subclasses may optimize).

        UNSTABLE: This feature is not part of the ACP spec yet.

        Args:
            session_id: Unique identifier for the session to resume

        Returns:
            SessionData if session was found and resumed, None otherwise
        """
        # Default: just delegate to load_session
        # Subclasses can override to skip history loading for efficiency
        return await self.load_session(session_id)

    async def load_rules(self, project_dir: str | None = None) -> None:
        """Load agent rules from global and project locations.

        Searches for AGENTS.md/CLAUDE.md files in:
        1. Global: ~/.config/wolfharness/AGENTS.md (user-wide rules)
        2. Project: {project_dir}/AGENTS.md (project-specific rules)

        Both are merged and staged for injection into the first prompt.
        Uses the agent's execution environment for filesystem access,
        making this work across local and remote (ACP) environments.

        Args:
            project_dir: Project directory to search for rules. Falls back to
                env.cwd if not provided.
        """
        from wolfharness_config.resolution import get_global_config_dir

        effective_project_dir = project_dir or self.env.cwd
        fs = self.env.get_fs()
        rules_file_names = ("AGENTS.md", "CLAUDE.md")
        rules_parts: list[str] = []
        # 1. Global rules from config directory
        global_dir = get_global_config_dir()
        for name in rules_file_names:
            path = global_dir / name
            if path.is_file():
                try:
                    content = path.read_text(encoding="utf-8")
                    rules_parts.append(f"## Global Rules\n\n{content}")
                    logger.debug("Loaded global rules", path=str(path))
                except OSError:
                    logger.exception("Failed to read global rules", path=str(path))
                break

        # 2. Project rules - use provided dir, env.cwd, or current directory
        if effective_project_dir is None:
            effective_project_dir = str(Path.cwd())

        if effective_project_dir:
            for name in rules_file_names:
                rules_path = f"{effective_project_dir}/{name}"
                try:
                    if fs:
                        # Use filesystem abstraction (works for local and ACP)
                        if await fs._exists(rules_path):
                            content_bytes = await fs._cat_file(rules_path)
                            content = content_bytes.decode("utf-8")
                            rules_parts.append(f"## Project Rules\n\n{content}")
                            logger.debug("Loaded project rules", path=rules_path)
                            break
                    else:
                        # Fallback to direct file access
                        local_path = Path(rules_path)
                        if local_path.is_file():
                            content = local_path.read_text(encoding="utf-8")
                            rules_parts.append(f"## Project Rules\n\n{content}")
                            logger.debug("Loaded project rules", path=rules_path)
                            break
                except (OSError, UnicodeDecodeError) as exc:
                    logger.debug("No project rules found", path=rules_path, error=str(exc))
                    break
                except (ConnectionError, RuntimeError, ValueError) as exc:
                    # Handles MCP/RequestError from remote filesystems (ACP mode)
                    logger.debug("No project rules found", path=rules_path, error=str(exc))

        # Stage combined rules for first prompt
        if rules_parts:
            combined = "\n\n".join(rules_parts)
            self.staged_content.add_text(combined)
            logger.info(
                "Staged agent rules",
                global_rules=bool(any("Global" in p for p in rules_parts)),
                project_rules=bool(any("Project" in p for p in rules_parts)),
            )
