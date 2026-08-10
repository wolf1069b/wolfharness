"""Base class for message processing nodes."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Self, overload
import warnings

from anyenv.signals import Signal

from wolfharness.log import get_logger
from wolfharness.messaging import ChatMessage
from wolfharness.talk import AggregatedTalkStats


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Coroutine
    from datetime import timedelta
    from types import TracebackType

    from evented.event_data import EventData
    from evented_config import EventConfig

    from wolfharness.agents.events import RichAgentStreamEvent
    from wolfharness.common_types import (
        AnyTransformFn,
        AsyncFilterFn,
        ProcessorCallback,
        QueueStrategy,
    )
    from wolfharness.delegation import AgentPool
    from wolfharness.host.context import HostContext
    from wolfharness.messaging.context import NodeContext
    from wolfharness.storage import StorageManager
    from wolfharness.talk import Talk, TeamTalk
    from wolfharness.talk.stats import AggregatedMessageStats, MessageStats
    from wolfharness.tools.base import FunctionTool
    from wolfharness.ui.base import InputProvider
    from wolfharness_config.forward_targets import ConnectionType
    from wolfharness_config.mcp_server import MCPServerConfig


logger = get_logger(__name__)

SourceType = Literal["agent", "team_parallel", "team_sequential"]
"""Type alias for source_type values used in event streaming.

Distinguishes the origin of a subagent event:
- ``"agent"``: a single agent (native, ACP, Claude Code, etc.)
- ``"team_parallel"``: a parallel team (``BaseTeam`` with ``mode="parallel"``)
- ``"team_sequential"``: a sequential team (``BaseTeam`` with ``mode="sequential"``)
"""


def get_source_type(node: MessageNode[Any, Any]) -> SourceType:
    """Return the source-type tag for *node* used in event streaming.

    Uses ``isinstance`` checks with **local** (deferred) imports so that
    calling this function never creates a circular import between
    ``messagenode`` ↔ ``base_team``.

    Args:
        node: Any :class:`MessageNode` instance.

    Returns:
        The corresponding :data:`SourceType` value.
    """
    from wolfharness.delegation.base_team import BaseTeam

    if isinstance(node, BaseTeam):
        return "team_parallel" if node.mode == "parallel" else "team_sequential"
    # Check it's at least a MessageNode — unknown subclasses get a warning
    if not isinstance(node, MessageNode):
        logger.warning("Unexpected node type %s, defaulting to 'agent'", type(node).__name__)
    return "agent"


class MessageNode[TDeps, TResult](ABC):
    """Base class for all message processing nodes."""

    message_received = Signal[ChatMessage[Any]]()
    """Signal emitted when node receives a message."""

    message_sent = Signal[ChatMessage[Any]]()
    """Signal emitted when node creates a message."""

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        display_name: str | None = None,
        mcp_servers: Sequence[str | MCPServerConfig] | None = None,
        agent_pool: AgentPool[Any] | None = None,
        enable_logging: bool = True,
        event_configs: Sequence[EventConfig] | None = None,
    ) -> None:
        """Initialize message node."""
        super().__init__()

        from wolfharness.mcp_server.manager import MCPManager
        from wolfharness.messaging import EventManager
        from wolfharness.messaging.connection_manager import ConnectionManager

        async def _event_handler(event: EventData) -> None:
            if prompt := event.to_prompt():
                await self.run(prompt)

        self._pending_tasks: set[asyncio.Task[Any]] = set()
        self._name = name or self.__class__.__name__
        self._display_name = display_name
        self.log = logger.bind(agent_name=self._name)
        self._agent_pool = agent_pool
        self.description = description
        self.connections = ConnectionManager(self)
        cfgs = list(event_configs) if event_configs else None
        self._events = EventManager(
            configs=cfgs,
            event_callbacks=[_event_handler],
            source_name=self._name,
        )
        name_ = f"node_{self._name}"
        # Share the pool's MCPManager when available to avoid duplicate
        # MCP subprocess spawning. The pool owns the lifecycle; agents
        # with a shared manager skip __aexit__ cleanup on it.
        # However, when the agent has its own MCP servers (agent-level),
        # create a dedicated MCPManager for them. Pool-level servers are
        # still accessible via agent_pool.mcp and are added separately by
        # the orchestrator via agent._external_capabilities.append().
        if agent_pool is not None and not mcp_servers:
            self._mcp_shared = True
            self.mcp = agent_pool.mcp
        else:
            self._mcp_shared = False
            self.mcp = MCPManager(name_, servers=mcp_servers, owner=self.name)
        self.enable_db_logging = enable_logging

    @property
    def agent_pool(self) -> AgentPool[Any] | None:
        """Compatibility shim: returns the pool reference.

        .. deprecated:: M2
            Use :attr:`host_context` instead. The ``agent_pool`` property
            will be removed in M3. ``host_context`` provides the same
            infrastructure fields via an immutable :class:`HostContext`.
        """
        warnings.warn(
            "agent_pool is deprecated, use host_context (HostContext) instead. "
            "See AGENTS.md for migration guide.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._agent_pool

    @agent_pool.setter
    def agent_pool(self, value: AgentPool[Any] | None) -> None:
        self._agent_pool = value

    def _bind_pool(self, pool: AgentPool[Any] | None) -> None:
        """Internal method to set the pool reference without DeprecationWarning.

        Used by Talk wiring and other internal code that needs to set
        the pool reference without going through the deprecated public
        property setter.
        """
        self._agent_pool = pool

    @property
    def host_context(self) -> HostContext | None:
        """Return HostContext from the pool, if available."""
        if self._agent_pool is not None:
            return self._agent_pool.get_context()
        return None

    async def log_session(
        self,
        session_id: str | None = None,
        initial_prompt: str | None = None,
        model: str | None = None,
        parent_session_id: str | None = None,
    ) -> None:
        """Log conversation to storage if enabled.

        Should be called at the start of run_stream() after session_id is set.
        For native agents, generate session_id first with uuid4().
        For wrapped agents (Claude Code), set session_id from SDK session first.

        Args:
            session_id: Optional session ID for the conversation.
            initial_prompt: Optional initial prompt to trigger title generation.
            model: Requested model identifier for this session.
            parent_session_id: Optional parent session ID.
        """
        if self.enable_db_logging and self.storage and session_id:
            await self.storage.log_session(
                session_id=session_id,
                node_name=self.name,
                model=model,
                initial_prompt=initial_prompt,
                parent_session_id=parent_session_id,
            )

    async def emit_agent_event(
        self, event: RichAgentStreamEvent[Any], source_session_id: str | None = None
    ) -> None:
        """Emit an agent stream event via the event manager.

        Args:
            event: The agent stream event to emit
            source_session_id: Optional ID of the session that produced the event
        """
        await self._events.emit_agent_event(event, source_session_id=source_session_id)

    async def __aenter__(self) -> Self:
        """Initialize base message node."""
        try:
            await self._events.__aenter__()
            if not self._mcp_shared:
                await self.mcp.__aenter__()
        except Exception as e:
            await self.__aexit__(type(e), e, e.__traceback__)

            raise RuntimeError(f"Failed to initialize {self.name}") from e
        else:
            return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up base resources."""
        await self._events.__aexit__(exc_type, exc_val, exc_tb)
        if not self._mcp_shared:
            await self.mcp.__aexit__(exc_type, exc_val, exc_tb)
        # Cancel and wait for all pending background tasks, then clear.
        if self._pending_tasks:
            for task in self._pending_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
        self._pending_tasks.clear()

    def spawn_task(
        self,
        coro_fn: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
    ) -> asyncio.Task[Any]:
        """Schedule a coroutine for background execution on this node.

        The created task is tracked in ``_pending_tasks`` and automatically
        removed when it completes. All pending tasks are awaited in
        ``__aexit__``.

        Args:
            coro_fn: Coroutine function to call.
            *args: Positional arguments passed to ``coro_fn``.

        Returns:
            The created :class:`asyncio.Task`.
        """
        task = asyncio.create_task(coro_fn(*args))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        return task

    def fire_and_forget(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Run a coroutine in the background without waiting for result.

        The coroutine is wrapped in a try/except that logs and swallows
        exceptions, preventing one non-critical task failure from
        propagating.

        Args:
            coro: Coroutine to run in the background.
        """

        async def _safe_run() -> None:
            try:
                await coro
            except Exception:
                self.log.exception("fire_and_forget task failed")

        task = asyncio.create_task(_safe_run())
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    @property
    def task_manager(self) -> _TaskManagerShim:
        """Deprecated: backward-compat shim for the old TaskManager API.

        .. deprecated::
            Use :meth:`spawn_task` or :attr:`_pending_tasks` directly.

        Returns:
            A shim object that provides the old TaskManager method surface
            (``complete_tasks``, ``cleanup_tasks``, ``fire_and_forget``,
            ``create_task``, ``_pending_tasks``) backed by this node's
            ``_pending_tasks`` set.
        """
        warnings.warn(
            "task_manager is deprecated. Use spawn_task() or _pending_tasks instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _TaskManagerShim(self)

    @property
    def connection_stats(self) -> AggregatedTalkStats:
        """Get stats for all active connections of this node."""
        stats = [talk.stats for talk in self.connections.get_connections()]
        return AggregatedTalkStats(stats=stats)

    def get_context(
        self,
        data: Any = None,
        input_provider: InputProvider | None = None,
    ) -> NodeContext:
        """Create a new context for this node.

        Args:
            data: Optional custom data to attach to the context
            input_provider: Optional input provider override

        Returns:
            A new NodeContext instance
        """
        raise NotImplementedError

    @property
    def storage(self) -> StorageManager | None:
        """Get storage manager from pool."""
        return self._agent_pool.storage if self._agent_pool else None

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._name

    @property
    def display_name(self) -> str:
        """Get human-readable display name, falls back to name."""
        return self._display_name or self._name

    @property
    def agent_type(self) -> str:
        """Return the agent-type string used for persistence.

        This is the *persistence-domain* identifier (stored in
        ``SessionData.agent_type``).  It differs from :data:`SourceType`
        which is the *event-domain* identifier used during streaming.

        Subclasses may override this to provide a more specific type
        string (e.g. ``"native"``, ``"acp"``, ``"claude_code"``).

        Returns:
            A string identifying the agent type for storage purposes.
        """
        return get_source_type(self)

    def to_tool(
        self, *, name: str | None = None, description: str | None = None, **kwargs: Any
    ) -> FunctionTool[TResult]:
        """Convert node to a callable tool.

        Args:
            name: Optional tool name override
            description: Optional tool description override
            **kwargs: Additional arguments for subclass customization

        Returns:
            Tool instance that can be registered
        """
        from wolfharness.tools.base import FunctionTool

        async def wrapped(prompt: str) -> TResult:
            result = await self.run(prompt)
            return result.content

        tool_name = name or f"ask_{self.name}"
        docstring = description or f"Get expert answer from {self.name}"
        if self.description:
            docstring = f"{docstring}\n\n{self.description}"
        wrapped.__doc__ = docstring
        wrapped.__name__ = tool_name
        return FunctionTool.from_callable(wrapped)

    @overload
    def __rshift__(
        self, other: MessageNode[Any, Any] | ProcessorCallback[Any]
    ) -> Talk[TResult]: ...

    @overload
    def __rshift__(
        self, other: Sequence[MessageNode[Any, Any] | ProcessorCallback[Any]]
    ) -> TeamTalk[TResult]: ...

    def __rshift__(
        self,
        other: MessageNode[Any, Any]
        | ProcessorCallback[Any]
        | Sequence[MessageNode[Any, Any] | ProcessorCallback[Any]],
    ) -> Talk[Any] | TeamTalk[Any]:
        """Connect agent to another agent or group.

        Example:
            agent >> other_agent  # Connect to single agent
            agent >> (agent2 & agent3)  # Connect to group
            agent >> "other_agent"  # Connect by name (needs pool)
        """
        return self.connect_to(other)

    @overload
    def connect_to(
        self,
        target: MessageNode[Any, Any] | ProcessorCallback[Any],
        *,
        queued: Literal[True],
        queue_strategy: Literal["concat"],
    ) -> Talk[str]: ...

    @overload
    def connect_to(
        self,
        target: MessageNode[Any, Any] | ProcessorCallback[Any],
        *,
        connection_type: ConnectionType = "run",
        name: str | None = None,
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: QueueStrategy = "latest",
        transform: AnyTransformFn[Any] | None = None,
        filter_condition: AsyncFilterFn | None = None,
        stop_condition: AsyncFilterFn | None = None,
        exit_condition: AsyncFilterFn | None = None,
    ) -> Talk[TResult]: ...

    @overload
    def connect_to(
        self,
        target: Sequence[MessageNode[Any, Any] | ProcessorCallback[Any]],
        *,
        queued: Literal[True],
        queue_strategy: Literal["concat"],
    ) -> TeamTalk[str]: ...

    @overload
    def connect_to(
        self,
        target: Sequence[MessageNode[Any, TResult] | ProcessorCallback[TResult]],
        *,
        connection_type: ConnectionType = "run",
        name: str | None = None,
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: QueueStrategy = "latest",
        transform: AnyTransformFn[Any] | None = None,
        filter_condition: AsyncFilterFn | None = None,
        stop_condition: AsyncFilterFn | None = None,
        exit_condition: AsyncFilterFn | None = None,
    ) -> TeamTalk[TResult]: ...

    @overload
    def connect_to(
        self,
        target: Sequence[MessageNode[Any, Any] | ProcessorCallback[Any]],
        *,
        connection_type: ConnectionType = "run",
        name: str | None = None,
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: QueueStrategy = "latest",
        transform: AnyTransformFn[Any] | None = None,
        filter_condition: AsyncFilterFn | None = None,
        stop_condition: AsyncFilterFn | None = None,
        exit_condition: AsyncFilterFn | None = None,
    ) -> TeamTalk: ...

    def connect_to(
        self,
        target: MessageNode[Any, Any]
        | ProcessorCallback[Any]
        | Sequence[MessageNode[Any, Any] | ProcessorCallback[Any]],
        *,
        connection_type: ConnectionType = "run",
        name: str | None = None,
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: QueueStrategy = "latest",
        transform: AnyTransformFn[Any] | None = None,
        filter_condition: AsyncFilterFn | None = None,
        stop_condition: AsyncFilterFn | None = None,
        exit_condition: AsyncFilterFn | None = None,
    ) -> Talk[Any] | TeamTalk:
        """Create connection(s) to target(s)."""
        # Handle callable case
        from wolfharness.agents import Agent
        from wolfharness.delegation.base_team import BaseTeam

        if callable(target):
            target = Agent.from_callback(target)
            if pool := self._agent_pool:
                target._bind_pool(pool)
        # we are explicit here just to make disctinction clear, we only want sequences
        # of message units
        if isinstance(target, Sequence) and not isinstance(target, BaseTeam):
            targets: list[MessageNode[Any, Any]] = []
            for t in target:
                match t:
                    case _ if callable(t):
                        other = Agent.from_callback(t)
                        if pool := self._agent_pool:
                            other._bind_pool(pool)
                        targets.append(other)
                    case MessageNode():
                        targets.append(t)
                    case _:
                        raise TypeError(f"Invalid node type: {type(t)}")
        else:
            targets = target  # type: ignore[assignment]
        return self.connections.create_connection(
            self,
            targets,
            connection_type=connection_type,
            priority=priority,
            name=name,
            delay=delay,
            queued=queued,
            queue_strategy=queue_strategy,
            transform=transform,
            filter_condition=filter_condition,
            stop_condition=stop_condition,
            exit_condition=exit_condition,
        )

    async def disconnect_all(self) -> None:
        """Disconnect from all nodes."""
        for target in list(self.connections.get_targets()):
            self.stop_passing_results_to(target)

    def stop_passing_results_to(self, other: MessageNode[Any, Any]) -> None:
        """Stop forwarding results to another node."""
        self.connections.disconnect(other)

    def _get_deps(self) -> TDeps | None:
        """Return dependencies for graph execution.

        Returns:
            The node's dependencies, or None if not configured.
        """
        return None

    def _build_single_node_graph(
        self,
    ) -> Any:
        """Build a single-node pydantic-graph wrapping this node.

        Returns:
            An immutable Graph ready for execution.
        """
        from wolfharness.messaging.graph_adapter import MessageNodeStep

        return MessageNodeStep(self).build_single_node_graph()

    async def _execute_node(self, *prompts: Any, **kwargs: Any) -> ChatMessage[TResult]:
        """Core execution logic without graph wrapping.

        Subclasses that do not override :meth:`run` must implement this
        method. Subclasses that override :meth:`run` (the default for
        all existing agent types) do not need to implement this method.

        Args:
            *prompts: Input prompts.
            **kwargs: Additional execution arguments.

        Returns:
            The resulting ChatMessage.

        Raises:
            NotImplementedError: If neither ``run()`` nor ``_execute_node()``
                is overridden.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _execute_node() or override run() directly."
        )

    async def run(self, *prompts: Any, **kwargs: Any) -> ChatMessage[TResult]:
        """Execute node with prompts via pydantic-graph single-node graph.

        Builds a single-node graph and runs it to completion, wrapping the
        graph run with :class:`SignalEmittingGraphRun` so that
        ``message_received`` and ``message_sent`` signals are emitted at
        step boundaries. Subclasses may override this method to provide
        custom execution logic; in that case the graph-based path is
        bypassed.

        Args:
            *prompts: Input prompts.
            **kwargs: Additional execution arguments.

        Returns:
            The resulting ChatMessage.
        """
        from pydantic_graph.id_types import NodeID

        from wolfharness.messaging.graph_adapter import AgentPoolState
        from wolfharness.messaging.signal_adapter import SignalEmittingGraphRun

        graph = self._build_single_node_graph()
        state = AgentPoolState(node=self, prompts=prompts, kwargs=kwargs)
        node_mapping: dict[NodeID, MessageNode[Any, Any]] = {NodeID(self.name): self}
        async with graph.iter(state=state, deps=self._get_deps(), inputs=None) as graph_run:
            signal_run: SignalEmittingGraphRun[Any, Any, Any] = SignalEmittingGraphRun(
                graph_run, node_mapping=node_mapping
            )
            async for _ in signal_run:
                pass
        return state.result  # type: ignore[return-value]  # ty: ignore[invalid-return-type]

    async def run_stream(
        self,
        *prompts: Any,
        **kwargs: Any,
    ) -> AsyncIterator[RichAgentStreamEvent[TResult]]:
        """Run with streaming output via pydantic-graph Graph.iter().

        Uses :meth:`Graph.iter` to drive execution step-by-step, wrapping
        the graph run with :class:`SignalEmittingGraphRun` so that
        ``message_received`` and ``message_sent`` signals are emitted at
        step boundaries. For nodes that do not override :meth:`run_stream`
        (e.g. most agent subclasses), this yields the final result wrapped
        in a :class:`StreamCompleteEvent`. Agent subclasses typically
        override this with rich event streaming.

        Args:
            *prompts: Input prompts.
            **kwargs: Additional execution arguments.

        Yields:
            RichAgentStreamEvent tokens during execution.
        """
        from pydantic_graph.id_types import NodeID

        from wolfharness.agents.events import RunErrorEvent, StreamCompleteEvent
        from wolfharness.messaging.graph_adapter import AgentPoolState
        from wolfharness.messaging.signal_adapter import SignalEmittingGraphRun

        graph = self._build_single_node_graph()
        state = AgentPoolState(node=self, prompts=prompts, kwargs=kwargs)
        node_mapping: dict[NodeID, MessageNode[Any, Any]] = {NodeID(self.name): self}

        async with graph.iter(state=state, deps=self._get_deps(), inputs=None) as graph_run:
            signal_run: SignalEmittingGraphRun[Any, Any, Any] = SignalEmittingGraphRun(
                graph_run, node_mapping=node_mapping
            )
            async for _ in signal_run:
                # Generic nodes do not produce intermediate stream events;
                # drain the event queue in case a subclass pushed events.
                while not state.event_queue.empty():
                    try:
                        event = state.event_queue.get_nowait()
                        yield event
                        if isinstance(event, StreamCompleteEvent | RunErrorEvent):
                            return
                    except asyncio.QueueEmpty:
                        break

        # Yield any remaining events after graph completion
        while not state.event_queue.empty():
            try:
                event = state.event_queue.get_nowait()
                yield event
                if isinstance(event, StreamCompleteEvent | RunErrorEvent):
                    return
            except asyncio.QueueEmpty:
                break

        # Yield the final result wrapped in StreamCompleteEvent
        if state.result is not None:
            yield StreamCompleteEvent(message=state.result)

    async def run_message(
        self,
        message: ChatMessage[Any],
        **kwargs: Any,
    ) -> ChatMessage[TResult]:
        """Run with an incoming ChatMessage (e.g., from Talk routing).

        Extracts content from the message, preserves session_id,
        and sets parent_id to track the message chain.

        Args:
            message: The incoming ChatMessage to process
            **kwargs: Additional arguments passed to run()

        Returns:
            Response ChatMessage with message chain tracked via parent_id
        """
        return await self.run(
            message.content,
            session_id=message.session_id,
            parent_id=message.message_id,
            **kwargs,
        )

    async def get_message_history(
        self, session_id: str | None = None, limit: int | None = None
    ) -> list[ChatMessage[Any]]:
        """Get message history from storage.

        Args:
            session_id: Optional session ID to query history for.
            limit: Maximum number of messages to return.

        Returns:
            List of chat messages from the session.
        """
        from wolfharness_config.session import SessionQuery

        if not self.enable_db_logging or not self.storage or not session_id:
            return []
        query = SessionQuery(name=session_id, limit=limit)
        return await self.storage.filter_messages(query)

    async def log_message(self, message: ChatMessage[Any]) -> None:
        """Handle message from chat signal."""
        if self.enable_db_logging and self.storage:
            await self.storage.log_message(message)

    @abstractmethod
    async def get_stats(self) -> MessageStats | AggregatedMessageStats:
        """Get message statistics for this node."""

    @abstractmethod
    def run_iter(self, *prompts: Any, **kwargs: Any) -> AsyncIterator[ChatMessage[Any]]:
        """Yield messages during execution."""


class _TaskManagerShim:
    """Backward-compat shim providing the old ``TaskManager`` API surface.

    Wraps a :class:`MessageNode`'s ``_pending_tasks`` set and delegates
    the deprecated TaskManager methods to the new asyncio-based patterns.
    """

    def __init__(self, node: MessageNode[Any, Any]) -> None:
        self._node = node

    @property
    def _pending_tasks(self) -> set[asyncio.Task[Any]]:
        """Direct access to the node's pending task set."""
        return self._node._pending_tasks

    def fire_and_forget(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Run coroutine without waiting for result."""
        self._node.fire_and_forget(coro)

    def create_task(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        name: str | None = None,
        priority: int = 0,
        delay: timedelta | None = None,
    ) -> asyncio.Task[Any]:
        """Create and track a new task.

        Args:
            coro: Coroutine to run.
            name: Optional name for the task.
            priority: Ignored (kept for API compat).
            delay: Ignored (kept for API compat).
        """
        del priority, delay  # unused, accepted for API compat

        async def _runner() -> Any:
            return await coro

        task = asyncio.create_task(_runner(), name=name) if name else asyncio.create_task(_runner())
        self._node._pending_tasks.add(task)
        task.add_done_callback(self._node._pending_tasks.discard)
        return task

    async def cleanup_tasks(self) -> None:
        """Wait for all pending tasks to complete."""
        if self._node._pending_tasks:
            await asyncio.gather(*self._node._pending_tasks, return_exceptions=True)
        self._node._pending_tasks.clear()

    async def complete_tasks(self, cancel: bool = False) -> None:
        """Wait for all pending tasks to complete."""
        if cancel:
            for task in self._node._pending_tasks:
                task.cancel()
        if self._node._pending_tasks:
            await asyncio.wait(self._node._pending_tasks)

    def is_busy(self) -> bool:
        """Check if there are any tasks pending."""
        return bool(self._node._pending_tasks)
