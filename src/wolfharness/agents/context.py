"""Runtime context models for Agents."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING, Any, Literal
import uuid

import anyio
from mcp.types import (
    ElicitRequestFormParams,
    ElicitRequestParams,
    ElicitRequestURLParams,
    ElicitResult,
    ErrorData,
)

from wolfharness.agents.prompt_injection import PromptInjectionManager
from wolfharness.log import get_logger
from wolfharness.messaging.context import NodeContext
from wolfharness.tools import CallDeferred
from wolfharness.ui.elicitation import normalize_elicit_content


if TYPE_CHECKING:
    from upathtools.filesystems import IsolatedMemoryFileSystem, OverlayFileSystem

    from wolfharness import Agent
    from wolfharness.agents.events import StreamEventEmitter
    from wolfharness.agents.native_agent.checkpoint import CheckpointManager
    from wolfharness.agents.native_agent.elicitation_bridge import ElicitationFutureRegistry
    from wolfharness.orchestrator.core import EventBus
    from wolfharness.orchestrator.run import RunHandle
    from wolfharness.tools.base import Tool


ConfirmationResult = Literal["allow", "skip", "abort_run", "abort_chain"]

logger = get_logger(__name__)


class _DeprecatedField:
    """Data descriptor that warns when a deprecated dataclass field is accessed."""

    def __init__(self, *, default_factory: Any, msg: str) -> None:
        self.default_factory = default_factory
        self.msg = msg

    def __get__(self, obj: Any, objtype: type[Any] | None = None) -> Any:
        if obj is None:
            return self
        value = obj.__dict__.get("session_id")
        if value is None:
            value = self.default_factory()
            obj.__dict__["session_id"] = value
        logger.warning(self.msg)
        return value

    def __set__(self, obj: Any, value: Any) -> None:
        logger.warning(self.msg)
        obj.__dict__["session_id"] = value


MAX_SUBAGENT_DEPTH: int = 5
"""Maximum nesting depth for subagent delegations."""


class SubagentDepthError(Exception):
    """Raised when subagent nesting exceeds MAX_SUBAGENT_DEPTH."""


@dataclass(kw_only=True)
class AgentRunContext:
    """Per-execution isolated state container for agent runs.

    This dataclass holds all state that is specific to a single run execution,
    ensuring isolation between concurrent runs. It is separate from AgentContext
    which is the PydanticAI context passed to tools.

    Attributes:
        cancelled: Whether the run has been cancelled.
        current_task: The asyncio.Task for the current run, if any.
        depth: Current delegation depth (0 = top-level run).
        event_bus: Optional event bus for cross-session event routing.
        injection_manager: Manages prompt injection and queuing for this run.
        session_id: Session ID for this run.
        deps: Optional dependencies passed to the run.
        start_time: Timestamp when the run started (for metrics).
    """

    cancelled: bool = False
    """Whether the run has been cancelled."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    """Unique identifier for this run."""

    current_task: asyncio.Task[Any] | None = None
    """The asyncio.Task for the current run, if any."""

    depth: int = 0
    """Current delegation depth (0 = top-level run)."""

    event_bus: EventBus | None = None
    """Optional event bus for cross-session event routing."""

    injection_manager: PromptInjectionManager = field(default_factory=PromptInjectionManager)
    """Manages prompt injection and queuing for this run."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    """Session ID for this run."""

    deps: Any = None
    """Optional dependencies passed to the run."""

    start_time: float = field(default_factory=time.perf_counter)
    """Timestamp when the run started (for metrics)."""

    completed: bool = False
    """Whether the run has completed (stream finished)."""

    terminal_tool_result: Any = None
    """Result returned by a terminal tool such as `attempt_completion`."""

    terminal_tool_name: str | None = None
    """Name of the terminal tool that completed the run."""

    checkpointed: bool = False
    """Whether the run has been checkpointed (deferred tools pending)."""

    elicitation_registry: ElicitationFutureRegistry | None = None
    """Per-session registry of pending elicitation futures.

    Set by ``get_agentlet()`` when the elicitation bridge capability is
    created. Used by ``resume_session()`` to resolve futures for in-process
    resume (the agent run is still alive but paused on elicitation).
    """

    cached_elicitation_responses: dict[str, ElicitResult] = field(default_factory=dict)
    """Cached elicitation responses for crash recovery.

    Maps ``tool_call_id`` to a pre-built ``ElicitResult``. When
    ``handle_elicitation()`` is called during re-execution, it checks
    this dict first and returns the cached response instead of deferring
    again. Populated by ``_resume_native_agent()`` from
    ``ElicitationResumePayload`` data.
    """

    checkpoint_manager: CheckpointManager | None = None
    """Checkpoint manager for durable elicitation.

    Set by ``get_agentlet()`` when the elicitation bridge capability is
    created. Used by ``handle_elicitation()`` to checkpoint the session
    before awaiting an elicitation future, enabling crash recovery.
    """

    elicitation_timeout: float | None = None
    """Timeout in seconds for elicitation responses.

    Set from agent config (``BaseAgentConfig.elicitation_timeout``) by
    ``get_agentlet()``. ``None`` means no timeout (infinite wait, the
    default). Configure explicitly in YAML to enable a timeout. Used
    by ``handle_elicitation()`` for ``asyncio.wait_for()``.
    """

    _run_handle: RunHandle | None = None
    """Run handle for this execution, set by RunHandle lifecycle."""

    child_done_events: dict[str, anyio.Event] = field(default_factory=dict)
    """Per-child-session done events for tracking subagent completion."""

    queued_steer_messages: list[str | list[Any]] = field(default_factory=list)
    """Steer messages queued during ``start()`` ``feedback_queue`` drain.

    Populated by ``RunHandle.start()`` when draining ``session.feedback_queue``
    (before ``active_agent_run`` is set). Drained by
    ``RunHandle.drain_queued_steer_messages()`` which is called by
    ``NativeTurn.execute()`` after setting ``active_agent_run``.
    """

    turn_id: str | None = None
    """Unique identifier for the current Turn, set by RunHandle.start().

    Generated as ``str(uuid.uuid4())`` before ``agent.create_turn()`` so
    that all events, journal entries, and snapshots within a single Turn
    share the same ``turn_id`` for idempotent crash recovery.
    """

    opencode_message_id: str | None = None
    """OpenCode message ID for the current user message.

    Set by the OpenCode server when routing a user message. This is the
    same ID space as ``request.message_id`` in the revert endpoint, so
    ``FileOpsTracker.record_change()`` uses this field (with ``turn_id``
    as fallback) as the ``message_id`` for file change tracking. Without
    this, file rollback would silently fail because ``turn_id`` (UUID)
    and the OpenCode message ID are different ID spaces.
    """

    _pending_enqueue_message_ids: list[str] = field(default_factory=list)
    """FIFO queue of message_ids from ``steer()``/``followup()`` calls.

    When ``steer()`` or ``followup()`` enqueues a message to
    ``agent_run``, the ``message_id`` is appended here BEFORE the
    ``enqueue()`` call. When ``EnqueuedMessagesEvent`` fires and
    ``EventMapper.handle_enqueued_messages()`` runs, it pops from this
    queue to reuse the same ``message_id`` instead of generating a new
    UUID. This ensures the fire-and-forget ``UserMessageInsertedEvent``
    (emitted by ``_schedule_user_message_emission()``) and the
    ``EnqueuedMessagesEvent``-derived event share the same
    ``message_id``, enabling converter-level dedup.
    """

    async def complete_background_task(self, child_session_id: str, message: str) -> None:
        """Signal that a background child task has completed.

        Pops and sets the done_event for the child session. Steer
        injection is handled separately by ``SessionPool.steer_from_background_task()``.
        """
        event = self.child_done_events.pop(child_session_id, None)
        if event is not None:
            event.set()


@dataclass(kw_only=True)
class AgentContext[TDeps = Any](NodeContext[TDeps]):
    """Runtime context for agent execution.

    Generically typed with AgentContext[Type of Dependencies]
    """

    tool_name: str | None = None
    """Name of the currently executing tool."""

    tool_call_id: str | None = None
    """ID of the current tool call."""

    tool_input: dict[str, Any] = field(default_factory=dict)
    """Input arguments for the current tool call."""

    model_name: str | None = None
    """Model name in provider:model format (e.g., 'anthropic:claude-haiku-4-5')."""

    run_ctx: AgentRunContext | None = None
    """Reference to the per-run context for accessing run-isolated state."""

    _pending_elicitation_deferral: dict[str, Any] | None = None
    """Side-channel for durable elicitation (MCP tools only).

    When ``handle_elicitation`` is called inside an MCP callback and
    raises ``CallDeferred``, the MCP ``elicitation_handler`` catches it
    and stores the elicitation params here. ``MCPClient.call_tool``
    checks this attribute after the MCP call returns and re-raises
    ``CallDeferred`` so the run can be checkpointed and resumed.
    """

    in_mcp_callback: bool = False
    """Whether this context is inside an MCP callback.

    Set to ``True`` by ``MCPClient.call_tool()`` before invoking the
    MCP tool. When ``True``, ``handle_elicitation()`` raises
    ``CallDeferred`` (FastMCP callback wrapper catches exceptions,
    so awaiting a future is not possible). When ``False``,
    ``handle_elicitation()`` awaits a future directly (local tools
    can suspend without ending the agent run).
    """

    @property
    def native_agent(self) -> Agent[TDeps, Any]:
        """Current agent, type-narrowed to native pydantic-ai Agent."""
        from wolfharness import Agent

        assert isinstance(self.node, Agent)
        return self.node  # ty: ignore[invalid-return-type]

    async def handle_elicitation(self, params: ElicitRequestParams) -> ElicitResult | ErrorData:  # noqa: PLR0911, PLR0915
        """Handle elicitation request for additional information.

        Three paths based on context:

        1. **Crash recovery**: If ``run_ctx.cached_elicitation_responses``
           has a cached response for this ``tool_call_id``, return it
           immediately. This allows re-executed tools to complete without
           deferring again.

        2. **MCP tools** (``in_mcp_callback=True``): Raises
           ``CallDeferred`` directly. FastMCP's callback wrapper catches
           exceptions, so the MCP ``elicitation_handler`` in
           ``MCPClient.call_tool()`` catches this and converts to a
           side-channel + sentinel. ``call_tool()`` then re-raises
           ``CallDeferred`` after the MCP call returns. The run ends
           with ``DeferredToolRequests`` and resumes via crash recovery.

        3. **Local tools** (``in_mcp_callback=False``): Checkpoints the
           session, emits ``ElicitationDeferredEvent``, registers a future
           in ``ElicitationFutureRegistry``, and **awaits the future**.
           The agent run suspends (not ends) at the ``await`` point.
           When the user responds, ``resume_session()`` resolves the
           future, ``handle_elicitation()`` returns the response, the
           tool function continues naturally, and the agent run resumes.
           No re-execution, no ``CallDeferred``, no ``DeferredToolRequests``.

        When durability is not supported, the provider's ``get_elicitation``
        is called directly (existing behavior).
        """
        # Path 1: Crash recovery — return cached response.
        if (
            self.run_ctx is not None
            and self.tool_call_id is not None
            and self.tool_call_id in self.run_ctx.cached_elicitation_responses
        ):
            logger.info(
                "handle_elicitation: Path 1 (crash recovery cached response)",
                tool_call_id=self.tool_call_id,
            )
            return self.run_ctx.cached_elicitation_responses[self.tool_call_id]

        provider = self.get_input_provider()
        if not provider.supports_durable_elicitation:
            logger.info(
                "handle_elicitation: non-durable path (provider.get_elicitation)",
                tool_call_id=self.tool_call_id,
            )
            return await provider.get_elicitation(params)

        # Build elicitation params dict (used in both MCP and local paths).
        match params:
            case ElicitRequestFormParams():
                elicitation_params: dict[str, Any] = {
                    "message": params.message,
                    "requestedSchema": params.requestedSchema,
                    "mode": params.mode,
                }
            case ElicitRequestURLParams():
                elicitation_params = {
                    "message": params.message,
                    "url": params.url,
                    "elicitationId": params.elicitationId,
                    "mode": params.mode,
                }
            case _:
                elicitation_params = {
                    "message": str(params),
                    "mode": None,
                }

        # Path 2: MCP tools — raise CallDeferred (FastMCP can't await).
        if self.in_mcp_callback:
            logger.info(
                "handle_elicitation: Path 2 (MCP CallDeferred)",
                tool_call_id=self.tool_call_id,
                tool_name=self.tool_name,
            )
            raise CallDeferred(
                metadata={
                    "elicitation": elicitation_params,
                    "deferred_kind": "elicitation",
                }
            )

        # Path 3: Local tools — checkpoint, emit event, await future.
        # The agent run suspends here without ending. When the future
        # resolves, handle_elicitation() returns and the tool continues.
        logger.info(
            "handle_elicitation: Path 3 (local tool, await future)",
            tool_call_id=self.tool_call_id,
            tool_name=self.tool_name,
        )
        run_ctx = self.run_ctx
        if run_ctx is None:
            # No run context — fall back to synchronous path.
            return await provider.get_elicitation(params)

        registry = run_ctx.elicitation_registry
        if registry is None:
            # No registry — fall back to synchronous path.
            return await provider.get_elicitation(params)

        handle = self.tool_call_id or run_ctx.run_id

        # Checkpoint the session for crash recovery.
        if run_ctx.checkpoint_manager is not None:
            try:
                from wolfharness.agents.events.events import ElicitationDeferredEvent
                from wolfharness.sessions.models import PendingDeferredCall

                pending_call = PendingDeferredCall(
                    tool_call_id=handle,
                    tool_name=self.tool_name or "",
                    deferred_kind="elicitation",
                    deferred_strategy="block",
                    elicitation_message=elicitation_params.get("message"),
                    elicitation_schema=elicitation_params.get("requestedSchema"),  # ty: ignore[invalid-argument-type]
                    elicitation_mode=elicitation_params.get("mode"),
                )
                # Emit event to EventBus for protocol converters.
                if run_ctx.event_bus is not None:
                    event = ElicitationDeferredEvent(
                        deferred_handle=handle,
                        message=elicitation_params.get("message", ""),
                        requested_schema=elicitation_params.get("requestedSchema", {}),  # ty: ignore[invalid-argument-type]
                        mode=elicitation_params.get("mode", "form"),
                        session_id=run_ctx.session_id,
                        timeout_seconds=run_ctx.elicitation_timeout,
                    )
                    await run_ctx.event_bus.publish(run_ctx.session_id, event)
                # Get message history for checkpoint from the active agent run.
                # active_agent_run is the authoritative pydantic-ai AgentRun
                # object, always available during tool execution (set in
                # NativeTurn.execute() inside the `async with agentlet.iter()` block).
                checkpoint_messages: list[Any] = []
                if run_ctx._run_handle is not None:
                    agent_run = run_ctx._run_handle.active_agent_run
                    if agent_run is not None:
                        checkpoint_messages = agent_run.all_messages()
                await run_ctx.checkpoint_manager.checkpoint(
                    session_id=run_ctx.session_id,
                    message_history=checkpoint_messages,
                    pending_calls=[pending_call],
                    agent_config_hash="",
                )
                run_ctx.checkpointed = True
                # Update session store status to "checkpointed" so
                # resume_session() can find it without relying on the
                # allow_active_run workaround. Also set pending_deferred_calls
                # so resume_session() knows which calls need resolution.
                pool = self.node.host_context
                if pool is not None and pool.session_pool is not None:
                    store = pool.session_pool.sessions.store
                    if store is not None:
                        try:
                            data = await store.load_session(run_ctx.session_id)
                            if data is not None and data.status == "active":
                                data = data.model_copy(
                                    update={
                                        "status": "checkpointed",
                                        "pending_deferred_calls": [pending_call],
                                    }
                                )
                                data.touch()
                                await store.save_session(data)
                        except Exception:
                            logger.debug(
                                "Failed to update session status to checkpointed",
                                session_id=run_ctx.session_id,
                                exc_info=True,
                            )
            except Exception:
                # Checkpoint failed — the in-process future await still
                # works, but crash recovery won't be available for this
                # elicitation. Log prominently so operators know durability
                # is degraded.
                logger.warning(
                    "Checkpoint failed for durable elicitation — "
                    "crash recovery unavailable for this call",
                    session_id=run_ctx.session_id,
                    tool_call_id=handle,
                    exc_info=True,
                )
        # Register future and await — agent run suspends here.
        timeout = run_ctx.elicitation_timeout
        future = registry.register(handle)

        # Broadcast the question to the OpenCode TUI so the user can
        # see and answer it. Without this, the TUI never shows the
        # question and the future times out.
        try:
            await provider.broadcast_elicitation_question(handle, params, shared_future=future)
        except Exception:
            logger.warning(
                "Failed to broadcast elicitation question",
                session_id=run_ctx.session_id,
                tool_call_id=handle,
                exc_info=True,
            )

        try:
            if timeout is not None:
                payload = await asyncio.wait_for(future, timeout=timeout)
            else:
                payload = await future
        except TimeoutError:
            from wolfharness.tasks.exceptions import RunAbortedError

            logger.warning(
                "Elicitation timed out — aborting agent run",
                session_id=run_ctx.session_id,
                tool_call_id=handle,
                timeout_seconds=timeout,
            )
            raise RunAbortedError(
                f"Elicitation timed out after {timeout}s"
                if timeout is not None
                else "Elicitation timed out"
            ) from None
        finally:
            try:
                registry.remove(handle)
            except Exception:
                logger.exception("Failed to remove elicitation handle from registry")
            try:
                provider.cleanup_elicitation_question(handle)
            except Exception:
                logger.exception("Failed to cleanup elicitation question")

        # Convert payload to ElicitResult.
        # payload can be either:
        # - ElicitationResumePayload (from registry.resolve, used by
        #   in-process resume and crash recovery)
        # - list[list[str]] (from provider.resolve_question, used by
        #   the OpenCode TUI REST endpoint)
        from mcp.types import ElicitResult as MCPElicitResult

        if isinstance(payload, list):
            schema = elicitation_params.get("requestedSchema", {})
            if (
                isinstance(schema, dict)
                and schema.get("type") == "object"
                and "properties" in schema
            ):
                props = schema["properties"]
                prop_keys = list(props.keys())
                content: dict[str, Any] = {}
                for i, key in enumerate(prop_keys[: len(payload)]):
                    answer_list = payload[i] if i < len(payload) else []
                    prop_schema = props.get(key, {})
                    if isinstance(prop_schema, dict) and (
                        prop_schema.get("type") == "array" or "items" in prop_schema
                    ):
                        content[key] = (
                            answer_list
                            if isinstance(answer_list, list)
                            else [answer_list]
                            if answer_list
                            else []
                        )
                    else:
                        content[key] = (
                            answer_list[0]
                            if isinstance(answer_list, list) and answer_list
                            else answer_list
                            if isinstance(answer_list, str)
                            else ""
                        )
                return MCPElicitResult(action="accept", content=content)

            is_multi = isinstance(schema, dict) and (
                schema.get("type") == "array" or "items" in schema
            )
            answer = payload[0] if payload else []
            if is_multi:
                return MCPElicitResult(
                    action="accept",
                    content={
                        "value": answer if isinstance(answer, list) else [answer] if answer else []
                    },
                )
            return MCPElicitResult(
                action="accept",
                content={
                    "value": answer[0]
                    if isinstance(answer, list) and answer
                    else answer
                    if isinstance(answer, str)
                    else ""
                },
            )

        # ElicitationResumePayload path.
        match payload.action:
            case "accept":
                return MCPElicitResult(
                    action="accept",
                    content=normalize_elicit_content(payload.content),
                )
            case "decline":
                return MCPElicitResult(action="decline")
            case "cancel":
                return MCPElicitResult(action="cancel")
            case _:
                return MCPElicitResult(action="decline")

    def get_session_state(self) -> Any | None:
        """Get the SessionState for the current run if available.

        Returns:
            The SessionState from SessionPool, or None if not in a pooled session.
        """
        if self.run_ctx is None:
            return None
        session_id = self.run_ctx.session_id
        if not session_id:
            return None
        pool = self.node.host_context
        if pool is None or pool.session_pool is None:
            return None
        return pool.session_pool.sessions.get_session(session_id)

    async def report_progress(self, progress: float, total: float | None, message: str) -> None:
        """Report progress by emitting event into the agent's stream."""
        from wolfharness.agents.events import ToolCallProgressEvent

        logger.info("Reporting tool call progress", progress=progress, total=total, message=message)
        progress_event = ToolCallProgressEvent(
            progress=int(progress),
            total=int(total) if total is not None else 100,
            message=message,
            tool_name=self.tool_name or "",
            tool_call_id=self.tool_call_id or "",
            tool_input=self.tool_input,
        )
        if self.run_ctx is not None and self.run_ctx.event_bus is not None:
            await self.run_ctx.event_bus.publish(self.run_ctx.session_id, progress_event)
        else:
            logger.debug(
                "report_progress called with no active run context or event_bus — event dropped"
            )

    @property
    def events(self) -> StreamEventEmitter:
        """Get event emitter with context automatically injected."""
        from wolfharness.agents.events import StreamEventEmitter

        event_bus = self.run_ctx.event_bus if self.run_ctx else None
        return StreamEventEmitter(self, event_bus=event_bus)

    async def handle_confirmation(self, tool: Tool, args: dict[str, Any]) -> ConfirmationResult:
        """Handle tool execution confirmation.

        Returns "allow" if:
        - No confirmation handler is set
        - Handler confirms the execution

        Args:
            tool: The tool being executed
            args: Arguments passed to the tool

        Returns:
            Confirmation result indicating how to proceed
        """
        provider = self.get_input_provider()
        # Get tool_confirmation_mode if available (NativeAgent only)
        # Other agents handle permission checks in their own way
        mode = getattr(self.agent, "tool_confirmation_mode", "per_tool")
        if (mode == "per_tool" and not tool.requires_confirmation) or mode == "never":
            return "allow"
        return await provider.get_tool_confirmation(self, tool.description or "")

    @property
    def internal_fs(self) -> IsolatedMemoryFileSystem:
        """Access agent's internal filesystem for tool state.

        Tools can use this to store logs, history, temporary files, etc.
        The filesystem is scoped to the agent instance.

        Returns:
            In-memory filesystem for this agent
        """
        return self.agent.internal_fs

    async def create_child_session(
        self,
        agent_name: str,
        agent_type: str,
        parent_session_id: str | None = None,
        *,
        spawn_mechanism: str = "foreground",
        description: str = "",
        tool_call_id: str | None = None,
        input_provider: Any = None,
        skip_agent_registration: bool = False,
        **metadata: Any,
    ) -> str:
        """Create a child session for a subagent delegation.

        When the agent pool and its session pool are available, the child
        session is created via ``SessionPool.create_session()`` so that
        parent-child relationships, project context, and working directory
        are inherited automatically.  When no pool is present (e.g. during
        standalone or test runs) a new session ID is generated without
        persistence.

        When ``run_ctx`` is set (i.e. the agent is running inside a pooled
        session), a ``SpawnSessionStart`` event is auto-emitted and a
        ``done_event`` is registered on ``run_ctx.child_done_events`` so
        that callers can await subagent completion.

        The agent is eagerly registered under the child session_id via
        ``get_or_create_session_agent()`` so that ``receive_request()`` and
        ``run_stream()`` can find it without a separate call.  If
        ``input_provider`` is given, it is passed to the agent registration
        call so it is baked into the cached agent instance.

        Args:
            agent_name: Name of the child agent.
            agent_type: Type of the child agent (``"native"``, ``"claude"``, etc.).
            parent_session_id: Explicit parent session ID.  When *None* the
                current node's ``session_id`` is used as the parent.
            spawn_mechanism: How the subagent is created — ``"foreground"``
                for synchronous delegation, ``"task"`` for background.
            description: Human-readable description of the spawn operation.
            tool_call_id: ID of the tool call that triggered the spawn.
            input_provider: Optional input provider for the child agent.
                Passed to ``get_or_create_session_agent`` so it is available
                on the cached agent instance.
            skip_agent_registration: When *True*, skip the eager
                ``get_or_create_session_agent()`` call.  Needed for teams
                whose node is created separately via
                ``create_team_from_config()``.
            **metadata: Additional metadata to attach to the child session.

        Returns:
            The child session ID string.
        """
        child_sid: str
        pool = self.node.host_context
        if pool is not None and pool.session_pool is not None:
            effective_parent = parent_session_id or self.node._events.session_id
            # Guard against MagicMock auto-generated attributes in tests:
            # _events.session_id may return a Mock when not explicitly set.
            if isinstance(effective_parent, str):
                # Delegate to SessionPool.create_child_session() which
                # handles ID generation and parent inheritance.
                state = await pool.session_pool.create_child_session(
                    parent_session_id=effective_parent,
                    agent_name=agent_name,
                    agent_type=agent_type,
                    **metadata,
                )
                child_sid = state.session_id
                # Eagerly register agent under child session_id so that
                # receive_request / run_stream can find it without a
                # separate get_or_create_session_agent call.
                # Skipped for teams — team nodes are created separately
                # via create_team_from_config() and don't need agent
                # registration.
                if not skip_agent_registration:
                    agent_kwargs: dict[str, Any] = {}
                    if input_provider is not None:
                        agent_kwargs["input_provider"] = input_provider
                    await pool.session_pool.sessions.get_or_create_session_agent(
                        child_sid,
                        agent_name,
                        **agent_kwargs,
                    )
            else:
                from wolfharness.utils.identifiers import generate_session_id

                child_sid = generate_session_id()
        else:
            # Fallback: no pool, no session_pool — generate ephemeral ID.
            from wolfharness.utils.identifiers import generate_session_id

            child_sid = generate_session_id()

        # Auto-emit SpawnSessionStart and register done_event when running
        # inside a pooled session (run_ctx is set).  In standalone/test mode
        # (run_ctx is None) this is skipped.
        if self.run_ctx is not None:
            child_depth = self.run_ctx.depth + 1
            if child_depth > MAX_SUBAGENT_DEPTH:
                raise SubagentDepthError(
                    f"Subagent depth {child_depth} exceeds limit {MAX_SUBAGENT_DEPTH}",
                )
            from wolfharness.agents.events.events import SpawnSessionStart

            event_spawn_mechanism: Literal["task", "spawn"] = (
                "task" if spawn_mechanism == "task" else "spawn"
            )
            # Resolve display_name from manifest config when available.
            child_display_name: str | None = None
            if pool is not None:
                child_config = pool.manifest.agents.get(agent_name)
                if child_config is not None and child_config.display_name is not None:
                    child_display_name = (
                        child_config.display_name
                        if child_config.display_name != agent_name
                        else None
                    )

            spawn_event = SpawnSessionStart(
                child_session_id=child_sid,
                parent_session_id=self.run_ctx.session_id,
                tool_call_id=tool_call_id or self.tool_call_id,
                spawn_mechanism=event_spawn_mechanism,
                source_name=agent_name,
                display_name=child_display_name,
                source_type="agent",
                depth=child_depth,
                description=description,
            )
            await self.events.emit_event(spawn_event)
            done_event = anyio.Event()
            self.run_ctx.child_done_events[child_sid] = done_event

        return child_sid

    @property
    def overlay_fs(self) -> OverlayFileSystem:
        """Access unified filesystem combining agent storage and VFS resources.

        Provides a layered view where writes go to agent's internal filesystem
        and reads fall through to VFS resources.

        Returns:
            OverlayFileSystem for this agent
        """
        return self.agent.overlay_fs
