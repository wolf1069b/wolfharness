"""Ephemeral run handle for agent execution lifecycle management.

In the per-prompt RunHandle model, each RunHandle executes exactly one
turn and terminates naturally. Session-level state (lifecycle dimensions,
conversation history, message routing) is owned by ``SessionState``.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Self
import uuid

import logfire

from wolfharness.agents.context import AgentRunContext
from wolfharness.agents.events import (
    RunErrorEvent,
    RunFailedEvent,
    RunStartedEvent,
    StreamCompleteEvent,
    UserMessageInsertedEvent,
)
from wolfharness.lifecycle import RunOutcome
from wolfharness.log import get_logger
from wolfharness.messaging import ChatMessage
from wolfharness.observability.spans import safe_span


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable
    from typing import Any

    from pydantic_ai import AgentRun
    from pydantic_ai.messages import ModelMessage

    from wolfharness.agents.base_agent import BaseAgent
    from wolfharness.agents.events.events import RichAgentStreamEvent
    from wolfharness.host.context import HostContext
    from wolfharness.host.registry import AgentRegistry
    from wolfharness.lifecycle.protocols import CommChannel
    from wolfharness.orchestrator.core import EventBus, SessionState


logger = get_logger(__name__)

# Check if EnqueuedMessagesEvent is available in the installed pydantic-ai version.
# When available, the stream itself emits display events for enqueued messages,
# making the manual _schedule_user_message_emission() call redundant for
# steer/followup when an active_agent_run exists.
try:
    from pydantic_ai.messages import EnqueuedMessagesEvent  # noqa: F401

    ENQUEUED_MESSAGES_AVAILABLE = True
except ImportError:  # pragma: no cover
    ENQUEUED_MESSAGES_AVAILABLE = False


def _has_invalid_json_args(part: Any) -> bool:
    """Backward-compatible alias for the shared invalid-JSON check.

    Delegates to :func:`wolfharness.utils.pydantic_ai_helpers.has_invalid_json_args`.
    """
    from wolfharness.utils.pydantic_ai_helpers import has_invalid_json_args

    return has_invalid_json_args(part)


def inject_cancelled_tool_results(messages: list[ModelMessage]) -> list[ModelMessage]:
    r"""Inject RetryPromptPart for unprocessed tool calls in message history.

    When a turn is cancelled mid-tool-call, the message history ends with a
    ``ModelResponse`` containing ``ToolCallPart``\s but no corresponding
    ``ModelRequest`` with tool results. PydanticAI rejects new user prompts
    in this state with:
    "Cannot provide a new user prompt when the message history contains
    unprocessed tool calls."

    This function scans the message history for trailing unprocessed tool
    calls and appends a ``ModelRequest`` with a ``RetryPromptPart`` for each,
    telling the model the tool was cancelled. This preserves the model's
    decision context (it knows it called the tool) while satisfying
    PydanticAI's message history validation.

    Additionally, some models (e.g. deepseek-v4-flash) may generate tool call
    arguments with invalid JSON. This poisons conversation history — the model
    API rejects subsequent requests with 400: "Assistant tool call
    function.arguments must be valid JSON." This function detects such invalid
    args in the trailing ``ModelResponse`` and replaces them with ``{}`` so the
    history can be sent to the model API without rejection.

    Only the last ``ModelResponse`` is checked: if invalid JSON existed in an
    earlier message, the model API would have rejected the request at that
    point and the conversation could not have continued.

    Args:
        messages: The message history to sanitize.

    Returns:
        A new list with cancelled tool results injected and invalid JSON args
        sanitized if needed.
    """
    from pydantic_ai.messages import ModelRequest, ModelResponse, RetryPromptPart, ToolCallPart

    if not messages:
        return list(messages)

    result = list(messages)

    # Check if the last message is a ModelResponse with tool calls.
    last_msg = result[-1]
    if not isinstance(last_msg, ModelResponse):
        return result

    # Collect tool calls that need results, sanitizing invalid JSON args.
    pending_tool_calls: list[ToolCallPart] = []
    needs_rebuild = False
    new_parts: list[Any] = []
    for part in last_msg.parts:
        match part:
            case ToolCallPart(tool_name=tool_name, tool_call_id=call_id) if tool_name and call_id:
                if _has_invalid_json_args(part):
                    # Replace invalid JSON args with {} to prevent model API
                    # 400 rejection on subsequent requests.
                    sanitized = ToolCallPart(
                        tool_name=part.tool_name,
                        args={},
                        tool_call_id=part.tool_call_id,
                    )
                    new_parts.append(sanitized)
                    pending_tool_calls.append(sanitized)
                    needs_rebuild = True
                else:
                    new_parts.append(part)
                    pending_tool_calls.append(part)
            case _:
                new_parts.append(part)

    if not pending_tool_calls:
        return result

    # If we sanitized any args, replace the last ModelResponse with the
    # rebuilt version.
    if needs_rebuild:
        result[-1] = ModelResponse(parts=new_parts)

    # Build a ModelRequest with RetryPromptPart for each pending tool call.
    retry_parts: list[ModelRequest] = [
        ModelRequest(
            parts=[
                RetryPromptPart(
                    content=(
                        f"Tool '{tc.tool_name}' was cancelled. "
                        "The user interrupted the run before the tool could complete."
                    ),
                    tool_name=tc.tool_name,
                    tool_call_id=tc.tool_call_id,
                ),
            ],
        )
        for tc in pending_tool_calls
    ]

    result.extend(retry_parts)
    return result


@dataclass
class RunHandle:
    """Ephemeral runtime handle for a single agent turn.

    In the per-prompt model, each ``RunHandle`` executes exactly one turn
    and terminates naturally. Session-level state (lifecycle dimensions,
    conversation history, message routing) is owned by ``SessionState``.

    ``start()`` is an async generator that yields stream events from a
    single turn, then exits. There is no idle loop — between turns,
    ``SessionState`` creates a new ``RunHandle`` for the next prompt.

    Attributes:
        run_id: Unique identifier for this run.
        session_id: Session this run belongs to.
        agent_type: Type of agent running (e.g. ``"native"``, ``"claude"``).
        outcome: Terminal outcome (``RunOutcome.COMPLETED``, ``FAILED``,
            ``CHECKPOINTED``) set when the run completes.
        agent: The agent instance driving turns.
        event_bus: Event bus for publishing stream events.
        session: Per-session state containing the turn lock.
        run_ctx: Per-run isolated state container.
        complete_event: Set after the turn completes and cleanup finishes.
        _cleanup_callback: Optional callback invoked with run_id during cleanup.
        active_agent_run: Reference to PydanticAI AgentRun, set by
            NativeTurn during execution and cleared in ``finally``.
        _message_history: Constructor-only field, derived from
            ``agent.conversation.get_history()`` at RunHandle creation.
    """

    run_id: str
    session_id: str
    agent_type: str
    outcome: RunOutcome | None = None
    agent: BaseAgent[Any, Any] | None = None
    event_bus: EventBus | None = None
    session: SessionState | None = None
    run_ctx: AgentRunContext = field(default_factory=AgentRunContext)
    complete_event: asyncio.Event = field(default_factory=asyncio.Event)
    _cleanup_callback: Callable[[str], None] | None = None
    active_agent_run: AgentRun[Any, Any] | None = None
    _cancel_fn: Callable[[], None] | None = None
    _message_history: list[ModelMessage] = field(default_factory=list)
    """Constructor-only field. Bridged from ``agent.conversation.get_history()``
    at RunHandle creation. NOT accumulated after turns — the next RunHandle
    gets a fresh copy from ``agent.conversation``.
    """
    _current_turn: Any = None
    """The current Turn being executed. Set by ``_execute_turn()``, read by
    ``_handle_turn_result()``."""
    _current_turn_failed: bool = False
    """Whether the current turn failed. Set by ``_execute_turn()``."""
    _interrupt_task: asyncio.Task[None] | None = None
    """Background task for agent._interrupt(), stored to prevent GC."""
    _emission_tasks: set[asyncio.Task[None]] = field(default_factory=set)
    """References to fire-and-forget UserMessageInsertedEvent emission tasks.

    Stored to prevent GC of tasks created by ``_schedule_user_message_emission()``.
    Tasks are removed via a ``done_callback`` when they complete.
    """

    # ------------------------------------------------------------------
    # HostContext injection (M3 task group 15) — now sourced from SessionState
    # ------------------------------------------------------------------
    _host_context: HostContext | None = None
    """HostContext for constructing per-turn AgentContextDeps.

    When set, ``start()`` constructs an ``AgentContextDeps`` per turn and
    injects it into ``run_ctx.deps`` so capabilities like
    ``SubagentCapability`` can access the delegation service.
    """
    _agent_registry: AgentRegistry | None = None
    """Read-only registry of compiled agents for delegation."""
    _resume_deferred_tool_results: Any = None
    """Deferred tool results from checkpoint, forwarded to ``agent.create_turn()``
    via ``**pydantic_ai_kwargs`` during resume. Only set by
    ``_create_run_handle()`` when resuming from a checkpoint."""
    _run_span: Any = None
    """Per-run root OTel span (``run.message``). Created in ``_start_run_handle``
    as a new trace root (not inheriting the caller's context). Stored here so
    ``_consume_run`` can end it when the run completes."""
    _run_context: Any = None
    """OTel Context containing ``_run_span``. Attached in ``_consume_run`` so
    all spans within the run (turn.native, tools, notifications) are children
    of ``run.message`` and share the same trace_id."""
    _enqueued_messages_available: bool = field(default=False)
    """Whether ``EnqueuedMessagesEvent`` is available in the installed
    pydantic-ai version. When ``True`` and ``active_agent_run`` is not
    ``None``, steer/followup skip ``_schedule_user_message_emission()``
    because the stream itself emits the display event via
    ``EnqueuedMessagesEvent``."""

    def __post_init__(self) -> None:
        """Initialize computed fields after dataclass construction."""
        self._enqueued_messages_available = ENQUEUED_MESSAGES_AVAILABLE

    @property
    def is_running(self) -> bool:
        """Whether the RunHandle is actively executing a turn.

        Returns:
            ``True`` if the turn has not yet completed (``complete_event``
            is not set).
        """
        return not self.complete_event.is_set()

    @property
    def _active_agent_run(self) -> AgentRun[Any, Any] | None:
        """Alias for ``active_agent_run``.

        Provides the underscore-prefixed access for internal callers
        that prefer the private naming convention.
        """
        return self.active_agent_run

    def _inject_agent_context(self) -> None:
        """Construct and inject AgentContextDeps into run_ctx.deps.

        Builds a fresh ``AgentContextDeps`` per turn using the host context,
        agent registry, and resource source. The AgentContextDeps is set as
        ``run_ctx.deps`` so pydantic-ai's ``RunContext.deps`` carries it
        into tool calls. Capabilities like ``SubagentCapability`` access
        it via ``ctx.deps``.

        When ``_host_context`` is None (standalone execution without a
        pool), this is a no-op — ``run_ctx.deps`` stays at its prior value.
        """
        if self._host_context is None:
            return
        from wolfharness.capabilities.agent_context import AgentContextDeps
        from wolfharness.capabilities.runloop_delegation import RunLoopDelegationService
        from wolfharness.host.context import RunScope

        registry = self._agent_registry
        if registry is None:
            return

        scope = RunScope(
            config_id=self._host_context.config_id or "default",
            tenant_id=self._host_context.tenant_id or "default",
            session_id=self.session_id,
        )
        delegation = RunLoopDelegationService(
            registry=registry,
            host=self._host_context,
            session_id=self.session_id,
        )
        ctx = AgentContextDeps(
            agent_registry=registry,
            delegation=delegation,
            session=self.session,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            scope=scope,
            host=self._host_context,
            extension_registry=(
                self._host_context.extension_registry if self._host_context is not None else None
            ),
            agent_name=self.agent.name if self.agent is not None else self.agent_type,
        )
        self.run_ctx.deps = ctx

    # ------------------------------------------------------------------
    # Per-prompt execution: single turn, then natural termination
    # ------------------------------------------------------------------

    async def start(
        self, initial_prompt: str | list[Any] = ""
    ) -> AsyncGenerator[RichAgentStreamEvent[Any]]:
        """Execute a single turn and yield stream events, then terminate.

        In the per-prompt model, ``start()`` executes exactly one turn
        (1 prompt) and exits naturally. There is no idle loop — the
        caller (``_consume_run()``) is responsible for creating a new
        ``RunHandle`` for the next prompt.

        Args:
            initial_prompt: The user prompt to process. Can be a ``list``
                for multimodal content (images, audio, etc.).
        """
        agent = self.agent
        event_bus = self.event_bus
        session = self.session
        if agent is None:
            raise RuntimeError("agent must be set before calling start()")
        # event_bus can be None for standalone execution (no EventBus).
        # In that case, session._comm_channel must be a DirectChannel
        # (set by _initialize_lifecycle_and_recovery()).
        if session is None:
            raise RuntimeError("session must be set before calling start()")

        # Set _run_handle on run_ctx so NativeTurn can access active_agent_run
        self.run_ctx._run_handle = self
        # Set current_task so cancel() can interrupt the running turn.
        self.run_ctx.current_task = asyncio.current_task()
        # Wire _cancel_fn so cancel() triggers agent._interrupt() (ACP
        # CancelNotification, native _iteration_task cancel).
        self._cancel_fn = self._create_cancel_fn()

        # Drain any steer messages that arrived while the session was idle.
        # These were enqueued to SessionState.feedback_queue by
        # SessionPool.steer_from_background_task() when no RunHandle was active,
        # or by steer() fallback during a race window.
        # We write directly to queued_steer_messages (NOT via self.steer())
        # because steer() would re-enqueue to feedback_queue when
        # active_agent_run is not yet set, creating an infinite loop.
        # NativeTurn.execute() will drain queued_steer_messages into
        # agent_run via drain_queued_steer_messages() after setting
        # active_agent_run.
        while not session.feedback_queue.empty():
            try:
                fb = session.feedback_queue.get_nowait()
                if fb.content_blocks is not None:
                    self.run_ctx.queued_steer_messages.append(fb.content_blocks)
                else:
                    self.run_ctx.queued_steer_messages.append(fb.content)
            except asyncio.QueueEmpty:
                break

        with safe_span(
            "orchestration.run_handle.start",
            session_id=self.session_id,
            agent_type=self.agent_type,
        ):
            from wolfharness.observability.trace import get_trace_id

            logfire.info(
                "RunHandle started (per-prompt)",
                trace_id=get_trace_id(),
                session_id=self.session_id,
                agent_type=self.agent_type,
            )
            try:
                async with session.turn_lock:
                    # Execute exactly one turn.
                    # CRITICAL: empty string must produce [] not [""].
                    current_prompts: list[str | list[Any]] = (
                        [initial_prompt] if initial_prompt else []
                    )
                    if not current_prompts and (agent is None or not agent.staged_content):
                        # No prompt and no staged_content — nothing to do.
                        # (issue #284: staged_content may have been injected
                        # by skill commands even when the prompt is empty)
                        return
                    # If we reach here with empty current_prompts but
                    # staged_content has content, execute a turn so
                    # NativeTurn.execute() can consume staged_content.

                    try:
                        async with contextlib.aclosing(
                            self._execute_turn(
                                agent,
                                event_bus,
                                session,
                                current_prompts,
                            ),
                        ) as turn_gen:
                            async for event in turn_gen:
                                yield event
                    except asyncio.CancelledError:
                        # External cancellation (e.g. session close).
                        # Publish RunFailedEvent and let the generator
                        # terminate naturally.
                        with contextlib.suppress(Exception):
                            await self._publish_cancelled_event(event_bus)
                        raise
            finally:
                # Per-turn cleanup: set complete_event.
                # Do NOT close lifecycle dimensions — they are session-owned.
                # Do NOT call agent.__aexit__() — that's session-level.
                self.complete_event.set()

    async def _safe_publish(
        self,
        comm: CommChannel,
        event: RichAgentStreamEvent[Any],
    ) -> None:
        """Publish an event to the CommChannel, suppressing errors if closed.

        During session shutdown (e.g. team member removal), the ProtocolChannel
        may be closed before the RunHandle finishes publishing events. In that
        case, ``comm.publish()`` raises ``RuntimeError("ProtocolChannel is
        closed; cannot publish.")``. This is a benign race — the session is
        already shutting down and no consumer remains to receive the event.

        Args:
            comm: The CommChannel to publish on.
            event: The event to publish.
        """
        try:
            await comm.publish(event)
        except RuntimeError as e:
            if "ProtocolChannel is closed" in str(e):
                logger.debug(
                    "ProtocolChannel closed during publish — session likely "
                    "shutting down. Event type: %s",
                    type(event).__name__,
                )
            else:
                raise

    async def _publish_cancelled_event(self, event_bus: EventBus | None) -> None:
        """Publish a RunFailedEvent for cancelled turns.

        Args:
            event_bus: The event bus to publish on, or ``None`` for
                standalone execution (events go to CommChannel only).
        """
        comm = self.session._comm_channel if self.session is not None else None
        cancelled_event = RunFailedEvent(
            run_id=self.run_id,
            session_id=self.session_id,
            exception=RuntimeError("Run cancelled"),
        )
        if comm is not None and event_bus is not None and not comm.publishes_to_event_bus:
            await event_bus.publish(self.session_id, cancelled_event)
        if comm is not None:
            await self._safe_publish(comm, cancelled_event)
        elif comm is None and event_bus is not None:
            await event_bus.publish(self.session_id, cancelled_event)

    async def _execute_turn(  # noqa: PLR0915
        self,
        agent: BaseAgent[Any, Any],
        event_bus: EventBus | None,
        session: SessionState,
        current_prompts: list[str | list[Any]],
    ) -> AsyncGenerator[RichAgentStreamEvent[Any]]:
        """Execute a single turn and yield stream events.

        Creates a Turn from the current prompts, publishes
        ``RunStartedEvent``, saves the user prompt to conversation
        history, then executes the Turn and yields each event. On
        exception, publishes and yields a ``RunErrorEvent``.

        Args:
            agent: The agent driving the turn.
            event_bus: The event bus for publishing events.
            session: The per-session state.
            current_prompts: Prompts for this turn.
        """
        comm = session._comm_channel
        assert comm is not None
        # Generate a unique turn_id for this Turn.
        turn_id = str(uuid.uuid4())
        self.run_ctx.turn_id = turn_id
        # Reset per-turn state.
        if self.run_ctx.cancelled:
            self.run_ctx.cancelled = False
        # Construct per-turn AgentContextDeps and inject as deps so
        # capabilities (SubagentCapability, etc.) can access the
        # delegation service, resource sources, and host.
        self._inject_agent_context()
        # Forward _resume_deferred_tool_results to agent.create_turn()
        # via **pydantic_ai_kwargs so it reaches NativeTurn → agentlet.iter().
        # Only set during resume from checkpoint; None for normal turns.
        create_turn_kwargs: dict[str, Any] = {}
        if self._resume_deferred_tool_results is not None:
            create_turn_kwargs["deferred_tool_results"] = self._resume_deferred_tool_results
        turn = agent.create_turn(
            prompts=current_prompts,  # type: ignore[arg-type]
            run_ctx=self.run_ctx,
            message_history=self._message_history,
            **create_turn_kwargs,
        )
        # Publish RunStartedEvent before turn.execute() so consumers
        # know a new turn is starting.
        run_started = RunStartedEvent(
            run_id=self.run_id,
            session_id=self.session_id,
            agent_name=self.agent.name if self.agent is not None else self.agent_type,
            parent_session_id=session.parent_session_id if session is not None else None,
        )
        if event_bus is not None and not comm.publishes_to_event_bus:
            await event_bus.publish(self.session_id, run_started)
        await self._safe_publish(comm, run_started)
        # Set _current_input_provider ContextVar so MCP elicitation can
        # access it during turn execution.
        if session.input_provider is not None:
            from wolfharness.mcp_server.manager import _current_input_provider

            _current_input_provider.set(session.input_provider)
        # Save user prompt to agent conversation before execution.
        # This ensures user messages are preserved even if the turn
        # fails or is cancelled.
        # Skip when current_prompts is empty — the real content comes from
        # staged_content (consumed by NativeTurn), and saving an empty user
        # message would pollute conversation history (issue #284).
        if current_prompts:
            from wolfharness.agents.native_agent.helpers import _summarize_content_block

            prompt_text = "\n".join(
                p if isinstance(p, str) else " ".join(_summarize_content_block(b) for b in p)
                for p in current_prompts
            )
            agent.conversation.add_chat_messages([
                ChatMessage(
                    content=prompt_text,
                    role="user",
                    name=agent.name,
                    session_id=self.session_id,
                ),
            ])
        # Store turn state for downstream sub-methods.
        self._current_turn = turn
        self._current_turn_failed = False
        turn_failed = False
        stream_complete_saved = False
        with safe_span(
            "orchestration.run_handle.execute_turn",
            turn_id=turn_id,
            session_id=self.session_id,
        ):
            try:
                async with contextlib.aclosing(turn.execute()) as event_gen:
                    async for event in event_gen:
                        if event_bus is not None and not comm.publishes_to_event_bus:
                            await event_bus.publish(self.session_id, event)
                        await self._safe_publish(comm, event)
                        # Save assistant final message to conversation BEFORE
                        # yielding. The _consume_run caller closes the generator
                        # immediately after receiving StreamCompleteEvent, which
                        # prevents any code after `yield event` from executing.
                        if isinstance(event, StreamCompleteEvent) and event.message is not None:
                            agent.conversation.add_chat_messages(
                                [event.message],
                                extend_last=True,
                            )
                            stream_complete_saved = True
                        yield event
                        if isinstance(event, RunErrorEvent):
                            turn_failed = True
                            break
                        if isinstance(event, StreamCompleteEvent):
                            break
            except (GeneratorExit, asyncio.CancelledError):
                # GeneratorExit: from aclose() on start() — let safe_span
                #   __exit__ run in the finally block, then propagate.
                # CancelledError: may be raised by anyio cancel scope cleanup
                #   inside pydantic-ai's Agent.iter() during GeneratorExit
                #   processing, or from task.cancel(). Propagate as-is so
                #   callers can suppress appropriately.
                raise
            except Exception as e:  # noqa: BLE001
                turn_failed = True
                error_event = RunErrorEvent(
                    message=str(e),
                    run_id=self.run_id,
                    agent_name=self.agent.name if self.agent is not None else self.agent_type,
                )
                if event_bus is not None and not comm.publishes_to_event_bus:
                    await event_bus.publish(self.session_id, error_event)
                await self._safe_publish(comm, error_event)
                yield error_event
            finally:
                self._current_turn_failed = turn_failed
                # Preserve partial history for ALL non-StreamCompleteEvent
                # exit paths. Without this:
                #
                # - RunErrorEvent (generic Exception): agent.conversation
                #   has only the user message — next turn loses context.
                # - CancelledError (cooperative cancel): _final_message IS
                #   set by NativeTurn but no StreamCompleteEvent is yielded,
                #   so the StreamCompleteEvent branch never fires.
                #
                # Use the private attribute to avoid raising when
                # _final_message was never set (e.g. generic Exception
                # before any output was produced).  Skip if the
                # StreamCompleteEvent branch already saved.
                if not stream_complete_saved and turn._final_message is not None:
                    agent.conversation.add_chat_messages(
                        [turn._final_message],
                        extend_last=True,
                    )

    @logfire.instrument("orchestration.run_handle.steer")
    def steer(
        self,
        message: str | list[Any],
        *,
        message_id: str | None = None,
        emit_user_message: bool = True,
    ) -> str | None:
        """Inject a steer message into the active turn.

        Called by ``SessionState`` when a RunHandle is active. Directly
        calls ``agent_run.enqueue()`` to inject the message into
        PydanticAI's pending message drain. If no ``agent_run`` is
        active, re-enqueues the message to ``session.feedback_queue``
        so it survives across RunHandle boundaries and is drained by
        the next RunHandle's ``start()``.

        Args:
            message: The steer message (plain text or structured content
                blocks).
            message_id: Optional message ID. Auto-generated as UUID4 if
                not provided.
            emit_user_message: When ``True`` (default), schedule a
                fire-and-forget ``UserMessageInsertedEvent`` publication
                so protocol frontends display the injected message. Set
                to ``False`` to suppress emission (e.g. internal
                chaining that should not produce a visible user message).

        Returns:
            The ``message_id`` string on success, ``None`` if no agent_run
            is active and the message was queued.
        """
        from wolfharness.lifecycle.types import Feedback

        # Construct Feedback with message_id and content_blocks.
        fb_kwargs: dict[str, Any] = {}
        if message_id is not None:
            fb_kwargs["message_id"] = message_id
        if isinstance(message, list):
            fb = Feedback(
                content="",
                is_steer=True,
                content_blocks=message,
                **fb_kwargs,
            )
        else:
            fb = Feedback(
                content=message,
                is_steer=True,
                **fb_kwargs,
            )

        agent_run = self.active_agent_run
        if agent_run is not None:
            # Append message_id to FIFO queue BEFORE enqueue so
            # handle_enqueued_messages() can reuse the same ID.
            self.run_ctx._pending_enqueue_message_ids.append(fb.message_id)
            if fb.content_blocks is not None:
                agent_run.enqueue(*fb.content_blocks, priority="asap")
            else:
                agent_run.enqueue(fb.content, priority="asap")
        elif self.session is not None:
            # No active agent_run — re-enqueue to session.feedback_queue so
            # the message survives across RunHandle boundaries. The next
            # RunHandle's start() will drain feedback_queue into
            # queued_steer_messages, and NativeTurn.execute() will drain
            # those into agent_run via drain_queued_steer_messages().
            self.session.feedback_queue.put_nowait(fb)
        elif fb.content_blocks is not None:
            # No session (standalone execution) — fallback to queued list.
            self.run_ctx.queued_steer_messages.append(fb.content_blocks)
        else:
            # No session (standalone execution) — fallback to queued list.
            self.run_ctx.queued_steer_messages.append(fb.content)

        # Fire-and-forget UserMessageInsertedEvent publication.
        # When EnqueuedMessagesEvent is available AND there is an active
        # agent_run, skip the fire-and-forget emission — the display event
        # will come from handle_enqueued_messages() with source="processed".
        # Otherwise, emit with source="accepted" as a fallback display.
        if emit_user_message and not (
            self._enqueued_messages_available and self.active_agent_run is not None
        ):
            self._schedule_user_message_emission(message, "steer", message_id=fb.message_id)

        return fb.message_id

    def drain_queued_steer_messages(self) -> None:
        """Drain ``queued_steer_messages`` into the active ``agent_run``.

        Called by ``NativeTurn.execute()`` immediately after setting
        ``active_agent_run``. Delivers any steer messages that arrived
        before ``active_agent_run`` was set (e.g., during ``start()``
        ``feedback_queue`` drain or during the race window between
        turn end and RunHandle cleanup).

        Each message is enqueued to ``agent_run`` with ``priority="asap"``
        so PydanticAI's ``PendingMessageDrainCapability`` injects it
        into the current model request.

        After draining, ``queued_steer_messages`` is cleared.

        !!! note "No-op when inactive"
            If ``active_agent_run`` is ``None``, this method is a no-op.
            Messages remain in ``queued_steer_messages`` for a later drain.
        """
        agent_run = self.active_agent_run
        if agent_run is None:
            return
        queued = self.run_ctx.queued_steer_messages
        if not queued:
            return
        self.run_ctx.queued_steer_messages = []
        for msg in queued:
            if isinstance(msg, list):
                agent_run.enqueue(*msg, priority="asap")
            else:
                agent_run.enqueue(msg, priority="asap")

    def followup(
        self,
        message: str | list[Any],
        *,
        emit_user_message: bool = False,
    ) -> str | None:
        """Queue a follow-up prompt for the next RunHandle.

        In the per-prompt model, a follow-up message is enqueued on
        ``SessionState.prompt_queue`` and will be drained by
        ``SessionController._consume_run()`` after the current
        RunHandle terminates.

        Args:
            message: The follow-up prompt content (plain text or
                structured content blocks).
            emit_user_message: When ``True``, schedule a fire-and-forget
                ``UserMessageInsertedEvent`` publication so protocol
                frontends display the queued message. Defaults to
                ``False`` — followup is usually internal chaining, not
                a user-visible action.

        Returns:
            A ``message_id`` string on success, ``None`` if
            no session is attached.
        """
        from wolfharness.utils.identifiers import ascending

        message_id = ascending("message")
        session = self.session
        if session is None:
            return None
        agent_run = self.active_agent_run
        if agent_run is not None:
            # Enqueue directly to the active agent_run so
            # PendingMessageDrainCapability fires EnqueuedMessagesEvent
            # for display. The prompt will be drained as a "when_idle"
            # message after the current node finishes.
            # Append message_id to FIFO queue BEFORE enqueue so
            # handle_enqueued_messages() can reuse the same ID.
            self.run_ctx._pending_enqueue_message_ids.append(message_id)
            if isinstance(message, list):
                agent_run.enqueue(*message, priority="when_idle")
            else:
                agent_run.enqueue(message, priority="when_idle")
        else:
            # No active agent_run — fall back to session.prompt_queue.
            # The next RunHandle's _consume_run() will drain this queue.
            session.prompt_queue.put_nowait(message)

        # Fire-and-forget UserMessageInsertedEvent publication.
        # When EnqueuedMessagesEvent is available AND there is an active
        # agent_run, skip the fire-and-forget emission — the display event
        # will come from handle_enqueued_messages() with source="processed".
        # Otherwise, emit with source="accepted" as a fallback display.
        if emit_user_message and not (
            self._enqueued_messages_available and self.active_agent_run is not None
        ):
            self._schedule_user_message_emission(message, "followup", message_id=message_id)

        return message_id

    def _schedule_user_message_emission(
        self,
        content: str | list[Any],
        delivery: Literal["steer", "followup"],
        *,
        message_id: str | None = None,
    ) -> None:
        """Schedule a fire-and-forget ``UserMessageInsertedEvent`` publication.

        Uses ``asyncio.get_running_loop().create_task()`` so the emission
        runs concurrently without blocking the caller. If no event loop
        is running (e.g. called from a sync context), the emission is
        silently skipped — the steer/followup itself has already succeeded.

        Args:
            content: The message content that was inserted.
            delivery: Delivery mode — ``"steer"`` or ``"followup"``.
            message_id: Optional message ID for dedup correlation. If
                ``None``, a new UUID is generated in the emission helper.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — emission silently skipped. The
            # steer/followup operation itself has already completed.
            return
        task = loop.create_task(
            self._emit_user_message_inserted(content, delivery, "accepted", message_id=message_id),
        )
        # Store reference to prevent GC of the fire-and-forget task.
        self._emission_tasks.add(task)
        task.add_done_callback(self._emission_tasks.discard)

    async def _emit_user_message_inserted(
        self,
        content: str | list[Any],
        delivery: Literal["initial", "steer", "followup"],
        source: Literal["accepted"],
        *,
        message_id: str | None = None,
    ) -> None:
        """Publish a ``UserMessageInsertedEvent`` to the EventBus.

        Wraps emission in a ``logfire.span`` to prevent orphan traces and
        catches all exceptions so a failing EventBus never breaks the
        steer/followup path.

        Args:
            content: The message content that was inserted.
            delivery: Delivery mode — ``"initial"``, ``"steer"``, or
                ``"followup"``.
            source: Originator — ``"accepted"`` for fire-and-forget
                fallback display events.
            message_id: Optional message ID for dedup correlation. If
                ``None``, a new UUID is generated.
        """
        with logfire.span(
            "event.user_message_inserted.emit",
            session_id=self.session_id,
        ):
            try:
                from wolfharness.utils.identifiers import ascending

                event: UserMessageInsertedEvent[Any] = UserMessageInsertedEvent(
                    session_id=self.session_id,
                    message_id=message_id or ascending("message"),
                    content=content,
                    delivery=delivery,
                    source=source,
                )
                if self.event_bus is not None:
                    await self.event_bus.publish(self.session_id, event)
            except Exception:
                logger.warning(
                    "Failed to emit UserMessageInsertedEvent",
                    exc_info=True,
                )

    def close(self) -> None:
        """Set complete_event and perform per-turn cleanup.

        In the per-prompt model, ``close()`` only sets ``complete_event``
        and clears the steer callback on SessionState. It does NOT close
        lifecycle dimensions (they are session-owned) and does NOT call
        ``agent.__aexit__()`` (that's session-level).

        Calling ``close()`` twice is a no-op: the second call returns
        immediately because ``complete_event`` is already set.
        """
        if self.complete_event.is_set():
            return
        self.complete_event.set()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Legacy lifecycle (old code paths — simplified for per-prompt model)
    # ------------------------------------------------------------------

    def complete(self) -> None:
        """Transition the run to completed and trigger cleanup."""
        self.outcome = RunOutcome.COMPLETED
        self._cleanup_run()

    def checkpoint(self) -> None:
        """Transition the run to checkpointed and trigger cleanup."""
        self.outcome = RunOutcome.CHECKPOINTED
        self._cleanup_run()

    def fail(
        self,
        exception: BaseException | None = None,
        *,
        event_bus: Any | None = None,
    ) -> None:
        """Transition the run to failed and trigger cleanup.

        Args:
            exception: Optional exception that caused the failure.
            event_bus: Optional event bus to publish RunFailedEvent on.
        """
        self.outcome = RunOutcome.FAILED
        if exception is not None:
            self.run_ctx.cancelled = True
        if event_bus is not None:
            self._event_task = asyncio.create_task(
                event_bus.publish(
                    self.session_id,
                    RunFailedEvent(
                        run_id=self.run_id,
                        session_id=self.session_id,
                        exception=exception or RuntimeError("Run failed without exception"),
                    ),
                )
            )
        self._cleanup_run()

    @property
    def cancelled(self) -> bool:
        """Whether the run was cancelled.

        Returns the ``run_ctx.cancelled`` flag, which is set by
        ``cancel()`` and may be reset at turn start.
        """
        return self.run_ctx.cancelled

    def cancel(self) -> None:
        """Cancel the run cooperatively.

        Sets the cancelled flag on the run context and calls the
        registered cancel function (wired in ``start()``) which
        schedules ``agent._interrupt()`` for subclass-specific cleanup.

        Also force-cancels ``run_ctx.current_task`` to inject
        ``CancelledError`` into a hung ``__aexit__``. The
        ``CancelledError`` is caught by ``NativeTurn.execute()``'s
        ``except asyncio.CancelledError`` handler which checks
        ``run_ctx.cancelled`` and exits gracefully.

        Idempotency guard: if ``complete_event`` is already set, the
        RunHandle has terminated and cancel is a no-op.
        """
        # Idempotency guard: if already complete, no-op.
        if self.complete_event.is_set():
            return

        self.run_ctx.cancelled = True

        if self._cancel_fn is not None:
            self._cancel_fn()

        # Force-cancel the task driving start() to break through __aexit__
        # hangs. The CancelledError will be caught by NativeTurn's except
        # handler which checks run_ctx.cancelled and exits gracefully. The
        # start() finally block will still run, setting complete_event and
        # releasing turn_lock.
        task = self.run_ctx.current_task
        if task is not None and not task.done():
            task.cancel()

    def _create_cancel_fn(self) -> Callable[[], None]:
        """Create a cancel function that schedules ``agent._interrupt()``.

        Returns a callable that, when invoked, schedules the agent's
        ``_interrupt`` coroutine as a background task. The task reference
        is stored in ``self._interrupt_task`` to prevent GC.
        """
        agent = self.agent
        run_ctx = self.run_ctx

        def _cancel() -> None:
            if agent is None:
                return
            coro = agent._interrupt(run_ctx)
            if asyncio.iscoroutine(coro):
                self._interrupt_task = asyncio.create_task(coro)

        return _cancel

    def _cleanup_run(self) -> None:
        """Invoke cleanup callback and signal completion.

        The complete_event is set *after* all cleanup so that waiters
        observe the handle only when it is fully settled.
        """
        if self._cleanup_callback is not None:
            self._cleanup_callback(self.run_id)
        self.complete_event.set()
