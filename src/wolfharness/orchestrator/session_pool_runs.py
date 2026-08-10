"""Run handle delegation mixin for SessionPool.

Extracted from session_pool.py as part of the session-debt-cleanup file split.
Contains RunHandle creation, run streaming, and run lifecycle methods.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any
import uuid

from wolfharness.agents.context import AgentRunContext
from wolfharness.agents.events import (
    RunErrorEvent,
    StreamCompleteEvent,
)
from wolfharness.log import get_logger
from wolfharness.orchestrator.run import RunHandle
from wolfharness.utils.pydantic_ai_helpers import flatten_prompts


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai.messages import ModelMessage

    from wolfharness.agents.base_agent import BaseAgent
    from wolfharness.delegation import AgentPool
    from wolfharness.orchestrator.event_bus import EventBus
    from wolfharness.orchestrator.session_controller import SessionController, SessionState


logger = get_logger(__name__)


class SessionPoolRunsMixin:
    """Mixin providing RunHandle delegation and streaming methods for SessionPool.

    Attributes:
        sessions: SessionController instance (provided by SessionPool).
        pool: AgentPool instance (provided by SessionPool).
        event_bus: EventBus instance (provided by SessionPool).
    """

    sessions: SessionController
    pool: AgentPool[Any]

    @property
    def event_bus(self) -> EventBus: ...  # type: ignore[empty-body]  # ty: ignore[empty-body]

    def _get_active_run_handle(self, session_id: str) -> RunHandle | None:
        """Get the active RunHandle for a session, if any.

        Returns:
            The RunHandle, or None if no active run exists.
        """
        session = self.sessions.get_session(session_id)
        if session is None or session.current_run_id is None:
            return None
        return self.sessions._runs.get(session.current_run_id)

    def _create_run_handle(
        self,
        session: SessionState,
        agent: BaseAgent[Any, Any],
        session_id: str,
        *,
        deps: Any = None,
        cached_elicitation_responses: dict[str, Any] | None = None,
        deferred_tool_results: Any = None,
        message_history: list[ModelMessage] | None = None,
    ) -> RunHandle:
        """Create and register a RunHandle without a background task.

        Unlike :meth:`SessionController._start_run_handle`, this does
        NOT create an asyncio task to consume ``start()``. The caller
        is responsible for draining ``start()``.

        Creates the RunHandle with ProtocolTrigger and ProtocolChannel
        lifecycle dimensions for protocol server integration, matching
        the pattern in :meth:`SessionController._start_run_handle`.

        Also wires ``_host_context`` and ``_agent_registry`` so that
        ``get_agentlet()`` can create a ``CheckpointManager`` and
        ``SubagentCapability`` can resolve agents — matching the
        infrastructure provided by ``_start_run_handle()``.

        Resume parameters (``cached_elicitation_responses``,
        ``deferred_tool_results``, ``message_history``) are only set
        by ``resume_session()``. Normal turns pass ``None`` for all
        three — runtime behavior is unchanged.

        Args:
            session: The session state.
            agent: The agent instance (native or ACP).
            session_id: The session identifier.
            deps: Optional dependencies to pass to the agent run context
                (e.g. delegation_depth from BackgroundTaskCapability).
            cached_elicitation_responses: Pre-populated elicitation
                responses for crash recovery resume.
            deferred_tool_results: Deferred tool results for resolving
                pending deferred calls during resume.
            message_history: Message history from checkpoint to
                initialize ``RunHandle._message_history``.

        Returns:
            The newly created and registered RunHandle.

        Raises:
            SessionBusyError: If the session already has an active run
                that is not completed.
        """
        from wolfharness.orchestrator.session_controller import SessionBusyError

        # Staleness check: prevent silently overwriting an active run.
        if session.current_run_id is not None and session.current_run_id in self.sessions._runs:
            existing = self.sessions._runs[session.current_run_id]
            if not existing.complete_event.is_set():
                raise SessionBusyError(session_id, session.current_run_id)

        event_bus = self.event_bus
        run_ctx = AgentRunContext(session_id=session_id, event_bus=event_bus, deps=deps)
        if cached_elicitation_responses is not None:
            run_ctx.cached_elicitation_responses = cached_elicitation_responses

        # Use lifecycle dimensions from SessionState (per-prompt migration).
        # _host_context and _agent_registry are also sourced from SessionState.
        run_handle = RunHandle(
            run_id=uuid.uuid4().hex,
            session_id=session_id,
            agent_type=agent.AGENT_TYPE,
            agent=agent,
            event_bus=event_bus,
            session=session,
            run_ctx=run_ctx,
            _host_context=session._host_context,
            _agent_registry=session._agent_registry,
        )
        if message_history is not None:
            # Handle both list[ModelMessage] and MessageHistory objects.
            # Some callers (e.g. subagent_tools.py) pass MessageHistory()
            # which is not directly iterable as list[ModelMessage].
            from wolfharness.messaging.message_history import MessageHistory

            if isinstance(message_history, MessageHistory):
                model_msgs: list[ModelMessage] = []
                for chat_msg in message_history.get_history():
                    model_msgs.extend(chat_msg.messages)
                run_handle._message_history = model_msgs
            else:
                run_handle._message_history = list(message_history)
        else:
            # Bridge from agent.conversation (per-prompt migration, task 1.5).
            model_messages: list[ModelMessage] = []
            conversation = agent.conversation
            if conversation is not None:
                for chat_msg in conversation.get_history():
                    model_messages.extend(chat_msg.messages)
            from wolfharness.orchestrator.run import inject_cancelled_tool_results

            run_handle._message_history = inject_cancelled_tool_results(model_messages)
        if deferred_tool_results is not None:
            run_handle._resume_deferred_tool_results = deferred_tool_results
        self.sessions._runs[run_handle.run_id] = run_handle
        session.current_run_id = run_handle.run_id
        return run_handle

    async def wait_for_completion(
        self,
        session_id: str,
        timeout: float | None = 300,
    ) -> str:
        """Wait for the active run on a session to complete.

        Args:
            session_id: The session to wait for.
            timeout: Maximum seconds to wait. Defaults to 300 seconds.

        Returns:
            The ``session_id`` on completion.

        Raises:
            SessionNotFoundError: If the session does not exist.
            asyncio.TimeoutError: If the run does not complete within
                ``timeout`` seconds.
        """
        return await self.sessions.wait_for_completion(session_id, timeout=timeout)

    def revoke_message(self, session_id: str, message_id: str) -> bool:
        """Revoke a pending steer or followup message by ID.

        Args:
            session_id: The session containing the message.
            message_id: The ID of the message to revoke.

        Returns:
            ``True`` if revoked, ``False`` if already delivered or not found.
        """
        return self.sessions.revoke_inject(session_id, message_id)

    @property
    def active_runs(self) -> list[RunHandle]:
        """Get all currently active (running) RunHandles."""
        return [rh for rh in self.sessions._runs.values() if rh.is_running]

    def get_run(self, run_id: str) -> RunHandle | None:
        """Get a RunHandle by ID.

        Args:
            run_id: The run ID to look up.

        Returns:
            The RunHandle, or None if not found.
        """
        return self.sessions._runs.get(run_id)

    def cancel_run(self, run_id: str) -> None:
        """Cancel a run by ID.

        Args:
            run_id: The run ID to cancel.

        Raises:
            ValueError: If no active run with the given ID exists.
        """
        run_handle = self.sessions._runs.get(run_id)
        if run_handle is None:
            raise ValueError("No active run found with ID: " + run_id)
        run_handle.cancel()

    async def run_stream(
        self,
        session_id: str,
        *prompts: Any,
        scope: str = "session",
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Process prompts and yield events.

        Convenience method for tests and standalone clients that want
        an async iterator over session events. Yields events directly
        from ``RunHandle.start()`` when no active run exists. If a run
        is already active, steers the message and falls back to EventBus.

        Args:
            session_id: The session to process the prompt for.
            *prompts: Prompts to process.
            scope: Subscription scope - "session" (exact match),
                "descendants" (self + children), or "subtree" (self + parent + siblings).
            **kwargs: Additional arguments passed to the turn runner
                (e.g. ``input_provider``).

        Yields:
            Events published to the EventBus for this session.
        """
        async for event in self._run_stream_run_turn(session_id, *prompts, scope=scope, **kwargs):
            yield event

    async def _run_stream_run_turn(  # noqa: PLR0915
        self,
        session_id: str,
        *prompts: Any,
        scope: str = "session",
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Handle run_stream via the RunHandle path.

        If no active run exists, creates a RunHandle and yields events
        directly from ``start()``. If a run is active, steers the
        message and yields from the EventBus subscription.

        Resume parameters (``cached_elicitation_responses``,
        ``deferred_tool_results``, ``message_history``) are extracted
        from ``**kwargs`` and forwarded to ``_create_run_handle()``.
        Only set by ``resume_session()``; normal turns pass ``None``.
        """
        session, _ = await self.sessions.get_or_create_session(session_id)
        if session.is_closing:
            return
        # Extract input_provider from kwargs and set on session BEFORE
        # get_or_create_session_agent() so the agent is created with the
        # correct input_provider and the session state is consistent.
        input_provider = kwargs.pop("input_provider", None)
        if input_provider is not None:
            session.input_provider = input_provider
        # Extract deps from kwargs so they are passed to AgentRunContext
        # for the child agent run (e.g. delegation_depth from
        # BackgroundTaskCapability._task_async).
        deps = kwargs.pop("deps", None)
        # Extract resume parameters — only set by resume_session().
        cached_elicitation_responses: dict[str, Any] | None = kwargs.pop(
            "cached_elicitation_responses", None
        )
        deferred_tool_results: Any = kwargs.pop("deferred_tool_results", None)
        message_history: list[ModelMessage] | None = kwargs.pop("message_history", None)
        agent = await self.sessions.get_or_create_session_agent(
            session_id, input_provider=input_provider
        )
        if agent is None:
            return
        # Flatten prompts: if any prompt is a list (multimodal content),
        # preserve structure; otherwise join strings.
        match prompts:
            case []:
                content: str | list[Any] = ""
            case [single]:
                content = single
            case _:
                content = flatten_prompts(prompts)

        run_id = session.current_run_id
        if run_id is not None:
            # Active run — steer and use EventBus
            run_handle = self.sessions._runs.get(run_id)
            if run_handle is not None:
                run_handle.steer(content)
            queue = await self.event_bus.subscribe(session_id, scope=scope)
            try:
                while True:
                    try:
                        event = await queue.get()
                    except asyncio.QueueShutDown:
                        break
                    yield event.event
                    if isinstance(event.event, StreamCompleteEvent | RunErrorEvent):
                        break
            finally:
                await self.event_bus.unsubscribe(session_id, queue)
            return

        # No active run — create RunHandle and yield from start().
        # Acquire _request_lock to prevent race with concurrent
        # receive_request() → _start_run_handle() on the same session.
        # Without this, both paths could see current_run_id is None and
        # create overlapping RunHandles.
        async with session._request_lock:
            # Re-check current_run_id inside the lock — another caller
            # may have created a run while we waited.
            if session.current_run_id is not None:
                # Active run appeared — steer and use EventBus
                run_handle = self.sessions._runs.get(session.current_run_id)
                if run_handle is not None:
                    run_handle.steer(content)
                queue = await self.event_bus.subscribe(session_id, scope=scope)
                try:
                    while True:
                        try:
                            event = await queue.get()
                        except asyncio.QueueShutDown:
                            break
                        yield event.event
                        if isinstance(event.event, StreamCompleteEvent | RunErrorEvent):
                            break
                finally:
                    await self.event_bus.unsubscribe(session_id, queue)
                return

            # Also subscribe to EventBus so that events published by tools
            # during turn execution (e.g. SpawnSessionStart from task() →
            # create_child_session()) are delivered to the consumer, not
            # just events yielded directly by start().
            run_handle = self._create_run_handle(
                session,
                agent,
                session_id,
                deps=deps,
                cached_elicitation_responses=cached_elicitation_responses,
                deferred_tool_results=deferred_tool_results,
                message_history=message_history,
            )
            self.event_bus.clear_replay_buffer(session_id)
            bus_queue = await self.event_bus.subscribe(session_id, scope=scope)
            gen = run_handle.start(content)
        # Lock released — the run is now registered and can be steered
        # by concurrent receive_request() calls.
        try:
            async for evt in gen:
                # Drain any tool-published events from EventBus before
                # yielding the start() event. This ensures SpawnSessionStart
                # and similar events appear before the StreamCompleteEvent.
                with contextlib.suppress(asyncio.QueueEmpty):
                    while True:
                        envelope = bus_queue.get_nowait()
                        yield envelope.event
                yield evt
                if isinstance(evt, StreamCompleteEvent | RunErrorEvent):
                    break
        finally:
            # gen.aclose() and subsequent cleanup may raise CancelledError
            # (a BaseException, not caught by ``except Exception``) or
            # RuntimeError from pydantic-ai's anyio cancel scope cleanup
            # during GeneratorExit. Use save-and-re-raise so cleanup steps
            # always run (run_id cleared, handle removed) and CancelledError
            # is re-raised.
            _cancelled: asyncio.CancelledError | None = None
            try:
                await gen.aclose()
            except asyncio.CancelledError as e:
                _cancelled = e
            except Exception:
                logger.exception("Failed to close run generator")
            try:
                await self.event_bus.unsubscribe(session_id, bus_queue)
            except asyncio.CancelledError as e:
                if _cancelled is None:
                    _cancelled = e
            except Exception:
                logger.exception("Failed to unsubscribe from EventBus")
            session.current_run_id = None
            self.sessions._runs.pop(run_handle.run_id, None)
            if _cancelled is not None:
                raise _cancelled
