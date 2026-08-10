"""Run lifecycle mixin for SessionController.

Extracted from session_controller.py as part of the session-debt-cleanup file split.
Contains run handle creation, message routing, and run lifecycle methods.

In the per-prompt RunHandle model, each RunHandle executes exactly one
turn. ``_consume_run()`` chains RunHandles by checking ``prompt_queue``
after each turn terminates.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
import uuid

import logfire
from opentelemetry.context import attach, detach

from wolfharness.agents.context import AgentRunContext
from wolfharness.agents.events import (
    RunErrorEvent,
    RunFailedEvent,
    UserMessageInsertedEvent,
)
from wolfharness.lifecycle.comm_channel import ProtocolChannel
from wolfharness.log import get_logger
from wolfharness.observability.spans import safe_span
from wolfharness.orchestrator.run import RunHandle, inject_cancelled_tool_results


if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

    from wolfharness.agents.base_agent import BaseAgent
    from wolfharness.delegation import AgentPool
    from wolfharness.orchestrator.event_bus import EventBus
    from wolfharness.orchestrator.session_controller import SessionState


logger = get_logger(__name__)


class SessionControllerRunsMixin:
    """Mixin providing run lifecycle and message routing methods for SessionController.

    Attributes:
        pool: The agent pool (provided by SessionController).
        _sessions: Active sessions dict (provided by SessionController).
        _runs: Active run handles (provided by SessionController).
        _lock: Global lock (provided by SessionController).
        _event_bus: Event bus for cross-turn events (provided by SessionController).
        _background_tasks: Background task references (provided by SessionController).
    """

    pool: AgentPool[Any]
    _sessions: dict[str, SessionState]
    _runs: dict[str, RunHandle]
    _lock: asyncio.Lock
    _event_bus: EventBus | None
    _background_tasks: set[asyncio.Task[Any]]

    def get_session(self, session_id: str) -> SessionState | None: ...

    async def _emit_user_message_inserted(
        self,
        session_id: str,
        content: str | list[Any],
        delivery: str,
        source: str,
        message_id: str | None = None,
        meta: Any = None,
    ) -> None:
        """Publish ``UserMessageInsertedEvent`` to the EventBus.

        Wraps emission in a ``logfire.span`` to prevent orphan traces.
        Catches all exceptions and logs a warning — emission failures
        must never break the routing path.

        When a ``ProtocolChannel`` is available on the active run handle,
        the event is published through the channel (which journals before
        publishing to the EventBus) instead of direct ``EventBus.publish()``.
        This ensures the event is journaled for crash-recovery replay and
        avoids double-publish when the channel already publishes to the
        EventBus.

        For idle sessions (no active run), the event is published directly
        to the EventBus (existing behavior).

        Args:
            session_id: The session the message was inserted into.
            content: Message content (text or multi-modal part list).
            delivery: ``"initial"``, ``"steer"``, or ``"followup"``.
            source: ``"accepted"`` (accept-time routing display).
            message_id: Optional message ID; auto-generated if ``None``.
            meta: Optional protocol-specific metadata for rich user message
                display. When set, protocol event consumers use it to
                reconstruct the full user message instead of falling back
                to text-only ``content``.
        """
        with logfire.span(
            "event.user_message_inserted.emit",
            session_id=session_id,
            delivery=delivery,
            source=source,
        ):
            try:
                from wolfharness.utils.identifiers import ascending

                event = UserMessageInsertedEvent(
                    session_id=session_id,
                    message_id=message_id or ascending("message"),
                    content=content,
                    delivery=delivery,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                    source=source,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                    meta=meta,
                )
                if self._event_bus is None:
                    return

                # P2: When a ProtocolChannel is available on the active
                # run, route through the channel so the event is journaled
                # and published to the EventBus by the channel (avoiding
                # double-publish).
                comm_channel = self._get_protocol_channel(session_id)
                if comm_channel is not None:
                    await comm_channel.publish(event)
                    return

                # Fall back: direct EventBus publish (idle session).
                await self._event_bus.publish(session_id, event)
            except Exception:
                logger.warning(
                    "Failed to emit UserMessageInsertedEvent",
                    exc_info=True,
                )

    def _get_protocol_channel(self, session_id: str) -> ProtocolChannel | None:
        """Return the ProtocolChannel for the active session, if available.

        Checks the session's ``_comm_channel`` field (set by
        ``SessionController`` when creating a ``RunHandle`` for protocol
        server sessions). Returns ``None`` if the session is idle, the
        session is missing, or the CommChannel is not a ProtocolChannel.

        Args:
            session_id: The session to check.

        Returns:
            The active ``ProtocolChannel`` instance, or ``None``.
        """
        session = self.get_session(session_id)
        if session is None or session.current_run_id is None:
            return None
        comm_channel = session._comm_channel
        if isinstance(comm_channel, ProtocolChannel):
            return comm_channel
        return None

    async def _consume_run(self, run_handle: RunHandle, initial_prompt: str | list[Any]) -> None:  # noqa: PLR0915
        """Drive RunHandle execution to completion, chaining prompts.

        In the per-prompt model, each RunHandle executes exactly one turn
        and terminates. This method drains the generator, then checks
        ``prompt_queue`` for queued followup prompts. If non-empty, it
        creates a new RunHandle and chains to the next turn.

        The ``_request_lock`` is held during the chaining check to prevent
        ``_route_message()`` from creating a concurrent RunHandle.

        Args:
            run_handle: The initial run handle whose ``start()`` to consume.
            initial_prompt: The first user prompt (text or structured content).
        """
        # Attach the per-run trace context so all spans within this run
        # (turn.native, tools, notifications, etc.) are children of
        # run.message and share the same trace_id.
        ctx_token = attach(run_handle._run_context) if run_handle._run_context is not None else None
        try:
            with safe_span(
                "session.consume_run",
                session_id=run_handle.session_id,
                run_id=run_handle.run_id,
            ):
                session = self.get_session(run_handle.session_id)
                current_prompt: str | list[Any] = initial_prompt
                current_handle = run_handle
                while True:
                    gen = current_handle.start(current_prompt)
                    turn_failed = False
                    try:
                        async for _event in gen:
                            pass
                    except Exception as exc:
                        logger.exception(
                            "RunHandle.start() raised for run_id=%s session_id=%s",
                            current_handle.run_id,
                            current_handle.session_id,
                        )
                        error_event = RunErrorEvent(
                            message=f"{type(exc).__name__}: {exc}",
                            run_id=current_handle.run_id,
                            agent_name=(
                                current_handle.agent.name
                                if current_handle.agent is not None
                                else current_handle.agent_type
                            ),
                        )
                        if self._event_bus is not None:
                            await self._event_bus.publish(current_handle.session_id, error_event)
                            await self._event_bus.publish(
                                current_handle.session_id,
                                RunFailedEvent(
                                    run_id=current_handle.run_id,
                                    session_id=current_handle.session_id,
                                    exception=exc,
                                ),
                            )
                        turn_failed = True

                        # Notify parent (lead) session if this is a team
                        # member that crashed.  Skip when the session is
                        # being closed (team_delete, shutdown_request, etc.)
                        # — that's a normal shutdown, not a crash.
                        if session is not None and not session.is_closing:
                            await self._notify_lead_of_member_crash(session, exc)

                    # Generator terminated naturally — clean up this RunHandle.
                    self._runs.pop(current_handle.run_id, None)

                    if turn_failed:
                        # On error, do NOT chain — mark idle and break.
                        if session is not None:
                            async with session._request_lock:
                                if session.current_run_id == current_handle.run_id:
                                    session.set_current_run_id(None)
                        break

                    # Check prompt_queue for chained prompts (holding _request_lock
                    # to prevent _route_message() from racing).
                    if session is None:
                        break
                    async with session._request_lock:
                        if session.current_run_id == current_handle.run_id:
                            session.set_current_run_id(None)
                        if session.is_closing:
                            break  # Session is closing — don't chain.
                        if session.prompt_queue.empty():
                            break  # No more prompts, session goes idle.
                        try:
                            next_prompt = session.prompt_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        # Messages in prompt_queue were already displayed by
                        # _route_message() — no need to emit again.
                        # Create a new RunHandle for the next prompt.
                        agent = current_handle.agent
                        if agent is None:
                            break
                        current_handle = self._create_per_prompt_handle(session, agent, next_prompt)
                        current_prompt = next_prompt
                        # Loop continues — execute the next turn.
        finally:
            if ctx_token is not None:
                detach(ctx_token)
            if run_handle._run_span is not None:
                run_handle._run_span.end()

    async def _notify_lead_of_member_crash(
        self,
        session: SessionState,
        exc: BaseException,
    ) -> None:
        """Notify the lead (parent) session when a team member crashes.

        Routes a concise notification through the unified ``_route_message``
        path with ``source="accepted"`` so it appears in the lead's conversation
        history and the lead's LLM can act on it in the next turn.

        Silently skips if the session is not a team member, has no parent,
        or the parent session is unavailable.  Any notification failure is
        logged as a warning and never re-raised.
        """
        parent_session_id = session.parent_session_id
        if parent_session_id is None:
            return
        if session.metadata.get("team_role") != "member":
            return
        member_name = session.metadata.get("team_member_name", session.agent_name)
        error_detail = f"{type(exc).__name__}: {exc}"
        notification = (
            f'\u26a0\ufe0f Member "{member_name}" exited abnormally:'
            f" {error_detail}. Use team_status to check details."
        )
        try:
            parent_session = self.get_session(parent_session_id)
            if parent_session is None or parent_session.is_closing:
                return
            parent_agent = await self.get_or_create_session_agent(  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
                parent_session_id,
            )
            if parent_agent is None:
                return
            await self._route_message(
                parent_session,
                parent_agent,
                parent_session_id,
                notification,
                priority="when_idle",
                source="accepted",
            )
        except Exception:
            logger.warning(
                "Failed to notify lead session %s of member %s abnormal exit",
                parent_session_id,
                member_name,
                exc_info=True,
            )

    def _create_per_prompt_handle(
        self,
        session: SessionState,
        agent: BaseAgent[Any, Any],
        prompt: str | list[Any],
    ) -> RunHandle:
        """Create a new RunHandle for a single prompt (chaining).

        Bridges ``_message_history`` from ``agent.conversation`` and
        passes SessionState lifecycle dimensions. This is the per-prompt
        creation path used by ``_consume_run()`` chaining.

        Args:
            session: The session state.
            agent: The agent instance.
            prompt: The prompt for this turn.

        Returns:
            A newly created and registered RunHandle.
        """
        # Bridge agent.conversation → list[ModelMessage]
        model_messages: list[ModelMessage] = []
        conversation = agent.conversation
        if conversation is not None:
            for chat_msg in conversation.get_history():
                model_messages.extend(chat_msg.messages)
        model_messages = inject_cancelled_tool_results(model_messages)

        run_ctx = AgentRunContext(
            session_id=session.session_id,
            event_bus=self._event_bus,
        )

        run_handle = RunHandle(
            run_id=uuid.uuid4().hex,
            session_id=session.session_id,
            agent_type=agent.AGENT_TYPE,
            agent=agent,
            event_bus=self._event_bus,
            session=session,
            run_ctx=run_ctx,
            _message_history=model_messages,
            _host_context=session._host_context,
            _agent_registry=session._agent_registry,
        )
        self._runs[run_handle.run_id] = run_handle
        session.set_current_run_id(run_handle.run_id)
        return run_handle

    @logfire.instrument("session.start_run_handle")
    def _start_run_handle(
        self,
        session: SessionState,
        agent: BaseAgent[Any, Any],
        session_id: str,
        content: str | list[Any],
        *,
        deps: Any = None,
        message_id: str | None = None,
    ) -> str | None:
        """Create, register, and launch a RunHandle for a single prompt.

        In the per-prompt model, the RunHandle executes one turn and
        terminates. ``_consume_run()`` handles chaining via
        ``prompt_queue``.

        Lifecycle dimensions are sourced from ``SessionState`` (initialized
        in ``get_or_create_session_agent()``). The initial prompt is passed
        directly to ``start()``.

        Args:
            session: The session state.
            agent: The agent instance (native or ACP).
            session_id: The session identifier.
            content: The initial prompt (text or structured content blocks).
            deps: Optional dependencies to pass to the agent run context.
            message_id: Optional message ID for the initial prompt.

        Returns:
            The ``message_id`` string on success, ``None`` if the handle
            is closing.
        """
        # Bridge agent.conversation → list[ModelMessage]
        model_messages: list[ModelMessage] = []
        conversation = agent.conversation
        if conversation is not None:
            for chat_msg in conversation.get_history():
                model_messages.extend(chat_msg.messages)
        model_messages = inject_cancelled_tool_results(model_messages)

        event_bus = self._event_bus
        run_ctx = AgentRunContext(session_id=session_id, event_bus=event_bus, deps=deps)

        # Reset the agent's _cancelled flag from any prior run.
        agent._cancelled = False

        run_handle = RunHandle(
            run_id=uuid.uuid4().hex,
            session_id=session_id,
            agent_type=agent.AGENT_TYPE,
            agent=agent,
            event_bus=event_bus,
            session=session,
            run_ctx=run_ctx,
            _message_history=model_messages,
            _host_context=session._host_context,
            _agent_registry=session._agent_registry,
        )

        # Create per-run root span as a new trace root (not inheriting
        # the caller's context). This ensures each run gets its own
        # trace_id, independent of the HTTP request that triggered it.
        from opentelemetry import trace as otel_trace
        from opentelemetry.context import Context

        tracer = otel_trace.get_tracer("wolfharness.run")
        run_span = tracer.start_span(
            "run.message",
            context=Context(),  # Empty context → new trace, no parent
            attributes={
                "session.id": session_id,
                "run.id": run_handle.run_id,
                "agent.type": agent.AGENT_TYPE,
            },
        )
        run_handle._run_span = run_span
        run_handle._run_context = otel_trace.set_span_in_context(run_span)

        self._runs[run_handle.run_id] = run_handle
        session.set_current_run_id(run_handle.run_id)

        # Generate message_id for the initial prompt.
        from wolfharness.lifecycle.types import Feedback

        fb_kwargs: dict[str, Any] = {}
        if message_id is not None:
            fb_kwargs["message_id"] = message_id
        if isinstance(content, list):
            fb = Feedback(content="", is_steer=False, content_blocks=content, **fb_kwargs)
        else:
            fb = Feedback(content=content, is_steer=False, **fb_kwargs)
        mid = fb.message_id

        task = asyncio.create_task(self._consume_run(run_handle, content))
        # Keep a strong reference to prevent GC from destroying the task.
        self._background_tasks.add(task)

        def _on_run_done(t: asyncio.Task[Any], rid: str = run_handle.run_id) -> None:
            self._background_tasks.discard(t)
            if not t.cancelled() and t.exception() is not None:
                logger.error(
                    "Background run task failed for run_id=%s: %s",
                    rid,
                    t.exception(),
                )
            self._cleanup_run(rid)

        task.add_done_callback(_on_run_done)
        return mid

    @logfire.instrument("session.route_message")
    async def _route_message(
        self,
        session: SessionState,
        agent: BaseAgent[Any, Any],
        session_id: str,
        content: str | list[Any],
        *,
        priority: str = "when_idle",
        deps: Any = None,
        message_id: str | None = None,
        delivery: str | None = None,
        meta: Any = None,
        source: str = "accepted",
    ) -> str | None:
        """Route a message to the appropriate handler based on session state.

        Idle sessions create a RunHandle via :meth:`_start_run_handle`.
        Busy sessions call ``RunHandle.steer()`` (``"asap"``) or enqueue
        to ``SessionState.prompt_queue`` (``"when_idle"``).

        Before the routing action, publishes ``UserMessageInsertedEvent``
        to the EventBus (if available) so protocol frontends can display
        the user message.

        Args:
            session: The live session state (must already exist).
            agent: The resolved agent instance for this session.
            session_id: Target session identifier.
            content: Message / prompt content (text or structured content
                blocks).
            priority: ``"when_idle"`` to queue, ``"asap"`` to inject.
            deps: Optional dependencies for the agent run context.
            message_id: Optional message ID.
            delivery: Optional delivery label (``"initial"``, ``"steer"``,
                ``"followup"``). If ``None``, inferred from session state
                and priority.
            meta: Optional protocol-specific metadata carried through to
                ``UserMessageInsertedEvent`` for rich user message display.
            source: Originator of the message — ``"accepted"`` (default)
                for all message sources including protocol handler
                requests and team-mode messages.
                coordination messages. Passed to
                ``UserMessageInsertedEvent.source``.

        Returns:
            The ``message_id`` string on success, ``None`` for rejection.
        """
        resolved = {"steer": "asap", "followup": "when_idle"}.get(priority, priority)
        # Generate message_id once so both _emit_user_message_inserted() and
        # steer()/followup() share the same ID for dedup.
        if message_id is None:
            from wolfharness.utils.identifiers import ascending

            message_id = ascending("message")
        async with session._request_lock:
            if session.closing or session.is_closing:
                return None
            # Stale-run detection: if current_run_id points to a missing
            # or completed run, clear it and start a new run.
            if session.current_run_id is not None:
                existing_run = self._runs.get(session.current_run_id)
                if existing_run is None or existing_run.complete_event.is_set():
                    session.set_current_run_id(None)
            if session.current_run_id is None:
                # Idle session — initial delivery.
                inferred_delivery = delivery or "initial"
                await self._emit_user_message_inserted(
                    session_id,
                    content,
                    delivery=inferred_delivery,
                    source=source,
                    message_id=message_id,
                    meta=meta,
                )
                # Per-run trace context is created in _start_run_handle
                # and attached in _consume_run — no need to attach here.
                return self._start_run_handle(
                    session,
                    agent,
                    session_id,
                    content,
                    deps=deps,
                    message_id=message_id,
                )
            run = self._runs.get(session.current_run_id) if session.current_run_id else None
            if run is not None:
                if resolved == "asap":
                    # Steer — inject into active turn.
                    inferred_delivery = delivery or "steer"
                    await self._emit_user_message_inserted(
                        session_id,
                        content,
                        delivery=inferred_delivery,
                        source=source,
                        message_id=message_id,
                        meta=meta,
                    )
                    return run.steer(content, message_id=message_id, emit_user_message=False)
                # Followup: enqueue to SessionState.prompt_queue.
                inferred_delivery = delivery or "followup"
                await self._emit_user_message_inserted(
                    session_id,
                    content,
                    delivery=inferred_delivery,
                    source=source,
                    message_id=message_id,
                    meta=meta,
                )
                from wolfharness.lifecycle.types import Feedback

                fb_kwargs: dict[str, Any] = {}
                if message_id is not None:
                    fb_kwargs["message_id"] = message_id
                if isinstance(content, list):
                    fb = Feedback(content="", is_steer=False, content_blocks=content, **fb_kwargs)
                else:
                    fb = Feedback(content=content, is_steer=False, **fb_kwargs)
                session.prompt_queue.put_nowait(fb.content_blocks or fb.content)
                return None  # Queued — _wait_and_finalize will early-return
        return None

    def cancel_run_for_session(self, session_id: str) -> bool:
        """Cancel the active run for a session.

        Only cancels the run if the RunHandle is still active
        (``complete_event`` is not set). If the handle has already
        completed, the cancel is skipped.

        Args:
            session_id: The session whose run should be cancelled.

        Returns:
            ``True`` if cancellation was initiated, ``False`` if the
            session/run was not found or already completed.
        """
        session = self.get_session(session_id)
        if session is None:
            return False
        run_id = session.current_run_id
        if run_id is None:
            return False
        run_handle = self._runs.get(run_id)
        if run_handle is None:
            return False
        if run_handle.complete_event.is_set():
            logger.warning(
                "cancel_run_for_session: RunHandle %s already completed, skipping cancel",
                run_id,
            )
            return False
        run_handle.cancel()
        return True

    def revoke_inject(self, session_id: str, message_id: str) -> bool:
        """Revoke a pending steer or followup message by ID.

        In the per-prompt model, revocation is handled by
        ``SessionState.revoke()`` which cancels queued steer messages
        in ``feedback_queue``.

        Args:
            session_id: The session containing the message.
            message_id: The ID of the message to revoke.

        Returns:
            ``True`` if revoked or already gone (idempotent), ``False``
            if the session is not found.
        """
        session = self.get_session(session_id)
        if session is None:
            return False
        return session.revoke(message_id)

    async def wait_for_completion(self, session_id: str, timeout: float | None = 300) -> str:
        """Wait for the active run on a session to complete.

        In the per-prompt model, ``complete_event`` fires when the
        RunHandle's single-turn generator terminates. If
        ``_consume_run()`` chains to a new RunHandle, the caller must
        re-check ``session.current_run_id`` to detect the new turn.

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
        from wolfharness.orchestrator.session_controller import SessionNotFoundError

        if timeout is None:
            timeout = 300
        session = self.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        run_id = session.current_run_id
        if run_id is None:
            return session_id
        run_handle = self._runs.get(run_id)
        if run_handle is None:
            return session_id
        # In per-prompt model, complete_event fires when the single-turn
        # generator terminates — same semantic as the old _turn_complete_event.
        async with asyncio.timeout(timeout):
            await run_handle.complete_event.wait()
        return session_id

    def _cleanup_run(self, run_id: str) -> None:
        """Clean up a run after it completes.

        Removes the handle from _runs and signals completion. If the run
        was cancelled (``_consume_run`` skipped its ``prompt_queue`` check
        because ``CancelledError`` is ``BaseException``), drains any
        messages left in ``prompt_queue`` by starting a new run.

        Args:
            run_id: The run ID to clean up.
        """
        run_handle = self._runs.pop(run_id, None)
        if run_handle is not None:
            run_handle.complete_event.set()
            # Clear current_run_id if it still points to this run.
            session = self.get_session(run_handle.session_id)
            if session is not None and session.current_run_id == run_id:
                session.set_current_run_id(None)
                # After cancel, prompt_queue may have messages that arrived
                # during the race window (between task.cancel() and
                # complete_event.set()). _consume_run's prompt_queue check
                # is skipped because CancelledError is BaseException, not
                # caught by except Exception. Drain the queue here.
                self._drain_prompt_queue(session, run_handle)

    def _drain_prompt_queue(
        self,
        session: SessionState,
        old_handle: RunHandle,
    ) -> None:
        """Check prompt_queue and start a new run for the first queued message.

        Called from ``_cleanup_run`` when a run was cancelled with messages
        still in ``prompt_queue``. Creates a new ``RunHandle`` and
        ``_consume_run`` task for the first queued message. Additional
        messages are handled by ``_consume_run``'s normal chaining logic.

        In the normal (non-cancel) flow, ``_consume_run`` pops the run
        from ``_runs`` before returning, so ``_cleanup_run`` sees
        ``run_handle is None`` and never calls this method.

        Args:
            session: The session state with a potentially non-empty queue.
            old_handle: The cancelled run's handle (for agent reference).
        """
        # Do not start new runs on a closing/closed session.
        if session.closing or session.is_closing:
            return
        if session.prompt_queue.empty():
            return
        try:
            next_prompt = session.prompt_queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        agent = old_handle.agent
        if agent is None:
            # Put the message back — cannot process without an agent.
            session.prompt_queue.put_nowait(next_prompt)
            return
        # Reset the agent's _cancelled flag from the cancelled run.
        agent._cancelled = False
        new_handle = self._create_per_prompt_handle(session, agent, next_prompt)
        task = asyncio.create_task(self._consume_run(new_handle, next_prompt))
        self._background_tasks.add(task)

        def _on_chain_done(
            t: asyncio.Task[Any],
            rid: str = new_handle.run_id,
        ) -> None:
            self._background_tasks.discard(t)
            if not t.cancelled() and t.exception() is not None:
                logger.error(
                    "Background run task failed for run_id=%s: %s",
                    rid,
                    t.exception(),
                )
            self._cleanup_run(rid)

        task.add_done_callback(_on_chain_done)
