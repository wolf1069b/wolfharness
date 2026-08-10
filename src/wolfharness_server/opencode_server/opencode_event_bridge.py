"""Event conversion and EventBus subscription mixin.

Extracted from session_pool_integration.py as part of the session-debt-cleanup
file split. Contains the event bridge mixin that implements the
ProtocolEventConsumerMixin hooks for OpenCodeSessionPoolIntegration,
handling event conversion, EventBus subscription, and the event consumer
lifecycle.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

from wolfharness.agents.events.events import (
    CustomEvent,
    RunErrorEvent,
    RunFailedEvent,
    RunStartedEvent,
    SpawnSessionStart,
    StreamCompleteEvent,
    UserMessageInsertedEvent,
)
from wolfharness.log import get_logger
from wolfharness.utils import identifiers as identifier
from wolfharness.utils.time_utils import now_ms
from wolfharness_server.opencode_server.event_adapter import OpenCodeEventAdapter
from wolfharness_server.opencode_server.event_processor_context import (
    EventProcessorContext,
)
from wolfharness_server.opencode_server.models import (
    AssistantMessage,
    MessageAbortedError,
    MessageAbortedErrorData,
    MessagePath,
    MessageTime,
    MessageUpdatedEvent,
    MessageWithParts,
    PartUpdatedEvent,
    SessionErrorEvent,
    SessionStatus,
    StepStartPart,
    TimeCreated,
    TokenCache,
    Tokens,
    UserMessage,
)
from wolfharness_server.opencode_server.opencode_message_bridge import (
    append_message_to_session,
)
from wolfharness_server.opencode_server.opencode_session_routes import (
    ensure_session,
    set_session_status,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from wolfharness.agents.base_agent import BaseAgent
    from wolfharness.orchestrator.core import EventBus, EventEnvelope
    from wolfharness_server.opencode_server.state import ServerState


logger = get_logger(__name__)


class OpenCodeEventBridgeMixin:
    """Mixin providing event conversion and EventBus consumer lifecycle.

    Implements the ProtocolEventConsumerMixin hooks for
    OpenCodeSessionPoolIntegration, handling event subscription, event
    conversion to OpenCode SSE events, and parent/child session tracking.

    Attributes:
        session_pool: The SessionPool instance (provided by main class).
        server_state: The OpenCode server state (provided by main class).
        _contexts: Per-session EventProcessorContext instances
            (provided by main class).
        _adapters: Per-session OpenCodeEventAdapter instances
            (provided by main class).
        _message_registered: Per-session message registration flags
            (provided by main class).
        _child_to_parent: Mapping of child session IDs to parent session IDs
            (provided by main class).
        _child_spawns: Mapping of child session IDs to SpawnSessionStart events
            (provided by main class).
        _children_of: Mapping of parent session IDs to child session ID sets
            (provided by main class).
        _resume_contexts: Per-session serialized context data for resume
            (provided by main class).
        _pending_message_ids: Pending canonical message IDs from REST handlers
            (provided by main class).
    """

    session_pool: Any  # SessionPool
    server_state: ServerState
    _contexts: dict[str, EventProcessorContext]
    _adapters: dict[str, OpenCodeEventAdapter]
    _message_registered: dict[str, bool]
    _steer_split_ids: dict[str, set[str]]
    _child_to_parent: dict[str, str]
    _child_spawns: dict[str, SpawnSessionStart]
    _children_of: dict[str, set[str]]
    _resume_contexts: dict[str, dict[str, Any]]
    _pending_message_ids: dict[str, str]
    _pending_message_metadata: dict[str, dict[str, str | None]]

    # Crash recovery replay guard: when True, skip side-effectful actions
    # like steer split (events are being replayed, not live). Set by the
    # ProtocolEventConsumerMixin during crash recovery replay.
    _replaying: bool = False

    if TYPE_CHECKING:

        def get_session_context_data(self, session_id: str) -> dict[str, Any] | None: ...
        def set_session_context_data(self, session_id: str, data: dict[str, Any]) -> None: ...
        async def _create_subagent_tool_part(self, session_id: str, event: Any) -> Any: ...
        async def start_event_consumer(self, session_id: str) -> None: ...
        async def stop_event_consumer(self, session_id: str) -> None: ...
        async def _update_parent_toolpart(
            self,
            parent_session_id: str,
            child_session_id: str,
            spawn_event: SpawnSessionStart,
            event: Any,
        ) -> None: ...
        async def _update_parent_toolpart_error(
            self,
            parent_session_id: str,
            child_session_id: str,
            spawn_event: SpawnSessionStart,
            event: Any,
        ) -> None: ...

    @property
    def event_bus(self) -> EventBus:
        """Return the EventBus instance to subscribe to."""
        return cast("EventBus", self.session_pool.event_bus)

    def _get_subscription_scope(self) -> str:
        """Return the EventBus subscription scope.

        Overridden to "session" so that only the exact session's events are
        consumed. Child session events are handled by separate consumers
        created in response to SpawnSessionStart (see _on_spawn_session_start).

        Returns:
            The subscription scope string.
        """
        return "session"

    def _create_assistant_message(self, session_id: str) -> tuple[str, MessageWithParts]:
        """Create a fresh assistant message for a new turn.

        Resolves the canonical message_id from pending IDs (set by the REST
        handler), agent/model info from session state and pending metadata,
        and constructs a ``MessageWithParts.assistant`` instance.

        Args:
            session_id: The session to create the message for.

        Returns:
            A tuple of (assistant_msg_id, assistant_msg).
        """
        assistant_msg_id = self._pending_message_ids.pop(session_id, None)
        if assistant_msg_id is None:
            assistant_msg_id = identifier.ascending("message")

        agent_name = "wolfharness"
        model_id, provider_id = self.server_state.resolve_default_model_info()
        session_state = self.session_pool.sessions.get_session(session_id)
        if session_state is not None:
            agent_name = session_state.agent_name
            # Team member sessions: use the member's display name as the
            # mode so the TUI footer shows e.g. "Critic-1" instead of the
            # registered agent name (e.g. "artisan").
            team_member_name = session_state.metadata.get("team_member_name")
            if team_member_name is not None:
                agent_name = team_member_name
            # For child sessions that bypass route_message(), resolve model
            # from the session's agent instance instead of the server default
            # (which is the parent/lead agent's model).  _pending_message_metadata
            # is only set by route_message() (REST handler path); child sessions
            # created via create_child_session() never go through that path.
            if session_state.agent is not None:
                agent_model_name = cast("BaseAgent[Any, Any]", session_state.agent).model_name
                model_variants = self.server_state.model_variants
                from wolfharness_server.shared.model_utils import (
                    _extract_provider,
                    _find_variant_name,
                )

                variant_name = (
                    _find_variant_name(model_variants, agent_model_name)
                    if agent_model_name is not None
                    else None
                )
                if variant_name:
                    config = model_variants[variant_name]
                    model_id, provider_id = variant_name, _extract_provider(config)
                elif isinstance(agent_model_name, str) and ":" in agent_model_name:
                    provider, model = agent_model_name.split(":", 1)
                    model_id, provider_id = model, provider
        pending_meta = self._pending_message_metadata.pop(session_id, None)
        if pending_meta is not None:
            pending_model_id = pending_meta.get("model_id")
            if pending_model_id is not None:
                model_id = pending_model_id
            pending_provider_id = pending_meta.get("provider_id")
            if pending_provider_id is not None:
                provider_id = pending_provider_id

        assistant_msg = MessageWithParts.assistant(
            message_id=assistant_msg_id,
            session_id=session_id,
            time=MessageTime(created=now_ms()),
            agent_name=agent_name,
            model_id=model_id,
            parent_id=session_id,
            provider_id=provider_id,
            path=MessagePath(
                cwd=self.server_state.working_dir,
                root=self.server_state.working_dir,
            ),
            mode=agent_name,
        )
        return assistant_msg_id, assistant_msg

    async def _finalize_assistant_time(
        self,
        session_id: str,
        *,
        warn: bool = False,
    ) -> None:
        """Finalize time.completed on the assistant message if not already set.

        Used by StreamCompleteEvent, RunFailedEvent, and D1 reset to ensure
        the assistant message's time.completed is set before the turn ends
        or a new turn begins.

        Args:
            session_id: The session whose assistant message to finalize.
            warn: If True, log a warning when finalizing an incomplete turn
                (indicates StreamCompleteEvent was missed — D2 red flag).
        """
        ctx = self._contexts.get(session_id)
        if ctx is None:
            return
        info = ctx.assistant_msg.info
        if isinstance(info, AssistantMessage) and info.time.completed is None:
            if warn:
                logger.warning(
                    "Finalizing incomplete turn — StreamCompleteEvent missed",
                    session_id=session_id,
                    previous_message_id=ctx.assistant_msg_id,
                )
            info.time.completed = now_ms()
            await self.server_state.broadcast_event(MessageUpdatedEvent.create(info))
            await append_message_to_session(self.server_state, session_id, ctx.assistant_msg)

    async def _persist_assistant_message(self, session_id: str) -> None:
        """Persist the finalized assistant message to storage.

        Called after _finalize_assistant_time on StreamCompleteEvent and
        RunFailedEvent. This replaces the storage persistence previously
        done by _wait_and_finalize in the synchronous POST /message path.
        """
        from wolfharness_server.opencode_server.routes.message_routes import (
            persist_message_to_storage,
        )

        ctx = self._contexts.get(session_id)
        if ctx is None:
            return
        await persist_message_to_storage(self.server_state, ctx.assistant_msg, session_id)

    async def _persist_context_for_resume(self, session_id: str) -> None:
        """Serialize and store the EventProcessorContext for session resume.

        Called after StreamCompleteEvent and RunFailedEvent handlers (after
        P4's ``_message_registered`` reset). Serializes the current context
        via ``EventProcessorContext.serialize()`` and stores it via
        ``set_session_context_data()`` so that a subsequent
        ``_before_consumer_loop`` call (for a resumed session in the same
        process) can restore accumulated state.

        If serialization fails, logs an error and continues — the turn must
        not crash because of a resume-context serialization issue.
        """
        ctx = self._contexts.get(session_id)
        if ctx is None:
            return
        try:
            serialized = ctx.serialize()
            self.set_session_context_data(session_id, serialized)
        except Exception:
            logger.exception(
                "Failed to serialize EventProcessorContext for resume",
                session_id=session_id,
            )

    async def _before_consumer_loop(self, session_id: str) -> None:
        """Set up per-session context before the consumer loop starts.

        On a fresh session, creates a new :class:`EventProcessorContext` and
        :class:`OpenCodeEventAdapter`.  On a **resumed** session (where
        :meth:`set_session_context_data` was called before ``start_event_consumer``),
        restores the context from the serialized data so that accumulated
        text, tool parts, and tracking state are preserved.

        !!! note
            Restored contexts are NOT re-broadcast to the frontend.
            The parts already exist on the client from the original session.

        Args:
            session_id: The session whose consumer is starting.
        """
        # --- Check for persisted resume context -------------------------------
        resume_data = self.get_session_context_data(session_id)
        if resume_data:
            ctx = EventProcessorContext.deserialize(
                resume_data,
                state=self.server_state,
                working_dir=self.server_state.working_dir,
            )
            event_adapter = OpenCodeEventAdapter(ctx)
            self._contexts[session_id] = ctx
            self._adapters[session_id] = event_adapter
            # On resume, the assistant message was already registered in the
            # original session. Mark it as such so _handle_event does not
            # re-broadcast MessageUpdatedEvent.
            self._message_registered[session_id] = True
            logger.info(
                "Restored EventProcessorContext from persisted data",
                session_id=session_id,
                response_text_len=len(ctx.response_text),
            )
            return

        # --- Fresh context (original behaviour) -------------------------------
        # D14: Use the canonical message_id from the REST handler if available
        # instead of generating an independent one. This resolves the dual
        # assistant_msg_id split-message issue.
        assistant_msg_id, assistant_msg = self._create_assistant_message(session_id)

        ctx = EventProcessorContext(
            session_id=session_id,
            assistant_msg_id=assistant_msg_id,
            assistant_msg=assistant_msg,
            state=self.server_state,
            working_dir=self.server_state.working_dir,
        )
        event_adapter = OpenCodeEventAdapter(ctx)
        self._contexts[session_id] = ctx
        self._adapters[session_id] = event_adapter
        self._message_registered[session_id] = False

    async def _on_spawn_session_start(self, session_id: str, envelope: EventEnvelope) -> None:
        """Handle SpawnSessionStart by creating ToolPart and child consumer.

        Args:
            session_id: The session whose consumer received the event.
            envelope: The event envelope containing the spawn session start event.
        """
        try:
            event = envelope.event
            if not isinstance(event, SpawnSessionStart):
                return
            child_id = event.child_session_id
            if not child_id or child_id == session_id:
                return

            ctx = self._contexts.get(session_id)
            if ctx is None:
                return

            # Best-effort: make child session visible in protocol state.
            # Failure here (e.g. incomplete mock, storage error) must not
            # block ToolPart creation or assistant message registration.
            try:
                await self._ensure_child_session_visible(session_id, event)
            except Exception:
                logger.warning(
                    "Failed to ensure child session visible",
                    session_id=session_id,
                    child_session_id=event.child_session_id,
                    exc_info=True,
                )

            # Ensure assistant message is registered before ToolPart creation
            if not self._message_registered.get(session_id, False):
                await append_message_to_session(self.server_state, session_id, ctx.assistant_msg)
                await self.server_state.broadcast_event(
                    MessageUpdatedEvent.create(ctx.assistant_msg.info)
                )
                self._message_registered[session_id] = True

            # Distinguish parent vs child events.  Child events arrive
            # raw via scope="descendants".
            # Use envelope.source_session_id because many streaming events
            # (e.g.PartDeltaEvent from pydantic-ai) do not carry a
            # session_id attribute on the payload itself.

            tool_part = await self._create_subagent_tool_part(session_id, event)
            if tool_part is not None:
                subagent_key = f"{event.depth}:{event.source_name}:{child_id}"
                ctx.add_subagent_tool_part(subagent_key, tool_part)

            # Track parent-child relationship for later ToolPart updates
            self._child_to_parent[child_id] = session_id
            self._child_spawns[child_id] = event
            self._children_of.setdefault(session_id, set()).add(child_id)

            # Start dedicated consumer for the child session
            await self.start_event_consumer(child_id)
        except Exception:
            logger.exception(
                "SpawnSessionStart handler failed",
                session_id=session_id,
                child_session_id=getattr(envelope.event, "child_session_id", None),
            )

    async def _ensure_child_session_visible(
        self,
        parent_session_id: str,
        spawn_event: SpawnSessionStart,
    ) -> None:
        """Create OpenCode-visible child session scaffolding for task navigation.

        SessionPool owns the execution session. OpenCode also needs a session
        model so the TUI can open the child task immediately, even before the
        child stream emits its first token.

        Note: User message creation is NOT done here — the
        ``UserMessageInsertedEvent`` from ``_route_message()`` is the sole
        creator of user messages via the EventProcessor. Creating a duplicate
        user message here would cause double-rendering in the TUI.
        """
        child_session_id = spawn_event.child_session_id
        # Use source_name (ASCII registered name) for the @xxx subagent pattern
        # so the TUI regex @(\w+) subagent matches. display_name may contain
        # non-ASCII characters (e.g. Chinese) that \w cannot match.
        source_name = spawn_event.source_name
        # Team members get a richer title with team name and role
        team_id = spawn_event.metadata.get("team_id")
        if team_id is not None:
            team_name = spawn_event.metadata.get("team_name", "")
            team_role = spawn_event.metadata.get("team_role", "member")
            role_label = "Lead" if team_role == "lead" else "Member"
            team_prefix = f"Team '{team_name}' · {role_label} " if team_name else ""
            title = f"{team_prefix}(@{source_name} subagent)"
        else:
            title = f"(@{source_name} subagent)"
        session = await ensure_session(
            self.server_state,
            child_session_id,
            parent_id=parent_session_id,
            title=title,
        )
        # ensure_session does not update the title if the session already
        # exists (fast path / store-first path). Update it explicitly so the
        # TUI shows the enriched title (e.g. "Team 'X' · Member (@Y subagent)").
        if title is not None and session.title != title:
            session.title = title
            from wolfharness_server.opencode_server.models import SessionUpdatedEvent

            await self.server_state.broadcast_event(SessionUpdatedEvent.create(session))

    async def _handle_event(  # noqa: PLR0915
        self, session_id: str, envelope: EventEnvelope
    ) -> None:
        """Handle a single event from the EventBus.

        Distinguishes parent vs child events (via the child-to-parent mapping),
        updates parent ToolParts on child completion/error, and converts all
        events to OpenCode SSE events via the adapter.

        Args:
            session_id: The session whose consumer received the event.
            envelope: The event envelope to handle.
        """
        try:
            event = envelope.event

            # SpawnSessionStart is handled in _on_spawn_session_start; skip here
            if isinstance(event, SpawnSessionStart):
                return

            # Check if this event originated from a child session.
            # Child events are handled by the child consumer (started via
            # _on_spawn_session_start). We only process parent events here.
            is_child_event = envelope.source_session_id != session_id

            if is_child_event:
                # Child completion/error: update parent ToolPart, then skip.
                # Other child events (PartDeltaEvent etc.) are handled by the
                # dedicated child consumer started in _on_spawn_session_start.
                parent_id = self._child_to_parent.get(envelope.source_session_id)
                if parent_id is not None:
                    spawn = self._child_spawns.get(envelope.source_session_id)
                    if isinstance(event, StreamCompleteEvent) and spawn is not None:
                        await self._update_parent_toolpart(
                            parent_session_id=parent_id,
                            child_session_id=envelope.source_session_id,
                            spawn_event=spawn,
                            event=event,
                        )
                    elif isinstance(event, RunErrorEvent) and spawn is not None:
                        await self._update_parent_toolpart_error(
                            parent_session_id=parent_id,
                            child_session_id=envelope.source_session_id,
                            spawn_event=spawn,
                            event=event,
                        )
                return

            # When scope="session", child events are received by the child
            # consumer itself (not the parent consumer). In that case,
            # session_id == envelope.source_session_id, so is_child_event is
            # False.  We still need to update the parent ToolPart when this
            # session is a child session.
            parent_id = self._child_to_parent.get(session_id)
            if parent_id is not None:
                spawn = self._child_spawns.get(session_id)
                if isinstance(event, StreamCompleteEvent) and spawn is not None:
                    await self._update_parent_toolpart(
                        parent_session_id=parent_id,
                        child_session_id=session_id,
                        spawn_event=spawn,
                        event=event,
                    )
                elif isinstance(event, RunErrorEvent) and spawn is not None:
                    await self._update_parent_toolpart_error(
                        parent_session_id=parent_id,
                        child_session_id=session_id,
                        spawn_event=spawn,
                        event=event,
                    )

            # Handle run lifecycle events for session status
            match event:
                case RunStartedEvent():
                    await set_session_status(
                        self.server_state, session_id, SessionStatus(type="busy")
                    )
                case StreamCompleteEvent():
                    await set_session_status(
                        self.server_state, session_id, SessionStatus(type="idle")
                    )
                    # D3: Finalize assistant message time.completed if not
                    # already set. The prompt_async path returns 204
                    # immediately and never sets time.completed on the
                    # assistant message, so the event bridge must do it here
                    # when StreamCompleteEvent arrives.
                    # NOTE: Do NOT reset _message_registered here. It must
                    # stay True so the next RunStartedEvent's D1 block fires
                    # to create a fresh assistant message. Resetting here
                    # would cause D1 to be skipped → all turns merge into
                    # one assistant message.
                    #
                    # Correct state machine:
                    #
                    #   Turn 1: RunStarted → D1 skip (first) → register → True
                    #           StreamComplete → finalize (keep True)
                    #   Turn 2: RunStarted → D1 fires (True) → new msg → False
                    #           → register → True
                    #           StreamComplete → finalize (keep True)
                    #
                    # If we reset to False at StreamComplete:
                    #   Turn 2: RunStarted → D1 skip (False) → register
                    #           with OLD msg_id → all turns merge → BUG
                    #
                    # NOTE: _finalize_assistant_time and _persist_assistant_message
                    # are called AFTER adapter.convert_event() below, because the
                    # EventProcessor (invoked by convert_event) populates
                    # ctx.input_tokens/output_tokens from msg.usage.  Calling
                    # finalize before convert_event would broadcast
                    # MessageUpdatedEvent with tokens=0, causing the TUI to
                    # show no token usage.
                case RunFailedEvent(exception=exc):
                    await set_session_status(
                        self.server_state, session_id, SessionStatus(type="idle")
                    )
                    # C3 fallback: If the agent crashed before any event
                    # triggered C3 registration, the assistant message was
                    # never appended to session state or broadcast via SSE.
                    # Register it now so _finalize_assistant_time can
                    # finalize and broadcast it.
                    if not self._message_registered.get(session_id, False):
                        ctx = self._contexts.get(session_id)
                        if ctx is not None:
                            await append_message_to_session(
                                self.server_state, session_id, ctx.assistant_msg
                            )
                            await self.server_state.broadcast_event(
                                MessageUpdatedEvent.create(ctx.assistant_msg.info)
                            )
                            self._message_registered[session_id] = True
                    # D3: Finalize time.completed for failed runs too,
                    # so the next turn's D1 reset doesn't log a
                    # false-positive warning about a missed
                    # StreamCompleteEvent.
                    await self._finalize_assistant_time(session_id)
                    # Set aborted error on the assistant message.
                    ctx = self._contexts.get(session_id)
                    if ctx is not None and isinstance(ctx.assistant_msg.info, AssistantMessage):
                        info = ctx.assistant_msg.info
                        if info.error is None:
                            if isinstance(exc, asyncio.CancelledError):
                                reason = "Request cancelled by user"
                            elif isinstance(exc, Exception):
                                reason = f"Error: {exc}"
                            else:
                                reason = "Run failed"
                            info.error = MessageAbortedError(
                                data=MessageAbortedErrorData(message=reason)
                            )
                            await self.server_state.broadcast_event(
                                MessageUpdatedEvent.create(info)
                            )
                    if isinstance(exc, Exception) and not isinstance(exc, asyncio.CancelledError):
                        await self.server_state.broadcast_event(
                            SessionErrorEvent.from_exception(exc, session_id=session_id)
                        )
                    # Persist the aborted assistant message to storage.
                    await self._persist_assistant_message(session_id)
                    # NOTE: Do NOT reset _message_registered here (same
                    # reasoning as StreamCompleteEvent path). The next
                    # RunStartedEvent's D1 block handles the reset.
                    # P3: Serialize context for resume, same as
                    # StreamCompleteEvent path.
                    await self._persist_context_for_resume(session_id)
                case RunErrorEvent(message=error_msg):
                    # RunErrorEvent is a terminal event (no trailing
                    # StreamCompleteEvent). Without this case, the session
                    # status stays "busy" forever because the match block
                    # never sets it to "idle".
                    #
                    # The EventProcessor.process() already yields
                    # SessionErrorEvent for RunErrorEvent, so we do NOT
                    # broadcast it here — only the session status and
                    # assistant message cleanup are our responsibility.
                    await set_session_status(
                        self.server_state, session_id, SessionStatus(type="idle")
                    )
                    # C3 fallback: same as RunFailedEvent — if no event
                    # triggered C3 registration, register now so
                    # _finalize_assistant_time can finalize and broadcast.
                    if not self._message_registered.get(session_id, False):
                        ctx = self._contexts.get(session_id)
                        if ctx is not None:
                            await append_message_to_session(
                                self.server_state, session_id, ctx.assistant_msg
                            )
                            await self.server_state.broadcast_event(
                                MessageUpdatedEvent.create(ctx.assistant_msg.info)
                            )
                            self._message_registered[session_id] = True
                    # D3: Finalize time.completed for errored runs too.
                    await self._finalize_assistant_time(session_id)
                    # Set aborted error on the assistant message, using
                    # the RunErrorEvent's message as the error reason.
                    ctx = self._contexts.get(session_id)
                    if ctx is not None and isinstance(ctx.assistant_msg.info, AssistantMessage):
                        info = ctx.assistant_msg.info
                        if info.error is None:
                            info.error = MessageAbortedError(
                                data=MessageAbortedErrorData(message=error_msg)
                            )
                            await self.server_state.broadcast_event(
                                MessageUpdatedEvent.create(info)
                            )
                    # Persist the aborted assistant message to storage.
                    await self._persist_assistant_message(session_id)
                    # NOTE: Do NOT reset _message_registered here (same
                    # reasoning as StreamCompleteEvent/RunFailedEvent).
                    # P3: Serialize context for resume.
                    await self._persist_context_for_resume(session_id)
                case _:
                    pass

            # C4: CustomEvent wraps SSE broadcast events (e.g.
            # SessionCreatedEvent) republished from the OpenCodeEventBridge.
            # These are not real agent events and must NOT trigger assistant
            # message registration. Only skip bridge-wrapped CustomEvents
            # (source="opencode_event_bridge"); tool-emitted CustomEvents
            # (source=None or tool name) may carry meaningful payload and
            # should fall through to adapter processing.
            if isinstance(event, CustomEvent) and event.source == "opencode_event_bridge":
                return

            ctx = self._contexts.get(session_id)
            if ctx is None:
                return

            # D1: On RunStartedEvent for a subsequent turn (consumer already
            # running from turn 1), reset per-turn state so turn 2 gets a
            # fresh assistant message ID instead of reusing turn 1's.
            # _before_consumer_loop() only runs once (consumer start is
            # idempotent), so turns 2+ need this explicit reset.
            if isinstance(event, RunStartedEvent) and self._message_registered.get(
                session_id, False
            ):
                # D2/D3: Finalize previous turn's assistant message if
                # StreamCompleteEvent was missed or not yet processed. This
                # prevents the previous turn's time.completed from being lost
                # when the D1 reset creates a new message. The warning makes
                # the D2 red flag (running turn killed by new turn) visible.
                await self._finalize_assistant_time(session_id, warn=True)

                assistant_msg_id, assistant_msg = self._create_assistant_message(session_id)
                ctx.assistant_msg_id = assistant_msg_id
                ctx.assistant_msg = assistant_msg
                # Reset per-turn mutable tracking state
                ctx.response_text = ""
                ctx.text_part = None
                ctx.reasoning_part = None
                ctx.tool_parts.clear()
                ctx.tool_outputs.clear()
                ctx.tool_inputs.clear()
                ctx.subagent_tool_parts.clear()
                ctx.is_errored = False
                ctx.input_tokens = 0
                ctx.output_tokens = 0
                ctx.total_cost = 0.0
                ctx.stream_start_ms = now_ms()

                self._message_registered[session_id] = False
                self._steer_split_ids.pop(session_id, None)

            # Update assistant message with real agent info from RunStartedEvent.
            # RunStartedEvent is the first event in a run and carries the real
            # agent_name from the RunLoop. This is more reliable than the session
            # state lookup in _before_consumer_loop (which may not have the
            # agent name for sessions created outside the REST handler).
            if isinstance(event, RunStartedEvent) and event.agent_name:
                msg_info = ctx.assistant_msg.info
                if isinstance(msg_info, AssistantMessage):
                    msg_info.agent = event.agent_name
                    # Team member sessions: prefer team_member_name over the
                    # registered agent name for the TUI footer display.
                    display_mode = event.agent_name
                    session_state = self.session_pool.sessions.get_session(session_id)
                    if session_state is not None:
                        team_member_name = session_state.metadata.get("team_member_name")
                        if team_member_name is not None:
                            display_mode = team_member_name
                    msg_info.mode = display_mode
            # NOTE: Do NOT overwrite ctx.assistant_msg_id from event.message_id.
            # NativeTurn generates its own UUID for _message_id (uuid4().hex)
            # which is different from the canonical assistant_msg_id generated
            # by the REST handler (identifier.ascending("message", ...)).
            # Overwriting causes a mismatch: parts get the NativeTurn UUID as
            # their message_id while the assistant message keeps the REST
            # handler's ID, so the UI cannot associate parts with the message.
            # The canonical assistant_msg_id from the REST handler is correct.

            # Steer split: When a steer UserMessageInsertedEvent arrives,
            # split the logical turn: finalize A1, create A2 with fresh ID.
            #
            # Only source="processed" events trigger the split. These are
            # processing-time events from EnqueuedMessagesEvent mapping.
            # source="accepted" events (fire-and-forget from steer()/followup())
            # are handled by the EventProcessor which creates UserMessage + SSE.
            # Dedup by message_id via _steer_split_ids prevents double splits
            # when both events fire for the same steer message.
            if (
                isinstance(event, UserMessageInsertedEvent)
                and event.delivery == "steer"
                and event.source == "processed"
                and not self._replaying
                and self._message_registered.get(session_id, False)
                and event.message_id not in self._steer_split_ids.setdefault(session_id, set())
            ):
                self._steer_split_ids[session_id].add(event.message_id)
                await self._finalize_assistant_time(session_id)

                assistant_msg_id, assistant_msg = self._create_assistant_message(session_id)
                ctx.assistant_msg_id = assistant_msg_id
                ctx.assistant_msg = assistant_msg
                # Reset per-turn mutable tracking state (same as D1 reset)
                ctx.response_text = ""
                ctx.text_part = None
                ctx.reasoning_part = None
                ctx.tool_parts.clear()
                ctx.tool_outputs.clear()
                ctx.tool_inputs.clear()
                ctx.subagent_tool_parts.clear()
                ctx.is_errored = False
                ctx.input_tokens = 0
                ctx.output_tokens = 0
                ctx.total_cost = 0.0
                ctx.stream_start_ms = now_ms()

                self._message_registered[session_id] = False

            # Register assistant message on first non-spawn, non-custom,
            # non-user-message-inserted event.
            # C3: The event bridge is the sole broadcast point for the assistant
            # message. This ensures the message is visible only when the agent
            # actually starts producing events, not before.
            # UserMessageInsertedEvent creates a user message, not an assistant
            # message, so it must not trigger assistant registration.
            # StreamCompleteEvent and RunFailedEvent are lifecycle finalizers
            # that already handled message finalization in the match block
            # above. They must not trigger re-registration (which would undo
            # P4's _message_registered reset).
            is_user_message_inserted = isinstance(event, UserMessageInsertedEvent)
            is_lifecycle_finalizer = isinstance(event, (StreamCompleteEvent, RunFailedEvent))
            if (
                not is_user_message_inserted
                and not is_lifecycle_finalizer
                and not self._message_registered.get(session_id, False)
            ):
                await append_message_to_session(self.server_state, session_id, ctx.assistant_msg)
                await self.server_state.broadcast_event(
                    MessageUpdatedEvent.create(ctx.assistant_msg.info)
                )
                # C3: Also broadcast a StepStartPart so the frontend sees the
                # step-start indicator when the agent actually begins work.
                step_start_part = StepStartPart(
                    id=identifier.ascending("part"),
                    message_id=ctx.assistant_msg_id,
                    session_id=session_id,
                )
                ctx.assistant_msg.parts.append(step_start_part)
                await self.server_state.broadcast_event(PartUpdatedEvent.create(step_start_part))
                self._message_registered[session_id] = True

            adapter = self._adapters.get(session_id)
            if adapter is None:
                return

            async for oc_event in adapter.convert_event(event):
                await self.server_state.broadcast_event(oc_event)

            # After adapter.convert_event for StreamCompleteEvent, the
            # EventProcessor has updated ctx.input_tokens/output_tokens and
            # ctx.total_cost from msg.usage.  Now finalize the assistant
            # message with the correct token/cost values so the TUI sees them
            # via MessageUpdatedEvent.
            if isinstance(event, StreamCompleteEvent):
                finalize_ctx = self._contexts.get(session_id)
                if finalize_ctx is not None and isinstance(
                    finalize_ctx.assistant_msg.info, AssistantMessage
                ):
                    info = finalize_ctx.assistant_msg.info
                    info.tokens = Tokens(
                        cache=TokenCache(read=0, write=0),
                        input=finalize_ctx.input_tokens,
                        output=finalize_ctx.output_tokens,
                        reasoning=0,
                    )
                    info.cost = finalize_ctx.total_cost
                await self._finalize_assistant_time(session_id)
                await self._persist_assistant_message(session_id)
                # P3: Serialize the EventProcessorContext and store it
                # so that a subsequent _before_consumer_loop (for a
                # resumed session in the same process) can restore the
                # accumulated state instead of creating a fresh context.
                await self._persist_context_for_resume(session_id)
        except Exception:
            logger.exception(
                "Event handler failed",
                session_id=session_id,
                event_type=type(envelope.event).__name__,
            )

    async def _after_consumer_loop(self, session_id: str) -> None:
        """Clean up per-session context after the consumer loop exits.

        Args:
            session_id: The session whose consumer has stopped.
        """
        # Stop any child consumers that were started from this session
        for child_id in list(self._children_of.get(session_id, [])):
            try:
                await self.stop_event_consumer(child_id)
            except Exception:
                logger.exception(
                    "Failed to stop child event consumer",
                    child_id=child_id,
                )
        self._children_of.pop(session_id, None)

        # Clean up per-session state
        self._contexts.pop(session_id, None)
        self._adapters.pop(session_id, None)
        self._message_registered.pop(session_id, None)
        self._child_to_parent.pop(session_id, None)
        self._child_spawns.pop(session_id, None)

    # ------------------------------------------------------------------
    # Backward-compatible wrappers (used by tests)
    # ------------------------------------------------------------------

    async def _start_event_consumer(self, session_id: str) -> None:
        """Backward-compatible wrapper for the mixin's start_event_consumer."""
        await self.start_event_consumer(session_id)
        logger.info("Started session-scoped event consumer", session_id=session_id)

    async def _stop_event_consumer(self, session_id: str) -> None:
        """Backward-compatible wrapper for the mixin's stop_event_consumer."""
        await self.stop_event_consumer(session_id)
        logger.info("Stopped session-scoped event consumer", session_id=session_id)

    async def subscribe_to_events(self, session_id: str) -> AsyncIterator[Any]:
        """Subscribe to session events and yield converted OpenCode events.

        Creates a minimal EventProcessorContext so that AgentPool events
        can be converted to OpenCode SSE events via OpenCodeEventAdapter.

        Args:
            session_id: The session to subscribe to.

        Yields:
            OpenCode Event objects.
        """
        assistant_msg_id = identifier.ascending("message")
        assistant_msg = MessageWithParts(
            info=UserMessage(
                id=assistant_msg_id,
                session_id=session_id,
                time=TimeCreated.now(),
            )
        )
        ctx = EventProcessorContext(
            session_id=session_id,
            assistant_msg_id=assistant_msg_id,
            assistant_msg=assistant_msg,
            state=self.server_state,
            working_dir=self.server_state.working_dir,
        )
        event_adapter = OpenCodeEventAdapter(ctx)
        event_stream = await self.session_pool.event_bus.subscribe(session_id)

        try:
            from wolfharness.orchestrator.core import drain_and_merge

            async for event in drain_and_merge(event_stream):
                async for oc_event in event_adapter.convert_event(event.event):
                    yield oc_event
        finally:
            await self.session_pool.event_bus.unsubscribe(session_id, event_stream)
