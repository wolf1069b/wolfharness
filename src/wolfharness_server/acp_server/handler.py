"""ACP Protocol Handler using SessionPool for session and turn management.

This module provides ``ACPProtocolHandler``, a protocol handler that delegates
ACP session lifecycle and prompt processing to the ``SessionPool`` orchestration
layer.

The handler bridges AgentPool's EventBus with the ACP protocol by running a
per-session event consumer loop that converts agent stream events to ACP
session updates.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, Literal
import uuid

import anyio

from acp.agent.acp_requests import ACPRequests
from acp.schema.capabilities import ClientCapabilities
from wolfharness.agents.events.events import ElicitationDeferredEvent, SpawnSessionStart
from wolfharness.log import get_logger
from wolfharness_server.acp_server.event_converter import ACPEventConverter, SubagentContext
from wolfharness_server.acp_server.input_provider import ACPInputProvider
from wolfharness_server.mixins import ConsumerShutdown, ProtocolEventConsumerMixin


if TYPE_CHECKING:
    from collections.abc import Sequence

    from acp import Client
    from acp.schema import ContentBlock, PromptResponse, StopReason
    from wolfharness.host.context import HostContext
    from wolfharness.orchestrator.core import EventBus, EventEnvelope
    from wolfharness_server.acp_server.session_manager import ACPSessionManager

logger = get_logger(__name__)


class ACPProtocolHandler(ProtocolEventConsumerMixin):
    """ACP protocol handler backed by SessionPool.

    Manages per-session event consumers that subscribe to the SessionPool's
    EventBus and forward converted events to the ACP client. Prompt handling
    is delegated to ``SessionPool.receive_request()``.

    Args:
        host_context: The host context containing the SessionPool.
        event_converter: Template converter used to derive per-session
            converters. The display mode is extracted from this instance.
        client: ACP client for sending session update notifications.
        client_capabilities: Client capabilities for elicitation support
            gating. If None, falls back to legacy request_permission.
        acp_agent: Reference to the ``AgentPoolACPAgent``, used for
            session resume operations that need agent-level context.
    """

    def __init__(
        self,
        host_context: HostContext,
        session_manager: ACPSessionManager,
        event_converter: ACPEventConverter,
        client: Client,
        client_capabilities: ClientCapabilities | None = None,
        acp_agent: Any = None,
    ) -> None:
        """Initialize the protocol handler."""
        super().__init__()
        self._host_context = host_context
        self.session_manager = session_manager
        self._event_converter_template = event_converter
        self.client = client
        self.client_capabilities = client_capabilities
        self._converters: dict[str, ACPEventConverter] = {}
        self._parent_of: dict[str, str] = {}
        self.acp_agent = acp_agent
        self._elicitation_tasks: dict[str, set[asyncio.Task[Any]]] = {}

    @property
    def event_bus(self) -> EventBus:
        """Return the EventBus instance to subscribe to."""
        session_pool = self._host_context.session_pool
        if session_pool is None:
            raise RuntimeError("SessionPool not available")
        return session_pool.event_bus

    def _get_subscription_scope(self) -> str:
        """Return the EventBus subscription scope.

        Overridden to "session" so that only the exact session's events are
        consumed.  Child session events are handled by separate consumers
        created in response to SpawnSessionStart (see _on_spawn_session_start).
        This prevents event interleaving when a parent and its background-task
        child run concurrently.

        Returns:
            The subscription scope string.
        """
        return "session"

    async def _on_spawn_session_start(self, session_id: str, envelope: EventEnvelope) -> None:
        """Start a dedicated consumer for the newly spawned child session.

        Background tasks (spawn_mechanism="task") are skipped since their
        events should remain server-side and not be streamed to the ACP client.
        Only sync subagents get a child consumer.

        Each child session gets its own converter via start_event_consumer,
        which creates a fresh ACPEventConverter in _before_consumer_loop.
        This replaces the old zed-specific forwarding pattern where child
        events were routed through the parent's converter.

        Args:
            session_id: The session whose consumer received the event.
            envelope: The event envelope containing the spawn session start event.
        """
        event = envelope.event
        if isinstance(event, SpawnSessionStart):
            child_sid = event.child_session_id
            if child_sid and child_sid != session_id:
                if (
                    event.spawn_mechanism == "task"
                    and self._event_converter_template.subagent_display_mode not in ("zed", "qwen")
                ):
                    return
                # Create child converter with subagent context
                client_supports_turn_complete = (
                    self.client_capabilities is not None
                    and self.client_capabilities.turn_complete is True
                )
                self._converters[child_sid] = ACPEventConverter(
                    subagent_display_mode=self._event_converter_template.subagent_display_mode,
                    raw_input_mode=self._event_converter_template.raw_input_mode,
                    client_supports_turn_complete=client_supports_turn_complete,
                    subagent_context=SubagentContext(
                        parent_tool_call_id=event.tool_call_id or "",
                        subagent_type=event.source_name or "",
                    ),
                )
                await self.start_event_consumer(child_sid)
                # Register parent-child relationship BEFORE starting closure
                # so the closure can safely pop the entry.
                self._parent_of[child_sid] = session_id
                done_event = self._consumer_done_events.get(child_sid)
                if done_event is not None:
                    task = asyncio.ensure_future(
                        self._await_child_and_notify(
                            parent_sid=session_id,
                            child_sid=child_sid,
                            done_event=done_event,
                        )
                    )
                    self._consumer_task_refs.append(task)
                else:
                    # Race: consumer already finished before we could grab
                    # the done_event. Notify immediately and clean up.
                    self._parent_of.pop(child_sid, None)
                    await self._notify_completed(parent_sid=session_id, child_sid=child_sid)

    async def _notify_completed(self, parent_sid: str, child_sid: str) -> None:
        """Send a subagent completion notification to the parent session.

        Looks up the parent session's converter and calls
        ``build_subagent_completed()`` to emit a ``ToolCallProgress``
        with ``status="completed"``, closing the tool call lifecycle
        started by ``SpawnSessionStart`` in zed mode.

        Args:
            parent_sid: The parent session that spawned the child.
            child_sid: The child session that has completed.
        """
        converter = self._converters.get(parent_sid)
        if converter is None:
            logger.debug(
                "Parent converter gone, skipping completion notification",
                parent_sid=parent_sid,
                child_sid=child_sid,
            )
            return
        try:
            async for update in converter.build_subagent_completed(child_session_id=child_sid):
                from acp.schema import SessionNotification

                notification = SessionNotification(
                    session_id=parent_sid,
                    update=update,
                )
                await self.client.session_update(notification)
        except (ConnectionResetError, BrokenPipeError):
            logger.debug(
                "Client disconnected during completion notification",
                parent_sid=parent_sid,
                child_sid=child_sid,
            )
        except Exception:
            logger.exception(
                "Failed to send subagent completion notification",
                parent_sid=parent_sid,
                child_sid=child_sid,
            )

    async def _await_child_and_notify(
        self,
        parent_sid: str,
        child_sid: str,
        done_event: anyio.Event,
    ) -> None:
        """Wait for a child consumer to finish, then notify the parent.

        Background closure that waits on the child session's
        ``done_event`` (set by the mixin's finally block when the
        consumer loop exits), then calls ``_notify_completed`` to
        deliver the completion notification to the parent session.

        Args:
            parent_sid: The parent session that spawned the child.
            child_sid: The child session to wait for.
            done_event: The child consumer's done event from
                ``_consumer_done_events``.
        """
        try:
            await done_event.wait()
            self._parent_of.pop(child_sid, None)
            await self._notify_completed(parent_sid, child_sid)
        except (ConnectionResetError, BrokenPipeError):
            logger.debug(
                "Client disconnected during child completion notification",
                child_sid=child_sid,
            )
        except Exception:
            logger.exception(
                "Error in child completion notification",
                child_sid=child_sid,
            )
        finally:
            task = asyncio.current_task()
            if task is not None:
                with contextlib.suppress(ValueError):
                    self._consumer_task_refs.remove(task)

    async def _before_consumer_loop(self, session_id: str) -> None:
        """Create per-session ACPEventConverter before loop starts.

        Args:
            session_id: The session whose consumer is starting.
        """
        if session_id in self._converters:
            return  # Already created by _on_spawn_session_start
        client_supports_turn_complete = (
            self.client_capabilities is not None and self.client_capabilities.turn_complete is True
        )
        converter = ACPEventConverter(
            subagent_display_mode=self._event_converter_template.subagent_display_mode,
            raw_input_mode=self._event_converter_template.raw_input_mode,
            client_supports_turn_complete=client_supports_turn_complete,
        )
        self._converters[session_id] = converter

    async def _handle_event(self, session_id: str, envelope: EventEnvelope) -> None:
        """Handle a single event from the EventBus.

        Args:
            session_id: The session whose consumer received the event.
            envelope: The event envelope to handle.

        Raises:
            ConsumerShutdown: When the ACP client connection is closed.
        """
        # Use envelope's source_session_id for routing (for child session routing)
        event_sid = envelope.source_session_id
        effective_sid = event_sid if event_sid else session_id

        # Intercept ElicitationDeferredEvent: send an actual elicitation/create
        # request to the ACP client (not just a notification). The client shows
        # a form, the user responds, and we resume the session with the response.
        if isinstance(envelope.event, ElicitationDeferredEvent):
            task = asyncio.create_task(
                self._handle_elicitation_deferred(effective_sid, envelope.event)
            )
            self._elicitation_tasks.setdefault(effective_sid, set()).add(task)

            def _discard_task(t: asyncio.Task[Any], sid: str = effective_sid) -> None:
                self._elicitation_tasks.get(sid, set()).discard(t)

            task.add_done_callback(_discard_task)
            return

        # Look up converter: try event's session first, fall back to consumer's session
        converter = self._converters.get(effective_sid) or self._converters.get(session_id)
        if converter is None:
            return

        try:
            async for update in converter.convert(envelope.event):
                from acp.schema import SessionNotification

                notification = SessionNotification(
                    session_id=effective_sid,
                    update=update,
                    field_meta=converter.subagent_meta,
                )
                await self.client.session_update(notification)
        except (ConnectionResetError, BrokenPipeError) as e:
            logger.debug(
                "Client connection closed gracefully",
                session_id=session_id,
                error=str(e),
            )
            raise ConsumerShutdown from e
        except anyio.ClosedResourceError as e:
            logger.debug(
                "Stream closed gracefully",
                session_id=session_id,
                error=str(e),
            )
            raise ConsumerShutdown from e
        except anyio.EndOfStream as e:
            logger.debug(
                "Stream closed gracefully",
                session_id=session_id,
                error=str(e),
            )
            raise ConsumerShutdown from e
        except Exception:
            logger.exception(
                "Failed to convert or send event",
                session_id=session_id,
                event_type=type(envelope.event).__name__,
            )

    async def _handle_elicitation_deferred(
        self,
        session_id: str,
        event: ElicitationDeferredEvent,
    ) -> None:
        """Send elicitation/create to ACP client and resume session with response.

        This runs as a background task so the event consumer loop is not blocked
        while waiting for the user to respond.

        Args:
            session_id: The session that has a pending elicitation.
            event: The elicitation deferred event with params.
        """
        from wolfharness.sessions.models import ElicitationResumePayload

        session_pool = self._host_context.session_pool
        if session_pool is None:
            logger.error(
                "Cannot handle elicitation: SessionPool not available",
                session_id=session_id,
            )
            return

        acp_requests = ACPRequests(client=self.client, session_id=session_id)

        try:
            raw_mode = event.mode or "form"
            if raw_mode == "url":
                elicitation_mode: Literal["form", "url"] = "url"
            else:
                elicitation_mode = "form"
            logger.info(
                "Elicitation: sending elicitation/create to ACP client",
                session_id=session_id,
                mode=elicitation_mode,
                deferred_handle=event.deferred_handle,
            )
            response = await acp_requests.elicitation_create(
                message=event.message,
                mode=elicitation_mode,
                requested_schema=event.requested_schema or {"type": "object"},
            )
            logger.info(
                "Elicitation: received response from ACP client",
                session_id=session_id,
                action=response.action,
                deferred_handle=event.deferred_handle,
            )
        except asyncio.CancelledError:
            logger.debug("Elicitation background task cancelled", session_id=session_id)
            raise
        except Exception:
            logger.exception(
                "Failed to send elicitation/create to ACP client",
                session_id=session_id,
            )
            return
        else:
            match response.action:
                case "accept":
                    from wolfharness.ui.elicitation import normalize_elicit_content

                    payload = ElicitationResumePayload(
                        deferred_handle=event.deferred_handle,
                        action="accept",
                        content=normalize_elicit_content(response.content),
                    )
                case "decline":
                    payload = ElicitationResumePayload(
                        deferred_handle=event.deferred_handle,
                        action="decline",
                    )
                case "cancel":
                    payload = ElicitationResumePayload(
                        deferred_handle=event.deferred_handle,
                        action="cancel",
                    )
                case _ as unreachable:
                    logger.warning(
                        "Unknown elicitation response action",
                        action=unreachable,
                        session_id=session_id,
                    )
                    return

        try:
            # Restart event consumer before resume — the original consumer
            # stopped after the turn ended (StreamCompleteEvent → consumer
            # loop exit → _after_consumer_loop cleanup). Without this, events
            # from the resumed agent run are published to EventBus but nobody
            # is listening, so the ACP client never receives them.
            logger.info(
                "Elicitation: starting event consumer before resume",
                session_id=session_id,
                deferred_handle=event.deferred_handle,
            )
            await self.start_event_consumer(session_id)
            logger.info(
                "Elicitation: calling resume_session",
                session_id=session_id,
                deferred_handle=event.deferred_handle,
                action=payload.action,
            )
            await session_pool.resume_session(
                session_id,
                deferred_tool_results={},
                elicitation_payloads=[payload],
            )
            logger.info(
                "Elicitation: resume_session completed",
                session_id=session_id,
                deferred_handle=event.deferred_handle,
            )
        except Exception:
            logger.exception(
                "Failed to resume session after elicitation response",
                session_id=session_id,
                deferred_handle=event.deferred_handle,
            )

    async def _after_consumer_loop(self, session_id: str) -> None:
        """Clean up per-session converter and parent-child tracking.

        IMPORTANT: Do NOT cancel elicitation tasks here. Elicitation tasks
        outlive the consumer loop by design — the consumer loop exits when
        the agent turn ends (tool is deferred), but the elicitation task is
        still waiting for the user's response. Cancelling it here would
        prevent ``resume_session`` from ever being called, leaving the
        client with no后续 events after submitting an elicitation response.

        The elicitation task itself calls ``start_event_consumer`` before
        ``resume_session``, so a fresh consumer is started when the session
        resumes. Elicitation tasks are cleaned up in ``close_session`` when
        the session is explicitly torn down.

        Args:
            session_id: The session whose consumer has stopped.
        """
        self._converters.pop(session_id, None)
        self._parent_of.pop(session_id, None)

    async def _event_consumer_loop(self, session_id: str) -> None:
        """Backward-compatible wrapper for mixin's consumer loop.

        Supports direct invocation (e.g., from tests) by lazily subscribing
        when no stream has been set up via ``start_event_consumer()``.

        Args:
            session_id: The session whose events to consume.
        """
        if self._consumer_streams.get(session_id) is None:
            stream = await self.event_bus.subscribe(
                session_id, scope=self._get_subscription_scope()
            )
            self._consumer_streams[session_id] = stream
        await super()._event_consumer_loop(session_id)

    async def _ensure_event_consumer(self, session_id: str) -> None:
        """Subscribe to EventBus once per session and start consumer loop.

        If a consumer task already exists and has not finished, this is a
        no-op.

        Args:
            session_id: The session to ensure a consumer for.
        """
        await self.start_event_consumer(session_id)
        logger.debug("Started event consumer", session_id=session_id)

    async def handle_prompt(  # noqa: PLR0915
        self,
        session_id: str,
        prompt: Sequence[ContentBlock],
        *,
        delivery: str | None = None,
    ) -> PromptResponse:
        """Process a prompt through the SessionPool.

        Ensures the session exists (via ``SessionPool.create_session``) and
        that an event consumer is running before delegating the prompt to
        ``SessionPool.receive_request()``.

        Args:
            session_id: The ACP session identifier.
            prompt: ACP content blocks from the prompt request.
            delivery: Optional delivery hint extracted from ACP ``_meta``.
                ``"steer"`` → inject mid-turn (``asap``), ``"followup"`` →
                queue for next turn (``when_idle``), ``None`` → default
                (``when_idle``).

        Returns:
            A ``PromptResponse`` with the stop reason.
        """
        from wolfharness.lifecycle.types import DeliveryMode
        from wolfharness_server.acp_server.converters import from_acp_content

        session_pool = self._host_context.session_pool
        if session_pool is None:
            logger.error("SessionPool not available", session_id=session_id)
            return self._prompt_response("end_turn")

        # Recover cwd from session_store for clients reconnecting after pool swaps.
        # Also detect checkpointed sessions that need context restoration.
        cwd = "."
        stored_data = None
        if self.session_manager.session_store is not None:
            try:
                stored_data = await self.session_manager.session_store.load_session(session_id)
                if stored_data is not None:
                    if stored_data.cwd:
                        cwd = stored_data.cwd
                    # Resume sessions that exist in storage but are not active in memory.
                    # This handles checkpointed sessions and normal sessions that lost
                    # in-memory state due to server restart/pool swap/TTL expiry.
                    if (
                        self.session_manager.get_session(session_id) is None
                        and self.acp_agent is not None
                    ):
                        logger.info(
                            "Resuming session",
                            session_id=session_id,
                            status=stored_data.status,
                        )
                        await self.session_manager.resume_session(
                            session_id=session_id,
                            client=self.client,
                            acp_agent=self.acp_agent,
                            client_capabilities=self.client_capabilities,
                            client_info=self.acp_agent.client_info,
                            subagent_display_mode=self.acp_agent.subagent_display_mode,
                            raw_input_mode=self.acp_agent.raw_input_mode,
                            connection_id=self.acp_agent._get_connection_id(),
                        )
                        # Re-subscribe EventBus for resumed session
                        await self._ensure_event_consumer(session_id)
            except Exception:
                logger.exception(
                    "Failed to load/resume session from store",
                    session_id=session_id,
                )

        # Ensure the session exists in the SessionPool (pass recovered cwd as metadata).
        # create_session is idempotent — no-op if the session already exists.
        await session_pool.create_session(session_id, cwd=cwd)

        # MCP tools are handled via McpConfigSnapshot → get_capabilities() →
        # MCPToolset, not through agent.tools.providers.
        acp_session = self.session_manager.get_session(session_id)

        # Start event consumer before processing so no events are dropped
        await self._ensure_event_consumer(session_id)

        # Convert ACP content blocks to agent prompts
        contents = [from_acp_content(block, fs=None) for block in prompt]

        # Split slash commands from content and execute local commands.
        # Commands inject expanded prompts into the SessionPool per-session
        # agent's staged_content, which the agent run loop consumes automatically.
        if acp_session is not None:
            from wolfharness_server.acp_server.session import SLASH_PATTERN, split_commands

            commands, non_command_content = split_commands(contents, acp_session.command_store)
            if commands:
                session_agent = await session_pool.sessions.get_or_create_session_agent(session_id)
                for command_text in commands:
                    if match := SLASH_PATTERN.match(command_text.strip()):
                        command_name = match.group(1)
                        args = match.group(2) or ""
                    else:
                        continue
                    # Check NodeCommand support via duck-typing to avoid import
                    cmd = acp_session.command_store.get_command(command_name)
                    if (
                        cmd is not None
                        and callable(supports_node := getattr(cmd, "supports_node", None))
                        and not supports_node(session_agent)
                    ):
                        logger.debug(
                            "Command not available for this node type",
                            command=command_name,
                        )
                        continue
                    # Use per-session agent context so expanded prompts land
                    # in the correct staged_content for the SessionPool turn.
                    agent_context = session_agent.get_context(data=acp_session)
                    cmd_ctx = acp_session.command_store.create_context(
                        data=agent_context,
                        output_writer=lambda msg: logger.debug("Command output", msg=msg),
                    )
                    command_str = f"{command_name} {args}".strip()
                    try:
                        await acp_session.command_store.execute_command(command_str, cmd_ctx)
                    except Exception:
                        logger.exception(
                            "Command execution failed",
                            session_id=session_id,
                            command=command_text,
                        )
                if not non_command_content and len(session_agent.staged_content) == 0:
                    return self._prompt_response("end_turn")
                contents = non_command_content

        # Create ACP input provider for elicitation and tool confirmations
        # through the ACP protocol (not falling back to StdlibInputProvider)
        acp_requests = ACPRequests(client=self.client, session_id=session_id)
        session_proxy = _ACPSessionProxy(
            requests=acp_requests,
            client_capabilities=self.client_capabilities,
            checkpoint_enabled=session_pool.sessions.store is not None,
        )
        input_provider = ACPInputProvider(session=session_proxy)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

        # Emit user_message_chunk notifications for the user's input before
        # processing the prompt. Per the ACP spec, the server should notify
        # the client of the user's message chunks.
        # Generate a message_id for the user message. The EventBus
        # UserMessageInsertedEvent (published by _route_message) is the
        # sole publication point — the ACPEventConverter reconstructs
        # the content blocks from meta.
        message_id = str(uuid.uuid4())

        # Build ACP meta from content blocks for the EventBus event.
        from wolfharness_server.acp_server.event_converter import ACPUserMessageMeta

        content_block_dicts = [block.model_dump() for block in prompt]
        acp_meta = ACPUserMessageMeta(content_blocks=content_block_dicts)

        # Map delivery hint to DeliveryMode for send_message().
        delivery_mode = DeliveryMode.STEER if delivery == "steer" else DeliveryMode.QUEUE

        stop_reason: StopReason = "end_turn"
        try:
            returned_id = await session_pool.send_message(
                session_id,
                contents,
                mode=delivery_mode,
                input_provider=input_provider,
                message_id=message_id,
                meta=acp_meta,
            )
            # Legacy clients (no turn_complete support) block until the run finishes
            # so they don't need session/update turn_complete notifications.
            if returned_id is not None and not (
                self.client_capabilities is not None and self.client_capabilities.turn_complete
            ):
                # Get run handle reference before waiting (cleaned up after completion).
                run_handle = session_pool._get_active_run_handle(session_id)
                try:
                    await session_pool.wait_for_completion(session_id)
                except TimeoutError:
                    # Turn hung — cancel the run to break through __aexit__ hang
                    session_pool.sessions.cancel_run_for_session(session_id)
                    raise
                # Check if run was cancelled after the turn completed.
                # When client sends session/cancel, cancel_session() calls
                # cancel_run_for_session() which sets run_ctx.cancelled.
                # The start() loop then publishes RunFailedEvent, which sets
                # complete_event. We detect the cancelled flag to
                # return stopReason="cancelled".
                if run_handle is not None and run_handle.cancelled:
                    stop_reason = "cancelled"
        except asyncio.CancelledError:
            logger.info("Prompt processing cancelled", session_id=session_id)
            stop_reason = "cancelled"
        except Exception:
            logger.exception("Prompt processing failed", session_id=session_id)
            stop_reason = "end_turn"

        return self._prompt_response(stop_reason)

    async def cancel_session(self, session_id: str) -> None:
        """Cancel the active run for a session and all its subagents.

        Delegates to ``SessionController.cancel_run_for_session()``, which
        cancels the per-session agent's background iteration task — the one
        actually driving the LLM API call.  All child (subagent) sessions
        are recursively cancelled first so that in-flight subagent runs
        stop immediately.

        According to ACP protocol spec, session/cancel is a notification
        (no response expected). The agent must respond to the ORIGINAL
        session/prompt request with stopReason: "cancelled". This is achieved
        without calling ``run_handle.fail()``: the ``start()`` loop detects
        the cancellation, publishes ``RunFailedEvent``, and sets
        ``complete_event`` — which ``handle_prompt()`` is waiting on.
        The ``cancelled`` flag on ``run_ctx`` is set by ``cancel()``, so
        ``handle_prompt()`` can detect it and return the correct stop_reason.

        The event consumer is NOT stopped here to allow the RunFailedEvent
        to be converted and sent as session/update before the turn completes.
        This ensures clients receive proper notification of the cancellation.

        Args:
            session_id: The session to cancel.
        """
        session_pool = self._host_context.session_pool
        if session_pool is None:
            logger.warning("SessionPool not available for cancel", session_id=session_id)
            return

        # Cancel all child (subagent) sessions first, depth-first.
        # Unlike _cancel_subagents() (used by close_session), this does NOT
        # pop _parent_of entries or stop event consumers — it only cancels
        # the RunHandle for each child so the session stays alive for
        # follow-up interaction.
        await self._cancel_subagent_runs(session_id)

        # Cancel the parent session's run last.
        session_pool.sessions.cancel_run_for_session(session_id)

        # The start() loop detects the cancelled flag, publishes
        # RunFailedEvent (which sets complete_event), and the
        # event consumer converts it to session/update with
        # stop_reason="cancelled". handle_prompt() unblocks on
        # complete_event and returns the cancelled stop_reason.
        # No explicit fail() call is needed here.

        # Note: Event consumer is NOT stopped here. It will continue running
        # until the RunFailedEvent is processed, which emits the appropriate
        # session/update (turn_complete with stop_reason="cancelled").

    async def _cancel_subagent_runs(self, parent_sid: str) -> None:
        """Recursively cancel runs for all child sessions of parent_sid.

        Walks the ``_parent_of`` tree depth-first and calls
        ``cancel_run_for_session()`` for each child.  Unlike
        :meth:`_cancel_subagents`, this does NOT modify ``_parent_of``
        or stop event consumers — it only cancels the active
        :class:`RunHandle` for each child session.

        Args:
            parent_sid: The session whose child runs should be cancelled.
        """
        session_pool = self._host_context.session_pool
        if session_pool is None:
            return
        children = [child for child, parent in self._parent_of.items() if parent == parent_sid]
        for child_sid in children:
            await self._cancel_subagent_runs(child_sid)
            session_pool.sessions.cancel_run_for_session(child_sid)

    async def _cancel_subagents(self, parent_sid: str) -> None:
        """Recursively cancel all child sessions of parent_sid.

        Walks the ``_parent_of`` tree depth-first, popping each child
        before recursing into its own children to prevent infinite loops
        on circular entries.  After the subtree is drained, each child's
        event consumer is stopped via ``stop_event_consumer()``, which
        cascades cancellation through the mixin's CancelScope.

        Args:
            parent_sid: The session whose child sessions should be cancelled.
        """
        children = [child for child, parent in self._parent_of.items() if parent == parent_sid]
        for child_sid in children:
            self._parent_of.pop(child_sid, None)
            await self._cancel_subagents(child_sid)
            await self.stop_event_consumer(child_sid)

    async def close_session(self, session_id: str) -> None:
        """Close a session and tear down its event consumer.

        Recursively cancels all child (subagent) sessions before stopping
        the parent's own consumer.  Then sends the EventBus sentinel to
        gracefully stop the consumer loop, waits for it to finish, and
        delegates to ``SessionPool.close_session()``.

        Args:
            session_id: The session to close.
        """
        session_pool = self._host_context.session_pool

        # Cancel all child sessions first (depth-first, pop-before-recurse)
        await self._cancel_subagents(session_id)

        # Cancel pending elicitation tasks for this session only.
        session_tasks = self._elicitation_tasks.pop(session_id, None)
        if session_tasks is not None:
            for task in session_tasks:
                if not task.done():
                    task.cancel()

        # Stop the event consumer (mixin's stop handles cancellation + unsubscribe)
        await self.stop_event_consumer(session_id)

        # Signal EventBus to close session
        if session_pool is not None:
            await session_pool.event_bus.close_session(session_id)

        # Delegate to SessionPool for final cleanup
        if session_pool is not None:
            try:
                await session_pool.close_session(session_id)
            except Exception:
                logger.exception("SessionPool close_session failed", session_id=session_id)

    def _prompt_response(self, stop_reason: StopReason) -> PromptResponse:
        """Build a minimal PromptResponse.

        Args:
            stop_reason: The ACP stop reason.

        Returns:
            A ``PromptResponse`` with the given stop reason.
        """
        from acp.schema import PromptResponse

        return PromptResponse(stop_reason=stop_reason)

    async def _emit_user_message_chunks(
        self,
        session_id: str,
        prompt: Sequence[ContentBlock],
        *,
        message_id: str | None = None,
    ) -> None:
        """Emit ``user_message_chunk`` SessionUpdate notifications for user input.

        Per the ACP spec, the server should emit ``user_message_chunk``
        notifications when a user message is processed. This method
        converts the user's content blocks to ``UserMessageChunk``
        session updates and sends them to the client before the agent
        begins processing.

        Args:
            session_id: The ACP session identifier.
            prompt: ACP content blocks from the prompt request.
            message_id: Optional message ID for correlating chunks. If
                provided, all chunks share this ID. If ``None``, each
                call to ``build_user_message_chunks`` generates its own.
        """
        from acp.schema import SessionNotification, TextContentBlock

        for block in prompt:
            if not isinstance(block, TextContentBlock):
                continue
            text = block.text
            if not text:
                continue
            chunks = ACPEventConverter.build_user_message_chunks(text, message_id=message_id)
            for chunk in chunks:
                notification = SessionNotification(
                    session_id=session_id,
                    update=chunk,
                )
                try:
                    await self.client.session_update(notification)
                except (ConnectionResetError, BrokenPipeError, anyio.ClosedResourceError):
                    logger.debug(
                        "Failed to send user_message_chunk",
                        session_id=session_id,
                    )


class _ACPSessionProxy:
    """Lightweight proxy providing the subset of ACPSession that ACPInputProvider needs.

    ACPProtocolHandler does not have a full ACPSession instance, but
    ACPInputProvider only needs ``requests`` and ``client_capabilities``.
    This proxy bridges the gap so elicitation/tool-confirmation flows
    through the ACP protocol instead of falling back to StdlibInputProvider.
    """

    def __init__(
        self,
        requests: ACPRequests,
        client_capabilities: ClientCapabilities | None = None,
        checkpoint_enabled: bool = False,
    ) -> None:
        self.requests = requests
        self.client_capabilities = client_capabilities or ClientCapabilities()
        self.checkpoint_enabled = checkpoint_enabled
