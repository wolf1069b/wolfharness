"""Messaging mixin for SessionPool.

Extracted from session_pool.py as part of the session-debt-cleanup file split.
Contains message routing, prompt processing, and message history methods.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import logfire

from wolfharness.agents.events import StreamCompleteEvent
from wolfharness.lifecycle.types import DeliveryMode, Feedback
from wolfharness.log import get_logger
from wolfharness.utils.pydantic_ai_helpers import flatten_prompts


if TYPE_CHECKING:
    from collections import OrderedDict

    from wolfharness.agents.base_agent import BaseAgent
    from wolfharness.delegation import AgentPool
    from wolfharness.messaging import ChatMessage
    from wolfharness.orchestrator.event_bus import EventBus
    from wolfharness.orchestrator.run import RunHandle
    from wolfharness.orchestrator.session_controller import SessionController, SessionState


logger = get_logger(__name__)


class SessionPoolMessagingMixin:
    """Mixin providing message routing and history methods for SessionPool.

    Attributes:
        sessions: SessionController instance (provided by SessionPool).
        pool: AgentPool instance (provided by SessionPool).
        event_bus: EventBus instance (provided by SessionPool).
        _message_cache: Per-session message cache (provided by SessionPool).
    """

    sessions: SessionController
    pool: AgentPool[Any]

    @property
    def event_bus(self) -> EventBus: ...  # type: ignore[empty-body]  # ty: ignore[empty-body]

    _message_cache: OrderedDict[str, list[ChatMessage[Any]]]
    _message_cache_maxsize: int

    if TYPE_CHECKING:

        def _create_run_handle(
            self,
            session: SessionState,
            agent: BaseAgent[Any, Any],
            session_id: str,
            *,
            deps: Any = None,
            cached_elicitation_responses: dict[str, Any] | None = None,
            deferred_tool_results: Any = None,
            message_history: list[Any] | None = None,
        ) -> RunHandle: ...

        async def create_session(
            self,
            session_id: str,
            agent_name: str | None = None,
            parent_session_id: str | None = None,
            lifecycle_policy: str | None = None,
            **metadata: Any,
        ) -> SessionState: ...

        async def close_session(self, session_id: str) -> None: ...

        def _get_active_run_handle(self, session_id: str) -> RunHandle | None: ...

        async def wait_for_completion(
            self,
            session_id: str,
            timeout: float | None = 300,
        ) -> str: ...

    def _evict_message_cache(self) -> None:
        """Evict LRU entries from _message_cache (provided by SessionPool)."""

    async def process_prompt(
        self,
        session_id: str,
        *prompts: Any,
        **kwargs: Any,
    ) -> None:
        """Process a prompt through the RunHandle lifecycle.

        Main entry point for protocol handlers.
        Events are delivered exclusively via EventBus.

        Args:
            session_id: The session to process the prompt for.
            *prompts: Prompts to process.
            **kwargs: Additional arguments passed to the agent.
        """
        await self._process_prompt_run_turn(session_id, *prompts, **kwargs)

    async def _process_prompt_run_turn(
        self,
        session_id: str,
        *prompts: Any,
        **kwargs: Any,
    ) -> None:
        """Handle process_prompt via the RunHandle path.

        If no active run exists, creates a RunHandle and drains
        ``start()`` to completion. If a run is active, steers the
        message into it.
        """
        from wolfharness.agents.events import RunErrorEvent, StreamCompleteEvent

        session, _ = await self.sessions.get_or_create_session(session_id)
        if session.is_closing:
            return
        # Extract input_provider from kwargs and set on session BEFORE
        # get_or_create_session_agent() so the agent is created with the
        # correct input_provider and the session state is consistent.
        input_provider = kwargs.pop("input_provider", None)
        if input_provider is not None:
            session.input_provider = input_provider
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
            run_handle = self.sessions._runs.get(run_id)
            if run_handle is not None:
                run_handle.steer(content)
            return

        run_handle = self._create_run_handle(session, agent, session_id)
        gen = run_handle.start(content)
        try:
            async for _event in gen:
                if isinstance(_event, StreamCompleteEvent | RunErrorEvent):
                    break
        finally:
            # gen.aclose() may raise CancelledError (a BaseException,
            # not caught by ``except Exception``). Use save-and-re-raise
            # so cleanup steps always run and CancelledError is re-raised.
            _cancelled: asyncio.CancelledError | None = None
            try:
                await gen.aclose()
            except asyncio.CancelledError as e:
                _cancelled = e
            except Exception:
                logger.exception("Failed to close run generator")
            session.current_run_id = None
            self.sessions._runs.pop(run_handle.run_id, None)
            if _cancelled is not None:
                raise _cancelled

    @logfire.instrument("session.send_message")
    async def send_message(
        self,
        session_id: str,
        content: str | list[Any],
        *,
        mode: DeliveryMode = DeliveryMode.QUEUE,
        message_id: str | None = None,
        deps: Any = None,
        input_provider: Any = None,
        meta: Any = None,
        source: str = "accepted",
    ) -> str | None:
        """Send a message to a session using the typed ``DeliveryMode`` enum.

        Maps ``DeliveryMode.STEER`` → ``priority="asap"`` (inject into
        active turn) and ``DeliveryMode.QUEUE`` → ``priority="when_idle"``
        (queue for next turn), then delegates to
        :meth:`SessionController._route_message`.

        Args:
            session_id: Target session.
            content: Message / prompt content (text or structured content
                blocks). List content is stored as ``Feedback.content_blocks``
                without stringification.
            mode: Delivery mode — ``STEER`` for mid-turn injection,
                ``QUEUE`` for next-turn queue (default).
            message_id: Optional message ID. Auto-generated as UUID4 if
                not provided. Returned on success.
            deps: Optional dependencies to pass to the agent run context
                (e.g. delegation_depth from BackgroundTaskCapability).
            input_provider: Optional input provider to set on the session
                before agent resolution. When provided, the session's
                ``input_provider`` is updated so the agent is created
                with the correct provider.
            meta: Protocol-specific metadata to carry through to
                ``UserMessageInsertedEvent``. When set, the event consumer
                uses it to reconstruct the full user message (e.g. OpenCode
                parts, ACP content blocks) instead of falling back to
                text-only content.
            source: Originator of the message — ``"accepted"`` (default)
                for protocol handler requests, ``"team"`` for team-mode
                coordination messages. Passed through to
                ``UserMessageInsertedEvent.source`` so protocol frontends
                can render team messages with a distinct visual style.

        Returns:
            The ``message_id`` string on success (both new runs and
            steer/followup), ``None`` on failure.

        Note:
            For ``DeliveryMode.QUEUE`` when the session is busy, the
            return value is ``None`` even though the message was
            successfully queued. Callers that need to distinguish
            ``None``-as-queued from ``None``-as-failure should verify
            session existence before calling.
        """
        priority = "asap" if mode is DeliveryMode.STEER else "when_idle"
        # Delegate directly to _route_message() to avoid the deprecated
        # receive_request() path. This requires session+agent resolution,
        # which mirrors what SessionController.receive_request() does.
        session = self.sessions.get_session(session_id)
        if session is None:
            return None
        session.last_active_at = time.monotonic()
        session.last_active_at_ns = time.time_ns()
        if input_provider is not None:
            session.input_provider = input_provider
        agent = await self.sessions.get_or_create_session_agent(
            session_id, input_provider=input_provider
        )
        if agent is None:
            return None
        return await self.sessions._route_message(
            session,
            agent,
            session_id,
            content,
            priority=priority,
            deps=deps,
            message_id=message_id,
            meta=meta,
            source=source,
        )

    async def run_agent(
        self,
        agent: str,
        prompt: str,
        parent_session_id: str | None = None,
        **metadata: Any,
    ) -> str:
        """Run an agent to completion and return the result text.

        Creates a temporary session, sends the prompt via
        :meth:`send_message` with ``DeliveryMode.QUEUE``, subscribes to
        the EventBus to capture the ``StreamCompleteEvent``, waits for
        run completion, then closes the session. The session is always
        cleaned up via ``try/finally`` even on error.

        Args:
            agent: Name of the agent to run.
            prompt: Input prompt for the agent.
            parent_session_id: Optional parent session for hierarchical
                sessions (e.g. subagent delegation).
            **metadata: Arbitrary metadata attached to the session.

        Returns:
            The final assistant response text.

        Raises:
            SessionNotFoundError: If the session cannot be created.
            asyncio.TimeoutError: If the run does not complete within
                the default timeout.
            Exception: Any error from the agent run is re-raised after
                session cleanup.
        """
        from wolfharness.agents.events import RunErrorEvent
        from wolfharness.utils.identifiers import generate_session_id

        session_id = generate_session_id()
        await self.create_session(
            session_id,
            agent_name=agent,
            parent_session_id=parent_session_id,
            **metadata,
        )

        # Subscribe to EventBus BEFORE sending the message to avoid
        # missing the StreamCompleteEvent in a race.
        bus_queue = await self.event_bus.subscribe(session_id, scope="session")
        result_text: str = ""
        try:
            msg_id = await self.send_message(
                session_id,
                prompt,
                mode=DeliveryMode.QUEUE,
            )
            if msg_id is None:
                msg = f"Failed to send message to session {session_id}"
                raise RuntimeError(msg)

            # Drain events from the EventBus until we capture the
            # StreamCompleteEvent or RunErrorEvent.
            while True:
                try:
                    envelope = await asyncio.wait_for(bus_queue.get(), timeout=120.0)
                except TimeoutError:
                    logger.warning(
                        "Agent execution timed out after 120 seconds in run_agent",
                        session_id=session_id,
                    )
                    msg = f"Agent execution timed out after 120 seconds for session {session_id}"
                    raise TimeoutError(msg) from None
                event = envelope.event
                if isinstance(event, StreamCompleteEvent):
                    content = event.message.content
                    result_text = content if isinstance(content, str) else str(content)
                    break
                if isinstance(event, RunErrorEvent):
                    raise RuntimeError(event.message)  # noqa: TRY004

            # Ensure the run has fully completed before closing.
            try:
                await self.wait_for_completion(session_id, timeout=10.0)
            except TimeoutError:
                # Turn hung — cancel the run to break through __aexit__ hang
                self.sessions.cancel_run_for_session(session_id)
        finally:
            try:
                await self.event_bus.unsubscribe(session_id, bus_queue)
            except Exception:
                logger.exception("Failed to unsubscribe from EventBus", session_id=session_id)
            try:
                await self.close_session(session_id)
            except Exception:
                logger.exception("Failed to close session", session_id=session_id)

        return result_text

    async def inject_prompt(self, session_id: str, message: str, **kwargs: Any) -> str | None:
        """Inject a message into a session.

        If the session has an active run, injects immediately via
        ``RunHandle.steer()``. Otherwise, returns None.

        Does NOT acquire session.turn_lock.

        Args:
            session_id: The session to inject into.
            message: The message to inject.
            **kwargs: Additional arguments passed to the agent run.

        Returns:
            message_id if injected into active turn, None otherwise.
        """
        run_handle = self._get_active_run_handle(session_id)
        if run_handle is not None:
            return run_handle.steer(message)
        return None

    async def queue_prompt(self, session_id: str, *prompts: Any, **kwargs: Any) -> str | None:
        """Queue prompts for a session.

        Similar to inject_prompt but for full prompts.
        Does NOT acquire session.turn_lock.

        Args:
            session_id: The session to queue prompts for.
            *prompts: Prompts to queue.
            **kwargs: Additional arguments passed to the agent run.

        Returns:
            message_id if queued into active turn, None otherwise.
        """
        run_handle = self._get_active_run_handle(session_id)
        if run_handle is not None:
            message = prompts[0] if prompts else ""
            return run_handle.followup(str(message))
        return None

    async def steer(
        self,
        session_id: str,
        message: str,
        *,
        emit_user_message: bool = True,
        **kwargs: Any,
    ) -> str | None:
        """Inject a steer message with agent-type-aware routing.

        Delegates to ``RunHandle.steer()`` when an active run exists.

        Args:
            session_id: Target session.
            message: The steer message to deliver.
            emit_user_message: When ``True`` (default), publish a
                ``UserMessageInsertedEvent`` so protocol frontends display
                the injected message.  Set to ``False`` to suppress.
            **kwargs: Additional arguments (ignored).

        Returns:
            message_id if delivered into active turn, None otherwise.
        """
        run_handle = self._get_active_run_handle(session_id)
        if run_handle is not None:
            return run_handle.steer(message, emit_user_message=emit_user_message)
        return None

    async def steer_from_background_task(self, session_id: str, message: str) -> str | None:
        """Inject a steer message from a background task completion.

        This is the preferred entry point for background task capabilities
        (e.g. ``SubagentCapability``, ``BackgroundTaskCapability``) because:
        - It emits ``UserMessageInsertedEvent(source="accepted")``
          for TUI display
        - It injects into the active RunHandle when one exists
        - It falls back to ``feedback_queue`` when no run is active
        - It survives across RunHandle boundaries (uses SessionState)

        Args:
            session_id: Target session.
            message: The steer message to deliver.

        Returns:
            message_id if delivered, None otherwise.
        """
        from wolfharness.agents.events.events import UserMessageInsertedEvent
        from wolfharness.utils.identifiers import ascending

        session = self.sessions._sessions.get(session_id)
        if session is None:
            return None
        # Publish UserMessageInsertedEvent FIRST (await, not fire-and-forget).
        # This ensures the TUI creates the UserMessage with a timestamp-based
        # message_id BEFORE the agent processes the steer and produces output.
        # Use self.event_bus (SessionPoolMessagingMixin property, always set by
        # SessionPool.__init__) instead of session._event_bus (SessionState's
        # field, only set by _initialize_lifecycle_and_recovery).
        #
        # Generate message_id ONCE and share with steer() so that
        # EnqueuedMessagesEvent-derived event uses the same ID for dedup.
        message_id = ascending("message")
        event_bus = self.event_bus
        if event_bus is not None:
            with logfire.span(
                "event.user_message_inserted.emit",
                session_id=session_id,
                delivery="steer",
                source="accepted",
            ):
                try:
                    event: UserMessageInsertedEvent[Any] = UserMessageInsertedEvent(
                        session_id=session_id,
                        message_id=message_id,
                        content=message,
                        delivery="steer",
                        source="accepted",
                    )
                    await event_bus.publish(session_id, event)
                except Exception:
                    logger.warning(
                        "Failed to emit UserMessageInsertedEvent from background task",
                        exc_info=True,
                    )
        # Try injecting into the active RunHandle.
        run_handle = self._get_active_run_handle(session_id)
        if run_handle is not None:
            return run_handle.steer(message, message_id=message_id, emit_user_message=False)
        # No active run — enqueue for next RunHandle via feedback_queue.
        fb = Feedback(content=message, is_steer=True)
        session.feedback_queue.put_nowait(fb)
        return fb.message_id

    async def followup(self, session_id: str, message: str, **kwargs: Any) -> str | None:
        """Queue a follow-up message with agent-type-aware routing.

        Delegates to ``RunHandle.followup()`` when an active run exists.

        Args:
            session_id: Target session.
            message: The follow-up message to deliver.
            **kwargs: Additional arguments (ignored).

        Returns:
            message_id if delivered into active turn, None otherwise.
        """
        run_handle = self._get_active_run_handle(session_id)
        if run_handle is not None:
            return run_handle.followup(message)
        return None

    async def get_messages(
        self,
        session_id: str,
    ) -> list[ChatMessage[Any]]:
        """Get message history for a session.

        Results are cached per session_id (full message list) to avoid
        repeated storage queries. Cache is invalidated by append_message,
        truncate_messages, and copy_messages.

        Args:
            session_id: The session to retrieve messages for.

        Returns:
            List of messages ordered by timestamp (oldest first).

        Raises:
            KeyError: If the session does not exist.
        """
        session = self.sessions.get_session(session_id)
        if session is None:
            raise KeyError(session_id)

        if session_id in self._message_cache:
            # Move to end (most recently used) for LRU ordering.
            self._message_cache.move_to_end(session_id)
            return list(self._message_cache[session_id])

        storage = self.pool.storage
        if storage is not None:
            messages = await storage.get_session_messages(session_id)
            self._message_cache[session_id] = list(messages)
            self._evict_message_cache()
            return messages

        return []

    async def append_message(
        self,
        session_id: str,
        message: ChatMessage[Any],
    ) -> str:
        """Append a message to a session's history.

        Args:
            session_id: The session to append to.
            message: The message to append.

        Returns:
            The ID of the appended message.

        Raises:
            KeyError: If the session does not exist.
        """
        session = self.sessions.get_session(session_id)
        if session is None:
            raise KeyError(session_id)

        storage = self.pool.storage
        if storage is not None:
            await storage.log_message(message=message)

        self._message_cache.pop(session_id, None)
        return message.message_id

    async def copy_messages(
        self,
        source_session_id: str,
        target_session_id: str,
        *,
        up_to_message_id: str | None = None,
    ) -> str | None:
        """Copy messages from one session to another.

        Used by share_session (copy all) and revert_session (copy up to
        a specific message).

        Args:
            source_session_id: Session to copy from.
            target_session_id: Session to copy to.
            up_to_message_id: If set, only copy messages up to and
                including this message ID. If None, copy all messages.

        Returns:
            The ID of the fork point message (last copied message),
            or None if no messages were copied.

        Raises:
            KeyError: If either session does not exist.
        """
        if self.sessions.get_session(source_session_id) is None:
            raise KeyError(source_session_id)
        if self.sessions.get_session(target_session_id) is None:
            raise KeyError(target_session_id)

        storage = self.pool.storage
        if storage is not None:
            result = await storage.fork_conversation(
                source_session_id=source_session_id,
                new_session_id=target_session_id,
                fork_from_message_id=up_to_message_id,
            )
            self._message_cache.pop(target_session_id, None)
            return result

        return None

    async def truncate_messages(
        self,
        session_id: str,
        up_to_message_id: str,
    ) -> int:
        """Truncate messages after a specific message ID.

        Used by revert_session to remove messages after the revert point.

        Args:
            session_id: The session to truncate.
            up_to_message_id: Keep messages up to and including this ID,
                remove everything after.

        Returns:
            Number of messages removed.

        Raises:
            KeyError: If the session does not exist.
        """
        session = self.sessions.get_session(session_id)
        if session is None:
            raise KeyError(session_id)

        storage = self.pool.storage
        if storage is not None:
            removed = await storage.truncate_messages(session_id, up_to_message_id)
            self._message_cache.pop(session_id, None)
            return removed

        return 0
