"""Session route handlers and session lifecycle routing mixin.

Extracted from session_pool_integration.py as part of the session-debt-cleanup
file split. Contains session creation, persistence, status management, and
the session routes mixin that provides session lifecycle methods for the
OpenCodeSessionPoolIntegration class.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from wolfharness.log import get_logger
from wolfharness.utils.identifiers import extract_timestamp_ms
from wolfharness.utils.time_utils import now_ms
from wolfharness_server.opencode_server.models import (
    SessionCreatedEvent,
    SessionErrorEvent,
    SessionStatus,
    SessionStatusEvent,
    TimeCreatedUpdated,
)
from wolfharness_server.opencode_server.opencode_message_bridge import (
    _reconstruct_tool_parts_from_checkpoint,
)


if TYPE_CHECKING:
    from wolfharness.orchestrator.core import SessionPool
    from wolfharness.sessions.models import SessionData
    from wolfharness_server.opencode_server.models.session import Session
    from wolfharness_server.opencode_server.state import ServerState


logger = get_logger(__name__)


async def set_session_status(
    state: ServerState,
    session_id: str,
    status: SessionStatus,
) -> None:
    """Set the status of a session.

    Broadcasts ``SessionStatusEvent`` via ``ServerState`` directly.
    Falls back to the in-memory session_status dict for legacy code paths.

    Args:
        state: The OpenCode server state.
        session_id: The session to update.
        status: The new session status.
    """
    await state.broadcast_event(SessionStatusEvent.create(session_id, status))


async def get_session_status(
    state: ServerState,
    session_id: str,
) -> SessionStatus | None:
    """Get the current status of a session.

    Delegates to OpenCodeSessionPoolIntegration when the feature flag is
    enabled, otherwise falls back to the ServerState in-memory dictionary.

    Args:
        state: The OpenCode server state.
        session_id: The session to look up.

    Returns:
        The session status, or None if not found and the fallback is used.
    """
    integration: OpenCodeSessionRoutesMixin | None = getattr(
        state, "session_pool_integration", None
    )
    if integration is not None:
        return await integration.get_session_status(session_id)

    return getattr(state, "session_status", {}).get(session_id)


async def ensure_session(
    state: ServerState,
    session_id: str,
    parent_id: str | None = None,
    title: str | None = None,
) -> Session:
    """Ensure a session exists with the given ID.

    Resolution order (store-first, non-overwriting):

    1. **In-memory hit** — if the session already exists in
       ``state.sessions``, return it immediately (broadcasts
       ``session.updated`` so the TUI can upsert).

    2. **Store hit** — if the session is absent from memory but present
       in the session store, convert the stored ``SessionData`` to a UI
       ``Session``, register all in-memory runtime state (messages,
       status, input-provider), mark idle, and broadcast
       ``session.created`` + ``session.updated``.  **Does NOT** call
       ``store.save()`` because the data is already persisted.

    3. **Store miss** — fall back to creating a brand-new session and
       persisting it (original behaviour).

    Concurrent calls for the same ``session_id`` are serialized by a
    per-session lock so that only one in-memory ``Session`` object is
    created.

    Args:
        state: The OpenCode server state.
        session_id: Unique identifier for the session
        parent_id: Optional parent session ID for fork relationships
        title: Optional session title. Defaults to "New Session" when None.

    Returns:
        The Session object (existing or newly created)
    """
    from wolfharness_server.opencode_server.converters import session_data_to_opencode
    from wolfharness_server.opencode_server.models import SessionUpdatedEvent

    # --- Fast path: already in memory -----------------------------------
    if session_id in state.sessions:
        session = state.sessions[session_id]
        await state.broadcast_event(SessionUpdatedEvent.create(session))
        return session

    # --- Serialise concurrent callers for the same session_id -----------
    if session_id not in state.session_locks:
        state.session_locks[session_id] = asyncio.Lock()
    try:
        async with state.session_locks[session_id]:
            if session_id in state.sessions:
                session = state.sessions[session_id]
                await state.broadcast_event(SessionUpdatedEvent.create(session))
                return session

            # --- Store-first path ------------------------------------------
            session_data = None
            if (
                state.pool.session_pool is not None
                and state.pool.session_pool.sessions.store is not None
            ):
                session_data = await state.pool.session_pool.sessions.store.load_session(session_id)
            if session_data is None:
                session_data = await state.pool.storage.load_session(session_id)

            if session_data is not None:
                session = session_data_to_opencode(session_data)

                state.sessions[session_id] = session
                state.ensure_runtime_session_state(session_id)
                state.ensure_input_provider(session_id)
                await state.mark_session_idle(session_id)

                # --- Checkpoint restoration (Task 27) ---------------------------
                if session_data.status == "checkpointed":
                    await _restore_checkpoint_state(state, session_data, session_id)
                    logger.info(
                        "ensure_session: restored checkpointed session",
                        session_id=session_id,
                        pending_call_count=len(session_data.pending_deferred_calls),
                    )

                # Sync input_provider to SessionPool's SessionState for all sessions
                input_provider = state.ensure_input_provider(session_id)
                if state.pool.session_pool is not None:
                    sp_session = state.pool.session_pool.sessions.get_session(session_id)
                    if sp_session is not None:
                        sp_session.input_provider = input_provider

                from wolfharness_server.opencode_server.models import (
                    SessionCreatedEvent,
                )

                await state.broadcast_event(SessionCreatedEvent.create(session))
                await state.broadcast_event(SessionUpdatedEvent.create(session))
                logger.info(
                    "ensure_session: loaded from store",
                    session_id=session_id,
                    parent_id=session_data.parent_id,
                )
                return session

            # --- Store-miss fallback: create new session -------------------
            return await _create_and_persist_session(state, session_id, parent_id, title=title)
    finally:
        state.session_locks.pop(session_id, None)


async def _create_and_persist_session(
    state: ServerState,
    session_id: str,
    parent_id: str | None,
    title: str | None = None,
) -> Session:
    """Create a brand-new session and persist it (store-miss fallback).

    Args:
        state: The OpenCode server state.
        session_id: Unique identifier for the session.
        parent_id: Optional parent session ID.
        title: Optional session title. Defaults to "New Session" when None.

    Returns:
        The newly created and persisted ``Session``.
    """
    from wolfharness_server.opencode_server.converters import opencode_to_session_data
    from wolfharness_server.opencode_server.models import (
        Session,
        SessionCreatedEvent,
        SessionUpdatedEvent,
    )
    from wolfharness_storage.opencode_provider import helpers

    now = extract_timestamp_ms(session_id) or now_ms()
    if parent_id is not None:
        parent_session = state.sessions.get(parent_id)
        if parent_session:
            project_id = parent_session.project_id
            directory = parent_session.directory
        else:
            project_id = helpers.compute_project_id(state.working_dir)
            directory = state.working_dir
    else:
        project_id = helpers.compute_project_id(state.working_dir)
        directory = state.working_dir
    session = Session(
        id=session_id,
        project_id=project_id,
        directory=directory,
        title=title or "New Session",
        version="1",
        time=TimeCreatedUpdated(created=now, updated=now),
        parent_id=parent_id,
    )

    id_ = state.pool.manifest.config_file_path
    session_data = opencode_to_session_data(session, agent_name=state.agent.name, pool_id=id_)

    # Delegate session persistence to SessionPool.create_session() to avoid
    # dual-write. SessionPool internally calls SessionController.get_or_create_session()
    # which persists via the store. When SessionPool or its store is unavailable,
    # fall back to direct storage save (tests only).
    try:
        sp = state.pool.session_pool
        if sp is not None and sp.sessions.store is not None:
            await sp.create_session(
                session_id,
                agent_name=state.agent.name,
                parent_session_id=parent_id,
                project_id=project_id,
                cwd=directory,
            )
        else:
            await state.pool.storage.save_session(session_data)
    except Exception:
        logger.warning(
            "Failed to persist session to storage, degrading to in-memory",
            session_id=session_id,
            exc_info=True,
        )

    state.sessions[session_id] = session
    state.ensure_runtime_session_state(session_id)
    await state.mark_session_idle(session_id)

    # Sync input_provider to SessionPool's SessionState for all sessions
    input_provider = state.ensure_input_provider(session_id)
    if state.pool.session_pool is not None:
        sp_session = state.pool.session_pool.sessions.get_session(session_id)
        if sp_session is not None:
            sp_session.input_provider = input_provider

    await state.broadcast_event(SessionCreatedEvent.create(session))
    await state.broadcast_event(SessionUpdatedEvent.create(session))
    logger.info(
        "ensure_session: created new session",
        session_id=session_id,
        parent_id=parent_id,
    )

    return session


async def _restore_checkpoint_state(
    state: ServerState,
    session_data: SessionData,
    session_id: str,
) -> None:
    """Restore opencode runtime state from a checkpointed session.

    Reconstructs running ToolParts for pending deferred calls and
    restores the parent/child spawn graph topology.

    Args:
        state: The OpenCode server state.
        session_data: The persisted SessionData with checkpoint metadata.
        session_id: The session being restored.
    """
    _reconstruct_tool_parts_from_checkpoint(state, session_id, session_data.pending_deferred_calls)
    _restore_spawn_topology_from_checkpoint(state, session_data, session_id)


def _restore_spawn_topology_from_checkpoint(
    state: ServerState,
    session_data: SessionData,
    session_id: str,
) -> None:
    """Restore parent/child spawn graph from checkpoint metadata.

    Reads ``spawn_children`` from the session's metadata and stores it
    on ``state.checkpoint_spawn_graph`` so that
    :class:`OpenCodeSessionPoolIntegration` can reconstruct
    ``_children_of``, ``_child_to_parent``, and ``_child_spawns`` maps
    when the consumer starts.

    Args:
        state: The OpenCode server state.
        session_data: The persisted SessionData with checkpoint metadata.
        session_id: The parent session being restored.
    """
    spawn_children: list[str] = session_data.metadata.get("spawn_children", [])
    if not hasattr(state, "checkpoint_spawn_graph"):
        state.checkpoint_spawn_graph = {}  # type: ignore[attr-defined]  # ty: ignore[invalid-assignment]
    state.checkpoint_spawn_graph[session_id] = list(spawn_children)  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    logger.debug(
        "Restored spawn topology from checkpoint",
        session_id=session_id,
        child_count=len(spawn_children),
    )


class OpenCodeSessionRoutesMixin:
    """Mixin providing session lifecycle and routing methods.

    Provides session creation, forking, closing, message routing, status
    management, and shutdown methods for the OpenCodeSessionPoolIntegration.

    Attributes:
        session_pool: The SessionPool instance (provided by main class).
        server_state: The OpenCode server state (provided by main class).
        _pending_message_ids: Pending canonical message IDs (provided by main class).
        _pending_message_metadata: Pending model metadata from REST handlers
            (provided by main class).
    """

    session_pool: SessionPool
    server_state: ServerState
    _pending_message_ids: dict[str, str]
    _pending_message_metadata: dict[str, dict[str, str | None]]
    _session_groups: dict[str, Any]
    _resume_contexts: dict[str, dict[str, Any]]

    if TYPE_CHECKING:

        async def _start_event_consumer(self, session_id: str) -> None: ...
        async def _stop_event_consumer(self, session_id: str) -> None: ...
        async def stop_event_consumer(self, session_id: str) -> None: ...

    async def create_session(
        self,
        session_id: str,
        agent_name: str | None = None,
        **metadata: Any,
    ) -> Any:
        """Create a session via SessionPool and start its status bridge.

        Uses get_or_create_session so the call is idempotent: bridge and
        consumer are only started when the session is actually new.

        Args:
            session_id: Unique identifier for the session.
            agent_name: Name of the agent to associate with the session.
            **metadata: Arbitrary metadata to attach to the session.

        Returns:
            The session state from the SessionPool.
        """
        from wolfharness_server.opencode_server.opencode_message_bridge import (
            _session_state_to_opencode,
        )

        state, was_created = await self.session_pool.sessions.get_or_create_session(
            session_id, agent_name, **metadata
        )
        if was_created:
            await self._start_event_consumer(session_id)

            # Broadcast session.created event so OpenCode clients can upsert
            session = _session_state_to_opencode(state)
            await self.server_state.broadcast_event(SessionCreatedEvent.create(session))

        return state

    async def fork_session(
        self,
        parent_session_id: str,
        new_session_id: str,
        agent_name: str | None = None,
    ) -> Any:
        """Fork a session, creating a child with a parent reference.

        Uses get_or_create_session so the call is idempotent: bridge is
        only started when the session is actually new.

        Args:
            parent_session_id: The parent session ID.
            new_session_id: The new child session ID.
            agent_name: Name of the agent for the child session.

        Returns:
            The child session state.
        """
        parent_state = self.session_pool.sessions.get_session(parent_session_id)
        metadata: dict[str, Any] = {}
        if parent_state is not None:
            # get_or_create_session may nest kwargs under a "metadata" key;
            # unwrap one level so the child inherits the actual metadata dict.
            raw = parent_state.metadata
            metadata = dict(raw.get("metadata", raw))
        state, was_created = await self.session_pool.sessions.get_or_create_session(
            new_session_id,
            agent_name=agent_name,
            parent_session_id=parent_session_id,
            **metadata,
        )
        if was_created:
            pass  # Forked session inherits parent's event consumer
        return state

    async def close_session(self, session_id: str) -> None:
        """Close a session and clean up its resources.

        Stops the session-scoped event consumer and status bridge,
        then delegates to SessionPool.close_session().

        Args:
            session_id: The session to close.
        """
        await self._stop_event_consumer(session_id)
        await self.session_pool.close_session(session_id)

    async def route_message(
        self,
        session_id: str,
        content: Any,
        priority: str = "when_idle",
        input_provider: Any | None = None,
        agent_name: str | None = None,
        message_id: str | None = None,
        model_id: str | None = None,
        provider_id: str | None = None,
        assistant_msg_id: str | None = None,
        meta: Any = None,
        **kwargs: Any,
    ) -> str | None:
        """Route a message through SessionPool.receive_request().

        Creates the session if it does not yet exist. Stores the input
        provider on the session for auto-resume.

        If the session is checkpointed and ``deferred_tool_results`` is provided
        (via ``**kwargs``), :meth:`SessionPool.resume_session` is called first to
        replay deferred results into the agent loop before accepting new input.

        Args:
            session_id: Target session.
            content: Message / prompt content.
            priority: "when_idle" to queue, "asap" to inject into active turn.
            input_provider: Optional input provider for the agent.
            agent_name: Agent to bind if the session must be created.
            message_id: Optional user message ID from the REST handler.
                Passed to ``send_message()`` so it flows to
                ``UserMessageInsertedEvent`` for dedup with the protocol
                handler's own emission.
            model_id: Optional model ID from the REST handler (for agent/model
                propagation). Stored as pending so ``_before_consumer_loop`` can
                set it on the assistant message instead of hardcoding
                "default".
            provider_id: Optional provider ID from the REST handler. Same
                propagation path as ``model_id``.
            assistant_msg_id: Optional canonical assistant message ID from the
                REST handler. Stored as pending so ``_before_consumer_loop``
                can reuse it instead of generating an independent
                ``assistant_msg_id`` (D14). Falls back to ``message_id`` if
                not provided.
            meta: Protocol-specific metadata to carry through to
                ``UserMessageInsertedEvent``. When set, the event consumer
                uses it to reconstruct the full user message (e.g. OpenCode
                parts) instead of falling back to text-only content.
            **kwargs: Additional arguments passed to the turn runner.
                Supports ``deferred_tool_results`` for checkpoint replay.

        Returns:
            The ``message_id`` string on success, ``None`` on failure.
        """
        # --- Checkpoint replay: resume session before new input ----------
        deferred_results = kwargs.pop("deferred_tool_results", None)
        if deferred_results is not None and self.session_pool.sessions.store is not None:
            stored = await self.session_pool.sessions.store.load_session(session_id)
            if stored is not None and stored.status == "checkpointed":
                await self.session_pool.resume_session(
                    session_id,
                    deferred_results,
                    source="opencode_route_message",
                )

        # Store the canonical assistant_msg_id so _before_consumer_loop can
        # reuse it instead of generating an independent assistant_msg_id (D14).
        # Falls back to message_id for backward compatibility.
        pending_id = assistant_msg_id if assistant_msg_id is not None else message_id
        if pending_id is not None:
            self._pending_message_ids[session_id] = pending_id
        # Store model metadata so _before_consumer_loop can propagate the
        # real agent/model onto the assistant message instead of hardcoding
        # "wolfharness"/"default"/"wolfharness".
        self._pending_message_metadata[session_id] = {
            "message_id": message_id,
            "model_id": model_id,
            "provider_id": provider_id,
        }

        session_state = self.session_pool.sessions.get_session(session_id)
        if session_state is None:
            await self.create_session(session_id, agent_name=agent_name)
        else:
            # Ensure event consumer is running even for pre-existing sessions.
            # Sessions created via other paths (e.g. get_or_load_session) don't
            # have the consumer started, which would leave EventBus events
            # unconsumed and the frontend blank.
            await self._start_event_consumer(session_id)

        if input_provider is not None:
            session_state = self.session_pool.sessions.get_session(session_id)
            if session_state is not None:
                session_state.input_provider = input_provider

        from wolfharness.lifecycle.types import DeliveryMode

        delivery_mode = DeliveryMode.STEER if priority == "asap" else DeliveryMode.QUEUE
        return await self.session_pool.send_message(
            session_id=session_id,
            content=content,
            mode=delivery_mode,
            input_provider=input_provider,
            message_id=message_id,
            meta=meta,
        )

    async def abort_session(self, session_id: str) -> None:
        """Abort the active run for a session.

        Args:
            session_id: The session whose run should be cancelled.
        """
        self.session_pool.sessions.cancel_run_for_session(session_id)
        await self.server_state.broadcast_event(
            SessionErrorEvent.create(
                session_id=session_id,
                error_name="SessionAborted",
                error_message="Session was aborted by the user",
            )
        )

    async def attach_input_provider(
        self,
        session_id: str,
        input_provider: Any,
    ) -> None:
        """Attach an input provider to a session.

        Args:
            session_id: The session to attach the provider to.
            input_provider: The input provider instance.
        """
        session_state = self.session_pool.sessions.get_session(session_id)
        if session_state is not None:
            session_state.input_provider = input_provider

    async def get_session_status(self, session_id: str) -> SessionStatus | None:
        """Get the current status of a session.

        Checks the SessionPool for active runs and falls back to the
        server state's session status cache.

        Args:
            session_id: The session to look up.

        Returns:
            The session status, or a default idle status if not found.
        """
        session = self.session_pool.sessions.get_session(session_id)
        if session is not None:
            run_id = session.current_run_id
            if run_id is not None:
                run_handle = self.session_pool.sessions._runs.get(run_id)
                if run_handle is not None and run_handle.is_running:
                    return SessionStatus(type="busy")

        return SessionStatus(type="idle")

    async def shutdown(self) -> None:
        """Shutdown the integration and stop all consumers and bridges."""
        for session_id in list(self._session_groups.keys()):
            try:
                await self.stop_event_consumer(session_id)
            except Exception:
                logger.exception(
                    "Failed to stop event consumer during shutdown",
                    session_id=session_id,
                )
        await self.session_pool.shutdown()

    def set_session_context_data(self, session_id: str, data: dict[str, Any]) -> None:
        """Store serialized EventProcessorContext data for session resume.

        The orchestrator calls this before :meth:`start_event_consumer` for
        a resumed session.  The data is consumed (popped) by
        :meth:`_before_consumer_loop` and used to reconstruct the context
        instead of creating a fresh one.

        Args:
            session_id: The session to store context data for.
            data: Serialized context dict from :meth:`EventProcessorContext.serialize`.
        """
        self._resume_contexts[session_id] = data

    def get_session_context_data(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve and consume serialized EventProcessorContext data for resume.

        Returns the stored data and removes it so it is consumed exactly once.

        Args:
            session_id: The session to retrieve context data for.

        Returns:
            The serialized context dict, or ``None`` if no resume data is set.
        """
        return self._resume_contexts.pop(session_id, None)
