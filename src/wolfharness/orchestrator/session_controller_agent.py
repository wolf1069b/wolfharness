"""Agent management mixin for SessionController.

Extracted from session_controller.py as part of the session-debt-cleanup file split.
Contains session creation, agent resolution, and session listing methods.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import anyio

from wolfharness.log import get_logger
from wolfharness_server.opencode_server.models.session_info import SessionInfo


if TYPE_CHECKING:
    import asyncio

    from wolfharness.agents.base_agent import BaseAgent
    from wolfharness.delegation import AgentPool
    from wolfharness.orchestrator.runtime_registry import RuntimeAgentRegistry
    from wolfharness_storage.protocols import SessionPersistence

# SessionState is imported at runtime (not under TYPE_CHECKING) because it is
# used in _get_or_create_session_locked. This works because session_controller.py
# defines SessionState before importing this module.
from wolfharness.orchestrator.session_controller import SessionState


logger = get_logger(__name__)


class SessionControllerAgentMixin:
    """Mixin providing agent and session management methods for SessionController.

    Attributes:
        pool: The agent pool (provided by SessionController).
        store: Session persistence store (provided by SessionController).
        _sessions: Active sessions dict (provided by SessionController).
        _session_agents: Per-session agent cache (provided by SessionController).
        _children: Parent→children mapping (provided by SessionController).
        _lock: Global lock (provided by SessionController).
        _todo_lock: Todo lock (provided by SessionController).
        _runtime_registry: Runtime agent registry (provided by SessionController).
    """

    pool: AgentPool[Any]
    store: SessionPersistence | None
    _sessions: dict[str, SessionState]
    _session_agents: dict[str, BaseAgent[Any, Any]]
    _children: dict[str, list[str]]
    _session_scopes: dict[str, anyio.CancelScope]
    _lock: asyncio.Lock
    _todo_lock: asyncio.Lock
    _runtime_registry: RuntimeAgentRegistry
    _event_bus: Any

    def _increment_mcp_count(self, _agent: Any) -> None: ...

    async def get_or_create_session(
        self,
        session_id: str,
        agent_name: str | None = None,
        parent_session_id: str | None = None,
        lifecycle_policy: str | None = None,
        **metadata: Any,
    ) -> tuple[SessionState, bool]:
        """Get or create a session.

        Uses single global lock for simplicity and safety.
        Session creation is infrequent - no need for DCL optimization.

        Args:
            session_id: Unique identifier for the session.
            agent_name: Name of the agent to associate with the session.
            parent_session_id: Optional parent session ID for hierarchical sessions.
            lifecycle_policy: Optional lifecycle policy override.
            **metadata: Arbitrary metadata to attach to the session.

        Returns:
            A tuple of (session_state, was_created) where was_created is True
            if the session was newly created, False if it already existed.
        """
        if not session_id or not session_id.strip():
            raise ValueError("session_id cannot be empty or whitespace")

        async with self._lock:
            return await self._get_or_create_session_locked(
                session_id, agent_name, parent_session_id, lifecycle_policy, **metadata
            )

    def _state_to_data(self, state: SessionState) -> Any:
        """Convert SessionState to persistable SessionData.

        Args:
            state: The session state to convert.

        Returns:
            Persistable session data.
        """
        from datetime import UTC, datetime

        from wolfharness.sessions.models import SessionData

        # Derive created_at from created_at_ns (epoch nanoseconds) so the
        # persisted timestamp matches the session ID's embedded timestamp.
        # created_at_wall is a separate get_now() call that can differ by
        # milliseconds, causing time.created ≠ ID-embedded timestamp.
        created_at = datetime.fromtimestamp(state.created_at_ns / 1_000_000_000, tz=UTC)
        # Derive last_active from last_active_at_ns for the same reason —
        # get_now() is a separate wall-clock call that can diverge.
        last_active = datetime.fromtimestamp(state.last_active_at_ns / 1_000_000_000, tz=UTC)

        return SessionData(
            session_id=state.session_id,
            agent_name=state.agent_name,
            parent_id=state.parent_session_id,
            project_id=state.metadata.get("project_id"),
            cwd=state.metadata.get("cwd"),
            agent_type=state.metadata.get("agent_type"),
            created_at=created_at,
            last_active=last_active,
            metadata=state.metadata,
        )

    async def _get_or_create_session_locked(
        self,
        session_id: str,
        agent_name: str | None = None,
        parent_session_id: str | None = None,
        lifecycle_policy: str | None = None,
        **metadata: Any,
    ) -> tuple[SessionState, bool]:
        """Get or create a session - caller MUST hold self._lock.

        This internal method avoids deadlock when called from
        get_or_create_session_agent() which already holds the lock.

        Args:
            session_id: Unique identifier for the session.
            agent_name: Name of the agent to associate with the session.
            parent_session_id: Optional parent session ID for hierarchical sessions.
            lifecycle_policy: Optional lifecycle policy override.
            **metadata: Arbitrary metadata to attach to the session.

        Returns:
            A tuple of (session_state, was_created) where was_created is True
            if the session was newly created, False if it already existed.
        """
        from wolfharness.orchestrator.session_controller import SessionLifecyclePolicy

        if session_id in self._sessions:
            state = self._sessions[session_id]
            state.last_active_at = time.monotonic()
            state.last_active_at_ns = time.time_ns()
            return state, False

        effective_policy = lifecycle_policy or (
            self._sessions.get(parent_session_id, SessionState("", "")).lifecycle_policy
            if parent_session_id and parent_session_id in self._sessions
            else SessionLifecyclePolicy.default()
        )

        # Ensure agent_name is always a real string (guards against Mock
        # attributes in tests where pool.main_agent_name is a MagicMock).
        _main_agent_name = self.pool.main_agent_name
        if not isinstance(_main_agent_name, str):
            _main_agent_name = "default"
        state = SessionState(
            session_id=session_id,
            agent_name=agent_name or _main_agent_name,
            parent_session_id=parent_session_id,
            lifecycle_policy=effective_policy,
            metadata=metadata,
            checkpoint_enabled=self.store is not None,
        )
        # Sync created_at_ns with the timestamp embedded in the session ID.
        # This ensures time.created order matches session ID lexicographic
        # order, since both come from the same now_ms() call inside
        # generate_session_id() / ascending() / descending().
        from wolfharness.utils.identifiers import extract_timestamp_ms

        ts_ms = extract_timestamp_ms(session_id)
        if ts_ms is not None:
            state.created_at_ns = ts_ms * 1_000_000
            state.last_active_at_ns = ts_ms * 1_000_000
        self._sessions[session_id] = state

        # Clear todos for new top-level sessions only (not subagents)
        # This prevents accumulation of todos from previous sessions
        # Use dedicated lock to prevent race conditions with concurrent sessions
        if parent_session_id is None and self.pool.todos is not None:
            _entries = self.pool.todos.entries
            if isinstance(_entries, (list, tuple)) and len(_entries) > 0:
                async with self._todo_lock:
                    # Double-check after acquiring lock
                    _entries = self.pool.todos.entries
                    if isinstance(_entries, (list, tuple)) and len(_entries) > 0:
                        cleared_count = len(_entries)
                        self.pool.todos.clear()
                        logger.info(
                            "Cleared todos for new top-level session",
                            session_id=session_id,
                            agent_name=state.agent_name,
                            cleared_entries=cleared_count,
                        )

        if parent_session_id and effective_policy in ("cascade", "bound"):
            parent_scope = self._session_scopes.get(parent_session_id)
            if parent_scope is not None:
                child_scope = anyio.CancelScope()
                self._session_scopes[session_id] = child_scope
            else:
                self._session_scopes[session_id] = anyio.CancelScope()
        else:
            self._session_scopes[session_id] = anyio.CancelScope()
        if self.store is not None:
            # Only save if no existing data — callers like
            # ACPSessionManager.create_session() may have already
            # persisted richer SessionData (with cwd, project_id, etc.)
            existing = await self.store.load_session(session_id)
            if existing is None:
                await self.store.save_session(self._state_to_data(state))
        if parent_session_id:
            self._children.setdefault(parent_session_id, []).append(session_id)
        logger.info("Created session", session_id=session_id, agent_name=state.agent_name)
        return state, True

    async def get_or_create_session_agent(
        self,
        session_id: str,
        agent_name: str | None = None,
        input_provider: Any | None = None,
    ) -> BaseAgent[Any, Any]:
        """Get or create a dedicated agent for a session.

        Delegates agent creation to ``AgentFactory.create_session_agent()``,
        handling caching, session lookup, and config resolution locally.

        NOTE: Always acquires self._lock to prevent races with close_session().

        Args:
            session_id: Unique identifier for the session.
            agent_name: Name of the agent to use.
            input_provider: Optional input provider for the agent.

        Returns:
            The agent instance (per-session or shared).
        """
        if not session_id or not session_id.strip():
            raise ValueError("session_id cannot be empty or whitespace")

        async with self._lock:
            if session_id in self._session_agents:
                agent = self._session_agents[session_id]
                # Update input_provider on cached agent if a new one is provided.
                # Without this, the agent keeps the stale (or None) input_provider
                # from when it was first cached, causing elicitation failures.
                if input_provider is not None:
                    session = self._sessions.get(session_id)
                    if session is not None:
                        session.input_provider = input_provider
                    agent._input_provider = input_provider
                return agent

            session, _was_created = await self._get_or_create_session_locked(session_id, agent_name)
            agent_name = agent_name or session.agent_name

            cfg = self.pool.manifest.agents.get(agent_name)
            if cfg is None:
                cfg = self._runtime_registry.lookup(agent_name)

            if cfg is None:
                # Config not found
                available_manifest = list(self.pool.manifest.agents.keys())
                available_runtime = self._runtime_registry.names()
                msg = (
                    f"Agent config not found: {agent_name!r}. "
                    f"Available in manifest: {available_manifest}. "
                    f"Available in runtime registry: {available_runtime}."
                )
                raise RuntimeError(msg)

            # Resolve parent agent for child sessions.
            parent_agent: BaseAgent[Any, Any] | None = None
            if session.parent_session_id:
                parent_state = self._sessions.get(session.parent_session_id)
                parent_agent = parent_state.agent if parent_state else None

            # Delegate creation to AgentFactory.
            agent = await self.pool._factory.create_session_agent(
                agent_name=agent_name,
                session_id=session_id,
                host_context=self.pool.get_context(),
                session=session,
                cfg=cfg,
                input_provider=input_provider,
                parent_agent=parent_agent,
            )

            # Cache and session state.
            if input_provider is not None:
                session.input_provider = input_provider
            self._session_agents[session_id] = agent
            session.agent = agent

            # Path A (child): False — parent manages lifecycle.
            # Paths B/C (main/non-native): True — close_session calls __aexit__.
            if session.parent_session_id:
                session.is_per_session_agent = False
            else:
                session.is_per_session_agent = True
                self._increment_mcp_count(agent)

            # Initialize lifecycle dimensions and run crash recovery once
            # per session (per-prompt RunHandle migration, task 1.2).
            await self._initialize_lifecycle_and_recovery(session, agent)

            logger.info("Created session agent", session_id=session_id, agent_name=agent_name)
            return agent

    async def _initialize_lifecycle_and_recovery(
        self,
        session: SessionState,
        agent: BaseAgent[Any, Any],
    ) -> None:
        """Initialize lifecycle dimensions and run crash recovery for a session.

        Creates the 6 lifecycle dimensions (Journal, SnapshotStore,
        CommChannel, EventTransport, TriggerSource) and stores them on
        ``SessionState``. Then runs the full crash recovery logic
        (``journal.resume()``, event replay, recovery strategy
        application, dimension subscription, initial snapshot).

        This runs ONCE per session at agent creation time, NOT per
        RunHandle. In the per-prompt model, dimensions persist across
        RunHandles and recovery is a session-level concern.

        Args:
            session: The session state to initialize dimensions on.
            agent: The agent instance for this session.
        """
        from wolfharness.lifecycle import (
            DirectChannel,
            InProcessTransport,
            MemoryJournal,
            MemorySnapshotStore,
            ProtocolChannel,
            ProtocolTrigger,
            RunState,
        )

        event_bus = self._event_bus

        # Create ProtocolChannel when EventBus is available (protocol
        # server sessions). Otherwise use DirectChannel (standalone).
        journal = MemoryJournal()
        if event_bus is not None:
            comm_channel: ProtocolChannel | DirectChannel = ProtocolChannel(
                journal=journal,
                event_bus=event_bus,
                session_id=session.session_id,
            )
        else:
            comm_channel = DirectChannel(journal)

        # Create SnapshotStore (in-memory for now; durable via
        # agent._lifecycle_config is handled separately).
        snapshot_store = MemorySnapshotStore()

        # EventTransport is always in-process for M2.
        event_transport = InProcessTransport()

        # TriggerSource: ProtocolTrigger for protocol sessions,
        # ImmediateTrigger for standalone.
        trigger_source: ProtocolTrigger | None = ProtocolTrigger()

        # Store dimensions on SessionState.
        session._journal = journal
        session._snapshot_store = snapshot_store
        session._comm_channel = comm_channel
        session._event_transport = event_transport
        session._trigger_source = trigger_source
        session._lifecycle_session_id = session.session_id

        # Wire HostContext and AgentRegistry on SessionState (task 1.6).
        from wolfharness.host.registry import AgentRegistry

        host_ctx = self.pool.get_context()
        agent_registry = AgentRegistry(
            dict.fromkeys(self.pool.manifest.agents),  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        )
        session._host_context = host_ctx
        session._agent_registry = agent_registry
        session._event_bus = event_bus

        # Run crash recovery (full _handle_recovery logic, task 1.2).
        if journal is not None and comm_channel is not None and snapshot_store is not None:
            resume_result = journal.resume(snapshot_store)
            if resume_result is not None:
                if resume_result.is_inflight:
                    # Replay journaled events through CommChannel.
                    comm_channel.set_replaying(True)
                    try:
                        for event in resume_result.events:
                            await comm_channel.publish(event)
                    finally:
                        comm_channel.set_replaying(False)
                    session._recovered_inflight_turn_id = resume_result.inflight_turn_id
                    # Apply recovery strategy.
                    if (
                        session._recover_strategy == "mark_interrupted"
                        and resume_result.inflight_turn_id is not None
                    ):
                        snapshot_store.save_turn_result(
                            resume_result.inflight_turn_id,
                            {"status": "interrupted"},
                        )
                    # "retry" strategy: the recovered prompt is stored
                    # on SessionState._resume_result for the first
                    # RunHandle to pick up.
                    session._resume_result = resume_result
                # Non-inflight: no special action needed.
            else:
                # Fresh start: save initial snapshot.
                snapshot_store.save(
                    {"state": RunState.IDLE.value, "run_id": None},
                )

            # Subscribe dimensions to the session (not to a RunHandle,
            # since RunHandle is now ephemeral).
            if trigger_source is not None:
                trigger_source.subscribe(session)
            comm_channel.attach(session)

    def list_sessions(self) -> list[SessionInfo]:
        """List all active sessions.

        Returns:
            A list of SessionInfo DTOs for all active sessions.
        """
        return [
            SessionInfo(
                session_id=s.session_id,
                agent_name=s.agent_name,
                created_at=s.created_at,
                last_active_at=s.last_active_at,
                is_per_session_agent=s.is_per_session_agent,
                status="busy" if s.current_run_id is not None else "idle",
            )
            for s in self._sessions.values()
        ]

    def get_session_agent(self, session_id: str) -> BaseAgent[Any, Any] | None:
        """Get the agent for a session.

        Returns the per-session agent if one exists, otherwise the shared
        agent that was assigned to the session.  If the session has no
        agent assigned yet, a warning is logged and None is returned.

        Args:
            session_id: The session ID to look up.

        Returns:
            The agent instance, or None if the session is unknown.
        """
        session = self._sessions.get(session_id)
        if session is None:
            logger.warning("Session not found", session_id=session_id)
            return None
        agent = self._session_agents.get(session_id)
        if agent is None:
            logger.warning(
                "No agent assigned for session - falling back to shared agent",
                session_id=session_id,
            )
            return None
        return agent
