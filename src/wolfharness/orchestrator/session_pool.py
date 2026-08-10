"""Session pool for high-level session management.

Extracted from orchestrator/core.py as part of the thin-wrapper refactor.
Combines session and turn management for protocol handlers.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import contextlib
import time
from typing import TYPE_CHECKING, Any, Final

from wolfharness.agents.events import (
    SessionResumeEvent,
)
from wolfharness.log import get_logger
from wolfharness.orchestrator.event_bus import EventBus
from wolfharness.orchestrator.session_controller import (
    CheckpointMismatchError,
    SessionBusyError,
    SessionController,
    SessionNotFoundError,
    SessionState,
)
from wolfharness.orchestrator.session_pool_config import SessionPoolConfig


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from wolfharness.agents.base_agent import BaseAgent
    from wolfharness.agents.native_agent import Agent
    from wolfharness.agents.native_agent.checkpoint import CheckpointData
    from wolfharness.delegation import AgentPool
    from wolfharness.messaging import ChatMessage
    from wolfharness.sessions.models import (
        ElicitationResumePayload,
        PendingDeferredCall,
        SessionData,
    )
    from wolfharness_storage.protocols import SessionPersistence


logger = get_logger(__name__)

DEFAULT_MAX_AUTO_RESUME: Final[int] = 10

# Mixin imports placed after session_controller imports to avoid circular issues.
from wolfharness.orchestrator.session_pool_messaging import SessionPoolMessagingMixin  # noqa: E402
from wolfharness.orchestrator.session_pool_runs import SessionPoolRunsMixin  # noqa: E402
from wolfharness.orchestrator.session_pool_teams import SessionPoolTeamsMixin  # noqa: E402


class SessionPool(
    SessionPoolMessagingMixin,
    SessionPoolRunsMixin,
    SessionPoolTeamsMixin,
):
    """High-level session pool combining session and turn management.

    This is the main interface used by protocol handlers.
    """

    def __init__(
        self,
        pool: AgentPool[Any],
        store: SessionPersistence | None = None,
        enable_auto_resume: bool = True,
        enable_event_bus: bool = True,
        max_auto_resume: int = DEFAULT_MAX_AUTO_RESUME,
        max_concurrent_runs: int | None = None,
        replay_buffer_size: int = 100,
        config: SessionPoolConfig | None = None,
    ) -> None:
        """Initialize the session pool.

        Args:
            pool: The agent pool to resolve agents from.
            store: Optional session store for persistence.
            enable_auto_resume: Whether to enable auto-resume loop.
            enable_event_bus: Whether to enable cross-turn event routing.
            max_auto_resume: Maximum auto-resume iterations.
            max_concurrent_runs: Maximum number of concurrent runs across all sessions.
            replay_buffer_size: Maximum number of events retained per session for replay.
            config: Optional SessionPoolConfig for tunable parameters
                (message cache size, TTL intervals). Defaults to
                ``SessionPoolConfig()`` with standard defaults.
        """
        self.pool = pool
        self._config = config or SessionPoolConfig()
        self.sessions = SessionController(
            pool,
            store=store,
            cleanup_callback=self.close_session,
            max_concurrent_runs=max_concurrent_runs,
            session_ttl_seconds=self._config.session_ttl_seconds,
            cleanup_interval_seconds=self._config.cleanup_interval_seconds,
            deferred_cleanup_interval_seconds=self._config.deferred_cleanup_interval_seconds,
        )
        self._event_bus = EventBus(
            session_controller=self.sessions,
            replay_buffer_size=replay_buffer_size,
        )
        self.sessions._event_bus = self._event_bus
        self._enable_auto_resume = enable_auto_resume
        self._enable_event_bus = enable_event_bus
        self._runs_lock: asyncio.Lock = asyncio.Lock()
        self._resume_locks: dict[str, asyncio.Lock] = {}
        self._resume_locks_lock = asyncio.Lock()
        self._message_cache: OrderedDict[str, list[ChatMessage[Any]]] = OrderedDict()
        self._message_cache_maxsize: int = self._config.message_cache_maxsize
        self._elicitation_registries: dict[str, Any] = {}

    async def start(self) -> None:
        """Start the session pool and background tasks."""
        await self.sessions.start_cleanup_task()
        logger.info("SessionPool started")

    async def shutdown(self) -> None:
        """Shutdown the session pool and cancel background tasks."""
        await self.sessions.stop_cleanup_task()
        active_sessions = list(self.sessions._sessions.keys())
        for session_id in active_sessions:
            try:
                await self.close_session(session_id)
            except Exception:
                logger.exception(
                    "Failed to close session during shutdown",
                    session_id=session_id,
                )
            except asyncio.CancelledError:
                logger.warning(
                    "CancelledError during shutdown of session",
                    session_id=session_id,
                )
        logger.info("SessionPool shut down")

    @property
    def event_bus(self) -> EventBus:
        """Get the event bus for cross-turn event routing."""
        return self._event_bus

    async def create_session(
        self,
        session_id: str,
        agent_name: str | None = None,
        parent_session_id: str | None = None,
        lifecycle_policy: str | None = None,
        **metadata: Any,
    ) -> SessionState:
        """Create or get a session.

        Args:
            session_id: Unique identifier for the session.
            agent_name: Name of the agent to associate with the session.
            parent_session_id: Optional parent session ID for hierarchical sessions.
            lifecycle_policy: Optional lifecycle policy override.
            **metadata: Arbitrary metadata to attach to the session.

        Returns:
            The session state.
        """
        if parent_session_id is not None and self.sessions.store is not None:
            parent_data = await self.sessions.store.load_session(parent_session_id)
            if parent_data is not None:
                metadata.setdefault("project_id", parent_data.project_id)
                metadata.setdefault("cwd", parent_data.cwd)
        state, _was_created = await self.sessions.get_or_create_session(
            session_id, agent_name, parent_session_id, lifecycle_policy, **metadata
        )
        return state

    async def create_child_session(
        self,
        parent_session_id: str,
        agent_name: str,
        agent_type: str = "native",
        *,
        session_id: str | None = None,
        lifecycle_policy: str | None = None,
        **metadata: Any,
    ) -> SessionState:
        """Create a child session linked to a parent session.

        This is a first-class API for hierarchical session creation.
        The child session inherits the parent's ``project_id`` and ``cwd``
        automatically. A sortable session ID is generated via
        :func:`wolfharness.utils.identifiers.generate_session_id` when not
        provided.

        Args:
            parent_session_id: The parent session ID to link to.
            agent_name: Name of the agent for the child session.
            agent_type: Type of the agent (``"native"``, ``"acp"``, etc.).
            session_id: Optional explicit session ID (generated if None).
            lifecycle_policy: Optional lifecycle policy override.
            **metadata: Additional metadata to attach to the child session.

        Returns:
            The child session state.
        """
        from wolfharness.utils.identifiers import generate_session_id

        child_session_id = session_id or generate_session_id()
        metadata.setdefault("agent_type", agent_type)
        return await self.create_session(
            child_session_id,
            agent_name=agent_name,
            parent_session_id=parent_session_id,
            lifecycle_policy=lifecycle_policy,
            **metadata,
        )

    async def _get_resume_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create per-session lock for resume serialization.

        Args:
            session_id: Session identifier.

        Returns:
            The per-session resume lock.
        """
        async with self._resume_locks_lock:
            lock = self._resume_locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._resume_locks[session_id] = lock
            return lock

    @contextlib.asynccontextmanager
    async def _with_resume_lock(
        self, session_id: str, *, allow_active_run: bool = False
    ) -> AsyncIterator[SessionState | None]:
        """Acquire per-session resume lock with state validation.

        Ensures only one resume runs per session at a time and that
        the session is in a resumable state (no active run, persisted
        status is ``"checkpointed"``).

        Args:
            session_id: Session to lock.
            allow_active_run: When True, skip the ``current_run_id`` check.
                Used for in-process elicitation resume where the agent run
                is intentionally still alive.

        Yields:
            The live ``SessionState``, or ``None`` if no live session exists.

        Raises:
            SessionBusyError: If the session has an active run or its
                persisted status is not ``"checkpointed"``.
        """
        resume_lock = await self._get_resume_lock(session_id)
        async with resume_lock:
            session = self.sessions.get_session(session_id)
            if not allow_active_run and session is not None and session.current_run_id is not None:
                raise SessionBusyError(session_id, session.current_run_id)

            if self.sessions.store is not None:
                current_data = await self.sessions.store.load_session(session_id)
                # When allow_active_run is True (in-process elicitation
                # resume), the persisted status may still be "active"
                # because the elicitation bridge checkpoint saves
                # checkpoint data but doesn't update the session store
                # status. Allow "active" in that case.
                if (
                    current_data is not None
                    and current_data.status != "checkpointed"
                    and (not allow_active_run or current_data.status != "active")
                ):
                    raise SessionBusyError(session_id, current_data.status)

            yield session

    async def _load_checkpoint_data(self, session_id: str) -> CheckpointData:
        """Load checkpoint data for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Checkpoint data.

        Raises:
            SessionNotFoundError: If no checkpoint exists for the session.
        """
        from wolfharness.agents.native_agent.checkpoint import CheckpointManager

        storage = self.pool.storage
        if storage is None:
            raise SessionNotFoundError(session_id)

        checkpoint_mgr = CheckpointManager(storage)
        data = await checkpoint_mgr.load_checkpoint(session_id)
        if data is None:
            raise SessionNotFoundError(session_id)
        return data

    async def _reconstruct_native_agent(
        self,
        session_id: str,
        agent_name: str,
    ) -> Agent[Any, Any]:
        """Reconstruct a native agent from config for session resume.

        Args:
            session_id: Session identifier.
            agent_name: Name of the agent configuration to use.

        Returns:
            A reconstructed native agent instance.

        Raises:
            SessionNotFoundError: If the agent config is not found.
        """
        from wolfharness.models.agents import NativeAgentConfig
        from wolfharness_config.context import ConfigContextManager

        cfg = self.pool.manifest.agents.get(agent_name)
        if cfg is None:
            raise SessionNotFoundError(session_id)

        if not isinstance(cfg, NativeAgentConfig):
            raise SessionNotFoundError(session_id)

        if cfg.name is None:
            cfg = cfg.model_copy(update={"name": agent_name})

        session = self.sessions.get_session(session_id)
        input_provider = session.input_provider if session else None

        with ConfigContextManager(self.pool._config_file_path):
            agent: Agent[Any, Any] = cfg.get_agent(
                input_provider=input_provider,
                pool=self.pool,
            )

        # Pool-level skill tools (load_skill/list_skills) are now owned by
        # SkillManagerCap, registered at pool scope in ExtensionRegistry.
        # No explicit injection needed here.

        await agent.__aenter__()
        return agent

    async def _reconstruct_acp_agent(
        self,
        session_id: str,
        agent_name: str,
    ) -> BaseAgent[Any, Any]:
        """Reconstruct an ACP agent from config for session resume.

        Args:
            session_id: Session identifier.
            agent_name: Name of the agent configuration to use.

        Returns:
            A reconstructed ACP agent with reopened subprocess.

        Raises:
            SessionNotFoundError: If the agent config is not found.
        """
        from wolfharness_config.context import ConfigContextManager

        cfg = self.pool.manifest.agents.get(agent_name)
        if cfg is None:
            raise SessionNotFoundError(session_id)

        if cfg.name is None:
            cfg = cfg.model_copy(update={"name": agent_name})

        session = self.sessions.get_session(session_id)
        input_provider = session.input_provider if session else None

        with ConfigContextManager(self.pool._config_file_path):
            agent: BaseAgent[Any, Any] = cfg.get_agent(
                input_provider=input_provider,
                pool=self.pool,
            )

        # Pool-level skill tools (load_skill/list_skills) are now owned by
        # SkillManagerCap, registered at pool scope in ExtensionRegistry.
        # No explicit injection needed here.

        await agent.__aenter__()
        return agent

    async def _resume_native_agent(  # noqa: PLR0915
        self,
        session_data: SessionData,
        checkpoint: CheckpointData,
        results: Any,
        elicitation_payloads: list[ElicitationResumePayload] | None = None,
    ) -> None:
        """Resume a native agent from checkpoint with deferred results.

        Routes resume through the SessionPool's normal turn management
        (``run_stream()`` → ``_run_stream_run_turn()`` →
        ``_create_run_handle()``) so the resumed turn has full RunHandle
        lifecycle support (journal, snapshot, event delivery, session
        coordination).

        The checkpoint's ``message_history`` is passed as a
        ``list[ModelMessage]`` (NOT wrapped in ``MessageHistory``) to
        initialize ``RunHandle._message_history``.

        Elicitation responses are delivered via TWO mechanisms:

        1. **DeferredToolResults** (primary): ``ToolReturnPart`` entries
           built from ``elicitation_payloads``, keyed by ``tool_call_id``.
           pydantic-ai matches these against the ``ModelResponse`` in
           ``message_history`` and uses the results directly, skipping
           tool execution. This is necessary because ``agentlet.iter()``
           starts from ``UserPromptNode`` and generates a NEW
           ``ModelResponse`` — it does NOT replay the old one.

        2. **cached_elicitation_responses** (fallback): Set on
           ``AgentRunContext`` for cases where the tool does re-execute
           and calls ``handle_elicitation()``.

        Args:
            session_data: Persisted session data.
            checkpoint: Checkpoint data with message_history and pending_calls.
            results: DeferredToolResults for resolving pending deferred calls.
            elicitation_payloads: Optional elicitation responses for crash
                recovery resume of elicitation deferred calls.

        Raises:
            SessionNotFoundError: If agent config is not found.
            RuntimeError: If agent.run() fails (pending_calls remain uncleared).
        """
        agent = await self._reconstruct_native_agent(
            session_data.session_id, session_data.agent_name
        )

        # Detect agent config drift between checkpoint and resume.
        # The hash check is advisory: if we can't compute the current hash
        # (e.g. agent has no tools attribute, or tools is a mock in tests),
        # we skip the comparison and proceed with resume.
        if session_data.agent_config_hash:
            try:
                from wolfharness.agents.native_agent.checkpoint import (
                    compute_agent_config_hash,
                )

                agent_tools = await agent._get_all_tools()
                current_hash = compute_agent_config_hash(agent_tools)
                if current_hash != session_data.agent_config_hash:
                    logger.warning(
                        "Agent config hash mismatch — tools may have changed since checkpoint",
                        session_id=session_data.session_id,
                        stored_hash=session_data.agent_config_hash,
                        current_hash=current_hash,
                    )
            except Exception:
                logger.debug(
                    "Could not compute agent config hash for drift check",
                    session_id=session_data.session_id,
                    exc_info=True,
                )

        # Build cached elicitation responses for crash recovery (fallback).
        cached_elicitation: dict[str, Any] = {}
        # Build DeferredToolResults from elicitation_payloads (primary mechanism).
        # pydantic-ai's agentlet.iter() starts from UserPromptNode, NOT from
        # ModelResponse in message_history. It generates a NEW ModelResponse
        # with a NEW tool_call_id, so cached_elicitation_responses (keyed by
        # the OLD tool_call_id) are never hit. By passing DeferredToolResults,
        # pydantic-ai matches them against the ModelResponse in message_history
        # and uses the results directly, skipping tool execution entirely.
        from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
        from pydantic_ai.tools import DeferredToolResults

        # Build a mapping from elicitation handle (PendingDeferredCall.tool_call_id)
        # to the actual ToolCallPart.tool_call_id in the ModelResponse.
        # This is needed because handle_elicitation() may use run_ctx.run_id as
        # the handle when agent_ctx.tool_call_id is not set (MCP tools without
        # AgentContext param). The DeferredToolResults must be keyed by the
        # ToolCallPart.tool_call_id (what pydantic-ai expects), not the
        # elicitation handle.
        pending_by_handle: dict[str, PendingDeferredCall] = {
            call.tool_call_id: call for call in checkpoint.pending_calls
        }
        # Find ToolCallPart.tool_call_id for each pending call.
        # Match by tool_name first; fall back to positional matching
        # (MCP tools without AgentContext param may have empty tool_name
        # in PendingDeferredCall, so name matching fails).
        tool_call_id_map: dict[str, str] = {}  # handle → ToolCallPart.tool_call_id
        last_model_response: ModelResponse | None = None
        for msg in reversed(checkpoint.message_history):
            if isinstance(msg, ModelResponse):
                last_model_response = msg
                break
        if last_model_response is not None:
            # Collect all ToolCallParts from the last ModelResponse
            tool_call_parts: list[ToolCallPart] = [
                part
                for part in last_model_response.parts
                if isinstance(part, ToolCallPart) and part.tool_call_id
            ]
            pending_calls_list = list(pending_by_handle.items())

            # First pass: match by tool_name
            matched_tc_indices: set[int] = set()
            for handle, pcall in pending_calls_list:
                if not pcall.tool_name:
                    continue
                for i, tc in enumerate(tool_call_parts):
                    if i in matched_tc_indices:
                        continue
                    if tc.tool_name == pcall.tool_name:
                        tool_call_id_map[handle] = tc.tool_call_id
                        matched_tc_indices.add(i)
                        break

            # Second pass: positional matching for unmatched handles
            unmatched_tc_indices = [
                i for i in range(len(tool_call_parts)) if i not in matched_tc_indices
            ]
            unmatched_handles = [h for h, _ in pending_calls_list if h not in tool_call_id_map]
            for handle, tc_idx in zip(unmatched_handles, unmatched_tc_indices, strict=False):
                tool_call_id_map[handle] = tool_call_parts[tc_idx].tool_call_id

        elicitation_tool_results: dict[str, Any] = {}
        if elicitation_payloads:
            from mcp.types import ElicitResult as MCPElicitResult

            from wolfharness.ui.elicitation import normalize_elicit_content

            for payload in elicitation_payloads:
                # Map elicitation handle to ToolCallPart.tool_call_id
                actual_tool_call_id = tool_call_id_map.get(
                    payload.deferred_handle, payload.deferred_handle
                )
                pcall_deferred: PendingDeferredCall | None = pending_by_handle.get(
                    payload.deferred_handle
                )
                tool_name = pcall_deferred.tool_name if pcall_deferred else ""
                match payload.action:
                    case "accept":
                        cached_elicitation[payload.deferred_handle] = MCPElicitResult(
                            action="accept",
                            content=normalize_elicit_content(payload.content),
                        )
                        elicitation_tool_results[actual_tool_call_id] = ToolReturnPart(
                            tool_name=tool_name,
                            content=payload.content or {},
                            tool_call_id=actual_tool_call_id,
                        )
                    case "decline":
                        cached_elicitation[payload.deferred_handle] = MCPElicitResult(
                            action="decline",
                        )
                        elicitation_tool_results[actual_tool_call_id] = ToolReturnPart(
                            tool_name=tool_name,
                            content="declined",
                            tool_call_id=actual_tool_call_id,
                        )
                    case "cancel":
                        cached_elicitation[payload.deferred_handle] = MCPElicitResult(
                            action="cancel",
                        )
                        elicitation_tool_results[actual_tool_call_id] = ToolReturnPart(
                            tool_name=tool_name,
                            content="cancelled",
                            tool_call_id=actual_tool_call_id,
                        )

        try:
            # Route through the pool's normal turn management so the
            # resumed turn has full RunHandle lifecycle (journal,
            # snapshot, event delivery, session coordination).
            # Pass message_history as list[ModelMessage] (NOT wrapped
            # in MessageHistory) — the pool path initializes
            # RunHandle._message_history directly.
            run_kwargs: dict[str, Any] = {
                "cached_elicitation_responses": cached_elicitation or None,
                "message_history": list(checkpoint.message_history),
            }
            # Build DeferredToolResults from elicitation_payloads.
            # This is the PRIMARY mechanism for crash recovery: pydantic-ai
            # matches these against the ModelResponse in message_history and
            # uses the results directly, skipping tool execution.
            # cached_elicitation_responses is kept as a fallback.
            if elicitation_tool_results:
                run_kwargs["deferred_tool_results"] = DeferredToolResults(
                    calls=elicitation_tool_results
                )
            elif getattr(results, "calls", None):
                # Non-elicitation deferred results from the caller
                run_kwargs["deferred_tool_results"] = results
            async for _ in self.run_stream(
                session_data.session_id,
                "",
                **run_kwargs,
            ):
                pass
        finally:
            await agent.__aexit__(None, None, None)

    async def _resume_acp_agent(
        self,
        session_data: SessionData,
        checkpoint: CheckpointData,
        results: Any,
    ) -> None:
        """Resume an ACP agent by reopening the subprocess and sending session/resume.

        Reopens the ACP subprocess and calls agent.run() to restart the
        session with restored state.

        Args:
            session_data: Persisted session data.
            checkpoint: Checkpoint data (used for metadata only; ACP agents
                manage their own message history).
            results: DeferredToolResults for resolving pending deferred calls.
        """
        agent = await self._reconstruct_acp_agent(session_data.session_id, session_data.agent_name)
        try:
            # ACP agents receive the resumed session context through run()
            run_fn: Any = agent.run
            await run_fn(
                message_history=list(checkpoint.message_history),
                deferred_tool_results=results,
            )
        finally:
            if hasattr(agent, "__aexit__"):
                await agent.__aexit__(None, None, None)

    async def _try_in_process_elicitation_resume(
        self,
        session_id: str,
        elicitation_payloads: list[ElicitationResumePayload],
    ) -> bool:
        """Attempt in-process elicitation resume by resolving futures.

        Checks if the agent run is still alive (in-process) and has an
        ``ElicitationFutureRegistry`` with pending futures for the given
        elicitation payloads. If so, resolves each future with the
        corresponding payload.

        Args:
            session_id: The session to check.
            elicitation_payloads: Elicitation responses to resolve.

        Returns:
            True if ALL payloads were resolved in-process, False otherwise.
        """
        run_handle = self._get_active_run_handle(session_id)
        if run_handle is None or run_handle.run_ctx is None:
            return False

        registry = run_handle.run_ctx.elicitation_registry
        if registry is None:
            return False

        all_resolved = True
        for payload in elicitation_payloads:
            if payload.deferred_handle not in registry:
                all_resolved = False
                continue
            registry.resolve(payload.deferred_handle, payload)

        if all_resolved:
            logger.debug(
                "In-process elicitation resume — all futures resolved",
                session_id=session_id,
                count=len(elicitation_payloads),
            )
        else:
            logger.debug(
                "In-process elicitation resume — some futures not found, "
                "falling back to crash recovery",
                session_id=session_id,
            )

        return all_resolved

    async def resume_session(  # noqa: PLR0915
        self,
        session_id: str,
        deferred_tool_results: Any,
        *,
        source: str = "resume_prompt",
        elicitation_payloads: list[ElicitationResumePayload] | None = None,
    ) -> None:
        """Resume a paused session with resolved deferred tool results.

        Loads the persisted SessionData, validates that deferred_tool_results
        cover all pending_deferred_calls (raising CheckpointMismatchError if not),
        and resumes execution via the appropriate path:

        - **In-process elicitation resume**: If ``elicitation_payloads`` are
          provided and the elicitation future still exists in the
          ``ElicitationFutureRegistry`` (agent run is still alive), the future
          is resolved directly.
        - **Crash recovery elicitation resume**: If ``elicitation_payloads``
          are provided but the future does NOT exist, the elicitation responses
          are pre-populated into ``AgentRunContext.cached_elicitation_responses``
          and the agent run is re-executed.
        - **Non-elicitation resume**: Native agent: load checkpoint →
          reconstruct agent → ``agent.run_stream()`` with deferred results.

        Args:
            session_id: Session to resume.
            deferred_tool_results: Results for pending deferred tool calls.
            source: Identifier for the entity triggering the resume.
            elicitation_payloads: Optional elicitation responses for resuming
                deferred elicitation calls.

        Raises:
            SessionNotFoundError: If the session does not exist in storage.
            SessionBusyError: If the session has an active run.
            CheckpointMismatchError: If results don't cover all pending calls.
        """
        store = self.sessions.store
        if store is None:
            raise SessionNotFoundError(session_id)

        # Load persisted session data
        data = await store.load_session(session_id)
        if data is None:
            raise SessionNotFoundError(session_id)

        # Separate elicitation and non-elicitation pending call IDs.
        # Must be done before the SessionBusyError check so we can detect
        # in-process elicitation resume (where the agent run is still alive).
        elicitation_call_ids: set[str] = {
            call.tool_call_id
            for call in data.pending_deferred_calls
            if call.deferred_kind == "elicitation"
        }
        non_elicitation_pending_ids: set[str] = {
            call.tool_call_id for call in data.pending_deferred_calls
        } - elicitation_call_ids

        # Fast-path: check for active run in live sessions (before lock).
        # EXCEPTION: in-process elicitation resume — the agent run is
        # intentionally still alive, paused on elicitation futures.
        session = self.sessions.get_session(session_id)
        has_in_process_elicitation = False
        if session is not None and session.current_run_id is not None:
            if elicitation_payloads is not None:
                run_handle = self._get_active_run_handle(session_id)
                if (
                    run_handle is not None
                    and run_handle.run_ctx is not None
                    and run_handle.run_ctx.elicitation_registry is not None
                    and len(run_handle.run_ctx.elicitation_registry) > 0
                    and all(
                        p.deferred_handle in run_handle.run_ctx.elicitation_registry
                        for p in elicitation_payloads
                    )
                ):
                    has_in_process_elicitation = True

            if not has_in_process_elicitation:
                raise SessionBusyError(session_id, session.current_run_id)

        provided_call_ids: set[str] = set(getattr(deferred_tool_results, "calls", {}).keys())

        missing = non_elicitation_pending_ids - provided_call_ids
        extra = provided_call_ids - non_elicitation_pending_ids - elicitation_call_ids
        if missing or extra:
            raise CheckpointMismatchError(
                session_id=session_id,
                expected=non_elicitation_pending_ids,
                provided=provided_call_ids,
                missing=missing,
                extra=extra,
            )

        # Validate elicitation_payloads cover all elicitation deferred calls.
        if elicitation_call_ids:
            provided_elicitation_ids: set[str] = {
                p.deferred_handle for p in (elicitation_payloads or [])
            }
            missing_elicitation = elicitation_call_ids - provided_elicitation_ids
            if missing_elicitation:
                raise CheckpointMismatchError(
                    session_id=session_id,
                    expected=elicitation_call_ids,
                    provided=provided_elicitation_ids,
                    missing=missing_elicitation,
                    extra=set(),
                )

        # Determine agent type
        agent_type = data.metadata.get("agent_type", "native")

        # Per-session resume lock with state validation.
        # For in-process elicitation resume, allow the active run to persist.
        async with self._with_resume_lock(
            session_id, allow_active_run=has_in_process_elicitation
        ) as session:
            try:
                # In-process elicitation resume: resolve futures so the
                # suspended agent run can continue naturally.
                if elicitation_payloads:
                    resolved = await self._try_in_process_elicitation_resume(
                        session_id, elicitation_payloads
                    )
                    if resolved:
                        # All futures resolved — agent run will continue.
                        data = data.model_copy(
                            update={
                                "status": "active",
                                "pending_deferred_calls": [],
                            }
                        )
                        data.touch()
                        await store.save_session(data)

                        if session is not None:
                            session.last_active_at = time.monotonic()
                            session.last_active_at_ns = time.time_ns()

                        total_resolved = len(elicitation_call_ids)
                        await self.event_bus.publish(
                            session_id,
                            SessionResumeEvent(
                                session_id=session_id,
                                resolved_call_count=total_resolved,
                                source=source,
                            ),
                        )

                        logger.info(
                            "In-process elicitation resume — futures resolved",
                            session_id=session_id,
                            count=total_resolved,
                        )
                        return

                    # In-process resolution failed (race condition).
                    # If the run is still active, we cannot start crash
                    # recovery — that would create a concurrent run.
                    if session is not None and session.current_run_id is not None:
                        raise SessionBusyError(session_id, session.current_run_id)  # noqa: TRY301

                # Load checkpoint data
                checkpoint = await self._load_checkpoint_data(session_id)

                # Mark session as resuming
                data = data.model_copy(update={"status": "resuming"})
                await store.save_session(data)

                # Route to appropriate resume path
                if agent_type == "acp":
                    await self._resume_acp_agent(data, checkpoint, deferred_tool_results)
                else:
                    await self._resume_native_agent(
                        data,
                        checkpoint,
                        deferred_tool_results,
                        elicitation_payloads=elicitation_payloads,
                    )

                # Clear pending_deferred_calls ONLY after agent.run() succeeds
                data = data.model_copy(
                    update={
                        "status": "active",
                        "pending_deferred_calls": [],
                    }
                )
                data.touch()
                await store.save_session(data)

                # Update live session if one exists
                if session is not None:
                    session.last_active_at = time.monotonic()
                    session.last_active_at_ns = time.time_ns()

                # Emit SessionResumeEvent
                total_resolved = len(non_elicitation_pending_ids) + len(elicitation_call_ids)
                await self.event_bus.publish(
                    session_id,
                    SessionResumeEvent(
                        session_id=session_id,
                        resolved_call_count=total_resolved,
                        source=source,
                    ),
                )

                logger.info(
                    "Session resumed successfully",
                    session_id=session_id,
                    agent_type=agent_type,
                    resolved_calls=total_resolved,
                )

            except Exception:
                # On failure, keep status as checkpointed and do NOT clear pending calls
                data = data.model_copy(update={"status": "checkpointed"})
                data.touch()
                await store.save_session(data)
                raise

    async def close_session(self, session_id: str) -> None:
        """Close a session.

        Delegates to :meth:`SessionController.close_session()` which
        implements the standardized 7-step cleanup ordering. If the
        delegate does not respond within 15 seconds, falls back to
        direct RunHandle cancellation and retries.

        SessionPool-level cleanup (message cache, EventBus safety net)
        is performed in the ``finally`` block regardless of outcome.

        Args:
            session_id: The session to close.
        """
        try:
            async with asyncio.timeout(15):
                await self.sessions.close_session(session_id)
        except TimeoutError:
            logger.warning(
                "SessionController.close_session() timed out (15s), "
                "falling back to direct RunHandle cancellation",
                session_id=session_id,
            )
            # Fallback: directly cancel RunHandle
            session = self.sessions.get_session(session_id)
            if session is not None and session.current_run_id is not None:
                run_handle = self.sessions._runs.get(session.current_run_id)
                if run_handle is not None:
                    run_handle.cancel()
            # Retry close after cancellation
            try:
                await self.sessions.close_session(session_id)
            except Exception:
                logger.exception(
                    "Fallback close also failed",
                    session_id=session_id,
                )
        finally:
            # SessionPool-level cleanup (not handled by SessionController)
            self._message_cache.pop(session_id, None)
            # EventBus cleanup as safety net (idempotent if already
            # done by _close_session_unlocked step 6)
            try:
                await self.event_bus.close_session(session_id)
            except asyncio.CancelledError:
                logger.warning(
                    "EventBus close_session interrupted by spurious cancellation",
                    session_id=session_id,
                )
            except Exception:
                logger.exception(
                    "Failed to close event bus session",
                    session_id=session_id,
                )

    async def _await_inflight_checkpoints(self) -> None:
        """Wait for any in-flight checkpoint operations to complete.

        During normal operation, checkpoint-on-close happens synchronously
        inside :meth:`close_session`, so there are no in-flight operations
        to await. This method is a future-proof hook for graceful teardown:
        if the checkpoint mechanism ever becomes asynchronous (e.g.,
        background flush), this method ensures the shutdown waits for
        completion.

        Called from :meth:`AgentPool.__aexit__` during pool shutdown.
        """
        # Currently no-op: all checkpoint operations complete synchronously
        # within SessionController.close_session() under its lock.
        logger.debug("No in-flight checkpoint operations to await")

    def _evict_message_cache(self) -> None:
        """Evict least-recently-used entries from ``_message_cache``.

        Evicts entries for **inactive** sessions only (sessions with no
        active run). Active sessions' messages are never evicted.
        Called after each cache insertion to enforce ``maxsize``.
        """
        if len(self._message_cache) <= self._message_cache_maxsize:
            return
        # Single-pass: collect all evictable session IDs (inactive or
        # non-existent sessions), then pop in bulk.
        candidates = [
            sid
            for sid in list(self._message_cache.keys())
            if (s := self.sessions.get_session(sid)) is None or s.current_run_id is None
        ]
        for sid in candidates:
            self._message_cache.pop(sid, None)
            if len(self._message_cache) <= self._message_cache_maxsize:
                return
        if len(self._message_cache) > self._message_cache_maxsize:
            # All cached sessions are active — cannot evict inactive ones.
            logger.warning(
                "Message cache at maxsize but all sessions are active",
                cache_size=len(self._message_cache),
                maxsize=self._message_cache_maxsize,
            )
