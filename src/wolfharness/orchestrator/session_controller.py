"""Session controller for per-session agent lifecycle management.

Extracted from orchestrator/core.py as part of the thin-wrapper refactor.
Manages session creation, run tracking, and agent resolution.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING, Any, ClassVar, Final

import anyio

from wolfharness.log import get_logger
from wolfharness.orchestrator.runtime_registry import RuntimeAgentRegistry
from wolfharness.utils.time_utils import get_now


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from datetime import datetime

    from wolfharness.delegation import AgentPool
    from wolfharness.host.context import HostContext
    from wolfharness.host.registry import AgentRegistry
    from wolfharness.lifecycle.protocols import (
        CommChannel,
        EventTransport,
        Journal,
        SnapshotStore,
        TriggerSource,
    )
    from wolfharness.lifecycle.types import Feedback, ResumeResult
    from wolfharness.models.pending_interaction import PendingPermission
    from wolfharness.orchestrator.event_bus import EventBus
    from wolfharness_storage.protocols import SessionPersistence


logger = get_logger(__name__)

DEFAULT_SESSION_TTL_SECONDS: Final[float] = 3600.0


class SessionNotFoundError(Exception):
    """Raised when a session cannot be found for resume."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session not found: {session_id}")
        self.session_id = session_id


class SessionBusyError(Exception):
    """Raised when trying to resume a session that has an active run."""

    def __init__(self, session_id: str, run_id: str) -> None:
        super().__init__(
            f"Session '{session_id}' already has an active run '{run_id}'. "
            "Wait for it to complete or cancel it first."
        )
        self.session_id = session_id
        self.run_id = run_id


class SessionClosedError(Exception):
    """Raised when rejecting pending elicitation futures on session close.

    This exception is set on pending ``asyncio.Future`` instances in
    ``ElicitationFutureRegistry`` when a session is closed, ensuring
    any suspended ``handle_elicitation()`` calls unblock immediately.

    Attributes:
        session_id: The session that was closed.
    """

    def __init__(self, session_id: str) -> None:
        """Initialize with the closed session ID.

        Args:
            session_id: The session that was closed.
        """
        self.session_id = session_id
        super().__init__(f"Session closed: {session_id}")


class CheckpointMismatchError(Exception):
    """Raised when deferred_tool_results don't cover all pending_deferred_calls."""

    def __init__(
        self,
        session_id: str,
        expected: set[str],
        provided: set[str],
        missing: set[str],
        extra: set[str],
    ) -> None:
        parts: list[str] = []
        if missing:
            parts.append(f"missing results for: {sorted(missing)}")
        if extra:
            parts.append(f"unexpected results for: {sorted(extra)}")
        msg = (
            f"Checkpoint mismatch for session '{session_id}': "
            + "; ".join(parts)
            + f". Expected tool_call_ids: {sorted(expected)}, provided: {sorted(provided)}."
        )
        super().__init__(msg)
        self.session_id = session_id
        self.expected = expected
        self.provided = provided
        self.missing = missing
        self.extra = extra


class SessionLifecyclePolicy:
    """Session lifecycle policy constants and helpers."""

    VALID: ClassVar[tuple[str, str, str]] = ("independent", "cascade", "bound")

    @classmethod
    def default(cls) -> str:
        return "cascade"

    @classmethod
    def is_valid(cls, policy: str) -> bool:
        return policy in cls.VALID


def _create_cancel_scope() -> anyio.CancelScope | None:
    """Create CancelScope if an event loop is running, else return None.

    Allows SessionState to be instantiated in synchronous contexts (e.g. tests)
    where no async event loop is available.
    """
    try:
        return anyio.CancelScope()
    except anyio.NoEventLoopError:
        return None


@dataclass
class SessionState:
    """Per-session state managed by the session pool.

    Attributes:
        session_id: Unique identifier for the session.
        agent_name: Name of the agent associated with this session.
        agent: The actual agent instance (shared or per-session).
        metadata: Arbitrary metadata attached to the session.
        created_at: Timestamp when the session was created.
        last_active_at: Timestamp of the most recent activity.
        closed_at: Timestamp when the session was closed, or None if active.
        is_per_session_agent: Whether the agent is dedicated to this session.
        turn_lock: Lock ensuring only one turn runs per session at a time.
        is_closing: Flag indicating the session is being closed.
    """

    session_id: str
    agent_name: str
    agent: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)
    last_active_at: float = field(default_factory=time.monotonic)
    created_at_ns: int = field(default_factory=time.time_ns)
    """Wall-clock creation timestamp (epoch nanoseconds, ``time.time_ns()``).

    Uses the same clock as session ID generation (``now_ms()`` →
    ``time.time_ns() // 1_000_000``), ensuring ``time.created`` order
    matches session ID lexicographic order.  Integer arithmetic avoids
    the float precision loss in the monotonic→epoch conversion.
    """
    last_active_at_ns: int = field(default_factory=time.time_ns)
    """Wall-clock last-activity timestamp (epoch nanoseconds, ``time.time_ns()``).

    Updated alongside ``last_active_at`` (monotonic) at every activity
    site.  Used for millisecond-precise ``time.updated`` in OpenCode
    session models via integer division (``// 1_000_000``).
    """
    created_at_wall: datetime = field(default_factory=get_now)
    """Wall-clock creation timestamp (UTC datetime) for persistence.

    ``created_at`` and ``last_active_at`` use ``time.monotonic()`` for
    elapsed-time calculations (idle detection, session age).  They must
    NOT be passed to ``datetime.fromtimestamp()`` — that function treats
    its argument as a Unix epoch timestamp, producing a 1970 datetime.
    This field stores the real wall-clock creation time so that
    ``_state_to_data()`` can persist a correct ``created_at``.
    """
    closed_at: float | None = None
    is_per_session_agent: bool = False
    turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    is_closing: bool = False
    parent_session_id: str | None = None
    lifecycle_policy: str = field(default_factory=SessionLifecyclePolicy.default)
    current_run_id: str | None = None
    cancel_scope: anyio.CancelScope | None = field(default_factory=_create_cancel_scope)
    _request_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _turn_owner_task: asyncio.Task[Any] | None = None
    input_provider: Any | None = None
    pending_questions: dict[str, Any] = field(default_factory=dict)
    """Pending questions stored on SessionState for per-session isolation."""
    checkpoint_enabled: bool = False
    """Whether durable elicitation/checkpointing is enabled for this session."""

    # ------------------------------------------------------------------
    # Lifecycle dimensions (per-prompt RunHandle migration)
    # ------------------------------------------------------------------
    # SessionState owns the 6 lifecycle dimensions previously held by
    # RunHandle. These persist across RunHandles for the session's
    # lifetime and are only closed during session close.
    _journal: Journal | None = None
    _snapshot_store: SnapshotStore | None = None
    _comm_channel: CommChannel | None = None
    _event_transport: EventTransport | None = None
    _trigger_source: TriggerSource | None = None
    _lifecycle_session_id: str = "default"
    _recover_strategy: str = "mark_interrupted"
    """Crash recovery strategy: ``"mark_interrupted"`` or ``"retry"``.

    Only active when both ``lifecycle.journal`` and ``lifecycle.snapshot``
    are durable.
    """
    _resume_result: ResumeResult | None = None
    """Result of crash recovery (``journal.resume()``), stored at session init.

    Set once in ``get_or_create_session_agent()`` when recovery runs.
    ``None`` for fresh starts or when no durable journal is configured.
    """
    _recovered_inflight_turn_id: str | None = None
    """Turn ID of the in-flight Turn detected during crash recovery.

    Set on SessionState during recovery in ``get_or_create_session_agent()``.
    Used by the ``"retry"`` strategy to check
    ``journal.get_tool_executions(turn_id)`` before re-executing.
    """

    # ------------------------------------------------------------------
    # HostContext injection (moved from RunHandle)
    # ------------------------------------------------------------------
    _host_context: HostContext | None = None
    """HostContext for constructing per-turn AgentContextDeps.

    When set, RunHandle constructs an ``AgentContextDeps`` per turn and
    injects it into ``run_ctx.deps`` so capabilities like
    ``SubagentCapability`` can access the delegation service.
    """
    _agent_registry: AgentRegistry | None = None
    """Read-only registry of compiled agents for delegation."""
    _event_bus: EventBus | None = None
    """EventBus reference for direct event publication from SessionState.

    Set by ``SessionController._initialize_lifecycle_and_recovery()``
    alongside the other lifecycle dimensions. Enables
    ``steer_from_background_task()`` to fire-and-forget
    ``UserMessageInsertedEvent`` without a controller reference.
    """
    _resume_deferred_tool_results: Any = None
    """Deferred tool results from checkpoint, forwarded to ``agent.create_turn()``
    via ``**pydantic_ai_kwargs`` during resume. Only set by
    ``_create_run_handle()`` when resuming from a checkpoint."""

    # ------------------------------------------------------------------
    # Queues for per-prompt message routing
    # ------------------------------------------------------------------
    prompt_queue: asyncio.Queue[str | list[Any]] = field(default_factory=asyncio.Queue)
    """Queue of followup prompts for the next RunHandle.

    When a RunHandle is active, followup messages are enqueued here.
    After the RunHandle terminates, ``_consume_run()`` drains this
    queue (holding ``_request_lock``) and creates a new RunHandle for
    each prompt in FIFO order.
    """
    feedback_queue: asyncio.Queue[Feedback] = field(default_factory=asyncio.Queue)
    """Queue of steer messages for delivery to the next RunHandle.

    When no RunHandle is active, steer messages are enqueued here.
    The next RunHandle drains this queue at turn start.
    """

    @property
    def closing(self) -> bool:
        """Alias for is_closing."""
        return self.is_closing

    @closing.setter
    def closing(self, value: bool) -> None:
        self.is_closing = value

    # ------------------------------------------------------------------
    # Per-prompt RunHandle: message routing helpers
    # ------------------------------------------------------------------

    def set_current_run_id(self, run_id: str | None) -> None:
        """Set ``current_run_id`` and publish a ``StateUpdate`` on transition.

        In the per-prompt model, ``RunState`` is eliminated. The state
        machine is expressed by ``current_run_id`` (None = idle,
        non-None = running). This helper publishes ``StateUpdate``
        events when ``current_run_id`` transitions, replacing the old
        ``RunHandle._transition()`` mechanism.

        Args:
            run_id: The new run ID, or ``None`` to mark idle.
        """
        old = self.current_run_id
        self.current_run_id = run_id
        if old == run_id:
            return
        # Publish StateUpdate via CommChannel when available.
        from wolfharness.agents.events import StateUpdate
        from wolfharness.lifecycle.types import RunState

        comm = self._comm_channel
        if comm is None:
            return
        new_state = RunState.RUNNING if run_id is not None else RunState.IDLE
        comm.on_state_change(new_state)
        state_event = StateUpdate(
            session_id=self._lifecycle_session_id,
            state=new_state,
            stop_reason=None,
        )
        with contextlib.suppress(Exception):
            asyncio.get_running_loop().create_task(comm.publish(state_event))

    def revoke(self, message_id: str) -> bool:
        """Revoke a queued steer message in ``feedback_queue`` by ID.

        In the per-prompt model, CommChannel's feedback queue is unused.
        Steer messages pending between turns live in
        ``SessionState.feedback_queue``. This method cancels any
        pending steer with the given ``message_id``.

        Args:
            message_id: The ID of the message to revoke.

        Returns:
            ``True`` if revoked or already gone (idempotent), ``False``
            if the message was not found in the queue.
        """
        # Drain and re-enqueue, skipping the target message_id.
        found = False
        remaining: list[Feedback] = []
        while not self.feedback_queue.empty():
            try:
                fb = self.feedback_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if fb.message_id == message_id:
                found = True
                # Do not re-enqueue — revoked.
            else:
                remaining.append(fb)
        for fb in remaining:
            self.feedback_queue.put_nowait(fb)
        return found


# Mixin imports placed after SessionState/exception definitions to avoid
# circular import issues (mixins need SessionState at runtime).
from wolfharness.orchestrator.session_controller_agent import (  # noqa: E402
    SessionControllerAgentMixin,
)
from wolfharness.orchestrator.session_controller_close import (  # noqa: E402
    SessionControllerCloseMixin,
)
from wolfharness.orchestrator.session_controller_runs import (  # noqa: E402
    SessionControllerRunsMixin,
)


class SessionController(
    SessionControllerAgentMixin,
    SessionControllerRunsMixin,
    SessionControllerCloseMixin,
):
    """Manages per-session agent lifecycle.

    Extracted from ACP's AgentPoolACPAgent._session_agents and
    OpenCode's ServerState._session_agents.

    Safety features:
    - Single global lock for session creation (no DCL)
    - Per-session turn lock for serialization
    - Explicit cleanup of all resources
    - Support for all agent types (with per-session agents for NativeAgentConfig only)
    """

    def __init__(
        self,
        pool: AgentPool[Any],
        store: SessionPersistence | None = None,
        cleanup_callback: Callable[[str], Awaitable[None]] | None = None,
        max_concurrent_runs: int | None = None,
        session_ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS,
        cleanup_interval_seconds: float | None = None,
        deferred_cleanup_interval_seconds: float = 60.0,
    ) -> None:
        """Initialize the session controller.

        Args:
            pool: The agent pool to resolve agents from.
            store: Optional session store for persistence.
            cleanup_callback: Optional callback invoked when a session is cleaned up.
            max_concurrent_runs: Maximum number of concurrent runs across all sessions.
            session_ttl_seconds: TTL for idle sessions in seconds.
            cleanup_interval_seconds: Interval for the session TTL cleanup
                loop. Defaults to ``session_ttl_seconds / 2``.
            deferred_cleanup_interval_seconds: Interval for the deferred
                call expiry cleanup loop in seconds.
        """
        self.pool = pool
        self.store = store
        self._cleanup_callback = cleanup_callback
        self._sessions: dict[str, SessionState] = {}
        self._session_agents: dict[str, Any] = {}
        self._children: dict[str, list[str]] = {}
        self._session_scopes: dict[str, anyio.CancelScope] = {}
        self._lock = asyncio.Lock()
        self._session_ttl_seconds: float = session_ttl_seconds
        self._cleanup_interval_seconds: float = (
            cleanup_interval_seconds
            if cleanup_interval_seconds is not None
            else session_ttl_seconds / 2
        )
        self._deferred_cleanup_interval_seconds: float = deferred_cleanup_interval_seconds
        self._cleanup_task: asyncio.Task[Any] | None = None
        self._deferred_cleanup_task: asyncio.Task[Any] | None = None
        self._mcp_max_processes: int = 100
        self._mcp_process_count: int = 0
        self._runs: dict[str, Any] = {}
        self._runs_lock: asyncio.Lock = asyncio.Lock()
        self._max_concurrent_runs: int | None = max_concurrent_runs
        self._event_bus: EventBus | None = None
        self._pending_run_ids: dict[str, str] = {}
        self._todo_lock: asyncio.Lock = asyncio.Lock()
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._runtime_registry = RuntimeAgentRegistry()

    @property
    def runtime_registry(self) -> RuntimeAgentRegistry:
        """Runtime agent registry for programmatically-created agents."""
        return self._runtime_registry

    def get_session(self, session_id: str) -> SessionState | None:
        """Get a session by ID.

        Args:
            session_id: The session ID to look up.

        Returns:
            The session state, or None if not found.
        """
        return self._sessions.get(session_id)

    def get_children(self, session_id: str) -> list[str]:
        """Get child session IDs for a session.

        Args:
            session_id: The parent session ID.

        Returns:
            List of child session IDs.
        """
        return list(self._children.get(session_id, []))

    def get_parent(self, session_id: str) -> SessionState | None:
        """Get the parent session state for a session.

        Args:
            session_id: The child session ID.

        Returns:
            The parent session state, or None if not found.
        """
        session = self._sessions.get(session_id)
        if session is None or session.parent_session_id is None:
            return None
        return self._sessions.get(session.parent_session_id)

    def find_sessions_by_agent_name(self, agent_name: str) -> list[SessionState]:
        """Find all active sessions associated with a given agent name.

        Args:
            agent_name: The agent name to search for.

        Returns:
            List of session states matching the agent name, excluding closing sessions.
        """
        return [
            s for s in self._sessions.values() if s.agent_name == agent_name and not s.is_closing
        ]

    def _count_mcp_processes(self) -> int:
        """Count active MCP processes across all per-session agents.

        Returns:
            The tracked MCP process count.
        """
        return self._mcp_process_count

    def _increment_mcp_count(self, _agent: Any) -> None:
        """Increment MCP process count when a per-session agent is created.

        Args:
            _agent: The agent whose creation triggered the increment.
        """
        self._mcp_process_count += 1

    def _decrement_mcp_count(self, _agent: Any) -> None:
        """Decrement MCP process count when a per-session agent is destroyed.

        Args:
            _agent: The agent whose destruction triggered the decrement.
        """
        self._mcp_process_count = max(0, self._mcp_process_count - 1)

    def list_pending_questions(self) -> list[Any]:
        """List all pending questions across sessions.

        Aggregates pending questions from each session's SessionState.

        Returns:
            A list of pending question objects.
        """
        result: list[Any] = []
        for session in self._sessions.values():
            result.extend(session.pending_questions.values())
        return result

    def cancel_all_pending_questions(self) -> list[str]:
        """Cancel all pending questions across all sessions.

        Iterates over every session, cancels each pending question's future,
        and returns the IDs of all cancelled questions.

        Returns:
            List of cancelled question IDs.
        """
        cancelled_ids: list[str] = []
        for session in self._sessions.values():
            for question_id, pending in list(session.pending_questions.items()):
                future = getattr(pending, "future", None)
                if future is not None and not future.done():
                    future.cancel()
                    cancelled_ids.append(question_id)
        return cancelled_ids

    def cancel_session_pending_questions(self, session_id: str) -> list[str]:
        """Cancel pending questions for a specific session.

        Args:
            session_id: The session whose pending questions should be cancelled.

        Returns:
            List of cancelled question IDs.
        """
        cancelled_ids: list[str] = []
        session = self._sessions.get(session_id)
        if session is None:
            return cancelled_ids
        for question_id, pending in list(session.pending_questions.items()):
            future = getattr(pending, "future", None)
            if future is not None and not future.done():
                future.cancel()
                cancelled_ids.append(question_id)
        return cancelled_ids

    def list_pending_permissions(self) -> list[PendingPermission]:
        """List all pending permissions across sessions.

        Returns:
            A list of pending permissions. Currently returns an empty list.
        """
        return []

    async def start_cleanup_task(self) -> None:
        """Start background tasks for session cleanup.

        Launches two background tasks:
        - ``_cleanup_loop``: periodically closes expired sessions (TTL-based).
        - ``_start_cleanup_loop``: periodically expires stale deferred calls.

        Both tasks are stored in ``_background_tasks`` to prevent garbage
        collection mid-execution (per asyncio.create_task best practice).
        """
        if self._cleanup_task is None:
            task = asyncio.create_task(self._cleanup_loop())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            self._cleanup_task = task
        if self._deferred_cleanup_task is None:
            task = asyncio.create_task(self._start_cleanup_loop())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            self._deferred_cleanup_task = task

    async def stop_cleanup_task(self) -> None:
        """Stop both background cleanup tasks."""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None
        if self._deferred_cleanup_task is not None:
            self._deferred_cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._deferred_cleanup_task
            self._deferred_cleanup_task = None

    async def _cleanup_loop(self) -> None:
        """Periodically scan and close expired sessions.

        Runs every ``cleanup_interval_seconds`` (default: 30 minutes).
        A session is expired if last_active_at is older than session_ttl_seconds.
        """
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval_seconds)
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Session cleanup failed")

    async def _cleanup_expired_sessions(self) -> None:
        """Close all sessions that have exceeded TTL.

        Sessions with an active run are never expired — the run itself
        is proof of activity regardless of ``last_active_at`` age.
        """
        now = time.monotonic()
        expired_sessions: list[str] = []

        async with self._lock:
            for session_id, session in list(self._sessions.items()):
                if session.current_run_id is not None:
                    continue
                if now - session.last_active_at > self._session_ttl_seconds:
                    expired_sessions.append(session_id)

        for session_id in expired_sessions:
            logger.info("Closing expired session", session_id=session_id)
            try:
                if self._cleanup_callback is not None:
                    await self._cleanup_callback(session_id)
                else:
                    await self.close_session(session_id)
            except Exception:
                logger.exception(
                    "Failed to close expired session during cleanup",
                    session_id=session_id,
                )

    async def _start_cleanup_loop(self) -> None:
        """Periodically scan and expire deferred calls whose timeout has elapsed.

        Runs indefinitely in a background task. Checks every
        ``deferred_cleanup_interval_seconds`` (default: 60 seconds)
        for pending deferred calls whose timeout has elapsed and removes
        them from the session data.
        """
        while True:
            try:
                await asyncio.sleep(self._deferred_cleanup_interval_seconds)
                if self.store is None:
                    continue
                async with self._lock:
                    for session_id in list(self._sessions.keys()):
                        data = await self.store.load_session(session_id)
                        if data is None:
                            continue
                        expired = self._check_expired_calls(data)
                        if expired:
                            remaining = [
                                c
                                for c in data.pending_deferred_calls
                                if c.tool_call_id not in {e.tool_call_id for e in expired}
                            ]
                            updated = data.model_copy(update={"pending_deferred_calls": remaining})
                            await self.store.save_session(updated)
                            logger.info(
                                "Removed expired deferred calls",
                                session_id=session_id,
                                count=len(expired),
                            )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Deferred call cleanup loop failed")
