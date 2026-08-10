"""Close lifecycle mixin for SessionController.

Extracted from session_controller.py as part of the session-debt-cleanup file split.
Contains session close, checkpoint-on-close, and cleanup methods.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import time
from typing import TYPE_CHECKING, Any

from wolfharness.log import get_logger


if TYPE_CHECKING:
    import anyio

    from wolfharness.delegation.pool import AgentPool
    from wolfharness.orchestrator.event_bus import EventBus
    from wolfharness.orchestrator.run import RunHandle
    from wolfharness.orchestrator.session_controller import SessionState
    from wolfharness.sessions.models import PendingDeferredCall, SessionData
    from wolfharness_storage.protocols import SessionPersistence


logger = get_logger(__name__)


class SessionControllerCloseMixin:
    """Mixin providing session close and checkpoint methods for SessionController.

    Attributes:
        store: Session persistence store (provided by SessionController).
        _sessions: Active sessions dict (provided by SessionController).
        _session_agents: Per-session agent cache (provided by SessionController).
        _children: Parent→children mapping (provided by SessionController).
        _session_scopes: Session cancel scopes (provided by SessionController).
        _lock: Global lock (provided by SessionController).
        _runs: Active run handles (provided by SessionController).
    """

    store: SessionPersistence | None
    pool: AgentPool[Any]
    _sessions: dict[str, SessionState]
    _session_agents: dict[str, Any]
    _children: dict[str, list[str]]
    _session_scopes: dict[str, anyio.CancelScope]
    _lock: asyncio.Lock
    _runs: dict[str, RunHandle]
    _event_bus: EventBus | None

    def _decrement_mcp_count(self, _agent: Any) -> None: ...

    async def _close_session_unlocked(  # noqa: PLR0915
        self, session_id: str, *, checkpointed: bool = False
    ) -> None:
        """Close a session with standardized 7-step cleanup ordering.

        Caller must hold ``self._lock``.

        Cleanup ordering (each step logs errors but continues):

        1. Cancel RunHandle with 10s timeout
        2. Await RunHandle completion
        3. MCP cleanup (``agent.mcp.cleanup_session``)
        4. Agent ``__aexit__``
        4b. Close lifecycle dimensions (Journal, CommChannel, EventTransport,
            TriggerSource) — per-prompt migration, task 3.11
        5. Session persistence (save final state as ``"closed"``)
           — skipped when ``checkpointed=True`` (status already saved
           as ``"checkpointed"`` by ``_save_close_checkpoint``)
        6. EventBus unsubscription
        6b. ExtensionRegistry session/turn cap cleanup
        7. Cascade close children (respecting lifecycle policies)

        Args:
            session_id: The session to close.
            checkpointed: When True, the session was already saved as
                ``"checkpointed"`` by ``_save_close_checkpoint`` and step 5
                must NOT overwrite the status with ``"closed"``.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return
        # Acquire _request_lock to prevent concurrent routing (send_message
        # checks session.closing inside _request_lock).  Lock ordering is
        # self._lock → session._request_lock — no other path reverses this.
        async with session._request_lock:
            session.is_closing = True
            session.closed_at = time.monotonic()

        # Step 1: Cancel RunHandle with 10s timeout
        run_handle: RunHandle | None = None
        if session.current_run_id is not None:
            run_handle = self._runs.get(session.current_run_id)
        if run_handle is not None:
            try:
                run_handle.close()
                if run_handle.run_ctx is not None:
                    run_handle.run_ctx.cancelled = True
                    for ev in list(run_handle.run_ctx.child_done_events.values()):
                        ev.set()
                    run_handle.run_ctx.child_done_events.clear()
                    # Reject all pending elicitation futures so blocked
                    # MCP tool calls unblock immediately.
                    registry = run_handle.run_ctx.elicitation_registry
                    if registry is not None:
                        from wolfharness.orchestrator.session_controller import SessionClosedError

                        registry.reject_all(SessionClosedError(session_id))
                # Step 2: Await RunHandle completion (10s timeout)
                try:
                    async with asyncio.timeout(10):
                        await run_handle.complete_event.wait()
                except TimeoutError:
                    logger.warning(
                        "Timeout waiting for RunHandle completion, cancelling",
                        session_id=session_id,
                    )
                    run_handle.cancel()
            except Exception:
                logger.exception(
                    "Failed to cancel RunHandle during close",
                    session_id=session_id,
                )
            finally:
                self._runs.pop(run_handle.run_id, None)
                if session.current_run_id == run_handle.run_id:
                    session.current_run_id = None

        # Step 3: MCP cleanup
        agent = self._session_agents.get(session_id)
        if agent is not None:
            try:
                await agent.mcp.cleanup_session(session_id)
            except Exception:
                logger.exception("Failed to cleanup MCP session", session_id=session_id)

        # Step 4: Agent __aexit__
        if agent is not None and session.is_per_session_agent:
            try:
                await agent.__aexit__(None, None, None)
            except Exception:
                logger.exception("Failed to exit agent context", session_id=session_id)
            finally:
                self._decrement_mcp_count(agent)

        # Step 4b: Close lifecycle dimensions (per-prompt migration, task 3.11).
        # These were previously closed in RunHandle.close() but are now
        # session-owned and must be closed here.
        import contextlib as ctxlib

        with ctxlib.suppress(Exception):
            if session._trigger_source is not None:
                session._trigger_source.close()
        with ctxlib.suppress(Exception):
            if session._comm_channel is not None:
                session._comm_channel.close()
        with ctxlib.suppress(Exception):
            if session._event_transport is not None:
                session._event_transport.close()

        # Step 5: Session persistence (save final state)
        # Skip when checkpointed=True — _save_close_checkpoint already
        # saved the session with status="checkpointed", and calling
        # _mark_session_closed() would overwrite it with "closed".
        if self.store is not None and not checkpointed:
            try:
                await self._mark_session_closed(session_id)
            except Exception:
                logger.exception(
                    "Failed to mark session as closed in store",
                    session_id=session_id,
                )

        # Step 6: EventBus unsubscription
        if self._event_bus is not None:
            try:
                await self._event_bus.close_session(session_id)
            except Exception:
                logger.exception(
                    "Failed to unsubscribe from EventBus during close",
                    session_id=session_id,
                )

        # Step 6b: Clear SESSION and TURN level caps from the ExtensionRegistry.
        # AGENT and POOL level caps are NOT affected (they outlive sessions).
        try:
            self.pool.extension_registry.clear_session(session_id)
        except Exception:
            logger.exception(
                "Failed to clear ExtensionRegistry session caps",
                session_id=session_id,
            )

        # Step 7: Cascade close children (respecting lifecycle policies)
        children = self._children.pop(session_id, [])
        for child_id in children:
            child_session = self._sessions.get(child_id)
            if child_session is not None and child_session.lifecycle_policy == "independent":
                continue
            try:
                await self._close_session_unlocked(child_id)
            except Exception:
                logger.exception(
                    "Failed to close child session during cascade close",
                    child_id=child_id,
                )

        # Final dict cleanup
        self._session_agents.pop(session_id, None)
        self._sessions.pop(session_id, None)

        # Remove from parent's children list
        if session.parent_session_id and session.parent_session_id in self._children:
            self._children[session.parent_session_id] = [
                cid for cid in self._children[session.parent_session_id] if cid != session_id
            ]

        logger.info("Closed session (unlocked path)", session_id=session_id)

    @staticmethod
    def _should_checkpoint_on_close(data: SessionData | None) -> bool:
        """Check whether a session should be checkpointed before close.

        A session needs checkpoint-on-close when it has pending deferred calls
        that must be preserved for later resume.

        Args:
            data: The session data loaded from the store, or None.

        Returns:
            True if the session has pending deferred calls that require
            checkpointing before releasing resources.
        """
        return data is not None and bool(data.pending_deferred_calls)

    @staticmethod
    def _check_expired_calls(session_data: SessionData) -> list[PendingDeferredCall]:
        """Return pending calls whose timeout has elapsed.

        Args:
            session_data: The session data to check for expired calls.

        Returns:
            A list of ``PendingDeferredCall`` entries whose timeout has
            elapsed. Returns an empty list if none have expired.
        """
        now = datetime.now()
        return [
            call
            for call in session_data.pending_deferred_calls
            if call.timeout is not None and (now - call.created_at) > call.timeout
        ]

    async def _save_close_checkpoint(self, session_id: str, data: SessionData) -> bool:
        """Save session data with checkpointed status before close.

        Marks the session as ``"checkpointed"`` so it can be located by
        :meth:`resume_session` later. Returns ``True`` on success, ``False``
        if the storage write fails (caller should NOT release resources).

        Args:
            session_id: Session identifier (for logging).
            data: The session data to persist as checkpointed.

        Returns:
            True if the checkpoint was saved successfully, False on failure.
        """
        try:
            data = data.model_copy(update={"status": "checkpointed"})
            data.touch()
            if self.store is not None:
                await self.store.save_session(data)
            logger.info(
                "Session checkpointed before close",
                session_id=session_id,
                pending_call_count=len(data.pending_deferred_calls),
            )
        except Exception:
            logger.exception(
                "Failed to save checkpoint before close",
                session_id=session_id,
            )
            return False
        else:
            return True

    async def _mark_session_closed(self, session_id: str) -> None:
        """Mark a session as closed in the store instead of deleting it.

        This preserves session data across server restarts so that clients
        can resume sessions via ``session/resume`` or ``session/load`` after
        a server restart.

        Args:
            session_id: Session identifier to mark as closed.
        """
        assert self.store is not None
        data = await self.store.load_session(session_id)
        if data is None:
            logger.debug("Session not in store, skipping close mark", session_id=session_id)
            return
        data = data.model_copy(update={"status": "closed"})
        data.touch()
        await self.store.save_session(data)
        logger.debug("Session marked as closed in store", session_id=session_id)

    async def _close_session_run_turn(self, session_id: str) -> None:
        """Close a session using the standardized 7-step cleanup ordering.

        Performs checkpoint-on-close logic (if pending deferred calls exist)
        before delegating to :meth:`_close_session_unlocked` which implements
        the full 7-step cleanup ordering under the lock.

        Args:
            session_id: The session to close.
        """
        # Checkpoint-on-close: if pending deferred calls exist, save as
        # checkpointed before releasing resources. If checkpoint fails,
        # keep session in memory so it can be retried.
        _checkpointed = False
        if self.store is not None:
            _data = await self.store.load_session(session_id)
            if self._should_checkpoint_on_close(_data):
                assert _data is not None
                _checkpointed = await self._save_close_checkpoint(session_id, _data)
                if not _checkpointed:
                    logger.warning(
                        "Checkpoint failed, keeping session in memory",
                        session_id=session_id,
                    )
                    return

        async with self._lock:
            await self._close_session_unlocked(session_id, checkpointed=_checkpointed)

    async def close_session(self, session_id: str) -> None:
        """Close a session and clean up resources.

        Delegates to :meth:`_close_session_run_turn` which performs
        checkpoint-on-close (if needed) and then the standardized
        7-step cleanup ordering via :meth:`_close_session_unlocked`.

        Args:
            session_id: The session to close.
        """
        await self._close_session_run_turn(session_id)
