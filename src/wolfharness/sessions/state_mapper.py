"""Session state machine mapping and invariant checking.

This module provides the formal mapping between ``SessionData.status``
(persisted, survives crashes) and ``RunState`` (transient, governs the
in-memory RunLoop). The two state machines serve different purposes and
are kept separate to avoid write amplification on every RunState transition.

Mapping Table
-------------

+--------------------------------------+------------------+----------------------+----------+
| Scenario                             | SessionData.status| RunState             | Valid?   |
+======================================+==================+======================+==========+
| Idle, no active turn                 | ``active``       | ``IDLE``             | Yes      |
| Idle, no RunHandle                   | ``active``       | ``None`` (no handle) | Yes      |
| Turn executing                       | ``active``       | ``RUNNING``          | Yes      |
| Closed                               | ``closed``       | ``DONE``             | Yes      |
| Closed, no RunHandle                 | ``closed``       | ``None`` (no handle) | Yes      |
| Checkpointed (post-checkpoint)       | ``checkpointed`` | ``None`` (no handle) | Yes *    |
| Resuming (RunHandle being created)   | ``resuming``     | ``IDLE``             | Yes      |
| Resuming (crash before RunHandle)    | ``resuming``     | ``None`` (no handle) | No **    |
| Active + crash left no RunHandle     | ``active``       | ``None`` (no handle) | Yes ***  |
+--------------------------------------+------------------+----------------------+----------+

* ``checkpointed`` + no RunHandle is a **valid** post-checkpoint state.
  The RunHandle was intentionally cleaned up after checkpointing. Do NOT
  reconcile.

** ``resuming`` + no RunHandle means the resume process was interrupted
   by a crash before the RunHandle could be created. Reconcile
   ``SessionData.status`` to ``active`` so the session can accept new
   prompts.

*** ``active`` + no RunHandle is valid — the session is idle and can
    accept new prompts. No reconciliation needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from wolfharness.lifecycle.types import RunState


# Valid SessionData.status values (source of truth)
SESSION_STATUS_ACTIVE = "active"
SESSION_STATUS_CHECKPOINTED = "checkpointed"
SESSION_STATUS_RESUMING = "resuming"
SESSION_STATUS_CLOSED = "closed"

#: All valid ``SessionData.status`` values.
VALID_SESSION_STATUSES: frozenset[str] = frozenset({
    SESSION_STATUS_ACTIVE,
    SESSION_STATUS_CHECKPOINTED,
    SESSION_STATUS_RESUMING,
    SESSION_STATUS_CLOSED,
})


@dataclass(frozen=True, kw_only=True)
class InvariantResult:
    """Result of an invariant check between ``SessionData.status`` and ``RunState``.

    Attributes:
        is_valid: Whether the combination is valid.
        action: What to do if invalid. One of:
            - ``"none"`` — no action needed.
            - ``"reconcile_status"`` — update ``SessionData.status`` to
              ``reconciled_status``.
            - ``"reconcile_run_state"`` — update in-memory ``RunState`` to
              ``reconciled_run_state``.
        reconciled_status: New ``SessionData.status`` if action is
            ``"reconcile_status"``, otherwise ``None``.
        reconciled_run_state: New ``RunState`` if action is
            ``"reconcile_run_state"``, otherwise ``None``.
        message: Human-readable description of the result.
    """

    is_valid: bool
    action: str
    reconciled_status: str | None = None
    reconciled_run_state: RunState | None = None
    message: str = ""


class SessionStateMapper:
    """Maps between ``SessionData.status`` and ``RunState`` and checks invariants.

    This class is stateless — all methods are static. It exists as a namespace
    to group the mapping logic and make it testable.
    """

    @staticmethod
    def status_to_expected_run_state(
        status: str,
        has_run_handle: bool,
    ) -> RunState | None:
        """Convert ``SessionData.status`` to the expected ``RunState``.

        Args:
            status: The current ``SessionData.status`` string.
            has_run_handle: Whether a ``RunHandle`` exists for the session.

        Returns:
            The expected ``RunState``, or ``None`` if no meaningful
            ``RunState`` mapping exists (e.g., ``checkpointed`` with no
            ``RunHandle``, or ``closed`` with no ``RunHandle``).
        """
        from wolfharness.lifecycle.types import RunState

        if status == SESSION_STATUS_ACTIVE:
            # active maps to IDLE when idle, RUNNING when a turn is
            # executing. Without a RunHandle, active is equivalent
            # to IDLE.
            return RunState.IDLE if has_run_handle else None
        if status == SESSION_STATUS_RESUMING:
            # resuming maps to IDLE (RunHandle being created).
            # Without a RunHandle, this is a crash state.
            return RunState.IDLE if has_run_handle else None
        if status == SESSION_STATUS_CHECKPOINTED:
            # checkpointed has no RunHandle — it was cleaned up
            # after checkpointing. If a RunHandle exists, it should
            # be IDLE.
            return RunState.IDLE if has_run_handle else None
        if status == SESSION_STATUS_CLOSED:
            # closed maps to DONE. Without a RunHandle, None.
            return RunState.DONE if has_run_handle else None
        # Unknown status — return None, let invariant checker
        # handle the mismatch.
        return None

    @staticmethod
    def run_state_to_expected_status(run_state: RunState) -> str:
        """Convert ``RunState`` to the expected ``SessionData.status``.

        Args:
            run_state: The current ``RunState`` of the ``RunHandle``.

        Returns:
            The expected ``SessionData.status`` string.
        """
        from wolfharness.lifecycle.types import RunState

        if run_state is RunState.IDLE:
            return SESSION_STATUS_ACTIVE
        if run_state is RunState.RUNNING:
            return SESSION_STATUS_ACTIVE
        if run_state is RunState.DONE:
            return SESSION_STATUS_CLOSED
        # Should not reach here with a valid RunState
        return SESSION_STATUS_ACTIVE

    @staticmethod
    def check_invariant(  # noqa: PLR0911
        status: str,
        run_state: RunState | None,
        has_run_handle: bool,
    ) -> InvariantResult:
        """Check consistency between ``SessionData.status`` and ``RunState``.

        This should be called at Turn boundaries (snapshot save points)
        where the system is in a stable state, not during transitions.

        Args:
            status: The current ``SessionData.status`` string.
            run_state: The current ``RunState`` of the ``RunHandle``,
                or ``None`` if no ``RunHandle`` exists.
            has_run_handle: Whether a ``RunHandle`` exists for the session.

        Returns:
            ``InvariantResult`` describing whether the combination is
            valid and what action (if any) to take.
        """
        # --- Carve-out (a): checkpointed + no RunHandle = valid ---
        if status == SESSION_STATUS_CHECKPOINTED and not has_run_handle:
            return InvariantResult(
                is_valid=True,
                action="none",
                message=(
                    "checkpointed + no RunHandle is a valid post-checkpoint "
                    "state — no reconciliation needed"
                ),
            )

        # --- Carve-out (b): resuming + no RunHandle = reconcile to active ---
        if status == SESSION_STATUS_RESUMING and not has_run_handle:
            return InvariantResult(
                is_valid=False,
                action="reconcile_status",
                reconciled_status=SESSION_STATUS_ACTIVE,
                message=(
                    "resuming + no RunHandle indicates crash before RunHandle "
                    "creation — reconciling status to 'active'"
                ),
            )

        # --- Carve-out (c): active + no RunHandle = valid (idle session) ---
        if status == SESSION_STATUS_ACTIVE and not has_run_handle:
            return InvariantResult(
                is_valid=True,
                action="none",
                message=(
                    "active + no RunHandle is valid — session is idle and can accept new prompts"
                ),
            )

        # --- closed + no RunHandle = valid (session was closed) ---
        if status == SESSION_STATUS_CLOSED and not has_run_handle:
            return InvariantResult(
                is_valid=True,
                action="none",
                message=("closed + no RunHandle is valid — session was closed"),
            )

        # --- Unknown status ---
        if status not in VALID_SESSION_STATUSES:
            return InvariantResult(
                is_valid=False,
                action="reconcile_status",
                reconciled_status=SESSION_STATUS_ACTIVE,
                message=(f"unknown status '{status}' — reconciling to 'active'"),
            )

        # --- Both status and run_state exist: check consistency ---
        if has_run_handle and run_state is not None:
            expected_status = SessionStateMapper.run_state_to_expected_status(run_state)
            if status == expected_status:
                return InvariantResult(
                    is_valid=True,
                    action="none",
                    message=f"status '{status}' is consistent with RunState.{run_state.name}",
                )
            # Mismatch: RunState is authoritative for transient state
            return InvariantResult(
                is_valid=False,
                action="reconcile_status",
                reconciled_status=expected_status,
                message=(
                    f"status '{status}' does not match expected '{expected_status}' "
                    f"for RunState.{run_state.name} — reconciling status"
                ),
            )

        # --- has_run_handle but run_state is None (shouldn't happen) ---
        if has_run_handle and run_state is None:
            return InvariantResult(
                is_valid=False,
                action="reconcile_run_state",
                reconciled_run_state=None,  # caller should set to IDLE
                message=("has_run_handle is True but run_state is None — inconsistent state"),
            )

        # Fallback: should not reach here
        return InvariantResult(
            is_valid=True,
            action="none",
            message="no invariant violation detected",
        )


__all__ = [
    "SESSION_STATUS_ACTIVE",
    "SESSION_STATUS_CHECKPOINTED",
    "SESSION_STATUS_CLOSED",
    "SESSION_STATUS_RESUMING",
    "VALID_SESSION_STATUSES",
    "InvariantResult",
    "SessionStateMapper",
]
