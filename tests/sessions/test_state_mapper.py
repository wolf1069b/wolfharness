"""Unit tests for SessionStateMapper.

Tests all state transition paths including:
- Normal transitions (active↔IDLE, active↔RUNNING, closed↔DONE)
- checkpointed + no RunHandle (valid, do NOT reconcile)
- resuming + crash (reconcile to active)
- active + no RunHandle (valid)
- Unknown status (reconcile to active)
- Status mismatch with RunState (reconcile to expected status)
"""

from __future__ import annotations

import pytest

from wolfharness.lifecycle.types import RunState
from wolfharness.sessions.state_mapper import (
    SESSION_STATUS_ACTIVE,
    SESSION_STATUS_CHECKPOINTED,
    SESSION_STATUS_CLOSED,
    SESSION_STATUS_RESUMING,
    VALID_SESSION_STATUSES,
    InvariantResult,
    SessionStateMapper,
)


pytestmark = pytest.mark.unit


class TestStatusToExpectedRunState:
    """Tests for SessionStateMapper.status_to_expected_run_state()."""

    @pytest.mark.parametrize(
        ("status", "has_run_handle", "expected"),
        [
            (SESSION_STATUS_ACTIVE, True, RunState.IDLE),
            (SESSION_STATUS_ACTIVE, False, None),
            (SESSION_STATUS_RESUMING, True, RunState.IDLE),
            (SESSION_STATUS_RESUMING, False, None),
            (SESSION_STATUS_CHECKPOINTED, True, RunState.IDLE),
            (SESSION_STATUS_CHECKPOINTED, False, None),
            (SESSION_STATUS_CLOSED, True, RunState.DONE),
            (SESSION_STATUS_CLOSED, False, None),
            ("unknown", True, None),
            ("unknown", False, None),
        ],
    )
    def test_status_to_run_state_mapping(
        self,
        status: str,
        has_run_handle: bool,
        expected: RunState | None,
    ) -> None:
        result = SessionStateMapper.status_to_expected_run_state(status, has_run_handle)
        assert result == expected


class TestRunStateToExpectedStatus:
    """Tests for SessionStateMapper.run_state_to_expected_status()."""

    @pytest.mark.parametrize(
        ("run_state", "expected"),
        [
            (RunState.IDLE, SESSION_STATUS_ACTIVE),
            (RunState.RUNNING, SESSION_STATUS_ACTIVE),
            (RunState.DONE, SESSION_STATUS_CLOSED),
        ],
    )
    def test_run_state_to_status_mapping(
        self,
        run_state: RunState,
        expected: str,
    ) -> None:
        result = SessionStateMapper.run_state_to_expected_status(run_state)
        assert result == expected


class TestCheckInvariantNormal:
    """Tests for normal (valid) state combinations."""

    def test_active_idle_with_run_handle(self) -> None:
        """Active + IDLE + RunHandle = valid."""
        result = SessionStateMapper.check_invariant(
            SESSION_STATUS_ACTIVE,
            RunState.IDLE,
            has_run_handle=True,
        )
        assert result.is_valid is True
        assert result.action == "none"

    def test_active_running_with_run_handle(self) -> None:
        """Active + RUNNING + RunHandle = valid."""
        result = SessionStateMapper.check_invariant(
            SESSION_STATUS_ACTIVE,
            RunState.RUNNING,
            has_run_handle=True,
        )
        assert result.is_valid is True
        assert result.action == "none"

    def test_closed_done_with_run_handle(self) -> None:
        """Closed + DONE + RunHandle = valid."""
        result = SessionStateMapper.check_invariant(
            SESSION_STATUS_CLOSED,
            RunState.DONE,
            has_run_handle=True,
        )
        assert result.is_valid is True
        assert result.action == "none"

    def test_active_no_run_handle(self) -> None:
        """Active + no RunHandle = valid (idle session)."""
        result = SessionStateMapper.check_invariant(
            SESSION_STATUS_ACTIVE,
            None,
            has_run_handle=False,
        )
        assert result.is_valid is True
        assert result.action == "none"
        assert "idle" in result.message.lower()

    def test_closed_no_run_handle(self) -> None:
        """Closed + no RunHandle = valid (session was closed)."""
        result = SessionStateMapper.check_invariant(
            SESSION_STATUS_CLOSED,
            None,
            has_run_handle=False,
        )
        assert result.is_valid is True
        assert result.action == "none"


class TestCheckInvariantCarveOuts:
    """Tests for invariant checker carve-outs (tasks 1.5a, 1.5b, 1.5c)."""

    def test_checkpointed_no_run_handle_is_valid(self) -> None:
        """Carve-out (a): checkpointed + no RunHandle = valid, do NOT reconcile."""
        result = SessionStateMapper.check_invariant(
            SESSION_STATUS_CHECKPOINTED,
            None,
            has_run_handle=False,
        )
        assert result.is_valid is True
        assert result.action == "none"
        assert result.reconciled_status is None
        assert "post-checkpoint" in result.message.lower()

    def test_resuming_no_run_handle_reconciles_to_active(self) -> None:
        """Carve-out (b): resuming + no RunHandle = reconcile to active."""
        result = SessionStateMapper.check_invariant(
            SESSION_STATUS_RESUMING,
            None,
            has_run_handle=False,
        )
        assert result.is_valid is False
        assert result.action == "reconcile_status"
        assert result.reconciled_status == SESSION_STATUS_ACTIVE
        assert "crash" in result.message.lower()

    def test_active_no_run_handle_is_valid(self) -> None:
        """Carve-out (c): active + no RunHandle = valid (already valid)."""
        result = SessionStateMapper.check_invariant(
            SESSION_STATUS_ACTIVE,
            None,
            has_run_handle=False,
        )
        assert result.is_valid is True
        assert result.action == "none"

    def test_checkpointed_with_run_handle_idle_is_mismatch(self) -> None:
        """Checkpointed + RunHandle(IDLE) = mismatch, reconcile to active.

        When a RunHandle exists with IDLE state, the session status should
        have already been updated to 'resuming' or 'active'. 'checkpointed'
        with an active RunHandle is a stale status that needs reconciliation.
        """
        result = SessionStateMapper.check_invariant(
            SESSION_STATUS_CHECKPOINTED,
            RunState.IDLE,
            has_run_handle=True,
        )
        assert result.is_valid is False
        assert result.action == "reconcile_status"
        assert result.reconciled_status == SESSION_STATUS_ACTIVE


class TestCheckInvariantMismatches:
    """Tests for mismatched state combinations."""

    def test_closed_with_running_state_reconciles_to_active(self) -> None:
        """Closed + RUNNING = mismatch, reconcile to active (RunState is authoritative)."""
        result = SessionStateMapper.check_invariant(
            SESSION_STATUS_CLOSED,
            RunState.RUNNING,
            has_run_handle=True,
        )
        assert result.is_valid is False
        assert result.action == "reconcile_status"
        assert result.reconciled_status == SESSION_STATUS_ACTIVE

    def test_checkpointed_with_running_state_reconciles_to_active(self) -> None:
        """Checkpointed + RUNNING = mismatch, reconcile to active."""
        result = SessionStateMapper.check_invariant(
            SESSION_STATUS_CHECKPOINTED,
            RunState.RUNNING,
            has_run_handle=True,
        )
        assert result.is_valid is False
        assert result.action == "reconcile_status"
        assert result.reconciled_status == SESSION_STATUS_ACTIVE

    def test_resuming_with_running_state_reconciles_to_active(self) -> None:
        """Resuming + RUNNING = mismatch, reconcile to active."""
        result = SessionStateMapper.check_invariant(
            SESSION_STATUS_RESUMING,
            RunState.RUNNING,
            has_run_handle=True,
        )
        assert result.is_valid is False
        assert result.action == "reconcile_status"
        assert result.reconciled_status == SESSION_STATUS_ACTIVE

    def test_active_with_done_state_reconciles_to_closed(self) -> None:
        """Active + DONE = mismatch, reconcile to closed."""
        result = SessionStateMapper.check_invariant(
            SESSION_STATUS_ACTIVE,
            RunState.DONE,
            has_run_handle=True,
        )
        assert result.is_valid is False
        assert result.action == "reconcile_status"
        assert result.reconciled_status == SESSION_STATUS_CLOSED

    def test_unknown_status_reconciles_to_active(self) -> None:
        """Unknown status = reconcile to active."""
        result = SessionStateMapper.check_invariant(
            "unknown_status",
            None,
            has_run_handle=False,
        )
        assert result.is_valid is False
        assert result.action == "reconcile_status"
        assert result.reconciled_status == SESSION_STATUS_ACTIVE

    def test_has_run_handle_but_none_run_state_is_invalid(self) -> None:
        """has_run_handle=True + run_state=None = inconsistent."""
        result = SessionStateMapper.check_invariant(
            SESSION_STATUS_ACTIVE,
            None,
            has_run_handle=True,
        )
        assert result.is_valid is False
        assert result.action == "reconcile_run_state"


class TestValidSessionStatuses:
    """Tests for the VALID_SESSION_STATUSES constant."""

    def test_contains_all_expected_statuses(self) -> None:
        assert SESSION_STATUS_ACTIVE in VALID_SESSION_STATUSES
        assert SESSION_STATUS_CHECKPOINTED in VALID_SESSION_STATUSES
        assert SESSION_STATUS_RESUMING in VALID_SESSION_STATUSES
        assert SESSION_STATUS_CLOSED in VALID_SESSION_STATUSES

    def test_does_not_contain_removed_statuses(self) -> None:
        assert "completed" not in VALID_SESSION_STATUSES
        assert "failed" not in VALID_SESSION_STATUSES

    def test_is_frozenset(self) -> None:
        assert isinstance(VALID_SESSION_STATUSES, frozenset)


class TestInvariantResult:
    """Tests for the InvariantResult dataclass."""

    def test_default_fields(self) -> None:
        result = InvariantResult(is_valid=True, action="none")
        assert result.reconciled_status is None
        assert result.reconciled_run_state is None
        assert result.message == ""

    def test_frozen(self) -> None:
        result = InvariantResult(is_valid=True, action="none")
        with pytest.raises(AttributeError):
            result.is_valid = False  # type: ignore[misc]
