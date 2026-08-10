"""Integration test: simulate crash with DurableJournal + DurableSnapshotStore.

Verifies that the SessionStateMapper invariant checker correctly reconciles
session status after a crash recovery flow.

Scenario:
1. Session status is "resuming" (persisted before crash)
2. Crash occurs before RunHandle is created
3. On restart, journal.resume() loads the snapshot
4. Invariant checker detects "resuming + no RunHandle" and reconciles to "active"
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import pytest

from wolfharness.lifecycle.journal import DurableJournal
from wolfharness.lifecycle.snapshot_store import DurableSnapshotStore
from wolfharness.sessions.state_mapper import (
    SESSION_STATUS_ACTIVE,
    SESSION_STATUS_CHECKPOINTED,
    SESSION_STATUS_RESUMING,
    SessionStateMapper,
)


if TYPE_CHECKING:
    from wolfharness.lifecycle.types import RunState


pytestmark = pytest.mark.integration


@pytest.fixture
def temp_db_dir() -> Path:
    """Create a temporary directory for SQLite databases."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestCrashRecoveryInvariant:
    """Test invariant checker reconciles after crash recovery."""

    def test_resuming_crash_reconciles_to_active(
        self,
        temp_db_dir: Path,
    ) -> None:
        """Simulate: crash during resume, no RunHandle created.

        1. Save snapshot with status="resuming" (pre-crash state)
        2. Call journal.resume() — returns ResumeResult
        3. No RunHandle exists (crash before creation)
        4. Invariant checker detects "resuming + no RunHandle" → reconcile to "active"
        """
        session_id = "test-crash-resuming-001"
        journal_path = f"sqlite:///{temp_db_dir / 'journal.db'}"
        snapshot_path = str(temp_db_dir / "snapshots.db")

        # Create durable journal and snapshot store
        journal = DurableJournal(journal_path, session_id)
        snapshot_store = DurableSnapshotStore(snapshot_path, session_id)

        # Save a snapshot with status="resuming" (simulating pre-crash state)
        # The snapshot state is a JSON-serializable dict representing the
        # persisted session state at the time of checkpoint.
        snapshot_store.save({
            "session_id": session_id,
            "status": SESSION_STATUS_RESUMING,
            "run_state": None,  # No RunState was set yet
        })

        # Simulate crash: no more journal entries, no RunHandle created
        # On restart, call journal.resume()
        resume_result = journal.resume(snapshot_store)

        # Resume should return a result (snapshot exists)
        assert resume_result is not None
        assert resume_result.is_inflight is False  # No in-flight turn

        # After crash recovery, there is no RunHandle
        has_run_handle = False
        run_state: RunState | None = None

        # The recovered session status is "resuming"
        recovered_status = SESSION_STATUS_RESUMING

        # Invariant checker should detect "resuming + no RunHandle" = crash state
        result = SessionStateMapper.check_invariant(
            recovered_status,
            run_state,
            has_run_handle,
        )

        # Should be invalid and reconcile to "active"
        assert result.is_valid is False
        assert result.action == "reconcile_status"
        assert result.reconciled_status == SESSION_STATUS_ACTIVE
        assert "crash" in result.message.lower()

    def test_checkpointed_no_run_handle_is_valid_after_recovery(
        self,
        temp_db_dir: Path,
    ) -> None:
        """Simulate: checkpointed session, no RunHandle after recovery.

        This is a valid state — checkpoint was intentional, RunHandle was
        cleaned up. The invariant checker should NOT reconcile.
        """
        session_id = "test-crash-checkpointed-001"
        journal_path = f"sqlite:///{temp_db_dir / 'journal.db'}"
        snapshot_path = str(temp_db_dir / "snapshots.db")

        journal = DurableJournal(journal_path, session_id)
        snapshot_store = DurableSnapshotStore(snapshot_path, session_id)

        # Save a snapshot with status="checkpointed"
        snapshot_store.save({
            "session_id": session_id,
            "status": SESSION_STATUS_CHECKPOINTED,
            "run_state": None,
        })

        # Simulate recovery
        resume_result = journal.resume(snapshot_store)
        assert resume_result is not None

        # After recovery, there is no RunHandle (checkpointed sessions
        # have their RunHandle cleaned up)
        has_run_handle = False
        run_state: RunState | None = None
        recovered_status = SESSION_STATUS_CHECKPOINTED

        # Invariant checker should say this is VALID
        result = SessionStateMapper.check_invariant(
            recovered_status,
            run_state,
            has_run_handle,
        )

        assert result.is_valid is True
        assert result.action == "none"
        assert result.reconciled_status is None

    def test_active_no_run_handle_is_valid_after_recovery(
        self,
        temp_db_dir: Path,
    ) -> None:
        """Simulate: active session, no RunHandle after recovery.

        This is valid — the session is idle and can accept new prompts.
        """
        session_id = "test-crash-active-001"
        journal_path = f"sqlite:///{temp_db_dir / 'journal.db'}"
        snapshot_path = str(temp_db_dir / "snapshots.db")

        journal = DurableJournal(journal_path, session_id)
        snapshot_store = DurableSnapshotStore(snapshot_path, session_id)

        snapshot_store.save({
            "session_id": session_id,
            "status": SESSION_STATUS_ACTIVE,
            "run_state": None,
        })

        resume_result = journal.resume(snapshot_store)
        assert resume_result is not None

        has_run_handle = False
        run_state: RunState | None = None
        recovered_status = SESSION_STATUS_ACTIVE

        result = SessionStateMapper.check_invariant(
            recovered_status,
            run_state,
            has_run_handle,
        )

        assert result.is_valid is True
        assert result.action == "none"

    def test_full_reconciliation_flow_after_crash(
        self,
        temp_db_dir: Path,
    ) -> None:
        """Test the full reconciliation flow: crash → resume → reconcile.

        1. Session was "resuming" when crash occurred
        2. journal.resume() loads the snapshot
        3. Invariant checker detects mismatch
        4. Status is reconciled to "active"
        5. After reconciliation, invariant check passes
        """
        session_id = "test-full-reconc-001"
        journal_path = f"sqlite:///{temp_db_dir / 'journal.db'}"
        snapshot_path = str(temp_db_dir / "snapshots.db")

        journal = DurableJournal(journal_path, session_id)
        snapshot_store = DurableSnapshotStore(snapshot_path, session_id)

        # Pre-crash: session was resuming
        snapshot_store.save({
            "session_id": session_id,
            "status": SESSION_STATUS_RESUMING,
        })

        # Crash + restart: resume from snapshot
        resume_result = journal.resume(snapshot_store)
        assert resume_result is not None

        # Step 1: Check invariant with recovered status
        has_run_handle = False
        run_state: RunState | None = None
        recovered_status = SESSION_STATUS_RESUMING

        result = SessionStateMapper.check_invariant(
            recovered_status,
            run_state,
            has_run_handle,
        )

        # Step 2: Reconcile
        assert result.is_valid is False
        assert result.action == "reconcile_status"
        assert result.reconciled_status == SESSION_STATUS_ACTIVE

        # Step 3: After reconciliation, check invariant again
        reconciled_status = result.reconciled_status
        assert reconciled_status is not None

        result_after = SessionStateMapper.check_invariant(
            reconciled_status,
            run_state,
            has_run_handle,
        )

        # Step 4: Should now be valid
        assert result_after.is_valid is True
        assert result_after.action == "none"
