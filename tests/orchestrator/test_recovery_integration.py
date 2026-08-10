"""L2 integration tests for crash recovery via Journal.resume().

Tests the real integration between Journal, SnapshotStore, and the
ResumeResult contract — using properly wired instances (not standalone
disconnected instances like the removed VCR tests).

Crash recovery is lifecycle-internal and does not depend on model API
responses, so these are L2 integration tests (not L3 VCR tests).

See: https://github.com/Leoyzen/wolfharness/issues/205
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from wolfharness.lifecycle import (
    DurableJournal,
    MemoryJournal,
    MemorySnapshotStore,
    ResumeResult,
)


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _MockEvent:
    """Simple event with turn_id for testing in-flight detection."""

    turn_id: str | None = None
    payload: str = ""


# ---------------------------------------------------------------------------
# Fresh start: no snapshot → resume() returns None
# ---------------------------------------------------------------------------


def test_fresh_start_resume_returns_none():
    """Empty journal + empty snapshot store → resume() returns None.

    This is the Protocol contract: None means 'fresh start, no prior state'.
    The caller (RunHandle._handle_recovery) handles this by saving an
    initial snapshot.
    """
    journal = MemoryJournal()
    snapshot_store = MemorySnapshotStore()
    result = journal.resume(snapshot_store)
    assert result is None


# ---------------------------------------------------------------------------
# Recovery after completed turn: snapshot exists, no in-flight turn
# ---------------------------------------------------------------------------


def test_resume_after_completed_turn_returns_resume_result():
    """After a turn completes, snapshot exists → ResumeResult(is_inflight=False).

    Simulates the post-turn state: events journaled during the turn,
    turn_result saved (turn completed successfully), snapshot saved
    capturing state after turn completion.
    """
    journal = MemoryJournal()
    snapshot_store = MemorySnapshotStore()

    # Simulate events from a completed turn
    journal.append(_MockEvent(turn_id="turn_001", payload="run_started"))  # jseq=1
    journal.append(_MockEvent(turn_id="turn_001", payload="part_delta"))  # jseq=2

    # Simulate turn completion: save turn_result, then snapshot
    snapshot_store.save_turn_result("turn_001", {"content": "hello"})
    snapshot_store.save({"state": "idle", "run_id": "r1", "last_turn": "turn_001"})  # sseq=1

    result = journal.resume(snapshot_store)

    assert isinstance(result, ResumeResult)
    assert result.is_inflight is False
    assert result.inflight_turn_id is None
    assert result.state is not None


# ---------------------------------------------------------------------------
# In-flight turn detection: snapshot exists, turn has no result
# ---------------------------------------------------------------------------


def test_resume_detects_inflight_turn():
    """Turn with events but no turn_result → ResumeResult(is_inflight=True).

    Simulates a crash mid-turn: snapshot was saved before the turn,
    events were journaled during the turn, but no turn_result was saved
    (crash happened before turn completion).
    """
    journal = MemoryJournal()
    snapshot_store = MemorySnapshotStore()

    # Snapshot from before the crash
    snapshot_store.save({"state": "running", "run_id": "r1"})  # sseq=1

    # Events from the interrupted turn (no turn_result saved)
    # First append (jseq=1) is not > sseq=1, so it's "before" snapshot.
    # Second append (jseq=2) is > sseq=1, so it's "since" snapshot.
    journal.append(_MockEvent(turn_id="turn_inflight", payload="run_started"))  # jseq=1
    journal.append(_MockEvent(turn_id="turn_inflight", payload="tool_call_start"))  # jseq=2

    result = journal.resume(snapshot_store)

    assert isinstance(result, ResumeResult)
    assert result.is_inflight is True
    assert result.inflight_turn_id == "turn_inflight"
    # Events since snapshot should include the interrupted turn's events
    assert len(result.events) >= 1


# ---------------------------------------------------------------------------
# Multiple turns: completed turn + in-flight turn
# ---------------------------------------------------------------------------


def test_resume_with_completed_then_inflight_turn():
    """Two turns: first completed, second in-flight → detects second as in-flight.

    Simulates: turn_001 completes (snapshot saved after), then turn_002
    starts but crashes before completion (no snapshot or turn_result saved).
    """
    journal = MemoryJournal()
    snapshot_store = MemorySnapshotStore()

    # First turn: completed
    journal.append(_MockEvent(turn_id="turn_001", payload="run_started"))  # jseq=1
    snapshot_store.save_turn_result("turn_001", {"content": "first response"})

    # Snapshot taken after first turn completion (sseq=1, == jseq of turn_001's last event)
    snapshot_store.save({"state": "idle", "run_id": "r1", "last_turn": "turn_001"})

    # Second turn: in-flight (crash before completion)
    journal.append(_MockEvent(turn_id="turn_002", payload="run_started"))  # jseq=2 > sseq=1
    journal.append(_MockEvent(turn_id="turn_002", payload="tool_call"))  # jseq=3 > sseq=1

    result = journal.resume(snapshot_store)

    assert isinstance(result, ResumeResult)
    assert result.is_inflight is True
    assert result.inflight_turn_id == "turn_002"


# ---------------------------------------------------------------------------
# Upsert events: entity-state events are included in recovery
# ---------------------------------------------------------------------------


def test_resume_includes_upsert_events():
    """Upsert events since snapshot are included in ResumeResult.events.

    Both append (delta) and upsert (entity-state) events that occur
    after the snapshot should appear in the recovery events list.
    """
    journal = MemoryJournal()
    snapshot_store = MemorySnapshotStore()

    snapshot_store.save({"state": "idle", "run_id": "r1"})  # sseq=1

    # First append (jseq=1) is not > sseq=1, so it's "before" snapshot.
    # Subsequent events (jseq >= 2) are "since" snapshot.
    journal.append(_MockEvent(turn_id="turn_001", payload="filler"))  # jseq=1
    journal.append(_MockEvent(turn_id="turn_001", payload="delta"))  # jseq=2
    journal.upsert(
        "tool_call:abc", _MockEvent(turn_id="turn_001", payload="state_update")
    )  # jseq=3

    result = journal.resume(snapshot_store)

    assert isinstance(result, ResumeResult)
    # Both the append (jseq=2) and upsert (jseq=3) events should be in recovery events
    assert len(result.events) >= 2


# ---------------------------------------------------------------------------
# DurableJournal: same recovery semantics
# ---------------------------------------------------------------------------


def test_durable_journal_resume_consistency(tmp_path: Any):
    """DurableJournal.resume() has the same None/ResumeResult contract as MemoryJournal.

    Verifies that the SQL-backed journal implementation produces the same
    recovery contract: None for fresh start, ResumeResult when state exists.
    """
    db_url = f"sqlite:///{tmp_path}/test_recovery.db"
    journal = DurableJournal(db_url, session_id="test")
    snapshot_store = MemorySnapshotStore()

    # Fresh start: no snapshot → None
    result = journal.resume(snapshot_store)
    assert result is None

    # With snapshot but no events → ResumeResult(is_inflight=False)
    snapshot_store.save({"state": "idle", "run_id": "r1"})
    result = journal.resume(snapshot_store)
    assert isinstance(result, ResumeResult)
    assert result.is_inflight is False

    journal.close()


# ---------------------------------------------------------------------------
# DurableJournal: in-flight turn detection with dataclass events
# ---------------------------------------------------------------------------


def test_durable_journal_resume_detects_inflight_turn(tmp_path: Any):
    """DurableJournal.resume() detects in-flight turns through JSON round-trip.

    Dataclass events are JSON-serialized on append and deserialized on
    resume. The turn_id field survives the round-trip (as a dict key),
    so _detect_inflight_turn can still find the in-flight turn.
    """
    db_url = f"sqlite:///{tmp_path}/test_recovery_inflight.db"
    journal = DurableJournal(db_url, session_id="test")
    snapshot_store = MemorySnapshotStore()

    # Snapshot before the crash
    snapshot_store.save({"state": "running", "run_id": "r1"})  # sseq=1

    # Events from interrupted turn (no turn_result saved)
    journal.append(_MockEvent(turn_id="turn_inflight", payload="run_started"))  # jseq=1
    journal.append(_MockEvent(turn_id="turn_inflight", payload="tool_call"))  # jseq=2

    result = journal.resume(snapshot_store)

    assert isinstance(result, ResumeResult)
    assert result.is_inflight is True
    assert result.inflight_turn_id == "turn_inflight"

    journal.close()
