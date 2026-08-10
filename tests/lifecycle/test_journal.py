"""Tests for Journal dimension: MemoryJournal and DurableJournal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from wolfharness.lifecycle import (
    DurableJournal,
    Journal,
    MemoryJournal,
    MemorySnapshotStore,
    ToolExecutionRecord,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


@dataclass
class _MockEvent:
    """Simple event with turn_id for testing in-flight detection."""

    turn_id: str | None = None
    payload: str = ""


# ---------------------------------------------------------------------------
# MemoryJournal — Protocol conformance
# ---------------------------------------------------------------------------


def test_memory_journal_protocol_conformance():
    """MemoryJournal satisfies the Journal Protocol."""
    assert isinstance(MemoryJournal(), Journal)


# ---------------------------------------------------------------------------
# MemoryJournal — append
# ---------------------------------------------------------------------------


def test_memory_journal_append_returns_monotonic_seq():
    """append() returns strictly increasing sequence numbers."""
    journal = MemoryJournal()
    seq1 = journal.append("event1")
    seq2 = journal.append("event2")
    seq3 = journal.append("event3")
    assert seq1 == 1
    assert seq2 == 2
    assert seq3 == 3


def test_memory_journal_append_starts_at_one():
    """First append returns seq=1."""
    journal = MemoryJournal()
    assert journal.append("first") == 1


# ---------------------------------------------------------------------------
# MemoryJournal — upsert
# ---------------------------------------------------------------------------


def test_memory_journal_upsert_creates_new_entry():
    """upsert() on a new key creates an entry and returns seq."""
    journal = MemoryJournal()
    seq = journal.upsert("key1", "value1")
    assert seq == 1


def test_memory_journal_upsert_replaces_existing():
    """upsert() on an existing key replaces the value and returns higher seq."""
    journal = MemoryJournal()
    seq1 = journal.upsert("key1", "value1")
    seq2 = journal.upsert("key1", "value2")
    assert seq2 > seq1


def test_memory_journal_upsert_different_keys_are_independent():
    """upsert() on different keys creates separate entries."""
    journal = MemoryJournal()
    seq1 = journal.upsert("key1", "value1")
    seq2 = journal.upsert("key2", "value2")
    assert seq1 == 1
    assert seq2 == 2


def test_memory_journal_append_and_upsert_share_seq():
    """append() and upsert() share the same monotonic counter."""
    journal = MemoryJournal()
    seq1 = journal.append("delta1")
    seq2 = journal.upsert("key1", "state1")
    seq3 = journal.append("delta2")
    assert seq1 == 1
    assert seq2 == 2
    assert seq3 == 3


# ---------------------------------------------------------------------------
# MemoryJournal — replay
# ---------------------------------------------------------------------------


async def test_memory_journal_replay_yields_in_seq_order() -> None:
    """replay() yields events in sequence order."""
    journal = MemoryJournal()
    journal.append("a")
    journal.append("b")
    journal.append("c")
    events = [event async for event in journal.replay()]
    assert events == ["a", "b", "c"]


async def test_memory_journal_replay_upsert_dedup() -> None:
    """replay() returns only the latest value per upsert key."""
    journal = MemoryJournal()
    journal.upsert("key1", "v1")
    journal.append("delta1")
    journal.upsert("key1", "v2")  # replaces v1
    events = [event async for event in journal.replay()]
    # v1 should be replaced by v2; delta1 preserved
    assert "v1" not in events
    assert "v2" in events
    assert "delta1" in events


async def test_memory_journal_replay_from_seq() -> None:
    """replay(from_seq=N) returns only events with seq >= N."""
    journal = MemoryJournal()
    journal.append("a")  # seq 1
    journal.append("b")  # seq 2
    journal.append("c")  # seq 3
    events = [event async for event in journal.replay(from_seq=2)]
    assert events == ["b", "c"]


async def test_memory_journal_replay_to_seq() -> None:
    """replay(to_seq=N) returns only events with seq <= N."""
    journal = MemoryJournal()
    journal.append("a")  # seq 1
    journal.append("b")  # seq 2
    journal.append("c")  # seq 3
    events = [event async for event in journal.replay(to_seq=2)]
    assert events == ["a", "b"]


async def test_memory_journal_replay_empty_journal() -> None:
    """replay() on empty journal yields nothing."""
    journal = MemoryJournal()
    events = [event async for event in journal.replay()]
    assert events == []


# ---------------------------------------------------------------------------
# MemoryJournal — resume
# ---------------------------------------------------------------------------


def test_memory_journal_resume_returns_none_without_snapshot():
    """resume() returns None when snapshot store has no snapshot."""
    journal = MemoryJournal()
    store = MemorySnapshotStore()
    result = journal.resume(store)
    assert result is None


def test_memory_journal_resume_normal_recovery():
    """resume() returns ResumeResult with is_inflight=False when no in-flight Turn."""
    journal = MemoryJournal()
    journal.append("event1")
    journal.append("event2")
    store = MemorySnapshotStore()
    # Manually set snapshot with seq=1 (before event2)
    store._snapshot = ("state1", 1)
    result = journal.resume(store)
    assert result is not None
    assert result.is_inflight is False
    assert result.state == "state1"
    assert result.inflight_turn_id is None
    # event2 has seq=2 > 1, so it should be in events
    assert "event2" in result.events


def test_memory_journal_resume_detects_inflight_turn():
    """resume() returns ResumeResult with is_inflight=True when Turn has no result."""
    journal = MemoryJournal()
    # Simulate: snapshot taken at seq=0, then events with turn_id
    journal.append(_MockEvent(turn_id="turn_001", payload="delta"))
    store = MemorySnapshotStore()
    store._snapshot = ("state0", 0)
    # No turn_result saved for turn_001
    result = journal.resume(store)
    assert result is not None
    assert result.is_inflight is True
    assert result.inflight_turn_id == "turn_001"


def test_memory_journal_resume_no_inflight_when_turn_result_exists():
    """resume() returns is_inflight=False when turn_result exists."""
    journal = MemoryJournal()
    journal.append(_MockEvent(turn_id="turn_001", payload="delta"))
    store = MemorySnapshotStore()
    store._snapshot = ("state0", 0)
    store.save_turn_result("turn_001", "result")
    result = journal.resume(store)
    assert result is not None
    assert result.is_inflight is False
    assert result.inflight_turn_id is None


# ---------------------------------------------------------------------------
# MemoryJournal — compact
# ---------------------------------------------------------------------------


def test_memory_journal_compact_removes_old_entries():
    """compact(before_seq=N) removes entries with seq < N."""
    journal = MemoryJournal()
    journal.append("a")  # seq 1
    journal.append("b")  # seq 2
    journal.append("c")  # seq 3
    journal.compact(before_seq=2)
    # Only seq >= 2 should remain
    import asyncio

    events = asyncio.run(_collect_replay(journal))
    assert "a" not in events
    assert "b" in events
    assert "c" in events


def test_memory_journal_compact_removes_old_upserts():
    """compact() removes upsert entries with seq < before_seq."""
    journal = MemoryJournal()
    journal.upsert("key1", "v1")  # seq 1
    journal.append("delta")  # seq 2
    journal.compact(before_seq=2)
    import asyncio

    events = asyncio.run(_collect_replay(journal))
    assert "v1" not in events
    assert "delta" in events


def test_memory_journal_compact_on_empty_does_not_crash():
    """compact() on empty journal is a no-op."""
    journal = MemoryJournal()
    journal.compact(before_seq=100)
    # Should not raise


def test_memory_journal_compact_zero_removes_nothing():
    """compact(before_seq=0) removes nothing (all seq >= 0)."""
    journal = MemoryJournal()
    journal.append("a")  # seq 1
    journal.compact(before_seq=0)
    import asyncio

    events = asyncio.run(_collect_replay(journal))
    assert "a" in events


# ---------------------------------------------------------------------------
# MemoryJournal — clear
# ---------------------------------------------------------------------------


def test_memory_journal_clear_resets_all():
    """clear() removes all entries and resets the seq counter."""
    journal = MemoryJournal()
    journal.append("a")
    journal.upsert("key1", "v1")
    journal.log_tool_execution(
        ToolExecutionRecord(
            turn_id="t1", tool_name="bash", args={}, result="ok", status="completed"
        )
    )
    journal.clear()
    # After clear, next append should return 1
    assert journal.append("new") == 1
    assert journal.get_tool_executions("t1") == []


# ---------------------------------------------------------------------------
# MemoryJournal — tool execution log
# ---------------------------------------------------------------------------


def test_memory_journal_log_tool_execution_roundtrip():
    """log_tool_execution() + get_tool_executions() round-trip."""
    journal = MemoryJournal()
    record1 = ToolExecutionRecord(
        turn_id="t1",
        tool_name="bash",
        args={"command": "ls"},
        result="file1\nfile2",
        status="completed",
    )
    record2 = ToolExecutionRecord(
        turn_id="t1",
        tool_name="read",
        args={"path": "/tmp"},
        result="content",
        status="completed",
    )
    journal.log_tool_execution(record1)
    journal.log_tool_execution(record2)
    results = journal.get_tool_executions("t1")
    assert len(results) == 2
    assert results[0].tool_name == "bash"
    assert results[1].tool_name == "read"
    assert results[0].status == "completed"
    assert results[1].result == "content"


def test_memory_journal_get_tool_executions_empty():
    """get_tool_executions() returns empty list for unknown turn_id."""
    journal = MemoryJournal()
    results = journal.get_tool_executions("nonexistent")
    assert results == []


def test_memory_journal_tool_executions_isolated_by_turn_id():
    """Tool executions are isolated by turn_id."""
    journal = MemoryJournal()
    journal.log_tool_execution(
        ToolExecutionRecord(
            turn_id="t1", tool_name="bash", args={}, result="r1", status="completed"
        )
    )
    journal.log_tool_execution(
        ToolExecutionRecord(
            turn_id="t2", tool_name="read", args={}, result="r2", status="completed"
        )
    )
    assert len(journal.get_tool_executions("t1")) == 1
    assert len(journal.get_tool_executions("t2")) == 1
    assert journal.get_tool_executions("t1")[0].tool_name == "bash"
    assert journal.get_tool_executions("t2")[0].tool_name == "read"


# ---------------------------------------------------------------------------
# DurableJournal — Protocol conformance
# ---------------------------------------------------------------------------


def test_durable_journal_protocol_conformance(tmp_path):
    """DurableJournal satisfies the Journal Protocol."""
    db_url = f"sqlite:///{tmp_path}/test_journal.db"
    journal = DurableJournal(db_url, session_id="test")
    assert isinstance(journal, Journal)
    journal.close()


# ---------------------------------------------------------------------------
# DurableJournal — append / upsert / replay
# ---------------------------------------------------------------------------


def test_durable_journal_append_returns_monotonic_seq(tmp_path):
    """DurableJournal append() returns strictly increasing sequence numbers."""
    db_url = f"sqlite:///{tmp_path}/test_append.db"
    journal = DurableJournal(db_url, session_id="test")
    seq1 = journal.append("event1")
    seq2 = journal.append("event2")
    seq3 = journal.append("event3")
    assert seq1 == 1
    assert seq2 == 2
    assert seq3 == 3
    journal.close()


def test_durable_journal_upsert_replaces_existing(tmp_path):
    """DurableJournal upsert() replaces by key and returns higher seq."""
    db_url = f"sqlite:///{tmp_path}/test_upsert.db"
    journal = DurableJournal(db_url, session_id="test")
    seq1 = journal.upsert("key1", "v1")
    seq2 = journal.upsert("key1", "v2")
    assert seq2 > seq1
    journal.close()


async def test_durable_journal_replay_yields_in_order(tmp_path) -> None:
    """DurableJournal replay() yields events in sequence order."""
    db_url = f"sqlite:///{tmp_path}/test_replay.db"
    journal = DurableJournal(db_url, session_id="test")
    journal.append("a")
    journal.append("b")
    journal.append("c")
    events = [event async for event in journal.replay()]
    assert events == ["a", "b", "c"]
    journal.close()


async def test_durable_journal_replay_upsert_dedup(tmp_path) -> None:
    """DurableJournal replay() returns only latest per upsert key."""
    db_url = f"sqlite:///{tmp_path}/test_replay_upsert.db"
    journal = DurableJournal(db_url, session_id="test")
    journal.upsert("key1", "v1")
    journal.append("delta1")
    journal.upsert("key1", "v2")
    events = [event async for event in journal.replay()]
    assert "v1" not in events
    assert "v2" in events
    assert "delta1" in events
    journal.close()


# ---------------------------------------------------------------------------
# DurableJournal — data survives re-instantiation
# ---------------------------------------------------------------------------


async def test_durable_journal_data_survives_reinstantiation(tmp_path) -> None:
    """DurableJournal data survives re-instantiation with same DB file."""
    db_url = f"sqlite:///{tmp_path}/test_persist.db"
    journal1 = DurableJournal(db_url, session_id="test")
    journal1.append("event1")
    journal1.append("event2")
    journal1.upsert("key1", "value1")
    journal1.close()

    # Re-instantiate with same DB file
    journal2 = DurableJournal(db_url, session_id="test")
    events = [event async for event in journal2.replay()]
    assert "event1" in events
    assert "event2" in events
    assert "value1" in events
    journal2.close()


def test_durable_journal_tool_log_survives_reinstantiation(tmp_path):
    """DurableJournal tool execution log survives re-instantiation."""
    db_url = f"sqlite:///{tmp_path}/test_tool_persist.db"
    journal1 = DurableJournal(db_url, session_id="test")
    journal1.log_tool_execution(
        ToolExecutionRecord(
            turn_id="t1",
            tool_name="bash",
            args={"command": "ls"},
            result="output",
            status="completed",
        )
    )
    journal1.close()

    journal2 = DurableJournal(db_url, session_id="test")
    results = journal2.get_tool_executions("t1")
    assert len(results) == 1
    assert results[0].tool_name == "bash"
    assert results[0].status == "completed"
    assert results[0].result == "output"
    journal2.close()


# ---------------------------------------------------------------------------
# DurableJournal — compact
# ---------------------------------------------------------------------------


def test_durable_journal_compact_removes_old_entries(tmp_path):
    """DurableJournal compact() removes entries with seq < before_seq."""
    db_url = f"sqlite:///{tmp_path}/test_compact.db"
    journal = DurableJournal(db_url, session_id="test")
    journal.append("a")  # seq 1
    journal.append("b")  # seq 2
    journal.append("c")  # seq 3
    journal.compact(before_seq=2)
    import asyncio

    events = asyncio.run(_collect_replay(journal))
    assert "a" not in events
    assert "b" in events
    assert "c" in events
    journal.close()


def test_durable_journal_compact_on_empty_does_not_crash(tmp_path):
    """DurableJournal compact() on empty journal does not crash."""
    db_url = f"sqlite:///{tmp_path}/test_compact_empty.db"
    journal = DurableJournal(db_url, session_id="test")
    journal.compact(before_seq=0)
    journal.compact(before_seq=100)
    journal.close()


# ---------------------------------------------------------------------------
# DurableJournal — clear
# ---------------------------------------------------------------------------


def test_durable_journal_clear_resets(tmp_path):
    """DurableJournal clear() removes all entries."""
    db_url = f"sqlite:///{tmp_path}/test_clear.db"
    journal = DurableJournal(db_url, session_id="test")
    journal.append("a")
    journal.upsert("key1", "v1")
    journal.log_tool_execution(
        ToolExecutionRecord(
            turn_id="t1", tool_name="bash", args={}, result="ok", status="completed"
        )
    )
    journal.clear()
    # After clear, next append should return 1
    assert journal.append("new") == 1
    assert journal.get_tool_executions("t1") == []
    journal.close()


# ---------------------------------------------------------------------------
# DurableJournal — resume
# ---------------------------------------------------------------------------


def test_durable_journal_resume_returns_none_without_snapshot(tmp_path):
    """DurableJournal resume() returns None when no snapshot exists."""
    db_url = f"sqlite:///{tmp_path}/test_resume_none.db"
    journal = DurableJournal(db_url, session_id="test")
    store = MemorySnapshotStore()
    result = journal.resume(store)
    assert result is None
    journal.close()


def test_durable_journal_resume_normal_recovery(tmp_path):
    """DurableJournal resume() returns ResumeResult with events since snapshot."""
    db_url = f"sqlite:///{tmp_path}/test_resume_normal.db"
    journal = DurableJournal(db_url, session_id="test")
    journal.append("event1")  # seq 1
    journal.append("event2")  # seq 2
    store = MemorySnapshotStore()
    store._snapshot = ("state1", 1)  # snapshot after event1
    result = journal.resume(store)
    assert result is not None
    assert result.is_inflight is False
    assert result.state == "state1"
    assert result.inflight_turn_id is None
    assert "event2" in result.events
    assert "event1" not in result.events
    journal.close()


def test_durable_journal_resume_detects_inflight_turn(tmp_path):
    """DurableJournal resume() detects in-flight Turn."""
    db_url = f"sqlite:///{tmp_path}/test_resume_inflight.db"
    journal = DurableJournal(db_url, session_id="test")
    # Use a dict event so turn_id survives JSON round-trip
    journal.append({"turn_id": "turn_001", "payload": "delta"})
    store = MemorySnapshotStore()
    store._snapshot = ("state0", 0)
    result = journal.resume(store)
    assert result is not None
    assert result.is_inflight is True
    assert result.inflight_turn_id == "turn_001"
    journal.close()


# ---------------------------------------------------------------------------
# DurableJournal — WAL mode verification
# ---------------------------------------------------------------------------


def test_durable_journal_enables_wal_mode(tmp_path):
    """DurableJournal sets WAL mode on SQLite databases."""
    import sqlite3

    db_path = tmp_path / "test_wal.db"
    db_url = f"sqlite:///{db_path}"
    journal = DurableJournal(db_url, session_id="test")
    journal.append("event1")
    journal.close()

    # Open the DB directly and check journal_mode
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("PRAGMA journal_mode")
    mode = cursor.fetchone()[0]
    conn.close()
    assert mode == "wal"


# ---------------------------------------------------------------------------
# DurableJournal — accepts Engine instance
# ---------------------------------------------------------------------------


def test_durable_journal_accepts_engine_instance(tmp_path):
    """DurableJournal accepts a SQLAlchemy Engine instance."""
    from sqlalchemy import create_engine

    db_url = f"sqlite:///{tmp_path}/test_engine.db"
    engine = create_engine(db_url)
    journal = DurableJournal(engine, session_id="test")
    seq = journal.append("event1")
    assert seq == 1
    journal.close()


# ---------------------------------------------------------------------------
# DurableJournal — corrupt DB handling
# ---------------------------------------------------------------------------


def test_durable_journal_corrupt_db_handled_gracefully(tmp_path):
    """DurableJournal with corrupt DB file raises a specific exception, not a crash."""
    db_path = tmp_path / "corrupt.db"
    # Write invalid data to the DB file
    db_path.write_text("not a database")
    db_url = f"sqlite:///{db_path}"
    # Creating the journal should raise a DatabaseError, not crash
    with pytest.raises(Exception, match="not a database"):
        DurableJournal(db_url, session_id="test")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _collect_replay(journal: Any) -> list[Any]:
    """Collect all events from journal.replay() into a list."""
    return [event async for event in journal.replay()]
