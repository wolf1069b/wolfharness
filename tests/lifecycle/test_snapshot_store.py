"""Tests for SnapshotStore implementations: MemorySnapshotStore and DurableSnapshotStore."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from wolfharness.lifecycle import (
    DurableSnapshotStore,
    MemorySnapshotStore,
    SnapshotStore,
)


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.unit


# --- Protocol conformance ---


def test_memory_snapshot_store_conforms_to_protocol():
    """MemorySnapshotStore is an instance of SnapshotStore Protocol."""
    store = MemorySnapshotStore()
    assert isinstance(store, SnapshotStore)


def test_durable_snapshot_store_conforms_to_protocol(tmp_path: Path):
    """DurableSnapshotStore is an instance of SnapshotStore Protocol."""
    store = DurableSnapshotStore(tmp_path / "test_proto.db", session_id="test")
    try:
        assert isinstance(store, SnapshotStore)
    finally:
        store.close()


# --- MemorySnapshotStore tests ---


def test_memory_snapshot_store_load_returns_none_when_empty():
    """load() returns None when no snapshot has been saved."""
    store = MemorySnapshotStore()
    assert store.load() is None


def test_memory_snapshot_store_save_load_round_trip():
    """save() stores state, load() returns (state, seq) tuple."""
    store = MemorySnapshotStore()
    state: dict[str, Any] = {"phase": "running", "count": 42}
    seq = store.save(state)
    assert seq == 1
    loaded = store.load()
    assert loaded is not None
    loaded_state, loaded_seq = loaded
    assert loaded_state == state
    assert loaded_seq == 1


def test_memory_snapshot_store_save_returns_monotonic_seq():
    """save() returns monotonically increasing sequence numbers."""
    store = MemorySnapshotStore()
    seq1 = store.save({"phase": "idle"})
    seq2 = store.save({"phase": "running"})
    seq3 = store.save({"phase": "done"})
    assert seq1 == 1
    assert seq2 == 2
    assert seq3 == 3
    assert seq1 < seq2 < seq3


def test_memory_snapshot_store_load_returns_latest_snapshot():
    """load() returns the most recently saved snapshot."""
    store = MemorySnapshotStore()
    store.save({"phase": "idle"})
    store.save({"phase": "running"})
    loaded = store.load()
    assert loaded is not None
    loaded_state, _ = loaded
    assert loaded_state == {"phase": "running"}


def test_memory_snapshot_store_save_turn_result_and_has_turn_result():
    """save_turn_result stores result, has_turn_result returns True."""
    store = MemorySnapshotStore()
    assert store.has_turn_result("turn_1") is False
    store.save_turn_result("turn_1", {"output": "hello"})
    assert store.has_turn_result("turn_1") is True


def test_memory_snapshot_store_has_turn_result_false_for_unknown():
    """has_turn_result returns False for unknown turn_id."""
    store = MemorySnapshotStore()
    store.save_turn_result("turn_1", {"output": "hello"})
    assert store.has_turn_result("turn_2") is False


def test_memory_snapshot_store_clear_resets_snapshot():
    """clear() resets the snapshot to None."""
    store = MemorySnapshotStore()
    store.save({"phase": "running"})
    store.clear()
    assert store.load() is None


def test_memory_snapshot_store_clear_resets_turn_results():
    """clear() resets turn results."""
    store = MemorySnapshotStore()
    store.save_turn_result("turn_1", {"output": "hello"})
    store.clear()
    assert store.has_turn_result("turn_1") is False


def test_memory_snapshot_store_clear_resets_seq():
    """clear() resets the sequence counter."""
    store = MemorySnapshotStore()
    store.save({"phase": "running"})
    store.clear()
    seq = store.save({"phase": "idle"})
    assert seq == 1


# --- DurableSnapshotStore tests ---


def test_durable_snapshot_store_load_returns_none_when_empty(tmp_path: Path):
    """load() returns None when no snapshot has been saved."""
    store = DurableSnapshotStore(tmp_path / "test_empty.db", session_id="test")
    try:
        assert store.load() is None
    finally:
        store.close()


def test_durable_snapshot_store_save_load_round_trip(tmp_path: Path):
    """save() stores state, load() returns (state, seq) tuple."""
    store = DurableSnapshotStore(tmp_path / "test_roundtrip.db", session_id="test")
    try:
        state: dict[str, Any] = {"phase": "running", "count": 42}
        seq = store.save(state)
        assert seq >= 1
        loaded = store.load()
        assert loaded is not None
        loaded_state, loaded_seq = loaded
        assert loaded_state == state
        assert loaded_seq == seq
    finally:
        store.close()


def test_durable_snapshot_store_persists_across_reinstantiation(tmp_path: Path):
    """Data survives re-instantiation with the same db path."""
    db_path = tmp_path / "persist.db"
    state: dict[str, Any] = {"phase": "running", "data": [1, 2, 3]}

    store1 = DurableSnapshotStore(db_path, session_id="test")
    seq = store1.save(state)
    store1.save_turn_result("turn_abc", {"result": "done"})
    store1.close()

    store2 = DurableSnapshotStore(db_path, session_id="test")
    try:
        loaded = store2.load()
        assert loaded is not None
        loaded_state, loaded_seq = loaded
        assert loaded_state == state
        assert loaded_seq == seq
        assert store2.has_turn_result("turn_abc") is True
    finally:
        store2.close()


def test_durable_snapshot_store_load_returns_none_for_corrupt(tmp_path: Path):
    """load() returns None for corrupt/partial snapshot data."""
    db_path = tmp_path / "corrupt.db"
    store = DurableSnapshotStore(db_path, session_id="test")
    store.save({"phase": "running"})
    store.close()

    # Corrupt the state_blob directly in the database
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE snapshots SET state_blob = ? WHERE id = 1", ("NOT_VALID_JSON{{{",))
    conn.commit()
    conn.close()

    store2 = DurableSnapshotStore(db_path, session_id="test")
    try:
        loaded = store2.load()
        assert loaded is None
    finally:
        store2.close()


def test_durable_snapshot_store_save_turn_result_and_has_turn_result(tmp_path: Path):
    """save_turn_result persists, has_turn_result checks DB."""
    store = DurableSnapshotStore(tmp_path / "turn_results.db", session_id="test")
    try:
        assert store.has_turn_result("turn_1") is False
        store.save_turn_result("turn_1", {"output": "completed"})
        assert store.has_turn_result("turn_1") is True
    finally:
        store.close()


def test_durable_snapshot_store_turn_result_persists_across_reinstantiation(tmp_path: Path):
    """Turn results persist across re-instantiation."""
    db_path = tmp_path / "turn_persist.db"
    store1 = DurableSnapshotStore(db_path, session_id="test")
    store1.save_turn_result("turn_abc", {"output": "done"})
    store1.close()

    store2 = DurableSnapshotStore(db_path, session_id="test")
    try:
        assert store2.has_turn_result("turn_abc") is True
    finally:
        store2.close()


def test_durable_snapshot_store_clear_removes_all(tmp_path: Path):
    """clear() removes all snapshots and turn results."""
    store = DurableSnapshotStore(tmp_path / "clear.db", session_id="test")
    try:
        store.save({"phase": "running"})
        store.save_turn_result("turn_1", {"output": "done"})
        store.clear()
        assert store.load() is None
        assert store.has_turn_result("turn_1") is False
    finally:
        store.close()


def test_durable_snapshot_store_clear_on_empty_does_not_crash(tmp_path: Path):
    """clear() on empty store does not crash."""
    store = DurableSnapshotStore(tmp_path / "empty_clear.db", session_id="test")
    try:
        store.clear()
        store.clear()
    finally:
        store.close()


def test_durable_snapshot_store_load_returns_latest(tmp_path: Path):
    """load() returns the most recently saved snapshot."""
    store = DurableSnapshotStore(tmp_path / "latest.db", session_id="test")
    try:
        store.save({"phase": "idle"})
        store.save({"phase": "running"})
        loaded = store.load()
        assert loaded is not None
        loaded_state, _ = loaded
        assert loaded_state == {"phase": "running"}
    finally:
        store.close()


def test_durable_snapshot_store_save_turn_result_overwrites(tmp_path: Path):
    """save_turn_result with same turn_id overwrites previous result."""
    store = DurableSnapshotStore(tmp_path / "overwrite.db", session_id="test")
    try:
        store.save_turn_result("turn_1", {"output": "first"})
        store.save_turn_result("turn_1", {"output": "second"})
        assert store.has_turn_result("turn_1") is True
    finally:
        store.close()


# --- Independent composability: MemoryJournal + DurableSnapshotStore ---


def test_memory_journal_plus_durable_snapshot_store_compose(tmp_path: Path):
    """MemoryJournal + DurableSnapshotStore work together independently.

    The SnapshotStore is independently composable — it does not depend
    on the Journal implementation. This test verifies that a
    DurableSnapshotStore can be used standalone without any Journal.
    """
    db_path = tmp_path / "compose.db"

    # Simulate RunLoop: save snapshot at turn boundary
    store = DurableSnapshotStore(db_path, session_id="test")
    try:
        state: dict[str, Any] = {
            "session_id": "sess_1",
            "turn_count": 1,
            "messages": ["user: hello", "assistant: hi"],
        }
        seq = store.save(state)
        store.save_turn_result("turn_001", {"response": "hi"})

        # Simulate crash recovery: re-instantiate and load
        store.close()
        store2 = DurableSnapshotStore(db_path, session_id="test")
        loaded = store2.load()
        assert loaded is not None
        loaded_state, loaded_seq = loaded
        assert loaded_state == state
        assert loaded_seq == seq
        assert store2.has_turn_result("turn_001") is True
        assert store2.has_turn_result("turn_999") is False
    finally:
        store2.close()
