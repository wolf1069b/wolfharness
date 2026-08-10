"""SnapshotStore implementations: MemorySnapshotStore and DurableSnapshotStore.

MemorySnapshotStore is the default in-memory implementation using plain dicts.
DurableSnapshotStore provides crash-safe SQL persistence with fsync on write.

Both implementations conform to the ``SnapshotStore`` Protocol defined in
``wolfharness.lifecycle.protocols``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType
    from typing import Self


import logfire

from wolfharness.log import get_logger


logger = get_logger(__name__)


class MemorySnapshotStore:
    """In-memory SnapshotStore using plain Python dicts.

    Does NOT persist data across process restarts. Used as the default
    when no SnapshotStore is explicitly provided.

    Attributes:
        _snapshot: The latest ``(state, last_journal_seq)`` tuple, or ``None``.
        _turn_results: Mapping of ``turn_id`` to result.
        _seq: Monotonically increasing sequence counter.
    """

    def __init__(self) -> None:
        """Initialize an empty in-memory snapshot store."""
        self._snapshot: tuple[Any, int] | None = None
        self._turn_results: dict[str, Any] = {}
        self._seq: int = 0

    def save(self, state: Any) -> int:
        """Persist a full state snapshot and return the sequence number.

        Args:
            state: The RunState snapshot to persist.

        Returns:
            The sequence number of the snapshot.
        """
        self._seq += 1
        self._snapshot = (state, self._seq)
        return self._seq

    def load(self) -> tuple[Any, int] | None:
        """Return the latest snapshot.

        Returns:
            Tuple of ``(state, last_journal_seq)`` if a snapshot exists,
            ``None`` otherwise.
        """
        return self._snapshot

    def save_turn_result(self, turn_id: str, result: Any) -> None:
        """Persist a completed Turn's result for idempotency.

        Args:
            turn_id: The Turn ID.
            result: The Turn's result.
        """
        self._turn_results[turn_id] = result

    def has_turn_result(self, turn_id: str) -> bool:
        """Check whether a Turn was already completed.

        Args:
            turn_id: The Turn ID to check.

        Returns:
            ``True`` if the Turn result was saved, ``False`` otherwise.
        """
        return turn_id in self._turn_results

    def clear(self) -> None:
        """Remove all snapshots and turn results."""
        self._snapshot = None
        self._turn_results.clear()
        self._seq = 0


class DurableSnapshotStore:
    """SQL-backed SnapshotStore with crash-safe atomic writes.

    Uses SQLite with WAL mode and explicit ``fsync`` on every write.
    Data persists across process restarts. Independently composable with
    any Journal implementation.

    The constructor accepts a database path. Tables are created on
    first use via ``_ensure_tables()``.
    """

    def __init__(self, db_path: str | Path, session_id: str) -> None:
        """Initialize the durable snapshot store.

        Args:
            db_path: Filesystem path to the SQLite database file.
            session_id: Session identifier for isolating entries in a
                shared database.
        """
        self._db_path: str = str(db_path)
        self._session_id: str = session_id
        self._lock: threading.Lock = threading.Lock()
        self._conn: sqlite3.Connection = sqlite3.connect(
            self._db_path,
            isolation_level=None,  # autocommit mode; we manage transactions
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Create tables if they do not exist."""
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    seq INTEGER NOT NULL DEFAULT 0,
                    state_blob TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """,
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_snapshots_session
                ON snapshots (session_id)
                """,
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turn_results (
                    turn_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    result_blob TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """,
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_turn_results_session
                ON turn_results (session_id)
                """,
            )

    @logfire.instrument("lifecycle.snapshot.save")
    def save(self, state: Any) -> int:
        """Persist a full state snapshot with crash-safe atomic write.

        Writes the state as a JSON blob within a transaction. The WAL
        mode and ``synchronous=FULL`` ensure the write is durable to
        disk before the method returns.

        Args:
            state: The RunState snapshot to persist.

        Returns:
            The sequence number of the snapshot.
        """
        state_blob = json.dumps(state, default=str)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            cursor = self._conn.execute(
                "INSERT INTO snapshots (session_id, state_blob) VALUES (?, ?)",
                (self._session_id, state_blob),
            )
            row_id = cursor.lastrowid
            self._conn.execute("COMMIT")
        if row_id is None:
            msg = "Failed to insert snapshot: no rowid returned"
            raise RuntimeError(msg)
        return row_id

    @logfire.instrument("lifecycle.snapshot.load")
    def load(self) -> tuple[Any, int] | None:
        """Return the latest snapshot, rejecting partial/corrupt data.

        Returns:
            Tuple of ``(state, last_journal_seq)`` if a valid snapshot
            exists, ``None`` otherwise.
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id, state_blob FROM snapshots "
                "WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                (self._session_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        row_id, state_blob = row
        try:
            state = json.loads(state_blob)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Corrupt snapshot detected (id=%s), returning None", row_id)
            return None
        return (state, row_id)

    def save_turn_result(self, turn_id: str, result: Any) -> None:
        """Persist a completed Turn's result to the database.

        Args:
            turn_id: The Turn ID.
            result: The Turn's result.
        """
        result_blob = json.dumps(result, default=str)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.execute(
                """
                INSERT INTO turn_results (turn_id, session_id, result_blob)
                VALUES (?, ?, ?)
                ON CONFLICT(turn_id) DO UPDATE SET result_blob = excluded.result_blob
                """,
                (turn_id, self._session_id, result_blob),
            )
            self._conn.execute("COMMIT")

    def has_turn_result(self, turn_id: str) -> bool:
        """Check whether a Turn was already completed.

        Args:
            turn_id: The Turn ID to check.

        Returns:
            ``True`` if the Turn result was saved, ``False`` otherwise.
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT 1 FROM turn_results WHERE turn_id = ? AND session_id = ?",
                (turn_id, self._session_id),
            )
            return cursor.fetchone() is not None

    def clear(self) -> None:
        """Remove all snapshots and turn results for this session from the database."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.execute(
                "DELETE FROM snapshots WHERE session_id = ?",
                (self._session_id,),
            )
            self._conn.execute(
                "DELETE FROM turn_results WHERE session_id = ?",
                (self._session_id,),
            )
            self._conn.execute("COMMIT")

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager, closing the connection."""
        with self._lock:
            self._conn.close()


__all__ = [
    "DurableSnapshotStore",
    "MemorySnapshotStore",
]
