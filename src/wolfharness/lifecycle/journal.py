"""Journal dimension: MemoryJournal and DurableJournal.

The Journal provides event-layer persistence for the RunLoop lifecycle.
It supports two write semantics:

- ``append(event)`` — delta events; each call creates a new record.
- ``upsert(key, event)`` — entity-state events; latest state per key
  replaces previous.

Both return a monotonically increasing sequence number.

``MemoryJournal`` is the default in-memory implementation (no persistence
across process restarts). ``DurableJournal`` is a SQL-backed implementation
with WAL mode for crash-safe persistence.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
import json
from typing import TYPE_CHECKING, Any

import logfire
from pydantic import BaseModel
from sqlalchemy import Column, Text, create_engine, event as sa_event, text
from sqlalchemy.orm import Session
from sqlmodel import Field, SQLModel
from sqlmodel.main import SQLModelConfig  # type: ignore[attr-defined]

from wolfharness.lifecycle.types import ResumeResult, ToolExecutionRecord
from wolfharness.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.engine import Engine

    from wolfharness.lifecycle.protocols import SnapshotStore


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# MemoryJournal
# ---------------------------------------------------------------------------


class MemoryJournal:
    """In-memory Journal implementation.

    Uses plain Python lists and dicts. Data is lost on process exit.
    This is the default when no Journal is explicitly provided.
    """

    def __init__(self) -> None:
        """Initialize an empty in-memory journal."""
        self._seq: int = 0
        self._entries: list[tuple[int, Any]] = []
        self._upserts: dict[str, tuple[int, Any]] = {}
        self._tool_log: dict[str, list[ToolExecutionRecord]] = {}

    def append(self, event: Any) -> int:
        """Create a new journal entry for a delta event.

        Args:
            event: The event to append.

        Returns:
            Monotonically increasing sequence number.
        """
        self._seq += 1
        self._entries.append((self._seq, event))
        return self._seq

    def upsert(self, key: str, event: Any) -> int:
        """Replace or create an entity-state entry by key.

        Args:
            key: Deduplication key (e.g. ``"tool_call:abc"``).
            event: The event to upsert.

        Returns:
            Monotonically increasing sequence number.
        """
        self._seq += 1
        self._upserts[key] = (self._seq, event)
        return self._seq

    async def replay(
        self,
        from_seq: int = 0,
        to_seq: int | None = None,
    ) -> AsyncIterator[Any]:
        """Return an async iterator of events in sequence order.

        Append entries are all returned, ordered by ``seq``.
        Upsert entries return only the latest state per key.

        Args:
            from_seq: Start sequence (inclusive). Defaults to 0.
            to_seq: End sequence (inclusive). ``None`` means no upper bound.
        """
        upper = to_seq if to_seq is not None else float("inf")
        merged: list[tuple[int, Any]] = []
        merged.extend((seq, evt) for seq, evt in self._entries if from_seq <= seq <= upper)
        for seq, evt in self._upserts.values():
            if from_seq <= seq <= upper:
                merged.append((seq, evt))
        merged.sort(key=lambda pair: pair[0])

        for _seq, evt in merged:
            yield evt

    def resume(self, snapshot_store: SnapshotStore) -> ResumeResult | None:
        """Coordinate snapshot and journal for crash recovery.

        Loads the latest snapshot, replays journal events since the
        snapshot, and determines if a Turn was in-flight.

        Args:
            snapshot_store: The snapshot store to load state from.

        Returns:
            ``ResumeResult`` if state exists, ``None`` for fresh start.
        """
        loaded = snapshot_store.load()
        if loaded is None:
            return None
        state, last_journal_seq = loaded

        merged_since: list[tuple[int, Any]] = []
        for seq, evt in self._entries:
            if seq > last_journal_seq:
                merged_since.append((seq, evt))
        for seq, evt in self._upserts.values():
            if seq > last_journal_seq:
                merged_since.append((seq, evt))
        merged_since.sort(key=lambda pair: pair[0])
        events_since_snapshot: list[Any] = [evt for _seq, evt in merged_since]

        inflight_turn_id = _detect_inflight_turn(events_since_snapshot, snapshot_store)

        return ResumeResult(
            is_inflight=inflight_turn_id is not None,
            state=state,
            events=events_since_snapshot,
            inflight_turn_id=inflight_turn_id,
        )

    def compact(self, before_seq: int) -> None:
        """Remove journal entries with ``seq < before_seq``.

        Args:
            before_seq: Remove entries with seq below this value.
        """
        self._entries = [(seq, event) for seq, event in self._entries if seq >= before_seq]
        self._upserts = {
            key: (seq, event) for key, (seq, event) in self._upserts.items() if seq >= before_seq
        }

    def clear(self) -> None:
        """Remove all journal entries, resetting the sequence counter."""
        self._seq = 0
        self._entries = []
        self._upserts = {}
        self._tool_log = {}

    def log_tool_execution(self, record: ToolExecutionRecord) -> None:
        """Store a tool execution record for idempotent crash recovery.

        Args:
            record: The tool execution record to store.
        """
        self._tool_log.setdefault(record.turn_id, []).append(record)

    def get_tool_executions(self, turn_id: str) -> list[ToolExecutionRecord]:
        """Retrieve all tool execution records for a Turn.

        Args:
            turn_id: The Turn ID to query.

        Returns:
            List of tool execution records for the given Turn.
        """
        return list(self._tool_log.get(turn_id, []))


# ---------------------------------------------------------------------------
# DurableJournal — SQL-backed with WAL mode
# ---------------------------------------------------------------------------


class _JournalEntry(SQLModel, table=True):
    """SQL model for journal entries."""

    __tablename__ = "lifecycle_journal"
    __table_args__ = {"sqlite_autoincrement": True}

    seq: int = Field(default=None, primary_key=True)
    """Monotonically increasing sequence number (autoincrement)."""

    session_id: str = Field(index=True)
    """Session identifier for isolation in shared DB."""

    entry_type: str = Field(index=True)
    """``"append"`` or ``"upsert"``."""

    upsert_key: str | None = Field(default=None, index=True)
    """Dedup key for upsert entries; ``None`` for append entries."""

    event_json: str = Field(sa_column=Column(Text))
    """JSON-serialized event payload."""

    model_config = SQLModelConfig(use_attribute_docstrings=True)  # pyright: ignore[reportCallIssue]


class _ToolLogEntry(SQLModel, table=True):
    """SQL model for tool execution log entries."""

    __tablename__ = "lifecycle_tool_log"

    id: int = Field(default=None, primary_key=True)
    """Auto-increment primary key."""

    session_id: str = Field(index=True)
    """Session identifier for isolation in shared DB."""

    turn_id: str = Field(index=True)
    """The Turn that executed this tool call."""

    tool_name: str
    """Name of the tool that was called."""

    args_json: str = Field(sa_column=Column(Text))
    """JSON-serialized input arguments."""

    result_json: str | None = Field(default=None, sa_column=Column(Text))
    """JSON-serialized result, or ``None`` if not completed."""

    status: str
    """Execution status (``"completed"``, ``"failed"``, etc.)."""

    model_config = SQLModelConfig(use_attribute_docstrings=True)  # pyright: ignore[reportCallIssue]


def _enable_wal(dbapi_conn: Any, connection_record: Any) -> None:
    """SQLAlchemy event listener to enable WAL mode on SQLite connections.

    Sets ``journal_mode=WAL`` and ``synchronous=NORMAL`` for crash-safe
    writes with good performance.
    """
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


# Known event types for deserialization reconstruction.
_KNOWN_EVENT_TYPES: dict[str, type] = {}


def _register_event_type(cls: type) -> type:
    """Register an event class for deserialization reconstruction.

    Args:
        cls: The event class to register.

    Returns:
        The same class (for use as a decorator).
    """
    _KNOWN_EVENT_TYPES[cls.__name__] = cls
    return cls


def _json_default(obj: Any) -> Any:
    """Custom JSON default handler for dataclasses, Pydantic models, and Enums.

    Adds a ``__event_type__`` marker to serialized dataclasses and
    Pydantic models so they can be reconstructed during deserialization.

    Args:
        obj: The object to serialize.

    Returns:
        A JSON-serializable representation of the object.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result: dict[str, Any] = dataclasses.asdict(obj)
        result["__event_type__"] = type(obj).__name__
        return result
    if isinstance(obj, BaseModel):
        result = obj.model_dump(mode="json")
        result["__event_type__"] = type(obj).__name__
        return result
    if isinstance(obj, Enum):
        return obj.value
    return str(obj)


def _serialize_event(event: Any) -> str:
    """Serialize an event to JSON string.

    Handles dataclasses, Pydantic models, and Enums via a custom
    default handler. Falls back to ``repr()`` if the event is not
    JSON-serializable.
    """
    try:
        return json.dumps(event, default=_json_default)
    except (TypeError, ValueError):
        return repr(event)


def _deserialize_event(event_json: str) -> Any:
    """Deserialize an event from JSON string.

    If the JSON contains a ``__event_type__`` marker matching a
    registered event class, reconstructs the original event object.
    Otherwise returns the parsed JSON value. If parsing fails,
    returns the raw string.

    Args:
        event_json: The JSON string to deserialize.

    Returns:
        The deserialized event object, parsed JSON value, or raw string.
    """
    try:
        data: Any = json.loads(event_json)
    except (json.JSONDecodeError, TypeError):
        return event_json

    if isinstance(data, dict):
        event_type_name = data.pop("__event_type__", None)
        if event_type_name is not None:
            cls = _KNOWN_EVENT_TYPES.get(event_type_name)
            if cls is not None:
                try:
                    if isinstance(cls, type) and issubclass(cls, BaseModel):
                        return cls.model_validate(data)
                    if dataclasses.is_dataclass(cls):
                        valid_fields: set[str] = {f.name for f in dataclasses.fields(cls)}
                        filtered_data: dict[str, Any] = {
                            k: v for k, v in data.items() if k in valid_fields
                        }
                        return cls(**filtered_data)
                except Exception:  # noqa: BLE001
                    pass  # Fall through to return dict

    return data


def _extract_turn_id(event: Any) -> str | None:
    """Extract turn_id from an event object if present.

    Checks for a ``turn_id`` attribute on the event. Returns ``None``
    if the event has no turn_id or is not an object with attributes.
    """
    try:
        turn_id: str | None = event.turn_id
    except AttributeError:
        pass
    else:
        return turn_id
    if isinstance(event, dict):
        value: Any = event.get("turn_id")
        if isinstance(value, str):
            return value
    return None


def _detect_inflight_turn(
    events: list[Any],
    snapshot_store: SnapshotStore,
) -> str | None:
    """Detect an in-flight Turn from events since the last snapshot.

    Returns the turn_id of the first event whose turn_id has no
    corresponding turn_result in the snapshot store.
    """
    for evt in events:
        turn_id = _extract_turn_id(evt)
        if turn_id is not None and not snapshot_store.has_turn_result(turn_id):
            return turn_id
    return None


class DurableJournal:
    """SQL-backed Journal with crash-safe writes.

    Uses SQLite with WAL mode (or any SQLAlchemy-supported database).
    Persists data across process restarts. Used when
    ``lifecycle.journal: durable`` is configured.

    The constructor accepts an ``Engine`` (or a database URL string)
    so callers can reuse an existing engine from the storage subsystem.
    All Protocol-required methods are synchronous; ``replay()`` is async
    as required by the Protocol.
    """

    def __init__(
        self,
        engine: Engine | str,
        session_id: str,
    ) -> None:
        """Initialize the durable journal.

        Args:
            engine: A SQLAlchemy ``Engine`` instance or a database URL
                string. When a URL string is provided, a new sync engine
                is created.
            session_id: Session identifier for isolating entries in a
                shared database.
        """
        if isinstance(engine, str):
            url = engine
            self._engine: Engine = create_engine(url)
        else:
            self._engine = engine

        self._session_id: str = session_id

        # Enable WAL mode for SQLite databases.
        if self._engine.dialect.name == "sqlite":
            sa_event.listen(self._engine, "connect", _enable_wal)

        self._init_tables()

    def _init_tables(self) -> None:
        """Create tables if they don't exist."""
        SQLModel.metadata.create_all(self._engine)

    @logfire.instrument("lifecycle.journal.append")
    def append(self, event: Any) -> int:
        """Create a new journal entry for a delta event.

        For crash-safety, the write is committed with fsync (WAL mode
        ensures this for SQLite with ``synchronous=NORMAL``).

        Args:
            event: The event to append.

        Returns:
            Monotonically increasing sequence number.
        """
        entry = _JournalEntry(
            session_id=self._session_id,
            entry_type="append",
            upsert_key=None,
            event_json=_serialize_event(event),
        )
        with Session(self._engine) as session:
            session.add(entry)
            session.commit()
            session.refresh(entry)
            return entry.seq

    @logfire.instrument("lifecycle.journal.upsert")
    def upsert(self, key: str, event: Any) -> int:
        """Replace or create an entity-state entry by key.

        Args:
            key: Deduplication key.
            event: The event to upsert.

        Returns:
            Monotonically increasing sequence number.
        """
        from sqlalchemy import delete as sa_delete

        with Session(self._engine) as session:
            session.execute(
                sa_delete(_JournalEntry).where(
                    _JournalEntry.upsert_key == key,  # type: ignore[arg-type]
                    _JournalEntry.session_id == self._session_id,  # type: ignore[arg-type]
                )
            )
            entry = _JournalEntry(
                session_id=self._session_id,
                entry_type="upsert",
                upsert_key=key,
                event_json=_serialize_event(event),
            )
            session.add(entry)
            session.commit()
            session.refresh(entry)
            return entry.seq

    async def replay(
        self,
        from_seq: int = 0,
        to_seq: int | None = None,
    ) -> AsyncIterator[Any]:
        """Return an async iterator of events in sequence order.

        Append entries are all returned, ordered by ``seq``.
        Upsert entries return only the latest state per key.

        Args:
            from_seq: Start sequence (inclusive). Defaults to 0.
            to_seq: End sequence (inclusive). ``None`` means no upper bound.
        """
        from sqlmodel import select

        with Session(self._engine) as session:
            stmt = (
                select(_JournalEntry)
                .where(_JournalEntry.seq >= from_seq)
                .where(_JournalEntry.session_id == self._session_id)
            )
            if to_seq is not None:
                stmt = stmt.where(_JournalEntry.seq <= to_seq)
            stmt = stmt.order_by(_JournalEntry.seq.asc())  # type: ignore[attr-defined]
            result = session.execute(stmt)
            rows: list[_JournalEntry] = list(result.scalars().all())

        # For upsert dedup: only yield the latest entry per key.
        # Since rows are ordered by seq ascending, the latest upsert
        # for a key comes last. Pre-compute the max seq per upsert key.
        latest_upsert_seq: dict[str, int] = {}
        for row in rows:
            if row.entry_type == "upsert" and row.upsert_key is not None:
                latest_upsert_seq[row.upsert_key] = row.seq

        for row in rows:
            if (
                row.entry_type == "upsert"
                and row.upsert_key is not None
                and row.seq != latest_upsert_seq[row.upsert_key]
            ):
                continue
            yield _deserialize_event(row.event_json)

    @logfire.instrument("lifecycle.journal.resume")
    def resume(self, snapshot_store: SnapshotStore) -> ResumeResult | None:
        """Coordinate snapshot and journal for crash recovery.

        Args:
            snapshot_store: The snapshot store to load state from.

        Returns:
            ``ResumeResult`` if state exists, ``None`` for fresh start.
        """
        loaded = snapshot_store.load()
        if loaded is None:
            return None
        state, last_journal_seq = loaded

        from sqlmodel import select

        with Session(self._engine) as session:
            stmt = (
                select(_JournalEntry)
                .where(_JournalEntry.seq > last_journal_seq)
                .where(_JournalEntry.session_id == self._session_id)
                .order_by(_JournalEntry.seq.asc())  # type: ignore[attr-defined]
            )
            result = session.execute(stmt)
            rows: list[_JournalEntry] = list(result.scalars().all())

        events_since_snapshot: list[Any] = [_deserialize_event(row.event_json) for row in rows]

        inflight_turn_id = _detect_inflight_turn(events_since_snapshot, snapshot_store)

        return ResumeResult(
            is_inflight=inflight_turn_id is not None,
            state=state,
            events=events_since_snapshot,
            inflight_turn_id=inflight_turn_id,
        )

    def compact(self, before_seq: int) -> None:
        """Remove journal entries with ``seq < before_seq``.

        Args:
            before_seq: Remove entries with seq below this value.
        """
        from sqlalchemy import delete as sa_delete

        with Session(self._engine) as session:
            session.execute(
                sa_delete(_JournalEntry).where(
                    _JournalEntry.seq < before_seq,  # type: ignore[arg-type]
                    _JournalEntry.session_id == self._session_id,  # type: ignore[arg-type]
                )
            )
            session.commit()

    def clear(self) -> None:
        """Remove all journal entries for this session, resetting the sequence counter."""
        from sqlalchemy import delete as sa_delete

        with Session(self._engine) as session:
            session.execute(
                sa_delete(_JournalEntry).where(
                    _JournalEntry.session_id == self._session_id  # type: ignore[arg-type]
                )
            )
            session.execute(
                sa_delete(_ToolLogEntry).where(
                    _ToolLogEntry.session_id == self._session_id  # type: ignore[arg-type]
                )
            )
            # Reset autoincrement only if the table is now completely empty
            # (i.e., no other sessions have entries in this shared DB).
            if self._engine.dialect.name == "sqlite":
                remaining = session.execute(text("SELECT COUNT(*) FROM lifecycle_journal")).scalar()
                if remaining == 0:
                    session.execute(
                        text("DELETE FROM sqlite_sequence WHERE name = 'lifecycle_journal'")
                    )
                    session.execute(
                        text("DELETE FROM sqlite_sequence WHERE name = 'lifecycle_tool_log'")
                    )
            session.commit()

    def log_tool_execution(self, record: ToolExecutionRecord) -> None:
        """Store a tool execution record for idempotent crash recovery.

        Args:
            record: The tool execution record to store.
        """
        entry = _ToolLogEntry(
            session_id=self._session_id,
            turn_id=record.turn_id,
            tool_name=record.tool_name,
            args_json=json.dumps(record.args, default=str),
            result_json=(
                json.dumps(record.result, default=str) if record.result is not None else None
            ),
            status=record.status,
        )
        with Session(self._engine) as session:
            session.add(entry)
            session.commit()

    def get_tool_executions(self, turn_id: str) -> list[ToolExecutionRecord]:
        """Retrieve all tool execution records for a Turn.

        Args:
            turn_id: The Turn ID to query.

        Returns:
            List of tool execution records for the given Turn.
        """
        from sqlmodel import select

        with Session(self._engine) as session:
            stmt = (
                select(_ToolLogEntry)
                .where(_ToolLogEntry.turn_id == turn_id)
                .where(_ToolLogEntry.session_id == self._session_id)
            )
            result = session.execute(stmt)
            rows: list[_ToolLogEntry] = list(result.scalars().all())

        return [
            ToolExecutionRecord(
                turn_id=row.turn_id,
                tool_name=row.tool_name,
                args=(json.loads(row.args_json) if row.args_json else {}),
                result=(json.loads(row.result_json) if row.result_json is not None else None),
                status=row.status,
            )
            for row in rows
        ]

    def close(self) -> None:
        """Dispose the database engine."""
        self._engine.dispose()


# Register known event types for deserialization reconstruction.
from wolfharness.agents.events.events import (  # noqa: E402
    MessageReplacementEvent,
    PlanUpdateEvent,
    RunErrorEvent,
    RunFailedEvent,
    RunStartedEvent,
    StateUpdate,
    StreamCompleteEvent,
    ToolCallUpdateEvent,
)


for _cls in [
    StateUpdate,
    ToolCallUpdateEvent,
    MessageReplacementEvent,
    PlanUpdateEvent,
    RunStartedEvent,
    RunErrorEvent,
    RunFailedEvent,
    StreamCompleteEvent,
]:
    _register_event_type(_cls)


__all__ = [
    "DurableJournal",
    "MemoryJournal",
]
