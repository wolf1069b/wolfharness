"""Tests for ``SQLMessagesMixin.truncate_messages``.

Validates that ``truncate_messages`` deletes the boundary message and every
message whose timestamp is greater than or equal to the boundary timestamp,
using a real in-memory SQLite database via async SQLAlchemy.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from wolfharness_config.storage import SQLStorageConfig
from wolfharness_storage.sql_provider.models import Message
from wolfharness_storage.sql_provider.sql_provider import SQLModelProvider


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def provider(tmp_path: Path) -> SQLModelProvider:
    """Create a SQL provider with a temp database (no auto-migration)."""
    db_path = tmp_path / "test_truncate.db"
    config = SQLStorageConfig(url=f"sqlite:///{db_path}", auto_migration=False)
    return SQLModelProvider(config)


def _make_message(
    *,
    msg_id: str,
    session_id: str,
    timestamp: datetime,
    role: str = "user",
    content: str = "",
) -> Message:
    """Build a ``Message`` row with only the fields truncate_messages cares about."""
    return Message(
        id=msg_id,
        session_id=session_id,
        timestamp=timestamp,
        role=role,
        content=content,
    )


async def _insert_messages(provider: SQLModelProvider, messages: list[Message]) -> None:
    """Insert message rows directly via async SQLAlchemy."""
    async with AsyncSession(provider.engine) as session:
        for msg in messages:
            session.add(msg)
        await session.commit()


async def _remaining_ids(provider: SQLModelProvider, session_id: str) -> list[str]:
    """Return the IDs of messages still present for a session, ordered by timestamp."""
    async with AsyncSession(provider.engine) as session:
        result = await session.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.timestamp.asc(), Message.id.asc())  # type: ignore[arg-type]
        )
        return [m.id for m in result.scalars().all()]


class TestTruncateMessages:
    """Unit tests for ``SQLMessagesMixin.truncate_messages``."""

    @pytest.mark.unit
    async def test_truncate_middle_preserves_earlier(self, provider: SQLModelProvider) -> None:
        """Truncating at message 3 of 5 deletes messages 3, 4, 5 and preserves 1, 2."""
        base = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        session_id = "s-truncate-middle"
        messages = [
            _make_message(
                msg_id=f"m{i}", session_id=session_id, timestamp=base + timedelta(seconds=i)
            )
            for i in range(1, 6)
        ]

        async with provider:
            await _insert_messages(provider, messages)
            deleted = await provider.truncate_messages(session_id, "m3")
            remaining = await _remaining_ids(provider, session_id)

        assert deleted == 3
        assert remaining == ["m1", "m2"]

    @pytest.mark.unit
    async def test_truncate_unknown_message_returns_zero(self, provider: SQLModelProvider) -> None:
        """Truncating with an unknown message_id returns 0 and deletes nothing."""
        base = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        session_id = "s-unknown"
        messages = [
            _make_message(
                msg_id=f"m{i}", session_id=session_id, timestamp=base + timedelta(seconds=i)
            )
            for i in range(1, 6)
        ]

        async with provider:
            await _insert_messages(provider, messages)
            deleted = await provider.truncate_messages(session_id, "nonexistent")
            remaining = await _remaining_ids(provider, session_id)

        assert deleted == 0
        assert remaining == ["m1", "m2", "m3", "m4", "m5"]

    @pytest.mark.unit
    async def test_truncate_first_deletes_all(self, provider: SQLModelProvider) -> None:
        """Truncating at the first message deletes every message in the session."""
        base = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        session_id = "s-truncate-first"
        messages = [
            _make_message(
                msg_id=f"m{i}", session_id=session_id, timestamp=base + timedelta(seconds=i)
            )
            for i in range(1, 6)
        ]

        async with provider:
            await _insert_messages(provider, messages)
            deleted = await provider.truncate_messages(session_id, "m1")
            remaining = await _remaining_ids(provider, session_id)

        assert deleted == 5
        assert remaining == []

    @pytest.mark.unit
    async def test_truncate_same_timestamp_deletes_both(self, provider: SQLModelProvider) -> None:
        """Two messages sharing a timestamp are both deleted when truncating at the first."""
        ts = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        session_id = "s-same-ts"
        messages = [
            _make_message(msg_id="m1", session_id=session_id, timestamp=ts),
            _make_message(msg_id="m2", session_id=session_id, timestamp=ts),
        ]

        async with provider:
            await _insert_messages(provider, messages)
            deleted = await provider.truncate_messages(session_id, "m1")
            remaining = await _remaining_ids(provider, session_id)

        assert deleted == 2
        assert remaining == []

    @pytest.mark.unit
    async def test_truncate_with_real_sqlalchemy_timestamps(
        self, provider: SQLModelProvider
    ) -> None:
        """Verify SQLite timestamp storage format works with the >= comparison.

        Inserts messages with timezone-aware datetimes, truncates, and verifies
        via direct SQL query that the UTCDateTime TypeDecorator round-trips
        correctly so that ``timestamp >= boundary`` matches as expected.
        """
        base = datetime(2026, 7, 21, 9, 30, 45, tzinfo=UTC)
        session_id = "s-real-ts"
        messages = [
            _make_message(msg_id="early", session_id=session_id, timestamp=base),
            _make_message(
                msg_id="mid",
                session_id=session_id,
                timestamp=base + timedelta(milliseconds=500),
            ),
            _make_message(
                msg_id="late",
                session_id=session_id,
                timestamp=base + timedelta(seconds=2),
            ),
        ]

        async with provider:
            await _insert_messages(provider, messages)

            # Truncate at "mid" — should delete "mid" and "late", keep "early".
            deleted = await provider.truncate_messages(session_id, "mid")
            remaining = await _remaining_ids(provider, session_id)

        assert deleted == 2
        assert remaining == ["early"]

    @pytest.mark.unit
    async def test_truncate_does_not_affect_other_sessions(
        self, provider: SQLModelProvider
    ) -> None:
        """Truncating one session does not touch messages in another session."""
        base = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        session_a = "s-a"
        session_b = "s-b"
        messages = [
            _make_message(msg_id="a1", session_id=session_a, timestamp=base),
            _make_message(msg_id="a2", session_id=session_a, timestamp=base + timedelta(seconds=1)),
            _make_message(msg_id="a3", session_id=session_a, timestamp=base + timedelta(seconds=2)),
            _make_message(msg_id="b1", session_id=session_b, timestamp=base),
            _make_message(msg_id="b2", session_id=session_b, timestamp=base + timedelta(seconds=1)),
        ]

        async with provider:
            await _insert_messages(provider, messages)
            deleted = await provider.truncate_messages(session_a, "a2")
            remaining_a = await _remaining_ids(provider, session_a)
            remaining_b = await _remaining_ids(provider, session_b)

        assert deleted == 2
        assert remaining_a == ["a1"]
        assert remaining_b == ["b1", "b2"]
