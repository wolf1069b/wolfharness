"""Tests for message ordering with message_id tiebreaker (issue #227/C2).

Verifies that storage providers sort messages by (timestamp, message_id)
so that messages with identical timestamps have deterministic ordering.
Also tests that SQL log_message preserves the original creation timestamp
instead of overwriting it with log time (R1).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from wolfharness.messaging import ChatMessage
from wolfharness_storage.memory_provider import MemoryStorageProvider


if TYPE_CHECKING:
    from wolfharness_storage.sql_provider import SQLModelProvider


# ---------------------------------------------------------------------------
# L1 Unit Tests — Memory provider (pure in-memory, no DB)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_memory_provider_same_timestamp_orders_by_message_id() -> None:
    """Messages with identical timestamps sort by message_id ascending.

    Given: two messages with the same timestamp but different message_ids
    When: retrieved via get_session_messages
    Then: they appear in message_id ascending order (deterministic)
    """
    ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    provider = MemoryStorageProvider()
    await provider.log_session(session_id="s1", node_name="agent")

    # Insert in reverse order — msg_b has a lexicographically larger message_id
    msg_b = ChatMessage(
        content="second",
        role="assistant",
        session_id="s1",
        message_id="msg_bbb",
        timestamp=ts,
    )
    msg_a = ChatMessage(
        content="first",
        role="user",
        session_id="s1",
        message_id="msg_aaa",
        timestamp=ts,
    )
    await provider.log_message(message=msg_b)
    await provider.log_message(message=msg_a)

    result = await provider.get_session_messages("s1")
    assert len(result) == 2
    assert result[0].message_id == "msg_aaa"
    assert result[1].message_id == "msg_bbb"


@pytest.mark.unit
async def test_memory_provider_timestamp_takes_priority_over_message_id() -> None:
    """Timestamp is the primary sort key; message_id is only a tiebreaker.

    Given: msg_old has an earlier timestamp but a larger message_id
    When: retrieved via get_session_messages
    Then: msg_old comes first because timestamp is primary
    """
    ts_old = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    ts_new = datetime(2025, 1, 1, 12, 0, 1, tzinfo=UTC)
    provider = MemoryStorageProvider()
    await provider.log_session(session_id="s1", node_name="agent")

    msg_old = ChatMessage(
        content="older",
        role="user",
        session_id="s1",
        message_id="msg_zzz",
        timestamp=ts_old,
    )
    msg_new = ChatMessage(
        content="newer",
        role="assistant",
        session_id="s1",
        message_id="msg_aaa",
        timestamp=ts_new,
    )
    await provider.log_message(message=msg_old)
    await provider.log_message(message=msg_new)

    result = await provider.get_session_messages("s1")
    assert len(result) == 2
    assert result[0].message_id == "msg_zzz"
    assert result[1].message_id == "msg_aaa"


@pytest.mark.unit
async def test_memory_provider_three_messages_same_timestamp() -> None:
    """Three messages with the same timestamp sort by message_id ascending.

    Given: three messages with identical timestamps and message_ids c, a, b
    When: retrieved via get_session_messages
    Then: they appear in order a, b, c (message_id ascending)
    """
    ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    provider = MemoryStorageProvider()
    await provider.log_session(session_id="s1", node_name="agent")

    for mid in ("msg_c", "msg_a", "msg_b"):
        await provider.log_message(
            message=ChatMessage(
                content=mid,
                role="user",
                session_id="s1",
                message_id=mid,
                timestamp=ts,
            ),
        )

    result = await provider.get_session_messages("s1")
    assert [m.message_id for m in result] == ["msg_a", "msg_b", "msg_c"]


# ---------------------------------------------------------------------------
# L2 Integration Tests — SQL provider (real SQLite DB)
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_sql_provider_same_timestamp_orders_by_message_id(
    sql_model_provider: SQLModelProvider,
) -> None:
    """SQL provider sorts same-timestamp messages by id ascending.

    Given: two messages with the same timestamp but different ids
    When: retrieved via get_session_messages
    Then: they appear in id ascending order (deterministic)
    """
    session_id = "test_ordering_001"
    await sql_model_provider.log_session(session_id=session_id, node_name="agent")
    ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

    # Insert in reverse order
    for mid in ("msg_zzz", "msg_aaa"):
        await sql_model_provider.log_message(
            message=ChatMessage(
                content=mid,
                role="user",
                session_id=session_id,
                message_id=mid,
                timestamp=ts,
                model_name="openai:test-model",
            ),
        )

    result = await sql_model_provider.get_session_messages(session_id)
    assert len(result) == 2
    assert result[0].message_id == "msg_aaa"
    assert result[1].message_id == "msg_zzz"


@pytest.mark.integration
async def test_sql_log_message_preserves_creation_timestamp(
    sql_model_provider: SQLModelProvider,
) -> None:
    """log_message stores the original message.timestamp, not get_now().

    Given: a ChatMessage with a specific creation timestamp
    When: logged and retrieved from SQL storage
    Then: the stored timestamp matches the original creation timestamp

    This is a regression test for R1: sql_messages.py used to store
    get_now() (log time) instead of message.timestamp (creation time).
    """
    session_id = "test_timestamp_001"
    await sql_model_provider.log_session(session_id=session_id, node_name="agent")

    # Use a timestamp far from "now" so we can detect if get_now() was used
    creation_ts = datetime(2024, 6, 15, 8, 30, 0, tzinfo=UTC)
    message_id = "msg_ts_test_001"

    await sql_model_provider.log_message(
        message=ChatMessage(
            content="test content",
            role="assistant",
            session_id=session_id,
            message_id=message_id,
            timestamp=creation_ts,
            model_name="openai:test-model",
        ),
    )

    stored = await sql_model_provider.get_message(message_id, session_id=session_id)
    assert stored is not None
    # The stored timestamp should match the original creation timestamp,
    # not the time when log_message was called.
    time_diff = abs((stored.timestamp - creation_ts).total_seconds())
    assert time_diff < 1.0, (
        f"Stored timestamp {stored.timestamp} differs from creation timestamp "
        f"{creation_ts} by {time_diff}s — log_message likely used get_now() "
        f"instead of message.timestamp"
    )


@pytest.mark.integration
async def test_sql_provider_three_messages_same_timestamp(
    sql_model_provider: SQLModelProvider,
) -> None:
    """SQL provider sorts three same-timestamp messages by id ascending.

    Given: three messages with identical timestamps and ids c, a, b
    When: retrieved via get_session_messages
    Then: they appear in order a, b, c (id ascending)
    """
    session_id = "test_ordering_002"
    await sql_model_provider.log_session(session_id=session_id, node_name="agent")
    ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

    for mid in ("msg_c", "msg_a", "msg_b"):
        await sql_model_provider.log_message(
            message=ChatMessage(
                content=mid,
                role="user",
                session_id=session_id,
                message_id=mid,
                timestamp=ts,
                model_name="openai:test-model",
            ),
        )

    result = await sql_model_provider.get_session_messages(session_id)
    assert [m.message_id for m in result] == ["msg_a", "msg_b", "msg_c"]


@pytest.mark.integration
async def test_sql_provider_timestamp_priority_with_different_times(
    sql_model_provider: SQLModelProvider,
) -> None:
    """SQL provider uses timestamp as primary sort key even when ids conflict.

    Given: msg_old (earlier ts, larger id) and msg_new (later ts, smaller id)
    When: retrieved via get_session_messages
    Then: msg_old comes first (timestamp primary, id secondary)
    """
    session_id = "test_ordering_003"
    await sql_model_provider.log_session(session_id=session_id, node_name="agent")
    ts_old = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    ts_new = datetime(2025, 1, 1, 12, 0, 1, tzinfo=UTC)

    await sql_model_provider.log_message(
        message=ChatMessage(
            content="old",
            role="user",
            session_id=session_id,
            message_id="msg_zzz",
            timestamp=ts_old,
            model_name="openai:test-model",
        ),
    )
    await sql_model_provider.log_message(
        message=ChatMessage(
            content="new",
            role="assistant",
            session_id=session_id,
            message_id="msg_aaa",
            timestamp=ts_new,
            model_name="openai:test-model",
        ),
    )

    result = await sql_model_provider.get_session_messages(session_id)
    assert len(result) == 2
    assert result[0].message_id == "msg_zzz"
    assert result[1].message_id == "msg_aaa"


# ---------------------------------------------------------------------------
# L1 Unit Tests — Timezone-aware datetime parsing (parse_iso_timestamp)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parse_iso_timestamp_naive_string_returns_aware() -> None:
    """parse_iso_timestamp returns UTC-aware datetime for naive ISO strings.

    Given: an ISO string without timezone info (e.g. "2025-01-01T12:00:00")
    When: parsed via parse_iso_timestamp
    Then: the result has tzinfo=UTC (not None)
    """
    from wolfharness.utils.time_utils import parse_iso_timestamp

    result = parse_iso_timestamp("2025-01-01T12:00:00")
    assert result.tzinfo is not None
    assert result.utcoffset() == UTC.utcoffset(None)


@pytest.mark.unit
def test_parse_iso_timestamp_aware_string_preserves_timezone() -> None:
    """parse_iso_timestamp preserves timezone for already-aware ISO strings.

    Given: an ISO string with explicit timezone (e.g. "...+05:00")
    When: parsed via parse_iso_timestamp
    Then: the result preserves the original timezone offset
    """
    from datetime import timedelta

    from wolfharness.utils.time_utils import parse_iso_timestamp

    result = parse_iso_timestamp("2025-01-01T12:00:00+05:00")
    assert result.tzinfo is not None
    assert result.utcoffset() == timedelta(hours=5)


@pytest.mark.unit
def test_parse_iso_timestamp_z_suffix_returns_utc() -> None:
    """parse_iso_timestamp handles 'Z' suffix and returns UTC-aware datetime.

    Given: an ISO string with 'Z' suffix (e.g. "2025-01-01T12:00:00Z")
    When: parsed via parse_iso_timestamp
    Then: the result has tzinfo=UTC
    """
    from wolfharness.utils.time_utils import parse_iso_timestamp

    result = parse_iso_timestamp("2025-01-01T12:00:00Z")
    assert result.tzinfo is not None
    assert result.utcoffset() == UTC.utcoffset(None)


@pytest.mark.unit
async def test_memory_provider_sort_does_not_crash_with_mixed_timestamps() -> None:
    """Sorting messages with different timezone awareness does not crash.

    Given: messages with timezone-aware timestamps (from get_now)
    When: sorted in memory provider
    Then: no TypeError from comparing naive and aware datetimes

    This is a regression test for the timezone-naive/aware comparison bug
    that could occur when timestamps come from external sources without
    timezone info.
    """
    ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    provider = MemoryStorageProvider()
    await provider.log_session(session_id="s1", node_name="agent")

    await provider.log_message(
        message=ChatMessage(
            content="msg_a",
            role="user",
            session_id="s1",
            message_id="msg_aaa",
            timestamp=ts,
        ),
    )
    await provider.log_message(
        message=ChatMessage(
            content="msg_b",
            role="assistant",
            session_id="s1",
            message_id="msg_bbb",
            timestamp=ts,
        ),
    )

    # Should not raise TypeError
    result = await provider.get_session_messages("s1")
    assert len(result) == 2
