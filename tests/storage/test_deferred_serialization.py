"""Tests for PendingDeferredCall serialization."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from wolfharness.sessions.models import PendingDeferredCall
from wolfharness.storage.serialization import (
    deserialize_pending_calls,
    serialize_pending_calls,
)


pytestmark = pytest.mark.unit


def _make_call(
    tool_call_id: str = "call_1",
    tool_name: str = "bash",
    deferred_kind: str = "external",
    deferred_strategy: str = "block",
    created_at: datetime | None = None,
    timeout: timedelta | None = None,
) -> PendingDeferredCall:
    return PendingDeferredCall(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        deferred_kind=deferred_kind,  # type: ignore[arg-type]
        deferred_strategy=deferred_strategy,  # type: ignore[arg-type]
        created_at=created_at or datetime(2025, 1, 1, tzinfo=UTC),
        timeout=timeout,
    )


def test_round_trip_single_call():
    """A single PendingDeferredCall survives round-trip serialization."""
    original = _make_call(tool_call_id="test_1", tool_name="subagent")
    json_str = serialize_pending_calls([original])
    result = deserialize_pending_calls(json_str)

    assert len(result) == 1
    assert result[0].tool_call_id == original.tool_call_id
    assert result[0].tool_name == original.tool_name
    assert result[0].deferred_kind == original.deferred_kind
    assert result[0].deferred_strategy == original.deferred_strategy
    assert result[0].created_at == original.created_at
    assert result[0].timeout == original.timeout


def test_round_trip_multiple_calls():
    """Multiple PendingDeferredCalls survive round-trip serialization."""
    calls = [
        _make_call(
            tool_call_id="call_a",
            tool_name="bash",
            deferred_kind="unapproved",
            deferred_strategy="continue",
        ),
        _make_call(
            tool_call_id="call_b",
            tool_name="subagent",
            deferred_kind="external",
            deferred_strategy="stream",
        ),
        _make_call(
            tool_call_id="call_c",
            tool_name="read",
            deferred_kind="external",
            deferred_strategy="block",
        ),
    ]
    json_str = serialize_pending_calls(calls)
    result = deserialize_pending_calls(json_str)

    assert len(result) == 3
    for original, restored in zip(calls, result, strict=False):
        assert restored.tool_call_id == original.tool_call_id
        assert restored.tool_name == original.tool_name
        assert restored.deferred_kind == original.deferred_kind
        assert restored.deferred_strategy == original.deferred_strategy


def test_empty_list():
    """Empty list serializes to '[]' and deserializes to empty list."""
    json_str = serialize_pending_calls([])
    assert json_str == "[]"
    result = deserialize_pending_calls("[]")
    assert result == []


def test_round_trip_with_timeout():
    """PendingDeferredCall with timeout survives round-trip."""
    original = _make_call(
        tool_call_id="timeout_test",
        timeout=timedelta(seconds=300),
    )
    json_str = serialize_pending_calls([original])
    result = deserialize_pending_calls(json_str)

    assert len(result) == 1
    assert result[0].timeout == timedelta(seconds=300)


def test_round_trip_without_timeout():
    """PendingDeferredCall without timeout (None) survives round-trip."""
    original = _make_call(
        tool_call_id="no_timeout",
        timeout=None,
    )
    json_str = serialize_pending_calls([original])
    result = deserialize_pending_calls(json_str)

    assert len(result) == 1
    assert result[0].timeout is None


def test_deserialize_none_returns_empty():
    """deserialize_pending_calls(None) returns empty list."""
    result = deserialize_pending_calls(None)
    assert result == []


def test_deserialize_empty_string_returns_empty():
    """deserialize_pending_calls('') returns empty list."""
    result = deserialize_pending_calls("")
    assert result == []


def test_deserialize_invalid_json_returns_empty():
    """deserialize_pending_calls with invalid JSON returns empty list."""
    result = deserialize_pending_calls("not valid json")
    assert result == []
