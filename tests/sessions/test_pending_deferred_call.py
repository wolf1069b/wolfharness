"""Tests for PendingDeferredCall model and SessionData extension fields."""

from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import TypeAdapter
import pytest

from wolfharness.sessions.models import PendingDeferredCall, SessionData


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# PendingDeferredCall
# ---------------------------------------------------------------------------


class TestPendingDeferredCall:
    """PendingDeferredCall model tests."""

    def test_construct_minimal(self) -> None:
        """A minimal PendingDeferredCall should construct with required fields."""
        call = PendingDeferredCall(
            tool_call_id="call_abc123",
            tool_name="bash",
            deferred_kind="external",
            deferred_strategy="block",
        )
        assert call.tool_call_id == "call_abc123"
        assert call.tool_name == "bash"
        assert call.deferred_kind == "external"
        assert call.deferred_strategy == "block"
        assert isinstance(call.created_at, datetime)
        assert call.timeout is None

    def test_construct_full(self) -> None:
        """A full PendingDeferredCall should accept an explicit timeout."""
        timeout = timedelta(seconds=300)
        call = PendingDeferredCall(
            tool_call_id="call_xyz",
            tool_name="subagent",
            deferred_kind="unapproved",
            deferred_strategy="continue",
            timeout=timeout,
        )
        assert call.tool_call_id == "call_xyz"
        assert call.tool_name == "subagent"
        assert call.deferred_kind == "unapproved"
        assert call.deferred_strategy == "continue"
        assert call.timeout == timeout

    def test_json_round_trip(self) -> None:
        """TypeAdapter JSON round-trip should produce identical data."""
        call = PendingDeferredCall(
            tool_call_id="call_abc",
            tool_name="bash",
            deferred_kind="external",
            deferred_strategy="block",
            timeout=timedelta(seconds=120),
        )
        adapter = TypeAdapter(PendingDeferredCall)
        data = adapter.dump_python(call, mode="json")
        restored = adapter.validate_python(data)
        assert restored.tool_call_id == call.tool_call_id
        assert restored.tool_name == call.tool_name
        assert restored.deferred_kind == call.deferred_kind
        assert restored.deferred_strategy == call.deferred_strategy
        # timedelta serializes as float seconds in JSON mode
        if isinstance(data["timeout"], (int, float)):
            assert restored.timeout == timedelta(seconds=data["timeout"])
        else:
            assert restored.timeout == call.timeout

    def test_json_round_trip_minimal(self) -> None:
        """TypeAdapter JSON round-trip with no timeout."""
        call = PendingDeferredCall(
            tool_call_id="call_min",
            tool_name="read",
            deferred_kind="external",
            deferred_strategy="stream",
        )
        adapter = TypeAdapter(PendingDeferredCall)
        data = adapter.dump_python(call, mode="json")
        restored = adapter.validate_python(data)
        assert restored.tool_call_id == call.tool_call_id
        assert restored.deferred_kind == "external"
        assert restored.deferred_strategy == "stream"
        assert restored.timeout is None

    def test_deferred_kind_rejects_invalid(self) -> None:
        """deferred_kind should reject values outside the Literal."""
        dct: dict = {
            "tool_call_id": "call_1",
            "tool_name": "bash",
            "deferred_kind": "bogus",
            "deferred_strategy": "block",
        }
        adapter = TypeAdapter(PendingDeferredCall)
        with pytest.raises(Exception, match="deferred_kind"):
            adapter.validate_python(dct)

    def test_deferred_strategy_rejects_invalid(self) -> None:
        """deferred_strategy should reject values outside the Literal."""
        dct: dict = {
            "tool_call_id": "call_1",
            "tool_name": "bash",
            "deferred_kind": "external",
            "deferred_strategy": "bogus",
        }
        adapter = TypeAdapter(PendingDeferredCall)
        with pytest.raises(Exception, match="deferred_strategy"):
            adapter.validate_python(dct)

    def test_model_copy_update(self) -> None:
        """model_copy(update=...) should produce a new immutable copy."""
        call = PendingDeferredCall(
            tool_call_id="call_copy",
            tool_name="read",
            deferred_kind="external",
            deferred_strategy="block",
        )
        updated = call.model_copy(update={"deferred_strategy": "stream"})
        # Original unchanged
        assert call.deferred_strategy == "block"
        # New copy has update
        assert updated.deferred_strategy == "stream"
        assert updated.tool_call_id == call.tool_call_id


# ---------------------------------------------------------------------------
# SessionData extension fields
# ---------------------------------------------------------------------------


class TestSessionDataExtension:
    """SessionData new field tests."""

    def test_default_pending_deferred_calls(self) -> None:
        """SessionData should default pending_deferred_calls to empty list."""
        session = SessionData(session_id="s1", agent_name="test")
        assert session.pending_deferred_calls == []

    def test_default_status(self) -> None:
        """SessionData should default status to 'active'."""
        session = SessionData(session_id="s1", agent_name="test")
        assert session.status == "active"

    def test_default_agent_config_hash_is_none(self) -> None:
        """SessionData should default agent_config_hash to None."""
        session = SessionData(session_id="s1", agent_name="test")
        assert session.agent_config_hash is None

    def test_agent_config_hash_accepts_string(self) -> None:
        """SessionData should accept a string agent_config_hash."""
        session = SessionData(
            session_id="s1",
            agent_name="test",
            agent_config_hash="sha256:abc123",
        )
        assert session.agent_config_hash == "sha256:abc123"

    def test_pending_deferred_calls_with_entries(self) -> None:
        """SessionData should accept a list of PendingDeferredCall."""
        calls = [
            PendingDeferredCall(
                tool_call_id="c1",
                tool_name="bash",
                deferred_kind="external",
                deferred_strategy="block",
            ),
            PendingDeferredCall(
                tool_call_id="c2",
                tool_name="subagent",
                deferred_kind="unapproved",
                deferred_strategy="block",
            ),
        ]
        session = SessionData(
            session_id="s1",
            agent_name="test",
            pending_deferred_calls=calls,
        )
        assert len(session.pending_deferred_calls) == 2
        assert session.pending_deferred_calls[0].tool_call_id == "c1"
        assert session.pending_deferred_calls[1].tool_call_id == "c2"

    def test_status_transition_active_to_checkpointed(self) -> None:
        """model_copy(update={"status": "checkpointed"}) should preserve all fields."""
        session = SessionData(
            session_id="s1",
            agent_name="test",
            agent_config_hash="sha256:def456",
        )
        checkpointed = session.model_copy(update={"status": "checkpointed"})
        assert checkpointed.status == "checkpointed"
        assert checkpointed.session_id == session.session_id
        assert checkpointed.agent_name == session.agent_name
        assert checkpointed.agent_config_hash == "sha256:def456"
        assert checkpointed.pending_deferred_calls == []
        # Original unchanged
        assert session.status == "active"

    def test_status_transition_round_trip(self) -> None:
        """Full status lifecycle: active → checkpointed → resuming → active."""
        session = SessionData(
            session_id="s1",
            agent_name="test",
            pending_deferred_calls=[
                PendingDeferredCall(
                    tool_call_id="c1",
                    tool_name="bash",
                    deferred_kind="external",
                    deferred_strategy="block",
                ),
            ],
        )
        # active → checkpointed
        checkpointed = session.model_copy(update={"status": "checkpointed"})
        assert checkpointed.status == "checkpointed"

        # checkpointed → resuming
        resuming = checkpointed.model_copy(update={"status": "resuming"})
        assert resuming.status == "resuming"
        assert len(resuming.pending_deferred_calls) == 1

        # resuming → active (with cleared pending calls)
        active_again = resuming.model_copy(
            update={"status": "active", "pending_deferred_calls": []}
        )
        assert active_again.status == "active"
        assert active_again.pending_deferred_calls == []

    def test_status_preserved_in_json_round_trip(self) -> None:
        """SessionData with new fields should survive JSON round-trip."""
        call = PendingDeferredCall(
            tool_call_id="call_rt",
            tool_name="bash",
            deferred_kind="external",
            deferred_strategy="block",
        )
        session = SessionData(
            session_id="s_rt",
            agent_name="test_agent",
            status="checkpointed",
            agent_config_hash="sha256:roundtrip",
            pending_deferred_calls=[call],
        )
        adapter = TypeAdapter(SessionData)
        data = adapter.dump_python(session, mode="json")
        restored = adapter.validate_python(data)

        assert restored.session_id == "s_rt"
        assert restored.status == "checkpointed"
        assert restored.agent_config_hash == "sha256:roundtrip"
        assert len(restored.pending_deferred_calls) == 1
        restored_call = restored.pending_deferred_calls[0]
        assert restored_call.tool_call_id == "call_rt"
        assert restored_call.deferred_kind == "external"
        assert restored_call.deferred_strategy == "block"
