"""Round-trip migration test: SQLModelProvider saves and loads all SessionData fields.

Verifies that save_session() → load_session() preserves:
- status (active, checkpointed, closed, resuming)
- pending_deferred_calls (serialized in metadata_json)
- checkpoint_data (preserved from existing row on update)
- all other SessionData fields
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from wolfharness.sessions.models import PendingDeferredCall, SessionData
from wolfharness_storage.protocols import CheckpointStore, SessionPersistence


if TYPE_CHECKING:
    from wolfharness_storage.sql_provider import SQLModelProvider


pytestmark = pytest.mark.unit


class TestRoundTripMigration:
    """Test full round-trip of SessionData through SQLModelProvider."""

    async def test_basic_round_trip(self, sql_model_provider: SQLModelProvider) -> None:
        """Basic save/load round-trip preserves all fields."""
        data = SessionData(
            session_id="rt-basic",
            agent_name="agent1",
            pool_id="pool1",
            project_id="proj1",
            cwd="/tmp",
            agent_type="native",
            metadata={"key": "value"},
        )
        await sql_model_provider.save_session(data)
        loaded = await sql_model_provider.load_session("rt-basic")

        assert loaded is not None
        assert loaded.session_id == "rt-basic"
        assert loaded.agent_name == "agent1"
        assert loaded.pool_id == "pool1"
        assert loaded.project_id == "proj1"
        assert loaded.cwd == "/tmp"
        assert loaded.agent_type == "native"
        assert loaded.metadata.get("key") == "value"
        assert loaded.status == "active"

    async def test_status_round_trip(self, sql_model_provider: SQLModelProvider) -> None:
        """All status values survive the round-trip."""
        for status in ("active", "checkpointed", "closed", "resuming"):
            data = SessionData(
                session_id=f"rt-status-{status}",
                agent_name="agent1",
                status=status,
            )
            await sql_model_provider.save_session(data)
            loaded = await sql_model_provider.load_session(f"rt-status-{status}")

            assert loaded is not None
            assert loaded.status == status, f"Status '{status}' not preserved"

    async def test_pending_deferred_calls_round_trip(
        self, sql_model_provider: SQLModelProvider
    ) -> None:
        """pending_deferred_calls are serialized in metadata_json and restored."""
        calls = [
            PendingDeferredCall(
                tool_call_id="tc-1",
                tool_name="elicit_tool",
                deferred_kind="elicitation",
                deferred_strategy="block",
                elicitation_message="Do you agree?",
            ),
            PendingDeferredCall(
                tool_call_id="tc-2",
                tool_name="bash",
                deferred_kind="unapproved",
                deferred_strategy="block",
            ),
        ]
        data = SessionData(
            session_id="rt-deferred",
            agent_name="agent1",
            pending_deferred_calls=calls,
        )
        await sql_model_provider.save_session(data)
        loaded = await sql_model_provider.load_session("rt-deferred")

        assert loaded is not None
        assert len(loaded.pending_deferred_calls) == 2
        assert loaded.pending_deferred_calls[0].tool_call_id == "tc-1"
        assert loaded.pending_deferred_calls[0].tool_name == "elicit_tool"
        assert loaded.pending_deferred_calls[0].deferred_kind == "elicitation"
        assert loaded.pending_deferred_calls[0].elicitation_message == "Do you agree?"
        assert loaded.pending_deferred_calls[1].tool_call_id == "tc-2"
        assert loaded.pending_deferred_calls[1].deferred_kind == "unapproved"

    async def test_empty_pending_deferred_calls(self, sql_model_provider: SQLModelProvider) -> None:
        """Empty pending_deferred_calls list works correctly."""
        data = SessionData(
            session_id="rt-empty-deferred",
            agent_name="agent1",
            pending_deferred_calls=[],
        )
        await sql_model_provider.save_session(data)
        loaded = await sql_model_provider.load_session("rt-empty-deferred")

        assert loaded is not None
        assert loaded.pending_deferred_calls == []

    async def test_update_clears_pending_deferred_calls(
        self, sql_model_provider: SQLModelProvider
    ) -> None:
        """Updating a session clears stale pending_deferred_calls."""
        calls = [
            PendingDeferredCall(
                tool_call_id="tc-clear",
                tool_name="bash",
                deferred_kind="unapproved",
                deferred_strategy="block",
            ),
        ]
        data = SessionData(
            session_id="rt-clear-deferred",
            agent_name="agent1",
            pending_deferred_calls=calls,
        )
        await sql_model_provider.save_session(data)

        # Update with empty list
        updated = data.model_copy(update={"pending_deferred_calls": []})
        await sql_model_provider.save_session(updated)
        loaded = await sql_model_provider.load_session("rt-clear-deferred")

        assert loaded is not None
        assert loaded.pending_deferred_calls == []

    async def test_checkpoint_data_preserved_on_save(
        self, sql_model_provider: SQLModelProvider
    ) -> None:
        """save_session preserves checkpoint_data from save_checkpoint."""
        session_id = "rt-checkpoint-preserve"

        # First, save a session
        data = SessionData(session_id=session_id, agent_name="agent1")
        await sql_model_provider.save_session(data)

        # Save checkpoint data
        await sql_model_provider.save_checkpoint(session_id, "[]", "[]")

        # Now update the session (e.g. status change)
        updated = data.model_copy(update={"status": "checkpointed"})
        await sql_model_provider.save_session(updated)

        # Verify checkpoint data still exists
        checkpoint = await sql_model_provider.load_checkpoint(session_id)
        assert checkpoint is not None
        assert checkpoint[0] == "[]"
        assert checkpoint[1] == "[]"

    async def test_upsert_no_duplicate(self, sql_model_provider: SQLModelProvider) -> None:
        """UPSERT does not create duplicate rows."""
        data = SessionData(session_id="rt-upsert", agent_name="agent1")
        await sql_model_provider.save_session(data)
        await sql_model_provider.save_session(data)
        await sql_model_provider.save_session(data)

        ids = await sql_model_provider.list_session_ids()
        assert ids.count("rt-upsert") == 1

    async def test_protocol_conformance(self, sql_model_provider: SQLModelProvider) -> None:
        """SQLModelProvider satisfies SessionPersistence + CheckpointStore Protocols."""
        assert isinstance(sql_model_provider, SessionPersistence)
        assert isinstance(sql_model_provider, CheckpointStore)
