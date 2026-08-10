"""Checkpoint save/load round-trip tests through SQLModelProvider.

Verifies:
- save_checkpoint → save_session (metadata update) → load_checkpoint preserves data
- delete_checkpoint works
- load_checkpoint returns None when no checkpoint exists
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from wolfharness.sessions.models import SessionData


if TYPE_CHECKING:
    from wolfharness_storage.sql_provider import SQLModelProvider


pytestmark = pytest.mark.unit


class TestCheckpointRoundTrip:
    """Test checkpoint save/load/delete through SQLModelProvider."""

    async def test_save_load_checkpoint(self, sql_model_provider: SQLModelProvider) -> None:
        """Basic checkpoint save/load round-trip."""
        session_id = "ckpt-basic"
        messages_json = '[{"role": "user", "content": "hello"}]'
        pending_calls_json = '[{"tool_call_id": "tc-1"}]'

        await sql_model_provider.save_checkpoint(session_id, messages_json, pending_calls_json)
        loaded = await sql_model_provider.load_checkpoint(session_id)

        assert loaded is not None
        assert loaded[0] == messages_json
        assert loaded[1] == pending_calls_json

    async def test_checkpoint_survives_session_update(
        self, sql_model_provider: SQLModelProvider
    ) -> None:
        """Checkpoint data survives a save_session call (metadata update)."""
        session_id = "ckpt-survive"

        # Save initial session
        data = SessionData(session_id=session_id, agent_name="agent1")
        await sql_model_provider.save_session(data)

        # Save checkpoint
        messages_json = '[{"role": "user", "content": "checkpoint me"}]'
        pending_calls_json = '[{"tool_call_id": "tc-survive"}]'
        await sql_model_provider.save_checkpoint(session_id, messages_json, pending_calls_json)

        # Update session (e.g. status → checkpointed)
        updated = data.model_copy(update={"status": "checkpointed"})
        await sql_model_provider.save_session(updated)

        # Checkpoint should still be there
        loaded = await sql_model_provider.load_checkpoint(session_id)
        assert loaded is not None
        assert loaded[0] == messages_json
        assert loaded[1] == pending_calls_json

    async def test_load_checkpoint_none(self, sql_model_provider: SQLModelProvider) -> None:
        """load_checkpoint returns None when no checkpoint exists."""
        result = await sql_model_provider.load_checkpoint("nonexistent-session")
        assert result is None

    async def test_delete_checkpoint(self, sql_model_provider: SQLModelProvider) -> None:
        """delete_checkpoint removes checkpoint data."""
        session_id = "ckpt-delete"

        await sql_model_provider.save_checkpoint(session_id, "[]", "[]")
        assert await sql_model_provider.load_checkpoint(session_id) is not None

        deleted = await sql_model_provider.delete_checkpoint(session_id)
        assert deleted is True

        assert await sql_model_provider.load_checkpoint(session_id) is None

    async def test_delete_checkpoint_not_found(self, sql_model_provider: SQLModelProvider) -> None:
        """delete_checkpoint returns False when no checkpoint exists."""
        deleted = await sql_model_provider.delete_checkpoint("nonexistent")
        assert deleted is False

    async def test_checkpoint_without_session_record(
        self, sql_model_provider: SQLModelProvider
    ) -> None:
        """save_checkpoint creates a minimal record if none exists."""
        session_id = "ckpt-no-session"

        await sql_model_provider.save_checkpoint(session_id, "[]", "[]")
        loaded = await sql_model_provider.load_checkpoint(session_id)

        assert loaded is not None
        assert loaded[0] == "[]"
        assert loaded[1] == "[]"
