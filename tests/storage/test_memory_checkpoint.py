"""Tests for checkpoint storage on MemoryStorageProvider."""

from __future__ import annotations

import json

import pytest

from wolfharness_storage.memory_provider.provider import MemoryStorageProvider


@pytest.fixture
def store() -> MemoryStorageProvider:
    """Return a fresh MemoryStorageProvider instance."""
    return MemoryStorageProvider()


class TestMemoryCheckpoint:
    """Tests for checkpoint save/load/delete operations."""

    @pytest.mark.unit
    async def test_save_and_load_checkpoint(self, store: MemoryStorageProvider) -> None:
        """save_checkpoint stores JSON messages and pending calls; load_checkpoint returns them."""
        session_id = "s1"
        messages_json = '[{"role":"user","content":"hello"}]'
        pending_calls: list[dict[str, object]] = [{"tool_call_id": "tc1", "tool_name": "bash"}]

        await store.save_checkpoint(session_id, messages_json, json.dumps(pending_calls))
        result = await store.load_checkpoint(session_id)

        assert result is not None
        assert result[0] == messages_json
        assert json.loads(result[1]) == pending_calls

    @pytest.mark.unit
    async def test_load_checkpoint_nonexistent(self, store: MemoryStorageProvider) -> None:
        """load_checkpoint returns None for a session that has no checkpoint."""
        result = await store.load_checkpoint("nonexistent")
        assert result is None

    @pytest.mark.unit
    async def test_delete_checkpoint(self, store: MemoryStorageProvider) -> None:
        """delete_checkpoint removes stored checkpoint data."""
        session_id = "s1"
        await store.save_checkpoint(session_id, "[]", json.dumps([]))

        deleted = await store.delete_checkpoint(session_id)
        assert deleted is True

        result = await store.load_checkpoint(session_id)
        assert result is None

    @pytest.mark.unit
    async def test_delete_checkpoint_nonexistent(self, store: MemoryStorageProvider) -> None:
        """delete_checkpoint returns False for a session with no checkpoint."""
        deleted = await store.delete_checkpoint("nonexistent")
        assert deleted is False

    @pytest.mark.unit
    async def test_session_delete_cleans_checkpoint(self, store: MemoryStorageProvider) -> None:
        """Deleting a session also cleans up its checkpoint data."""
        from wolfharness.sessions import SessionData

        session_id = "s1"
        data = SessionData(session_id=session_id, agent_name="test_agent")
        await store.save_session(data)

        # Save a checkpoint
        await store.save_checkpoint(session_id, "[]", json.dumps([]))

        # Delete the session
        deleted = await store.delete_session(session_id)
        assert deleted is True

        # Checkpoint should be cleaned up too
        result = await store.load_checkpoint(session_id)
        assert result is None

    @pytest.mark.unit
    async def test_overwrite_checkpoint(self, store: MemoryStorageProvider) -> None:
        """save_checkpoint overwrites existing checkpoint data for the same session."""
        session_id = "s1"

        await store.save_checkpoint(session_id, "old", json.dumps([]))
        await store.save_checkpoint(session_id, "new", json.dumps([{"tool_call_id": "tc2"}]))

        result = await store.load_checkpoint(session_id)
        assert result is not None
        assert result[0] == "new"
        assert json.loads(result[1]) == [{"tool_call_id": "tc2"}]
