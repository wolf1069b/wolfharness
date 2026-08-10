"""Tests for checkpoint storage on SQLModelProvider.

Validates that save_checkpoint/load_checkpoint/delete_checkpoint persist
checkpoint data (messages_json + pending_calls_json) in the SQL database and
that the data survives connection close/reopen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from wolfharness.sessions import SessionData
from wolfharness_config.storage import SQLStorageConfig
from wolfharness_storage.sql_provider import SQLModelProvider


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def provider(tmp_path: Path) -> SQLModelProvider:
    """Create a SQL provider with temp database (no auto-migration to avoid Alembic multi-heads)."""
    db_path = tmp_path / "test_checkpoints.db"
    config = SQLStorageConfig(url=f"sqlite:///{db_path}", auto_migration=False)
    return SQLModelProvider(config)


class TestSQLCheckpoint:
    """Tests for checkpoint save/load/delete on SQLModelProvider."""

    @pytest.mark.unit
    async def test_save_and_load_checkpoint(self, provider: SQLModelProvider) -> None:
        """save_checkpoint stores JSON messages and pending calls; load_checkpoint returns them."""
        async with provider:
            await provider.save_session(SessionData(session_id="s1", agent_name="test_agent"))
            await provider.save_checkpoint(
                "s1",
                '[{"role":"user","content":"hello"}]',
                '[{"tool_call_id":"tc1","tool_name":"bash"}]',
            )
            result = await provider.load_checkpoint("s1")

        assert result is not None
        messages_json, pending_calls_json = result
        assert messages_json == '[{"role":"user","content":"hello"}]'
        assert pending_calls_json == '[{"tool_call_id":"tc1","tool_name":"bash"}]'

    @pytest.mark.unit
    async def test_load_checkpoint_nonexistent(self, provider: SQLModelProvider) -> None:
        """load_checkpoint returns None for a session with no checkpoint."""
        async with provider:
            await provider.save_session(SessionData(session_id="s1", agent_name="test_agent"))
            result = await provider.load_checkpoint("s1")

        assert result is None

    @pytest.mark.unit
    async def test_load_checkpoint_unknown_session(self, provider: SQLModelProvider) -> None:
        """load_checkpoint returns None for a session that doesn't exist in the database."""
        async with provider:
            result = await provider.load_checkpoint("nonexistent")

        assert result is None

    @pytest.mark.unit
    async def test_delete_checkpoint(self, provider: SQLModelProvider) -> None:
        """delete_checkpoint removes stored checkpoint data from the database."""
        async with provider:
            await provider.save_session(SessionData(session_id="s1", agent_name="test_agent"))
            await provider.save_checkpoint("s1", "[]", "[]")

            deleted = await provider.delete_checkpoint("s1")
            assert deleted is True

            result = await provider.load_checkpoint("s1")
            assert result is None

    @pytest.mark.unit
    async def test_delete_checkpoint_nonexistent(self, provider: SQLModelProvider) -> None:
        """delete_checkpoint returns False for a session with no checkpoint."""
        async with provider:
            await provider.save_session(SessionData(session_id="s1", agent_name="test_agent"))
            deleted = await provider.delete_checkpoint("s1")

        assert deleted is False

    @pytest.mark.unit
    async def test_checkpoint_survives_close_reopen(self, provider: SQLModelProvider) -> None:
        """Checkpoint data persists after closing and reopening the database connection."""
        async with provider:
            await provider.save_session(SessionData(session_id="s1", agent_name="test_agent"))
            await provider.save_checkpoint(
                "s1",
                '[{"role":"user","content":"survive test"}]',
                '[{"tool_call_id":"tc42"}]',
            )

        # Reopen with a fresh provider (new connection, same file)
        db_path = provider.engine.url.database
        new_provider = SQLModelProvider(
            SQLStorageConfig(url=f"sqlite:///{db_path}", auto_migration=False)
        )
        async with new_provider:
            result = await new_provider.load_checkpoint("s1")

        assert result is not None
        messages_json, pending_calls_json = result
        assert messages_json == '[{"role":"user","content":"survive test"}]'
        assert pending_calls_json == '[{"tool_call_id":"tc42"}]'

    @pytest.mark.unit
    async def test_overwrite_checkpoint(self, provider: SQLModelProvider) -> None:
        """save_checkpoint overwrites existing checkpoint data for the same session."""
        async with provider:
            await provider.save_session(SessionData(session_id="s1", agent_name="test_agent"))
            await provider.save_checkpoint("s1", "old", "old_calls")
            await provider.save_checkpoint("s1", "new", "new_calls")

            result = await provider.load_checkpoint("s1")

        assert result is not None
        messages_json, pending_calls_json = result
        assert messages_json == "new"
        assert pending_calls_json == "new_calls"

    @pytest.mark.unit
    async def test_session_delete_cleans_checkpoint(self, provider: SQLModelProvider) -> None:
        """Deleting a session also removes its checkpoint data from the database.

        Since checkpoint_data is a column on the conversation table,
        deleting the conversation row naturally removes the checkpoint data.
        """
        async with provider:
            await provider.save_session(SessionData(session_id="s1", agent_name="test_agent"))
            await provider.save_checkpoint("s1", "[]", "[]")

            deleted = await provider.delete_session("s1")
            assert deleted is True

            result = await provider.load_checkpoint("s1")
            assert result is None

    @pytest.mark.unit
    async def test_save_checkpoint_without_prior_session(self, provider: SQLModelProvider) -> None:
        """save_checkpoint succeeds even when no Conversation record exists.

        This is the ACP session scenario: SessionStore uses a separate Session table,
        and StorageManager.save_session() is never called. The SQL provider should
        create a minimal Conversation record on demand (upsert pattern) rather than
        raising ValueError.
        """
        async with provider:
            # No save_session() call — directly save checkpoint
            await provider.save_checkpoint(
                "s-no-conv",
                '[{"role":"user","content":"hello"}]',
                '[{"tool_call_id":"tc1","tool_name":"bash"}]',
            )
            result = await provider.load_checkpoint("s-no-conv")

        assert result is not None
        messages_json, pending_calls_json = result
        assert messages_json == '[{"role":"user","content":"hello"}]'
        assert pending_calls_json == '[{"tool_call_id":"tc1","tool_name":"bash"}]'

    @pytest.mark.unit
    async def test_save_checkpoint_without_session_then_overwrite(
        self, provider: SQLModelProvider
    ) -> None:
        """save_checkpoint can overwrite a checkpoint created without prior session."""
        async with provider:
            await provider.save_checkpoint("s-upsert", "old", "old_calls")
            await provider.save_checkpoint("s-upsert", "new", "new_calls")
            result = await provider.load_checkpoint("s-upsert")

        assert result is not None
        messages_json, pending_calls_json = result
        assert messages_json == "new"
        assert pending_calls_json == "new_calls"
