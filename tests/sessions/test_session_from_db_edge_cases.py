"""Edge case tests for _session_from_db().

Tests that _session_from_db() handles:
- checkpoint_data=None (no checkpoint saved)
- checkpoint_data={} (empty checkpoint dict)
- metadata_json without _pending_deferred_calls key
- metadata_json=None
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from wolfharness.sessions.models import SessionData
from wolfharness_storage.sql_provider.models import Conversation


if TYPE_CHECKING:
    from wolfharness_storage.sql_provider import SQLModelProvider


pytestmark = pytest.mark.unit


class TestSessionFromDbEdgeCases:
    """Test _session_from_db() with edge case inputs."""

    def _make_row(
        self,
        session_id: str = "edge-1",
        metadata_json: dict[str, Any] | None = None,
        checkpoint_data: dict[str, Any] | None = None,
        status: str = "active",
    ) -> Conversation:
        """Create a Conversation row with specific field values."""
        return Conversation(
            id=session_id,
            agent_name="agent1",
            metadata_json=metadata_json if metadata_json is not None else {},
            checkpoint_data=checkpoint_data,
            status=status,
        )

    def test_checkpoint_data_none(self, sql_model_provider: SQLModelProvider) -> None:
        """_session_from_db handles checkpoint_data=None."""
        row = self._make_row(checkpoint_data=None)
        result = sql_model_provider._session_from_db(row)

        assert result.session_id == "edge-1"
        assert result.status == "active"
        assert result.pending_deferred_calls == []

    def test_checkpoint_data_empty_dict(self, sql_model_provider: SQLModelProvider) -> None:
        """_session_from_db handles checkpoint_data={}."""
        row = self._make_row(checkpoint_data={})
        result = sql_model_provider._session_from_db(row)

        assert result.session_id == "edge-1"
        assert result.pending_deferred_calls == []

    def test_metadata_without_pending_deferred_calls(
        self, sql_model_provider: SQLModelProvider
    ) -> None:
        """_session_from_db handles metadata without _pending_deferred_calls key."""
        row = self._make_row(metadata_json={"title": "My Session", "custom": "data"})
        result = sql_model_provider._session_from_db(row)

        assert result.pending_deferred_calls == []
        assert result.metadata.get("custom") == "data"
        # Title should be merged into metadata
        assert result.metadata.get("title") == "My Session"

    def test_metadata_empty_dict(self, sql_model_provider: SQLModelProvider) -> None:
        """_session_from_db handles empty metadata_json."""
        row = self._make_row(metadata_json={})
        result = sql_model_provider._session_from_db(row)

        assert result.pending_deferred_calls == []
        assert result.metadata == {}

    def test_status_none_defaults_to_active(self, sql_model_provider: SQLModelProvider) -> None:
        """_session_from_db defaults status to 'active' when row.status is None."""
        row = self._make_row(status=None)  # type: ignore[arg-type]
        result = sql_model_provider._session_from_db(row)

        assert result.status == "active"

    def test_status_checkpointed(self, sql_model_provider: SQLModelProvider) -> None:
        """_session_from_db reads status='checkpointed' correctly."""
        row = self._make_row(status="checkpointed")
        result = sql_model_provider._session_from_db(row)

        assert result.status == "checkpointed"

    def test_title_overrides_metadata_title(self, sql_model_provider: SQLModelProvider) -> None:
        """Conversation.title overrides metadata_json['title']."""
        row = self._make_row(metadata_json={"title": "old title"})
        row.title = "new title"
        result = sql_model_provider._session_from_db(row)

        assert result.title == "new title"
        assert result.metadata.get("title") == "new title"

    async def test_save_load_with_no_metadata(self, sql_model_provider: SQLModelProvider) -> None:
        """Save and load a session with default (empty) metadata."""
        data = SessionData(session_id="edge-no-meta", agent_name="agent1")
        await sql_model_provider.save_session(data)
        loaded = await sql_model_provider.load_session("edge-no-meta")

        assert loaded is not None
        assert loaded.pending_deferred_calls == []
        assert loaded.status == "active"
