"""Tests for ISP Protocol decomposition and StorageProviderAdapter."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from wolfharness_storage.adapter import StorageProviderAdapter
from wolfharness_storage.base import StorageProvider
from wolfharness_storage.protocols import (
    CheckpointStore,
    CommandLog,
    MessagePersistence,
    ProjectStoreProtocol,
    SessionMetadata,
    SessionPersistence,
    StatsAggregator,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Protocol isinstance checks for StorageProviderAdapter
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock StorageProvider with all methods as AsyncMocks."""
    provider = MagicMock(spec=StorageProvider)
    # async methods need to be AsyncMock
    for name in [
        "save_session",
        "load_session",
        "delete_session",
        "list_session_ids",
        "load_sessions_batch",
        "update_sdk_session_id",
        "log_message",
        "get_session_messages",
        "get_message",
        "get_message_ancestry",
        "filter_messages",
        "delete_session_messages",
        "truncate_messages",
        "fork_conversation",
        "log_session",
        "update_session_title",
        "get_session_title",
        "get_sessions",
        "get_filtered_conversations",
        "get_session_counts",
        "get_session_stats",
        "log_command",
        "get_commands",
        "save_project",
        "get_project",
        "get_project_by_name",
        "get_project_by_worktree",
        "list_projects",
        "delete_project",
        "touch_project",
        "save_checkpoint",
        "load_checkpoint",
        "delete_checkpoint",
        "reset",
    ]:
        setattr(provider, name, AsyncMock())
    # sync method
    provider.aggregate_stats = MagicMock(return_value={})
    return provider


@pytest.fixture
def adapter(mock_provider: MagicMock) -> StorageProviderAdapter:
    return StorageProviderAdapter(mock_provider)


class TestAdapterProtocolConformance:
    """Test that StorageProviderAdapter passes isinstance for all 7 Protocols."""

    def test_isinstance_session_persistence(self, adapter: StorageProviderAdapter) -> None:
        assert isinstance(adapter, SessionPersistence)

    def test_isinstance_message_persistence(self, adapter: StorageProviderAdapter) -> None:
        assert isinstance(adapter, MessagePersistence)

    def test_isinstance_session_metadata(self, adapter: StorageProviderAdapter) -> None:
        assert isinstance(adapter, SessionMetadata)

    def test_isinstance_command_log(self, adapter: StorageProviderAdapter) -> None:
        assert isinstance(adapter, CommandLog)

    def test_isinstance_project_store(self, adapter: StorageProviderAdapter) -> None:
        assert isinstance(adapter, ProjectStoreProtocol)

    def test_isinstance_checkpoint_store(self, adapter: StorageProviderAdapter) -> None:
        assert isinstance(adapter, CheckpointStore)

    def test_isinstance_stats_aggregator(self, adapter: StorageProviderAdapter) -> None:
        assert isinstance(adapter, StatsAggregator)

    def test_isinstance_all_seven(self, adapter: StorageProviderAdapter) -> None:
        protocols = [
            SessionPersistence,
            MessagePersistence,
            SessionMetadata,
            CommandLog,
            ProjectStoreProtocol,
            CheckpointStore,
            StatsAggregator,
        ]
        for p in protocols:
            assert isinstance(adapter, p), f"Adapter failed isinstance check for {p.__name__}"


# ---------------------------------------------------------------------------
# Adapter delegation tests
# ---------------------------------------------------------------------------


class TestAdapterDelegation:
    """Test that StorageProviderAdapter delegates correctly."""

    async def test_save_session_delegates(
        self, adapter: StorageProviderAdapter, mock_provider: MagicMock
    ) -> None:
        data = MagicMock()
        await adapter.save_session(data)
        mock_provider.save_session.assert_called_once_with(data)

    async def test_load_session_delegates(
        self, adapter: StorageProviderAdapter, mock_provider: MagicMock
    ) -> None:
        mock_provider.load_session.return_value = None
        result = await adapter.load_session("test-id")
        mock_provider.load_session.assert_called_once_with("test-id")
        assert result is None

    async def test_delete_session_delegates(
        self, adapter: StorageProviderAdapter, mock_provider: MagicMock
    ) -> None:
        mock_provider.delete_session.return_value = True
        result = await adapter.delete_session("test-id")
        mock_provider.delete_session.assert_called_once_with("test-id")
        assert result is True

    async def test_list_session_ids_delegates(
        self, adapter: StorageProviderAdapter, mock_provider: MagicMock
    ) -> None:
        mock_provider.list_session_ids.return_value = ["id1", "id2"]
        result = await adapter.list_session_ids(pool_id="pool1", agent_name="agent1")
        mock_provider.list_session_ids.assert_called_once_with(
            pool_id="pool1", agent_name="agent1", cwd=None
        )
        assert result == ["id1", "id2"]

    async def test_save_checkpoint_delegates(
        self, adapter: StorageProviderAdapter, mock_provider: MagicMock
    ) -> None:
        await adapter.save_checkpoint("sid", "msgs", "calls")
        mock_provider.save_checkpoint.assert_called_once_with("sid", "msgs", "calls")

    async def test_load_checkpoint_delegates(
        self, adapter: StorageProviderAdapter, mock_provider: MagicMock
    ) -> None:
        mock_provider.load_checkpoint.return_value = ("msgs", "calls")
        result = await adapter.load_checkpoint("sid")
        assert result == ("msgs", "calls")

    def test_aggregate_stats_delegates(
        self, adapter: StorageProviderAdapter, mock_provider: MagicMock
    ) -> None:
        rows: list[tuple[str | None, str | None, Any, Any]] = []
        adapter.aggregate_stats(rows, "agent")
        mock_provider.aggregate_stats.assert_called_once_with(rows, "agent")


# ---------------------------------------------------------------------------
# Provider Protocol conformance
# ---------------------------------------------------------------------------


class TestProviderProtocolConformance:
    """Test that concrete providers pass isinstance checks."""

    def test_memory_storage_provider_satisfies_session_persistence(self) -> None:
        from wolfharness_storage.memory_provider.provider import MemoryStorageProvider

        provider = MemoryStorageProvider()
        assert isinstance(provider, SessionPersistence)

    def test_memory_storage_provider_satisfies_checkpoint_store(self) -> None:
        from wolfharness_storage.memory_provider.provider import MemoryStorageProvider

        provider = MemoryStorageProvider()
        assert isinstance(provider, CheckpointStore)

    def test_memory_storage_provider_isinstance(self) -> None:
        from wolfharness_storage.memory_provider.provider import MemoryStorageProvider

        provider = MemoryStorageProvider()
        protocols = [
            SessionPersistence,
            MessagePersistence,
            SessionMetadata,
            CommandLog,
            ProjectStoreProtocol,
            CheckpointStore,
            StatsAggregator,
        ]
        for p in protocols:
            assert isinstance(provider, p), (
                f"MemoryStorageProvider failed isinstance for {p.__name__}"
            )
