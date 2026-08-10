"""Tests for checkpoint save/load/delete on StorageManager and MemoryStorageProvider."""

from __future__ import annotations

from datetime import timedelta

from pydantic_ai import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
import pytest

from wolfharness.sessions.models import PendingDeferredCall
from wolfharness.storage.manager import StorageManager
from wolfharness.storage.serialization import serialize_messages
from wolfharness_config.storage import MemoryStorageConfig, StorageConfig
from wolfharness_storage.memory_provider.provider import MemoryStorageProvider


pytestmark = pytest.mark.unit


# ── helpers ──────────────────────────────────────────────────────────────


def make_messages() -> list[ModelMessage]:
    """Create a small set of ModelMessages for testing."""
    return [
        ModelRequest(parts=[UserPromptPart(content="Hello")]),
        ModelResponse(
            parts=[
                TextPart(content="Hi there!"),
                ToolCallPart(
                    tool_name="bash",
                    args="ls",
                    tool_call_id="call_1",
                ),
            ]
        ),
    ]


def make_pending_calls() -> list[PendingDeferredCall]:
    """Create a list of pending deferred calls."""
    return [
        PendingDeferredCall(
            tool_call_id="call_1",
            tool_name="bash",
            deferred_kind="external",
            deferred_strategy="block",
        ),
    ]


# ── MemoryStorageProvider unit tests ─────────────────────────────────────


class TestMemoryProviderCheckpoints:
    """Checkpoint operations on the in-memory provider directly."""

    @pytest.fixture
    def provider(self) -> MemoryStorageProvider:
        """Fresh in-memory provider."""
        return MemoryStorageProvider()

    async def test_save_load_roundtrip(self, provider: MemoryStorageProvider) -> None:
        """Save and load checkpoint returns same data."""
        msgs_json = '["message json"]'
        calls_json = '["calls json"]'

        await provider.save_checkpoint("s1", msgs_json, calls_json)
        result = await provider.load_checkpoint("s1")

        assert result == (msgs_json, calls_json)

    async def test_load_missing_returns_none(self, provider: MemoryStorageProvider) -> None:
        """Loading non-existent checkpoint returns None."""
        assert await provider.load_checkpoint("nonexistent") is None

    async def test_overwrite_is_idempotent(self, provider: MemoryStorageProvider) -> None:
        """Second save overwrites cleanly."""
        await provider.save_checkpoint("s2", "old_msgs", "old_calls")
        await provider.save_checkpoint("s2", "new_msgs", "new_calls")

        result = await provider.load_checkpoint("s2")
        assert result == ("new_msgs", "new_calls")

    async def test_delete_removes(self, provider: MemoryStorageProvider) -> None:
        """Delete removes checkpoint data."""
        await provider.save_checkpoint("s3", "msgs", "calls")
        assert await provider.load_checkpoint("s3") is not None

        deleted = await provider.delete_checkpoint("s3")
        assert deleted is True
        assert await provider.load_checkpoint("s3") is None

    async def test_delete_missing_returns_false(self, provider: MemoryStorageProvider) -> None:
        """Deleting non-existent checkpoint returns False."""
        assert await provider.delete_checkpoint("nonexistent") is False

    async def test_cleanup_clears_checkpoints(self, provider: MemoryStorageProvider) -> None:
        """cleanup() removes all checkpoints."""
        await provider.save_checkpoint("s1", "m", "c")
        await provider.save_checkpoint("s2", "m", "c")

        provider.cleanup()

        assert await provider.load_checkpoint("s1") is None
        assert await provider.load_checkpoint("s2") is None


# ── StorageManager integration tests ─────────────────────────────────────


class TestStorageManagerCheckpoints:
    """Checkpoint operations through StorageManager with MemoryProvider."""

    @pytest.fixture
    def manager(self) -> StorageManager:
        """StorageManager backed by a single MemoryStorageProvider."""
        config = StorageConfig(providers=[MemoryStorageConfig()])
        return StorageManager(config)

    @pytest.fixture
    def messages(self) -> list[ModelMessage]:
        """Real ModelMessage instances for roundtrip."""
        return make_messages()

    @pytest.fixture
    def pending_calls(self) -> list[PendingDeferredCall]:
        """Real PendingDeferredCall instances."""
        return make_pending_calls()

    async def test_save_load_roundtrip(
        self,
        manager: StorageManager,
        messages: list[ModelMessage],
        pending_calls: list[PendingDeferredCall],
    ) -> None:
        """Full serialization roundtrip through StorageManager."""
        messages_json = serialize_messages(messages) or ""
        await manager.save_checkpoint("s1", messages_json, pending_calls)

        result = await manager.load_checkpoint("s1")
        assert result is not None
        loaded_msgs, loaded_calls = result

        assert len(loaded_msgs) == len(messages)
        assert len(loaded_calls) == len(pending_calls)
        assert loaded_calls[0].tool_call_id == pending_calls[0].tool_call_id
        assert loaded_calls[0].tool_name == pending_calls[0].tool_name

    async def test_load_missing_returns_none(self, manager: StorageManager) -> None:
        """Loading non-existent checkpoint returns None."""
        result = await manager.load_checkpoint("nonexistent")
        assert result is None

    async def test_overwrite_is_idempotent(
        self,
        manager: StorageManager,
        messages: list[ModelMessage],
        pending_calls: list[PendingDeferredCall],
    ) -> None:
        """Second save overwrites cleanly (no stale data)."""
        # First save with data
        messages_json = serialize_messages(messages) or ""
        await manager.save_checkpoint("s2", messages_json, pending_calls)

        # Second save with different data
        new_calls = [
            PendingDeferredCall(
                tool_call_id="call_2",
                tool_name="read",
                deferred_kind="unapproved",
                deferred_strategy="block",
                timeout=timedelta(seconds=30),
            ),
        ]
        await manager.save_checkpoint("s2", messages_json, new_calls)

        result = await manager.load_checkpoint("s2")
        assert result is not None
        _, loaded_calls = result
        assert len(loaded_calls) == 1
        assert loaded_calls[0].tool_call_id == "call_2"

    async def test_delete_checkpoint(
        self,
        manager: StorageManager,
        messages: list[ModelMessage],
        pending_calls: list[PendingDeferredCall],
    ) -> None:
        """delete_checkpoint removes stored data."""
        messages_json = serialize_messages(messages) or ""
        await manager.save_checkpoint("s3", messages_json, pending_calls)
        assert await manager.load_checkpoint("s3") is not None

        await manager.delete_checkpoint("s3")
        assert await manager.load_checkpoint("s3") is None

    async def test_delete_session_also_deletes_checkpoint(
        self,
        manager: StorageManager,
        messages: list[ModelMessage],
        pending_calls: list[PendingDeferredCall],
    ) -> None:
        """When a session is deleted, its checkpoint is also cleaned up."""
        messages_json = serialize_messages(messages) or ""
        await manager.save_checkpoint("s4", messages_json, pending_calls)
        assert await manager.load_checkpoint("s4") is not None

        # `delete_session` should call `delete_checkpoint` internally
        await manager.delete_session("s4")
        assert await manager.load_checkpoint("s4") is None

    async def test_empty_data_roundtrip(self, manager: StorageManager) -> None:
        """Checkpoint with empty messages and empty pending calls works."""
        await manager.save_checkpoint("s_empty", "", [])

        result = await manager.load_checkpoint("s_empty")
        assert result is not None
        loaded_msgs, loaded_calls = result
        assert loaded_msgs == []
        assert loaded_calls == []
