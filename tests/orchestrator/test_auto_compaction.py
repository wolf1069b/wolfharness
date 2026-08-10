"""Tests for auto-compaction before checkpoint.

Verifies that CheckpointManager.checkpoint() triggers compaction
when message count or serialized byte size exceeds configured thresholds,
and skips compaction when thresholds are not exceeded.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from pydantic_ai import ToolReturnPart
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
import pytest

from wolfharness.agents.native_agent.checkpoint import CheckpointManager
from wolfharness.storage.serialization import messages_adapter


pytestmark = pytest.mark.unit


def _make_small_messages(count: int) -> list[ModelMessage]:
    """Create simple text-only messages (each well under 10 bytes)."""
    messages: list[ModelMessage] = []
    for i in range(count):
        messages.append(ModelRequest(parts=[UserPromptPart(content=f"m{i}")]))
        messages.append(ModelResponse(parts=[TextPart(content=f"r{i}")]))
    return messages


def _make_messages_with_large_output(count: int) -> list[ModelMessage]:
    """Create messages where each has a tool output >1000 characters.

    These messages trigger TruncateToolOutputs during compaction, making it
    detectable by searching for the truncation suffix in the saved JSON.
    """
    messages: list[ModelMessage] = []
    for i in range(count):
        messages.append(ModelRequest(parts=[UserPromptPart(content=f"msg{i}")]))
        messages.append(
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        content="x" * 2000,
                        tool_name="test_tool",
                    )
                ]
            )
        )
        messages.append(ModelResponse(parts=[TextPart(content=f"resp{i}")]))
    return messages


@pytest.fixture
def mock_storage() -> AsyncMock:
    """Create a mock StorageManager with an async save_checkpoint."""
    mock = AsyncMock()
    mock.save_checkpoint = AsyncMock()
    return mock


class TestAutoCompaction:
    """Tests for compaction-trigger logic in CheckpointManager.checkpoint()."""

    async def test_below_threshold_skips_compaction(self, mock_storage: AsyncMock) -> None:
        """Compaction is NOT triggered when message count and byte size are below thresholds."""
        messages = _make_small_messages(2)
        original_json = messages_adapter.dump_json(messages).decode()

        mgr = CheckpointManager(
            mock_storage,
            compaction_threshold_messages=999_999,
            compaction_threshold_bytes=999_999_999,
        )
        await mgr.checkpoint(
            session_id="test-session",
            message_history=messages,
            pending_calls=[],
        )

        mock_storage.save_checkpoint.assert_called_once()
        saved_json = mock_storage.save_checkpoint.call_args.kwargs["messages_json"]
        # Without compaction the saved JSON should match the original serialization
        assert saved_json == original_json

    async def test_above_message_threshold_triggers_compaction(
        self, mock_storage: AsyncMock
    ) -> None:
        """Compaction IS triggered when message count exceeds threshold."""
        messages = _make_messages_with_large_output(50)
        original_json = messages_adapter.dump_json(messages).decode()

        mgr = CheckpointManager(
            mock_storage,
            compaction_threshold_messages=1,
            compaction_threshold_bytes=999_999_999,
        )
        await mgr.checkpoint(
            session_id="test-session",
            message_history=messages,
            pending_calls=[],
        )

        mock_storage.save_checkpoint.assert_called_once()
        saved_json = mock_storage.save_checkpoint.call_args.kwargs["messages_json"]

        # Compaction should have truncated the large tool outputs
        assert "... [truncated]" in saved_json, "Expected compaction to truncate tool outputs"
        assert saved_json != original_json, "Expected compacted JSON to differ from original"

    async def test_above_byte_threshold_triggers_compaction(self, mock_storage: AsyncMock) -> None:
        """Compaction IS triggered when serialized byte size exceeds threshold."""
        messages = _make_messages_with_large_output(5)
        original_json = messages_adapter.dump_json(messages).decode()

        # 10-byte threshold is trivially exceeded by any message
        mgr = CheckpointManager(
            mock_storage,
            compaction_threshold_messages=999_999,
            compaction_threshold_bytes=10,
        )
        await mgr.checkpoint(
            session_id="test-session",
            message_history=messages,
            pending_calls=[],
        )

        mock_storage.save_checkpoint.assert_called_once()
        saved_json = mock_storage.save_checkpoint.call_args.kwargs["messages_json"]

        # Compaction should have truncated the large tool outputs
        assert "... [truncated]" in saved_json, "Expected compaction to truncate tool outputs"
        assert saved_json != original_json, "Expected compacted JSON to differ from original"
