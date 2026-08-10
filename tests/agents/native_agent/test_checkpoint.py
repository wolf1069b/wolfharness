"""Tests for CheckpointManager and CheckpointData.

Verifies checkpoint serialization, roundtrip integrity, mid-stream prevention,
storage failure handling, and event emission.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
import pytest

from wolfharness.agents.native_agent.checkpoint import (
    CheckpointData,
    CheckpointManager,
    MidStreamCheckpointError,
)
from wolfharness.sessions.models import PendingDeferredCall
from wolfharness.storage.manager import StorageManager


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from wolfharness.agents.events.events import ToolCallDeferredEvent


class TestCheckpointData:
    """Tests for CheckpointData dataclass."""

    def test_create_empty(self) -> None:
        """CheckpointData can be created with empty fields."""
        data = CheckpointData()
        assert data.message_history == []
        assert data.pending_calls == []

    def test_create_with_data(self) -> None:
        """CheckpointData holds message_history and pending_calls."""
        msgs: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="hello")]),
            ModelResponse(parts=[TextPart(content="hi")]),
        ]
        calls = [
            PendingDeferredCall(
                tool_call_id="tc-1",
                tool_name="bash",
                deferred_kind="external",
                deferred_strategy="block",
            ),
        ]
        data = CheckpointData(message_history=msgs, pending_calls=calls)
        assert len(data.message_history) == 2
        assert len(data.pending_calls) == 1
        assert data.pending_calls[0].tool_call_id == "tc-1"


@pytest.fixture
def storage_manager() -> StorageManager:
    """Create a StorageManager with no providers (for unit testing)."""
    from wolfharness_config.storage import StorageConfig

    config = StorageConfig(providers=[])
    return StorageManager(config=config)


@pytest.fixture
def sample_messages() -> list[ModelMessage]:
    """Create sample message_history with a ToolCallPart pending."""
    return [
        ModelRequest(parts=[UserPromptPart(content="run a script")]),
        ModelResponse(
            parts=[
                TextPart(content="I'll run that script for you."),
                ToolCallPart(
                    tool_name="bash",
                    args={"command": "./deploy.sh"},
                    tool_call_id="tc-deploy-001",
                ),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="read",
                    content="file contents here",
                    tool_call_id="tc-read-001",
                )
            ]
        ),
    ]


@pytest.fixture
def sample_pending_calls() -> list[PendingDeferredCall]:
    """Create sample pending deferred calls."""
    return [
        PendingDeferredCall(
            tool_call_id="tc-deploy-001",
            tool_name="bash",
            deferred_kind="external",
            deferred_strategy="block",
        ),
    ]


class TestCheckpointManagerRoundtrip:
    """Tests for checkpoint save/load roundtrip."""

    @pytest.mark.anyio
    async def test_roundtrip_save_and_load(
        self,
        storage_manager: StorageManager,
        sample_messages: list[ModelMessage],
        sample_pending_calls: list[PendingDeferredCall],
    ) -> None:
        """Checkpoint saves messages + pending calls, load returns correct CheckpointData."""
        manager = CheckpointManager(storage_manager)

        # Setup mock provider that can save/load checkpoints
        mock_provider = MagicMock()
        mock_provider.can_load_history = True
        mock_provider.can_store_projects = True
        mock_provider.save_checkpoint = AsyncMock()
        mock_provider.load_checkpoint = AsyncMock(
            return_value=(
                CheckpointManager._serialize_messages(sample_messages),
                CheckpointManager._serialize_pending_calls(sample_pending_calls),
            )
        )
        storage_manager.providers = [mock_provider]  # type: ignore[assignment]

        session_id = "session-roundtrip-001"

        # Save checkpoint
        await manager.checkpoint(
            session_id=session_id,
            message_history=sample_messages,
            pending_calls=sample_pending_calls,
        )

        # Verify save was called
        assert mock_provider.save_checkpoint.call_count >= 1

        # Load checkpoint
        data = await manager.load_checkpoint(session_id)

        assert data is not None
        assert len(data.message_history) == 3
        assert len(data.pending_calls) == 1
        assert data.pending_calls[0].tool_call_id == "tc-deploy-001"
        assert data.pending_calls[0].tool_name == "bash"

    @pytest.mark.anyio
    async def test_load_checkpoint_not_found(
        self,
        storage_manager: StorageManager,
    ) -> None:
        """load_checkpoint returns None when no checkpoint exists."""
        manager = CheckpointManager(storage_manager)

        mock_provider = MagicMock()
        mock_provider.can_load_history = True
        mock_provider.load_checkpoint = AsyncMock(return_value=None)
        storage_manager.providers = [mock_provider]  # type: ignore[assignment]

        data = await manager.load_checkpoint("nonexistent-session")
        assert data is None

    @pytest.mark.anyio
    async def test_roundtrip_empty_data(
        self,
        storage_manager: StorageManager,
    ) -> None:
        """Checkpoint handles empty messages and empty pending calls."""
        manager = CheckpointManager(storage_manager)

        mock_provider = MagicMock()
        mock_provider.can_load_history = True
        mock_provider.save_checkpoint = AsyncMock()
        mock_provider.load_checkpoint = AsyncMock(
            return_value=(
                CheckpointManager._serialize_messages([]),
                "[]",
            )
        )
        storage_manager.providers = [mock_provider]  # type: ignore[assignment]

        session_id = "session-empty-001"

        await manager.checkpoint(
            session_id=session_id,
            message_history=[],
            pending_calls=[],
        )

        data = await manager.load_checkpoint(session_id)
        assert data is not None
        assert data.message_history == []
        assert data.pending_calls == []


class TestCheckpointMidStreamPrevention:
    """Tests for mid-stream checkpoint prevention."""

    @pytest.mark.anyio
    async def test_mid_stream_raises_error(
        self,
        storage_manager: StorageManager,
        sample_messages: list[ModelMessage],
        sample_pending_calls: list[PendingDeferredCall],
    ) -> None:
        """Checkpoint raises MidStreamCheckpointError when called mid-stream."""
        manager = CheckpointManager(storage_manager)
        manager._is_mid_stream = True

        with pytest.raises(MidStreamCheckpointError, match="mid-stream"):
            await manager.checkpoint(
                session_id="session-001",
                message_history=sample_messages,
                pending_calls=sample_pending_calls,
            )

    @pytest.mark.anyio
    async def test_clean_boundary_succeeds(
        self,
        storage_manager: StorageManager,
        sample_messages: list[ModelMessage],
        sample_pending_calls: list[PendingDeferredCall],
    ) -> None:
        """Checkpoint succeeds at clean boundary (not mid-stream)."""
        manager = CheckpointManager(storage_manager)
        manager._is_mid_stream = False

        mock_provider = MagicMock()
        mock_provider.can_load_history = True
        mock_provider.save_checkpoint = AsyncMock()
        storage_manager.providers = [mock_provider]  # type: ignore[assignment]

        # Should not raise
        await manager.checkpoint(
            session_id="session-001",
            message_history=sample_messages,
            pending_calls=sample_pending_calls,
        )

        assert mock_provider.save_checkpoint.call_count >= 1

    def test_set_mid_stream_flag(self, storage_manager: StorageManager) -> None:
        """Can set and clear mid-stream flag."""
        manager = CheckpointManager(storage_manager)
        assert manager._is_mid_stream is False
        manager._is_mid_stream = True
        assert manager._is_mid_stream is True
        manager._is_mid_stream = False
        assert manager._is_mid_stream is False


class TestCheckpointStorageFailure:
    """Tests for storage failure handling.

    StorageManager.save_checkpoint() catches exceptions internally (logged,
    not re-raised). The CheckpointManager's checkpoint() call completes
    without error even when underlying storage fails. The key invariant is
    that the manager's internal state is preserved, so the agent can
    continue running.
    """

    @pytest.mark.anyio
    async def test_storage_failure_does_not_crash(
        self,
        storage_manager: StorageManager,
        sample_messages: list[ModelMessage],
        sample_pending_calls: list[PendingDeferredCall],
    ) -> None:
        """Storage write failure is caught internally; checkpoint call completes."""
        manager = CheckpointManager(storage_manager)

        mock_provider = MagicMock()
        mock_provider.can_load_history = True
        mock_provider.save_checkpoint = AsyncMock(side_effect=RuntimeError("disk full"))
        storage_manager.providers = [mock_provider]  # type: ignore[assignment]

        # Should NOT raise — StorageManager catches errors internally
        await manager.checkpoint(
            session_id="session-001",
            message_history=sample_messages,
            pending_calls=sample_pending_calls,
        )

    @pytest.mark.anyio
    async def test_storage_failure_preserves_state(
        self,
        storage_manager: StorageManager,
        sample_messages: list[ModelMessage],
        sample_pending_calls: list[PendingDeferredCall],
    ) -> None:
        """Storage failure does NOT clear internal state (agent continues)."""
        manager = CheckpointManager(storage_manager)

        mock_provider = MagicMock()
        mock_provider.can_load_history = True
        mock_provider.save_checkpoint = AsyncMock(side_effect=RuntimeError("disk full"))
        storage_manager.providers = [mock_provider]  # type: ignore[assignment]

        await manager.checkpoint(
            session_id="session-001",
            message_history=sample_messages,
            pending_calls=sample_pending_calls,
        )

        # Manager state should be unchanged (agent continues running)
        assert manager._is_mid_stream is False

    @pytest.mark.anyio
    async def test_not_implemented_provider_skipped(
        self,
        storage_manager: StorageManager,
        sample_messages: list[ModelMessage],
        sample_pending_calls: list[PendingDeferredCall],
    ) -> None:
        """Providers that raise NotImplementedError for save_checkpoint are skipped."""
        manager = CheckpointManager(storage_manager)

        mock_provider = MagicMock()
        mock_provider.can_load_history = True
        mock_provider.save_checkpoint = AsyncMock(side_effect=NotImplementedError)
        storage_manager.providers = [mock_provider]  # type: ignore[assignment]

        # Should NOT raise — NotImplementedError is caught internally
        await manager.checkpoint(
            session_id="session-001",
            message_history=sample_messages,
            pending_calls=sample_pending_calls,
        )


class TestCheckpointEventEmission:
    """Tests for ToolCallDeferredEvent emission."""

    @pytest.mark.anyio
    async def test_checkpoint_emits_deferred_events(
        self,
        storage_manager: StorageManager,
        sample_messages: list[ModelMessage],
        sample_pending_calls: list[PendingDeferredCall],
    ) -> None:
        """Checkpoint emits ToolCallDeferredEvent per pending call."""
        manager = CheckpointManager(storage_manager)

        mock_provider = MagicMock()
        mock_provider.can_load_history = True
        mock_provider.save_checkpoint = AsyncMock()
        storage_manager.providers = [mock_provider]  # type: ignore[assignment]

        # Collect emitted events
        emitted_events: list[ToolCallDeferredEvent] = []

        async def collect_event(event: ToolCallDeferredEvent) -> None:
            emitted_events.append(event)

        # Monkey-patch the emit method
        manager._emit_deferred_event = collect_event  # type: ignore[assignment]

        await manager.checkpoint(
            session_id="session-event-001",
            message_history=sample_messages,
            pending_calls=sample_pending_calls,
        )

        assert len(emitted_events) == 1
        event = emitted_events[0]
        assert event.tool_call_id == "tc-deploy-001"
        assert event.tool_name == "bash"
        assert event.deferred_strategy == "block"
        assert event.status == "pending"
        assert event.session_id == "session-event-001"

    @pytest.mark.anyio
    async def test_multiple_pending_calls_emit_multiple_events(
        self,
        storage_manager: StorageManager,
    ) -> None:
        """Multiple pending calls produce multiple events."""
        manager = CheckpointManager(storage_manager)

        mock_provider = MagicMock()
        mock_provider.can_load_history = True
        mock_provider.save_checkpoint = AsyncMock()
        storage_manager.providers = [mock_provider]  # type: ignore[assignment]

        pending_calls = [
            PendingDeferredCall(
                tool_call_id="tc-1",
                tool_name="bash",
                deferred_kind="external",
                deferred_strategy="block",
            ),
            PendingDeferredCall(
                tool_call_id="tc-2",
                tool_name="subagent",
                deferred_kind="external",
                deferred_strategy="block",
            ),
        ]

        emitted_events: list[ToolCallDeferredEvent] = []

        async def collect_event(event: ToolCallDeferredEvent) -> None:
            emitted_events.append(event)

        manager._emit_deferred_event = collect_event  # type: ignore[assignment]

        await manager.checkpoint(
            session_id="session-multi-001",
            message_history=[],
            pending_calls=pending_calls,
        )

        assert len(emitted_events) == 2
        assert {e.tool_call_id for e in emitted_events} == {"tc-1", "tc-2"}
        assert all(e.status == "pending" for e in emitted_events)


class TestCheckpointSerialization:
    """Tests for message serialization helpers."""

    def test_serialize_messages_roundtrip(
        self,
        sample_messages: list[ModelMessage],
    ) -> None:
        """Serialized messages deserialize back to original."""
        json_str = CheckpointManager._serialize_messages(sample_messages)
        assert json_str is not None

        from wolfharness.storage.serialization import deserialize_messages

        restored = deserialize_messages(json_str)
        assert len(restored) == len(sample_messages)

    def test_serialize_empty_messages(self) -> None:
        """Empty message list serializes to None."""
        result = CheckpointManager._serialize_messages([])
        assert result is None

    def test_serialize_pending_calls(self) -> None:
        """Pending calls serialize to JSON string."""
        calls = [
            PendingDeferredCall(
                tool_call_id="tc-1",
                tool_name="bash",
                deferred_kind="external",
                deferred_strategy="block",
            ),
        ]
        json_str = CheckpointManager._serialize_pending_calls(calls)
        assert json_str is not None
        assert "tc-1" in json_str
        assert "bash" in json_str

    def test_serialize_empty_pending_calls(self) -> None:
        """Empty pending calls serialize to '[]'."""
        result = CheckpointManager._serialize_pending_calls([])
        assert result == "[]"


class TestCheckpointAtomicConsistency:
    """Tests for atomic checkpoint consistency."""

    @pytest.mark.anyio
    async def test_partial_save_error_handled(
        self,
        storage_manager: StorageManager,
        sample_messages: list[ModelMessage],
        sample_pending_calls: list[PendingDeferredCall],
    ) -> None:
        """Storage failure during save is logged, checkpoint call completes cleanly."""
        manager = CheckpointManager(storage_manager)

        mock_provider = MagicMock()
        mock_provider.can_load_history = True
        mock_provider.save_checkpoint = AsyncMock(side_effect=RuntimeError("connection lost"))
        storage_manager.providers = [mock_provider]  # type: ignore[assignment]

        # Should NOT raise — StorageManager handles errors internally
        await manager.checkpoint(
            session_id="session-atomic-001",
            message_history=sample_messages,
            pending_calls=sample_pending_calls,
        )

        # The key invariant: internal state preserved, agent can continue
        assert manager._is_mid_stream is False

    @pytest.mark.anyio
    async def test_save_checkpoint_receives_correct_args(
        self,
        storage_manager: StorageManager,
        sample_messages: list[ModelMessage],
        sample_pending_calls: list[PendingDeferredCall],
    ) -> None:
        """save_checkpoint receives messages_json and pending_calls list."""
        manager = CheckpointManager(storage_manager)

        mock_provider = MagicMock()
        mock_provider.can_load_history = True
        mock_provider.save_checkpoint = AsyncMock()
        storage_manager.providers = [mock_provider]  # type: ignore[assignment]

        await manager.checkpoint(
            session_id="session-args-001",
            message_history=sample_messages,
            pending_calls=sample_pending_calls,
        )

        # Check that StorageManager.save_checkpoint was called with correct args
        # The save_checkpoint on StorageManager serializes pending_calls via TypeAdapter
        # and calls provider.save_checkpoint(session_id, messages_json, pending_calls_json)
        assert mock_provider.save_checkpoint.call_count >= 1
        call_args = mock_provider.save_checkpoint.call_args
        assert call_args is not None
        args, _kwargs = call_args
        assert args[0] == "session-args-001"  # session_id
        assert isinstance(args[1], str)  # messages_json
        assert isinstance(args[2], str)  # pending_calls_json


class TestCheckpointFailureLogging:
    """Tests for checkpoint failure logging.

    When StorageManager.save_checkpoint() fails (all providers fail),
    CheckpointManager.checkpoint() should log an error, not log
    "Checkpoint saved". The checkpoint call should still not raise
    (agent continues running), but the failure must be visible in logs.
    """

    @pytest.mark.anyio
    async def test_failure_logs_error_not_saved(
        self,
        storage_manager: StorageManager,
        sample_messages: list[ModelMessage],
        sample_pending_calls: list[PendingDeferredCall],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When all providers fail, 'Checkpoint save FAILED' is logged, not 'Checkpoint saved'."""
        import logging

        manager = CheckpointManager(storage_manager)

        mock_provider = MagicMock()
        mock_provider.can_load_history = True
        mock_provider.save_checkpoint = AsyncMock(side_effect=RuntimeError("disk full"))
        storage_manager.providers = [mock_provider]  # type: ignore[assignment]

        with caplog.at_level(logging.DEBUG):
            await manager.checkpoint(
                session_id="session-fail-001",
                message_history=sample_messages,
                pending_calls=sample_pending_calls,
            )

        # Should NOT contain "Checkpoint saved" on failure
        saved_logs = [r for r in caplog.records if "Checkpoint saved" in r.message]
        assert len(saved_logs) == 0, "Should not log 'Checkpoint saved' when save failed"

        # Should contain an error-level log about checkpoint failure
        error_logs = [
            r
            for r in caplog.records
            if r.levelno >= logging.ERROR and "checkpoint" in r.message.lower()
        ]
        assert len(error_logs) >= 1, "Should log an error when checkpoint save fails"

    @pytest.mark.anyio
    async def test_success_logs_saved(
        self,
        storage_manager: StorageManager,
        sample_messages: list[ModelMessage],
        sample_pending_calls: list[PendingDeferredCall],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When save succeeds, 'Checkpoint saved' is logged normally."""
        import logging

        manager = CheckpointManager(storage_manager)

        mock_provider = MagicMock()
        mock_provider.can_load_history = True
        mock_provider.save_checkpoint = AsyncMock()
        storage_manager.providers = [mock_provider]  # type: ignore[assignment]

        with caplog.at_level(logging.DEBUG):
            await manager.checkpoint(
                session_id="session-ok-001",
                message_history=sample_messages,
                pending_calls=sample_pending_calls,
            )

        saved_logs = [r for r in caplog.records if "Checkpoint saved" in r.message]
        assert len(saved_logs) == 1

    @pytest.mark.anyio
    async def test_storage_manager_returns_success_flag(self) -> None:
        """StorageManager.save_checkpoint() returns True when any provider succeeds."""
        from wolfharness_config.storage import StorageConfig

        config = StorageConfig(providers=[])
        mgr = StorageManager(config=config)

        mock_provider = MagicMock()
        mock_provider.can_load_history = True
        mock_provider.save_checkpoint = AsyncMock()
        mgr.providers = [mock_provider]  # type: ignore[assignment]

        result = await mgr.save_checkpoint("s1", "[]", [])
        assert result is True

    @pytest.mark.anyio
    async def test_storage_manager_returns_failure_flag(self) -> None:
        """StorageManager.save_checkpoint() returns False when all providers fail."""
        from wolfharness_config.storage import StorageConfig

        config = StorageConfig(providers=[])
        mgr = StorageManager(config=config)

        mock_provider = MagicMock()
        mock_provider.can_load_history = True
        mock_provider.save_checkpoint = AsyncMock(side_effect=RuntimeError("disk full"))
        mgr.providers = [mock_provider]  # type: ignore[assignment]

        result = await mgr.save_checkpoint("s1", "[]", [])
        assert result is False

    @pytest.mark.anyio
    async def test_storage_manager_returns_false_no_providers(self) -> None:
        """StorageManager.save_checkpoint() returns False when no providers configured."""
        from wolfharness_config.storage import StorageConfig

        config = StorageConfig(providers=[])
        mgr = StorageManager(config=config)

        result = await mgr.save_checkpoint("s1", "[]", [])
        assert result is False
