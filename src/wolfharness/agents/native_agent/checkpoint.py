"""Checkpoint manager for durable execution.

Provides CheckpointData dataclass for checkpoint state and CheckpointManager
for serializing, persisting, and restoring agent execution state at deferred
tool boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from wolfharness.agents.events.events import ToolCallDeferredEvent
from wolfharness.log import get_logger
from wolfharness.storage.serialization import (
    deferred_calls_adapter,
    messages_adapter,
)


if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

    from wolfharness.sessions.models import PendingDeferredCall
    from wolfharness.storage.manager import StorageManager

logger = get_logger(__name__)


class MidStreamCheckpointError(RuntimeError):
    """Raised when checkpoint is attempted mid-stream (during ModelRequestNode streaming).

    Checkpoints MUST only occur at clean boundaries between model turns.
    """


@dataclass(kw_only=True)
class CheckpointData:
    """Checkpoint state containing message history and pending deferred calls.

    Attributes:
        message_history: The serialized/deserialized pydantic-ai message history.
        pending_calls: Unresolved deferred tool calls awaiting external resolution.
    """

    message_history: list[ModelMessage] = field(default_factory=list)
    """Complete message history at the checkpoint boundary."""

    pending_calls: list[PendingDeferredCall] = field(default_factory=list)
    """Unresolved deferred tool calls pending external resolution."""


def compute_agent_config_hash(tools: list[Any]) -> str:
    """Compute a deterministic SHA-256 hash of agent tool configurations.

    Hashes the deferred-relevant fields of each tool to detect config
    drift between checkpoint and resume.

    Args:
        tools: List of Tool objects or tool config dicts.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    import hashlib
    import json

    tool_data: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, dict):
            entry = {
                "name": tool.get("name", ""),
                "deferred": tool.get("deferred", False),
                "deferred_kind": tool.get("deferred_kind", "external"),
                "deferred_strategy": tool.get("deferred_strategy", "block"),
            }
        else:
            entry = {
                "name": getattr(tool, "name", ""),
                "deferred": getattr(tool, "deferred", False),
                "deferred_kind": getattr(tool, "deferred_kind", "external"),
                "deferred_strategy": getattr(tool, "deferred_strategy", "block"),
            }
        # Include parameter schema if available (non-deferred fields excluded)
        if hasattr(tool, "model_dump_json"):
            entry["param_schema"] = tool.model_dump_json(
                exclude={
                    "name",
                    "deferred",
                    "deferred_kind",
                    "deferred_strategy",
                    "created_at",
                    "updated_at",
                }
            )
        elif hasattr(tool, "parameters_json_schema"):
            entry["param_schema"] = json.dumps(tool.parameters_json_schema, sort_keys=True)
        tool_data.append(entry)

    # Sort for deterministic output
    tool_data.sort(key=lambda x: x["name"])
    canonical = json.dumps(tool_data, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


class CheckpointManager:
    """Manages checkpoint lifecycle for durable execution.

    Handles serialization of message_history via ModelMessagesTypeAdapter,
    atomic storage via StorageManager, event emission, and checkpoint loading.

    Mid-stream checkpoint prevention: the `_is_mid_stream` flag must be
    False for a checkpoint to proceed. Callers (DeferredToolBridge) are
    responsible for setting this flag before/after streaming operations.

    On storage failure, the checkpoint is aborted and the error propagates —
    the agent continues running with its resources intact.
    """

    def __init__(
        self,
        storage_manager: StorageManager,
        compaction_threshold_messages: int = 500,
        compaction_threshold_bytes: int = 5_242_880,
    ) -> None:
        """Initialize the checkpoint manager.

        Args:
            storage_manager: Storage backend for persisting checkpoint data.
            compaction_threshold_messages: Minimum message count that triggers
                compaction before checkpoint.
            compaction_threshold_bytes: Minimum serialized byte size that
                triggers compaction before checkpoint.
        """
        self._storage = storage_manager
        self._is_mid_stream: bool = False
        self._compaction_threshold_messages = compaction_threshold_messages
        self._compaction_threshold_bytes = compaction_threshold_bytes

    async def checkpoint(
        self,
        *,
        session_id: str,
        message_history: list[ModelMessage],
        pending_calls: list[PendingDeferredCall],
        agent_config_hash: str | None = None,
    ) -> None:
        """Save a checkpoint of the current agent execution state.

        Serializes message_history via ModelMessagesTypeAdapter and stores
        both messages and pending_calls atomically via StorageManager.
        Emits a ToolCallDeferredEvent for each pending call.
        Logs an error if the storage save fails (but does not raise,
        so the agent can continue running).

        Args:
            session_id: Session identifier.
            message_history: Current pydantic-ai message history.
            pending_calls: Deferred tool calls awaiting external resolution.
            agent_config_hash: Optional SHA-256 hash of agent tool configuration
                for drift detection on resume.

        Raises:
            MidStreamCheckpointError: If checkpoint is attempted mid-stream.
        """
        if self._is_mid_stream:
            raise MidStreamCheckpointError(
                f"Cannot checkpoint session '{session_id}' mid-stream. "
                "Checkpoints must occur at clean boundaries between model turns."
            )

        # Serialize messages via ModelMessagesTypeAdapter
        messages_json = (
            messages_adapter.dump_json(message_history).decode() if message_history else None
        )

        # Auto-compact message history if above thresholds
        messages_json_for_save = messages_json
        if messages_json and len(messages_json) > 0:
            message_count = len(message_history)
            byte_size = len(
                messages_json.encode() if isinstance(messages_json, str) else messages_json
            )
            if (
                message_count > self._compaction_threshold_messages
                or byte_size > self._compaction_threshold_bytes
            ):
                from wolfharness.messaging.compaction import (
                    CompactionPipeline,
                    KeepLastMessages,
                    TruncateToolOutputs,
                )

                pipeline = CompactionPipeline(
                    steps=[
                        TruncateToolOutputs(max_length=1000),
                        KeepLastMessages(count=500),
                    ]
                )
                compacted = await pipeline.apply(message_history)
                messages_json_for_save = messages_adapter.dump_json(compacted).decode()
                logger.info(
                    "Compacted message history before checkpoint",
                    session_id=session_id,
                    original_count=message_count,
                    compacted_count=len(compacted),
                )

        # Store atomically via StorageManager (messages + pending_calls together)
        save_success = await self._storage.save_checkpoint(
            session_id=session_id,
            messages_json=messages_json_for_save or "[]",
            pending_calls=pending_calls,
        )

        # Emit deferred events for protocol visibility
        for call in pending_calls:
            await self._emit_deferred_event(
                ToolCallDeferredEvent(
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                    deferred_strategy=call.deferred_strategy,
                    deferred_handle=call.tool_call_id,
                    status="pending",
                    session_id=session_id,
                )
            )

        if save_success:
            logger.info(
                "Checkpoint saved",
                session_id=session_id,
                message_count=len(message_history),
                pending_call_count=len(pending_calls),
                agent_config_hash=agent_config_hash,
            )
        else:
            logger.error(
                "Checkpoint save FAILED — no provider succeeded",
                session_id=session_id,
                message_count=len(message_history),
                pending_call_count=len(pending_calls),
                agent_config_hash=agent_config_hash,
            )

    async def load_checkpoint(self, session_id: str) -> CheckpointData | None:
        """Load and deserialize a checkpoint from storage.

        Args:
            session_id: Session identifier.

        Returns:
            CheckpointData with message_history and pending_calls, or None
            if no checkpoint exists for this session.
        """
        result = await self._storage.load_checkpoint(session_id)
        if result is None:
            logger.debug("No checkpoint found", session_id=session_id)
            return None

        messages, calls = result
        return CheckpointData(message_history=messages, pending_calls=calls)

    async def _emit_deferred_event(self, event: ToolCallDeferredEvent) -> None:
        """Emit a ToolCallDeferredEvent for protocol visibility.

        Override in tests or subclass to capture events.

        Args:
            event: The deferred event to emit.
        """
        logger.debug(
            "Emitting deferred event",
            tool_call_id=event.tool_call_id,
            tool_name=event.tool_name,
            strategy=event.deferred_strategy,
        )

    @staticmethod
    def _serialize_messages(messages: list[ModelMessage]) -> str | None:
        """Serialize a list of ModelMessage to JSON string.

        Uses ModelMessagesTypeAdapter for pydantic-ai compatible serialization.

        Args:
            messages: ModelMessage list to serialize.

        Returns:
            JSON string or None if messages is empty.
        """
        if not messages:
            return None
        return messages_adapter.dump_json(messages).decode()

    @staticmethod
    def _serialize_pending_calls(calls: list[PendingDeferredCall]) -> str:
        """Serialize a list of PendingDeferredCall to JSON string.

        Args:
            calls: Pending deferred calls to serialize.

        Returns:
            JSON string (empty list returns "[]").
        """
        if not calls:
            return "[]"
        return deferred_calls_adapter.dump_json(calls).decode()
