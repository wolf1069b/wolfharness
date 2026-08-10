"""Lifecycle types: RunState, DeliveryMode, Prompt, Feedback, etc.

These are the foundational data structures for the M2 lifecycle subsystem.
M2 uses plain dataclasses; M6 will upgrade to Pydantic models with the same
field names and types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import uuid


class RunState(Enum):
    """State machine states for RunLoop.

    - ``IDLE`` — waiting for a prompt, no active Turn.
    - ``RUNNING`` — a Turn is executing.
    - ``DONE`` — ``close()`` was called; no further Turns will execute.
    """

    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"


class RunOutcome(Enum):
    """Terminal outcome for a completed RunLoop.

    Set on ``RunHandle.outcome`` when the run reaches ``RunState.DONE``.
    ``None`` means the run has not yet terminated (still ``IDLE`` or
    ``RUNNING``) or was closed without a specific outcome.

    - ``COMPLETED`` — Run finished normally.
    - ``FAILED`` — Run finished with an error.
    - ``CHECKPOINTED`` — Run state persisted for later resumption.
    """

    COMPLETED = "completed"
    FAILED = "failed"
    CHECKPOINTED = "checkpointed"


class DeliveryMode(Enum):
    """Delivery mode for feedback messages.

    Unifies the naming across protocols:

    - ACP v2 (RFD #1261): ``mode: "steer" | "queue"``
    - OpenCode: ``SessionDelivery.Delivery = ["steer", "queue"]``
    - PydanticAI internal: ``"asap"`` (drain before model request) /
      ``"when_idle"`` (drain after node run)

    The enum values (``"steer"``, ``"queue"``) match the ACP v2 and
    OpenCode wire formats directly. ``Feedback.mode`` uses the same
    string values so ``DeliveryMode`` can be used without conversion.
    """

    STEER = "steer"
    QUEUE = "queue"


@dataclass
class Prompt:
    """Incoming prompt delivered to the RunLoop by a TriggerSource.

    Attributes:
        content: The prompt text.
        priority: Delivery priority (``"normal"`` or ``"asap"``).
        metadata: Extensible metadata (source, tags, etc.).
    """

    content: str
    priority: str = "normal"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Feedback:
    """User feedback for a bidirectional CommChannel.

    Attributes:
        content: The feedback text.
        is_steer: ``True`` if this should steer the active Turn
            (injected mid-Turn); ``False`` if it is a followup
            queued for the next Turn.
        message_id: Unique identifier for this feedback message.
            Auto-generated as a UUID4 string. Callers can override
            with an explicit value for protocol-provided IDs.
        content_blocks: Structured/multimodal content blocks, or
            ``None`` for plain-text feedback. When present, the
            pipeline carries structured content through without
            stringification.
        mode: Delivery mode — ``"steer"`` or ``"queue"``. Auto-derived
            from ``is_steer`` in ``__post_init__`` if not explicitly
            set. Matches ACP v2 and OpenCode wire format values.
    """

    content: str
    is_steer: bool
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content_blocks: list[Any] | None = None
    mode: str = ""

    def __post_init__(self) -> None:
        """Auto-derive ``mode`` from ``is_steer`` if not explicitly set."""
        if not self.mode:
            self.mode = DeliveryMode.STEER.value if self.is_steer else DeliveryMode.QUEUE.value


@dataclass
class ResumeResult:
    """Result of ``Journal.resume()`` during crash recovery.

    Attributes:
        is_inflight: Whether a Turn was interrupted mid-execution.
        state: The recovered ``RunState`` snapshot, or ``None``.
        events: Events from the journal since the snapshot.
        inflight_turn_id: Turn ID of the interrupted Turn, if any.
    """

    is_inflight: bool
    state: RunState | None
    events: list[Any]
    inflight_turn_id: str | None


@dataclass
class ToolExecutionRecord:
    """Record of a single tool execution within a Turn.

    Stored in the Journal's tool execution log for idempotent
    crash recovery.

    Attributes:
        turn_id: The Turn that executed this tool call.
        tool_name: Name of the tool that was called.
        args: Input arguments passed to the tool.
        result: The tool's return value (``None`` if not completed).
        status: Execution status (``"completed"``, ``"failed"``, etc.).
    """

    turn_id: str
    tool_name: str
    args: dict[str, Any]
    result: Any | None
    status: str


@dataclass(kw_only=True)
class EventEnvelope:
    """Language-agnostic serialization format for event transport.

    Designed to be forward-compatible with M6's Pydantic model.
    All events are serialized to ``EventEnvelope`` before transport.

    Attributes:
        schema_version: Schema version string (default ``"1.0.0"``).
        event_type: Type identifier for the payload.
        session_id: Session the event belongs to.
        turn_id: Turn the event belongs to (``None`` if not Turn-scoped).
        timestamp: ISO 8601 formatted timestamp string.
        payload: The event payload as a JSON-serializable dict.
        seq: Monotonically increasing sequence number set by
            Journal-backed transports; ``None`` for ``InProcessTransport``.
        metadata: Extensible metadata.
    """

    schema_version: str = "1.0.0"
    event_type: str
    session_id: str
    turn_id: str | None = None
    timestamp: str
    payload: dict[str, Any] = field(default_factory=dict)
    seq: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "DeliveryMode",
    "EventEnvelope",
    "Feedback",
    "Prompt",
    "ResumeResult",
    "RunOutcome",
    "RunState",
    "ToolExecutionRecord",
]
