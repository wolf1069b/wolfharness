"""Event processor context for OpenCode server.

Holds mutable state for event processing per session/level.
This context is designed for recursive subagent handling where each
child session gets its own child context.

Supports serialization/deserialization for durable session resume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from wolfharness_server.opencode_server.models import MessageWithParts
    from wolfharness_server.opencode_server.models.parts import (
        ReasoningPart,
        TextPart,
        ToolPart,
    )
    from wolfharness_server.opencode_server.state import ServerState


def _model_validate_or_none(model_cls: type[Any], data: Any) -> Any | None:
    """Safely validate model data; returns None on failure instead of raising.

    Args:
        model_cls: Pydantic model class to validate against.
        data: Raw dict or model instance to validate.

    Returns:
        Validated model instance, or None if validation failed.
    """
    if data is None:
        return None
    try:
        return model_cls.model_validate(data)
    except Exception:  # noqa: BLE001
        return None


@dataclass
class EventProcessorContext:
    """Mutable state context for the EventProcessor.

    Holds all tracking state that changes during stream processing:
    - Token and cost tracking
    - Tool call state accumulation
    - Text and reasoning accumulation
    - Subagent tool part tracking

    Contexts are created per session/level. For recursive subagent handling,
    each child session gets its own child EventProcessorContext.

    Args:
        session_id: The OpenCode session ID for this context.
        assistant_msg_id: The assistant message ID for updates.
        assistant_msg: The mutable assistant message to append parts to.
        state: The server state for session management and event routing.
        working_dir: Working directory for path context.
    """

    # Context identifier fields
    session_id: str
    assistant_msg_id: str
    assistant_msg: MessageWithParts
    state: ServerState
    working_dir: str

    # --- mutable tracking state ---

    # Text accumulation
    response_text: str = field(default="", init=False)
    text_part: TextPart | None = field(default=None, init=False)
    reasoning_part: ReasoningPart | None = field(default=None, init=False)

    # Token and cost tracking
    input_tokens: int = field(default=0, init=False)
    output_tokens: int = field(default=0, init=False)
    total_cost: float = field(default=0.0, init=False)
    stream_start_ms: int = field(default=0, init=False)

    # Tool call tracking
    tool_parts: dict[str, ToolPart] = field(default_factory=dict, init=False)
    tool_outputs: dict[str, str] = field(default_factory=dict, init=False)
    tool_inputs: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    # Unified diff text accumulated from DiffContentItem in ToolCallProgressEvent,
    # carried to ToolStateCompleted.metadata.diff for TUI diff rendering.
    tool_diffs: dict[str, str] = field(default_factory=dict, init=False)

    # Subagent tool parts tracking (key: "depth:source_name" -> ToolPart)
    subagent_tool_parts: dict[str, ToolPart] = field(default_factory=dict, init=False)

    # Error flag: set when RunErrorEvent is processed for this context's subagent,
    # preventing a subsequent StreamCompleteEvent from overriding the error state.
    is_errored: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        from wolfharness.utils.time_utils import now_ms

        self.stream_start_ms = now_ms()

    # --- serialization / deserialization for durable resume ---

    def serialize(self) -> dict[str, Any]:
        """Serialize mutable state to a JSON-compatible dict.

        The ``state`` (ServerState) and ``working_dir`` are runtime
        dependencies and are NOT serialized — they must be provided at
        deserialization time.

        Returns:
            A dict containing all reconstructable mutable state.
        """
        return {
            "session_id": self.session_id,
            "assistant_msg_id": self.assistant_msg_id,
            "assistant_msg": self.assistant_msg.model_dump(),
            "response_text": self.response_text,
            "text_part": self.text_part.model_dump() if self.text_part is not None else None,
            "reasoning_part": (
                self.reasoning_part.model_dump() if self.reasoning_part is not None else None
            ),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_cost": self.total_cost,
            "stream_start_ms": self.stream_start_ms,
            "tool_parts": {tc_id: tp.model_dump() for tc_id, tp in self.tool_parts.items()},
            "tool_outputs": dict(self.tool_outputs),
            "tool_inputs": dict(self.tool_inputs),
            "subagent_tool_parts": {
                key: tp.model_dump() for key, tp in self.subagent_tool_parts.items()
            },
            "is_errored": self.is_errored,
        }

    @classmethod
    def deserialize(
        cls,
        data: dict[str, Any],
        *,
        state: ServerState,
        working_dir: str,
    ) -> EventProcessorContext:
        """Reconstruct an EventProcessorContext from serialized data.

        The ``state`` and ``working_dir`` are runtime dependencies injected
        by the caller (they are NOT persisted in the serialized form).

        Args:
            data: Serialized context dict from :meth:`serialize`.
            state: The current server state.
            working_dir: The current working directory.

        Returns:
            A fully reconstructed EventProcessorContext.

        Raises:
            ValueError: If required fields are missing from ``data``.
        """
        from wolfharness_server.opencode_server.models import MessageWithParts
        from wolfharness_server.opencode_server.models.parts import (
            ReasoningPart,
            TextPart,
            ToolPart,
        )

        # --- reconstruct assistant message ---
        assistant_msg_raw: dict[str, Any] = data.get("assistant_msg", {})
        if not assistant_msg_raw:
            raise ValueError("Missing 'assistant_msg' in serialized context data")
        assistant_msg = _model_validate_or_none(MessageWithParts, assistant_msg_raw)
        if assistant_msg is None:
            raise ValueError("Failed to reconstruct assistant_msg")

        # --- create bare context (__post_init__ sets stream_start_ms) ---
        ctx = cls(
            session_id=data.get("session_id", ""),
            assistant_msg_id=data.get("assistant_msg_id", ""),
            assistant_msg=assistant_msg,
            state=state,
            working_dir=working_dir,
        )

        # --- restore mutable fields (overrides __post_init__ defaults) ---
        ctx.response_text = data.get("response_text", "")
        ctx.input_tokens = data.get("input_tokens", 0)
        ctx.output_tokens = data.get("output_tokens", 0)
        ctx.total_cost = data.get("total_cost", 0.0)
        ctx.is_errored = data.get("is_errored", False)

        # stream_start_ms: prefer serialized value (over __post_init__ timestamp)
        serialized_start: int = data.get("stream_start_ms", 0)
        if serialized_start > 0:
            ctx.stream_start_ms = serialized_start

        # --- reconstruct text_part ---
        text_part_raw: dict[str, Any] | None = data.get("text_part")
        if text_part_raw is not None:
            ctx.text_part = _model_validate_or_none(TextPart, text_part_raw)

        # --- reconstruct reasoning_part ---
        reasoning_part_raw: dict[str, Any] | None = data.get("reasoning_part")
        if reasoning_part_raw is not None:
            ctx.reasoning_part = _model_validate_or_none(ReasoningPart, reasoning_part_raw)

        # --- reconstruct tool_parts ---
        tool_parts_raw: dict[str, Any] = data.get("tool_parts", {})
        ctx.tool_parts = {}
        for tc_id, tp_raw in tool_parts_raw.items():
            tp = _model_validate_or_none(ToolPart, tp_raw)
            if tp is not None:
                ctx.tool_parts[tc_id] = tp

        # --- reconstruct tool_outputs ---
        ctx.tool_outputs = dict(data.get("tool_outputs", {}))

        # --- reconstruct tool_inputs ---
        ctx.tool_inputs = dict(data.get("tool_inputs", {}))

        # --- reconstruct subagent_tool_parts ---
        subagent_tool_parts_raw: dict[str, Any] = data.get("subagent_tool_parts", {})
        ctx.subagent_tool_parts = {}
        for key, tp_raw in subagent_tool_parts_raw.items():
            tp = _model_validate_or_none(ToolPart, tp_raw)
            if tp is not None:
                ctx.subagent_tool_parts[key] = tp

        return ctx

    # --- public read-only accessors ---

    @property
    def text_accumulated(self) -> str:
        """Return the accumulated response text."""
        return self.response_text

    @property
    def has_text_part(self) -> bool:
        """Return True if a text part has been created."""
        return self.text_part is not None

    @property
    def has_reasoning_part(self) -> bool:
        """Return True if a reasoning part has been created."""
        return self.reasoning_part is not None

    # --- state update helpers ---

    def accumulate_text(self, delta: str) -> None:
        """Accumulate text into the response."""
        self.response_text += delta

    def set_text(self, text: str) -> None:
        """Set the response text (used for initial text)."""
        self.response_text = text

    def update_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Update token counts."""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def update_cost(self, total_cost: float) -> None:
        """Update the total cost."""
        self.total_cost = total_cost

    def add_tool_part(self, tool_call_id: str, tool_part: ToolPart) -> None:
        """Register a tool part for tracking.

        Args:
            tool_call_id: The unique identifier for the tool call.
            tool_part: The ToolPart to track.
        """
        self.tool_parts[tool_call_id] = tool_part

    def remove_tool_part(self, tool_call_id: str) -> ToolPart | None:
        """Remove and return a tracked tool part.

        Args:
            tool_call_id: The tool call ID to remove.

        Returns:
            The removed ToolPart or None if not found.
        """
        return self.tool_parts.pop(tool_call_id, None)

    def get_tool_part(self, tool_call_id: str) -> ToolPart | None:
        """Get a tracked tool part without removing it.

        Args:
            tool_call_id: The tool call ID to look up.

        Returns:
            The ToolPart or None if not found.
        """
        return self.tool_parts.get(tool_call_id)

    def has_tool_part(self, tool_call_id: str) -> bool:
        """Check if a tool part is being tracked.

        Args:
            tool_call_id: The tool call ID to check.

        Returns:
            True if the tool part exists in tracking.
        """
        return tool_call_id in self.tool_parts

    def set_tool_output(self, tool_call_id: str, output: str) -> None:
        """Set the accumulated output for a tool call.

        Args:
            tool_call_id: The tool call ID.
            output: The output string to set or append to.
        """
        self.tool_outputs[tool_call_id] = output

    def append_tool_output(self, tool_call_id: str, delta: str) -> None:
        """Append to the accumulated output for a tool call.

        Args:
            tool_call_id: The tool call ID.
            delta: The text to append.
        """
        current = self.tool_outputs.get(tool_call_id, "")
        self.tool_outputs[tool_call_id] = current + delta

    def get_tool_output(self, tool_call_id: str) -> str:
        """Get the accumulated output for a tool call.

        Args:
            tool_call_id: The tool call ID.

        Returns:
            The accumulated output string or empty string if not found.
        """
        return self.tool_outputs.get(tool_call_id, "")

    def set_tool_input(self, tool_call_id: str, tool_input: dict[str, Any]) -> None:
        """Set the input parameters for a tool call.

        Args:
            tool_call_id: The tool call ID.
            tool_input: The input parameters dictionary.
        """
        self.tool_inputs[tool_call_id] = tool_input

    def get_tool_input(self, tool_call_id: str) -> dict[str, Any] | None:
        """Get the input parameters for a tool call.

        Args:
            tool_call_id: The tool call ID.

        Returns:
            The input parameters dictionary or None if not found.
        """
        return self.tool_inputs.get(tool_call_id)

    def add_subagent_tool_part(self, subagent_key: str, tool_part: ToolPart) -> None:
        """Register a subagent tool part for tracking.

        Args:
            subagent_key: The composite key "depth:source_name" for the subagent.
            tool_part: The ToolPart to track.
        """
        self.subagent_tool_parts[subagent_key] = tool_part

    def get_subagent_tool_part(self, subagent_key: str) -> ToolPart | None:
        """Get a tracked subagent tool part.

        Args:
            subagent_key: The composite key "depth:source_name" for the subagent.

        Returns:
            The ToolPart or None if not found.
        """
        return self.subagent_tool_parts.get(subagent_key)

    def has_subagent_tool_part(self, subagent_key: str) -> bool:
        """Check if a subagent tool part is being tracked.

        Args:
            subagent_key: The composite key "depth:source_name" for the subagent.

        Returns:
            True if the subagent tool part exists in tracking.
        """
        return subagent_key in self.subagent_tool_parts
