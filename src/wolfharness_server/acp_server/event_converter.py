"""Convert agent stream events to ACP notifications.

This module provides a stateful converter that transforms agent stream events
into ACP session update objects. The converter tracks tool call state but does
not perform any I/O - it yields notification objects that can be emitted
by the caller.

This separation enables easy testing without mocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import TYPE_CHECKING, Any, Literal
import uuid

from pydantic import BaseModel
from pydantic_ai import (
    FinalResultEvent,
    FunctionToolResultEvent,
    NativeToolCallPart,
    NativeToolReturnPart,
    OutputToolCallEvent,
    OutputToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)

from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    ContentToolCallContent,
    Cost,
    ToolCallLocation,
    ToolCallProgress,
    ToolCallStart,
    TurnCompleteUpdate,
    Usage,
    UsageUpdate,
    UserMessageChunk,
)
from acp.utils import generate_tool_title, infer_tool_kind, to_acp_content_blocks
from wolfharness.agents.events import (
    CompactionEvent,
    CustomEvent,
    DiffContentItem,
    ElicitationDeferredEvent,
    FileContentItem,
    LocationContentItem,
    PartDeltaEvent as AgentPoolPartDeltaEvent,
    PartStartEvent as AgentPoolPartStartEvent,
    PlanUpdateEvent,
    RunErrorEvent,
    RunFailedEvent,
    RunStartedEvent,
    SessionResumeEvent,
    SpawnSessionStart,
    StepUsageEvent,
    StreamCompleteEvent,
    SubAgentEvent,
    TerminalContentItem,
    TextContentItem,
    ToolCallCompleteEvent,
    ToolCallDeferredEvent,
    ToolCallProgressEvent,
    ToolCallStartEvent,
    ToolResultMetadataEvent,
    UserMessageInsertedEvent,
)
from wolfharness.log import get_logger
from wolfharness.utils.pydantic_ai_helpers import safe_args_as_dict


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai.usage import UsageBase

    from acp.schema import TextContentBlock
    from acp.schema.tool_call import ToolCallContent, ToolCallKind
    from wolfharness.agents.events import RichAgentStreamEvent

logger = get_logger(__name__)


@dataclass(frozen=True)
class ACPUserMessageMeta:
    """Protocol-specific metadata for ACP user messages.

    Carries serialized ContentBlock data so the ACPEventConverter can
    reconstruct the full user message from the EventBus event without
    the protocol handler broadcasting separately.
    """

    content_blocks: list[dict[str, Any]]
    """Serialized ContentBlock data as dicts.

    Each dict is the ``model_dump()`` output of an ACP ContentBlock.
    The converter deserializes each dict back to the appropriate
    ContentBlock type.
    """


# Type alias for all session updates the converter can yield
ACPSessionUpdate = (
    AgentMessageChunk
    | AgentThoughtChunk
    | UserMessageChunk
    | ToolCallStart
    | ToolCallProgress
    | AgentPlanUpdate
    | UsageUpdate
    | TurnCompleteUpdate
)


def get_compaction_text(trigger: str) -> str:
    if trigger == "auto":
        return "\n\n---\n\n📦 **Context compaction** triggered. Summarizing...\n\n---\n\n"
    return "\n\n---\n\n📦 **Manual compaction** requested. Summarizing...\n\n---\n\n"


def _to_acp_usage(usage: UsageBase) -> Usage:
    """Convert a pydantic-ai ``UsageBase`` to an ACP ``Usage`` object.

    Accepts both ``RunUsage`` (per-step deltas) and ``RequestUsage``
    (final turn totals) since both share the same token fields through
    ``UsageBase``.

    Maps token fields from the pydantic-ai usage representation to the
    ACP schema.  ``thought_tokens`` is extracted from the ``details``
    dict (key ``"reasoning_tokens"``).  Cache fields are set to ``None``
    when zero so they are omitted from the serialized output.

    Args:
        usage: The pydantic-ai usage object to convert (``RunUsage`` for
            per-step deltas, ``RequestUsage`` for final turn totals).

    Returns:
        An ACP ``Usage`` object with the corresponding token counts.
    """
    thought = usage.details.get("reasoning_tokens") or None
    return Usage(
        total_tokens=usage.total_tokens,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        thought_tokens=thought,
        cached_read_tokens=usage.cache_read_tokens or None,
        cached_write_tokens=usage.cache_write_tokens or None,
    )


@dataclass
class _ToolState:
    """Internal state for a single tool call."""

    tool_call_id: str
    tool_name: str
    title: str
    kind: ToolCallKind
    raw_input: dict[str, Any]
    started: bool = False
    has_content: bool = False


class SubagentSessionInfo(BaseModel):
    """Information about a subagent session for contextual display.

    Used by protocols that need to correlate subagent output with the
    parent session's message stream (e.g., Zed subagent display mode).
    """

    session_id: str
    """The child session ID spawned for subagent work."""

    message_start_index: int | None = None
    """Index of the message where subagent output begins (for ordering)."""

    message_end_index: int | None = None
    """Index of the message where subagent output ends (for ordering)."""


# ============================================================================
# Event Converter
# ============================================================================


@dataclass
class SubagentContext:
    """Parent context for a child session converter."""

    parent_tool_call_id: str
    subagent_type: str


@dataclass
class ACPEventConverter:
    """Converts agent stream events to ACP session updates.

    Stateful converter that tracks tool calls and subagent content,
    yielding ACP schema objects without performing I/O.

    Example:
        ```python
        converter = ACPEventConverter()
        async for event in agent.run_stream(...):
            async for update in converter.convert(event):
                await client.session_update(SessionNotification(session_id=sid, update=update))
        ```
    """

    # Deprecated: kept for backward compatibility of constructor calls
    subagent_display_mode: Literal["legacy", "zed", "qwen"] = "legacy"
    """How to display subagent output. "legacy" (default), "zed", or "qwen"."""

    raw_input_mode: Literal["dict", "skip", "json_str"] = "dict"
    """How to emit tool call raw_input in ACP session updates:
    - "dict": Parse args as dict (default; partial JSON returns empty dict)
    - "skip": Omit raw_input in ToolCallStart; deliver via ToolCallProgress
    - "json_str": Emit raw_input as a JSON string instead of a dict
    """

    # Feature flag for TurnCompleteUpdate emission
    client_supports_turn_complete: bool = False
    """Whether the connected ACP client supports TurnCompleteUpdate.

    When True, the converter yields TurnCompleteUpdate on StreamCompleteEvent.
    When False (default), no TurnCompleteUpdate is emitted for backward
    compatibility with clients that do not handle the update type.
    """

    subagent_context: SubagentContext | None = None
    """Parent context for child session converters. None for root sessions."""

    # Internal state
    _tool_states: dict[str, _ToolState] = field(default_factory=dict)
    """Active tool call states."""

    _current_message_id: str = ""
    _current_tool_inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Current tool inputs by tool_call_id."""

    _subagent_headers: set[str] = field(default_factory=set)
    """Track which subagent headers have been sent (for inline mode)."""

    _subagent_content: dict[str, list[str]] = field(default_factory=dict)
    """Accumulated content per subagent (for tool_box mode)."""

    _child_sessions: set[str] = field(default_factory=set)
    """Track child session IDs that have been spawned."""

    _subagent_tool_call_ids: dict[str, str] = field(default_factory=dict)
    """Map child_session_id to tool_call_id for zed mode subagent tracking."""

    last_usage: Usage | None = field(default=None, init=False)
    """Usage from the last completed stream, if available."""

    _displayed_message_ids: set[str] = field(default_factory=set, init=False)
    """Per-session set of message_ids already displayed as UserMessageChunk.

    When ``steer()``/``followup()`` emits a fire-and-forget
    ``UserMessageInsertedEvent`` AND ``EnqueuedMessagesEvent`` also
    produces one with the same ``message_id`` (reused via
    ``_pending_enqueue_message_ids``), the converter skips the second
    event. This set is NOT cleared in ``reset()`` — it persists for the
    entire session lifetime.
    """

    def _format_raw_input(self, raw_input: dict[str, Any] | None) -> Any:
        """Format raw_input for ACP session updates based on raw_input_mode.

        Args:
            raw_input: The parsed tool arguments dict (or None).

        Returns:
            - "dict" mode: the dict as-is (or None if empty/None)
            - "skip" mode: None (raw_input delivered later via ToolCallProgress)
            - "json_str" mode: JSON string representation (or None if empty/None)
        """
        if not raw_input:
            return None
        match self.raw_input_mode:
            case "skip":
                return None
            case "json_str":
                return json.dumps(raw_input, ensure_ascii=False)
            case _:
                return raw_input

    def _build_subagent_field_meta(
        self,
        child_session_id: str,
        message_start_index: int | None = None,
        message_end_index: int | None = None,
    ) -> dict[str, Any] | None:
        """Build subagent field metadata for ACP session updates.

        Returns ``None`` when *child_session_id* is empty so that callers
        can skip emitting subagent-related fields without additional
        branching.

        Args:
            child_session_id: The spawned child session ID.
            message_start_index: Optional start index for message range.
            message_end_index: Optional end index for message range.
        """
        if not child_session_id:
            return None
        return {
            "subagent_session_info": SubagentSessionInfo(
                session_id=child_session_id,
                message_start_index=message_start_index,
                message_end_index=message_end_index,
            ).model_dump(exclude_none=True),
            "tool_name": "task",
        }

    def reset(self) -> None:
        """Reset converter state for a new run."""
        self._current_message_id = ""
        self._tool_states.clear()
        self._current_tool_inputs.clear()
        self._subagent_headers.clear()
        self.last_usage = None
        self._subagent_content.clear()
        self._child_sessions.clear()
        self._subagent_tool_call_ids.clear()
        self.cleanup()

    @staticmethod
    def build_user_message_chunks(
        text: str,
        *,
        message_id: str | None = None,
    ) -> list[UserMessageChunk]:
        """Build UserMessageChunk notifications for a user's text input.

        Args:
            text: The user's input text.
            message_id: Optional message ID for correlating chunks.

        Returns:
            List of UserMessageChunk session update objects.
        """
        if not text:
            return []
        mid = message_id or str(uuid.uuid4())
        return [UserMessageChunk.text(text, message_id=mid)]

    @staticmethod
    def _content_to_text_blocks(content: str | list[Any]) -> list[TextContentBlock]:
        """Convert ``UserMessageInsertedEvent.content`` to text content blocks.

        Handles both plain string content and multi-modal part lists:
        - ``str`` → single ``TextContentBlock``
        - ``list`` → iterate, extract text from dicts with ``"type": "text"``
          or bare strings; skip non-text items (images, etc.) in v1.

        Args:
            content: The event content — plain text or multi-modal part list.

        Returns:
            List of ``TextContentBlock`` instances (may be empty).
        """
        from acp.schema import TextContentBlock

        if isinstance(content, str):
            if not content:
                return []
            return [TextContentBlock(text=content)]
        if isinstance(content, list):
            blocks: list[TextContentBlock] = []
            for item in content:
                if isinstance(item, str):
                    if item:
                        blocks.append(TextContentBlock(text=item))
                elif isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        blocks.append(TextContentBlock(text=text))
            return blocks
        return []

    def cleanup(self) -> None:
        """Clean up converter state.

        Idempotent — safe to call multiple times.
        """

    @property
    def subagent_meta(self) -> dict[str, Any] | None:
        """Build _meta dict for subagent notifications. None for root sessions."""
        if self.subagent_context is None:
            return None
        return {
            "parentToolCallId": self.subagent_context.parent_tool_call_id,
            "subagentType": self.subagent_context.subagent_type,
            "provenance": "subagent",
        }

    # =========================================================================
    # V2_EXTENSION: ACP V2 protocol hooks (no-op on V1)
    #
    # These hooks are placeholders for ACP V2 concepts:
    #   - _on_state_change   → session/update with state_change notification
    #   - _on_out_of_turn_update → out-of-turn content delivery
    #
    # Reference: ACP V2 spec (unstable) — state_change, out_of_turn_update
    # =========================================================================

    def _on_state_change(self, state: str) -> None:
        """Handle agent processing state transitions.

        # V2_EXTENSION: In ACP V2 this would emit a ``session/update``
        # notification with a ``state_change`` payload. On V1 this is
        # a no-op.

        Args:
            state: The new processing state (e.g. ``"idle"``, ``"running"``).
        """
        # V2_EXTENSION: emit state_change session/update in ACP V2

    def _on_out_of_turn_update(self) -> None:
        """Handle out-of-turn content updates.

        # V2_EXTENSION: In ACP V2 this would deliver content that arrives
        # outside of an active turn (e.g. background notifications,
        # deferred tool completions). On V1 this is a no-op.
        """
        # V2_EXTENSION: deliver out-of-turn content in ACP V2

    async def cancel_pending_tools(self) -> AsyncIterator[ToolCallProgress]:
        """Cancel all pending tool calls.

        Yields ToolCallProgress notifications with status="completed" for all
        tool calls that were started but not completed. This should be called
        when the stream is interrupted to properly clean up client-side state.

        Note:
            Uses status="completed" since ACP doesn't have a "cancelled" status.
            This signals to the client that we're done with these tool calls.

        Yields:
            ToolCallProgress notifications for each pending tool call
        """
        for tool_call_id, state in list(self._tool_states.items()):
            if state.started:
                yield ToolCallProgress(tool_call_id=tool_call_id, status="completed")
        # Clean up all state
        self.reset()

    async def build_subagent_completed(
        self,
        child_session_id: str,
    ) -> AsyncIterator[ToolCallProgress]:
        """Emit a completion notification for a subagent session in zed mode.

        Yields a ToolCallProgress with status="completed" and subagent
        field metadata, closing the tool call lifecycle started by
        SpawnSessionStart in zed mode. In legacy mode, this is a no-op.

        Args:
            child_session_id: The child session ID that has completed.
        """
        if self.subagent_display_mode != "zed":
            return
        tool_call_id = self._subagent_tool_call_ids.pop(child_session_id, None)
        if not tool_call_id:
            return
        field_meta = self._build_subagent_field_meta(child_session_id=child_session_id)
        yield ToolCallProgress(
            tool_call_id=tool_call_id,
            status="completed",
            field_meta=field_meta,
        )

    def _get_or_create_tool_state(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> _ToolState:
        """Get existing tool state or create a new one."""
        if tool_call_id not in self._tool_states:
            self._tool_states[tool_call_id] = _ToolState(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                title=generate_tool_title(tool_name, tool_input),
                kind=infer_tool_kind(tool_name),
                raw_input=tool_input,
            )
        return self._tool_states[tool_call_id]

    def _cleanup_tool_state(self, tool_call_id: str) -> None:
        """Remove tool state after completion."""
        self._tool_states.pop(tool_call_id, None)
        self._current_tool_inputs.pop(tool_call_id, None)

    def _get_message_id(self, event: RichAgentStreamEvent[Any]) -> str:
        """Extract message_id from event, or reuse known ID, or generate UUID.

        Maintains a sticky ``_current_message_id`` that persists across
        events within the same turn. This ensures that all chunks of a
        single message (e.g., successive ``ThinkingPartDelta`` events)
        share the same ``message_id`` even if the source events carry
        inconsistent or empty IDs.

        For ACP agents that may not send ``message_id`` on every chunk,
        the first event's ID becomes the canonical ID for the entire
        turn. Falls back to a one-off UUID only when no event has ever
        carried a message_id.

        Args:
            event: The stream event to extract message_id from.

        Returns:
            A consistent message_id for the current turn segment.
        """
        match event:
            case (
                AgentPoolPartStartEvent(message_id=mid) | AgentPoolPartDeltaEvent(message_id=mid)
            ) if mid:
                self._current_message_id = mid
                return mid
            case _:
                if self._current_message_id:
                    return self._current_message_id
                mid = str(uuid.uuid4())
                self._current_message_id = mid
                return mid

    async def convert(  # noqa: PLR0915
        self, event: RichAgentStreamEvent[Any]
    ) -> AsyncIterator[ACPSessionUpdate]:
        """Convert an agent event to zero or more ACP session updates."""
        from acp.schema import (
            FileEditToolCallContent,
            PlanEntry as ACPPlanEntry,
            TerminalToolCallContent,
        )
        from wolfharness_server.acp_server.syntax_detection import format_zed_code_block

        match event:
            # Text output
            case (
                PartStartEvent(part=TextPart(content=delta))
                | PartDeltaEvent(delta=TextPartDelta(content_delta=delta))
            ):
                yield AgentMessageChunk.text(delta, message_id=self._get_message_id(event))

            # User message inserted (steer/followup display)
            case UserMessageInsertedEvent(source="processed"):
                # ACP has no split concept — processed events carry no
                # useful info beyond what the accepted event already
                # displayed.
                return

            case UserMessageInsertedEvent(
                message_id=mid,
                content=event_content,
                meta=event_meta,
            ):
                # Dedup: skip if this message_id was already displayed.
                # This happens when both the fire-and-forget emission
                # from steer()/followup() and the EnqueuedMessagesEvent
                # produce a UserMessageInsertedEvent with the same ID.
                if mid and mid in self._displayed_message_ids:
                    return
                # Use meta.content_blocks if available, otherwise fall back
                # to text-only content.
                if isinstance(event_meta, ACPUserMessageMeta):
                    from acp.schema import TextContentBlock

                    blocks: list[Any] = []
                    for block_dict in event_meta.content_blocks:
                        block_type = block_dict.get("type", "")
                        if block_type == "text":
                            text = block_dict.get("text", "")
                            if text:
                                blocks.append(TextContentBlock(text=text))
                    for block in blocks:
                        yield UserMessageChunk(
                            content=block,
                            message_id=mid or None,
                        )
                else:
                    # Fall back to text-only content.
                    blocks = self._content_to_text_blocks(event_content)
                    for block in blocks:
                        yield UserMessageChunk(
                            content=block,
                            message_id=mid or None,
                        )
                # Mark this message_id as displayed for dedup.
                if mid:
                    self._displayed_message_ids.add(mid)

            # Thinking/reasoning
            case (
                PartStartEvent(part=ThinkingPart(content=delta))
                | PartDeltaEvent(delta=ThinkingPartDelta(content_delta=delta))
            ):
                if delta is not None:
                    yield AgentThoughtChunk.text(delta, message_id=self._get_message_id(event))

            # Builtin tool call started (e.g., WebSearchTool, CodeExecutionTool)
            case PartStartEvent(part=NativeToolCallPart() as part):
                tool_call_id = part.tool_call_id
                tool_input = safe_args_as_dict(part, default={})
                self._current_tool_inputs[tool_call_id] = tool_input
                state = self._get_or_create_tool_state(tool_call_id, part.tool_name, tool_input)
                if not state.started:
                    state.started = True
                    yield ToolCallStart(
                        tool_call_id=tool_call_id,
                        title=state.title,
                        kind=state.kind,
                        raw_input=self._format_raw_input(state.raw_input),
                        status="pending",
                    )

            # Builtin tool completed
            case PartStartEvent(part=NativeToolReturnPart(content=out, tool_call_id=tc_id)):
                tool_state = self._tool_states.get(tc_id)
                if tool_state and tool_state.has_content:
                    yield ToolCallProgress(tool_call_id=tc_id, status="completed", raw_output=out)
                else:
                    converted = to_acp_content_blocks(out)
                    yield ToolCallProgress(
                        tool_call_id=tc_id,
                        status="completed",
                        raw_output=out,
                        content=[ContentToolCallContent(content=block) for block in converted],
                    )
                self._cleanup_tool_state(tc_id)

            # Regular tool call started (e.g., question_for_user)
            case PartStartEvent(part=ToolCallPart() as part):
                tool_call_id = part.tool_call_id
                tool_input = safe_args_as_dict(part, default={})
                self._current_tool_inputs[tool_call_id] = tool_input
                state = self._get_or_create_tool_state(tool_call_id, part.tool_name, tool_input)
                if not state.started:
                    state.started = True
                    yield ToolCallStart(
                        tool_call_id=tool_call_id,
                        title=state.title,
                        kind=state.kind,
                        raw_input=self._format_raw_input(state.raw_input),
                        status="pending",
                    )

            case PartStartEvent(part=part):
                logger.debug("Received unhandled PartStartEvent", part=part)

            # Tool call streaming delta
            case PartDeltaEvent(delta=ToolCallPartDelta() as delta):
                # Streaming deltas are not forwarded to ACP client.
                # Tool call state is managed via ToolCallStartEvent and
                # ToolCallProgressEvent from EventMapper.
                pass

            # Tool completed successfully
            case FunctionToolResultEvent(part=ToolReturnPart(content=out), tool_call_id=tc_id):
                # Handle async generator content
                tool_state = self._tool_states.get(tc_id)
                if tool_state and tool_state.has_content:
                    yield ToolCallProgress(tool_call_id=tc_id, status="completed", raw_output=out)
                else:
                    converted = to_acp_content_blocks(out)
                    content_items = [ContentToolCallContent(content=block) for block in converted]
                    yield ToolCallProgress(
                        tool_call_id=tc_id,
                        status="completed",
                        raw_output=out,
                        content=content_items,
                    )
                self._cleanup_tool_state(tc_id)

            # Tool failed with retry
            case FunctionToolResultEvent(part=RetryPromptPart() as result, tool_call_id=tc_id):
                error_message = result.model_response()
                content = ContentToolCallContent.text(f"Error: {error_message}")
                yield ToolCallProgress(tool_call_id=tc_id, status="failed", content=[content])
                self._cleanup_tool_state(tc_id)

            # Tool emits its own start event
            case ToolCallStartEvent(
                tool_call_id=tc_id,
                tool_name=tool_name,
                title=title,
                kind=kind,
                locations=loc_items,
                raw_input=raw_input,
            ):
                state = self._get_or_create_tool_state(tc_id, tool_name, raw_input or {})
                acp_locations = [ToolCallLocation(path=i.path, line=i.line) for i in loc_items]
                # If not started, send start notification
                if not state.started:
                    state.started = True
                    yield ToolCallStart(
                        tool_call_id=tc_id,
                        title=title,
                        kind=kind,
                        raw_input=self._format_raw_input(raw_input),
                        locations=acp_locations or None,
                        status="pending",
                    )
                else:
                    # Send update with tool-provided details
                    yield ToolCallProgress(
                        tool_call_id=tc_id,
                        title=title,
                        kind=kind,
                        locations=acp_locations or None,
                    )

            # Tool progress event - create state if needed (tool may emit progress before SDK event)
            case ToolCallProgressEvent(
                tool_call_id=tool_call_id,
                title=title,
                status=status,
                items=items,
                progress=progress,
                total=total,
                message=message,
                tool_input=tool_input,
                tool_name=tool_name,
            ) if tool_call_id:
                # Get or create state - handles race where tool emits before SDK event
                state = self._get_or_create_tool_state(
                    tool_call_id, tool_name or "unknown", tool_input or {}
                )
                # Update state with tool_input and tool_name from the event
                if tool_input is not None:
                    state.raw_input = tool_input
                    state.title = generate_tool_title(tool_name or state.tool_name, tool_input)
                if tool_name is not None and state.tool_name == "unknown":
                    state.tool_name = tool_name
                # Emit start if this is the first event for this tool call
                if not state.started:
                    state.started = True
                    yield ToolCallStart(
                        tool_call_id=tool_call_id,
                        title=title or state.title,
                        kind=state.kind,
                        raw_input=self._format_raw_input(state.raw_input),
                        status="pending",
                    )
                acp_content: list[ToolCallContent] = []
                locations: list[ToolCallLocation] = []
                for item in items:
                    match item:
                        case TerminalContentItem(terminal_id=tid):
                            acp_content.append(TerminalToolCallContent(terminal_id=tid))
                        case TextContentItem(text=text):
                            acp_content.append(ContentToolCallContent.text(text))
                        case FileContentItem(
                            content=file_content,
                            path=file_path,
                            start_line=start_line,
                            end_line=end_line,
                        ):
                            formatted = format_zed_code_block(
                                file_content, file_path, start_line, end_line
                            )
                            acp_content.append(ContentToolCallContent.text(formatted))
                            locations.append(ToolCallLocation(path=file_path, line=start_line or 0))
                        case DiffContentItem(path=diff_path, old_text=old, new_text=new):
                            acp_content.append(
                                FileEditToolCallContent(path=diff_path, old_text=old, new_text=new)
                            )
                            locations.append(ToolCallLocation(path=diff_path))
                            state.has_content = True
                        case LocationContentItem(path=loc_path, line=loc_line):
                            location = ToolCallLocation(path=loc_path, line=loc_line)
                            locations.append(location)

                # Build title: use provided title, or format MCP numeric progress
                effective_title = title
                if not effective_title and (progress is not None or message):
                    # MCP-style numeric progress - format into title
                    if progress is not None and total:
                        pct = int(progress / total * 100)
                        effective_title = f"{message} ({pct}%)" if message else f"Progress: {pct}%"
                    elif message:
                        effective_title = message

                # Use the actual status from the event (failed, completed, in_progress)
                yield ToolCallProgress(
                    tool_call_id=tool_call_id,
                    title=effective_title,
                    status=status or "in_progress",
                    content=acp_content or None,
                    locations=locations or None,
                    raw_input=self._format_raw_input(state.raw_input),
                )
                if acp_content:
                    state.has_content = True

            case ToolCallCompleteEvent(
                tool_call_id=tc_id,
                tool_name=tool_name,
                tool_result=result,
                metadata=meta,
            ):
                # ToolCallCompleteEvent is produced by EventMapper from
                # FunctionToolResultEvent. When metadata contains
                # ``is_error=True``, the original part was a
                # RetryPromptPart (tool failure).
                is_error = bool(meta and meta.get("is_error"))
                completion_status: Literal["completed", "failed"] = (
                    "failed" if is_error else "completed"
                )
                tool_state = self._tool_states.get(tc_id)
                if is_error:
                    error_text = str(result) if result else "Tool execution failed"
                    content = ContentToolCallContent.text(f"Error: {error_text}")
                    yield ToolCallProgress(
                        tool_call_id=tc_id,
                        status=completion_status,
                        content=[content],
                    )
                elif tool_state and tool_state.has_content:
                    yield ToolCallProgress(
                        tool_call_id=tc_id,
                        status=completion_status,
                        raw_output=result,
                    )
                else:
                    converted = to_acp_content_blocks(result)
                    content_items = [ContentToolCallContent(content=block) for block in converted]
                    yield ToolCallProgress(
                        tool_call_id=tc_id,
                        status=completion_status,
                        raw_output=result,
                        content=content_items,
                    )
                self._cleanup_tool_state(tc_id)

            case ToolResultMetadataEvent(tool_call_id=tc_id, metadata=_meta):
                # Sidechannel metadata for tool results (e.g., diffs,
                # diagnostics stripped by Claude SDK). Enrich existing
                # tool state if present; otherwise log and skip.
                meta_state = self._tool_states.get(tc_id)
                if meta_state:
                    meta_state.has_content = True
                else:
                    logger.debug(
                        "ToolResultMetadataEvent for unknown tool call",
                        tool_call_id=tc_id,
                    )

            case FinalResultEvent():
                pass  # No notification needed

            case StepUsageEvent(
                step_index=step_index,
                step_usage=step_usage,
                cumulative_usage=cumulative_usage,
            ):
                logger.info(
                    "ACP converter emitting per-step UsageUpdate",
                    step_index=step_index,
                    used=step_usage.total_tokens,
                    cumulative=cumulative_usage.total_tokens,
                )
                # Emit a per-step UsageUpdate with DELTA token values.
                # ACP clients should accumulate these per-step deltas and
                # then REPLACE the running total with the final
                # UsageUpdate from StreamCompleteEvent (which carries
                # cumulative totals).
                acp_usage = _to_acp_usage(step_usage)
                cumulative_acp = _to_acp_usage(cumulative_usage)
                yield UsageUpdate(
                    used=acp_usage.total_tokens,
                    size=cumulative_usage.total_tokens,
                    field_meta={
                        "step_index": step_index,
                        "step_usage": acp_usage.model_dump(exclude_none=True),
                        "cumulative_usage": cumulative_acp.model_dump(exclude_none=True),
                    },
                )

            case StreamCompleteEvent(message=message, cancelled=cancelled):
                request_usage = message.usage
                # Build the final cumulative Usage via the shared helper.
                self.last_usage = _to_acp_usage(request_usage)
                cost_obj: Cost | None = None
                if message.cost_info and message.cost_info.total_cost:
                    cost_obj = Cost(
                        amount=float(message.cost_info.total_cost),
                        currency="USD",
                    )
                # Final turn-level UsageUpdate with cumulative totals.
                # ACP clients that received per-step UsageUpdate notifications
                # (from StepUsageEvent) should REPLACE their running "used"
                # total with this final value, NOT add it on top — this
                # UsageUpdate carries the turn's complete cumulative usage.
                yield UsageUpdate(
                    used=request_usage.total_tokens,
                    size=request_usage.total_tokens,  # best approximation
                    cost=cost_obj,
                )
                # Turn-complete signal: explicit end-of-turn barrier for clients.
                # Based on draft RFD PR #644 (not yet merged into ACP spec).
                # See: https://github.com/agentclientprotocol/agent-client-protocol/pull/644
                async for cancel_update in self.cancel_pending_tools():
                    yield cancel_update
                if self.client_supports_turn_complete:
                    yield TurnCompleteUpdate(
                        stop_reason="cancelled" if cancelled else "end_turn",
                    )

            case PlanUpdateEvent(entries=entries):
                acp_entries = [
                    ACPPlanEntry(content=e.content, priority=e.priority, status=e.status)
                    for e in entries
                ]
                yield AgentPlanUpdate(entries=acp_entries)

            case CompactionEvent(trigger=trigger, phase=phase) if phase == "starting":
                text = get_compaction_text(trigger)
                yield AgentMessageChunk.text(text, message_id=self._get_message_id(event))

            case CompactionEvent(phase="completed"):
                # Signal compaction completion to the client
                yield AgentMessageChunk.text(
                    "\n\n---\n\n✅ **Context compaction complete.**\n\n---\n\n",
                    message_id=self._get_message_id(event),
                )

            case SpawnSessionStart(
                child_session_id=child_session_id,
                source_name=source_name,
                description=description,
                spawn_mechanism=spawn_mechanism,
            ):
                if self.subagent_display_mode == "legacy":
                    icon = "⚡" if spawn_mechanism == "spawn" else "🚀"
                    text = f"\n{icon} **`{source_name}`**: {description}\n"
                    yield AgentMessageChunk.text(text, message_id=self._get_message_id(event))
                    self._child_sessions.add(child_session_id)
                elif self.subagent_display_mode == "zed":
                    tool_call_id = event.tool_call_id or str(uuid.uuid4())
                    self._subagent_tool_call_ids[child_session_id] = tool_call_id
                    field_meta = self._build_subagent_field_meta(
                        child_session_id=child_session_id, message_start_index=0
                    )
                    yield ToolCallStart(
                        tool_call_id=tool_call_id,
                        title=f"{source_name}: {description}" if description else source_name,
                        kind="other",
                        status="pending",
                        field_meta=field_meta,
                    )
                elif self.subagent_display_mode == "qwen":
                    tool_call_id = event.tool_call_id or str(uuid.uuid4())
                    yield ToolCallStart(
                        tool_call_id=tool_call_id,
                        title=f"{source_name}: {description}" if description else source_name,
                        kind="other",
                        status="pending",
                    )

            case RunStartedEvent(run_id=run_id, agent_name=agent_name):
                # ACP has no explicit "run started" notification.
                # Log for debugging; clients infer start from first event.
                logger.debug("Run started", run_id=run_id, agent_name=agent_name)

            case SubAgentEvent(
                source_name=source_name,
                event=inner_event,
                depth=depth,
                child_session_id=child_session_id,
            ):
                # SubAgentEvent wraps events from delegated agents/teams.
                # Child sessions have their own consumer that handles
                # their events directly. This wrapper provides metadata
                # for protocols that want to annotate nested activity.
                # For ACP, we skip re-converting the inner event (it's
                # already handled by the child consumer) and just log.
                logger.debug(
                    "SubAgent event",
                    source_name=source_name,
                    depth=depth,
                    child_session_id=child_session_id,
                    inner_event_type=type(inner_event).__name__,
                )

            case SessionResumeEvent(
                session_id=_sess_id,
                resolved_call_count=call_count,
                source=resume_source,
            ):
                # Backend-only log — no ACP notification needed for resume.
                logger.info(
                    "Session resumed",
                    session_id=_sess_id,
                    resolved_call_count=call_count,
                    source=resume_source,
                )

            case CustomEvent(event_type=ev_type, source=ev_source):
                # Generic custom events — log for debugging, no ACP output
                logger.debug(
                    "Custom event",
                    event_type=ev_type,
                    source=ev_source,
                )

            case PartEndEvent(index=idx, part=ended_part):
                # Part boundary detection — no ACP notification needed,
                # but log for debugging.
                logger.debug(
                    "Part ended",
                    index=idx,
                    part_kind=ended_part.part_kind,
                )

            case OutputToolCallEvent(part=part):
                # Output tool calls (structured output submission) —
                # no ACP notification needed, handled internally by
                # PydanticAI for result validation.
                logger.debug(
                    "Output tool call",
                    tool_name=part.tool_name,
                )

            case OutputToolResultEvent(part=part):
                # Output tool results — no ACP notification needed.
                logger.debug(
                    "Output tool result",
                    tool_name=part.tool_name,
                )

            case RunErrorEvent(message=message, agent_name=agent_name):
                # TurnCompleteUpdate is required here — without it, clients
                # with turn_complete support stay stuck in "running" state.
                # Use "refusal" stop reason so clients know the turn ended
                # abnormally (e.g. elicitation timeout), not successfully.
                agent_prefix = f"[{agent_name}] " if agent_name else ""
                error_text = f"\n\n❌ **Error**: {agent_prefix}{message}\n\n"
                yield AgentMessageChunk.text(error_text, message_id=self._get_message_id(event))
                async for cancel_update in self.cancel_pending_tools():
                    yield cancel_update
                if self.client_supports_turn_complete:
                    yield TurnCompleteUpdate(stop_reason="refusal")

            case RunFailedEvent(run_id=run_id, exception=exc):
                # Display run failure as agent text and signal turn completion.
                # Unlike RunErrorEvent (agent-level), RunFailedEvent indicates
                # the run itself crashed — the session cannot continue.

                # Check if this is a cancellation (session/cancel notification)
                import asyncio

                is_cancellation = isinstance(exc, asyncio.CancelledError) or (
                    isinstance(exc, RuntimeError) and "cancelled" in str(exc).lower()
                )

                stop_reason: Literal["end_turn", "cancelled"] = (
                    "cancelled" if is_cancellation else "end_turn"
                )
                if not is_cancellation:
                    error_text = f"\n\n❌ **Run Failed** [{run_id}]: {exc}\n\n"
                    yield AgentMessageChunk.text(error_text, message_id=self._get_message_id(event))
                async for cancel_update in self.cancel_pending_tools():
                    yield cancel_update
                if self.client_supports_turn_complete:
                    yield TurnCompleteUpdate(stop_reason=stop_reason)

            case ToolCallDeferredEvent(
                tool_call_id=tc_id,
                tool_name=tool_name,
                deferred_strategy=_strategy,
                deferred_handle=deferred_handle,
                status="pending",
            ):
                # Create or get tool state for the deferred call
                state = self._get_or_create_tool_state(tc_id, tool_name, {})
                if not state.started:
                    state.started = True
                    yield ToolCallStart(
                        tool_call_id=tc_id,
                        title=f"Deferred: {tool_name}",
                        kind=state.kind,
                        raw_input=self._format_raw_input(state.raw_input),
                        status="pending",
                        field_meta={"deferred_handle": deferred_handle},
                    )

            case ToolCallDeferredEvent():
                # Deferred events with status "resolved" or "expired" are no-ops
                # on the converter side — resolution is handled by the session pool
                pass

            case ElicitationDeferredEvent(
                deferred_handle=deferred_handle,
                message=message,
                requested_schema=requested_schema,
                mode=mode,
            ):
                # Emit a tool_call_deferred notification with elicitation params
                # in _meta so the ACP client can render an elicitation UI.
                elicitation_call_id = f"elicitation_{deferred_handle}"
                state = self._get_or_create_tool_state(elicitation_call_id, "elicitation", {})
                if not state.started:
                    state.started = True
                    yield ToolCallStart(
                        tool_call_id=elicitation_call_id,
                        title=f"Elicitation: {message}",
                        kind=state.kind,
                        raw_input=self._format_raw_input(state.raw_input),
                        status="pending",
                        field_meta={
                            "deferred_handle": deferred_handle,
                            "elicitation": True,
                            "elicitation_message": message,
                            "elicitation_schema": requested_schema,
                            "elicitation_mode": mode,
                        },
                    )

            case _:
                # Graceful fallback for unknown event types
                # Handles future events like ToolRequiresAuthEvent without crashing
                logger.debug("Unhandled event", event_type=type(event).__name__)
