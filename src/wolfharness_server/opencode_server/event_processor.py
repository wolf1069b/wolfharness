"""Event processor for OpenCode server.

Translates RichAgentStreamEvent objects from the agent event system
into OpenCode SSE Event objects. Uses EventProcessorContext for mutable
state, enabling stateless recursive processing.
"""

from __future__ import annotations

from dataclasses import dataclass
import difflib
from typing import TYPE_CHECKING, Any

from pydantic_ai import FunctionToolCallEvent
from pydantic_ai.messages import (
    PartDeltaEvent as PydanticPartDeltaEvent,
    PartStartEvent,
    TextPart as PydanticTextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart as PydanticToolCallPart,
)

from wolfharness.agents.events import (
    DiffContentItem,
    ElicitationDeferredEvent,
    FileContentItem,
    LocationContentItem,
    RunErrorEvent,
    StepUsageEvent,
    StreamCompleteEvent,
    TextContentItem,
    ToolCallCompleteEvent,
    ToolCallDeferredEvent,
    ToolCallProgressEvent,
    ToolCallStartEvent,
    UserMessageInsertedEvent,
)
from wolfharness.agents.events.infer_info import derive_rich_tool_info
from wolfharness.log import get_logger
from wolfharness.utils import identifiers as identifier
from wolfharness.utils.pydantic_ai_helpers import safe_args_as_dict
from wolfharness.utils.time_utils import now_ms
from wolfharness_server.opencode_server.converters import _convert_params_for_ui
from wolfharness_server.opencode_server.models import (
    MessageUpdatedEvent,
    Part,
    PartDeltaEvent,
    PartUpdatedEvent,
    SessionErrorEvent,
    SessionStatusEvent,
    TimeCreated,
    TokenCache,
    Tokens,
)

# Cross-layer import: McpToolsChangedEvent is an OpenCode SSE event that
# EventProcessor creates from core-layer ChangeEvent(kind="tools_changed").
from wolfharness_server.opencode_server.models.events import McpToolsChangedEvent
from wolfharness_server.opencode_server.models.message import (
    AssistantMessage,
    MessageWithParts,
    UserMessage,
)
from wolfharness_server.opencode_server.models.parts import (
    FilePart,
    ReasoningPart,
    StepFinishPart,
    StepStartPart,
    TextPart,
    TimeStart,
    TimeStartEnd,
    TimeStartEndCompacted,
    TimeStartEndOptional,
    ToolPart,
    ToolStateCompleted,
    ToolStateError,
    ToolStateRunning,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence

    from wolfharness.agents.events import ToolCallContentItem
    from wolfharness.agents.events.events import RichAgentStreamEvent
    from wolfharness.messaging import ChatMessage
    from wolfharness_server.opencode_server.event_processor_context import (
        EventProcessorContext,
    )
    from wolfharness_server.opencode_server.models.events import Event
    from wolfharness_server.opencode_server.models.parts import ToolState
    from wolfharness_server.opencode_server.models.session import SessionStatusType
logger = get_logger(__name__)


@dataclass(frozen=True)
class OpenCodeUserMessageMeta:
    """Protocol-specific metadata for OpenCode user messages.

    Carries serialized Part data (TextPart, FilePart, etc.) so the
    EventProcessor can reconstruct the full user message from the
    EventBus event without the protocol handler broadcasting separately.
    """

    parts: list[dict[str, Any]]
    """Serialized Part data (TextPart, FilePart, etc.) as dicts.

    Each dict is the ``model_dump()`` output of an OpenCode Part model.
    The EventProcessor deserializes each dict back to the appropriate
    Part type using the ``type`` discriminator field.
    """


class EventProcessor:
    """Processes RichAgentStreamEvent objects into OpenCode SSE events.

    Stateless processor that uses EventProcessorContext for all mutable state.
    This design enables recursive processing with different contexts at different
    depths (e.g., for subagent handling).

    The processor yields OpenCode Event objects ready for broadcasting.
    """

    def __init__(self) -> None:
        """Initialize the event processor."""

    @staticmethod
    def create_mcp_tools_changed_event(server: str) -> McpToolsChangedEvent:
        """Create an McpToolsChangedEvent for tool list refresh notification.

        Called by the server's ``_watch_mcp_tool_changes`` task when a
        ``ChangeEvent(kind="tools_changed")`` is received from
        ``McpServerCap.on_change()``. The resulting event is broadcast
        to connected OpenCode clients so they can refresh their tool lists.

        Args:
            server: Name of the MCP server whose tools changed.

        Returns:
            ``McpToolsChangedEvent`` ready for broadcasting.
        """
        return McpToolsChangedEvent.create(server=server)

    async def process(
        self,
        event: RichAgentStreamEvent[Any],
        ctx: EventProcessorContext,
    ) -> AsyncIterator[Event]:
        """Process a single agent event and yield OpenCode SSE events.

        Args:
            event: The agent stream event to process.
            ctx: The event processor context holding mutable state.

        Yields:
            OpenCode Event objects for broadcasting.
        """
        match event:
            case PartStartEvent(part=PydanticTextPart(content=delta)):
                for e in self._process_text_start(ctx, delta):
                    yield e

            case PydanticPartDeltaEvent(delta=TextPartDelta(content_delta=delta)) if delta:
                for e in self._process_text_delta(ctx, delta):
                    yield e

            case PartStartEvent(part=ThinkingPart(content=delta)):
                for e in self._process_thinking_start(ctx, delta):
                    yield e

            case PydanticPartDeltaEvent(delta=ThinkingPartDelta(content_delta=delta)):
                for e in self._process_thinking_delta(ctx, delta):
                    yield e

            case ToolCallStartEvent(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                raw_input=raw_input,
                title=title,
            ):
                for e in self._process_tool_call_start(
                    ctx,
                    tool_name,  # ty: ignore[invalid-argument-type]
                    tool_call_id,  # ty: ignore[invalid-argument-type]
                    raw_input,
                    title,  # ty: ignore[invalid-argument-type]
                ):
                    yield e

            case (
                FunctionToolCallEvent(part=tc_part)
                | PartStartEvent(part=PydanticToolCallPart() as tc_part)
            ) if not ctx.has_tool_part(tc_part.tool_call_id):
                for e in self._process_pydantic_tool_call(ctx, tc_part):
                    yield e
            case (
                FunctionToolCallEvent(part=tc_part)
                | PartStartEvent(part=PydanticToolCallPart() as tc_part)
            ) if ctx.has_tool_part(tc_part.tool_call_id):
                # Tool part already exists (from ToolCallStartEvent), update input if empty
                for e in self._update_tool_call_input(ctx, tc_part):
                    yield e

            case ToolCallProgressEvent(
                tool_call_id=tool_call_id,
                title=title,
                items=items,
                tool_name=tool_name,
                tool_input=event_tool_input,
            ) if tool_call_id:
                for e in self._process_tool_progress(
                    ctx, tool_call_id, title, items, tool_name, event_tool_input
                ):
                    yield e

            case ToolCallDeferredEvent(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                deferred_handle=deferred_handle,
                deferred_strategy=strategy,
            ):
                for e in self._process_tool_deferred(
                    ctx, tool_call_id, tool_name, deferred_handle, strategy
                ):
                    yield e

            case ElicitationDeferredEvent(
                deferred_handle=deferred_handle,
                message=message,
                requested_schema=requested_schema,
                mode=mode,
            ):
                for e in self._process_elicitation_deferred(
                    ctx, deferred_handle, message, requested_schema, mode
                ):
                    yield e

            case ToolCallCompleteEvent(
                tool_call_id=tool_call_id,
                tool_result=result,
                metadata=event_metadata,
            ) if ctx.has_tool_part(tool_call_id):
                for e in self._process_tool_complete(ctx, tool_call_id, result, event_metadata):
                    yield e

            case StepUsageEvent() as step_usage_event:
                logger.info(
                    "OpenCode event processor processing StepUsageEvent",
                    step_index=step_usage_event.step_index,
                    input=step_usage_event.step_usage.input_tokens,
                    output=step_usage_event.step_usage.output_tokens,
                )
                for e in self._process_step_usage(ctx, step_usage_event):
                    yield e

            case StreamCompleteEvent(message=msg, cancelled=cancelled) if msg:
                for e in self._process_stream_complete(ctx, msg):
                    yield e
                status: SessionStatusType = "cancelled" if cancelled else "idle"
                yield SessionStatusEvent.create(
                    session_id=ctx.session_id,
                    status_type=status,
                )

            case RunErrorEvent() as run_error_event:
                yield SessionErrorEvent.create(
                    session_id=ctx.session_id,
                    error_name=run_error_event.code or "RunError",
                    error_message=run_error_event.message,
                )

            case UserMessageInsertedEvent(source="processed"):
                # Processing-time event: do NOT create UserMessage.
                # Split is handled by opencode_event_bridge, not EventProcessor.
                return

            case UserMessageInsertedEvent(
                message_id=mid,
                content=event_content,
                timestamp=ts,
                meta=event_meta,
                source=event_source,
            ):
                async for e in self._process_user_message_inserted(
                    ctx, mid, event_content, ts, event_meta, event_source
                ):
                    yield e

    def _process_text_start(
        self,
        ctx: EventProcessorContext,
        delta: str,
    ) -> Iterator[Event]:
        """Process the start of a text part.

        Args:
            ctx: The event processor context.
            delta: The initial text content.

        Yields:
            PartUpdatedEvent for the created text part.
        """
        ctx.set_text(delta)
        # Close out any active reasoning part before text starts
        if ctx.reasoning_part is not None:
            start_time = ctx.reasoning_part.time.start if ctx.reasoning_part.time else now_ms()
            final_reasoning = ReasoningPart(
                id=ctx.reasoning_part.id,
                message_id=ctx.assistant_msg_id,
                session_id=ctx.session_id,
                text=ctx.reasoning_part.text,
                time=TimeStartEndOptional(start=start_time, end=now_ms()),
                metadata=ctx.reasoning_part.metadata,
            )
            ctx.assistant_msg.update_part(final_reasoning)
            ctx.reasoning_part = final_reasoning
            yield PartUpdatedEvent.create(final_reasoning)
            ctx.reasoning_part = None

        text_part = TextPart(
            id=identifier.ascending("part"),
            message_id=ctx.assistant_msg_id,
            session_id=ctx.session_id,
            text=delta,
        )
        ctx.text_part = text_part
        ctx.assistant_msg.parts.append(text_part)
        yield PartUpdatedEvent.create(text_part)

    def _process_text_delta(
        self,
        ctx: EventProcessorContext,
        delta: str,
    ) -> Iterator[Event]:
        """Process an incremental text delta.

        Args:
            ctx: The event processor context.
            delta: The text delta to append.

        Yields:
            PartUpdatedEvent for the updated text part.
        """
        ctx.accumulate_text(delta)
        if ctx.text_part is not None:
            updated = TextPart(
                id=ctx.text_part.id,
                message_id=ctx.assistant_msg_id,
                session_id=ctx.session_id,
                text=ctx.response_text,
            )
            ctx.assistant_msg.update_part(updated)
            ctx.text_part = updated
            yield PartDeltaEvent.create(
                session_id=ctx.session_id,
                message_id=ctx.assistant_msg_id,
                part_id=updated.id,
                delta=delta,
            )
        else:
            # No text part exists yet (no PartStartEvent received)
            # Create one now with the accumulated text
            text_part = TextPart(
                id=identifier.ascending("part"),
                message_id=ctx.assistant_msg_id,
                session_id=ctx.session_id,
                text=ctx.response_text,
            )
            ctx.text_part = text_part
            ctx.assistant_msg.parts.append(text_part)
            # Part doesn't exist on frontend yet, send full PartUpdatedEvent
            yield PartUpdatedEvent.create(text_part)

    def _process_thinking_start(
        self,
        ctx: EventProcessorContext,
        delta: str | None,
    ) -> Iterator[Event]:
        """Process the start of a thinking/reasoning part.

        Args:
            ctx: The event processor context.
            delta: The initial thinking content.

        Yields:
            PartUpdatedEvent for the created reasoning part.
        """
        # Skip None (no content), but preserve empty strings and whitespace
        if delta is None:
            return

        reasoning_part_id = identifier.ascending("part")
        reasoning_part = ReasoningPart(
            id=reasoning_part_id,
            message_id=ctx.assistant_msg_id,
            session_id=ctx.session_id,
            text=delta,
            time=TimeStartEndOptional(start=now_ms()),
        )
        ctx.reasoning_part = reasoning_part
        ctx.assistant_msg.parts.append(reasoning_part)
        yield PartUpdatedEvent.create(reasoning_part)

    def _process_thinking_delta(
        self,
        ctx: EventProcessorContext,
        delta: str | None,
    ) -> Iterator[Event]:
        """Process an incremental thinking delta.

        Args:
            ctx: The event processor context.
            delta: The thinking delta to append.

        Yields:
            PartUpdatedEvent for the updated or created reasoning part.
        """
        # Skip None (no content), but preserve empty strings and whitespace
        if delta is None:
            return

        if ctx.reasoning_part is not None:
            # Update existing reasoning part
            updated = ReasoningPart(
                id=ctx.reasoning_part.id,
                message_id=ctx.assistant_msg_id,
                session_id=ctx.session_id,
                text=ctx.reasoning_part.text + delta,
                time=ctx.reasoning_part.time,
            )
            ctx.assistant_msg.update_part(updated)
            ctx.reasoning_part = updated
            yield PartDeltaEvent.create(
                session_id=ctx.session_id,
                message_id=ctx.assistant_msg_id,
                part_id=updated.id,
                delta=delta,
            )
        else:
            # No reasoning part exists yet (e.g., after text reset or orphaned delta)
            # Create a new reasoning part with the delta content
            reasoning_part_id = identifier.ascending("part")
            reasoning_part = ReasoningPart(
                id=reasoning_part_id,
                message_id=ctx.assistant_msg_id,
                session_id=ctx.session_id,
                text=delta,
                time=TimeStartEndOptional(start=now_ms()),
            )
            ctx.reasoning_part = reasoning_part
            ctx.assistant_msg.parts.append(reasoning_part)
            # Part doesn't exist on frontend yet, send full PartUpdatedEvent
            yield PartUpdatedEvent.create(reasoning_part)

    def _process_tool_call_start(
        self,
        ctx: EventProcessorContext,
        tool_name: str,
        tool_call_id: str,
        raw_input: dict[str, Any] | None,
        title: str | None,
    ) -> Iterator[Event]:
        """Process the start of a tool call (rich events).

        Args:
            ctx: The event processor context.
            tool_name: The name of the tool being called.
            tool_call_id: The unique identifier for this tool call.
            raw_input: The raw input arguments for the tool.
            title: Optional display title for the tool call.

        Yields:
            PartUpdatedEvent for the created or updated tool part.
        """
        ui_input = _convert_params_for_ui(raw_input) if raw_input else {}

        if ctx.has_tool_part(tool_call_id):
            # Update existing part with the custom title
            existing = ctx.get_tool_part(tool_call_id)
            if existing is not None:
                existing_input = ctx.get_tool_input(tool_call_id) or {}
                ctx.set_tool_input(tool_call_id, ui_input or existing_input)
                tool_input = ctx.get_tool_input(tool_call_id) or {}
                running_state = ToolStateRunning(
                    time=TimeStart(start=ctx.stream_start_ms),
                    input=tool_input,
                    title=title,
                )
                updated = ToolPart(
                    id=existing.id,
                    message_id=existing.message_id,
                    session_id=existing.session_id,
                    tool=existing.tool,
                    call_id=existing.call_id,
                    state=running_state,
                )
                ctx.add_tool_part(tool_call_id, updated)
                ctx.assistant_msg.update_part(updated)
                yield PartUpdatedEvent.create(updated)
        else:
            # Create new tool part
            ctx.set_tool_input(tool_call_id, ui_input)
            ctx.set_tool_output(tool_call_id, "")
            ts = TimeStart(start=now_ms())
            tool_state = ToolStateRunning(time=ts, input=ui_input, title=title)
            tool_part = ToolPart(
                id=identifier.ascending("part"),
                message_id=ctx.assistant_msg_id,
                session_id=ctx.session_id,
                tool=tool_name,
                call_id=tool_call_id,
                state=tool_state,
            )
            ctx.add_tool_part(tool_call_id, tool_part)
            ctx.assistant_msg.parts.append(tool_part)
            yield PartUpdatedEvent.create(tool_part)

    def _process_tool_deferred(
        self,
        ctx: EventProcessorContext,
        tool_call_id: str,
        tool_name: str,
        deferred_handle: str,
        strategy: str,
    ) -> Iterator[Event]:
        """Process a deferred tool call event.

        Produces a ToolPart with state=ToolStateRunning and deferred metadata.
        If a ToolPart for the given tool_call_id already exists with a completed
        or error state, the event is deduplicated (skipped).

        Args:
            ctx: The event processor context.
            tool_call_id: The unique identifier for the deferred tool call.
            tool_name: The name of the tool that was deferred.
            deferred_handle: Correlation ID for resolving the deferred call.
            strategy: How the agent handles the deferral (block/continue/stream).

        Yields:
            PartUpdatedEvent for the created deferred tool part, or nothing
            if the tool part already exists in a completed/error state.
        """
        # Deduplication: skip if ToolPart already exists and is completed/error
        existing = ctx.get_tool_part(tool_call_id)
        if existing is not None:
            match existing.state:
                case ToolStateCompleted() | ToolStateError():
                    return  # Already finalized, skip replay
                case _:
                    pass  # Running or not yet in final state, continue

        title = f"[Deferred] {tool_name}"
        if strategy and strategy != "block":
            title = f"[Deferred:{strategy}] {tool_name}"

        ts = TimeStart(start=now_ms())
        tool_state = ToolStateRunning(
            time=ts,
            input={},
            title=title,
        )
        metadata: dict[str, Any] = {
            "deferred": True,
            "deferred_handle": deferred_handle,
        }
        tool_part = ToolPart(
            id=identifier.ascending("part"),
            message_id=ctx.assistant_msg_id,
            session_id=ctx.session_id,
            tool=tool_name,
            call_id=tool_call_id,
            state=tool_state,
            metadata=metadata,
        )
        ctx.add_tool_part(tool_call_id, tool_part)
        ctx.assistant_msg.parts.append(tool_part)
        yield PartUpdatedEvent.create(tool_part)

    def _process_elicitation_deferred(
        self,
        ctx: EventProcessorContext,
        deferred_handle: str,
        message: str,
        requested_schema: dict[str, Any],
        mode: str,
    ) -> Iterator[Event]:
        """Process an elicitation deferred event.

        Produces a ToolPart with state=ToolStateRunning and elicitation metadata
        so the OpenCode client can render an elicitation form.

        Args:
            ctx: The event processor context.
            deferred_handle: Correlation ID for resolving the deferred elicitation.
            message: Human-readable message describing what is being elicited.
            requested_schema: JSON schema describing the expected response structure.
            mode: Elicitation mode hint for client rendering.

        Yields:
            PartUpdatedEvent for the created elicitation tool part.
        """
        elicitation_call_id = f"elicitation_{deferred_handle}"

        # Deduplication: skip if ToolPart already exists and is finalized
        existing = ctx.get_tool_part(elicitation_call_id)
        if existing is not None:
            match existing.state:
                case ToolStateCompleted() | ToolStateError():
                    return
                case _:
                    pass

        ts = TimeStart(start=now_ms())
        tool_state = ToolStateRunning(
            time=ts,
            input={},
            title=f"[Elicitation] {message}",
        )
        metadata: dict[str, Any] = {
            "deferred": True,
            "deferred_handle": deferred_handle,
            "elicitation": True,
            "elicitation_message": message,
            "elicitation_schema": requested_schema,
            "elicitation_mode": mode,
        }
        tool_part = ToolPart(
            id=identifier.ascending("part"),
            message_id=ctx.assistant_msg_id,
            session_id=ctx.session_id,
            tool="elicitation",
            call_id=elicitation_call_id,
            state=tool_state,
            metadata=metadata,
        )
        ctx.add_tool_part(elicitation_call_id, tool_part)
        ctx.assistant_msg.parts.append(tool_part)
        yield PartUpdatedEvent.create(tool_part)

    def _process_pydantic_tool_call(
        self,
        ctx: EventProcessorContext,
        tc_part: PydanticToolCallPart,
    ) -> Iterator[Event]:
        """Process a pydantic-ai tool call event (fallback for pydantic-ai agents).

        Args:
            ctx: The event processor context.
            tc_part: The pydantic-ai tool call part.

        Yields:
            PartUpdatedEvent for the created tool part.
        """
        tool_call_id = tc_part.tool_call_id
        tool_name = tc_part.tool_name
        raw_input = safe_args_as_dict(tc_part)
        ui_input = _convert_params_for_ui(raw_input)

        ctx.set_tool_input(tool_call_id, ui_input)
        ctx.set_tool_output(tool_call_id, "")

        rich_info = derive_rich_tool_info(tool_name, raw_input)
        ts = TimeStart(start=now_ms())
        tool_state = ToolStateRunning(time=ts, input=ui_input, title=rich_info.title)
        tool_part = ToolPart(
            id=identifier.ascending("part"),
            message_id=ctx.assistant_msg_id,
            session_id=ctx.session_id,
            tool=tool_name,
            call_id=tool_call_id,
            state=tool_state,
        )
        ctx.add_tool_part(tool_call_id, tool_part)
        ctx.assistant_msg.parts.append(tool_part)
        yield PartUpdatedEvent.create(tool_part)

    def _update_tool_call_input(
        self,
        ctx: EventProcessorContext,
        tc_part: PydanticToolCallPart,
    ) -> Iterator[Event]:
        """Update existing tool part with input from pydantic ToolCallPart.

        This handles the case where ToolCallStartEvent (from ctx.events.tool_call_start())
        arrives before PartStartEvent, creating an empty tool part that needs to be
        populated with actual arguments from the pydantic event.

        Args:
            ctx: The event processor context.
            tc_part: The pydantic-ai tool call part containing args.

        Yields:
            PartUpdatedEvent if the tool part was updated with new input.
        """
        tool_call_id = tc_part.tool_call_id
        existing_input = ctx.get_tool_input(tool_call_id) or {}

        # Only update if current input is empty and we have args
        if not existing_input and tc_part.args:
            raw_input = safe_args_as_dict(tc_part)
            if raw_input:
                ui_input = _convert_params_for_ui(raw_input)
                ctx.set_tool_input(tool_call_id, ui_input)

                # Update the existing tool part with new input
                existing = ctx.get_tool_part(tool_call_id)
                if existing is not None:
                    existing_title = _extract_title_from_tool_state(existing.state)
                    tool_state = ToolStateRunning(
                        time=TimeStart(start=now_ms()),
                        input=ui_input,
                        title=existing_title or tc_part.tool_name,
                    )
                    updated = ToolPart(
                        id=existing.id,
                        message_id=existing.message_id,
                        session_id=existing.session_id,
                        tool=existing.tool,
                        call_id=existing.call_id,
                        state=tool_state,
                    )
                    ctx.add_tool_part(tool_call_id, updated)
                    ctx.assistant_msg.update_part(updated)
                    yield PartUpdatedEvent.create(updated)

    def _process_tool_progress(
        self,
        ctx: EventProcessorContext,
        tool_call_id: str,
        title: str | None,
        items: Sequence[ToolCallContentItem],
        tool_name: str | None,
        event_tool_input: dict[str, Any] | None,
    ) -> Iterator[Event]:
        """Process tool call progress updates.

        Args:
            ctx: The event processor context.
            tool_call_id: The unique identifier for this tool call.
            title: Optional display title for the tool call.
            items: Content items representing progress output.
            tool_name: Optional tool name (for new tool parts).
            event_tool_input: Optional input parameters (for new tool parts).

        Yields:
            PartUpdatedEvent for the updated or created tool part.
        """
        new_output = ""
        for item in items:
            match item:
                case TextContentItem(text=text):
                    new_output += text
                case FileContentItem(content=content):
                    new_output += content
                case LocationContentItem():
                    pass
                case DiffContentItem(path=path, old_text=old, new_text=new):
                    # Convert structured diff to unified diff text for TUI rendering.
                    # The opencode TUI Edit component reads metadata.diff as a string
                    # (createTwoFilesPatch format). Accumulate per tool_call_id;
                    # _process_tool_complete merges it into ToolStateCompleted.metadata.
                    #
                    # Use lineterm="" + splitlines(keepends=False) so no line carries
                    # an embedded \n. Join with \n and add trailing \n to ensure every
                    # line is properly terminated — the npm "diff" parser strictly
                    # validates hunk line counts and fails on missing terminators.
                    old_str = old or ""
                    new_str = new or ""
                    diff_iter = difflib.unified_diff(
                        old_str.splitlines(keepends=False),
                        new_str.splitlines(keepends=False),
                        fromfile=path,
                        tofile=path,
                        lineterm="",
                    )
                    diff_text = "\n".join(diff_iter)
                    if diff_text:
                        diff_text += "\n"
                    ctx.tool_diffs[tool_call_id] = diff_text

        if new_output:
            ctx.append_tool_output(tool_call_id, new_output)

        if ctx.has_tool_part(tool_call_id):
            existing = ctx.get_tool_part(tool_call_id)
            if existing is not None:
                existing_title = _extract_title_from_tool_state(existing.state)
                # Update tool input from progress event if existing input is empty.
                # This happens when the model streams tool call arguments: the
                # initial ToolCallStartEvent arrives with empty raw_input, and
                # the EventMapper emits a ToolCallProgressEvent with the complete
                # tool_input once args are assembled.
                if event_tool_input and not ctx.get_tool_input(tool_call_id):
                    ui_input = _convert_params_for_ui(event_tool_input)
                    ctx.set_tool_input(tool_call_id, ui_input)
                tool_input = ctx.get_tool_input(tool_call_id) or {}
                accumulated_output = ctx.get_tool_output(tool_call_id)
                tool_state = ToolStateRunning(
                    time=TimeStart(start=now_ms()),
                    title=title or existing_title,
                    input=tool_input,
                    metadata={"output": accumulated_output} if accumulated_output else None,
                )
                updated = ToolPart(
                    id=existing.id,
                    message_id=existing.message_id,
                    session_id=existing.session_id,
                    tool=existing.tool,
                    call_id=existing.call_id,
                    state=tool_state,
                )
                ctx.add_tool_part(tool_call_id, updated)
                ctx.assistant_msg.update_part(updated)
                yield PartUpdatedEvent.create(updated)
        else:
            # Create new tool part from progress event
            ui_input = _convert_params_for_ui(event_tool_input) if event_tool_input else {}
            ctx.set_tool_input(tool_call_id, ui_input)
            accumulated_output = ctx.get_tool_output(tool_call_id)
            tool_state = ToolStateRunning(
                time=TimeStart(start=now_ms()),
                input=ui_input,
                title=title or tool_name or "Running...",
                metadata={"output": accumulated_output} if accumulated_output else None,
            )
            tool_part = ToolPart(
                id=identifier.ascending("part"),
                message_id=ctx.assistant_msg_id,
                session_id=ctx.session_id,
                tool=tool_name or "unknown",
                call_id=tool_call_id,
                state=tool_state,
            )
            ctx.add_tool_part(tool_call_id, tool_part)
            ctx.assistant_msg.parts.append(tool_part)
            yield PartUpdatedEvent.create(tool_part)

    def _process_tool_complete(
        self,
        ctx: EventProcessorContext,
        tool_call_id: str,
        result: Any,
        event_metadata: dict[str, Any] | None,
    ) -> Iterator[Event]:
        """Process tool call completion.

        Args:
            ctx: The event processor context.
            tool_call_id: The unique identifier for this tool call.
            result: The result of the tool execution.
            event_metadata: Optional metadata about the tool execution.

        Yields:
            PartUpdatedEvent for the completed tool part.
        """
        existing = ctx.get_tool_part(tool_call_id)
        if existing is None:
            return

        result_str = _format_tool_output(result)
        tool_input = ctx.get_tool_input(tool_call_id) or {}
        is_error = isinstance(result, dict) and result.get("error")
        start = ctx.stream_start_ms

        new_state: ToolStateCompleted | ToolStateError
        if is_error:
            t = TimeStartEnd(start=start, end=now_ms())
            error_string = str(result.get("error", "Unknown error"))
            new_state = ToolStateError(error=error_string, input=tool_input, time=t)
        else:
            # Merge accumulated diff text (from DiffContentItem in progress events)
            # into completion metadata so the TUI Edit component can render it.
            # Also set diagnostics=[] when diff content exists: this triggers the
            # Write component's code-block branch (props.metadata.diagnostics !==
            # undefined) so the written content is visible instead of "Preparing
            # write...". An empty diagnostics array renders no error messages.
            diff_text = ctx.tool_diffs.pop(tool_call_id, "")
            merged_metadata: dict[str, Any] = dict(event_metadata or {})
            if diff_text:
                merged_metadata["diff"] = diff_text
                merged_metadata.setdefault("diagnostics", [])
            new_state = ToolStateCompleted(
                title="Completed",
                input=tool_input,
                output=result_str,
                metadata=merged_metadata,
                time=TimeStartEndCompacted(start=start, end=now_ms()),
            )

        updated = ToolPart(
            id=existing.id,
            message_id=existing.message_id,
            session_id=existing.session_id,
            tool=existing.tool,
            call_id=existing.call_id,
            state=new_state,
        )
        ctx.add_tool_part(tool_call_id, updated)
        ctx.assistant_msg.update_part(updated)
        yield PartUpdatedEvent.create(updated)

    def _process_step_usage(
        self,
        ctx: EventProcessorContext,
        event: StepUsageEvent,
    ) -> Iterator[Event]:
        """Process a per-step usage event into a StepFinishPart.

        Emits a ``StepFinishPart`` with token counts from ``step_usage``
        and the ``step_index`` from the event.  This is distinct from the
        final cumulative ``StepFinishPart`` emitted by
        ``_process_stream_complete`` / ``finalize()`` — both are emitted
        in a turn that has per-step usage.

        Args:
            ctx: The event processor context.
            event: The step usage event with per-step and cumulative usage.

        Yields:
            ``PartUpdatedEvent`` for the created ``StepFinishPart``.
        """
        step_usage = event.step_usage
        details = step_usage.details or {}
        reasoning_tokens = details.get("reasoning_tokens", 0)

        cache = TokenCache(
            read=step_usage.cache_read_tokens or 0,
            write=step_usage.cache_write_tokens or 0,
        )
        input_tokens = step_usage.input_tokens or 0
        output_tokens = step_usage.output_tokens or 0
        total = input_tokens + output_tokens + reasoning_tokens + cache.read + cache.write
        tokens = Tokens(
            cache=cache,
            input=input_tokens,
            output=output_tokens,
            reasoning=reasoning_tokens,
            total=total,
        )
        # Emit a per-step StepStartPart so the client sees the step
        # boundary.  The turn-level StepStartPart is emitted at turn
        # start; per-step ones ensure each LLM call step gets a proper
        # start marker that pairs with the StepFinishPart below.
        step_start = StepStartPart(
            id=identifier.ascending("part"),
            message_id=ctx.assistant_msg_id,
            session_id=ctx.session_id,
        )
        ctx.assistant_msg.parts.append(step_start)
        yield PartUpdatedEvent.create(step_start)

        step_finish = StepFinishPart(
            id=identifier.ascending("part"),
            message_id=ctx.assistant_msg_id,
            session_id=ctx.session_id,
            tokens=tokens,
            cost=0.0,
            step_index=event.step_index,
        )
        ctx.assistant_msg.parts.append(step_finish)
        yield PartUpdatedEvent.create(step_finish)

        # Update the message-level `tokens` field with per-step delta
        # values.  The TUI sidebar reads this field to show per-step
        # token cost.  Using per-step deltas (not cumulative) so the
        # user sees the actual cost of each step rather than the
        # running total which grows quickly because each LLM request
        # re-sends the entire context (system prompt + history).
        ctx.update_tokens(input_tokens, output_tokens)
        if isinstance(ctx.assistant_msg.info, AssistantMessage):
            ctx.assistant_msg.info.tokens = Tokens(
                cache=cache,
                input=input_tokens,
                output=output_tokens,
                reasoning=reasoning_tokens,
            )
            yield MessageUpdatedEvent.create(ctx.assistant_msg.info)

    def _process_stream_complete(
        self,
        ctx: EventProcessorContext,
        msg: ChatMessage[Any],
    ) -> Iterator[Event]:
        """Process stream completion and update token/cost tracking.

        Args:
            ctx: The event processor context.
            msg: The completed chat message with usage and cost info.

        Yields:
            Final events including text part timing update and step finish part.
        """
        if not ctx.response_text:
            final_content = _message_content_to_text(msg.content)
            if final_content:
                ctx.set_text(final_content)

        # Update token and cost tracking from the message
        if msg.usage:
            ctx.update_tokens(
                msg.usage.input_tokens or 0,
                msg.usage.output_tokens or 0,
            )
        if msg.cost_info and msg.cost_info.total_cost:
            ctx.update_cost(float(msg.cost_info.total_cost))

        # Update model info from the real inference result.
        # If the model changed (e.g., child session initially showed parent's
        # model), emit a MessageUpdatedEvent so the TUI receives the correction.
        model_changed = False
        if isinstance(ctx.assistant_msg.info, AssistantMessage):
            from wolfharness_server.shared.model_utils import resolve_model_info_from_response

            resolved_model_id, resolved_provider_id = resolve_model_info_from_response(
                msg.model_name, msg.provider_name, ctx.state.model_variants
            )
            if resolved_model_id != ctx.assistant_msg.info.model_id:
                ctx.assistant_msg.info.model_id = resolved_model_id
                model_changed = True
            if resolved_provider_id != ctx.assistant_msg.info.provider_id:
                ctx.assistant_msg.info.provider_id = resolved_provider_id
                model_changed = True
            if model_changed:
                yield MessageUpdatedEvent.create(ctx.assistant_msg.info)

        response_time = now_ms()
        start = ctx.stream_start_ms

        # Close out any active reasoning part before finalizing text
        if ctx.reasoning_part is not None:
            reasoning_start = ctx.reasoning_part.time.start if ctx.reasoning_part.time else start
            final_reasoning = ReasoningPart(
                id=ctx.reasoning_part.id,
                message_id=ctx.assistant_msg_id,
                session_id=ctx.session_id,
                text=ctx.reasoning_part.text,
                time=TimeStartEndOptional(start=reasoning_start, end=response_time),
                metadata=ctx.reasoning_part.metadata,
            )
            ctx.assistant_msg.update_part(final_reasoning)
            ctx.reasoning_part = None
            yield PartUpdatedEvent.create(final_reasoning)

        # Final text part
        if ctx.response_text and ctx.text_part is None:
            # Text was never streamed incrementally — create a text part now
            text_part = TextPart(
                id=identifier.ascending("part"),
                message_id=ctx.assistant_msg_id,
                session_id=ctx.session_id,
                text=ctx.response_text,
                time=TimeStartEndOptional(start=start, end=response_time),
            )
            ctx.assistant_msg.parts.append(text_part)
            yield PartUpdatedEvent.create(text_part)
        elif ctx.text_part is not None:
            # Update streamed text part with final timing
            final_text_part = TextPart(
                id=ctx.text_part.id,
                message_id=ctx.assistant_msg_id,
                session_id=ctx.session_id,
                text=ctx.response_text,
                time=TimeStartEndOptional(start=start, end=response_time),
            )
            ctx.assistant_msg.update_part(final_text_part)
            yield PartUpdatedEvent.create(final_text_part)

        # Step finish part
        cache = TokenCache(read=0, write=0)
        tokens = Tokens(
            cache=cache,
            input=ctx.input_tokens,
            output=ctx.output_tokens,
            reasoning=0,
        )
        step_finish = StepFinishPart(
            id=identifier.ascending("part"),
            message_id=ctx.assistant_msg_id,
            session_id=ctx.session_id,
            tokens=tokens,
            cost=ctx.total_cost,
        )
        ctx.assistant_msg.parts.append(step_finish)
        yield PartUpdatedEvent.create(step_finish)

    async def _process_user_message_inserted(
        self,
        ctx: EventProcessorContext,
        message_id: str,
        content: str | list[Any],
        timestamp: float,
        meta: Any = None,
        source: str = "internal",
    ) -> AsyncIterator[Event]:
        """Process a UserMessageInsertedEvent into OpenCode SSE events.

        Creates a ``UserMessage`` with parts from the event, appends it to
        the session state, and yields ``MessageUpdatedEvent`` and
        ``PartUpdatedEvent`` for each part.

        When ``meta`` is an :class:`OpenCodeUserMessageMeta`, uses
        ``meta.parts`` to deserialize rich parts (TextPart, FilePart, etc.).
        When ``meta`` is ``None``, falls back to text-only ``content`` →
        ``TextPart``.

        For ``source="accepted"`` messages (fire-and-forget from
        ``steer()``/``followup()``), both ``MessageUpdatedEvent`` and
        ``PartUpdatedEvent`` are yielded because there is no ``sync()``
        to load parts from DB.

        ``source="processed"`` events never reach this method — they are
        filtered out by the match-case in ``process()`` and handled
        exclusively by ``opencode_event_bridge`` for steer split.

        Args:
            ctx: The event processor context.
            message_id: Unique ID for the inserted user message.
            content: Message content — plain text or multi-modal part list.
            timestamp: Wall-clock time the event was created (epoch seconds).
            meta: Optional protocol-specific metadata carrying serialized
                Part data for rich user message reconstruction.
            source: Where the message originated — ``"accepted"``
                (fire-and-forget emission from steer()/followup()).

        Yields:
            ``MessageUpdatedEvent`` for the user message, followed by
            ``PartUpdatedEvent`` for each part.
        """
        # Convert epoch seconds to milliseconds for OpenCode's TimeCreated
        created_ms = int(timestamp * 1000)

        user_message = UserMessage(
            id=message_id,
            session_id=ctx.session_id,
            time=TimeCreated(created=created_ms),
        )
        user_msg_with_parts = MessageWithParts(info=user_message)

        # Reconstruct parts from meta or fall back to text-only content.
        if isinstance(meta, OpenCodeUserMessageMeta):
            for part_dict in meta.parts:
                part = _deserialize_part(part_dict, user_message.id, ctx.session_id)
                if part is not None:
                    user_msg_with_parts.parts.append(part)
        elif isinstance(content, str):
            if content:
                user_msg_with_parts.add_text_part(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    if item:
                        user_msg_with_parts.add_text_part(item)
                elif isinstance(item, dict) and "text" in item:
                    text_val = item["text"]
                    if isinstance(text_val, str) and text_val:
                        user_msg_with_parts.add_text_part(text_val)
                elif hasattr(item, "parts") and isinstance(item.parts, list):
                    # Handle ModelRequest containing SystemPromptPart etc.
                    for part in item.parts:
                        if hasattr(part, "content") and isinstance(part.content, str):
                            user_msg_with_parts.add_text_part(part.content)
                elif hasattr(item, "content") and isinstance(item.content, str):
                    # Handle pydantic-ai ModelRequestPart (e.g. SystemPromptPart)
                    # by extracting its content as text for TUI display.
                    user_msg_with_parts.add_text_part(item.content)

        # Append to session state
        from wolfharness_server.opencode_server.opencode_message_bridge import (
            append_message_to_session,
        )

        await append_message_to_session(ctx.state, ctx.session_id, user_msg_with_parts)

        # Always yield MessageUpdatedEvent so the TUI sees the message info
        yield MessageUpdatedEvent.create(user_message)

        # P1: Always yield PartUpdatedEvent for each part, regardless of
        # source. The TUI has no optimistic mechanism — it relies entirely
        # on SSE events for parts. Without PartUpdatedEvent, user messages
        # appear empty after the initial sync() (which only runs once per
        # session). Part IDs from _deserialize_part() preserve the original
        # IDs from meta (line 1077), so they match DB-stored parts — no
        # duplicate risk for new messages.
        for part in user_msg_with_parts.parts:
            yield PartUpdatedEvent.create(part)


def _deserialize_part(
    part_dict: dict[str, Any],
    message_id: str,
    session_id: str,
) -> Part | None:
    """Deserialize a Part dict back to the appropriate Part type.

    Uses the ``type`` discriminator field to select the correct Part
    subclass. Overrides ``id``, ``message_id``, and ``session_id`` to
    ensure the part belongs to the given message and session.

    Args:
        part_dict: The serialized Part data (from ``model_dump()``).
        message_id: The message ID to assign to the part.
        session_id: The session ID to assign to the part.

    Returns:
        The deserialized Part, or ``None`` if the type is unknown.
    """
    part_type = part_dict.get("type", "")
    # Map type discriminator to Part subclass.
    type_to_cls: dict[str, type[Part]] = {
        "text": TextPart,
        "file": FilePart,
    }
    cls = type_to_cls.get(part_type)
    if cls is None:
        return None
    # Override identity fields to ensure consistency.
    data = {
        **part_dict,
        "id": part_dict.get("id", identifier.ascending("part")),
        "message_id": message_id,
        "session_id": session_id,
    }
    return cls.model_validate(data)


def _extract_title_from_tool_state(state: ToolState) -> str:
    """Extract the title from a tool state without getattr.

    Args:
        state: The tool state to extract title from.

    Returns:
        The title string or empty string if no title available.
    """
    match state:
        case ToolStateRunning(title=title):
            return title or ""
        case ToolStateCompleted(title=title):
            return title or ""
        case ToolStateError() | _:
            return ""


def _format_tool_output(result: Any) -> str:
    """Format a tool result for display in OpenCode ToolStateCompleted.output.

    Mirrors the type-branch logic in ``chat_message_to_opencode`` so streaming
    and persistence paths produce identical output.

    Args:
        result: The raw tool result (str, dict, list, or other).

    Returns:
        A display-ready string.
    """
    import anyenv

    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, (dict, list)):
        try:
            return anyenv.dump_json(result, indent=True)
        except Exception:  # noqa: BLE001
            return str(result)
    return str(result)


def _message_content_to_text(content: Any) -> str:
    """Convert final `ChatMessage` content into display text."""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        from wolfharness.agents.native_agent.helpers import _summarize_content_block

        return " ".join(_summarize_content_block(c) for c in content)
    from wolfharness.agents.native_agent.helpers import _summarize_content_block

    return _summarize_content_block(content)
