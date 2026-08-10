"""Event stream events.

Unified event system using ToolCallProgressEvent for all tool-related progress updates.
Rich content (terminals, diffs, locations) is conveyed through the items field.

## UI Content vs Agent Return Values (ACP Protocol)

Tools can control what the UI displays separately from what the agent receives:

- Emit `tool_call_progress` with content items for UI display
- Return raw/structured data for the agent
- Session layer populates `content` (for UI) and `raw_output` (for agent) separately

Use `replace_content=True` for streaming/final content that should replace previous state.
If no content is emitted, the return value is automatically converted for UI (fallback).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai import (
    AgentStreamEvent,
    PartDeltaEvent as PyAIPartDeltaEvent,
    PartStartEvent as PyAIPartStartEvent,
    RunUsage,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPartDelta,
)

from wolfharness.messaging import ChatMessage  # noqa: TC001


if TYPE_CHECKING:
    from collections.abc import Sequence

    from wolfharness.lifecycle.types import RunState
    from wolfharness.messaging.messages import TokenCost
    from wolfharness.tools.base import ToolKind
    from wolfharness.utils.todos import PlanEntry


ToolCallStatus = Literal["pending", "in_progress", "completed", "failed"]
"""Status of a tool call."""


# Lifecycle events (aligned with AG-UI protocol)


@dataclass(kw_only=True)
class PartStartEvent(PyAIPartStartEvent):
    """Part start event."""

    session_id: str = ""
    """ID of the session that emitted this event."""

    message_id: str = ""
    """ID of the message this event belongs to."""

    @classmethod
    def thinking(cls, index: int, content: str, *, message_id: str = "") -> PartStartEvent:
        return cls(index=index, part=ThinkingPart(content=content), message_id=message_id)

    @classmethod
    def text(cls, index: int, content: str, *, message_id: str = "") -> PartStartEvent:
        return cls(index=index, part=TextPart(content=content), message_id=message_id)


@dataclass(kw_only=True)
class PartDeltaEvent(PyAIPartDeltaEvent):
    """Part delta event."""

    session_id: str = ""
    """ID of the session that emitted this event."""

    message_id: str = ""
    """ID of the message this event belongs to."""

    @classmethod
    def thinking(cls, index: int, content: str, *, message_id: str = "") -> PartDeltaEvent:
        return cls(
            index=index, delta=ThinkingPartDelta(content_delta=content), message_id=message_id
        )

    @classmethod
    def text(cls, index: int, content: str, *, message_id: str = "") -> PartDeltaEvent:
        return cls(index=index, delta=TextPartDelta(content_delta=content), message_id=message_id)

    @classmethod
    def tool_call(
        cls, index: int, content: str, tool_call_id: str, *, message_id: str = ""
    ) -> PartDeltaEvent:
        delta = ToolCallPartDelta(args_delta=content, tool_call_id=tool_call_id)
        return cls(index=index, delta=delta, message_id=message_id)


@dataclass(kw_only=True)
class RunStartedEvent:
    """Signals the start of an agent run."""

    session_id: str = ""
    """ID of the session."""
    run_id: str
    """ID of the agent run (unique per request/response cycle)."""
    agent_name: str | None = None
    """Name of the agent starting the run."""
    parent_session_id: str | None = None
    """ID of the parent session when this is a subagent run."""
    event_kind: Literal["run_started"] = "run_started"
    """Event type identifier."""


@dataclass(frozen=True, kw_only=True)
class StepErrorMetadata:
    """Diagnostic metadata for step-level errors.

    Captures the pydantic-ai node context and exception details when
    a step fails during ``NativeTurn.execute()``. Populated only at the
    generic exception handler inside the while loop; the outer catch in
    ``_execute_turn()`` leaves ``step_error`` as ``None`` because no
    node context is available.
    """

    node_type: str
    """Type name of the pydantic-ai node that was executing (e.g. ``ModelRequestNode``)."""

    exception_type: str
    """Type name of the exception that caused the error."""

    exception_message: str
    """Message from the exception that caused the error."""


@dataclass(kw_only=True)
class RunErrorEvent:
    """Signals an error during an agent run."""

    message: str
    """Error message."""
    code: str | None = None
    """Error code."""
    run_id: str | None = None
    """ID of the agent run that failed."""
    agent_name: str | None = None
    """Name of the agent that errored."""
    step_error: StepErrorMetadata | None = None
    """Diagnostic metadata from the step that failed, if available."""
    event_kind: Literal["run_error"] = "run_error"
    """Event type identifier."""


@dataclass(kw_only=True)
class RunFailedEvent:
    """Event indicating a run failed with an error."""

    run_id: str
    """ID of the agent run that failed."""
    session_id: str
    """ID of the session the run belonged to."""
    exception: BaseException
    """The exception that caused the failure."""
    event_kind: Literal["run_failed"] = "run_failed"
    """Event type identifier."""


@dataclass(kw_only=True)
class ToastInfo:
    """Toast notification from an agent.

    Emitted via the state_updated signal to display non-chat notifications
    (errors, warnings, info) without polluting the conversation history.
    """

    message: str
    """Toast message text."""
    level: Literal["error", "warning", "info", "success"] = "info"
    """Toast severity level."""
    duration: int | None = None
    """Display duration in milliseconds; None for persistent."""
    action: dict[str, str] | None = None
    """Optional action button {label: ..., command: ...}."""


# Unified tool call content models (dataclass versions of ACP schema models)


@dataclass(kw_only=True)
class TerminalContentItem:
    """Embed a terminal for live output display."""

    type: Literal["terminal"] = "terminal"
    """Content type identifier."""
    terminal_id: str
    """The ID of the terminal being embedded."""


@dataclass(kw_only=True)
class DiffContentItem:
    """File modification shown as a diff."""

    type: Literal["diff"] = "diff"
    """Content type identifier."""
    path: str
    """The file path being modified."""
    old_text: str | None = None
    """The original content (None for new files)."""
    new_text: str
    """The new content after modification."""


@dataclass(kw_only=True)
class LocationContentItem:
    """A file location being accessed or modified.

    Note: line defaults to 0 (not None) to ensure ACP clients render clickable paths.
    The ACP spec allows None, but some clients (e.g., Claude Code) require the field present.
    """

    type: Literal["location"] = "location"
    """Content type identifier."""
    path: str
    """The file path being accessed or modified."""
    line: int = 0
    """Line number within the file (0 = beginning/unspecified)."""


@dataclass(kw_only=True)
class TextContentItem:
    """Simple text content."""

    type: Literal["text"] = "text"
    """Content type identifier."""
    text: str
    """The text content."""


@dataclass(kw_only=True)
class FileContentItem:
    """File content with metadata for rich display.

    Carries structured data about file content. Formatting (e.g., Zed-style
    code blocks) happens at the ACP layer based on client capabilities.
    """

    type: Literal["file"] = "file"
    """Content type identifier."""
    content: str
    """The file content."""
    path: str
    """The file path."""
    language: str | None = None
    """Language for syntax highlighting (inferred from path if not provided)."""
    start_line: int | None = None
    """Starting line number (1-based) if showing a range."""
    end_line: int | None = None
    """Ending line number (1-based) if showing a range."""


# Union type for all tool call content items
ToolCallContentItem = (
    TerminalContentItem | DiffContentItem | LocationContentItem | TextContentItem | FileContentItem
)


@dataclass(kw_only=True)
class StreamCompleteEvent[TContent]:
    """Event indicating streaming is complete with final message."""

    message: ChatMessage[TContent]
    """The final chat message with all metadata."""
    cancelled: bool = False
    """Whether the run was cancelled before completion."""
    session_id: str = ""
    """ID of the session that emitted this event."""
    event_kind: Literal["stream_complete"] = "stream_complete"
    """Event type identifier."""


@dataclass(kw_only=True)
class ToolCallStartEvent:
    """Event indicating a tool call has started with rich ACP metadata."""

    tool_call_id: str
    """The ID of the tool call."""
    tool_name: str
    """The name of the tool being called."""
    title: str
    """Human-readable title describing what the tool is doing."""
    kind: ToolKind = "other"
    """Tool kind (read, edit, delete, move, search, execute, think, fetch, other)."""
    content: list[ToolCallContentItem] = field(default_factory=list)
    """Content produced by the tool call."""
    locations: list[LocationContentItem] = field(default_factory=list)
    """File locations affected by this tool call."""
    raw_input: dict[str, Any] = field(default_factory=dict)
    """The raw input parameters sent to the tool."""
    session_id: str = ""
    """ID of the session that emitted this event."""

    event_kind: Literal["tool_call_start"] = "tool_call_start"
    """Event type identifier."""


@dataclass(kw_only=True)
class ToolCallProgressEvent:
    """Unified tool call progress event with rich content support.

    This event carries a title and rich content items (terminals, diffs, locations, text)
    that map directly to ACP tool call notifications.

    Use the classmethod constructors for common patterns:
    - process_started() - Process start with terminal
    - process_output() - Process output update
    - process_exit() - Process completion
    - process_killed() - Process termination
    - process_released() - Process cleanup
    - file_operation() - File read/write/delete
    - file_edit() - File edit with diff
    """

    tool_call_id: str
    """The ID of the tool call."""
    status: Literal["pending", "in_progress", "completed", "failed"] = "in_progress"
    """Current execution status."""
    title: str | None = None
    """Human-readable title describing the operation."""

    # Rich content items
    items: Sequence[ToolCallContentItem] = field(default_factory=list)
    """Rich content items (terminals, diffs, locations, text)."""
    replace_content: bool = False
    """If True, items replace existing content instead of appending."""

    # Legacy fields for backwards compatibility
    progress: int | None = None
    """The current progress of the tool call."""
    total: int | None = None
    """The total progress of the tool call."""
    message: str | None = None
    """Progress message."""
    tool_name: str | None = None
    """The name of the tool being called."""
    tool_input: dict[str, Any] | None = None
    """The input provided to the tool."""
    session_id: str = ""
    """ID of the session that emitted this event."""

    event_kind: Literal["tool_call_progress"] = "tool_call_progress"
    """Event type identifier."""

    @classmethod
    def process_started(
        cls,
        *,
        tool_call_id: str,
        process_id: str,
        command: str,
        success: bool = True,
        error: str | None = None,
        tool_name: str | None = None,
    ) -> ToolCallProgressEvent:
        """Create event for process start.

        Args:
            tool_call_id: Tool call identifier
            process_id: Process/terminal identifier
            command: Command being executed
            success: Whether process started successfully
            error: Error message if failed
            tool_name: Optional tool name
        """
        status: Literal["in_progress", "failed"] = "in_progress" if success else "failed"
        title = f"Running: {command}" if success else f"Failed to start: {command}"
        if error:
            title = f"{title} - {error}"

        items: list[ToolCallContentItem] = [TerminalContentItem(terminal_id=process_id)]
        if error:
            items.append(TextContentItem(text=f"Error: {error}"))

        return cls(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            status=status,
            title=title,
            items=items,
        )

    @classmethod
    def process_output(
        cls,
        *,
        tool_call_id: str,
        process_id: str,
        output: str,
        tool_name: str | None = None,
    ) -> ToolCallProgressEvent:
        """Create event for process output.

        Args:
            tool_call_id: Tool call identifier
            process_id: Process/terminal identifier
            output: Process output
            tool_name: Optional tool name
        """
        items = [TerminalContentItem(terminal_id=process_id)]
        title = f"Output: {output[:50]}..." if len(output) > 50 else output  # noqa: PLR2004

        return cls(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            status="in_progress",
            title=title,
            items=items,
        )

    @classmethod
    def process_exit(
        cls,
        *,
        tool_call_id: str,
        process_id: str,
        exit_code: int,
        final_output: str | None = None,
        tool_name: str | None = None,
    ) -> ToolCallProgressEvent:
        """Create event for process exit.

        Args:
            tool_call_id: Tool call identifier
            process_id: Process/terminal identifier
            exit_code: Process exit code
            final_output: Final process output
            tool_name: Optional tool name
        """
        success = exit_code == 0
        status_icon = "✓" if success else "✗"
        title = f"Process exited [{status_icon} exit {exit_code}]"

        items: list[ToolCallContentItem] = [TerminalContentItem(terminal_id=process_id)]
        if final_output:
            items.append(TextContentItem(text=final_output))

        return cls(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            status="in_progress",
            title=title,
            items=items,
        )

    @classmethod
    def process_killed(
        cls,
        *,
        tool_call_id: str,
        process_id: str,
        success: bool = True,
        error: str | None = None,
        tool_name: str | None = None,
    ) -> ToolCallProgressEvent:
        """Create event for process kill.

        Args:
            tool_call_id: Tool call identifier
            process_id: Process/terminal identifier
            success: Whether kill succeeded
            error: Error message if failed
            tool_name: Optional tool name
        """
        title = (
            f"Killed process {process_id}" if success else f"Failed to kill process {process_id}"
        )
        if error:
            title = f"{title} - {error}"

        status: Literal["in_progress", "failed"] = "in_progress" if success else "failed"
        items = [TerminalContentItem(terminal_id=process_id)]

        return cls(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            status=status,
            title=title,
            items=items,
        )

    @classmethod
    def process_released(
        cls,
        *,
        tool_call_id: str,
        process_id: str,
        success: bool = True,
        error: str | None = None,
        tool_name: str | None = None,
    ) -> ToolCallProgressEvent:
        """Create event for process resource release.

        Args:
            tool_call_id: Tool call identifier
            process_id: Process/terminal identifier
            success: Whether release succeeded
            error: Error message if failed
            tool_name: Optional tool name
        """
        title = (
            f"Released process {process_id}"
            if success
            else f"Failed to release process {process_id}"
        )
        if error:
            title = f"{title} - {error}"

        status: Literal["in_progress", "failed"] = "in_progress" if success else "failed"
        items = [TerminalContentItem(terminal_id=process_id)]

        return cls(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            status=status,
            title=title,
            items=items,
        )

    @classmethod
    def file_operation(
        cls,
        *,
        tool_call_id: str,
        operation: Literal["read", "write", "delete", "list", "edit"],
        path: str,
        success: bool,
        error: str | None = None,
        tool_name: str | None = None,
        line: int = 0,
    ) -> ToolCallProgressEvent:
        """Create event for file operation.

        Args:
            tool_call_id: Tool call identifier
            operation: File operation type
            path: File path
            success: Whether operation succeeded
            error: Error message if failed
            tool_name: Optional tool name
            line: Line number for navigation (0 = beginning)
        """
        status: Literal["completed", "failed"] = "completed" if success else "failed"
        title = f"{operation.capitalize()}: {path}"
        if error:
            title = f"{title} - {error}"

        items: list[ToolCallContentItem] = [LocationContentItem(path=path, line=line)]
        if error:
            items.append(TextContentItem(text=f"Error: {error}"))

        return cls(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            status=status,
            title=title,
            items=items,
        )

    @classmethod
    def file_edit(
        cls,
        *,
        tool_call_id: str,
        path: str,
        old_text: str,
        new_text: str,
        status: ToolCallStatus = "in_progress",
        tool_name: str | None = None,
    ) -> ToolCallProgressEvent:
        """Create event for file edit with diff.

        Args:
            tool_call_id: Tool call identifier
            path: File path being edited
            old_text: Original file content
            new_text: New file content
            status: Edit status
            tool_name: Optional tool name
        """
        items: list[ToolCallContentItem] = [
            DiffContentItem(path=path, old_text=old_text, new_text=new_text),
        ]

        return cls(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            status=status,
            title=f"Editing: {path}",
            items=items,
            replace_content=True,  # Streaming diffs should replace, not accumulate
        )


@dataclass(kw_only=True)
class CommandOutputEvent:
    """Event for slash command output."""

    command: str
    """The command name that was executed."""
    output: str
    """The output text from the command."""
    event_kind: Literal["command_output"] = "command_output"
    """Event type identifier."""


@dataclass(kw_only=True)
class CommandCompleteEvent:
    """Event indicating slash command execution is complete."""

    command: str
    """The command name that was completed."""
    success: bool
    """Whether the command executed successfully."""
    event_kind: Literal["command_complete"] = "command_complete"
    """Event type identifier."""


@dataclass(kw_only=True)
class ToolCallCompleteEvent:
    """Event indicating tool call is complete with both input and output."""

    tool_name: str
    """The name of the tool that was called."""
    tool_call_id: str
    """The ID of the tool call."""
    tool_input: dict[str, Any]
    """The input provided to the tool."""
    tool_result: Any
    """The result returned by the tool."""
    agent_name: str
    """The name of the agent that made the tool call."""
    message_id: str
    """The message ID associated with this tool call."""
    metadata: dict[str, Any] | None = None
    """Optional metadata for UI/client use (diffs, diagnostics, etc.)."""
    session_id: str = ""
    """ID of the session that emitted this event."""
    event_kind: Literal["tool_call_complete"] = "tool_call_complete"
    """Event type identifier."""


@dataclass(kw_only=True)
class ToolResultMetadataEvent:
    """Sidechannel event carrying tool result metadata stripped by Claude SDK.

    The Claude SDK strips the `_meta` field from MCP CallToolResult when converting
    to ToolResultBlock, losing UI-only metadata (diffs, diagnostics, etc.).

    This event provides a sidechannel to preserve that metadata:
    - Tool returns ToolResult with metadata
    - ToolManagerBridge emits this event with metadata before converting
    - The agent correlates by tool_call_id and enriches ToolCallCompleteEvent
    - Downstream consumers (OpenCode, ACP) receive complete events with metadata

    This avoids polluting LLM context with UI-only data while preserving it for clients.
    """

    tool_call_id: str
    """The ID of the tool call this metadata belongs to."""
    metadata: dict[str, Any]
    """Metadata for UI/client use (diffs, diagnostics, etc.)."""
    event_kind: Literal["tool_result_metadata"] = "tool_result_metadata"
    """Event type identifier."""


@dataclass(kw_only=True)
class CustomEvent[T]:
    """Generic custom event that can be emitted during tool execution."""

    event_data: T
    """The custom event data of any type."""
    event_type: str = "custom"
    """Type identifier for the custom event."""
    source: str | None = None
    """Optional source identifier (tool name, etc.)."""
    event_kind: Literal["custom"] = "custom"
    """Event type identifier."""


@dataclass(kw_only=True)
class PlanUpdateEvent:
    """Event indicating plan state has changed."""

    entries: list[PlanEntry]
    """Current plan entries."""
    tool_call_id: str | None = None
    """Tool call ID for ACP notifications."""
    event_kind: Literal["plan_update"] = "plan_update"
    """Event type identifier."""


@dataclass(kw_only=True)
class SubAgentEvent:
    """Event wrapping activity from a subagent or team member.

    Used to propagate events from delegated agents/teams into the parent stream,
    allowing the consumer (UI/server) to decide how to render nested activity.
    """

    source_name: str
    """Name of the agent or team that produced this event."""
    source_type: Literal["agent", "team_parallel", "team_sequential"]
    """Type of source: agent, parallel team, or sequential team."""
    event: RichAgentStreamEvent[Any]
    """The actual event from the subagent/team."""
    depth: int = 1
    """Nesting depth (1 = direct child, 2 = grandchild, etc.)."""
    child_session_id: str | None = None
    """ID of the child session for this subagent run."""
    parent_session_id: str | None = None
    """ID of the parent session that spawned this subagent."""
    tool_call_id: str | None = None
    """ID of the tool call that spawned this subagent."""
    model_id: str | None = None
    """Model identifier for the subagent (e.g., 'openai:gpt-4o'). Propagated to UI for display."""
    mode: str | None = None
    """Mode identifier for the subagent (e.g., 'code', 'ask'). Maps to OpenCode mode display."""
    path: list[str] = field(default_factory=list)
    """List of session_ids that this event has traversed, starting from source."""
    event_kind: Literal["subagent"] = "subagent"
    """Event type identifier."""


@dataclass(kw_only=True)
class SpawnSessionStart:
    """Event indicating a subsession (spawn/subagent) is being created.

    This event explicitly signals when a subsession is created, replacing the need
    for protocol adapters to hardcode detection of specific tool calls.
    """

    child_session_id: str
    """ID of the child session being created."""
    parent_session_id: str
    """ID of the parent session that is spawning the child."""
    tool_call_id: str | None = None
    """ID of the tool call that spawned this subsession, if applicable."""
    spawn_mechanism: Literal["task", "spawn"]
    """How the subagent was created: 'task' for task-based, 'spawn' for direct spawn."""
    source_name: str
    """Name of the agent or team being spawned."""
    display_name: str | None = None
    """Display name for the subagent, if different from source_name.

    Falls back to source_name when None.
    """
    source_type: Literal["agent", "team_parallel", "team_sequential"]
    """Type of source being spawned: agent, parallel team, or sequential team."""
    depth: int = 1
    """Nesting depth (1 = direct child of the root session, 2 = grandchild, etc.)."""
    description: str
    """Human-readable description of the spawn operation."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata associated with the spawn operation."""
    model_id: str | None = None
    """Model identifier for the subagent (e.g., 'openai:gpt-4o'). Propagated to UI for display."""
    mode: str | None = None
    """Mode identifier for the subagent (e.g., 'code', 'ask'). Maps to OpenCode mode display."""
    event_kind: Literal["spawn_session_start"] = "spawn_session_start"
    """Event type identifier."""


@dataclass(kw_only=True)
class CompactionEvent:
    """Event indicating context compaction is starting or completed.

    This is a semantic event that consumers (ACP, OpenCode) handle differently:
    - ACP: Converts to a text message for display
    - OpenCode: Emits session.compacted SSE event
    """

    session_id: str
    """The session ID being compacted."""
    trigger: Literal["auto", "manual"] = "auto"
    """What triggered the compaction (auto = context overflow, manual = slash command)."""
    phase: Literal["starting", "completed"] = "starting"
    """Current phase of compaction."""
    event_kind: Literal["compaction"] = "compaction"
    """Event type identifier."""


@dataclass(kw_only=True)
class ToolCallDeferredEvent:
    """Event indicating a tool call has been deferred for durable execution.

    This event is emitted when a tool execution cannot be completed immediately
    and must be persisted for later resumption. The deferred_handle identifies
    the external resource needed to resolve the call.
    """

    tool_call_id: str
    """The ID of the deferred tool call."""
    tool_name: str
    """The name of the tool that was deferred."""
    deferred_strategy: Literal["block", "continue", "stream"]
    """How the agent should handle the deferral: block, continue, or stream."""
    deferred_handle: str = ""
    """Opaque handle for resolving the deferred call externally."""
    status: Literal["pending", "resolved", "expired"]
    """Current status of the deferred call."""
    session_id: str = ""
    """ID of the session that emitted this event."""
    event_kind: Literal["tool_call_deferred"] = "tool_call_deferred"
    """Event type identifier."""


@dataclass(kw_only=True)
class ElicitationDeferredEvent:
    """Event indicating an elicitation request has been deferred for user response.

    Emitted when an MCP server's elicitation request cannot be resolved immediately
    and must be persisted for later resumption. The deferred_handle identifies
    the pending call that will be resolved when the user responds.
    """

    deferred_handle: str
    """Opaque handle for resolving the deferred elicitation call."""

    message: str
    """Human-readable message describing what is being elicited."""

    requested_schema: dict[str, Any]
    """JSON schema describing the expected response structure."""

    mode: str
    """Elicitation mode hint (e.g., 'form', 'inline') for client rendering."""

    session_id: str = ""
    """ID of the session that emitted this event."""
    event_kind: Literal["elicitation_deferred"] = "elicitation_deferred"
    """Event type identifier."""

    timeout_seconds: float | None = None
    """Elicitation timeout in seconds, for frontend countdown display.

    ``None`` means no timeout (infinite wait). Set from
    ``AgentRunContext.elicitation_timeout`` when the event is published.
    """


@dataclass(kw_only=True)
class SessionResumeEvent:
    """Event indicating a session has been resumed from a checkpoint.

    Emitted when a previously suspended session resumes execution,
    carrying metadata about the deferred calls that were resolved.
    """

    session_id: str
    """ID of the resumed session."""
    resolved_call_count: int
    """Number of deferred calls that were resolved before resumption."""
    source: str = ""
    """Identifier for the entity that triggered the resume."""
    event_kind: Literal["session_resume"] = "session_resume"
    """Event type identifier."""


@dataclass(kw_only=True)
class StateUpdate:
    """Event indicating a RunLoop state transition.

    Published on every state transition (idle → running → idle → done).
    For ``IDLE`` transitions after crash recovery, ``stop_reason`` SHALL
    be ``"crash_recovery"``.
    """

    session_id: str
    """ID of the session whose state changed."""
    state: RunState
    """The new RunState."""
    stop_reason: str | None = None
    """Optional reason for the transition (e.g. ``"crash_recovery"``)."""
    event_kind: Literal["state_update"] = "state_update"
    """Event type identifier."""


@dataclass(kw_only=True)
class ToolCallUpdateEvent:
    """Entity-state event for tool call updates.

    Represents the latest state of a tool call. Uses upsert semantics
    in the Journal (key: ``f"tool_call:{tool_call_id}"``) so only the
    most recent state per tool call is retained.
    """

    tool_call_id: str
    """The ID of the tool call."""
    tool_name: str
    """The name of the tool being called."""
    status: ToolCallStatus = "in_progress"
    """Current execution status."""
    title: str | None = None
    """Human-readable title describing the operation."""
    tool_input: dict[str, Any] = field(default_factory=dict)
    """The input parameters sent to the tool."""
    tool_result: Any | None = None
    """The result returned by the tool (``None`` if not completed)."""
    session_id: str = ""
    """ID of the session that emitted this event."""
    event_kind: Literal["tool_call_update"] = "tool_call_update"
    """Event type identifier."""


@dataclass(kw_only=True)
class MessageReplacementEvent:
    """Entity-state event for message replacement.

    Indicates that a previous message should be replaced with new content.
    Uses upsert semantics in the Journal (key: ``f"msg:{message_id}"``).
    """

    message_id: str
    """The ID of the message to replace."""
    content: str
    """The replacement content."""
    session_id: str = ""
    """ID of the session that emitted this event."""
    event_kind: Literal["message_replacement"] = "message_replacement"
    """Event type identifier."""


@dataclass(frozen=True, kw_only=True)
class UserMessageInsertedEvent[T]:
    """User message inserted into the conversation mid-run.

    Emitted when a steer/followup message is injected into an active session
    so that protocol frontends can display it as a user message. Supports
    multi-modal content via ``list[Any]``.

    The ``meta`` field carries protocol-specific metadata (e.g. serialized
    parts for OpenCode, content blocks for ACP) so that the EventBus is
    the sole publication point and protocol handlers do not need to
    broadcast user messages themselves.
    """

    session_id: str = ""
    """ID of the session the message was inserted into."""

    message_id: str = ""
    """Unique ID for the inserted user message."""

    content: str | list[Any] = ""
    """Message content — plain text or multi-modal part list."""

    delivery: Literal["initial", "steer", "followup"] = "initial"
    """How the message was delivered to the run."""

    source: Literal["accepted", "processed"] = "accepted"
    """Originator of the inserted message.

    ``"accepted"`` indicates the event was produced at message accept
    time (routing) — when the message enters the session via
    ``_route_message()``, ``send_message()``, or fire-and-forget
    emission from ``_schedule_user_message_emission()``. This is a
    fallback display event that does NOT trigger a turn split.

    ``"processed"`` indicates the event was produced at model
    processing time — from ``EnqueuedMessagesEvent`` mapping when the
    steer message enters run history and the model is about to process
    it, making it the split trigger for protocol frontends.
    """

    timestamp: float = field(default_factory=time.time)
    """Wall-clock time the event was created (epoch seconds)."""

    meta: T | None = None
    """Protocol-specific metadata for rich user message display.

    When set, protocol event consumers use this to reconstruct the full
    user message (e.g. OpenCode parts, ACP content blocks) instead of
    falling back to text-only ``content``.
    """


@dataclass(kw_only=True)
class StepUsageEvent:
    """Per-step token usage, emitted after each LLM call within a turn.

    Emitted by ``NativeTurn.execute()`` after each ``agent_run.next(node)``
    when the step involved an LLM call (``step_usage.requests > 0``).
    Carries the per-step delta and the running cumulative total.
    """

    step_index: int
    """Zero-based index of this LLM step within the current turn (resets per turn)."""

    step_usage: RunUsage
    """Per-step delta usage (difference from previous step). NOT ``RequestUsage``
    because ``RequestUsage.requests`` is a read-only property returning 1."""

    cumulative_usage: RunUsage
    """Running cumulative usage for the entire turn (snapshot copy, not live reference)."""

    cost_info: TokenCost | None = None
    """Per-step cost. Always ``None`` — per-step cost calculation is a non-goal."""

    event_kind: Literal["step_usage"] = "step_usage"
    """Event type discriminator (all events use ``event_kind``, NOT ``event_type``)."""


type RichAgentStreamEvent[OutputDataT] = (
    AgentStreamEvent
    | StreamCompleteEvent[OutputDataT]
    | RunStartedEvent
    | RunErrorEvent
    | RunFailedEvent
    | ToolCallStartEvent
    | ToolCallProgressEvent
    | ToolCallCompleteEvent
    | ToolCallDeferredEvent
    | ElicitationDeferredEvent
    | SessionResumeEvent
    | PlanUpdateEvent
    | CompactionEvent
    | SubAgentEvent
    | SpawnSessionStart
    | ToolResultMetadataEvent
    | CustomEvent[Any]
    | StateUpdate
    | ToolCallUpdateEvent
    | MessageReplacementEvent
    | UserMessageInsertedEvent[Any]
    | StepUsageEvent
)


type StreamWithCommandsEvent[OutputDataT] = (
    RichAgentStreamEvent[OutputDataT] | CommandOutputEvent | CommandCompleteEvent
)
