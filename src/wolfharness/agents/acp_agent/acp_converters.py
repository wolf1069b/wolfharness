"""ACP to native event converters.

This module provides conversion from ACP session updates to native wolfharness
streaming events, enabling ACPAgent to yield the same event types as native agents.

This is the reverse of the conversion done in acp_server/session.py handle_event().

Also provides ACPMessageAccumulator for converting ACP notification streams
into ChatMessage objects (the reverse of replaying).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, assert_never, overload
from uuid import uuid4

from pydantic_ai import (
    AudioUrl,
    BinaryContent,
    BinaryImage,
    CachePoint,
    DocumentUrl,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    RunUsage,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.messages import UploadedFile

from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AudioContentBlock,
    BlobResourceContents,
    ContentToolCallContent,
    EmbeddedResourceContentBlock,
    FileEditToolCallContent,
    ImageContentBlock,
    ResourceContentBlock,
    SessionConfigSelectOption,
    TerminalToolCallContent,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    UsageUpdate,
    UserMessageChunk,
)
from wolfharness.agents.events import (
    DiffContentItem,
    LocationContentItem,
    PartDeltaEvent,
    PlanUpdateEvent,
    StepUsageEvent,
    TerminalContentItem,
    ToolCallCompleteEvent,
    ToolCallProgressEvent,
    ToolCallStartEvent,
)


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pydantic_ai import FinishReason, ModelMessage, ModelResponsePart, UserContent

    from acp.schema import (
        ContentBlock,
        HttpMcpServer,
        McpServer,
        SessionConfigOption,
        SessionModelState,
        SessionModeState,
        SessionUpdate,
        SseMcpServer,
        StdioMcpServer,
        StopReason,
        ToolCallContent,
        ToolCallLocation,
    )
    from acp.schema.mcp import AcpMcpServer
    from wolfharness.agents.events import RichAgentStreamEvent, ToolCallContentItem
    from wolfharness.agents.modes import ModeCategory, ModeInfo
    from wolfharness.messaging.messages import ChatMessage
    from wolfharness_config.mcp_server import (
        AcpMCPServerConfig,
        MCPServerConfig,
        SSEMCPServerConfig,
        StdioMCPServerConfig,
        StreamableHTTPMCPServerConfig,
    )

STOP_REASON_MAP: dict[StopReason, FinishReason] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "max_turn_requests": "length",
    "refusal": "content_filter",
    "cancelled": "error",
}


def get_modes(
    config_options: list[SessionConfigOption],
    available_modes: SessionModeState | None,
    available_models: SessionModelState | None,
) -> list[ModeCategory]:
    from wolfharness.agents.modes import ModeCategory, ModeInfo

    categories: list[ModeCategory] = []

    if config_options:
        for config_opt in config_options:
            # Extract options from the config (ungrouped or grouped)
            mode_infos: list[ModeInfo] = []
            for i in config_opt.options:
                opts = [i] if isinstance(i, SessionConfigSelectOption) else i.options
                mode_infos.extend(
                    ModeInfo(
                        id=sub_opt.value,
                        name=sub_opt.name,
                        description=sub_opt.description or "",
                        category_id=config_opt.id,
                    )
                    for sub_opt in opts
                )

            categories.append(
                ModeCategory(
                    id=config_opt.id,
                    name=config_opt.name,
                    available_modes=mode_infos,
                    current_mode_id=config_opt.current_value,
                    category=config_opt.category or "other",
                )
            )
        return categories

    # Legacy: Convert ACP SessionModeState to ModeCategory
    if available_modes:
        modes = [
            ModeInfo(
                id=m.id,
                name=m.name,
                description=m.description or "",
                category_id="mode",
            )
            for m in available_modes.available_modes
        ]
        categories.append(
            ModeCategory(
                id="mode",
                name="Mode",
                available_modes=modes,
                current_mode_id=available_modes.current_mode_id,
                category="mode",
            )
        )

    # Legacy: Convert ACP SessionModelState to ModeCategory
    if available_models:
        models = [
            ModeInfo(
                id=m.model_id,
                name=m.name,
                description=m.description or "",
                category_id="model",
            )
            for m in available_models.available_models
        ]
        categories.append(
            ModeCategory(
                id="model",
                name="Model",
                available_modes=models,
                current_mode_id=available_models.current_model_id,
                category="model",
            )
        )

    return categories


def to_finish_reason(stop_reason: StopReason) -> FinishReason:
    return STOP_REASON_MAP.get(stop_reason, "stop")


def convert_acp_locations(
    locations: Sequence[ToolCallLocation] | None,
) -> list[LocationContentItem]:
    """Convert ACP ToolCallLocation list to native LocationContentItem list."""
    return [LocationContentItem(path=loc.path, line=loc.line or 0) for loc in locations or []]


def convert_acp_content(content: Sequence[ToolCallContent] | None) -> list[ToolCallContentItem]:
    """Convert ACP ToolCallContent list to native ToolCallContentItem list."""
    if not content:
        return []

    result: list[ToolCallContentItem] = []
    for item in content:
        match item:
            case TerminalToolCallContent(terminal_id=terminal_id):
                result.append(TerminalContentItem(terminal_id=terminal_id))
            case FileEditToolCallContent(path=path, old_text=old_text, new_text=new_text):
                result.append(DiffContentItem(path=path, old_text=old_text, new_text=new_text))
            case ContentToolCallContent(content=TextContentBlock(text=text)):
                from wolfharness.agents.events import TextContentItem

                result.append(TextContentItem(text=text))
    return result


def convert_to_acp_content(prompts: Sequence[UserContent]) -> list[ContentBlock]:
    """Convert pydantic-ai UserContent to ACP ContentBlock format.

    Handles text, images, audio, video, and document content types.

    Args:
        prompts: pydantic-ai UserContent items

    Returns:
        List of ACP ContentBlock items
    """
    content_blocks: list[ContentBlock] = []

    for item in prompts:
        match item:
            case str(text):
                content_blocks.append(TextContentBlock(text=text))

            case BinaryImage(data=data, media_type=media_type):
                encoded = base64.b64encode(data).decode("utf-8")
                content_blocks.append(ImageContentBlock(data=encoded, mime_type=media_type))

            case BinaryContent(data=data, media_type=typ):
                encoded = base64.b64encode(data).decode()
                # Handle different media types
                if item.is_image:
                    content_blocks.append(ImageContentBlock(data=encoded, mime_type=typ))
                elif item.is_audio:
                    content_blocks.append(AudioContentBlock(data=encoded, mime_type=typ))
                elif item.is_document:
                    uri = f"data:application/pdf;base64,{encoded[:50]}..."
                    blob_resource = BlobResourceContents(blob=encoded, mime_type=typ, uri=uri)
                    content_blocks.append(EmbeddedResourceContentBlock(resource=blob_resource))
                else:
                    # Generic binary as embedded resource
                    uri = f"data:{typ or 'application/octet-stream'};base64,..."
                    blob_resource = BlobResourceContents(blob=encoded, mime_type=typ, uri=uri)
                    content_blocks.append(EmbeddedResourceContentBlock(resource=blob_resource))

            case (
                AudioUrl(url=url, media_type=typ)
                | DocumentUrl(url=url, media_type=typ)
                | VideoUrl(url=url, media_type=typ)
                | ImageUrl(url=url, media_type=typ)
                # FileUrl(url=url, media_type=typ)
            ):
                name = type(item).__name__.removesuffix("Url")
                content_blocks.append(ResourceContentBlock(uri=url, name=name, mime_type=typ))

            case TextContent(content=text):
                content_blocks.append(TextContentBlock(text=text))

            case CachePoint() | UploadedFile():
                pass

            case _ as unreachable:
                assert_never(unreachable)

    return content_blocks


def acp_to_native_event(  # noqa: PLR0911
    update: SessionUpdate,
    *,
    step_index: int = 0,
    cumulative_usage: RunUsage | None = None,
) -> RichAgentStreamEvent[Any] | None:
    """Convert ACP session update to native streaming event.

    Args:
        update: ACP SessionUpdate from session/update notification
        step_index: Zero-based LLM step index within the current turn (resets per turn).
        cumulative_usage: Running cumulative usage for the turn. If ``None``, an empty
            ``RunUsage`` is used. The caller (``ACPTurn.execute()``) is responsible for
            maintaining and passing the correct accumulator.

    Returns:
        Corresponding native event, or None if no mapping exists
    """
    match update:
        # Text message chunks -> PartDeltaEvent with TextPartDelta
        case AgentMessageChunk(content=TextContentBlock(text=text), message_id=msg_id):
            return PartDeltaEvent.text(index=0, content=text, message_id=msg_id or "")

        # Thought chunks -> PartDeltaEvent with ThinkingPartDelta
        case AgentThoughtChunk(content=TextContentBlock(text=text), message_id=msg_id):
            return PartDeltaEvent.thinking(index=0, content=text, message_id=msg_id or "")

        # User message echo - usually ignored
        case UserMessageChunk():
            return None

        # Tool call start -> ToolCallStartEvent
        case ToolCallStart(
            tool_call_id=tool_call_id,
            title=title,
            kind=kind,
            content=content,
            locations=locations,
            raw_input=raw_input,
        ):
            return ToolCallStartEvent(
                tool_call_id=tool_call_id,
                tool_name=title,  # ACP uses title, not separate tool_name
                title=title,
                kind="other" if kind == "subagent" else (kind or "other"),
                content=convert_acp_content(list(content) if content else None),
                locations=convert_acp_locations(list(locations) if locations else None),
                raw_input=raw_input or {},
            )

        # Tool call progress -> ToolCallProgressEvent or ToolCallCompleteEvent
        case ToolCallProgress(
            tool_call_id=tool_call_id,
            status=status,
            title=title,
            content=content,
            raw_output=raw_output,
        ):
            # If completed, return ToolCallCompleteEvent for metadata injection
            if status == "completed":
                return ToolCallCompleteEvent(
                    tool_call_id=tool_call_id,
                    tool_name=title or "unknown",
                    tool_input={},  # ACP doesn't provide input in progress updates
                    tool_result=str(raw_output) if raw_output else "",
                    agent_name="",  # Will be set by agent
                    message_id="",  # Will be set by ACPTurn from _message_id
                    metadata=None,  # Will be injected by agent from metadata accumulator
                )
            # Otherwise return progress event
            return ToolCallProgressEvent(
                tool_call_id=tool_call_id,
                status=status or "in_progress",
                title=title,
                items=convert_acp_content(list(content) if content else None),
                message=str(raw_output) if raw_output else None,
            )

        # Plan update -> PlanUpdateEvent
        case AgentPlanUpdate(entries=entries):
            from wolfharness.utils.todos import PlanEntry

            native_entries = [
                PlanEntry(content=e.content, priority=e.priority, status=e.status) for e in entries
            ]
            return PlanUpdateEvent(entries=native_entries)

        # Usage update -> StepUsageEvent
        # ACP's UsageUpdate.used is "tokens currently in context" — a best-effort
        # cumulative measure with no input/output split. We map it to output_tokens
        # and set input_tokens=0. The step_index and cumulative_usage are provided
        # by the caller (ACPTurn.execute()); defaults are used when called standalone.
        case UsageUpdate(used=used):
            return StepUsageEvent(
                step_index=step_index,
                step_usage=RunUsage(output_tokens=used),
                cumulative_usage=(cumulative_usage or RunUsage()),
            )

        case _:
            return None


@overload
def mcp_config_to_acp(config: StdioMCPServerConfig) -> StdioMcpServer: ...


@overload
def mcp_config_to_acp(config: SSEMCPServerConfig) -> SseMcpServer: ...


@overload
def mcp_config_to_acp(config: StreamableHTTPMCPServerConfig) -> HttpMcpServer: ...


@overload
def mcp_config_to_acp(config: AcpMCPServerConfig) -> AcpMcpServer: ...


@overload
def mcp_config_to_acp(config: MCPServerConfig) -> McpServer: ...


def mcp_config_to_acp(config: MCPServerConfig) -> McpServer:
    """Convert native MCPServerConfig to ACP McpServer format.

    If the config has tool filtering (enabled_tools or disabled_tools),
    the server is wrapped with mcp-filter proxy to apply the filtering.

    Args:
        config: wolfharness MCP server configuration

    Returns:
        ACP-compatible McpServer instance, or None if conversion not possible
    """
    from acp.schema.common import EnvVariable
    from acp.schema.mcp import AcpMcpServer, HttpMcpServer, SseMcpServer, StdioMcpServer
    from wolfharness_config.mcp_server import (
        AcpMCPServerConfig,
        SSEMCPServerConfig,
        StdioMCPServerConfig,
        StreamableHTTPMCPServerConfig,
    )

    # If filtering is configured, wrap with mcp-filter first
    if config.needs_tool_filtering():
        config = config.wrap_with_mcp_filter()

    match config:
        case StdioMCPServerConfig(command=command, args=args):
            envs = [EnvVariable(name=k, value=v) for k, v in config.get_env_vars().items()]
            args = list(args) if args else []
            return StdioMcpServer(name=config.name or command, command=command, args=args, env=envs)

        case SSEMCPServerConfig(url=url):
            return SseMcpServer(name=config.name or str(url), url=url, headers=[])

        case StreamableHTTPMCPServerConfig(url=url):
            return HttpMcpServer(name=config.name or str(url), url=url)

        case AcpMCPServerConfig(acp_id=acp_id):
            return AcpMcpServer(name=config.name or config.client_id, id=acp_id)

        case _ as unreachable:
            assert_never(unreachable)


# ============================================================================
# ACP Notifications → ChatMessages Converter
# ============================================================================


@dataclass
class _PendingToolCall:
    """Tracks a tool call from start to completion."""

    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    result: Any = None
    completed: bool = False


@dataclass
class ACPMessageAccumulator:
    """Accumulates ACP session notifications into ChatMessage objects.

    This is the reverse of ACPEventConverter - instead of converting native events
    to ACP notifications for "replaying", this converts ACP notifications back
    into ChatMessage objects.

    The accumulator processes a stream of notifications and groups them by role:
    - UserMessageChunk → accumulates into ChatMessage with role="user"
    - Everything else → accumulates into ChatMessage with role="assistant"

    Message boundaries occur when the role switches.

    Example:
        ```python
        accumulator = ACPMessageAccumulator()
        for notification in acp_notifications:
            accumulator.process(notification.update)
        messages = accumulator.finalize()
        ```
    """

    session_id: str | None = None
    """Optional session ID for generated messages."""

    agent_name: str | None = None
    """Optional agent name for generated messages."""

    model_name: str | None = None
    """Optional model name for generated messages."""

    # Internal state
    _current_role: Literal["user", "assistant"] | None = field(default=None, repr=False)
    """Current role being accumulated."""

    _text_buffer: list[str] = field(default_factory=list, repr=False)
    """Accumulated text content."""

    _thinking_buffer: list[str] = field(default_factory=list, repr=False)
    """Accumulated thinking content."""

    _pending_tool_calls: dict[str, _PendingToolCall] = field(default_factory=dict, repr=False)
    """Tool calls in progress, keyed by tool_call_id."""

    _completed_tool_calls: list[_PendingToolCall] = field(default_factory=list, repr=False)
    """Tool calls that have completed."""

    _messages: list[ChatMessage[str]] = field(default_factory=list, repr=False)
    """Accumulated messages."""

    _last_parent_id: str | None = field(default=None, repr=False)
    """Parent ID for linking messages."""

    _current_message_id: str = field(default="", repr=False)
    """Current message ID from ACP updates, preserved for finalize."""

    def reset(self) -> None:
        """Reset accumulator state for a new conversation."""
        self._current_role = None
        self._text_buffer.clear()
        self._thinking_buffer.clear()
        self._pending_tool_calls.clear()
        self._completed_tool_calls.clear()
        self._messages.clear()
        self._last_parent_id = None
        self._current_message_id = ""

    def _finalize_current_message(self) -> None:
        """Finalize the current message and add it to the messages list."""
        from wolfharness.messaging.messages import ChatMessage

        if self._current_role is None:
            return
        message_id = self._current_message_id if self._current_message_id else str(uuid4())
        if self._current_role == "user":
            # Build user message
            content = "".join(self._text_buffer)
            parts: list[UserPromptPart] = [UserPromptPart(content=content)]
            model_message = ModelRequest(parts=parts, run_id=message_id)

            msg: ChatMessage[str] = ChatMessage(
                content=content,
                role="user",
                message_id=message_id,
                session_id=self.session_id,
                parent_id=self._last_parent_id,
                messages=[model_message],
                name=None,
            )
        else:
            # Build assistant message with proper parts
            # Structure: ModelResponse(ToolCallPart) -> ModelRequest(ToolReturnPart)
            #         -> ModelResponse(TextPart)
            all_messages: list[ModelMessage] = []
            response_parts: list[ModelResponsePart] = []
            # Add thinking part if present
            if self._thinking_buffer:
                thinking_content = "".join(self._thinking_buffer)
                response_parts.append(ThinkingPart(content=thinking_content))

            # Process completed tool calls - each creates a response + request pair
            for tc in self._completed_tool_calls:
                # First, a ModelResponse with the ToolCallPart
                call_part = ToolCallPart(
                    tool_name=tc.tool_name,
                    args=tc.args,
                    tool_call_id=tc.tool_call_id,
                )
                tool_call_response = ModelResponse(parts=[call_part], model_name=self.model_name)
                all_messages.append(tool_call_response)
                # Then, a ModelRequest with the ToolReturnPart
                return_part = ToolReturnPart(
                    tool_name=tc.tool_name,
                    content=tc.result if tc.result is not None else "",
                    tool_call_id=tc.tool_call_id,
                )
                tool_return_request = ModelRequest(parts=[return_part])
                all_messages.append(tool_return_request)

            # Add pending (incomplete) tool calls to response parts
            response_parts.extend(
                ToolCallPart(
                    tool_name=tc.tool_name,
                    args=tc.args,
                    tool_call_id=tc.tool_call_id,
                )
                for tc in self._pending_tool_calls.values()
            )

            # Add text part if present
            if self._text_buffer:
                text_content = "".join(self._text_buffer)
                response_parts.append(TextPart(content=text_content))

            # Add final response with text/thinking/pending tool calls
            if response_parts:
                final_response = ModelResponse(
                    parts=response_parts,
                    model_name=self.model_name,
                    run_id=message_id,
                )
                all_messages.append(final_response)

            msg = ChatMessage(
                content="".join(self._text_buffer),
                role="assistant",
                message_id=message_id,
                session_id=self.session_id,
                parent_id=self._last_parent_id,
                messages=all_messages,
                name=self.agent_name,
                model_name=self.model_name,
            )

        self._messages.append(msg)
        self._last_parent_id = message_id
        # Clear buffers for next message
        self._text_buffer.clear()
        self._thinking_buffer.clear()
        self._pending_tool_calls.clear()
        self._completed_tool_calls.clear()
        self._current_role = None
        self._current_message_id = ""

    def _switch_role(self, new_role: Literal["user", "assistant"]) -> None:
        """Switch to a new role, finalizing the current message if needed."""
        if self._current_role is not None and self._current_role != new_role:
            self._finalize_current_message()
        self._current_role = new_role

    def _update_message_id(self, message_id: str | None) -> None:
        """Update the current message ID, finalizing if it changed.

        A change in message_id indicates a new message has started,
        so we finalize the current message before processing the new one.
        """
        if message_id and message_id != self._current_message_id:
            if self._current_role is not None:
                self._finalize_current_message()
            self._current_message_id = message_id

    def process(self, update: SessionUpdate) -> None:
        """Process a single ACP session update."""
        match update:
            # User message chunks → switch to user role
            case UserMessageChunk(content=content_block, message_id=msg_id):
                self._update_message_id(msg_id)
                self._switch_role("user")
                if isinstance(content_block, TextContentBlock):
                    self._text_buffer.append(content_block.text)

            # Agent message chunks → switch to assistant role
            case AgentMessageChunk(content=content_block, message_id=msg_id):
                self._update_message_id(msg_id)
                self._switch_role("assistant")
                if isinstance(content_block, TextContentBlock):
                    self._text_buffer.append(content_block.text)

            # Agent thought chunks → assistant role thinking
            case AgentThoughtChunk(content=content_block, message_id=msg_id):
                self._update_message_id(msg_id)
                self._switch_role("assistant")
                if isinstance(content_block, TextContentBlock):
                    self._thinking_buffer.append(content_block.text)

            # Tool call start → track pending tool call
            case ToolCallStart(tool_call_id=tool_call_id, title=title, raw_input=raw_input):
                self._switch_role("assistant")
                self._pending_tool_calls[tool_call_id] = _PendingToolCall(
                    tool_call_id=tool_call_id,
                    tool_name=title,
                    args=raw_input if isinstance(raw_input, dict) else {},
                )

            # Tool call progress → update or complete tool call
            case ToolCallProgress(tool_call_id=tool_call_id, status=status, raw_output=raw_output):
                self._switch_role("assistant")
                if tool_call_id in self._pending_tool_calls:
                    tc = self._pending_tool_calls[tool_call_id]
                    if raw_output is not None:
                        tc.result = raw_output
                    if status == "completed":
                        tc.completed = True
                        self._completed_tool_calls.append(tc)
                        del self._pending_tool_calls[tool_call_id]

            # Plan updates, mode updates, etc. - just ensure assistant role
            case AgentPlanUpdate() | _:
                if self._current_role is None:
                    self._current_role = "assistant"

    def finalize(self) -> list[ChatMessage[str]]:
        """Finalize accumulation and return all messages.

        This should be called after all notifications have been processed.
        It finalizes any pending message and returns the complete list.

        Returns:
            List of ChatMessage objects accumulated from the notifications
        """
        self._finalize_current_message()
        return list(self._messages)


def acp_notifications_to_messages(
    notifications: Iterable[SessionUpdate],
    *,
    session_id: str | None = None,
    agent_name: str | None = None,
    model_name: str | None = None,
) -> list[ChatMessage[str]]:
    """Convert a sequence of ACP notifications to ChatMessage objects.

    Convenience function that creates an accumulator, processes all notifications,
    and returns the finalized messages.

    Args:
        notifications: Iterable of SessionNotification or SessionUpdate objects
        session_id: Optional session ID for generated messages
        agent_name: Optional agent name for generated messages
        model_name: Optional model name for generated messages

    Returns:
        List of ChatMessage objects

    Example:
        ```python
        # From a list of notifications
        messages = acp_notifications_to_messages(notifications)

        # With metadata
        messages = acp_notifications_to_messages(
            notifications,
            session_id="conv-123",
            agent_name="goose",
            model_name="claude-3-opus",
        )
        ```
    """
    accumulator = ACPMessageAccumulator(
        session_id=session_id,
        agent_name=agent_name,
        model_name=model_name,
    )
    for item in notifications:
        accumulator.process(item)
    return accumulator.finalize()
