"""The main Agent. Can do all sort of crazy things."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai import (
    BaseToolCallPart,
    BaseToolReturnPart,
    BinaryContent,
    BinaryImage,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelResponse,
    PartStartEvent,
    TextPart,
)
from pydantic_ai.messages import (
    AudioUrl,
    DocumentUrl,
    ImageUrl,
    TextContent,
    ThinkingPart,
    UploadedFile,
    VideoUrl,
)

from wolfharness.agents.events import ToolCallCompleteEvent, UserMessageInsertedEvent
from wolfharness.agents.modes import ModeCategory, ModeInfo
from wolfharness.utils.pydantic_ai_helpers import safe_args_as_dict


if TYPE_CHECKING:
    from tokonomics.model_discovery import ModelInfo

    from wolfharness.agents.context import AgentRunContext
    from wolfharness.agents.events import RichAgentStreamEvent
    from wolfharness_config.nodes import ToolConfirmationMode


async def process_tool_event(
    agent_name: str,
    event: RichAgentStreamEvent[Any],
    pending_tool_calls: dict[str, BaseToolCallPart],
    message_id: str,
    run_ctx: AgentRunContext | None = None,
) -> ToolCallCompleteEvent | None:
    """Process tool-related events and return combined event when complete.

    Always returns the combined event; the caller decides how to route it
    (enqueue locally, publish to EventBus, etc.).

    Args:
        agent_name: Name of the agent
        event: The streaming event to process
        pending_tool_calls: Dict tracking in-progress tool calls by ID
        message_id: Message ID for the combined event
        run_ctx: Optional per-run context (unused, kept for API compatibility).

    Returns:
        ToolCallCompleteEvent if a tool call completed, None otherwise.
    """
    # Note: BuiltinToolCallEvent/BuiltinToolResultEvent are deprecated.
    # Both function and builtin tools use PartStartEvent with BaseToolCallPart/BaseToolReturnPart.
    match event:
        case (
            PartStartEvent(part=BaseToolCallPart() as tool_part)
            | FunctionToolCallEvent(part=tool_part)
        ):
            pending_tool_calls[tool_part.tool_call_id] = tool_part
        case (
            PartStartEvent(part=BaseToolReturnPart(tool_call_id=call_id, content=content))
            | FunctionToolResultEvent(
                part=BaseToolReturnPart(tool_call_id=call_id, content=content)
            )
        ):
            if call_info := pending_tool_calls.pop(call_id, None):
                return ToolCallCompleteEvent(
                    tool_name=call_info.tool_name,
                    tool_call_id=call_id,
                    tool_input=safe_args_as_dict(call_info),
                    tool_result=content,
                    agent_name=agent_name,
                    message_id=message_id,
                )
        case UserMessageInsertedEvent():
            pass  # Not a tool event
    return None


def extract_text_from_messages(messages: list[Any], include_interruption_note: bool = False) -> str:
    """Extract text content from pydantic-ai messages.

    Args:
        messages: List of ModelRequest/ModelResponse messages
        include_interruption_note: Whether to append interruption notice

    Returns:
        Concatenated text content from all ModelResponse TextParts
    """
    content = "".join(
        part.content
        for msg in messages
        if isinstance(msg, ModelResponse)
        for part in msg.parts
        if isinstance(part, TextPart | ThinkingPart)
    )
    if include_interruption_note:
        if content:
            content += "\n\n"
        content += "[Request interrupted by user]"
    return content


def get_permission_category(current_mode: ToolConfirmationMode) -> ModeCategory:
    """Get permission mode category using native ToolConfirmationMode values."""
    return ModeCategory(
        id="mode",
        name="Tool Confirmation",
        available_modes=[
            ModeInfo(
                id="always",
                name="Always",
                description="Always require confirmation for all tools",
                category_id="mode",
            ),
            ModeInfo(
                id="never",
                name="Never",
                description="Never require confirmation (auto-approve all)",
                category_id="mode",
            ),
            ModeInfo(
                id="per_tool",
                name="Per Tool",
                description="Require confirmation only for tools marked as needing it",
                category_id="mode",
            ),
        ],
        current_mode_id=current_mode,
        category="mode",
    )


def get_model_category(current_model: str, models: list[ModelInfo]) -> ModeCategory:
    return ModeCategory(
        id="model",
        name="Model",
        available_modes=[
            ModeInfo(
                id=m.id,
                name=m.name or m.id,
                description=m.description or "",
                category_id="model",
            )
            for m in models
        ],
        current_mode_id=current_model,
        category="model",
    )


def _summarize_content_block(block: Any) -> str:
    """Produce a short text summary of a content block for logging/display.

    Handles all pydantic-ai UserContent variant types, producing meaningful
    placeholders instead of raw ``repr()`` output for binary/URL types.
    Delegates multimodal content description to
    :func:`describe_multimodal_content`.

    Args:
        block: A content block from ``UserPromptPart.content`` — may be a
            ``str``, ``TextContent``, ``BinaryImage``, ``BinaryContent``,
            ``ImageUrl``, ``AudioUrl``, ``VideoUrl``, ``DocumentUrl``,
            ``UploadedFile``, or any other object.

    Returns:
        A short text string suitable for display in logs, ChatMessage.content,
        and other text-only contexts.
    """
    if isinstance(block, str):
        return block
    if isinstance(block, TextContent):
        return block.content
    if isinstance(
        block,
        BinaryImage | BinaryContent | ImageUrl | AudioUrl | VideoUrl | DocumentUrl | UploadedFile,
    ):
        from wolfharness.capabilities.modality_utils import describe_multimodal_content

        return describe_multimodal_content(block)
    return f"[{type(block).__name__}]"
