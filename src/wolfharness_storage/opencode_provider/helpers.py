"""Helper functions for OpenCode SQLite storage provider.

Stateless conversion and utility functions for working with OpenCode's
SQLite-based format. Converts between raw database rows and domain models.
"""

from __future__ import annotations

import base64
from decimal import Decimal
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter
from pydantic_ai import (
    BinaryContent,
    FileUrl,
    ModelRequest,
    ModelResponse,
    RequestUsage,
    RunUsage,
    TextContent,
    TextPart as PydanticTextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from wolfharness.log import get_logger
from wolfharness.messaging import ChatMessage, TokenCost
from wolfharness.utils import identifiers as identifier
from wolfharness.utils.pydantic_ai_helpers import to_user_content
from wolfharness.utils.time_utils import ms_to_datetime
from wolfharness_server.opencode_server.models.message import (
    AssistantMessage,
    MessageInfo,
    UserMessage,
)
from wolfharness_server.opencode_server.models.parts import (
    FilePart,
    Part,
    ReasoningPart,
    TextPart,
    ToolPart,
    ToolStateCompleted,
)
from wolfharness_server.opencode_server.models.session import Session


if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from pydantic_ai.messages import UserContent


logger = get_logger(__name__)

_message_info_adapter: TypeAdapter[MessageInfo] = TypeAdapter(MessageInfo)
_part_adapter: TypeAdapter[Part] = TypeAdapter(Part)


def compute_project_id(directory: str) -> str:
    """Compute OpenCode project ID from directory.

    This is the **OpenCode external contract** for project identification.
    OpenCode uses the root commit SHA1 of the git repository as the project ID,
    which determines the session directory structure:
    ``storage/session/{project_id}/``. This ID is stable across different
    machines that share the same git history.

    If not in a git repository, returns ``'global'``.

    !!! note "Dual ID scheme"
        This is DIFFERENT from ``generate_project_id()`` in
        ``wolfharness_storage.project_store``, which hashes the canonical worktree
        path for AgentPool-internal ``ProjectData`` identification. The two IDs
        serve distinct purposes and are not interchangeable:
        - ``compute_project_id`` — git root commit SHA1 (OpenCode session layout)
        - ``generate_project_id`` — path SHA1 (AgentPool project registry)

    Args:
        directory: Project directory path

    Returns:
        Project ID (root commit SHA1 or 'global')
    """
    try:
        # Get the git root directory
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=True,
        )
        git_root = result.stdout.strip()

        # Get the root commit(s)
        result = subprocess.run(
            ["git", "rev-list", "--max-parents=0", "--all"],
            cwd=git_root,
            capture_output=True,
            text=True,
            check=True,
        )
        root_commits = [c.strip() for c in result.stdout.strip().split("\n") if c.strip()]

        if root_commits:
            # Sort and return the first root commit
            return sorted(root_commits)[0]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Not in a git repo or no root commits found
    return "global"


def parse_message_info(data: dict[str, Any], *, message_id: str, session_id: str) -> MessageInfo:
    """Parse a message JSON data dict into a typed MessageInfo model.

    Injects the DB column fields (id, sessionID) into the data dict before
    validation, matching how OpenCode itself reconstructs messages from DB rows.

    Args:
        data: The JSON 'data' field from the message table
        message_id: Message ID from the DB id column
        session_id: Session ID from the DB session_id column

    Returns:
        Validated UserMessage or AssistantMessage
    """
    data["id"] = message_id
    data["sessionID"] = session_id
    return _message_info_adapter.validate_python(data)


def parse_part(data: dict[str, Any], *, part_id: str, message_id: str, session_id: str) -> Part:
    """Parse a part JSON data dict into a typed Part model.

    Injects the DB column fields (id, messageID, sessionID) into the data dict
    before validation, matching how OpenCode itself reconstructs parts from DB rows.

    Args:
        data: The JSON 'data' field from the part table
        part_id: Part ID from the DB id column
        message_id: Message ID from the DB message_id column
        session_id: Session ID from the DB session_id column

    Returns:
        Validated Part (TextPart, ToolPart, ReasoningPart, etc.)
    """
    data["id"] = part_id
    data["messageID"] = message_id
    data["sessionID"] = session_id
    return _part_adapter.validate_python(data)


def extract_text_content(parts: list[Part]) -> str:
    """Extract text content from typed parts for display.

    Groups consecutive reasoning parts into a single <thinking> block
    and only wraps them if there are also non-reasoning parts present.

    Args:
        parts: List of typed Part models

    Returns:
        Combined text content from all text and reasoning parts
    """
    text_segments: list[str] = []
    reasoning_segments: list[str] = []
    has_text = False

    for part in parts:
        if isinstance(part, TextPart):
            if part.text:
                has_text = True
                # Flush any accumulated reasoning before this text
                if reasoning_segments:
                    combined = "\n".join(reasoning_segments)
                    text_segments.append(f"<thinking>\n{combined}\n</thinking>")
                    reasoning_segments.clear()
                text_segments.append(part.text)
        elif isinstance(part, ReasoningPart) and part.text:
            reasoning_segments.append(part.text)

    # Flush remaining reasoning
    if reasoning_segments:
        combined = "\n".join(reasoning_segments)
        if has_text:
            text_segments.append(f"<thinking>\n{combined}\n</thinking>")
        else:
            # Entire message is thinking — no need for wrapper tags
            text_segments.append(combined)

    return "\n".join(text_segments)


def convert_user_content_to_parts(
    content: str | Sequence[UserContent],
    message_id: str,
    session_id: str,
    part_counter_start: int,
) -> list[TextPart]:
    """Convert user content from pydantic-ai UserPromptPart to OpenCode TextParts.

    Handles both simple string content and structured UserContent lists.
    Non-text content types (BinaryContent, FileUrl, etc.) are skipped with a
    warning since the OpenCode storage provider only persists text parts.

    Args:
        content: String or list of UserContent items from a UserPromptPart
        message_id: The parent message ID
        session_id: The parent session ID
        part_counter_start: Starting counter for generating part IDs

    Returns:
        List of TextPart models for text content items
    """
    if isinstance(content, str):
        part_id = identifier.ascending("part")
        return [
            TextPart(
                id=part_id,
                message_id=message_id,
                session_id=session_id,
                text=content,
            )
        ]

    parts: list[TextPart] = []
    for item in content:
        match item:
            case str():
                part_id = identifier.ascending("part")
                parts.append(
                    TextPart(
                        id=part_id,
                        message_id=message_id,
                        session_id=session_id,
                        text=item,
                    )
                )
            case TextContent():
                part_id = identifier.ascending("part")
                parts.append(
                    TextPart(
                        id=part_id,
                        message_id=message_id,
                        session_id=session_id,
                        text=item.content,
                    )
                )
            case BinaryContent():
                logger.warning(
                    "Skipping BinaryContent in user message",
                    media_type=item.media_type,
                    message_id=message_id,
                )
            case FileUrl():
                logger.warning(
                    "Skipping FileUrl in user message",
                    url=item.url,
                    message_id=message_id,
                )
            case _:
                logger.warning(
                    "Skipping unsupported UserContent type",
                    content_type=type(item).__name__,
                    message_id=message_id,
                )
    return parts


def _build_user_pydantic_messages(
    parts: list[Part],
    timestamp: datetime,
) -> list[ModelRequest | ModelResponse]:
    """Build ModelRequest from user message parts."""
    user_content: list[UserContent] = []
    for part in parts:
        if isinstance(part, TextPart):
            if part.text:
                user_content.append(part.text)
        elif isinstance(part, FilePart):
            url = part.url
            mime = part.mime
            if url.startswith("data:") and ";base64," in url:
                mime_part, b64_data = url.split(";base64,", 1)
                media_type = mime_part.replace("data:", "")
                data = base64.b64decode(b64_data)
                user_content.append(BinaryContent(data=data, media_type=media_type))
            elif url:
                content_item = to_user_content(url, mime)
                user_content.append(content_item)
    if user_content:
        user_part = UserPromptPart(content=user_content, timestamp=timestamp)
        return [ModelRequest(parts=[user_part], timestamp=timestamp)]
    return []


def _build_assistant_pydantic_messages(
    msg: AssistantMessage,
    parts: list[Part],
    timestamp: datetime,
) -> list[ModelRequest | ModelResponse]:
    """Build ModelResponse (+ optional ModelRequest for tool returns) from assistant parts."""
    result: list[ModelRequest | ModelResponse] = []
    response_parts: list[PydanticTextPart | ToolCallPart | ThinkingPart] = []
    tool_return_parts: list[ToolReturnPart] = []

    tokens = msg.tokens
    cache = tokens.cache
    usage = RequestUsage(
        input_tokens=tokens.input,
        output_tokens=tokens.output,
        cache_read_tokens=cache.read,
        cache_write_tokens=cache.write,
    )

    for part in parts:
        if isinstance(part, TextPart):
            if part.text:
                response_parts.append(PydanticTextPart(content=part.text))
        elif isinstance(part, ReasoningPart):
            if part.text:
                thinking_kwargs: dict[str, Any] = {"content": part.text}
                meta = part.metadata
                if meta:
                    if meta.get("thinking_id") is not None:
                        thinking_kwargs["id"] = meta["thinking_id"]
                    if meta.get("provider_name") is not None:
                        thinking_kwargs["provider_name"] = meta["provider_name"]
                    if meta.get("signature") is not None:
                        thinking_kwargs["signature"] = meta["signature"]
                    if meta.get("provider_details") is not None:
                        thinking_kwargs["provider_details"] = meta["provider_details"]
                response_parts.append(ThinkingPart(**thinking_kwargs))
        elif isinstance(part, ToolPart):
            tc_part = ToolCallPart(
                tool_name=part.tool,
                args=part.state.input,
                tool_call_id=part.call_id,
            )
            response_parts.append(tc_part)

            if isinstance(part.state, ToolStateCompleted) and part.state.output:
                return_part = ToolReturnPart(
                    tool_name=part.tool,
                    content=part.state.output,
                    tool_call_id=part.call_id,
                    timestamp=timestamp,
                )
                tool_return_parts.append(return_part)

    if response_parts:
        model_response = ModelResponse(
            parts=response_parts,
            usage=usage,
            model_name=msg.model_id,
            timestamp=timestamp,
        )
        result.append(model_response)

    if tool_return_parts:
        result.append(ModelRequest(parts=tool_return_parts, timestamp=timestamp))

    return result


def build_pydantic_messages(
    msg: MessageInfo,
    parts: list[Part],
    timestamp: datetime,
) -> list[ModelRequest | ModelResponse]:
    """Build pydantic-ai messages from typed OpenCode models.

    In OpenCode's model, assistant messages contain both tool calls AND their
    results in the same message. We split these into:
    - ModelResponse with ToolCallPart (the call)
    - ModelRequest with ToolReturnPart (the result)

    Args:
        msg: Typed UserMessage or AssistantMessage
        parts: List of typed Part models
        timestamp: Message timestamp

    Returns:
        List of pydantic-ai messages (ModelRequest and/or ModelResponse)
    """
    if isinstance(msg, UserMessage):
        return _build_user_pydantic_messages(parts, timestamp)
    return _build_assistant_pydantic_messages(msg, parts, timestamp)


def to_chat_message(
    *,
    msg: MessageInfo,
    parts: list[Part],
) -> ChatMessage[str]:
    """Convert typed OpenCode message + parts to ChatMessage.

    Args:
        msg: Typed UserMessage or AssistantMessage
        parts: List of typed Part models

    Returns:
        ChatMessage with content, pydantic messages, cost info etc.
    """
    timestamp = ms_to_datetime(msg.time.created)
    content = extract_text_content(parts)
    pydantic_messages = build_pydantic_messages(msg, parts, timestamp)

    cost_info = None
    provider_details: dict[str, Any] = {}
    parent_id: str | None = None
    model_name: str | None = None
    agent_name: str | None = msg.agent if msg.agent != "default" else None

    if isinstance(msg, AssistantMessage):
        tokens = msg.tokens
        cache = tokens.cache
        input_tokens = tokens.input + cache.read
        output_tokens = tokens.output
        if input_tokens or output_tokens:
            usage = RunUsage(input_tokens=input_tokens, output_tokens=output_tokens)
            cost = Decimal(str(msg.cost))
            cost_info = TokenCost(token_usage=usage, total_cost=cost)
        if msg.finish:
            provider_details["finish_reason"] = msg.finish
        parent_id = msg.parent_id
        model_name = msg.model_id
        agent_name = msg.agent if msg.agent != "default" else None
    elif isinstance(msg, UserMessage) and msg.model is not None:
        model_name = msg.model.model_id

    return ChatMessage[str](
        content=content,
        session_id=msg.session_id,
        role=msg.role,
        message_id=msg.id,
        name=agent_name,
        model_name=model_name,
        cost_info=cost_info,
        timestamp=timestamp,
        parent_id=parent_id,
        messages=pydantic_messages,
        provider_details=provider_details,
    )


def read_session(session_path: Path) -> Session | None:
    """Read a session from a JSON file.

    Args:
        session_path: Path to the session JSON file

    Returns:
        Session model if successful, None if file is invalid or missing
    """
    import anyenv

    if not isinstance(session_path, Path):
        session_path = Path(session_path)

    if not session_path.exists():
        return None

    try:
        content = session_path.read_text(encoding="utf-8")
        data = anyenv.load_json(content)
        return Session.model_validate(data)
    except (anyenv.JsonLoadError, ValueError, OSError, Exception) as e:  # noqa: BLE001
        logger.warning("Failed to parse session file", path=str(session_path), error=str(e))
        return None
