"""Pure functions for conversation compaction archiving.

These helpers support ``VikingCapability._handle_compaction()`` and are
kept as module-level pure functions so they can be unit-tested without
any mock or async fixture.

Token estimation uses a CJK-aware heuristic: ASCII characters are
estimated at 4 chars/token, while CJK characters (Unified Ideographs
and Extension A) count as 1 token each.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text length.

    Uses a CJK-aware heuristic:
    - ASCII / non-CJK characters: ``len(text) // 4``
    - CJK characters (U+4E00-U+9FFF, U+3400-U+4DBF): 1 token each

    Args:
        text: The text to estimate.

    Returns:
        Estimated number of tokens.
    """
    cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf")
    ascii_count = len(text) - cjk_count
    return (ascii_count // 4) + cjk_count


def _split_archivable(
    messages: list[Any],
    keep_recent_turns: int,
) -> tuple[list[Any], list[Any]]:
    """Split messages into archivable (old) and keep (recent) lists.

    A "turn" is defined as one user message plus any subsequent assistant
    messages until the next user message. The last ``keep_recent_turns``
    turns are kept; everything before that is archivable.

    Args:
        messages: The full message list from ``request_context.messages``.
        keep_recent_turns: Number of recent turns to keep.

    Returns:
        A ``(archivable, keep)`` tuple of two lists.
    """
    if keep_recent_turns <= 0:
        return list(messages), []

    # Find indices of all user messages — each marks the start of a turn.
    user_msg_indices: list[int] = []
    for i, msg in enumerate(messages):
        if _is_user_message(msg):
            user_msg_indices.append(i)

    # If there are fewer turns than keep_recent_turns, keep everything.
    if len(user_msg_indices) <= keep_recent_turns:
        return [], list(messages)

    # The split point is the start index of the first "keep" turn.
    split_idx = user_msg_indices[-keep_recent_turns]
    archivable = list(messages[:split_idx])
    keep = list(messages[split_idx:])
    return archivable, keep


def _is_user_message(msg: Any) -> bool:
    """Check if a message is a user message (ModelRequest with UserPromptPart).

    Args:
        msg: The message to check.

    Returns:
        ``True`` if the message contains at least one ``UserPromptPart``.
    """
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    if not isinstance(msg, ModelRequest):
        return False
    return any(isinstance(part, UserPromptPart) for part in msg.parts)


def _serialize_messages(messages: list[Any]) -> str:
    """Format a list of messages as markdown for archiving.

    Each message is rendered with a role header and its textual content.

    Args:
        messages: The messages to serialize.

    Returns:
        A markdown string representing the conversation.
    """
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            user_lines = [
                f"## User\n\n{_extract_text(part.content)}\n"
                for part in msg.parts
                if isinstance(part, UserPromptPart)
            ]
            lines.extend(user_lines)
        elif isinstance(msg, ModelResponse):
            asst_lines = [
                f"## Assistant\n\n{part.content}\n"
                for part in msg.parts
                if isinstance(part, TextPart)
            ]
            lines.extend(asst_lines)
    return "\n".join(lines)


def _extract_text(content: Any) -> str:
    """Extract textual content from a message part's content field.

    Handles both plain string content and list content (which may contain
    ``TextPart`` or ``BinaryContent`` objects alongside strings).

    Args:
        content: The content field from a ``UserPromptPart``.

    Returns:
        The extracted text, or ``str(content)`` as fallback.
    """
    from pydantic_ai.messages import TextPart

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, TextPart):
                parts.append(item.content)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _summarize_messages(messages: list[Any]) -> str:
    """Generate a brief summary of archived messages.

    Takes the first 200 characters of each message's content and
    concatenates them with role headers. This is a v1 placeholder
    strategy — a future version could use an LLM call.

    Args:
        messages: The messages to summarize.

    Returns:
        A condensed summary string.
    """
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            user_lines = [
                f"**User:** {_extract_text(part.content)[:200]}"
                for part in msg.parts
                if isinstance(part, UserPromptPart)
            ]
            lines.extend(user_lines)
        elif isinstance(msg, ModelResponse):
            asst_lines = [
                f"**Assistant:** {part.content[:200]}"
                for part in msg.parts
                if isinstance(part, TextPart)
            ]
            lines.extend(asst_lines)
    return "\n\n".join(lines)


def _replace_old_messages(
    request_context: ModelRequestContext,
    archivable: list[Any],
    keep: list[Any],
    archive_uri: str,
    summary: str,
) -> ModelRequestContext:
    """Create a new ModelRequestContext with archivable messages replaced.

    Removes the archivable messages from the front of the list and
    inserts a new ``ModelRequest`` containing a ``SystemPromptPart`` with
    the summary and a reference to the archive URI. The kept messages
    follow immediately after.

    Args:
        request_context: The original model request context.
        archivable: The messages to archive (removed from the front).
        keep: The messages to keep.
        archive_uri: The Viking URI where archived content is stored.
        summary: A brief summary of the archived conversation.

    Returns:
        A new ``ModelRequestContext`` with the modified message list.
    """
    from pydantic_ai.messages import ModelRequest, SystemPromptPart

    archive_message = ModelRequest(
        parts=[
            SystemPromptPart(
                content=(
                    f"[Conversation history archived. Summary:\n{summary}\n\n"
                    f"Full archive available at: {archive_uri}\n"
                    f"Use the viking_expand tool with this URI to retrieve the "
                    f"full conversation if needed.]"
                ),
            ),
        ],
    )

    new_messages = [archive_message, *keep]
    return replace(request_context, messages=new_messages)
