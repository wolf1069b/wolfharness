"""Auto conversation ingestion helpers for VikingCapability.

Provides pure-function message sanitization and conversation-pair extraction,
plus an async ingestion coroutine that writes conversation turns to a Viking
session via the SDK client.

These helpers are called from ``VikingCapability._handle_auto_ingest()`` during
``before_model_request`` (lazy ingestion of the previous turn) and flushed
synchronously in ``after_run()``.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Sequence


# Regex patterns for stripping injected XML blocks.
# ``re.DOTALL`` ensures the pattern matches across newlines.
_RECALL_RE = re.compile(r"<openviking-recall>.*?</openviking-recall>", re.DOTALL)
_PROFILE_RE = re.compile(r"<openviking-profile>.*?</openviking-profile>", re.DOTALL)
_CONTEXT_RE = re.compile(r"<openviking-context\b[^>]*>.*?</openviking-context>", re.DOTALL)
_SESSION_ARCHIVE_RE = re.compile(r"<session-archive\b[^>]*>.*?</session-archive>", re.DOTALL)
_USER_PROFILE_RE = re.compile(r"<user-profile\b[^>]*>.*?</user-profile>", re.DOTALL)

_REPLACEMENT = "[recalled context omitted]"

# Template for the memory-intent marker appended to deferred remember
# captures. The marker is the only channel through which the extraction
# LLM can see the caller's intent (the server MemoryPolicy has no
# dedicated intent field), so it must survive sanitization — it does,
# since sanitization only strips ``<openviking-recall>`` /
# ``<openviking-profile>`` blocks.
_MEMORY_INTENT_TEMPLATE = "<memory-intent>{reason}</memory-intent>"


def _sanitize_message(content: str, enabled: bool = True) -> str:
    """Strip injected Viking XML blocks from message content.

    Replaces ``<openviking-recall>...</openviking-recall>``,
    ``<openviking-profile>...</openviking-profile>``,
    ``<openviking-context>...</openviking-context>``,
    ``<session-archive>...</session-archive>``, and
    ``<user-profile>...</user-profile>`` blocks with
    ``[recalled context omitted]`` to prevent feedback loops where
    recalled context is re-ingested as original conversation.

    Args:
        content: The raw message content string.
        enabled: When ``False``, return the content unchanged.

    Returns:
        The sanitized content string.
    """
    if not enabled:
        return content
    sanitized = _RECALL_RE.sub(_REPLACEMENT, content)
    sanitized = _PROFILE_RE.sub(_REPLACEMENT, sanitized)
    sanitized = _CONTEXT_RE.sub(_REPLACEMENT, sanitized)
    sanitized = _SESSION_ARCHIVE_RE.sub(_REPLACEMENT, sanitized)
    return _USER_PROFILE_RE.sub(_REPLACEMENT, sanitized)


def _extract_conversation_pairs(
    messages: Sequence[Any],
    start_idx: int,
) -> list[dict[str, str]]:
    """Extract user/assistant conversation pairs from model messages.

    Scans ``messages`` starting at ``start_idx``, extracting text content
    from ``ModelRequest`` (user prompts) and ``ModelResponse`` (assistant
    text) objects. Only string content is extracted — binary content and
    tool calls are skipped.

    Args:
        messages: Sequence of ``ModelRequest`` / ``ModelResponse`` objects.
        start_idx: Index to start scanning from (inclusive).

    Returns:
        A list of ``{"role": "user"|"assistant", "content": str}`` dicts
        representing the conversation since ``start_idx``.
    """
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

    pairs: list[dict[str, str]] = []
    for msg in messages[start_idx:]:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    content = _extract_text_content(part.content)
                    if content:
                        pairs.append({"role": "user", "content": content})
        elif isinstance(msg, ModelResponse):
            text_parts = [p for p in msg.parts if isinstance(p, TextPart)]
            if text_parts:
                combined = "\n".join(p.content for p in text_parts)
                pairs.append({"role": "assistant", "content": combined})
    return pairs


def _extract_full_trace(
    messages: Sequence[Any],
    start_idx: int,
) -> list[dict[str, Any]]:
    """Extract full conversation trace including tool calls and results.

    Unlike ``_extract_conversation_pairs`` which only captures user/assistant
    text, this function extracts **all** part types from pydantic-ai messages
    and produces Viking-compatible ``{role, parts}`` dicts:

    - ``UserPromptPart`` → ``{"type": "text", "text": "..."}``
    - ``TextPart`` / ``ThinkingPart`` → ``{"type": "text", "text": "..."}``
    - ``ToolCallPart`` → ``{"type": "tool", "tool_name": ..., "tool_status": "running", ...}``
    - ``ToolReturnPart`` → ``{"type": "tool", "tool_status": "completed"/"error"}``

    Args:
        messages: Sequence of ``ModelRequest`` / ``ModelResponse`` objects.
        start_idx: Index to start scanning from (inclusive).

    Returns:
        A list of ``{"role": "user"|"assistant", "parts": [...]}`` dicts
        in Viking-compatible format.
    """
    from pydantic_ai.messages import (
        ModelRequest,
        TextPart,
        ThinkingPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )

    result: list[dict[str, Any]] = []
    for msg in messages[start_idx:]:
        parts: list[dict[str, Any]] = []
        role = "user" if isinstance(msg, ModelRequest) else "assistant"

        for part in msg.parts:
            part_role = _trace_part_role(part, default=role)
            if part_role != role:
                if parts:
                    result.append({"role": role, "parts": parts})
                    parts = []
                role = part_role

            if isinstance(part, UserPromptPart):
                content = _extract_text_content(part.content)
                if content:
                    _append_text_part(parts, _sanitize_message(content))

            elif isinstance(part, TextPart):
                if part.content.strip():
                    _append_text_part(parts, _sanitize_message(part.content))

            elif isinstance(part, ThinkingPart):
                if part.content.strip():
                    _append_text_part(parts, _sanitize_message(part.content), subtype="thinking")

            elif isinstance(part, ToolCallPart):
                tool_part: dict[str, Any] = {
                    "type": "tool",
                    "tool_name": part.tool_name or "",
                    "tool_status": "running",
                }
                if part.tool_call_id:
                    tool_part["tool_id"] = part.tool_call_id
                if part.args is not None:
                    tool_part["tool_input"] = _serialize_tool_args(part.args)
                parts.append(tool_part)

            elif isinstance(part, ToolReturnPart):
                outcome = getattr(part, "outcome", None)
                tool_part = {
                    "type": "tool",
                    "tool_name": part.tool_name or "",
                    "tool_status": "error" if outcome == "failed" else "completed",
                }
                if part.tool_call_id:
                    tool_part["tool_id"] = part.tool_call_id
                if part.content is not None:
                    tool_part["tool_output"] = _serialize_tool_output(part.content)
                parts.append(tool_part)

        if parts:
            result.append({"role": role, "parts": parts})

    return result


def _trace_part_role(part: Any, *, default: str) -> str:
    from pydantic_ai.messages import ToolCallPart, ToolReturnPart, UserPromptPart

    if isinstance(part, (ToolCallPart, ToolReturnPart)):
        return "assistant"
    if isinstance(part, UserPromptPart):
        return "user"
    return default


def _extract_text_content(content: Any) -> str:
    """Extract human-readable text from pydantic-ai prompt content."""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for item in content:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            value = item.get("text") or item.get("content")
            text = value if isinstance(value, str) else ""
        else:
            value = getattr(item, "text", None)
            if not isinstance(value, str):
                value = getattr(item, "content", None)
            text = value if isinstance(value, str) else ""
        if text.strip():
            chunks.append(text.strip())
    return "\n".join(chunks)


def _append_text_part(
    parts: list[dict[str, Any]],
    text: str,
    *,
    subtype: str | None = None,
) -> None:
    """Append text while coalescing adjacent text chunks with matching subtype."""
    stripped = text.strip()
    if not stripped:
        return
    if parts and parts[-1].get("type") == "text" and parts[-1].get("subtype") == subtype:
        parts[-1]["text"] = f"{parts[-1].get('text', '')}\n{stripped}"
        return
    part: dict[str, Any] = {"type": "text", "text": stripped}
    if subtype is not None:
        part["subtype"] = subtype
    parts.append(part)


def _serialize_tool_args(args: Any) -> Any:
    """Serialize tool call arguments to a JSON-safe dict.

    Handles string-encoded JSON (parsed), native dicts (passed through),
    and other types (wrapped in ``{"value": ...}``).
    """
    if isinstance(args, str):
        try:
            return json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return {"value": args}
    if isinstance(args, dict):
        return args
    if isinstance(args, list):
        return {"items": args}
    return {"value": str(args)} if args is not None else {}


def _serialize_tool_output(content: Any) -> str:
    """Serialize tool return content to a string.

    Preserves the complete textual payload so Viking has the auditable
    tool trace. Display-layer truncation should happen outside ingestion.
    """
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(content)


async def _ingest_conversation(
    client: Any,
    messages: list[dict[str, str]],
    *,
    session_id: str,
    source_type: str = "wolfharness",
    keep_recent_turns: int = 0,
) -> Any:
    """Write conversation messages to a Viking session.

    Creates a new session, adds each message, and commits with the
    configured retention policy. Errors are caught and logged by the
    caller — this function raises on failure.

    Args:
        client: The Viking SDK ``AsyncHTTPClient`` instance.
        messages: Conversation pairs from ``_extract_conversation_pairs``.
        session_id: Viking session ID for the new session.
        source_type: Source type metadata for the session.
        keep_recent_turns: Number of recent turns to retain after commit.
            When 0, no retention parameter is passed to ``commit_session``.

    Returns:
        The commit response from ``commit_session`` (e.g.
        ``{"archive_uri": ..., "task_id": ...}`` on SDK clients that
        expose asynchronous extraction).
    """
    await client.create_session(session_id=session_id)
    for msg in messages:
        await client.add_message(session_id, msg["role"], msg["content"])
    commit_kwargs: dict[str, Any] = {}
    if keep_recent_turns > 0:
        commit_kwargs["keep_recent_count"] = keep_recent_turns
    return await client.commit_session(session_id, **commit_kwargs)


async def read_memory_diff(client: Any, archive_uri: str) -> dict[str, Any]:
    """Read and parse the extraction memory diff for a committed archive.

    The server writes ``memory_diff.json`` under the archive after the
    asynchronous Phase 2 extraction completes. Handles both structured
    (dict) and serialized (str/bytes) ``read()`` responses.

    Args:
        client: The Viking SDK ``AsyncHTTPClient`` instance.
        archive_uri: Archive URI returned by ``commit_session``.

    Returns:
        The parsed diff dict. Raises ``ValueError`` if the payload is
        not valid JSON.
    """
    raw = await client.read(f"{archive_uri}/memory_diff.json")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, bytes):
        raw = raw.decode()
    if isinstance(raw, str):
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    return {}


def format_memory_diff_summary(diff: dict[str, Any]) -> str:
    """Format a memory diff into a single-line steer summary.

    Recognizes ``added``/``updated``/``deleted`` keys in either plain
    or ``*_uris`` form. Returns an empty string when the diff carries
    no URIs (nothing worth notifying).

    Args:
        diff: Parsed memory diff dict.

    Returns:
        A one-line summary, or ``""`` when empty.
    """
    sections: list[str] = []
    for label, keys in (
        ("added", ("added", "added_uris")),
        ("updated", ("updated", "updated_uris")),
        ("deleted", ("deleted", "deleted_uris")),
    ):
        uris = next((diff.get(key) for key in keys if diff.get(key)), None)
        if uris:
            sections.append(f"{label}: {', '.join(str(uri) for uri in uris)}")
    if not sections:
        return ""
    return f"[Viking memory updated] {'; '.join(sections)}"
