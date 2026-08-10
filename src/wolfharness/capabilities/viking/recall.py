"""Auto semantic recall helpers for VikingCapability.

Provides pure-function helpers for extracting user prompts, ranking and
deduplicating search results, formatting recall blocks, and injecting
system messages — used by ``VikingCapability._handle_auto_recall()``.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import TYPE_CHECKING, Any

from wolfharness.capabilities.viking.utils import truncate_text


if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext

# Maximum content snippet length shown per hit in the recall block.
_SNIPPET_LIMIT = 500
# Number of leading content characters used for deduplication hashing.
_DEDUP_PREFIX_LENGTH = 200


def _extract_latest_user_prompt(messages: list[Any]) -> str | None:
    """Extract the text content of the latest ``UserPromptPart`` in messages.

    Scans messages in reverse order, returning the first ``UserPromptPart``
    found with string content. Multimodal content (list) is skipped — only
    plain-text prompts are used for recall queries.

    Args:
        messages: The list of messages from ``ModelRequestContext.messages``.

    Returns:
        The user prompt string, or ``None`` if no text UserPromptPart is found.
    """
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    for msg in reversed(messages):
        if not isinstance(msg, ModelRequest):
            continue
        for part in reversed(msg.parts):
            if not isinstance(part, UserPromptPart):
                continue
            content = part.content
            if isinstance(content, str) and content.strip():
                return content
    return None


def _rank_and_dedup(
    hits: list[dict[str, Any]],
    query: str,
    lexical_boost: float = 0.1,
    category_boost: float = 0.05,
    context_types: list[str] | None = None,
    min_score: float = 0.3,
) -> list[dict[str, Any]]:
    """Rank and deduplicate search hits by composite score.

    Composite score: ``base_score + (lexical_overlap_count * lexical_boost)
    + (category_boost if context_type == "memory")``.

    Deduplication uses a hash of the first 200 characters of content to
    remove duplicate hits before ranking.

    Args:
        hits: List of hit dicts from ``client.search()`` or ``client.find()``.
        query: The original query string — used for lexical overlap counting.
        lexical_boost: Score boost per overlapping word between query and content.
        category_boost: Score boost applied to hits with ``context_type="memory"``.
        context_types: If non-None, filter hits to only these context types.
        min_score: Minimum composite score to include a hit in the result.

    Returns:
        Sorted list of hit dicts with an added ``_composite_score`` key,
        filtered by ``min_score`` and deduplicated by content hash.
    """
    if not hits:
        return []

    # Filter by context_types if specified
    filtered: list[dict[str, Any]] = []
    for hit in hits:
        if context_types is not None:
            ct = str(hit.get("context_type", ""))
            if ct and ct not in context_types:
                continue
        filtered.append(hit)

    # Deduplicate by first 200 chars of content
    seen_hashes: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for hit in filtered:
        content = str(hit.get("content", hit.get("text", hit.get("abstract", ""))))
        content_prefix = content[:_DEDUP_PREFIX_LENGTH]
        content_hash = hashlib.md5(content_prefix.encode("utf-8")).hexdigest()
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)
        deduped.append(hit)

    # Score each hit
    query_words = set(query.lower().split())
    scored: list[dict[str, Any]] = []
    for hit in deduped:
        base_score = float(hit.get("score", hit.get("similarity", 0.0)) or 0.0)
        content = str(hit.get("content", hit.get("text", hit.get("abstract", ""))))
        content_words = set(content.lower().split())
        overlap_count = len(query_words & content_words)
        composite = base_score + (overlap_count * lexical_boost)
        if str(hit.get("context_type", "")) == "memory":
            composite += category_boost
        hit_scored = {**hit, "_composite_score": composite}
        scored.append(hit_scored)

    # Filter by min_score and sort descending
    result = [h for h in scored if h["_composite_score"] >= min_score]
    result.sort(key=lambda h: h["_composite_score"], reverse=True)
    return result


def _format_recall_block(
    hits: list[dict[str, Any]],
    session_context: dict[str, Any] | None = None,
    max_tokens: int = 2000,
) -> str:
    """Format ranked hits as an ``<openviking-recall>`` XML block.

    Includes optional session context (from ``get_session_context()``) and
    each hit's URI, score, and content snippet. The total block is truncated
    to approximately ``max_tokens`` characters (using chars as a proxy for
    tokens with a 4:1 chars-to-tokens heuristic).

    Args:
        hits: Ranked and deduplicated hit dicts (from ``_rank_and_dedup()``).
        session_context: Optional session context dict from ``get_session_context()``.
        max_tokens: Maximum token budget — content is truncated to
            ``max_tokens * 4`` characters with a truncation indicator.

    Returns:
        A formatted ``<openviking-recall>`` XML block string. Returns an
        empty string if both hits and session_context are empty.
    """
    if not hits and not session_context:
        return ""

    max_chars = max_tokens * 4  # rough chars-to-tokens heuristic

    lines: list[str] = ["<openviking-recall>"]

    if session_context:
        lines.append("  <session-context>")
        # Session context may contain recent memories or conversation summary
        for key, value in session_context.items():
            value_str = str(value) if value is not None else ""
            if value_str:
                lines.append(f"    <{key}>{value_str}</{key}>")
        lines.append("  </session-context>")

    for hit in hits:
        uri = str(hit.get("uri", hit.get("path", "?")))
        score = hit.get("_composite_score", hit.get("score", 0.0))
        content = str(hit.get("content", hit.get("text", hit.get("abstract", ""))))
        snippet = content[:_SNIPPET_LIMIT] if len(content) > _SNIPPET_LIMIT else content
        lines.append(f'  <hit uri="{uri}" score="{score:.4f}">')
        lines.append(f"    {snippet}")
        lines.append("  </hit>")

    lines.append("</openviking-recall>")
    block = "\n".join(lines)

    return truncate_text(block, max_chars)


def _inject_system_message(
    request_context: ModelRequestContext,
    recall_block: str,
) -> ModelRequestContext:
    """Inject a system message with recall block before the latest user message.

    Creates a new ``ModelRequest`` containing a ``SystemPromptPart`` with the
    recall block text, and inserts it into ``request_context.messages``
    immediately before the last ``ModelRequest`` that contains a
    ``UserPromptPart``.

    Args:
        request_context: The original model request context.
        recall_block: The formatted recall XML block to inject.

    Returns:
        A new ``ModelRequestContext`` with the injected system message.
        If no user message is found, the original context is returned unchanged.
    """
    if not recall_block.strip():
        return request_context

    from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart

    messages = list(request_context.messages)

    # Find the index of the last ModelRequest containing a UserPromptPart
    insert_idx: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, ModelRequest) and any(isinstance(p, UserPromptPart) for p in msg.parts):
            insert_idx = i
            break

    if insert_idx is None:
        return request_context

    system_msg = ModelRequest(parts=[SystemPromptPart(content=recall_block)])
    new_messages = [*messages[:insert_idx], system_msg, *messages[insert_idx:]]

    return replace(request_context, messages=new_messages)
