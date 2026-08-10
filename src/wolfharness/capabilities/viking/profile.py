"""Profile injection helpers for VikingCapability.

Provides pure-function helpers for deriving a context hint from the run
context and formatting memory search results as an ``<openviking-profile>``
XML block — used by ``VikingCapability._handle_profile_inject()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from wolfharness.capabilities.viking.utils import truncate_text


_SNIPPET_LIMIT = 500


if TYPE_CHECKING:
    from pydantic_ai import RunContext


def _derive_context_hint(ctx: RunContext[Any]) -> str:
    """Extract a context hint from the run context's deps.

    Tries (in order):
    1. Agent name from ``deps.agent_name`` (if present and non-empty).
    2. Session metadata from ``deps.session_metadata`` (if present, a dict
       with a ``"topic"`` or ``"description"`` key).
    3. First 100 characters of the latest user prompt.

    Args:
        ctx: The pydantic-ai run context.

    Returns:
        A context hint string, or an empty string if nothing can be derived.
    """
    deps = ctx.deps

    # Tier 1: agent name
    try:
        agent_name = deps.agent_name
    except AttributeError:
        agent_name = ""
    if isinstance(agent_name, str) and agent_name:
        return agent_name

    # Tier 2: session metadata
    try:
        session_metadata = deps.session_metadata
    except AttributeError:
        session_metadata = None
    if isinstance(session_metadata, dict):
        for key in ("topic", "description"):
            value = session_metadata.get(key)
            if isinstance(value, str) and value:
                return value

    # Tier 3: latest user prompt keywords (first 100 chars)
    messages = _extract_messages_from_ctx(ctx)
    prompt = _extract_latest_user_prompt_text(messages)
    if prompt:
        return prompt[:100]

    return ""


def _format_profile_block(
    results: dict[str, Any] | list[Any],
    max_tokens: int = 1000,
) -> str:
    """Format memory search results as an ``<openviking-profile>`` XML block.

    Parses the SDK ``find()`` response (dict with grouped keys or a flat list
    of hits) and renders sections for project context, historical decisions,
    and relevant resources. The total block is truncated to approximately
    ``max_tokens`` characters (using chars as a proxy with a 4:1 heuristic).

    Args:
        results: The raw response from ``client.find()`` — either a dict
            with ``"hits"``, ``"results"``, or Viking grouped keys
            (``"memories"``, ``"resources"``, ``"skills"``), or a flat list
            of hit dicts.
        max_tokens: Maximum token budget — content is truncated to
            ``max_tokens * 4`` characters with a truncation indicator.

    Returns:
        A formatted ``<openviking-profile>`` XML block string. Returns an
        empty string if there are no hits.
    """
    hits = _extract_hits(results)
    if not hits:
        return ""

    max_chars = max_tokens * 4  # rough chars-to-tokens heuristic

    lines: list[str] = ["<openviking-profile>"]

    # Categorize hits into sections
    memories: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        context_type = str(hit.get("context_type", ""))
        if context_type == "memory":
            memories.append(hit)
        elif context_type == "resource":
            resources.append(hit)
        else:
            other.append(hit)

    if memories:
        lines.append("  <project-context>")
        lines.extend(f"    {_format_hit(hit)}" for hit in memories)
        lines.append("  </project-context>")

    if resources:
        lines.append("  <relevant-resources>")
        lines.extend(f"    {_format_hit(hit)}" for hit in resources)
        lines.append("  </relevant-resources>")

    if other:
        lines.append("  <historical-decisions>")
        lines.extend(f"    {_format_hit(hit)}" for hit in other)
        lines.append("  </historical-decisions>")

    lines.append("</openviking-profile>")
    block = "\n".join(lines)

    return truncate_text(block, max_chars)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_messages_from_ctx(ctx: RunContext[Any]) -> list[Any]:
    """Extract the messages list from the run context, if available.

    Returns:
        The messages list, or an empty list if not accessible.
    """
    try:
        messages = ctx.messages
    except AttributeError:
        return []
    if isinstance(messages, list):
        return messages
    return []


def _extract_latest_user_prompt_text(messages: list[Any]) -> str:
    """Extract the text content of the latest ``UserPromptPart``.

    Scans messages in reverse order, returning the first ``UserPromptPart``
    found with string content.

    Returns:
        The user prompt string, or an empty string if none found.
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
    return ""


def _extract_hits(results: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    """Extract hit dicts from an SDK find() response.

    Handles dict responses (with ``hits``, ``results``, or Viking grouped
    keys) and flat list responses.

    Returns:
        A list of hit dicts. Non-dict entries are skipped.
    """
    if isinstance(results, dict):
        hits: list[Any] = (
            results.get("hits")
            or results.get("results")
            or (
                results.get("memories", [])
                + results.get("resources", [])
                + results.get("skills", [])
            )
        )
    else:
        hits = results

    return [h for h in hits if isinstance(h, dict)]


def _format_hit(hit: dict[str, Any]) -> str:
    """Format a single hit as an XML entry string.

    Args:
        hit: A hit dict from search results.

    Returns:
        A formatted string with URI, score (if present), and content snippet.
    """
    uri = str(hit.get("uri", hit.get("path", "?")))
    score = hit.get("score", hit.get("similarity"))
    content = str(hit.get("content", hit.get("text", hit.get("abstract", ""))))
    snippet = content[:_SNIPPET_LIMIT] if len(content) > _SNIPPET_LIMIT else content

    parts: list[str] = [f'<item uri="{uri}"']
    if score is not None:
        if isinstance(score, float):
            parts.append(f' score="{score:.4f}"')
        else:
            parts.append(f' score="{score}"')
    parts.append(">")
    parts.append(snippet)
    parts.append("</item>")
    return "".join(parts)
