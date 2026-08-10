"""Prunable-tools list builder and injector for V2 context pruning.

Builds the ``<prunable-tools>`` numbered list from message history and
injects it into the last ``ModelRequest`` as a ``SystemPromptPart`` or
``UserPromptPart``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from wolfharness.capabilities.dcp.strategies import _is_pruned
from wolfharness.capabilities.dcp.token_utils import estimate_tokens


if TYPE_CHECKING:
    from wolfharness.capabilities.dcp.config import DCPConfig
    from wolfharness.capabilities.dcp.state import DCPState

__all__ = ["META_TOOL_NAMES", "build_prunable_list", "inject_prunable_list"]

_ARGS_SUMMARY_MAX_LEN = 60

#: Tool names whose returns are meta-tools (context management tools).
#: These are auto-pruned by the pipeline and excluded from the
#: ``<prunable-tools>`` list so the model never sees them as prune targets.
META_TOOL_NAMES: frozenset[str] = frozenset({"prune", "distill", "decompress"})


def build_prunable_list(
    messages: list[ModelMessage],
    state: DCPState,
    config: DCPConfig,
) -> str:
    """Build the ``<prunable-tools>`` numbered list from message history.

    Scans ``messages`` for ``ToolReturnPart`` instances, filtering out
    protected tools, meta-tool returns (prune/distill/decompress),
    already-pruned parts, and ``None`` content.  Each surviving part is
    assigned a sequential numeric ID and its ``tool_call_id`` is stored
    in ``state.tool_id_list``.

    Args:
        messages: The conversation message list to scan.
        state: The DCP state — ``tool_id_list`` is cleared and rebuilt.
        config: The DCP config — ``protected_tools`` is used for filtering.

    Returns:
        Formatted ``<prunable-tools>`` block string, or empty string if
        no prunable tools are found.
    """
    # Clear and rebuild the ID list each time — includes ALL tools
    # (both active and pruned) so numeric IDs are stable and the
    # model can reference pruned tools for decompress.
    state.tool_id_list = []

    # First pass: build a map of tool_call_id -> call args for identification.
    call_args_map: dict[str, str] = {}
    for msg in messages:
        parts = getattr(msg, "parts", [])
        for part in parts:
            if isinstance(part, ToolCallPart) and part.tool_call_id:
                call_args_map[part.tool_call_id] = str(part.args)

    entries: list[str] = []

    for msg in messages:
        parts = getattr(msg, "parts", [])
        for part in parts:
            if not isinstance(part, ToolReturnPart):
                continue

            # Skip protected tools.
            if part.tool_name in config.protected_tools:
                continue

            # Skip meta-tool returns (prune/distill/decompress) — these are
            # auto-pruned by the pipeline and should never appear as prune
            # targets for the model.
            if part.tool_name in META_TOOL_NAMES:
                continue

            # Skip None content.
            if part.content is None:
                continue

            # Assign sequential numeric ID (includes pruned tools).
            numeric_id = len(state.tool_id_list)
            state.tool_id_list.append(part.tool_call_id)

            is_pruned = _is_pruned(part)

            if is_pruned:
                # Show pruned tools with a marker — model can decompress them.
                entries.append(f"{numeric_id}: {part.tool_name} [pruned]")
            else:
                # Use tool call args for identification (truncated to 60 chars).
                # Falls back to return content if call args not found.
                args_summary = call_args_map.get(part.tool_call_id, "")
                if not args_summary:
                    args_summary = str(part.content)
                args_summary = args_summary[:_ARGS_SUMMARY_MAX_LEN]

                # Estimate tokens for this single part.
                tokens = estimate_tokens([ModelRequest(parts=[part])])

                entries.append(f"{numeric_id}: {part.tool_name}, {args_summary} (~{tokens} tokens)")

    if not entries:
        return ""

    preamble = (
        "The following tools have been invoked and are available for pruning. "
        "This list does not mandate immediate action. Consider your current goals "
        "and the resources you need before pruning valuable tool inputs or outputs. "
        "Consolidate your prunes for efficiency; it is rarely worth pruning a single "
        "tiny tool output. Keep the context free of noise."
    )
    lines = ["<prunable-tools>", preamble, *entries, "</prunable-tools>"]
    return "\n".join(lines)


def _strip_old_injections(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Remove old ``<prunable-tools>`` / ``<compress-context>`` injections.

    Scans all messages for ``SystemPromptPart`` or ``UserPromptPart``
    instances whose content starts with ``<prunable-tools>`` or
    ``<compress-context>`` and removes them.
    """
    new_messages: list[ModelMessage] = []
    for msg in messages:
        parts = getattr(msg, "parts", [])
        if not parts:
            new_messages.append(msg)
            continue
        filtered = [
            p
            for p in parts
            if not (
                isinstance(p, (SystemPromptPart, UserPromptPart))
                and isinstance(p.content, str)
                and (
                    p.content.strip().startswith("<prunable-tools>")
                    or p.content.strip().startswith("<compress-context>")
                )
            )
        ]
        if len(filtered) == len(parts):
            new_messages.append(msg)
        else:
            new_messages.append(dataclasses.replace(msg, parts=filtered))  # type: ignore[arg-type]
    return new_messages


def inject_prunable_list(
    messages: list[ModelMessage],
    text: str,
    role: str = "system",
) -> list[ModelMessage]:
    """Inject the ``<prunable-tools>`` text into the last ``ModelRequest``.

    First strips old injections from previous react-loop steps.
    Then appends a part (``SystemPromptPart`` or ``UserPromptPart``
    depending on ``role``) to the last ``ModelRequest``.

    Args:
        messages: The conversation message list (never mutated).
        text: The ``<prunable-tools>`` text to inject.  If empty,
            ``messages`` is returned unchanged.
        role: ``"system"`` (default) or ``"user"`` — controls whether
            the injection uses ``SystemPromptPart`` or ``UserPromptPart``.

    Returns:
        A new message list with old injections stripped and the new
        injection appended.  The input list is never mutated.
    """
    if not text:
        return messages

    # Strip old injections from previous react-loop steps.
    messages = _strip_old_injections(messages)

    part: SystemPromptPart | UserPromptPart = (
        UserPromptPart(content=text) if role == "user" else SystemPromptPart(content=text)
    )

    # Find the last ModelRequest index.
    last_request_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], ModelRequest):
            last_request_idx = i
            break

    if last_request_idx == -1:
        # No ModelRequest — append a new one.
        new_request = ModelRequest(parts=[part])
        return [*messages, new_request]

    # Append part to the last ModelRequest's parts.
    last_request = messages[last_request_idx]
    new_parts = [*last_request.parts, part]
    new_request = dataclasses.replace(last_request, parts=new_parts)  # type: ignore[arg-type, assignment]

    # Build new list without mutating input.
    return [*messages[:last_request_idx], new_request, *messages[last_request_idx + 1 :]]
