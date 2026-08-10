"""Context pressure nudge system for Dynamic Context Pruning.

Generates nudge prompts that inform the model about context
window pressure, encouraging it to use pruning tools before the context
becomes critically full.

The nudge text includes:
- Current token usage as ``N / M tokens (P%)``
- Context management protocol and immediate action items
- Top-3 largest prunable tool outputs by estimated token count

``nudge_counter`` increment happens in ``capability.py`` ``before_run``,
NOT in this module.  The counter reset (to 0) also happens externally
after the nudge is injected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai.messages import (
    ModelRequest,
    ToolReturnPart,
)

from wolfharness.capabilities.dcp.state import WatermarkLevel
from wolfharness.capabilities.dcp.strategies import _is_pruned
from wolfharness.capabilities.dcp.token_utils import estimate_tokens


if TYPE_CHECKING:
    from wolfharness.capabilities.dcp.config import DCPConfig
    from wolfharness.capabilities.dcp.state import DCPState

__all__ = ["build_nudge_text"]

_TOP_N = 3


def _estimate_part_tokens(part: ToolReturnPart) -> int:
    """Estimate token count for a single ``ToolReturnPart``.

    Wraps the part in a throwaway ``ModelRequest`` because
    ``estimate_tokens`` accepts only a list of ``ModelMessage``.
    """
    return estimate_tokens([ModelRequest(parts=[part])])


def _collect_prunable_parts(
    state: DCPState,
    config: DCPConfig,
) -> list[ToolReturnPart]:
    """Collect prunable ``ToolReturnPart`` instances from ``state.current_messages``.

    Filters out protected tools, already-pruned parts, and ``None`` content.
    Only parts whose ``tool_call_id`` appears in ``state.tool_id_list`` are
    included.
    """
    if state.current_messages is None:
        return []

    id_set = set(state.tool_id_list)
    result: list[ToolReturnPart] = []

    for msg in state.current_messages:
        parts = getattr(msg, "parts", [])
        for part in parts:
            if not isinstance(part, ToolReturnPart):
                continue
            if part.tool_call_id not in id_set:
                continue
            if part.tool_name in config.protected_tools:
                continue
            if _is_pruned(part):
                continue
            if part.content is None:
                continue
            result.append(part)

    return result


def build_nudge_text(state: DCPState, config: DCPConfig) -> str:
    """Build context pressure nudge text from current state and config.

    The nudge urgency scales with the current watermark level:

    - **NORMAL** (< info_threshold): Gentle informational note, no action expected.
    - **INFO** (info_threshold ~ warning_threshold): Suggestive tone, consider reviewing context.
    - **WARNING** (warning_threshold ~ critical_threshold): Moderate urgency, recommend taking \
action.
    - **CRITICAL** (critical_threshold+): Forceful, immediate action required.

    The text also includes top-3 largest prunable tool outputs by
    estimated token count (only at INFO and above).

    Args:
        state: The current DCP session state.
        config: The DCP configuration.

    Returns:
        A nudge string describing the context pressure situation.
    """
    current = state.current_tokens
    maximum = config.max_context_tokens
    pct = round(current / maximum * 100) if maximum > 0 else 0
    level = state.watermark_level

    # Collect and rank prunable tool outputs.
    prunable = _collect_prunable_parts(state, config)
    ranked = sorted(
        prunable,
        key=_estimate_part_tokens,
        reverse=True,
    )
    top = ranked[:_TOP_N]

    lines: list[str] = ["<system-reminder>"]

    if level >= WatermarkLevel.CRITICAL:
        # --- CRITICAL (90%+): Forceful, immediate action ---
        lines.append("CRITICAL CONTEXT WARNING")
        lines.append(
            f"Your context window is almost full ({current} / {maximum} tokens ({pct}%). "
            "Strict adherence to context hygiene is required.",
        )
        lines.append("")
        lines.append("PROTOCOL")
        lines.append(
            "You should prioritize context management, but do not interrupt a critical "
            "atomic operation if one is in progress. Once the immediate step is done, "
            "you must perform context management.",
        )
        lines.append("")
        lines.append("IMMEDIATE ACTION REQUIRED")
        lines.append(
            "- KNOWLEDGE PRESERVATION: If holding valuable raw data you POTENTIALLY will "
            "need in your task, use the `distill` tool. Produce a high-fidelity distillation "
            "to preserve insights - be thorough. Use `decompress` to restore previously "
            "pruned content if needed.",
        )
        _append_prunable_list(lines, top)

    elif level >= WatermarkLevel.WARNING:
        # --- WARNING (75-89%): Moderate urgency, recommend action ---
        lines.append("Context Pressure Warning")
        lines.append(
            f"Your context window is filling up ({current} / {maximum} tokens ({pct}%). "
            "You should prioritize context management soon.",
        )
        lines.append("")
        lines.append("RECOMMENDED ACTIONS")
        lines.append(
            "- Use `distill` to condense valuable tool outputs into high-signal summaries "
            "before the context gets too full.",
        )
        lines.append(
            "- Use `prune` to remove tool outputs that are noise, irrelevant, or superseded.",
        )
        _append_prunable_list(lines, top)

    elif level >= WatermarkLevel.INFO:
        # --- INFO (60-74%): Suggestive, consider reviewing ---
        lines.append("Context Status")
        lines.append(
            f"Context usage is moderate ({current} / {maximum} tokens ({pct}%). "
            "Consider reviewing tool outputs that are no longer needed.",
        )
        lines.append("")
        lines.append("SUGGESTED ACTIONS (not urgent)")
        lines.append(
            "- `distill`: condense key findings from tool calls to preserve insights.",
        )
        lines.append(
            "- `prune`: remove tool outputs that yielded no value or are superseded.",
        )
        _append_prunable_list(lines, top)

    else:
        # --- NORMAL (< 60%): Gentle note, no action expected ---
        lines.append("Context Status")
        lines.append(
            f"Context usage is low ({current} / {maximum} tokens ({pct}%). "
            "No context management action needed right now.",
        )
        lines.append(
            "Tools `distill` and `prune` are available if the context grows later.",
        )

    lines.append("</system-reminder>")

    return "\n".join(lines)


def _append_prunable_list(lines: list[str], top: list[ToolReturnPart]) -> None:
    """Append the largest prunable tool outputs to the nudge lines.

    Args:
        lines: The nudge text lines being built (modified in place).
        top: The top-N prunable tool parts, sorted by token count descending.
    """
    if top:
        lines.append(
            "- NOISE REMOVAL: If you read files or ran commands that yielded no value, "
            "use the `prune` tool to remove them. If newer tools supersede older ones, "
            "prune the old.",
        )
        lines.append("")
        lines.append("Largest prunable tool outputs:")
        for i, part in enumerate(top):
            tokens = _estimate_part_tokens(part)
            summary = str(part.content)[:60]
            lines.append(f"  {i + 1}. {part.tool_name} (~{tokens} tokens): {summary}")
    else:
        lines.append(
            "- NOISE REMOVAL: If you read files or ran commands that yielded no value, "
            "use the `prune` tool to remove them. If newer tools supersedes older ones, "
            "prune the old.",
        )
        lines.append("")
        lines.append("No prunable tool outputs detected.")
