"""Strategy helpers for Dynamic Context Pruning.

Provides metadata dual-view helpers, exact deduplication, thinking-content
stripping, and the inlined ``purge_failed_tool_inputs`` strategy that
prunes tool OUTPUTS (``ToolReturnPart.content``) of failed tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, runtime_checkable
from uuid import uuid4

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from wolfharness.capabilities.dcp.state import CompressionBlock, DCPState


if TYPE_CHECKING:
    from wolfharness.capabilities.dcp.block_store import CompressionBlockStore
    from wolfharness.capabilities.dcp.config import DCPConfig


__all__ = [
    "PrunableState",
    "purge_failed_tool_inputs",
]

#: Minimum occurrences of a tool call signature before deduplication triggers.
DEDUP_MIN_COUNT: int = 2


# ---------------------------------------------------------------------------
# PruneMetadata — TypedDict for metadata dual-view (compile pattern).
# ---------------------------------------------------------------------------


class PruneMetadata(TypedDict):
    """Metadata stored on ``ToolReturnPart`` when pruned.

    Enables idempotent re-pruning (``_is_pruned`` check) and decompress
    restoration.  Uses ``TypedDict`` instead of ``dict[str, Any]`` for
    type safety.

    Attributes:
        _prune_original: The original ``content`` value before pruning.
        _prune_kind: The kind of pruning (``"prune"``, ``"dedup"``, etc.).
        _prune_summary: Optional summary text (for distill).
    """

    _prune_original: str | object
    _prune_kind: str
    _prune_summary: str | None


# ---------------------------------------------------------------------------
# Metadata dual-view helpers.
# ---------------------------------------------------------------------------


def _prune_part(
    part: ToolReturnPart,
    replacement: str,
    kind: str,
    summary: str | None = None,
) -> ToolReturnPart:
    """Replace a ``ToolReturnPart``'s content, storing the original in metadata.

    Idempotent: if the part is already pruned (``_is_pruned`` returns
    ``True``), the part is returned unchanged.

    Uses ``dataclasses.replace`` to create a new part — never mutates
    the original.

    Args:
        part: The ``ToolReturnPart`` to prune.
        replacement: The new content string (e.g. ``"[pruned]"``).
        kind: The pruning kind (e.g. ``"prune"``, ``"dedup"``).
        summary: Optional summary text for distill.

    Returns:
        A new ``ToolReturnPart`` with replaced content and metadata.
    """
    if _is_pruned(part):
        return part

    metadata: PruneMetadata = PruneMetadata(
        _prune_original=part.content,
        _prune_kind=kind,
        _prune_summary=summary,
    )
    return replace(part, content=replacement, metadata=metadata)


def _is_pruned(part: ToolReturnPart) -> bool:
    """Check whether a ``ToolReturnPart`` has already been pruned.

    A part is considered pruned if its ``metadata`` is a ``dict``
    containing the ``_prune_kind`` key.

    Args:
        part: The ``ToolReturnPart`` to check.

    Returns:
        ``True`` if the part has been pruned, ``False`` otherwise.
    """
    return isinstance(part.metadata, dict) and "_prune_kind" in part.metadata


# ---------------------------------------------------------------------------
# Protocol for duck-typing state objects passed to DCP strategies.
# ---------------------------------------------------------------------------


@runtime_checkable
class PrunableState(Protocol):
    """Structural protocol for state objects passed to DCP strategies.

    Both ``_PrunableStateAdapter`` and ``DCPState`` satisfy this protocol
    since they expose ``pruned_tools: set[str]`` and ``current_turn: int``.
    Using a protocol avoids ``cast`` / ``Any`` while allowing structural
    typing.
    """

    pruned_tools: set[str]
    current_turn: int


# ---------------------------------------------------------------------------
# _PrunableStateAdapter — temporary state for DCP strategy calls.
# ---------------------------------------------------------------------------


@dataclass
class _PrunableStateAdapter:
    """Temporary state adapter for calling DCP strategies.

    ``DCPState`` does not have a ``pruned_tools`` field.  When DCP
    strategies like ``purge_failed_tool_inputs`` need access to that
    field, this adapter provides it as a local, ephemeral set that is
    discarded after the call.

    Attributes:
        pruned_tools: Ephemeral set populated by DCP strategies.
        current_turn: Current conversation turn (from ``DCPState``).
        applied_action_ids: Already-applied action IDs (from ``DCPState``).
        block_store: Optional block store for recording ``CompressionBlock``s.
    """

    pruned_tools: set[str] = field(default_factory=set)
    current_turn: int = 0
    applied_action_ids: set[str] = field(default_factory=set)
    block_store: CompressionBlockStore | None = None


# ---------------------------------------------------------------------------
# Config adapter — wraps ``DCPConfig`` for DCP strategy keyword names.
# ---------------------------------------------------------------------------


class _StrategyConfigAdapter:
    """Adapt ``DCPConfig`` fields to DCP strategy keyword names.

    DCP strategies expect ``iteration_protection`` and ``purge_error_iterations``
    keyword arguments, while ``DCPConfig`` uses ``step_protection``
    and ``purge_error_steps``.  This adapter bridges the naming gap so the
    capability can pass a single config object to DCP strategies without
    duplicating field mappings.
    """

    def __init__(self, config: DCPConfig) -> None:
        """Store the wrapped config.

        Args:
            config: The ``DCPConfig`` to adapt.
        """
        self._config = config

    @property
    def protected_tools(self) -> set[str]:
        """Return the set of protected tool names from the wrapped config."""
        return self._config.protected_tools

    @property
    def step_protection(self) -> int:
        """Return ``step_protection`` (maps to DCP ``iteration_protection``)."""
        return self._config.step_protection

    @property
    def purge_error_steps(self) -> int:
        """Return ``purge_error_steps`` (maps to DCP ``purge_error_iterations``)."""
        return self._config.purge_error_steps


# ---------------------------------------------------------------------------
# Apply pruned tools — replace ToolReturnPart content via _prune_part.
# ---------------------------------------------------------------------------


def _apply_pruned_tools(
    messages: list[ModelMessage],
    adapter: _PrunableStateAdapter,
    session_id: str = "default",
) -> None:
    """Replace ``ToolReturnPart`` content for pruned tool call IDs.

    For each ``tool_call_id`` in ``adapter.pruned_tools`` (excluding those
    in ``adapter.applied_action_ids``), replace the ``content`` of the
    matching ``ToolReturnPart`` using ``_prune_part`` with kind
    ``"dedup"``.  A ``CompressionBlock`` of kind ``"dedup"`` is created
    and stored in the adapter's block store for each pruned tool.  After
    processing, ``adapter.pruned_tools`` is cleared.

    Messages are mutated in-place at the list level — individual parts are
    replaced via ``dataclasses.replace()`` (never mutated in-place).

    Args:
        messages: The conversation message list (mutated in-place at list level).
        adapter: The state adapter with ``pruned_tools`` to apply and
            ``applied_action_ids`` for dedup tracking.
        session_id: The session ID for block store namespace isolation.
    """
    if not adapter.pruned_tools:
        return

    # Only process IDs not already applied.
    to_apply = adapter.pruned_tools - adapter.applied_action_ids
    if not to_apply:
        return

    for i, msg in enumerate(messages):
        if not isinstance(msg, (ModelRequest, ModelResponse)):
            continue

        new_parts: list[Any] = []
        modified = False
        for part in msg.parts:
            if isinstance(part, ToolReturnPart) and part.tool_call_id in to_apply:
                new_parts.append(_prune_part(part, "[pruned]", "dedup"))
                modified = True

                # Record a compression block.
                block = CompressionBlock(
                    block_id=f"cb_{uuid4().hex[:12]}",
                    original_tool_call_ids=(part.tool_call_id,) if part.tool_call_id else (),
                    compressed_content="[pruned]",
                    kind="dedup",
                )
                if adapter.block_store is not None:
                    adapter.block_store.put(session_id, block)

                # Mark as applied.
                if part.tool_call_id:
                    adapter.applied_action_ids.add(part.tool_call_id)
            else:
                new_parts.append(part)

        if modified:
            messages[i] = replace(msg, parts=new_parts)  # type: ignore[arg-type]

    # Clear pruned_tools after applying.
    adapter.pruned_tools.clear()


# ---------------------------------------------------------------------------
# Exact-match deduplication.
# ---------------------------------------------------------------------------


def _tool_signature(part: ToolCallPart) -> str:
    """Compute a canonical tool call signature for deduplication.

    The signature is ``tool_name + json.dumps(sorted(normalized(args)))``
    where ``normalized(args)`` converts string args to a dict and sorts
    keys, ensuring that calls with the same tool name and identical
    arguments produce the same signature regardless of key ordering.

    Args:
        part: The ``ToolCallPart`` to compute a signature for.

    Returns:
        A string signature uniquely identifying the tool call's
        name and arguments.
    """
    tool_name = part.tool_name
    args = part.args

    if args is None:
        normalized: dict[str, object] = {}
    elif isinstance(args, str):
        try:
            parsed = json.loads(args)
            normalized = parsed if isinstance(parsed, dict) else {"_value": parsed}
        except (json.JSONDecodeError, TypeError):
            normalized = {"_raw": args}
    elif isinstance(args, dict):
        normalized = args
    else:
        normalized = {"_value": str(args)}

    sorted_args = json.dumps(sorted(normalized.items()))
    return f"{tool_name}{sorted_args}"


def _dedup_exact(
    messages: list[ModelMessage],
    state: DCPState,
    config: DCPConfig,
) -> None:
    """Exact-match deduplication of tool calls.

    Computes a tool signature (``tool_name + json.dumps(sorted(normalized(args)))``)
    for each ``ToolCallPart``.  When the same signature appears multiple
    times, only the **last** occurrence is kept; earlier occurrences have
    their corresponding ``ToolReturnPart`` pruned with ``"[duplicate removed]"``
    via ``_prune_part``.

    Protected tools (from ``config.protected_tools``) are exempt from
    deduplication.

    This function mutates messages in-place at the list level — individual
    parts are replaced via ``_prune_part`` (which uses
    ``dataclasses.replace``).

    Args:
        messages: The conversation message list (mutated in-place).
        state: The DCP state (not populated — kept for interface
            consistency with the capability's Phase 2 pipeline).
        config: The config providing ``protected_tools``.
    """
    if not messages:
        return

    protected = config.protected_tools

    # Map signature → list of (msg_idx, part_idx) for all occurrences.
    seen: dict[str, list[tuple[int, int]]] = {}

    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, (ModelRequest, ModelResponse)):
            continue
        for part_idx, part in enumerate(msg.parts):
            if not isinstance(part, ToolCallPart):
                continue
            if part.tool_name in protected:
                continue
            sig = _tool_signature(part)
            if sig not in seen:
                seen[sig] = []
            seen[sig].append((msg_idx, part_idx))

    # For each signature with multiple occurrences, prune earlier ones.
    # We need to find the ToolReturnPart that corresponds to each
    # ToolCallPart (matched by tool_call_id) and prune it.
    for positions in seen.values():
        if len(positions) < DEDUP_MIN_COUNT:
            continue

        # Keep the last occurrence; prune all earlier ones.
        earlier_positions = positions[:-1]

        for msg_idx, part_idx in earlier_positions:
            msg = messages[msg_idx]
            if not isinstance(msg, (ModelRequest, ModelResponse)):
                continue

            # Find the ToolCallPart at this position to get tool_call_id.
            call_part = msg.parts[part_idx]
            if not isinstance(call_part, ToolCallPart):
                continue

            tool_call_id = call_part.tool_call_id
            if not tool_call_id:
                continue

            # Find and prune the corresponding ToolReturnPart across
            # all messages (it may be in a different message).
            for search_idx, search_msg in enumerate(messages):
                if not isinstance(search_msg, (ModelRequest, ModelResponse)):
                    continue

                new_parts: list[Any] = []
                search_modified = False
                for search_part in search_msg.parts:
                    if (
                        isinstance(search_part, ToolReturnPart)
                        and search_part.tool_call_id == tool_call_id
                        and not _is_pruned(search_part)
                    ):
                        new_parts.append(
                            _prune_part(search_part, "[duplicate removed]", "dedup"),
                        )
                        search_modified = True
                    else:
                        new_parts.append(search_part)

                if search_modified:
                    messages[search_idx] = replace(search_msg, parts=new_parts)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Thinking-content stripping (clear_thinking).
# ---------------------------------------------------------------------------


def _strip_thinking_content(
    messages: list[ModelMessage],
) -> tuple[list[ModelMessage], int]:
    """Remove ``ThinkingPart`` from assistant messages before the last user message.

    Finds the last ``ModelRequest`` containing a ``UserPromptPart`` and
    strips all ``ThinkingPart`` instances from ``ModelResponse`` messages
    before that index.  Messages at or after the last user message are
    left untouched — the most recent thinking remains available for
    reasoning continuity.

    This is re-applied on every ``before_model_request`` iteration because
    ``ctx.state.message_history`` restores original content each cycle
    (same ephemeral pattern as ``_re_prune_messages``).

    Args:
        messages: The conversation message list (never mutated).

    Returns:
        A tuple of ``(new_messages, stripped_count)`` where
        ``new_messages`` is a new list with thinking parts removed and
        ``stripped_count`` is the number of ``ThinkingPart`` instances
        removed.
    """
    # Find the last user message index.
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    last_user_idx = i
                    break
            if last_user_idx >= 0:
                break

    if last_user_idx < 0:
        # No user message found — nothing to strip.
        return messages, 0

    result: list[ModelMessage] = []
    stripped_count = 0

    for i, msg in enumerate(messages):
        if i < last_user_idx and isinstance(msg, ModelResponse):
            new_parts = []
            modified = False
            for part in msg.parts:  # type: ignore[assignment]
                if isinstance(part, ThinkingPart):
                    stripped_count += 1
                    modified = True
                    continue
                new_parts.append(part)
            if modified:
                result.append(replace(msg, parts=new_parts))  # type: ignore[arg-type]
            else:
                result.append(msg)
        else:
            result.append(msg)

    return result, stripped_count


# ---------------------------------------------------------------------------
# purge_failed_tool_inputs — inlined from DCP strategies/purge_errors.py.
#
# Prunes tool OUTPUTS (ToolReturnPart.content) of failed tools after
# sufficient iterations have elapsed.  Protection is iteration-based:
# the last N tool-call-containing messages (counted from the end) are
# protected from purging.
# ---------------------------------------------------------------------------


def _has_tool_call(msg: ModelMessage) -> bool:
    """Check if a message contains any tool-related part."""
    if not isinstance(msg, (ModelRequest, ModelResponse)):
        return False
    return any(isinstance(p, (ToolCallPart, ToolReturnPart, RetryPromptPart)) for p in msg.parts)


def purge_failed_tool_inputs(
    messages: list[ModelMessage],
    state: PrunableState,
    *,
    purge_error_iterations: int,
    iteration_protection: int,
    protected_tools: set[str],
) -> None:
    """Mark failed tool call outputs for pruning after ``purge_error_iterations``.

    Scans messages for error indicators (``RetryPromptPart`` and
    ``ToolReturnPart`` with ``outcome != 'success'``), then marks the
    corresponding tool call's ``tool_call_id`` for pruning once
    ``purge_error_iterations`` tool-call iterations have elapsed since
    the error.

    The ``RetryPromptPart`` error message itself is preserved — only the
    tool call OUTPUT (``ToolReturnPart.content``) is marked for pruning
    (the caller replaces it with a placeholder via ``_apply_pruned_tools``).

    Recent errors within ``iteration_protection`` tool-call iterations
    from the end are retained.

    Tools listed in ``protected_tools`` are exempt from pruning.

    Args:
        messages: The full conversation message history.
        state: The prunable state to modify (adds to ``pruned_tools``).
        purge_error_iterations: Number of tool-call iterations after which
            failed tool outputs become eligible for pruning.
        iteration_protection: Number of recent tool-call iterations to
            protect from pruning.
        protected_tools: Set of tool names exempt from pruning.
    """
    if not messages:
        return

    # Build a list of (message_index, tool_call_id, tool_name) for error tools,
    # and count tool-call iterations from the end for protection.
    error_entries: list[tuple[int, str, str]] = []

    # Count tool-call-containing messages to determine protection window.
    tool_call_iteration_at: list[int | None] = [None] * len(messages)
    iteration_counter = 0
    for i, msg in enumerate(messages):
        if _has_tool_call(msg):
            iteration_counter += 1
        tool_call_iteration_at[i] = iteration_counter

    total_iterations = iteration_counter

    # Collect error tool_call_ids with their message index and tool_name.
    for idx, msg in enumerate(messages):
        if not isinstance(msg, (ModelRequest, ModelResponse)):
            continue
        for part in msg.parts:
            if isinstance(part, RetryPromptPart):
                tool_call_id = part.tool_call_id
                if tool_call_id:
                    error_entries.append((idx, tool_call_id, part.tool_name or ""))
            elif isinstance(part, ToolReturnPart):
                if part.outcome != "success":
                    tool_call_id = part.tool_call_id
                    if tool_call_id:
                        error_entries.append((idx, tool_call_id, part.tool_name))

    if not error_entries:
        return

    # Mark for pruning if enough iterations have elapsed.
    for msg_idx, tool_call_id, tool_name in error_entries:
        if tool_name in protected_tools:
            continue
        msg_iteration = tool_call_iteration_at[msg_idx]
        if msg_iteration is None:
            continue
        iterations_elapsed = total_iterations - msg_iteration
        # Skip if within protection window
        if iterations_elapsed < iteration_protection:
            continue
        if iterations_elapsed >= purge_error_iterations:
            state.pruned_tools.add(tool_call_id)
