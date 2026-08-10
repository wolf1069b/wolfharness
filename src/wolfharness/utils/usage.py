"""Usage diff helper for per-step token tracking.

Provides ``diff_usage()`` to compute the per-step delta between two
``RunUsage`` snapshots taken before and after ``agent_run.next(node)``
in ``NativeTurn.execute()``.
"""

from __future__ import annotations

from pydantic_ai import RunUsage


# Fields to diff between two RunUsage snapshots.
# Excludes computed properties (total_tokens, cache_hit_ratio, has_values,
# opentelemetry_attributes) and methods (incr).
_USAGE_FIELDS: tuple[str, ...] = (
    "input_tokens",
    "cache_write_tokens",
    "cache_read_tokens",
    "output_tokens",
    "input_audio_tokens",
    "cache_audio_read_tokens",
    "output_audio_tokens",
    "requests",
    "tool_calls",
)


def diff_usage(curr: RunUsage, prev: RunUsage) -> RunUsage:
    """Compute the field-by-field difference between two ``RunUsage`` snapshots.

    Returns a new ``RunUsage`` where each token field is ``curr - prev``.
    For the ``details: dict[str, int]`` field: for each key in
    ``curr.details``, compute ``result[key] = curr.details[key] -
    prev.details.get(key, 0)``.  Keys only in ``curr`` are included
    as-is; keys only in ``prev`` are dropped.

    The helper SHALL NOT modify either input argument.

    Args:
        curr: Current (newer) RunUsage snapshot.
        prev: Previous (older) RunUsage snapshot.

    Returns:
        A new ``RunUsage`` representing the delta.
    """
    kwargs: dict[str, int | dict[str, int]] = {}
    for field_name in _USAGE_FIELDS:
        kwargs[field_name] = getattr(curr, field_name) - getattr(prev, field_name)

    # Diff the details dict
    details_diff: dict[str, int] = {}
    for key, value in curr.details.items():
        details_diff[key] = value - prev.details.get(key, 0)

    kwargs["details"] = details_diff
    return RunUsage(**kwargs)  # type: ignore[arg-type]
