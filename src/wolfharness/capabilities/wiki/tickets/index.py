"""Build-root index injection helpers for WikiBuildCapability.

Provides a pure-function helper for formatting the config-resolved wiki
build roots as an ``<openviking-index>`` XML block — used by
``WikiBuildCapability.before_model_request()``. This is B-scheme: roots
come from configuration (wiki/raw/bom), not a live-server enumeration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wolfharness.capabilities.viking.utils import truncate_text


if TYPE_CHECKING:
    from collections.abc import Sequence


_INDEX_BLOCK = (
    "<openviking-index>\n"
    "Wiki resources are addressed by viking:// URIs. Active build roots "
    "for this session:\n"
    "{roots}\n"
    "Use list_children / get_resource / read_resource to navigate and "
    "never guess or splice namespaces.\n"
    "</openviking-index>"
)


def _format_index_block(
    roots: Sequence[tuple[str, str]],
    *,
    max_tokens: int,
    limit: int,
) -> str:
    """Format config-resolved build roots as an ``<openviking-index>`` block.

    Renders the first ``limit`` ``(label, uri)`` pairs as aligned
    ``- label:  uri`` lines, then budget-scales the whole block to
    approximately ``max_tokens`` characters (chars-to-tokens 4:1
    heuristic, same as Viking's index/profile blocks).

    Args:
        roots: ``(label, uri)`` pairs for each resolved build root.
        max_tokens: Maximum token budget — content is truncated to
            ``max_tokens * 4`` characters with a truncation indicator.
        limit: Maximum number of root pairs to include.

    Returns:
        A formatted ``<openviking-index>`` XML block string. Returns an
        empty string if there are no roots.
    """
    truncated = roots[:limit]
    if not truncated:
        return ""

    root_lines = "\n".join(f"- {label}:  {uri}" for label, uri in truncated)
    max_chars = max_tokens * 4  # rough chars-to-tokens heuristic
    return truncate_text(_INDEX_BLOCK.format(roots=root_lines), max_chars)
