"""Dynamic resource-namespace index injection helpers for VikingCapability.

Provides a pure-function helper for formatting live resource namespaces
as an ``<openviking-index>`` XML block — used by
``VikingCapability._handle_index_inject()``. This is the dynamic
counterpart to the static ``_VIKING_INSTRUCTIONS``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wolfharness.capabilities.viking.utils import truncate_text


if TYPE_CHECKING:
    from collections.abc import Sequence


_INDEX_BLOCK = (
    "<openviking-index>\n"
    "Viking resources are addressed by viking:// URIs. Live resource "
    "namespaces under viking://resources/:\n"
    "{namespaces}\n"
    'Use viking_ls("<uri>") to enumerate deeper, viking_search/viking_find '
    "to locate content, viking_read to fetch.\n"
    "</openviking-index>"
)


def _format_index_block(
    namespaces: Sequence[str],
    *,
    max_tokens: int,
    limit: int,
) -> str:
    """Format live resource namespaces as an ``<openviking-index>`` XML block.

    Joins at most ``limit`` namespace names with ``", "`` and renders the
    block (contract consumed by external prompts). The total block is
    truncated to approximately ``max_tokens`` characters (using chars as
    a proxy with a 4:1 heuristic, same as ``_format_profile_block``).

    Args:
        namespaces: Discovered resource namespace names.
        max_tokens: Maximum token budget — content is truncated to
            ``max_tokens * 4`` characters with a truncation indicator.
        limit: Maximum number of namespace names to include.

    Returns:
        A formatted ``<openviking-index>`` XML block string. Returns an
        empty string if there are no namespaces.
    """
    truncated = namespaces[:limit]
    if not truncated:
        return ""

    max_chars = max_tokens * 4  # rough chars-to-tokens heuristic
    block = _INDEX_BLOCK.format(namespaces=", ".join(truncated))
    return truncate_text(block, max_chars)
