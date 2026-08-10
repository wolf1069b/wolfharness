"""TUI event routing filter — diagnostic re-implementation of OpenCode TUI routing.

Re-implements the 3-rule event routing filter from the OpenCode TUI's
event.ts as a pure Python function for diagnostic purposes. This allows
debugging why events may be dropped by the TUI's client-side filter.

The 3 rules (evaluated in order):
1. **Sync events always dropped**: if `payload.type == "sync"` → False
2. **Global directory always passes**: if `directory == "global"` → True
3. **Directory must match exactly**: `event.directory == project_directory`
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from wolfharness_server.opencode_server.models.base import OpenCodeBaseModel


if TYPE_CHECKING:
    from wolfharness_server.opencode_server.models.events import GlobalEvent


RoutingReason = Literal[
    "sync_dropped",
    "global_directory",
    "directory_match",
    "directory_mismatch",
]
"""Reason strings explaining why an event passes or fails the routing filter."""


class RoutingCheckResponse(OpenCodeBaseModel):
    """Response for GET /global/routing-check endpoint."""

    would_pass: bool
    """Whether the event would pass the TUI routing filter."""

    reason: RoutingReason
    """Explanation for the routing decision."""


def tui_event_filter(
    event: GlobalEvent,
    project_directory: str,
) -> tuple[bool, RoutingReason]:
    """Re-implements OpenCode TUI event routing filter for diagnostic purposes.

    Evaluates the 3-rule filter in priority order and returns both
    the pass/fail result and the reason for the decision.

    Args:
        event: The GlobalEvent to check against the routing filter.
        project_directory: The server's working directory for directory matching.

    Returns:
        A tuple of (would_pass, reason) where reason explains which rule
        determined the outcome.
    """
    # Rule 1: sync events always dropped
    if event.payload.get("type") == "sync":
        return (False, "sync_dropped")

    # Rule 2: global directory always passes (except sync, handled above)
    if event.directory == "global":
        return (True, "global_directory")

    # Rule 3: directory must match exactly (string comparison, no normalization)
    if event.directory == project_directory:
        return (True, "directory_match")
    return (False, "directory_mismatch")
