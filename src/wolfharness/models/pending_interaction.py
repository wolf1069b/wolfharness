"""Generic pending interaction types for agent sessions.

These Protocol types define the interface for pending questions and permissions
across all agent types and protocols.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from datetime import datetime


class PendingQuestion(Protocol):
    """A pending question waiting for user response.

    This Protocol defines the common interface for pending questions
    across all protocols (OpenCode, ACP, AG-UI, etc.).
    """

    id: str
    """Unique identifier for this pending question."""

    session_id: str
    """The session this question belongs to."""

    tool_name: str
    """The name of the tool that generated this question."""

    content: str
    """The question content/prompt."""

    created_at: datetime
    """When this question was created."""


class PendingPermission(Protocol):
    """A pending permission request waiting for user approval.

    This Protocol defines the common interface for pending permissions
    across all protocols (OpenCode, ACP, AG-UI, etc.).
    """

    id: str
    """Unique identifier for this pending permission."""

    session_id: str
    """The session this permission belongs to."""

    tool_name: str
    """The name of the tool requesting permission."""

    content: str
    """Description of what permission is being requested."""

    created_at: datetime
    """When this permission request was created."""
