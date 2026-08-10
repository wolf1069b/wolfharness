"""OpenCode-specific pending question and permission types.

Concrete dataclasses implementing the generic PendingQuestion and
PendingPermission Protocols for the OpenCode protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class OpenCodePendingQuestion:
    """A pending question in the OpenCode protocol.

    Implements the PendingQuestion Protocol with OpenCode-specific
    serialization compatibility.
    """

    id: str
    """Unique identifier for this pending question."""

    session_id: str
    """The session this question belongs to."""

    tool_name: str
    """The name of the tool that generated this question."""

    content: str
    """The question content/prompt."""

    created_at: datetime = field(default_factory=datetime.utcnow)
    """When this question was created."""


@dataclass
class OpenCodePendingPermission:
    """A pending permission request in the OpenCode protocol.

    Implements the PendingPermission Protocol with OpenCode-specific
    serialization compatibility.
    """

    id: str
    """Unique identifier for this pending permission."""

    session_id: str
    """The session this permission belongs to."""

    tool_name: str
    """The name of the tool requesting permission."""

    content: str
    """Description of what permission is being requested."""

    created_at: datetime = field(default_factory=datetime.utcnow)
    """When this permission request was created."""
