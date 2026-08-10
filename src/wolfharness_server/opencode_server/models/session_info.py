"""SessionInfo DTO for listing sessions via SessionController."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SessionInfo:
    """Session information DTO for listing sessions.

    Attributes:
        session_id: Unique identifier for the session.
        agent_name: Name of the agent associated with this session.
        created_at: Timestamp when the session was created (monotonic).
        last_active_at: Timestamp of the most recent activity (monotonic).
        is_per_session_agent: Whether the agent is dedicated to this session.
        status: Current session status ("idle" or "busy").
    """

    session_id: str
    agent_name: str
    created_at: float
    last_active_at: float
    is_per_session_agent: bool
    status: str
