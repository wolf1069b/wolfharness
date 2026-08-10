"""ISP Protocols for storage provider decomposition.

Decomposes the monolithic ``StorageProvider`` god class (43 methods, 7 concerns)
into 7 focused Protocols following the Interface Segregation Principle.

Each Protocol covers exactly one concern:

- ``SessionPersistence`` — CRUD for session data
- ``MessagePersistence`` — message log + ancestry + fork
- ``SessionMetadata`` — session metadata, titles, counts, stats
- ``CommandLog`` — command history logging
- ``ProjectStore`` — project CRUD
- ``CheckpointStore`` — checkpoint save/load/delete
- ``StatsAggregator`` — stats aggregation + reset

Consumers should depend on the narrowest Protocol they need.
``StorageProviderAdapter`` wraps a legacy ``StorageProvider`` and satisfies all 7.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from wolfharness.common_types import JsonValue
    from wolfharness.messaging import ChatMessage, TokenCost
    from wolfharness.sessions.models import ProjectData, SessionData
    from wolfharness_config.session import SessionQuery
    from wolfharness_storage.models import ConversationData, QueryFilters, StatsFilters


# ---------------------------------------------------------------------------
# 1. Session Persistence
# ---------------------------------------------------------------------------


@runtime_checkable
class SessionPersistence(Protocol):
    """Protocol for session CRUD persistence."""

    async def save_session(self, data: SessionData) -> None:
        """Save or update session data."""
        ...

    async def load_session(self, session_id: str) -> SessionData | None:
        """Load session data by ID."""
        ...

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if deleted, False if not found."""
        ...

    async def list_session_ids(
        self,
        *,
        pool_id: str | None = None,
        agent_name: str | None = None,
        cwd: str | None = None,
    ) -> list[str]:
        """List session IDs, optionally filtered."""
        ...

    async def load_sessions_batch(
        self,
        session_ids: list[str],
        *,
        agent_name: str | None = None,
    ) -> list[SessionData]:
        """Load multiple sessions by IDs in a single query."""
        ...

    async def update_sdk_session_id(
        self,
        session_id: str,
        sdk_session_id: str,
    ) -> None:
        """Update the external SDK session ID for a session."""
        ...


# ---------------------------------------------------------------------------
# 2. Message Persistence
# ---------------------------------------------------------------------------


@runtime_checkable
class MessagePersistence(Protocol):
    """Protocol for message logging and retrieval."""

    async def log_message(self, *, message: ChatMessage[Any]) -> None:
        """Log a message."""
        ...

    async def get_session_messages(
        self,
        session_id: str,
        *,
        include_ancestors: bool = False,
    ) -> list[ChatMessage[Any]]:
        """Get all messages for a session."""
        ...

    async def get_message(
        self,
        message_id: str,
        *,
        session_id: str | None = None,
    ) -> ChatMessage[Any] | None:
        """Get a single message by ID."""
        ...

    async def get_message_ancestry(
        self,
        message_id: str,
        *,
        session_id: str | None = None,
    ) -> list[ChatMessage[Any]]:
        """Get the ancestry chain of a message."""
        ...

    async def filter_messages(self, query: SessionQuery) -> list[ChatMessage[Any]]:
        """Get messages matching query."""
        ...

    async def delete_session_messages(self, session_id: str) -> int:
        """Delete all messages for a session. Returns count deleted."""
        ...

    async def truncate_messages(
        self,
        session_id: str,
        up_to_message_id: str,
    ) -> int:
        """Delete the message with up_to_message_id and all after it. Returns count removed."""
        ...

    async def fork_conversation(
        self,
        *,
        source_session_id: str,
        new_session_id: str,
        fork_from_message_id: str | None = None,
        new_agent_name: str | None = None,
    ) -> str | None:
        """Fork a conversation at a specific point."""
        ...


# ---------------------------------------------------------------------------
# 3. Session Metadata
# ---------------------------------------------------------------------------


@runtime_checkable
class SessionMetadata(Protocol):
    """Protocol for session metadata, titles, and statistics queries."""

    async def log_session(
        self,
        *,
        session_id: str,
        node_name: str,
        start_time: datetime | None = None,
        model: str | None = None,
        agent_type: str | None = None,
        parent_session_id: str | None = None,
    ) -> None:
        """Log a conversation/session."""
        ...

    async def update_session_title(self, session_id: str, title: str) -> None:
        """Update the title of a conversation."""
        ...

    async def get_session_title(self, session_id: str) -> str | None:
        """Get the title of a conversation."""
        ...

    async def get_sessions(self, filters: QueryFilters) -> list[ConversationData]:
        """Get filtered conversations with their messages."""
        ...

    async def get_filtered_conversations(
        self,
        agent_name: str | None = None,
        period: str | None = None,
        since: datetime | None = None,
        query: str | None = None,
        model: str | None = None,
        limit: int | None = None,
        *,
        compact: bool = False,
        include_tokens: bool = False,
    ) -> list[ConversationData]:
        """Get filtered conversations with formatted output."""
        ...

    async def get_session_counts(
        self,
        *,
        agent_name: str | None = None,
    ) -> tuple[int, int]:
        """Get counts of conversations and messages."""
        ...

    async def get_session_stats(self, filters: StatsFilters) -> dict[str, dict[str, Any]]:
        """Get conversation statistics grouped by specified criterion."""
        ...


# ---------------------------------------------------------------------------
# 4. Command Log
# ---------------------------------------------------------------------------


@runtime_checkable
class CommandLog(Protocol):
    """Protocol for command history logging."""

    async def log_command(
        self,
        *,
        agent_name: str,
        session_id: str,
        command: str,
        context_type: type | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> None:
        """Log a command."""
        ...

    async def get_commands(
        self,
        agent_name: str,
        session_id: str,
        *,
        limit: int | None = None,
        current_session_only: bool = False,
    ) -> list[str]:
        """Get command history."""
        ...


# ---------------------------------------------------------------------------
# 5. Project Store
# ---------------------------------------------------------------------------


@runtime_checkable
class ProjectStoreProtocol(Protocol):
    """Protocol for project CRUD operations.

    Named ``ProjectStoreProtocol`` to avoid clash with the existing
    ``ProjectStore`` class in ``wolfharness_storage.project_store``.
    """

    async def save_project(self, project: ProjectData) -> None:
        """Save or update a project."""
        ...

    async def get_project(self, project_id: str) -> ProjectData | None:
        """Get a project by ID."""
        ...

    async def get_project_by_name(self, name: str) -> ProjectData | None:
        """Get a project by friendly name."""
        ...

    async def get_project_by_worktree(self, worktree: str) -> ProjectData | None:
        """Get a project by worktree path."""
        ...

    async def list_projects(self, limit: int | None = None) -> list[ProjectData]:
        """List all projects, ordered by last_active descending."""
        ...

    async def delete_project(self, project_id: str) -> bool:
        """Delete a project. Returns True if deleted, False if not found."""
        ...

    async def touch_project(self, project_id: str) -> None:
        """Update project's last_active timestamp."""
        ...


# ---------------------------------------------------------------------------
# 6. Checkpoint Store
# ---------------------------------------------------------------------------


@runtime_checkable
class CheckpointStore(Protocol):
    """Protocol for checkpoint persistence."""

    async def save_checkpoint(
        self,
        session_id: str,
        messages_json: str,
        pending_calls_json: str,
    ) -> None:
        """Save checkpoint data atomically."""
        ...

    async def load_checkpoint(self, session_id: str) -> tuple[str, str] | None:
        """Load checkpoint data. Returns (messages_json, pending_calls_json) or None."""
        ...

    async def delete_checkpoint(self, session_id: str) -> bool:
        """Delete checkpoint data. Returns True if deleted, False if not found."""
        ...


# ---------------------------------------------------------------------------
# 7. Stats Aggregator
# ---------------------------------------------------------------------------


@runtime_checkable
class StatsAggregator(Protocol):
    """Protocol for stats aggregation and reset."""

    def aggregate_stats(
        self,
        rows: Sequence[tuple[str | None, str | None, datetime, TokenCost | None]],
        group_by: Literal["agent", "model", "hour", "day"],
    ) -> dict[str, dict[str, Any]]:
        """Aggregate statistics data by specified grouping."""
        ...

    async def reset(
        self,
        *,
        agent_name: str | None = None,
        hard: bool = False,
    ) -> tuple[int, int]:
        """Reset storage. Returns (conversations deleted, messages deleted)."""
        ...
