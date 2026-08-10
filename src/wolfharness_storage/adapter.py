"""Adapter wrapping legacy ``StorageProvider`` to satisfy all 7 ISP Protocols.

``StorageProviderAdapter`` delegates every method to the wrapped
``StorageProvider`` instance, making it pass ``isinstance`` checks for all
seven Protocols defined in ``wolfharness_storage.protocols``.

This enables gradual migration: consumers depend on the narrow Protocol
they need, while the underlying storage remains a single ``StorageProvider``
subclass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from wolfharness.common_types import JsonValue
    from wolfharness.messaging import ChatMessage, TokenCost
    from wolfharness.sessions.models import ProjectData, SessionData
    from wolfharness_config.session import SessionQuery
    from wolfharness_storage.base import StorageProvider
    from wolfharness_storage.models import ConversationData, QueryFilters, StatsFilters


class StorageProviderAdapter:
    """Wraps a ``StorageProvider`` to satisfy all 7 ISP Protocols.

    Each method delegates directly to the wrapped provider.
    Passes ``isinstance`` checks for ``SessionPersistence``,
    ``MessagePersistence``, ``SessionMetadata``, ``CommandLog``,
    ``ProjectStoreProtocol``, ``CheckpointStore``, and ``StatsAggregator``.
    """

    def __init__(self, provider: StorageProvider) -> None:
        self._provider = provider

    # -- SessionPersistence ----------------------------------------------

    async def save_session(self, data: SessionData) -> None:
        await self._provider.save_session(data)

    async def load_session(self, session_id: str) -> SessionData | None:
        return await self._provider.load_session(session_id)

    async def delete_session(self, session_id: str) -> bool:
        return await self._provider.delete_session(session_id)

    async def list_session_ids(
        self,
        *,
        pool_id: str | None = None,
        agent_name: str | None = None,
        cwd: str | None = None,
    ) -> list[str]:
        return await self._provider.list_session_ids(
            pool_id=pool_id, agent_name=agent_name, cwd=cwd
        )

    async def load_sessions_batch(
        self,
        session_ids: list[str],
        *,
        agent_name: str | None = None,
    ) -> list[SessionData]:
        return await self._provider.load_sessions_batch(session_ids, agent_name=agent_name)

    async def update_sdk_session_id(
        self,
        session_id: str,
        sdk_session_id: str,
    ) -> None:
        await self._provider.update_sdk_session_id(session_id, sdk_session_id)

    # -- MessagePersistence ----------------------------------------------

    async def log_message(self, *, message: ChatMessage[Any]) -> None:
        await self._provider.log_message(message=message)

    async def get_session_messages(
        self,
        session_id: str,
        *,
        include_ancestors: bool = False,
    ) -> list[ChatMessage[Any]]:
        return await self._provider.get_session_messages(
            session_id, include_ancestors=include_ancestors
        )

    async def get_message(
        self,
        message_id: str,
        *,
        session_id: str | None = None,
    ) -> ChatMessage[Any] | None:
        return await self._provider.get_message(message_id, session_id=session_id)

    async def get_message_ancestry(
        self,
        message_id: str,
        *,
        session_id: str | None = None,
    ) -> list[ChatMessage[Any]]:
        return await self._provider.get_message_ancestry(message_id, session_id=session_id)

    async def filter_messages(self, query: SessionQuery) -> list[ChatMessage[Any]]:
        return await self._provider.filter_messages(query)

    async def delete_session_messages(self, session_id: str) -> int:
        return await self._provider.delete_session_messages(session_id)

    async def truncate_messages(
        self,
        session_id: str,
        up_to_message_id: str,
    ) -> int:
        return await self._provider.truncate_messages(session_id, up_to_message_id)

    async def fork_conversation(
        self,
        *,
        source_session_id: str,
        new_session_id: str,
        fork_from_message_id: str | None = None,
        new_agent_name: str | None = None,
    ) -> str | None:
        return await self._provider.fork_conversation(
            source_session_id=source_session_id,
            new_session_id=new_session_id,
            fork_from_message_id=fork_from_message_id,
            new_agent_name=new_agent_name,
        )

    # -- SessionMetadata -------------------------------------------------

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
        await self._provider.log_session(
            session_id=session_id,
            node_name=node_name,
            start_time=start_time,
            model=model,
            agent_type=agent_type,
            parent_session_id=parent_session_id,
        )

    async def update_session_title(self, session_id: str, title: str) -> None:
        await self._provider.update_session_title(session_id, title)

    async def get_session_title(self, session_id: str) -> str | None:
        return await self._provider.get_session_title(session_id)

    async def get_sessions(self, filters: QueryFilters) -> list[ConversationData]:
        return await self._provider.get_sessions(filters)

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
        return await self._provider.get_filtered_conversations(
            agent_name=agent_name,
            period=period,
            since=since,
            query=query,
            model=model,
            limit=limit,
            compact=compact,
            include_tokens=include_tokens,
        )

    async def get_session_counts(
        self,
        *,
        agent_name: str | None = None,
    ) -> tuple[int, int]:
        return await self._provider.get_session_counts(agent_name=agent_name)

    async def get_session_stats(self, filters: StatsFilters) -> dict[str, dict[str, Any]]:
        return await self._provider.get_session_stats(filters)

    # -- CommandLog ------------------------------------------------------

    async def log_command(
        self,
        *,
        agent_name: str,
        session_id: str,
        command: str,
        context_type: type | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> None:
        await self._provider.log_command(
            agent_name=agent_name,
            session_id=session_id,
            command=command,
            context_type=context_type,
            metadata=metadata,
        )

    async def get_commands(
        self,
        agent_name: str,
        session_id: str,
        *,
        limit: int | None = None,
        current_session_only: bool = False,
    ) -> list[str]:
        return await self._provider.get_commands(
            agent_name,
            session_id,
            limit=limit,
            current_session_only=current_session_only,
        )

    # -- ProjectStoreProtocol --------------------------------------------

    async def save_project(self, project: ProjectData) -> None:
        await self._provider.save_project(project)

    async def get_project(self, project_id: str) -> ProjectData | None:
        return await self._provider.get_project(project_id)

    async def get_project_by_name(self, name: str) -> ProjectData | None:
        return await self._provider.get_project_by_name(name)

    async def get_project_by_worktree(self, worktree: str) -> ProjectData | None:
        return await self._provider.get_project_by_worktree(worktree)

    async def list_projects(self, limit: int | None = None) -> list[ProjectData]:
        return await self._provider.list_projects(limit)

    async def delete_project(self, project_id: str) -> bool:
        return await self._provider.delete_project(project_id)

    async def touch_project(self, project_id: str) -> None:
        await self._provider.touch_project(project_id)

    # -- CheckpointStore -------------------------------------------------

    async def save_checkpoint(
        self,
        session_id: str,
        messages_json: str,
        pending_calls_json: str,
    ) -> None:
        await self._provider.save_checkpoint(session_id, messages_json, pending_calls_json)

    async def load_checkpoint(self, session_id: str) -> tuple[str, str] | None:
        return await self._provider.load_checkpoint(session_id)

    async def delete_checkpoint(self, session_id: str) -> bool:
        return await self._provider.delete_checkpoint(session_id)

    # -- StatsAggregator -------------------------------------------------

    def aggregate_stats(
        self,
        rows: Sequence[tuple[str | None, str | None, datetime, TokenCost | None]],
        group_by: Literal["agent", "model", "hour", "day"],
    ) -> dict[str, dict[str, Any]]:
        return self._provider.aggregate_stats(rows, group_by)

    async def reset(
        self,
        *,
        agent_name: str | None = None,
        hard: bool = False,
    ) -> tuple[int, int]:
        return await self._provider.reset(agent_name=agent_name, hard=hard)
