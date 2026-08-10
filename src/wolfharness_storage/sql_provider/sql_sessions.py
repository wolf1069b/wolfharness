"""Session persistence mixin for SQLModelProvider.

Extracted from sql_provider.py as part of the session-debt-cleanup file split.
Contains session metadata, session persistence, checkpoint, command log, and stats methods.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from pydantic_ai.usage import RunUsage
from sqlalchemy import insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel, desc, select

from wolfharness.log import get_logger
from wolfharness.messaging import TokenCost
from wolfharness.utils.parse_time import parse_time_period
from wolfharness.utils.time_utils import get_now
from wolfharness_storage.models import QueryFilters
from wolfharness_storage.sql_provider.models import (
    CommandHistory,
    Conversation,
    Message,
)
from wolfharness_storage.sql_provider.utils import (
    format_conversation,
    to_chat_message,
)


try:
    from sqlalchemy.dialects.postgresql import insert as pg_insert
except ImportError:
    pg_insert = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
try:
    from sqlalchemy.dialects.mysql import insert as mysql_insert
except ImportError:
    mysql_insert = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]


if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncEngine

    from wolfharness.common_types import JsonValue
    from wolfharness.sessions.models import SessionData
    from wolfharness_storage.models import ConversationData, StatsFilters


logger = get_logger(__name__)


class SQLSessionsMixin:
    """Mixin providing session persistence, metadata, checkpoint, and stats methods.

    Attributes:
        engine: Async database engine (provided by SQLModelProvider).
        session: Active async session (provided by SQLModelProvider).
    """

    engine: AsyncEngine
    session: AsyncSession | None

    if TYPE_CHECKING:

        def aggregate_stats(
            self,
            rows: Any,
            group_by: str,
        ) -> dict[str, dict[str, Any]]: ...

        async def _init_database(self, auto_migrate: bool = True) -> None: ...

    def _get_insert_stmt(self) -> Any:
        """Get appropriate insert statement for database dialect.

        Invariant (PR #10): branch on ``engine.dialect.name`` only. Do not prefer
        ``pg_insert`` merely because psycopg is installed while connected to MySQL.

        Returns:
            SQLAlchemy insert statement with dialect-specific conflict handling support.
        """
        dialect_name = self.engine.dialect.name

        if dialect_name == "sqlite":
            return sqlite_insert(Conversation)
        if dialect_name == "postgresql" and pg_insert is not None:
            return pg_insert(Conversation)
        if dialect_name in ("mysql", "mariadb") and mysql_insert is not None:
            return mysql_insert(Conversation)
        # Generic fallback (or dialect without dialect-specific insert helper)
        return insert(Conversation)

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
        """Log conversation to database.

        ``parent_session_id`` maps to ``Conversation.parent_id`` and the
        ``conversation.parent_id`` column (RFC-0011; migration
        ``2f5ee67f43ce_add_parent_id_to_conversation``). The ORM model must keep this
        field in sync with the INSERT ``values()`` call below.

        Uses upsert semantics to handle duplicate session IDs gracefully.
        If the session already exists, it will be silently ignored.
        """
        from wolfharness_storage.sql_provider.models import Conversation

        async with AsyncSession(self.engine) as session:
            # Soft validation: check if parent exists (warn but don't crash)
            if parent_session_id:
                result = await session.execute(
                    select(Conversation).where(Conversation.id == parent_session_id)
                )
                if not result.scalar_one_or_none():
                    logger.warning(
                        "Parent session not found",
                        parent_session_id=parent_session_id,
                        session_id=session_id,
                    )

            now = start_time or get_now()

            # Conversation.parent_id (models.Conversation) stores parent_session_id for hierarchy.
            # Use dialect-specific upsert to avoid UNIQUE constraint violations
            stmt = self._get_insert_stmt().values(
                id=session_id,
                agent_name=node_name,
                parent_id=parent_session_id,
                title=None,
                start_time=now,
                model=model,
            )

            # Apply dialect-specific "insert or ignore duplicate PK" semantics
            dialect_name = self.engine.dialect.name
            if dialect_name in ("sqlite", "postgresql") and hasattr(stmt, "on_conflict_do_nothing"):
                # Conversation.id is the primary key (indexed); required for ON CONFLICT target.
                stmt = stmt.on_conflict_do_nothing(index_elements=["id"])
            elif dialect_name in ("mysql", "mariadb") and hasattr(stmt, "on_duplicate_key_update"):
                # MySQL/MariaDB have no on_conflict_do_nothing; update PK to itself is a no-op
                stmt = stmt.on_duplicate_key_update(id=stmt.inserted.id)

            await session.execute(stmt)
            await session.commit()

    async def update_session_title(self, session_id: str, title: str) -> None:
        """Update the title of a conversation.

        Only writes the ``title`` column.  ``_session_from_db`` always
        syncs ``row.title`` → ``metadata["title"]`` on read, so this
        single write is sufficient.
        """
        async with AsyncSession(self.engine) as session:
            result = await session.execute(
                select(Conversation).where(Conversation.id == session_id)
            )
            conversation = result.scalar_one_or_none()
            if conversation:
                conversation.title = title
                session.add(conversation)
                await session.commit()

    async def get_session_title(self, session_id: str) -> str | None:
        """Get the title of a conversation."""
        async with AsyncSession(self.engine) as session:
            result = await session.execute(
                select(Conversation.title).where(Conversation.id == session_id)
            )
            return result.scalar_one_or_none()

    async def log_command(
        self,
        *,
        agent_name: str,
        session_id: str,
        command: str,
        context_type: type | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> None:
        """Log command to database."""
        async with AsyncSession(self.engine) as session:
            history = CommandHistory(
                session_id=session_id,
                agent_name=agent_name,
                command=command,
                context_type=context_type.__name__ if context_type else None,
                context_metadata=metadata or {},
            )
            session.add(history)
            await session.commit()

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
        # Convert period to since if provided
        if period:
            since = get_now() - parse_time_period(period)

        # Create filters
        filters = QueryFilters(
            agent_name=agent_name,
            since=since,
            query=query,
            model=model,
            limit=limit,
        )

        # Use existing get_sessions method
        return [
            format_conversation(i, i["messages"], compact=compact, include_tokens=include_tokens)
            for i in await self.get_sessions(filters)
        ]

    async def get_commands(
        self,
        agent_name: str,
        session_id: str,
        *,
        limit: int | None = None,
        current_session_only: bool = False,
    ) -> list[str]:
        """Get command history from database."""
        async with AsyncSession(self.engine) as session:
            query = select(CommandHistory)
            if current_session_only:
                query = query.where(CommandHistory.session_id == str(session_id))
            else:
                query = query.where(CommandHistory.agent_name == agent_name)

            query = query.order_by(desc(CommandHistory.timestamp))
            if limit:
                query = query.limit(limit)

            result = await session.execute(query)
            return [h.command for h in result.scalars()]

    async def get_sessions(self, filters: QueryFilters) -> list[ConversationData]:
        """Get filtered conversations using SQL queries."""
        async with AsyncSession(self.engine) as session:
            results: list[ConversationData] = []
            # Base conversation query
            conv_query = select(Conversation)
            if filters.agent_name:
                conv_query = conv_query.where(Conversation.agent_name == filters.agent_name)
            # Apply time filters if provided
            if filters.since:
                conv_query = conv_query.where(Conversation.start_time >= filters.since)
            if filters.limit:
                conv_query = conv_query.limit(filters.limit)
            conv_result = await session.execute(conv_query)
            for conv in conv_result.scalars().all():
                # Get messages for this conversation
                msg_query = select(Message).where(Message.session_id == conv.id)

                if filters.query:
                    msg_query = msg_query.where(Message.content.contains(filters.query))  # type: ignore
                if filters.model:
                    msg_query = msg_query.where(Message.model_name == filters.model)

                msg_query = msg_query.order_by(Message.timestamp.asc(), Message.id.asc())  # type: ignore
                msg_result = await session.execute(msg_query)
                messages = msg_result.scalars().all()

                if not messages:
                    continue
                chat_msgs = [to_chat_message(msg) for msg in messages]
                results.append(format_conversation(conv, chat_msgs))

            return results

    async def get_session_stats(self, filters: StatsFilters) -> dict[str, dict[str, Any]]:
        """Get statistics using SQL aggregations."""
        from wolfharness_storage.sql_provider.models import Conversation, Message

        async with AsyncSession(self.engine) as session:
            # Base query for stats
            query = (
                select(  # type: ignore[call-overload]  # ty: ignore[no-matching-overload]
                    Message.model,
                    Conversation.agent_name,
                    Message.timestamp,
                    Message.total_tokens,
                    Message.input_tokens,
                    Message.output_tokens,
                )
                .join(Conversation, Message.session_id == Conversation.id)
                .where(Message.timestamp > filters.cutoff)
            )

            if filters.agent_name:
                query = query.where(Conversation.agent_name == filters.agent_name)

            # Execute query and get raw data
            result = await session.execute(query)
            rows = [
                (
                    model,
                    agent,
                    timestamp,
                    TokenCost(
                        token_usage=RunUsage(
                            input_tokens=input_tokens or 0,
                            output_tokens=output_tokens or 0,
                        ),
                        total_cost=Decimal(0),  # We don't store this in DB
                    )
                    if total or input_tokens or output_tokens
                    else None,
                )
                for model, agent, timestamp, total, input_tokens, output_tokens in result.all()
            ]

        # Use base class aggregation
        return self.aggregate_stats(rows, filters.group_by)

    async def reset(self, *, agent_name: str | None = None, hard: bool = False) -> tuple[int, int]:
        """Reset database storage."""
        from sqlalchemy import text

        from wolfharness_storage.sql_provider.queries import (
            DELETE_AGENT_CONVERSATIONS,
            DELETE_AGENT_MESSAGES,
            DELETE_ALL_CONVERSATIONS,
            DELETE_ALL_MESSAGES,
        )

        async with AsyncSession(self.engine) as session:
            if hard:
                if agent_name:
                    msg = "Hard reset cannot be used with agent_name"
                    raise ValueError(msg)
                # Drop and recreate all tables
                async with self.engine.begin() as conn:
                    await conn.run_sync(SQLModel.metadata.drop_all)
                await session.commit()
                # Recreate schema
                await self._init_database()
                return 0, 0

            # Get counts first
            conv_count, msg_count = await self.get_session_counts(agent_name=agent_name)

            # Delete data
            if agent_name:
                await session.execute(text(DELETE_AGENT_MESSAGES), {"agent": agent_name})
                await session.execute(text(DELETE_AGENT_CONVERSATIONS), {"agent": agent_name})
            else:
                await session.execute(text(DELETE_ALL_MESSAGES))
                await session.execute(text(DELETE_ALL_CONVERSATIONS))

            await session.commit()
        return conv_count, msg_count

    async def get_session_counts(self, *, agent_name: str | None = None) -> tuple[int, int]:
        """Get conversation and message counts."""
        from wolfharness_storage.sql_provider import Conversation, Message

        if not self.session:
            msg = "Session not initialized. Use provider as async context manager."
            raise RuntimeError(msg)
        if agent_name:
            conv_query = select(Conversation).where(Conversation.agent_name == agent_name)
            msg_query = (
                select(Message).join(Conversation).where(Conversation.agent_name == agent_name)
            )
        else:
            conv_query = select(Conversation)
            msg_query = select(Message)

        conv_result = await self.session.execute(conv_query)
        msg_result = await self.session.execute(msg_query)
        conv_count = len(conv_result.scalars().all())
        msg_count = len(msg_result.scalars().all())

        return conv_count, msg_count

    def _session_from_db(self, row: Conversation) -> SessionData:
        """Convert database Conversation row to SessionData.

        Mirrors the former ``SQLSessionStore._from_db_model`` logic: merges ``title``
        into ``metadata``, maps ``start_time`` → ``created_at``, reads
        ``status`` from the column, and deserializes
        ``pending_deferred_calls`` from ``metadata_json["_pending_deferred_calls"]``.

        ``Conversation.title`` is the single source of truth — it always
        overrides ``metadata_json["title"]`` so that ``update_session_title``
        (which only writes the column) is sufficient.
        """
        from pydantic import TypeAdapter

        from wolfharness.sessions.models import PendingDeferredCall, SessionData

        metadata = row.metadata_json or {}
        # Conversation.title is authoritative — always sync to metadata
        if row.title:
            metadata = {**metadata, "title": row.title}

        # Deserialize pending_deferred_calls from metadata
        pending_calls: list[PendingDeferredCall] = []
        raw_calls = metadata.pop("_pending_deferred_calls", None)
        if raw_calls:
            calls_adapter = TypeAdapter(list[PendingDeferredCall])
            pending_calls = calls_adapter.validate_python(raw_calls)

        return SessionData(
            session_id=row.id,
            agent_name=row.agent_name,
            pool_id=row.pool_id,
            project_id=row.project_id,
            parent_id=row.parent_id,
            version=row.version,
            cwd=row.cwd,
            agent_type=row.agent_type,
            sdk_session_id=row.sdk_session_id,
            created_at=row.start_time,
            last_active=row.last_active or get_now(),
            metadata=metadata,
            status=row.status or "active",
            pending_deferred_calls=pending_calls,
        )

    async def save_session(self, data: SessionData) -> None:
        """Save or update session data.

        Uses dialect-aware UPSERT to avoid the race condition in
        delete-then-insert. Preserves ``checkpoint_data`` from the
        existing row so that session status updates do not destroy
        checkpoint data saved by ``save_checkpoint()``.

        Serializes ``pending_deferred_calls`` into ``metadata_json``
        under the ``_pending_deferred_calls`` key so they survive the
        roundtrip (same approach as the former ``SQLSessionStore._to_db_model``).
        """
        from pydantic import TypeAdapter

        from wolfharness.sessions.models import PendingDeferredCall

        # Serialize pending_deferred_calls into metadata_json
        metadata = dict(data.metadata)
        if data.pending_deferred_calls:
            calls_adapter = TypeAdapter(list[PendingDeferredCall])
            metadata["_pending_deferred_calls"] = calls_adapter.dump_python(
                data.pending_deferred_calls, mode="json"
            )
        elif "_pending_deferred_calls" in metadata:
            metadata.pop("_pending_deferred_calls", None)

        values: dict[str, Any] = {
            "id": data.session_id,
            "agent_name": data.agent_name,
            "pool_id": data.pool_id,
            "project_id": data.project_id,
            "parent_id": data.parent_id,
            "title": data.title,
            "version": data.version,
            "cwd": data.cwd,
            "agent_type": data.agent_type,
            "sdk_session_id": data.sdk_session_id,
            "start_time": data.created_at,
            "last_active": data.last_active,
            "metadata_json": metadata,
            "status": data.status,
        }

        update_values = {k: v for k, v in values.items() if k != "id"}

        async with AsyncSession(self.engine) as session:
            dialect_name = self.engine.dialect.name
            insert_stmt = self._get_insert_stmt()
            if dialect_name == "sqlite" or (dialect_name == "postgresql" and pg_insert is not None):
                stmt = insert_stmt.values(**values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["id"],
                    set_=update_values,
                )
                await session.execute(stmt)
            elif dialect_name in ("mysql", "mariadb") and mysql_insert is not None:
                stmt = insert_stmt.values(**values)
                stmt = stmt.on_duplicate_key_update(**update_values)
                await session.execute(stmt)
            else:
                # Generic fallback: load existing, update or insert
                existing = await session.get(Conversation, data.session_id)
                if existing is None:
                    session.add(Conversation(**values))
                else:
                    for key, value in update_values.items():
                        setattr(existing, key, value)
            await session.commit()

    async def load_session(self, session_id: str) -> SessionData | None:
        """Load session data by ID."""
        async with AsyncSession(self.engine) as session:
            result = await session.execute(
                select(Conversation).where(Conversation.id == session_id)
            )
            row = result.scalars().first()
            return self._session_from_db(row) if row else None

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        from sqlalchemy import delete as sa_delete

        async with AsyncSession(self.engine) as session:
            stmt = sa_delete(Conversation).where(Conversation.id == session_id)  # type: ignore[arg-type]
            result = await session.execute(stmt)
            await session.commit()
            deleted: bool = result.rowcount > 0  # type: ignore[attr-defined]
            if deleted:
                logger.debug("Deleted session", session_id=session_id)
            return deleted

    async def list_session_ids(
        self,
        *,
        pool_id: str | None = None,
        agent_name: str | None = None,
        cwd: str | None = None,
    ) -> list[str]:
        """List session IDs, optionally filtered."""
        async with AsyncSession(self.engine) as session:
            stmt = select(Conversation.id)
            if pool_id is not None:
                stmt = stmt.where(Conversation.pool_id == pool_id)
            if agent_name is not None:
                stmt = stmt.where(Conversation.agent_name == agent_name)
            if cwd is not None:
                stmt = stmt.where(Conversation.cwd == cwd)
            stmt = stmt.order_by(Conversation.last_active.desc())  # type: ignore[attr-defined]
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def save_checkpoint(
        self,
        session_id: str,
        messages_json: str,
        pending_calls_json: str,
    ) -> None:
        """Save checkpoint data atomically for a session.

        Stores serialized messages and pending deferred calls together
        in the ``checkpoint_data`` JSON column so they can be restored on resume.

        If no ``Conversation`` record exists for the given ``session_id``
        (e.g. ACP sessions that use ``SessionStore`` but never call
        ``save_session``), a minimal record is created automatically so
        that checkpoint data has a row to attach to.

        Args:
            session_id: Session identifier
            messages_json: JSON-serialized list of ModelMessage
            pending_calls_json: JSON-serialized list of PendingDeferredCall
        """
        async with AsyncSession(self.engine) as session:
            result = await session.execute(
                select(Conversation).where(Conversation.id == session_id)
            )
            conv = result.scalar_one_or_none()
            if conv is None:
                conv = Conversation(
                    id=session_id,
                    agent_name="unknown",
                    status="checkpointed",
                )
                logger.debug(
                    "Created minimal Conversation for checkpoint",
                    session_id=session_id,
                )
            conv.checkpoint_data = {
                "messages_json": messages_json,
                "pending_calls": pending_calls_json,
            }
            session.add(conv)
            await session.commit()
            logger.debug("Saved checkpoint", session_id=session_id)

    async def load_checkpoint(self, session_id: str) -> tuple[str, str] | None:
        """Load checkpoint data from the database.

        Returns:
            Tuple of (messages_json, pending_calls_json) or None if no checkpoint exists.
        """
        async with AsyncSession(self.engine) as session:
            result = await session.execute(
                select(Conversation.checkpoint_data).where(Conversation.id == session_id)
            )
            checkpoint_data = result.scalar_one_or_none()
            if checkpoint_data is None:
                return None
            return (
                checkpoint_data.get("messages_json", "[]"),
                checkpoint_data.get("pending_calls", "[]"),
            )

    async def delete_checkpoint(self, session_id: str) -> bool:
        """Delete checkpoint data for a session.

        Args:
            session_id: Session identifier

        Returns:
            ``True`` if checkpoint was deleted, ``False`` if not found
        """
        async with AsyncSession(self.engine) as session:
            result = await session.execute(
                select(Conversation).where(Conversation.id == session_id)
            )
            conv = result.scalar_one_or_none()
            if conv is None or conv.checkpoint_data is None:
                return False
            conv.checkpoint_data = None
            session.add(conv)
            await session.commit()
            logger.debug("Deleted checkpoint", session_id=session_id)
            return True

    async def load_sessions_batch(
        self,
        session_ids: list[str],
        *,
        agent_name: str | None = None,
    ) -> list[SessionData]:
        """Load multiple sessions by IDs in a single query.

        Avoids the N+1 problem of calling ``load_session`` per ID by fetching
        all matching rows in one SQL statement.

        Args:
            session_ids: List of session identifiers to load
            agent_name: Optional filter to return only sessions for this agent

        Returns:
            List of found SessionData objects, ordered by last_active descending
        """
        if not session_ids:
            return []

        async with AsyncSession(self.engine) as session:
            stmt = select(Conversation).where(Conversation.id.in_(session_ids))  # type: ignore[attr-defined]
            if agent_name is not None:
                stmt = stmt.where(Conversation.agent_name == agent_name)
            stmt = stmt.order_by(Conversation.last_active.desc())  # type: ignore[attr-defined]
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._session_from_db(row) for row in rows]
