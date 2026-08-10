"""Message persistence mixin for SQLModelProvider.

Extracted from sql_provider.py as part of the session-debt-cleanup file split.
Contains all message-related database operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import desc, select

from wolfharness.log import get_logger
from wolfharness.utils.time_utils import get_now
from wolfharness_storage.sql_provider.models import Message
from wolfharness_storage.sql_provider.utils import (
    build_message_query,
    parse_model_info,
    to_chat_message,
)


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from wolfharness.messaging import ChatMessage
    from wolfharness_config.session import SessionQuery


logger = get_logger(__name__)

# Dialect-specific insert helpers (imported at module level for reuse).
from sqlalchemy.dialects.sqlite import insert as sqlite_insert  # noqa: E402


try:
    from sqlalchemy.dialects.postgresql import insert as pg_insert
except ImportError:
    pg_insert = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
try:
    from sqlalchemy.dialects.mysql import insert as mysql_insert
except ImportError:
    mysql_insert = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]


class SQLMessagesMixin:
    """Mixin providing message persistence methods for SQLModelProvider.

    Attributes:
        engine: Async database engine (provided by SQLModelProvider).
    """

    engine: AsyncEngine

    async def filter_messages(self, query: SessionQuery) -> list[ChatMessage[str]]:
        """Filter messages using SQL queries."""
        async with AsyncSession(self.engine) as session:
            stmt = build_message_query(query)
            result = await session.execute(stmt)
            messages = result.scalars().all()
            return [to_chat_message(msg) for msg in messages]

    async def log_message(self, *, message: ChatMessage[Any]) -> None:
        """Log message to database.

        Message persistence is intentionally idempotent. Streaming UIs may log
        a placeholder message before the agent run completes, then log the same
        message ID again with final content, tool results, token usage, and
        finish metadata.
        """
        from wolfharness.storage.serialization import serialize_messages

        provider, model_name = parse_model_info(message.model_name)
        cost_info = message.cost_info
        values = {
            "session_id": message.session_id or "",
            "id": message.message_id,
            "parent_id": message.parent_id,
            "content": str(message.content),
            "role": message.role,
            "name": message.name,
            "model": message.model_name,
            "model_provider": provider,
            "model_name": model_name,
            "response_time": message.response_time,
            "total_tokens": cost_info.token_usage.total_tokens if cost_info else None,
            "input_tokens": cost_info.token_usage.input_tokens if cost_info else None,
            "output_tokens": cost_info.token_usage.output_tokens if cost_info else None,
            "cost": float(cost_info.total_cost) if cost_info else None,
            "provider_name": message.provider_name,
            "provider_response_id": message.provider_response_id,
            "messages": serialize_messages(message.messages),
            "finish_reason": message.finish_reason,
            "timestamp": message.timestamp,
        }

        async with AsyncSession(self.engine) as session:
            update_values = {key: value for key, value in values.items() if key != "id"}
            dialect_name = self.engine.dialect.name
            if dialect_name == "sqlite":
                stmt = sqlite_insert(Message).values(**values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["id"],
                    set_=update_values,
                )
                await session.execute(stmt)
            elif dialect_name == "postgresql" and pg_insert is not None:
                pg_stmt = pg_insert(Message).values(**values)
                pg_stmt = pg_stmt.on_conflict_do_update(
                    index_elements=["id"],
                    set_=update_values,
                )
                await session.execute(pg_stmt)
            elif dialect_name in ("mysql", "mariadb") and mysql_insert is not None:
                mysql_stmt = mysql_insert(Message).values(**values)
                mysql_stmt = mysql_stmt.on_duplicate_key_update(**update_values)
                await session.execute(mysql_stmt)
            else:
                existing = await session.get(Message, message.message_id)
                if existing is None:
                    session.add(Message(**values))
                else:
                    for key, value in update_values.items():
                        setattr(existing, key, value)
            await session.commit()

    async def get_session_messages(
        self,
        session_id: str,
        *,
        include_ancestors: bool = False,
    ) -> list[ChatMessage[str]]:
        """Get all messages for a session.

        Args:
            session_id: ID of the conversation
            include_ancestors: If True, traverse parent_id chain to include
                messages from ancestor conversations (for forked convos).

        Returns:
            List of messages ordered by timestamp.
        """
        async with AsyncSession(self.engine) as session:
            # Get messages for this conversation
            result = await session.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.timestamp.asc(), Message.id.asc())  # type: ignore
            )
            messages = [to_chat_message(m) for m in result.scalars().all()]

            if not include_ancestors or not messages:
                return messages

            # Find the first message's parent_id to get ancestor chain
            first_msg = messages[0]
            if first_msg.parent_id:
                ancestors = await self.get_message_ancestry(
                    first_msg.parent_id, session_id=session_id
                )
                return ancestors + messages

            return messages

    async def get_message(
        self,
        message_id: str,
        *,
        session_id: str | None = None,
    ) -> ChatMessage[str] | None:
        """Get a single message by ID.

        When ``session_id`` is set, the message must belong to that session.
        """
        async with AsyncSession(self.engine) as session:
            stmt = select(Message).where(Message.id == message_id)
            if session_id is not None:
                stmt = stmt.where(Message.session_id == session_id)
            result = await session.execute(stmt)
            msg = result.scalar_one_or_none()
            return to_chat_message(msg) if msg else None

    async def get_message_ancestry(
        self,
        message_id: str,
        *,
        session_id: str | None = None,
    ) -> list[ChatMessage[str]]:
        """Get the ancestry chain of a message.

        Traverses parent_id chain to build full history.
        """
        ancestors: list[ChatMessage[str]] = []
        current_id: str | None = message_id

        async with AsyncSession(self.engine) as session:
            while current_id:
                result = await session.execute(select(Message).where(Message.id == current_id))
                msg = result.scalar_one_or_none()
                if not msg:
                    break
                ancestors.append(to_chat_message(msg))
                current_id = msg.parent_id

        # Reverse to get oldest first
        ancestors.reverse()
        return ancestors

    async def fork_conversation(
        self,
        *,
        source_session_id: str,
        new_session_id: str,
        fork_from_message_id: str | None = None,
        new_agent_name: str | None = None,
    ) -> str | None:
        """Fork a conversation at a specific point.

        Creates a new conversation record. The fork point message_id is returned
        so callers can set it as parent_id for new messages.
        """
        from wolfharness_storage.sql_provider.models import Conversation

        async with AsyncSession(self.engine) as session:
            # Get source conversation
            result = await session.execute(
                select(Conversation).where(Conversation.id == source_session_id)
            )
            source_conv = result.scalar_one_or_none()
            if not source_conv:
                msg = f"Source conversation not found: {source_session_id}"
                raise ValueError(msg)

            # Determine fork point
            fork_point_id: str | None = None
            if fork_from_message_id:
                # Verify the message exists and belongs to the source conversation
                msg_result = await session.execute(
                    select(Message).where(
                        Message.id == fork_from_message_id,
                        Message.session_id == source_session_id,
                    )
                )
                if not msg_result.scalar_one_or_none():
                    raise ValueError(f"Message {fork_from_message_id} not found in conversation")
                fork_point_id = fork_from_message_id
            else:
                # Fork from the last message
                msg_result = await session.execute(
                    select(Message)
                    .where(Message.session_id == source_session_id)
                    .order_by(desc(Message.timestamp))
                    .limit(1)
                )
                if last_msg := msg_result.scalar_one_or_none():
                    fork_point_id = last_msg.id

            # Create new conversation
            agent_name = new_agent_name or source_conv.agent_name
            new_conv = Conversation(
                id=new_session_id,
                agent_name=agent_name,
                title=f"{source_conv.title or 'Conversation'} (fork)"
                if source_conv.title
                else None,
                start_time=get_now(),
            )
            session.add(new_conv)
            await session.commit()

            return fork_point_id

    async def delete_session_messages(self, session_id: str) -> int:
        """Delete all messages for a session."""
        from sqlalchemy import delete, func

        async with AsyncSession(self.engine) as session:
            # First count messages to return
            count_result = await session.execute(
                select(func.count()).where(Message.session_id == session_id)
            )
            count = count_result.scalar() or 0
            # Then delete
            await session.execute(
                delete(Message).where(Message.session_id == session_id)  # type: ignore[arg-type]
            )
            await session.commit()
            return count

    async def truncate_messages(self, session_id: str, up_to_message_id: str) -> int:
        """Delete the message with up_to_message_id and all messages after it.

        Finds the boundary message by ``session_id`` + ``up_to_message_id``,
        reads its timestamp, then deletes every message in the session whose
        timestamp is greater than or equal to that boundary timestamp.

        Args:
            session_id: ID of the conversation to truncate.
            up_to_message_id: ID of the message that marks the truncation
                boundary. This message and everything after it is deleted.

        Returns:
            The count of deleted messages. Returns 0 when the boundary
            message does not exist in the session.
        """
        from sqlalchemy import delete, func

        async with AsyncSession(self.engine) as session:
            # Find the boundary message to get its timestamp.
            boundary_result = await session.execute(
                select(Message.timestamp).where(
                    Message.id == up_to_message_id,
                    Message.session_id == session_id,
                )
            )
            boundary_timestamp = boundary_result.scalar_one_or_none()
            if boundary_timestamp is None:
                return 0

            # Count the messages that will be deleted.
            count_result = await session.execute(
                select(func.count()).where(
                    Message.session_id == session_id,
                    Message.timestamp >= boundary_timestamp,
                )
            )
            count = count_result.scalar() or 0

            # Delete the boundary message and everything after it.
            await session.execute(
                delete(Message).where(
                    Message.session_id == session_id,  # type: ignore[arg-type]
                    Message.timestamp >= boundary_timestamp,  # type: ignore[arg-type]
                )
            )
            await session.commit()
            return count
