"""Message format conversion and OpenCode message handling mixin.

Extracted from session_pool_integration.py as part of the session-debt-cleanup
file split. Contains message conversion utilities and the message bridge mixin
that provides tool-part creation/update methods for subagent sessions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from wolfharness.log import get_logger
from wolfharness.utils.time_utils import now_ms
from wolfharness_server.opencode_server.converters import (
    chat_message_to_opencode,
    opencode_to_chat_message,
)
from wolfharness_server.opencode_server.models import (
    MessagePath,
    MessageTime,
    MessageWithParts,
    PartUpdatedEvent,
    TimeCreatedUpdated,
)
from wolfharness_server.opencode_server.models.parts import (
    TimeStart,
    TimeStartEnd,
    TimeStartEndCompacted,
    ToolPart,
    ToolStateCompleted,
    ToolStateError,
    ToolStateRunning,
)
from wolfharness_server.opencode_server.models.session import Session


# ToolState variants that have a ``metadata`` attribute.
# ``ToolStatePending`` does NOT have ``metadata`` — only Running, Completed,
# and Error states do.
_ToolStateWithMetadata = ToolStateRunning | ToolStateCompleted | ToolStateError


if TYPE_CHECKING:
    from wolfharness.agents.events.events import (
        RunErrorEvent,
        SpawnSessionStart,
        StreamCompleteEvent,
    )
    from wolfharness_server.opencode_server.state import ServerState


logger = get_logger(__name__)


def _apply_revert_filter(
    session: Session | None,
    messages: list[MessageWithParts],
) -> list[MessageWithParts]:
    """Apply soft-hide filter based on ``session.revert``.

    If the session has a revert point set, messages at and after the revert
    message are hidden from the returned list.  Messages before the revert
    point are returned unchanged.  The underlying message list is not
    modified — this is a read-only filter.

    Args:
        session: The OpenCode Session model (may be ``None`` if not cached).
        messages: The full message list.

    Returns:
        Filtered message list, or the original list if no revert is set
        or the revert message_id is not found (defensive fallback).
    """
    if session is None or session.revert is None:
        return messages

    revert_msg_id = session.revert.message_id
    for i, msg in enumerate(messages):
        if msg.info.id == revert_msg_id:
            return messages[:i]

    # Defensive fallback: revert message_id not found in the message list.
    # This can happen if the message was already deleted or the ID is stale.
    return messages


async def get_messages_for_session(
    state: ServerState,
    session_id: str,
    *,
    prefer_in_memory: bool = True,
) -> list[MessageWithParts]:
    """Get messages for a session from SessionPool or fall back to ServerState.

    When ``prefer_in_memory`` is True (default, used by sync/TUI), the
    in-memory ``state.messages`` cache is preferred because it retains
    the ORIGINAL part IDs that match SSE PartUpdatedEvent. DB-reconstructed
    messages get NEW part IDs, causing TUI duplication.

    When ``prefer_in_memory`` is False (used by share/fork), the DB
    (SessionPool) is preferred because it has the complete history.

    A soft-hide filter is applied to all return paths: when
    ``session.revert`` is set, messages at and after the revert point are
    hidden from the result, even though they still exist in the DB and
    in-memory state.

    Args:
        state: The OpenCode server state.
        session_id: The session ID to get messages for.
        prefer_in_memory: If True, prefer in-memory messages (for sync/TUI).
            If False, prefer DB (for share/fork).

    Returns:
        List of MessageWithParts for the session.
    """
    messages: list[MessageWithParts] = getattr(state, "messages", {}).get(session_id, []) or []

    cached_session = state.sessions.get(session_id)

    # Fast-path: in-memory messages retain original part IDs matching SSE.
    if prefer_in_memory and messages:
        return _apply_revert_filter(cached_session, messages)

    pool = state.pool_or_none
    session_pool = getattr(pool, "session_pool", None) if pool is not None else None
    if session_pool is not None:
        try:
            sp_messages = await session_pool.get_messages(session_id)
        except (KeyError, TypeError):
            sp_messages = []
        if sp_messages:
            agent = state.agent
            existing_agent = session_pool.sessions.get_session_agent(session_id)
            if existing_agent is not None:
                agent = existing_agent
            default_model_id, default_provider_id = state.resolve_default_model_info()
            model_variants = state.model_variants
            converted = [
                chat_message_to_opencode(
                    chat_msg,
                    session_id=session_id,
                    working_dir=state.working_dir,
                    agent_name=agent.name,
                    model_id=default_model_id,
                    provider_id=default_provider_id,
                    model_variants=model_variants,
                )
                for chat_msg in sp_messages
            ]
            return _apply_revert_filter(cached_session, converted)
    return _apply_revert_filter(cached_session, messages)


async def append_message_to_session(
    state: ServerState,
    session_id: str,
    msg: MessageWithParts,
) -> None:
    """Append a message to a session's history.

    Writes to SessionPool when the feature flag is enabled.
    Also writes to the in-memory messages dict when present for
    backward compatibility with tests and legacy code paths.

    Args:
        state: The OpenCode server state.
        session_id: The session ID to append to.
        msg: The OpenCode message to append.
    """
    pool = state.pool_or_none
    session_pool = getattr(pool, "session_pool", None) if pool is not None else None
    if session_pool is not None:
        chat_msg = opencode_to_chat_message(msg, session_id=session_id)
        try:
            await session_pool.append_message(session_id, chat_msg)
        except (KeyError, TypeError):
            logger.warning(
                "Failed to append message to SessionPool",
                session_id=session_id,
                exc_info=True,
            )
        except ValueError as exc:
            # The storage provider (e.g. MemoryProvider) raises ValueError
            # for duplicate message IDs. In the sync path (POST /message),
            # the REST handler pre-stores the assistant message before the
            # event bridge tries to store it with the same canonical ID
            # (via _pending_message_ids). This is expected — the message
            # is already in storage, so we skip the duplicate write
            # gracefully instead of propagating the error.
            if "Duplicate message ID" in str(exc):
                logger.debug(
                    "Message already in storage, skipping duplicate write",
                    session_id=session_id,
                )
            else:
                raise

    # Always mirror to the in-memory dict when present for backward compatibility.
    # For duplicate writes (pre-stored by REST handler), skip the in-memory
    # append too — the message was already added by the REST handler's call.
    messages = getattr(state, "messages", None)
    if messages is not None:
        messages.setdefault(session_id, [])
        msg_id = msg.info.id
        already_in_memory = any(m.info.id == msg_id for m in messages[session_id])
        if not already_in_memory:
            messages[session_id].append(msg)
            logger.info(
                "append_message_to_session APPENDED",
                session_id=session_id,
                message_id=msg_id,
                role=getattr(msg.info, "role", "unknown"),
                list_len=len(messages[session_id]),
                all_ids=[m.info.id for m in messages[session_id]],
            )


async def set_messages_for_session(
    state: ServerState,
    session_id: str,
    messages: list[MessageWithParts],
) -> None:
    """Replace all in-memory messages for a session.

    This is a bulk operation used after compaction/summarization when
    the UI-visible message list should be reset to a specific set.
    SessionPool storage is managed separately via storage.replace_conversation_messages.

    Args:
        state: The OpenCode server state.
        session_id: The session ID to update.
        messages: The new message list.
    """
    in_memory_messages = getattr(state, "messages", None)
    if in_memory_messages is not None:
        in_memory_messages[session_id] = list(messages)


def _session_state_to_opencode(state: Any) -> Session:
    """Convert SessionPool SessionState to OpenCode Session model.

    Args:
        state: SessionState from SessionPool.

    Returns:
        OpenCode Session model.
    """
    from wolfharness_storage.opencode_provider import helpers

    created_ms = state.created_at_ns // 1_000_000
    updated_ms = state.last_active_at_ns // 1_000_000
    directory = state.metadata.get("cwd", "")
    project_id = state.metadata.get("project_id", "")
    if not project_id and directory:
        project_id = helpers.compute_project_id(directory)
    if not project_id:
        project_id = "default"

    return Session(
        id=state.session_id,
        project_id=project_id,
        directory=directory,
        title=state.metadata.get("title", "New Session"),
        version="1",
        time=TimeCreatedUpdated(created=created_ms, updated=updated_ms),
        parent_id=state.parent_session_id,
    )


def _reconstruct_tool_parts_from_checkpoint(
    state: ServerState,
    session_id: str,
    pending_calls: list[Any],
) -> None:
    """Reconstruct running ToolParts from pending deferred calls.

    Creates an assistant message (if one does not exist) and appends
    a ``ToolPart`` with ``ToolStateRunning`` for each pending deferred
    call. This restores the visual tool state in the OpenCode TUI so
    the user sees what tools were in-flight at checkpoint time.

    Args:
        state: The OpenCode server state.
        session_id: The session to reconstruct ToolParts for.
        pending_calls: Unresolved deferred tool calls from the checkpoint.
    """
    if not pending_calls:
        return

    from wolfharness.utils import identifiers as identifier
    from wolfharness.utils.time_utils import now_ms
    from wolfharness_server.opencode_server.models.parts import (
        TimeStart,
        ToolPart,
        ToolStateRunning,
    )

    # Create an assistant message to hold the ToolParts
    assistant_msg_id = identifier.ascending("message")

    # Agent/model propagation: look up the real agent_name from the
    # session state instead of hardcoding "wolfharness". Falls back to
    # "wolfharness" when the session state is unavailable.
    agent_name = "wolfharness"
    pool = state.pool_or_none
    session_pool = pool.session_pool if pool is not None else None
    if session_pool is not None:
        session_state = session_pool.sessions.get_session(session_id)
        if session_state is not None:
            agent_name = session_state.agent_name

    default_model_id, default_provider_id = state.resolve_default_model_info()

    assistant_msg = MessageWithParts.assistant(
        message_id=assistant_msg_id,
        session_id=session_id,
        time=MessageTime(created=now_ms()),
        agent_name=agent_name,
        model_id=default_model_id,
        parent_id=session_id,
        provider_id=default_provider_id,
        path=MessagePath(cwd=state.working_dir, root=state.working_dir),
    )

    for call in pending_calls:
        ts = TimeStart(start=now_ms())
        running_state = ToolStateRunning(
            time=ts,
            input={
                "description": call.tool_name,
                "tool_call_id": call.tool_call_id,
            },
            metadata={"deferred": True, "deferred_strategy": call.deferred_strategy},
            title=call.tool_name,
        )
        tool_part = ToolPart(
            id=identifier.ascending("part"),
            message_id=assistant_msg_id,
            session_id=session_id,
            tool=call.tool_name,
            call_id=call.tool_call_id,
            state=running_state,
        )
        assistant_msg.parts.append(tool_part)

    # Register in the in-memory message list
    messages = getattr(state, "messages", None)
    if messages is not None:
        messages.setdefault(session_id, [])
        messages[session_id].append(assistant_msg)


class OpenCodeMessageBridgeMixin:
    """Mixin providing message format conversion and tool-part management.

    Provides methods for creating and updating ToolParts that represent
    subagent sessions in the parent session's message history.

    Attributes:
        server_state: The OpenCode server state (provided by main class).
    """

    server_state: ServerState

    async def _create_subagent_tool_part(
        self,
        parent_session_id: str,
        spawn_event: SpawnSessionStart,
    ) -> ToolPart | None:
        """Create a ToolPart in the parent session representing a subagent.

        This replaces the ToolPart creation that previously happened inside
        EventProcessor._process_subagent_event when events were wrapped in
        SubAgentEvent.

        Args:
            parent_session_id: The parent session ID.
            spawn_event: The spawn event containing subagent metadata.

        Returns:
            The created ToolPart, or None if one already exists for this child.
        """
        from wolfharness.utils import identifiers as identifier

        # Find the parent session's latest assistant message from in-memory state
        # (not via get_messages_for_session, which may return copies when
        # SessionPool message storage is enabled).
        messages = getattr(self.server_state, "messages", {}).get(parent_session_id, []) or []
        assistant_msg = None
        for msg in reversed(messages):
            if msg.info.role == "assistant":
                assistant_msg = msg
                break

        if assistant_msg is None:
            logger.warning(
                "No assistant message found for parent session %s, skipping ToolPart creation",
                parent_session_id,
            )
            return None

        # Check if ToolPart already exists for this child session
        child_session_id = spawn_event.child_session_id
        for part in assistant_msg.parts:
            if (
                isinstance(part, ToolPart)
                and part.metadata is not None
                and part.metadata.get("sessionId") == child_session_id
            ):
                logger.debug("ToolPart already exists for child session %s", child_session_id)
                return None

        display_name = spawn_event.display_name or spawn_event.source_name or "subagent"
        # Team members get a "Team ·" prefix to distinguish from regular subagents
        team_id = spawn_event.metadata.get("team_id")
        if team_id is not None:
            team_name = spawn_event.metadata.get("team_name", "")
            team_role = spawn_event.metadata.get("team_role", "member")
            role_label = "Lead" if team_role == "lead" else "Member"
            tool_title = f"Team · {display_name}"
            card_description = f"{role_label} in '{team_name}'" if team_name else role_label
        else:
            tool_title = display_name
            card_description = spawn_event.description or tool_title
        ts = TimeStart(start=now_ms())
        running_state = ToolStateRunning(
            time=ts,
            input={
                "description": card_description,
                "subagent_type": tool_title,
                "prompt": spawn_event.metadata.get("prompt", ""),
            },
            metadata={
                "sessionId": child_session_id,
                "title": tool_title,
                "model_id": spawn_event.model_id,
                "mode": spawn_event.mode,
                "source_type": spawn_event.source_type,
                "background": spawn_event.spawn_mechanism == "task",
            },
            title=tool_title,
        )
        tool_part = ToolPart(
            id=identifier.ascending("part"),
            message_id=assistant_msg.info.id,
            session_id=parent_session_id,
            tool="task",
            call_id=identifier.ascending("part"),
            state=running_state,
        )
        assistant_msg.parts.append(tool_part)
        await self.server_state.broadcast_event(PartUpdatedEvent.create(tool_part))
        logger.debug(
            "Created ToolPart for child session %s in parent %s",
            child_session_id,
            parent_session_id,
        )
        return tool_part

    async def _update_parent_toolpart(
        self,
        parent_session_id: str,
        child_session_id: str,
        spawn_event: SpawnSessionStart,
        event: StreamCompleteEvent[Any],
    ) -> None:
        """Update parent ToolPart to Completed when child subagent finishes.

        Args:
            parent_session_id: The parent session ID.
            child_session_id: The child session ID.
            spawn_event: The spawn event containing subagent metadata.
            event: The StreamCompleteEvent from the child.
        """
        # Find the ToolPart for this child session across ALL assistant
        # messages (not just the last one). A background task may start in
        # turn 1 (ToolPart in msg_A1) and complete after turn 2 has started
        # (msg_A2 is now the last assistant message). Searching only the last
        # message would miss the ToolPart in the earlier message.
        messages = self.server_state.messages.get(parent_session_id, [])
        assistant_msg: MessageWithParts | None = None
        tool_part: ToolPart | None = None
        for msg in reversed(messages):
            if msg.info.role != "assistant":
                continue
            for part in msg.parts:
                metadata = (
                    part.state.metadata
                    if isinstance(part, ToolPart) and isinstance(part.state, _ToolStateWithMetadata)
                    else None
                )
                if (
                    isinstance(part, ToolPart)
                    and isinstance(metadata, dict)
                    and metadata.get("sessionId") == child_session_id
                ):
                    tool_part = part
                    assistant_msg = msg
                    break
            if tool_part is not None:
                break

        if tool_part is None or assistant_msg is None:
            logger.warning(
                "No ToolPart found for child session %s in parent %s",
                child_session_id,
                parent_session_id,
            )
            return

        display_name = spawn_event.display_name or spawn_event.source_name or "subagent"
        team_id = spawn_event.metadata.get("team_id")
        if team_id is not None:
            team_name = spawn_event.metadata.get("team_name", "")
            team_role = spawn_event.metadata.get("team_role", "member")
            role_label = "Lead" if team_role == "lead" else "Member"
            tool_title = f"Team · {display_name}"
            card_description = f"{role_label} in '{team_name}'" if team_name else role_label
        else:
            tool_title = display_name
            card_description = spawn_event.description or tool_title
        complete_msg = event.message
        content = str(complete_msg.content) if complete_msg.content else "(no output)"

        start_time = (
            tool_part.state.time.start
            if isinstance(tool_part.state, ToolStateRunning)
            else now_ms()
        )
        completed_state = ToolStateCompleted(
            input={
                "description": card_description,
                "subagent_type": tool_title,
                "prompt": spawn_event.metadata.get("prompt", ""),
            },
            output=content,
            title=tool_title,
            metadata={
                "sessionId": child_session_id,
                "title": tool_title,
                "model_id": spawn_event.model_id,
                "mode": spawn_event.mode,
                "source_type": spawn_event.source_type,
                "background": spawn_event.spawn_mechanism == "task",
            },
            time=TimeStartEndCompacted(start=start_time, end=now_ms()),
        )
        updated = ToolPart(
            id=tool_part.id,
            message_id=tool_part.message_id,
            session_id=tool_part.session_id,
            tool=tool_part.tool,
            call_id=tool_part.call_id,
            state=completed_state,
        )

        # Replace the old part in the message
        for i, part in enumerate(assistant_msg.parts):
            if part.id == tool_part.id:
                assistant_msg.parts[i] = updated
                break

        await self.server_state.broadcast_event(PartUpdatedEvent.create(updated))
        logger.debug(
            "Updated ToolPart to Completed for child session %s in parent %s",
            child_session_id,
            parent_session_id,
        )

    async def _update_parent_toolpart_error(
        self,
        parent_session_id: str,
        child_session_id: str,
        spawn_event: SpawnSessionStart,
        event: RunErrorEvent,
    ) -> None:
        """Update parent ToolPart to Error when child subagent fails.

        Args:
            parent_session_id: The parent session ID.
            child_session_id: The child session ID.
            spawn_event: The spawn event containing subagent metadata.
            event: The RunErrorEvent from the child.
        """
        # Find the ToolPart for this child session across ALL assistant
        # messages (not just the last one). See _update_parent_toolpart for
        # the cross-turn background task rationale.
        messages = self.server_state.messages.get(parent_session_id, [])
        assistant_msg: MessageWithParts | None = None
        tool_part: ToolPart | None = None
        for msg in reversed(messages):
            if msg.info.role != "assistant":
                continue
            for part in msg.parts:
                metadata = (
                    part.state.metadata
                    if isinstance(part, ToolPart) and isinstance(part.state, _ToolStateWithMetadata)
                    else None
                )
                if (
                    isinstance(part, ToolPart)
                    and isinstance(metadata, dict)
                    and metadata.get("sessionId") == child_session_id
                ):
                    tool_part = part
                    assistant_msg = msg
                    break
            if tool_part is not None:
                break

        if tool_part is None or assistant_msg is None:
            return

        display_name = spawn_event.display_name or spawn_event.source_name or "subagent"
        team_id = spawn_event.metadata.get("team_id")
        if team_id is not None:
            team_name = spawn_event.metadata.get("team_name", "")
            team_role = spawn_event.metadata.get("team_role", "member")
            role_label = "Lead" if team_role == "lead" else "Member"
            tool_title = f"Team · {display_name}"
            card_description = f"{role_label} in '{team_name}'" if team_name else role_label
        else:
            tool_title = display_name
            card_description = spawn_event.description or tool_title
        error_msg = event.message or "Unknown error"

        start_time = (
            tool_part.state.time.start
            if isinstance(tool_part.state, ToolStateRunning)
            else now_ms()
        )
        error_state = ToolStateError(
            error=error_msg,
            input={
                "description": card_description,
                "subagent_type": tool_title,
                "prompt": spawn_event.metadata.get("prompt", ""),
            },
            metadata={
                "sessionId": child_session_id,
                "title": tool_title,
                "model_id": spawn_event.model_id,
                "mode": spawn_event.mode,
                "source_type": spawn_event.source_type,
                "background": spawn_event.spawn_mechanism == "task",
            },
            time=TimeStartEnd(start=start_time, end=now_ms()),
        )
        updated = ToolPart(
            id=tool_part.id,
            message_id=tool_part.message_id,
            session_id=tool_part.session_id,
            tool=tool_part.tool,
            call_id=tool_part.call_id,
            state=error_state,
        )

        # Replace the old part in the message
        for i, part in enumerate(assistant_msg.parts):
            if part.id == tool_part.id:
                assistant_msg.parts[i] = updated
                break

        await self.server_state.broadcast_event(PartUpdatedEvent.create(updated))
        logger.debug(
            "Updated ToolPart to Error for child session %s in parent %s",
            child_session_id,
            parent_session_id,
        )
