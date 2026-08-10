"""Session routes."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
import shlex
from typing import TYPE_CHECKING, Any

from anyenv.text_sharing.opencode import Message, MessagePart, OpenCodeSharer
from fastapi import APIRouter, HTTPException
from pydantic_ai import FileUrl
from slashed import CommandContext

from wolfharness.log import get_logger
from wolfharness.repomap import RepoMap, find_src_files
from wolfharness.utils import identifiers as identifier
from wolfharness.utils.identifiers import extract_timestamp_ms
from wolfharness.utils.time_utils import now_ms
from wolfharness_server.opencode_server.command_validation import validate_command
from wolfharness_server.opencode_server.converters import (
    chat_message_to_opencode,
    opencode_to_session_data,
    session_data_to_opencode,
)
from wolfharness_server.opencode_server.dependencies import StateDep
from wolfharness_server.opencode_server.input_provider import OpenCodeInputProvider
from wolfharness_server.opencode_server.models import (
    AssistantMessage,
    CommandExecutedEvent,
    CommandRequest,
    FileDiff,
    MessagePath,
    MessageTime,
    MessageUpdatedEvent,
    MessageWithParts,
    OpenCodeBaseModel,
    PartDeltaEvent,
    PartUpdatedEvent,
    PermissionAskedProperties,
    PermissionReplyRequest,
    PermissionResolvedEvent,
    Session,
    SessionCreatedEvent,
    SessionCreateRequest,
    SessionDeletedEvent,
    SessionDiffEvent,
    SessionForkRequest,
    SessionIdleEvent,
    SessionInitRequest,
    SessionRevert,
    SessionShare,
    SessionStatus,
    SessionStatusEvent,
    SessionUpdatedEvent,
    SessionUpdateRequest,
    ShellRequest,
    StepFinishPart,
    StepStartPart,
    SummarizeRequest,
    TextPart,
    TimeCreatedUpdated,
    Todo,
    Tokens,
)
from wolfharness_server.opencode_server.session_pool_integration import (
    append_message_to_session,
    get_messages_for_session,
    get_session_status as _get_single_session_status,
    set_messages_for_session,
    set_session_status,
)
from wolfharness_server.opencode_server.stream_adapter import OpenCodeStreamAdapter
from wolfharness_server.opencode_server.todo_utils import build_opencode_todos
from wolfharness_storage.opencode_provider import helpers
from wolfharness_storage.protocols import SessionPersistence


if TYPE_CHECKING:
    from wolfharness.sessions.models import SessionData
    from wolfharness_server.opencode_server.state import ServerState

logger = get_logger(__name__)


def _resolve_session_create_agent(state: ServerState, requested_agent: str | None) -> str:
    """Resolve the agent to bind to a newly created OpenCode session."""
    default_agent = state.agent.name or "default"
    if not requested_agent or requested_agent == "default":
        return default_agent

    pool = state.pool
    if requested_agent not in pool.manifest.agents:
        raise HTTPException(status_code=400, detail=f"Unknown agent: {requested_agent}")
    return requested_agent


async def _get_session_messages_from_pool(
    state: ServerState,
    session_id: str,
) -> list[MessageWithParts]:
    """Get messages for a session from SessionPool via get_messages_for_session.

    Delegates to :func:`get_messages_for_session` which handles feature-flag
    routing and ChatMessage-to-MessageWithParts conversion.
    """
    return await get_messages_for_session(state, session_id, prefer_in_memory=False)


class _CommandOutputCapture:
    """Output writer that captures command output to a string buffer."""

    def __init__(self) -> None:
        self._buffer: list[str] = []

    async def print(self, message: str) -> None:
        """Write a message to the buffer."""
        self._buffer.append(message)

    def __str__(self) -> str:
        """Get the captured output as a single string."""
        return "\n".join(self._buffer)


class _DiscardOutputWriter:
    """Output writer that discards all command output.

    Used for skill commands where ``ctx.print`` output (e.g.
    "Loading skill: ...") must NOT be rendered as an assistant ``TextPart``.
    Mirrors ACP's ``handler.py:582``
    (``output_writer=lambda msg: logger.debug(...)``).
    """

    async def print(self, message: str) -> None:
        """Discard the message (log at debug level only)."""
        logger.debug("Skill command output discarded: %s", message)


def _process_skill_template(template: str, arguments: str | None) -> str:
    """Process skill template with placeholder substitution like opencode.

    Args:
        template: The skill instruction template.
        arguments: The command arguments string.

    Returns:
        The processed template with placeholders replaced.

    Supported placeholders:
        - $1, $2, etc.: Positional arguments
        - $ARGUMENTS: All arguments as a single string
    """
    args = arguments.split() if arguments else []

    # Find numbered placeholders $1, $2, etc.
    import re

    placeholder_regex = r"\$(\d+|ARGUMENTS)"
    placeholders = re.findall(placeholder_regex, template)

    # Find the highest numbered placeholder
    last_pos = 0
    for p in placeholders:
        if p.isdigit():
            last_pos = max(last_pos, int(p))

    # Replace placeholders
    def replace_placeholder(match: re.Match[str]) -> str:
        placeholder = match.group(1)
        if placeholder == "ARGUMENTS":
            return arguments or ""
        pos = int(placeholder)
        idx = pos - 1
        if idx >= len(args):
            return ""
        if pos == last_pos and idx < len(args):
            # Last placeholder swallows remaining args
            return " ".join(args[idx:])
        return args[idx] if idx < len(args) else ""

    result = re.sub(placeholder_regex, replace_placeholder, template)

    # If no placeholders and arguments exist, wrap in user_request tag
    if not placeholders and arguments and arguments.strip():
        result = result + "\n\n<user_request>\n\n" + arguments + "\n\n</user_request>"

    return result


async def _execute_slashed_command(  # noqa: PLR0915
    state: ServerState,
    session_id: str,
    request: CommandRequest,
) -> MessageWithParts:
    """Execute a slashed command from the CommandStore.

    Args:
        state: The server state containing the command store and agent.
        session_id: The session ID for this command execution.
        request: The command request with command name and arguments.

    Returns:
        MessageWithParts containing the command output.

    Raises:
        HTTPException: 404 if command store not initialized or command not found.
        HTTPException: 500 if command execution fails.
    """
    # Validate command store is available
    if state.command_store is None:
        raise HTTPException(status_code=404, detail="Command store not initialized")

    # Retrieve command from store
    command = state.command_store.get_command(request.command)
    if command is None:
        raise HTTPException(status_code=404, detail=f"Command not found: {request.command}")

    # Create assistant message (before execution)
    now = now_ms()
    # D14: Use request.message_id if provided for end-to-end ID consistency.
    assistant_msg_id = identifier.ascending("message", request.message_id)
    assistant_message = AssistantMessage(
        id=assistant_msg_id,
        session_id=session_id,
        parent_id="",
        model_id=request.model or "default",
        provider_id="opencode",
        mode="command",
        agent=request.agent or "default",
        path=MessagePath(cwd=state.working_dir, root=state.working_dir),
        time=MessageTime(created=now),
    )

    # Initialize message with parts
    message_with_parts = MessageWithParts(info=assistant_message, parts=[])

    # Store message in state and broadcast
    await append_message_to_session(state, session_id, message_with_parts)
    await state.broadcast_event(MessageUpdatedEvent.create(assistant_message))

    try:
        # Mark session as busy
        await set_session_status(state, session_id, SessionStatus(type="busy"))
        await state.broadcast_event(
            SessionStatusEvent.create(session_id, SessionStatus(type="busy"))
        )

        # Add step-start part to indicate command is running
        part_id = identifier.ascending("part")
        step_start = StepStartPart(id=part_id, message_id=assistant_msg_id, session_id=session_id)
        message_with_parts.parts.append(step_start)
        await state.broadcast_event(PartUpdatedEvent.create(step_start))

        # Parse arguments
        args = request.arguments.split() if request.arguments else []

        # Create command context with output capture
        output_capture = _CommandOutputCapture()
        session_agent = state.agent
        cmd_ctx = CommandContext(
            output=output_capture,
            data=session_agent.get_context(),
            command_store=state.command_store,
        )

        # Execute command
        try:
            await command.execute(cmd_ctx, args, {})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Command execution failed: {e}") from e

        # Get command output
        output_text = str(output_capture) if output_capture else "Command executed"

        # Create text part with output
        text_part = TextPart(
            id=identifier.ascending("part"),
            message_id=assistant_msg_id,
            session_id=session_id,
            text=output_text,
        )
        message_with_parts.parts.append(text_part)
        await state.broadcast_event(PartUpdatedEvent.create(text_part))

        # Run agent to process the loaded skill context
        try:
            # Create adapter to stream agent events through existing text_part
            adapter = OpenCodeStreamAdapter(
                state=state,
                session_id=session_id,
                assistant_msg_id=assistant_msg_id,
                assistant_msg=message_with_parts,
                working_dir=state.working_dir,
            )
            # Build prompt including user arguments
            user_request = (
                request.arguments if request.arguments else "请使用已加载的 skill context"
            )
            agent_prompt = (
                f"用户执行了命令 '{request.command}' 并说: {user_request}\n\n"
                "请使用已加载的 skill context 来回答用户的请求。"
            )

            session_pool = (
                state.pool_or_none.session_pool if state.pool_or_none is not None else None
            )
            if session_pool is not None:
                input_provider = state.ensure_input_provider(session_id)
                iterator = session_pool.run_stream(
                    session_id,
                    agent_prompt,
                    scope="session",
                    input_provider=input_provider,
                    message_id=assistant_msg_id,
                )
            else:
                # Fallback to direct agent if SessionPool not available
                agent = state.agent
                iterator = agent.run_stream(agent_prompt, session_id=session_id)  # type: ignore[assignment]

            async for oc_event in adapter.process_stream(iterator):
                await state.broadcast_event(oc_event)
            # Append adapter's response to text_part
            if adapter.response_text:
                text_part.text = f"{output_text}\n\n{adapter.response_text}"
                await state.broadcast_event(PartUpdatedEvent.create(text_part))
        except Exception:  # noqa: BLE001
            # Command already executed, ignore agent errors
            pass

        # Add step-finish part to indicate command completed
        step_finish = StepFinishPart(
            id=identifier.ascending("part"),
            message_id=assistant_msg_id,
            session_id=session_id,
        )
        message_with_parts.parts.append(step_finish)
        await state.broadcast_event(PartUpdatedEvent.create(step_finish))

        # Broadcast command.executed event
        await state.broadcast_event(
            CommandExecutedEvent.create(
                name=request.command,
                session_id=session_id,
                arguments=request.arguments or "",
                message_id=assistant_msg_id,
            )
        )
    finally:
        await state.mark_session_idle(session_id)

    return message_with_parts


async def _execute_skill_command(
    state: ServerState,
    session_id: str,
    request: CommandRequest,
) -> MessageWithParts:
    """Execute a skill command through the normal prompt lifecycle.

    Routes skill commands (``category == "skill"``) through
    ``send_message`` → EventBus-only consumption (fire-and-forget),
    matching the ACP protocol's correct skill handling pattern.

    Key differences from ``_execute_slashed_command``:
    - Discards ``ctx.print`` output (no TextPart capture).
    - Injects skill instructions + args into the per-session agent's
      ``staged_content`` (via ``skill_bridge.execute_skill``).
    - Passes empty content to ``send_message`` — the model receives
      instructions and args exclusively from ``staged_content``.
    - Passes ``meta=OpenCodeUserMessageMeta(parts=[...])`` with the raw
      command string for TUI display (avoids empty user message bubble).
    - Returns a placeholder immediately (fire-and-forget); the response
      is delivered via SSE events.
    - Does NOT call ``run_stream``, ``_process_message``, or build a
      second hardcoded prompt.
    - All within ``execute_command``'s existing session lock.

    Args:
        state: The server state containing the command store and session pool.
        session_id: The session ID for this command execution.
        request: The command request with command name and arguments.

    Returns:
        MessageWithParts containing the assistant message placeholder
        (response arrives via SSE events).

    Raises:
        HTTPException: 404 if command store not initialized or command not found.
        HTTPException: 500 if skill execution fails.
    """
    if state.command_store is None:
        raise HTTPException(status_code=404, detail="Command store not initialized")

    # Retrieve skill command from store
    command = state.command_store.get_command(request.command)
    if command is None:
        raise HTTPException(status_code=404, detail=f"Command not found: {request.command}")

    # Get per-session agent for staged_content injection (not state.agent)
    # This matches ACP's handler.py:558 pattern: get_or_create_session_agent
    # ensures the agent that send_message runs is the same one whose
    # staged_content receives the skill instructions.
    session_pool = state.pool_or_none.session_pool if state.pool_or_none is not None else None
    if session_pool is None:
        raise HTTPException(status_code=500, detail="SessionPool not available for skill commands")

    session_agent = await session_pool.sessions.get_or_create_session_agent(
        session_id,
        agent_name=request.agent,
    )

    # Execute the skill — injects into per-session agent's staged_content
    # and discards ctx.print output (no TextPart capture).
    # Use shlex.split for consistent quoting semantics with the ACP path
    # (which uses the shlex-based slashed parser).
    output_writer = _DiscardOutputWriter()
    args = shlex.split(request.arguments) if request.arguments else []
    cmd_ctx = CommandContext(
        output=output_writer,
        data=session_agent.get_context(),
        command_store=state.command_store,
    )
    try:
        await command.execute(cmd_ctx, args, {})
    except Exception:
        logger.exception("Skill execution failed for command: %s", request.command)
        # Fire-and-forget design: return placeholder even on skill execution
        # failure. The error is logged; the TUI shows an empty assistant
        # message that will eventually receive an error via SSE.
        user_msg_id = identifier.ascending("message")
        assistant_msg_id = identifier.ascending("message", request.message_id)
        assistant_message = AssistantMessage(
            id=assistant_msg_id,
            session_id=session_id,
            parent_id=user_msg_id,
            model_id=request.model or "default",
            provider_id="opencode",
            mode="command",
            agent=request.agent or "default",
            path=MessagePath(cwd=state.working_dir, root=state.working_dir),
            time=MessageTime(created=now_ms()),
        )
        return MessageWithParts(info=assistant_message, parts=[])

    # Build raw command string for TUI display (e.g. "/lodestone analyze this")
    raw_command = f"/{request.command}"
    if request.arguments:
        raw_command = f"{raw_command} {request.arguments}"

    # Build meta for TUI display — send_message(content="") causes
    # EventProcessor to create an empty user message (no parts) because
    # `if content:` is falsy for empty string. Passing meta with serialized
    # TextPart parts allows the EventProcessor to reconstruct the user
    # message from meta.parts, displaying the raw command in the TUI.
    # This mirrors the normal path (message_routes.py:700-701).
    from wolfharness_server.opencode_server.event_processor import (
        OpenCodeUserMessageMeta,
    )

    user_msg_id = identifier.ascending("message")
    assistant_msg_id = identifier.ascending("message", request.message_id)
    now = now_ms()

    # Create assistant message placeholder (in memory only — NOT pre-persisted).
    # The event bridge (_before_consumer_loop) creates and broadcasts the
    # assistant message when the first real agent event arrives, reusing
    # the assistant_msg_id passed via integration.route_message.
    assistant_message = AssistantMessage(
        id=assistant_msg_id,
        session_id=session_id,
        parent_id=user_msg_id,
        model_id=request.model or "default",
        provider_id="opencode",
        mode="command",
        agent=request.agent or "default",
        path=MessagePath(cwd=state.working_dir, root=state.working_dir),
        time=MessageTime(created=now),
    )
    placeholder = MessageWithParts(info=assistant_message, parts=[])

    # Build serialized TextPart for the raw command string
    text_part_dict = TextPart(
        id=identifier.ascending("part"),
        message_id=user_msg_id,
        session_id=session_id,
        text=raw_command,
    ).model_dump()
    route_meta = OpenCodeUserMessageMeta(parts=[text_part_dict])

    # Route through SessionPool with empty content — the model gets
    # instructions + args exclusively from staged_content (which
    # skill_bridge.execute_skill already populated).
    # Prefer integration.route_message for assistant_msg_id propagation
    # (ensures HTTP response ID matches SSE-delivered ID).
    input_provider = state.ensure_input_provider(session_id)
    integration = state.session_pool_integration
    if integration is not None:
        await integration.route_message(
            session_id=session_id,
            content="",
            input_provider=input_provider,
            agent_name=request.agent,
            message_id=user_msg_id,
            assistant_msg_id=assistant_msg_id,
            model_id=None,
            provider_id=None,
            meta=route_meta,
        )
    else:
        # Fallback: integration not available. In production, server.py
        # always sets session_pool_integration when session_pool is not
        # None, so this branch is effectively unreachable. It exists for
        # test setups without integration. Note: send_message does not
        # support assistant_msg_id, so the SSE-delivered ID may diverge
        # from the HTTP placeholder in this path (cosmetic mismatch only).
        from wolfharness.lifecycle.types import DeliveryMode

        await session_pool.send_message(
            session_id=session_id,
            content="",
            mode=DeliveryMode.QUEUE,
            input_provider=input_provider,
            message_id=user_msg_id,
            meta=route_meta,
        )

    # Note: busy/idle status is NOT explicitly set here. The event bridge
    # handles session status transitions: RunStartedEvent → busy,
    # StreamCompleteEvent → idle. This matches the normal /message path
    # (message_routes.py), which also does not set busy/idle explicitly.

    # Broadcast command.executed event (signals dispatch, not completion)
    await state.broadcast_event(
        CommandExecutedEvent.create(
            name=request.command,
            session_id=session_id,
            arguments=request.arguments or "",
            message_id=assistant_msg_id,
        )
    )

    return placeholder


async def get_or_load_session(state: ServerState, session_id: str) -> Session | None:
    """Get session from cache or load via session-scoped agent.

    Returns None if session not found.
    Uses ``state.agent`` to access the shared server agent for loading
    session conversation history from storage.

    For subagent sessions (child sessions), we prioritize the in-memory version
    because parts are streamed in real-time and may not be immediately persisted
    to storage. This ensures users see the latest message state when viewing
    subagent sessions.
    """
    # For subagent/child sessions: prioritize in-memory messages if available.
    # This is critical because subagent parts are streamed in real-time to memory
    # but are only persisted at completion, not after each part update.
    # A session is considered a subagent session if it has a parent_id.
    cached_session = state.sessions.get(session_id)
    is_subagent_session = cached_session is not None and cached_session.parent_id is not None

    if is_subagent_session and len(await get_messages_for_session(state, session_id)) > 0:
        return cached_session

    # If the session is cached in memory AND registered in SessionController,
    # we have it from create_session or a previous load. Since each session
    # now has its own agent instance, we only need to reload history on
    # cold-start recovery after server restart (when cached_session is None).
    # However, list_sessions may cache session metadata in state.sessions
    # without registering in SessionController — in that case, we must
    # continue with the full loading path to create the agent and register.
    if (
        cached_session is not None
        and state.session_controller is not None
        and state.session_controller.get_session(session_id) is not None
    ):
        return cached_session

    # Load from SessionPool store when available
    session_pool = state.pool.session_pool
    if session_pool is not None and session_pool.sessions.store is not None:
        data = await session_pool.sessions.store.load_session(session_id)
        if data is not None:
            session = session_data_to_opencode(data)
            state.sessions[session_id] = session
            state.ensure_runtime_session_state(session_id)
            if await _get_single_session_status(state, session_id) is None:
                await state.mark_session_idle(session_id)
            # Load conversation history from agent via SessionController
            agent = await session_pool.sessions.get_or_create_session_agent(session_id)
            # Defense-in-depth: skip replacement if messages already exist
            # (concurrent request may have already loaded and appended).
            # See issue #192.
            existing_msgs = state.messages.get(session_id, [])
            if not existing_msgs:
                default_model_id, default_provider_id = state.resolve_default_model_info()
                model_variants = state.model_variants
                await set_messages_for_session(
                    state,
                    session_id,
                    [
                        chat_message_to_opencode(
                            chat_msg,
                            session_id=session_id,
                            working_dir=state.working_dir,
                            agent_name=agent.name,
                            model_id=default_model_id,
                            provider_id=default_provider_id,
                            model_variants=model_variants,
                        )
                        for chat_msg in agent.conversation.chat_messages
                    ],
                )
            state.ensure_input_provider(session_id)
            await state.broadcast_event(SessionUpdatedEvent.create(session))
            return session

    # Fallback: load via agent.load_session()
    existing_messages = (
        await get_messages_for_session(state, session_id) if is_subagent_session else []
    )
    if session_pool is not None:
        # Avoid auto-creating a session for nonexistent session IDs.
        # If the session is not in memory (state.sessions or SessionController)
        # and not in the persistent store (checked above), it doesn't exist.
        # Return early instead of calling get_or_create_session_agent which
        # would auto-create the session as a side effect (and may raise).
        if cached_session is None and (
            state.session_controller is None
            or state.session_controller.get_session(session_id) is None
        ):
            return None
        agent = await session_pool.sessions.get_or_create_session_agent(session_id)
    else:
        agent = state.agent
    data = await agent.load_session(session_id)
    if data is None:
        return cached_session

    session = session_data_to_opencode(data)
    state.sessions[session_id] = session
    state.ensure_runtime_session_state(session_id)
    if await _get_single_session_status(state, session_id) is None:
        await state.mark_session_idle(session_id)

    if not (is_subagent_session and existing_messages):
        # Defense-in-depth: also skip if any messages already exist in memory
        # (concurrent request may have already loaded and appended).
        # See issue #192.
        all_existing_msgs = state.messages.get(session_id, [])
        if not all_existing_msgs:
            default_model_id, default_provider_id = state.resolve_default_model_info()
            model_variants = state.model_variants
            await set_messages_for_session(
                state,
                session_id,
                [
                    chat_message_to_opencode(
                        chat_msg,
                        session_id=session_id,
                        working_dir=state.working_dir,
                        agent_name=agent.name,
                        model_id=default_model_id,
                        provider_id=default_provider_id,
                        model_variants=model_variants,
                    )
                    for chat_msg in agent.conversation.chat_messages
                ],
            )

    state.ensure_input_provider(session_id)
    await state.broadcast_event(SessionUpdatedEvent.create(session))
    return session


router = APIRouter(prefix="/session", tags=["session"])


async def _query_store_session_ids(
    store: SessionPersistence,
    cwd: str | None,
) -> list[str]:
    """Query the store for persisted session IDs, optionally filtered by cwd.

    All current store implementations (SQLModelProvider, MemoryStorageProvider)
    have ``list_session_ids`` and filter by cwd correctly.
    """
    return await store.list_session_ids(cwd=cwd)


async def _load_sessions_from_store(
    store: SessionPersistence,
    session_ids: list[str],
) -> list[SessionData]:
    """Load sessions from the store in batch.

    Delegates to ``store.load_sessions_batch(ids)`` (base class provides
    default fallback to individual ``load_session`` calls for providers
    that don't override it). Returns empty list for empty input.
    """
    if not session_ids:
        return []
    return await store.load_sessions_batch(session_ids)


@router.get("")
async def list_sessions(  # noqa: PLR0915
    state: StateDep,
    directory: str | None = None,
    roots: bool | None = None,
    start: int | None = None,
    search: str | None = None,
    limit: int | None = None,
) -> list[Session]:
    """List all sessions.

    Queries the session store for persisted sessions (source of truth),
    then overlays in-memory active sessions for real-time status.
    Includes newly created sessions not yet persisted.

    Query params:
        directory: Filter sessions by directory (overrides default cwd).
                   The OpenCode SDK auto-injects this param.
        roots: Only return root sessions (no parentID)
        start: Filter sessions updated on or after this timestamp (ms since epoch)
        search: Filter sessions by title (case-insensitive)
        limit: Maximum number of sessions to return
    """
    effective_cwd = directory or state.base_path
    sessions: list[Session] = []

    if state.session_controller is not None:
        session_pool = state.pool.session_pool
        store: SessionPersistence | None = (
            session_pool.sessions.store
            if session_pool is not None and session_pool.sessions is not None
            else None
        )

        if store is not None:
            try:
                # Capture in-memory sessions BEFORE store load (store load will
                # populate state.sessions, overwriting any existing cache entries)
                in_memory_sessions: dict[str, Session] = {}
                for info in state.session_controller.list_sessions():
                    cached = state.sessions.get(info.session_id)
                    if cached is not None:
                        in_memory_sessions[info.session_id] = cached

                # D2: Query store first for persisted session IDs
                session_ids = await _query_store_session_ids(store, effective_cwd)
                # D3: Batch load all sessions
                store_session_data = await _load_sessions_from_store(store, session_ids)

                # Convert to OpenCode Session objects and cache in state.sessions
                sessions_by_id: dict[str, Session] = {}
                for data in store_session_data:
                    session = session_data_to_opencode(data)
                    # Don't overwrite in-memory cached sessions
                    if data.session_id not in in_memory_sessions:
                        state.sessions[data.session_id] = session
                    sessions_by_id[data.session_id] = session

                # D4: Overlay in-memory cached sessions (fresher status: busy/idle)
                for session_id, cached in in_memory_sessions.items():
                    if session_id in sessions_by_id:
                        sessions_by_id[session_id] = cached

                # D5: Append in-memory-only sessions not in store results
                # (newly created, not yet persisted), filtered by cwd
                resolved_cwd = Path(effective_cwd).resolve() if effective_cwd else None
                for session_id, cached in in_memory_sessions.items():
                    if session_id not in sessions_by_id and (
                        resolved_cwd is None
                        or (cached.directory and Path(cached.directory).resolve() == resolved_cwd)
                    ):
                        sessions_by_id[session_id] = cached

                sessions = list(sessions_by_id.values())
            except Exception:
                # D7: Store query failure — degrade to in-memory only
                logger.warning(
                    "Failed to query store for sessions, falling back to in-memory only",
                    exc_info=True,
                )
                for info in state.session_controller.list_sessions():
                    cached = state.sessions.get(info.session_id)
                    if cached is not None:
                        sessions.append(cached)
        else:
            # D9: Store is None — in-memory only
            for info in state.session_controller.list_sessions():
                cached = state.sessions.get(info.session_id)
                if cached is not None:
                    sessions.append(cached)

        # D6: Python-level cwd filter (defensive safety net for future custom stores)
        if effective_cwd:
            resolved_cwd = Path(effective_cwd).resolve()
            sessions = [
                s for s in sessions if s.directory and Path(s.directory).resolve() == resolved_cwd
            ]
    else:
        # Legacy path: load via agent.list_sessions()
        for data in await state.agent.list_sessions(cwd=effective_cwd):
            session = session_data_to_opencode(data)
            state.sessions[data.session_id] = session
            sessions.append(session)

    # D8: Re-sort merged list by time.updated descending
    sessions.sort(key=lambda s: s.time.updated, reverse=True)

    # Apply filters
    if roots:
        sessions = [s for s in sessions if s.parent_id is None]
    if start is not None:
        sessions = [s for s in sessions if s.time.updated >= start]
    if search:
        lower_search = search.lower()
        sessions = [s for s in sessions if lower_search in s.title.lower()]
    if limit is not None:
        sessions = sessions[:limit]
    return sessions


@router.post("")
async def create_session(state: StateDep, request: SessionCreateRequest | None = None) -> Session:
    """Create a new session and persist to storage."""
    session_id = identifier.descending("session")
    now = extract_timestamp_ms(session_id) or now_ms()
    base_path = state.base_path
    project_id = helpers.compute_project_id(base_path)
    agent_name = _resolve_session_create_agent(state, request.agent if request else None)
    session = Session(
        id=session_id,
        project_id=project_id,
        directory=base_path,
        title=request.title if request and request.title else "New Session",
        version="1",
        time=TimeCreatedUpdated(created=now, updated=now),
        parent_id=request.parent_id if request else None,
    )

    # Delegate session creation to SessionPool
    session_pool = state.pool.session_pool
    if session_pool is not None:
        try:
            await session_pool.create_session(
                session_id=session_id,
                agent_name=agent_name,
                parent_session_id=session.parent_id,
                project_id=project_id,
                cwd=base_path,
                title=session.title,
            )
        except Exception:
            logger.exception(
                "SessionPool session creation failed, falling back to in-memory",
                session_id=session_id,
            )
    # Cache in memory
    state.sessions[session_id] = session
    state.ensure_runtime_session_state(session_id)
    await state.mark_session_idle(session_id)
    state.ensure_input_provider(session_id)
    agent = state.agent
    agent.session_id = session_id
    agent.conversation.chat_messages.clear()
    await state.broadcast_event(SessionCreatedEvent.create(session))
    # Broadcast session.updated so the CLI TUI can upsert the session into
    # its SolidJS store.  The CLI TUI's sync.tsx event handler processes
    # session.updated (upsert) but NOT session.created (insert-only), so
    # without this event the TUI would rely solely on the async REST
    # session.sync() call, causing a "black screen" delay while the store
    # is empty.
    await state.broadcast_event(SessionUpdatedEvent.create(session))
    return session


@router.get("/status")
async def get_session_status(state: StateDep) -> dict[str, SessionStatus]:
    """Get status for all sessions.

    Returns only non-idle sessions. If all sessions are idle, returns empty dict.
    Delegates to :func:`_get_single_session_status` for each session so the
    SessionPool integration is consulted when the feature flag is enabled.
    """
    result = {}
    for session_id in list(state.sessions.keys()):
        status = await _get_single_session_status(state, session_id)
        if status is not None and status.type != "idle":
            result[session_id] = status
    return result


@router.get("/{session_id}")
async def get_session(session_id: str, state: StateDep) -> Session:
    """Get session details.

    Loads from storage if not in memory cache.
    """
    # Fast path for subagent/child sessions already in memory:
    # Skip get_or_load_session (which may load from storage) because the
    # parent agent is streaming and subagent parts are in memory.
    cached_session = state.sessions.get(session_id)
    if cached_session is not None and cached_session.parent_id is not None:
        return cached_session

    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/{session_id}/message")
async def get_session_messages(
    session_id: str,
    state: StateDep,
    limit: int | None = None,
) -> list[MessageWithParts]:
    """Get all messages for a session.

    Retrieves all messages in a session, including user prompts and AI responses.
    Loads from storage if session not in memory cache.

    Args:
        session_id: The session ID to get messages for
        state: The server state for accessing storage and sessions
        limit: Optional maximum number of messages to return

    Returns:
        List of messages with their parts
    """
    # Fast path for subagent/child sessions already in memory:
    # Skip get_or_load_session (which may load from storage) because the
    # parent agent is streaming and subagent parts are in memory.
    cached_session = state.sessions.get(session_id)
    if cached_session is not None and cached_session.parent_id is not None:
        messages = await get_messages_for_session(state, session_id)
        if limit is not None and limit > 0:
            messages = messages[-limit:]
        return messages

    # Ensure session is loaded
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = await get_messages_for_session(state, session_id)
    if limit is not None and limit > 0:
        messages = messages[-limit:]
    logger.info(
        "sync() returning {count} messages for session {sid}: {ids}",
        count=len(messages),
        sid=session_id,
        ids=[m.info.id for m in messages],
    )
    return messages


@router.get("/{session_id}/sync")
async def sync_session_messages(
    session_id: str,
    state: StateDep,
    limit: int | None = None,
) -> list[MessageWithParts]:
    """Alias for ``GET /{session_id}/message``.

    The OpenCode TUI (v1.18+) requests ``/session/{id}/sync`` to load
    conversation history.  Without this route, the request falls through
    to the catch-all web-UI proxy (which forwards to ``app.opencode.ai``)
    and the TUI receives empty cloud data instead of local session
    history.

    This route simply delegates to :func:`get_session_messages` so both
    ``/message`` and ``/sync`` return identical data.
    """
    return await get_session_messages(session_id, state, limit)


@router.get("/{session_id}/children")
async def get_session_children(
    session_id: str,
    state: StateDep,
) -> list[Session]:
    """Get all child sessions for a given session.

    Returns a list of sessions where parent_id matches the provided session_id.
    Queries both memory cache and database for complete results.
    """
    children: list[Session] = []
    seen_ids: set[str] = set()

    # Check all cached sessions first
    for session in state.sessions.values():
        if session.parent_id == session_id:
            children.append(session)
            seen_ids.add(session.id)

    # Query database for child sessions not in memory
    try:
        session_pool = state.pool.session_pool
        if session_pool is not None:
            child_ids = session_pool.sessions.get_children(session_id)
            for child_id in child_ids:
                if child_id not in seen_ids:
                    child_session = await get_or_load_session(state, child_id)
                    if child_session:
                        children.append(child_session)
                        seen_ids.add(child_id)
    except Exception:  # noqa: BLE001
        # Graceful fallback if store doesn't support list_sessions or query fails
        pass

    # Sort by creation time (ascending) so left/right navigation in the
    # OpenCode UI shows child sessions in creation order.
    children.sort(key=lambda s: s.time.created)

    return children


@router.patch("/{session_id}")
async def update_session(
    session_id: str,
    request: SessionUpdateRequest,
    state: StateDep,
) -> Session:
    """Update session properties and persist changes.

    Supports updating title and archiving via time.archived.
    """
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    updates: dict[str, Any] = {}
    if request.title is not None:
        updates["title"] = request.title
    # Always update the 'updated' timestamp
    updates["time"] = TimeCreatedUpdated(created=session.time.created, updated=now_ms())
    session = session.model_copy(update=updates)
    state.sessions[session_id] = session  # Update cache
    id_ = state.pool.manifest.config_file_path
    session_data = opencode_to_session_data(session, agent_name=state.agent.name, pool_id=id_)
    if state.pool.session_pool and state.pool.session_pool.sessions.store:
        await state.pool.session_pool.sessions.store.save_session(session_data)
    await state.broadcast_event(SessionUpdatedEvent.create(session))
    return session


@router.delete("/{session_id}")
async def delete_session(session_id: str, state: StateDep) -> bool:
    """Delete a session from both cache and storage."""
    # Check if session exists (in cache or storage)
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Remove from cache
    state.sessions.pop(session_id, None)
    state.reverted_messages.pop(session_id, None)
    # Delegate session cleanup to OpenCodeSessionPoolIntegration
    integration = state.session_pool_integration
    if integration is not None:
        await integration.close_session(session_id)
    # Ensure store delete if close_session did not handle it
    session_pool = state.pool.session_pool
    if session_pool is not None and session_pool.sessions.store is not None:
        await session_pool.sessions.store.delete_session(session_id)
    await state.broadcast_event(SessionDeletedEvent.create(session_id))
    return True


@router.post("/{session_id}/abort")
async def abort_session(session_id: str, state: StateDep) -> bool:
    """Abort a running session by interrupting the agent."""
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Cancel pending questions FIRST so the agent doesn't resume
    # after the user answers a question that was already in-flight.
    state.cancel_session_pending_questions(session_id)

    # Use SessionPool-based agent-aware abort when a SessionController is
    # available.  For native (per-session) agents we call interrupt() on the
    # dedicated agent instance.  For non-native shared agents we only cancel
    # the RunHandle so we don't kill the shared agent for all sessions.
    sp_session = None
    if state.session_controller is not None:
        sp_session = state.session_controller.get_session(session_id)

    if sp_session is not None:
        # Native agents: interrupt the per-session agent.
        # interrupt() internally calls cancel_run_for_session() which
        # calls run_handle.cancel() — so we MUST NOT call
        # session_pool.cancel_run() again for per-session agents.
        # A double cancel() kills the start() generator because the
        # second cancel throws CancelledError at _idle_event.wait()
        # inside _idle_loop(), which is OUTSIDE the except handler.
        if sp_session.is_per_session_agent and state.session_controller is not None:
            session_agent = state.session_controller.get_session_agent(session_id)
            if session_agent is not None:
                try:
                    await session_agent.interrupt()
                    # Give a moment for the cancellation to propagate
                    await asyncio.sleep(0.1)
                except Exception:
                    logger.warning(
                        "interrupt() failed for session %s during abort",
                        session_id,
                        exc_info=True,
                    )
        # Non-per-session (shared) agents: can't interrupt the shared
        # agent, so only cancel the RunHandle.
        elif sp_session.current_run_id is not None:
            session_pool = state.pool.session_pool
            if session_pool is not None:
                with contextlib.suppress(ValueError):
                    session_pool.cancel_run(sp_session.current_run_id)
    else:
        # Fallback: legacy behavior when SessionController is unavailable
        session_pool = state.pool.session_pool
        if session_pool is not None:
            session_pool.sessions.cancel_run_for_session(session_id)

        # Interrupt the shared server agent to cancel any ongoing stream
        try:
            await state.agent.interrupt()
            # Give a moment for the cancellation to propagate
            await asyncio.sleep(0.1)
        except Exception:  # noqa: BLE001
            pass

    # Re-cancel pending questions after interrupt to catch any questions
    # that were created AFTER the initial cancel but BEFORE the interrupt
    # took effect (TOCTOU race — the agent may ask questions between
    # cancel_session_pending_questions and interrupt taking effect).
    state.cancel_session_pending_questions(session_id)

    # Update and broadcast session status to notify clients
    await set_session_status(state, session_id, SessionStatus(type="idle"))
    await state.broadcast_event(SessionStatusEvent.create(session_id, SessionStatus(type="idle")))
    await state.broadcast_event(SessionIdleEvent.create(session_id))
    return True


@router.post("/{session_id}/fork")
async def fork_session(  # noqa: D417
    session_id: str,
    state: StateDep,
    request: SessionForkRequest | None = None,
    directory: str | None = None,
) -> Session:
    """Fork a session, optionally at a specific message.

    Creates a new session with:
    - parent_id pointing to the original session
    - Copies all messages (or up to message_id if specified)
    - Independent conversation history from that point forward

    Args:
        session_id: The session to fork from
        request: Optional fork parameters (message_id to fork from)
        directory: Optional directory for the forked session

    Returns:
        The newly created forked session
    """
    # Get the original session
    original_session = await get_or_load_session(state, session_id)
    if original_session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get messages from the original session
    original_messages = await _get_session_messages_from_pool(state, session_id)
    messages_to_copy: list[MessageWithParts] = []
    if request and request.message_id:
        for msg in original_messages:
            messages_to_copy.append(msg)
            if msg.info.id == request.message_id:
                break
        else:
            detail = f"Message {request.message_id} not found in session"
            raise HTTPException(status_code=404, detail=detail)
    else:
        messages_to_copy = list(original_messages)

    # Create the new forked session
    new_session_id = identifier.descending("session")
    now = extract_timestamp_ms(new_session_id) or now_ms()
    # Use provided directory or inherit from original session
    fork_directory = directory if directory else original_session.directory
    forked_session = Session(
        id=new_session_id,
        project_id=original_session.project_id,
        directory=fork_directory,
        title=f"{original_session.title} (fork)",
        version="1",
        time=TimeCreatedUpdated(created=now, updated=now),
        parent_id=session_id,  # Link to original session
    )

    # Delegate forked session creation to SessionPool
    session_pool = state.pool.session_pool
    if session_pool is not None:
        try:
            await session_pool.create_session(
                session_id=new_session_id,
                agent_name=state.agent.name,
                parent_session_id=session_id,
                project_id=original_session.project_id,
                cwd=fork_directory,
                title=forked_session.title,
            )
        except Exception:
            logger.exception(
                "SessionPool forked session creation failed, falling back to in-memory",
                session_id=new_session_id,
            )

    # Copy messages in storage via SessionPool
    if session_pool is not None:
        with contextlib.suppress(KeyError, TypeError):
            await session_pool.copy_messages(
                session_id,
                new_session_id,
                up_to_message_id=request.message_id if request else None,
            )

    # Cache in memory
    state.sessions[new_session_id] = forked_session
    await state.mark_session_idle(new_session_id)
    # Copy messages to the new session (with updated session_id references)
    copied_messages: list[MessageWithParts] = []
    for msg_with_parts in messages_to_copy:
        new_info = msg_with_parts.info.model_copy(update={"session_id": new_session_id})
        new_parts = [
            part.model_copy(update={"session_id": new_session_id}) for part in msg_with_parts.parts
        ]
        copied_messages.append(MessageWithParts(info=new_info, parts=new_parts))
    if session_pool is not None:
        in_memory_messages = getattr(state, "messages", None)
        if in_memory_messages is not None:
            in_memory_messages[new_session_id] = list(copied_messages)
    else:
        for msg_with_parts in copied_messages:
            await append_message_to_session(state, new_session_id, msg_with_parts)
    if session_pool is not None:
        fork_agent = await session_pool.sessions.get_or_create_session_agent(new_session_id)
        fork_agent.conversation.chat_messages.clear()
        from wolfharness_server.opencode_server.converters import opencode_to_chat_message

        for msg_with_parts in copied_messages:
            chat_msg = opencode_to_chat_message(msg_with_parts, session_id=new_session_id)
            fork_agent.conversation.chat_messages.append(chat_msg)
    # Broadcast session created event
    await state.broadcast_event(SessionCreatedEvent.create(forked_session))
    # Also broadcast session.updated so the CLI TUI upserts the forked
    # session into its store immediately (same reason as create_session).
    await state.broadcast_event(SessionUpdatedEvent.create(forked_session))
    return forked_session


@router.post("/{session_id}/init")
async def init_session(  # noqa: D417,PLR0915
    session_id: str,
    state: StateDep,
    request: SessionInitRequest | None = None,
) -> bool:
    """Initialize a session by analyzing the codebase and creating AGENTS.md.

    Generates a repository map, reads README if present, and runs the agent
    with a prompt to create an AGENTS.md file with project-specific context.

    Args:
        session_id: The session to initialize
        request: Optional model/provider override for the init task

    Returns:
        True when the init task has been started (runs async)
    """
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    fs = state.fs
    working_dir = state.working_dir
    try:
        all_files = await find_src_files(fs, working_dir)
        repo_map = RepoMap(fs=fs, root_path=working_dir, max_tokens=4000)
        repomap_content = await repo_map.get_map(all_files) or "No repository map generated."
    except Exception as e:  # noqa: BLE001
        repomap_content = f"Error generating repository map: {e}"

    # Try to read README.md
    readme_content = ""
    for readme_name in ["README.md", "readme.md", "README", "readme.txt"]:
        try:
            readme_path = f"{working_dir}/{readme_name}".replace("//", "/")
            content = await fs._cat_file(readme_path)
            readme_content = content.decode("utf-8") if isinstance(content, bytes) else content
            break
        except Exception:  # noqa: BLE001
            continue

    # Build the init prompt
    prompt_parts = [
        "Please analyze this codebase and create an AGENTS.md file in the project root.",
        "",
        "<repository-structure>",
        repomap_content,
        "</repository-structure>",
    ]
    if readme_content:
        prompt_parts.extend(["", "<readme>", readme_content, "</readme>"])
    prompt_parts.extend([
        "",
        "Include:",
        "1. Build/lint/test commands - especially for running a single test",
        "2. Code style guidelines (imports, formatting, types, naming conventions, error handling)",
        "",
        (
            "The file will be given to AI coding agents working in this repository. "
            "Keep it around 150 lines."
        ),
        "",
        (
            "If there are existing rules (.cursor/rules/, .cursorrules, "
            ".github/copilot-instructions.md), incorporate them."
        ),
    ])

    init_prompt = "\n".join(prompt_parts)

    session_pool = state.pool_or_none.session_pool if state.pool_or_none is not None else None
    if session_pool is not None:
        # Get or create agent and optionally set model before fire-and-forget
        agent = state.agent
        if request and request.model_id and request.provider_id:
            requested_model = f"{request.provider_id}:{request.model_id}"
            try:
                available_models = await agent.get_available_models()
                if available_models:
                    valid_ids = [m.id_override if m.id_override else m.id for m in available_models]
                    if requested_model in valid_ids:
                        await agent.set_model(requested_model)
            except Exception:  # noqa: BLE001
                pass

        # Fire-and-forget through SessionPool; RunHandle is stored
        # in SessionController._runs for cancellation tracking.
        await session_pool.send_message(session_id, init_prompt)
        return True

    # Fallback: run the agent in the background directly
    async def run_init() -> None:
        agent = state.agent
        try:
            if request and request.model_id and request.provider_id:
                requested_model = f"{request.provider_id}:{request.model_id}"
                try:
                    available_models = await agent.get_available_models()
                    if available_models:
                        valid_ids = [
                            m.id_override if m.id_override else m.id for m in available_models
                        ]
                        if requested_model in valid_ids:
                            await agent.set_model(requested_model)
                except Exception:  # noqa: BLE001
                    pass

            await agent.run(init_prompt)
        finally:
            # Per-session agent: model changes are session-local, no need
            # to restore the original model.
            pass

    state.create_background_task(run_init(), name=f"init_{session_id}")

    return True


@router.get("/{session_id}/todo")
async def get_session_todos(session_id: str, state: StateDep) -> list[Todo]:
    """Get todos for a session.

    Returns todos from the agent pool's TodoTracker.
    """
    # Fast path for subagent/child sessions already in memory:
    # Skip get_or_load_session (which may load from storage) because the
    # parent agent is streaming and subagent parts are in memory.
    cached_session = state.sessions.get(session_id)
    if cached_session is not None and cached_session.parent_id is not None:
        tracker = state.pool.todos
        return build_opencode_todos(tracker, Todo)

    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get todos from pool's TodoTracker
    tracker = state.pool.todos
    return build_opencode_todos(tracker, Todo)


@router.get("/{session_id}/diff")
async def get_session_diff(
    session_id: str,
    state: StateDep,
    message_id: str | None = None,
) -> list[FileDiff]:
    """Get file diffs for a session.

    Returns a list of file changes with unified diffs.
    Optionally filter to changes since a specific message.
    """
    # Fast path for subagent/child sessions already in memory:
    # Skip get_or_load_session (which may load from storage) because the
    # parent agent is streaming and subagent parts are in memory.
    cached_session = state.sessions.get(session_id)
    if cached_session is not None and cached_session.parent_id is not None:
        file_ops = state.pool.file_ops
        if not file_ops.changes:
            return []
        changes = file_ops.get_changes_since(message_id) if message_id else file_ops.changes
        return [FileDiff.from_file_change(change) for change in changes]

    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    file_ops = state.pool.file_ops
    if not file_ops.changes:
        return []
    # Optionally filter by message_id
    changes = file_ops.get_changes_since(message_id) if message_id else file_ops.changes
    return [FileDiff.from_file_change(change) for change in changes]


@router.post("/{session_id}/shell")
async def run_shell_command(
    session_id: str,
    request: ShellRequest,
    state: StateDep,
) -> MessageWithParts:
    """Run a shell command directly."""
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Validate command for security issues
    validate_command(request.command, state.working_dir)
    now = now_ms()
    # Create assistant message for the shell output
    assistant_msg_id = identifier.ascending("message")
    assistant_message = AssistantMessage(
        id=assistant_msg_id,
        session_id=session_id,
        parent_id="",  # Shell commands don't have a parent user message
        model_id=request.model.model_id if request.model else "shell",  # ty: ignore[invalid-argument-type]
        provider_id=request.model.provider_id if request.model else "local",  # ty: ignore[invalid-argument-type]
        mode="shell",
        agent=request.agent,
        path=MessagePath(cwd=state.working_dir, root=state.working_dir),
        time=MessageTime(created=now),
    )

    # Initialize message with empty parts
    assistant_msg_with_parts = MessageWithParts(info=assistant_message, parts=[])
    await append_message_to_session(state, session_id, assistant_msg_with_parts)
    # Broadcast message created
    await state.broadcast_event(MessageUpdatedEvent.create(assistant_message))
    try:
        # Mark session as busy
        await set_session_status(state, session_id, SessionStatus(type="busy"))
        await state.broadcast_event(
            SessionStatusEvent.create(session_id, SessionStatus(type="busy"))
        )
        # Add step-start part
        part_id = identifier.ascending("part")
        step_start = StepStartPart(id=part_id, message_id=assistant_msg_id, session_id=session_id)
        assistant_msg_with_parts.parts.append(step_start)
        await state.broadcast_event(PartUpdatedEvent.create(step_start))
        # Execute the command via standalone shell_env (not agent.env)
        output_text = ""
        success = False
        try:
            result = await state.shell_env.execute_command(request.command)
            success = result.success
            if success:
                output_text = str(result.result) if result.result else ""
            else:
                output_text = f"Error: {result.error}" if result.error else "Command failed"
        except Exception as e:  # noqa: BLE001
            output_text = f"Error executing command: {e}"

        response_time = now_ms()
        # Create text part with output
        text_part = TextPart(
            id=identifier.ascending("part"),
            message_id=assistant_msg_id,
            session_id=session_id,
            text=f"$ {request.command}\n{output_text}",
        )
        assistant_msg_with_parts.parts.append(text_part)
        await state.broadcast_event(PartUpdatedEvent.create(text_part))
        part_id = identifier.ascending("part")
        step_finish = StepFinishPart(id=part_id, message_id=assistant_msg_id, session_id=session_id)
        assistant_msg_with_parts.parts.append(step_finish)
        await state.broadcast_event(PartUpdatedEvent.create(step_finish))
        # Update message with completion time
        time_ = MessageTime(created=now, completed=response_time)
        updated_assistant = assistant_message.model_copy(update={"time": time_})
        assistant_msg_with_parts.info = updated_assistant
        await state.broadcast_event(MessageUpdatedEvent.create(updated_assistant))
    finally:
        await state.mark_session_idle(session_id)
    return assistant_msg_with_parts


@router.get("/{session_id}/permissions")
async def get_pending_permissions(
    session_id: str, state: StateDep
) -> list[PermissionAskedProperties]:
    """Get all pending permission requests for a session.

    Returns a list of pending permissions awaiting user response.
    """
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get the input provider for this session
    input_provider: OpenCodeInputProvider | None = None
    if state.session_controller is not None:
        sp_session = state.session_controller.get_session(session_id)
        if sp_session is not None:
            input_provider = sp_session.input_provider
    if input_provider is None:
        return []

    return input_provider.get_pending_permissions()


@router.post("/{session_id}/permissions/{permission_id}")
async def respond_to_permission(
    session_id: str,
    permission_id: str,
    body: PermissionReplyRequest,
    state: StateDep,
) -> bool:
    """Respond to a pending permission request.

    The response can be:
    - "once": Allow this tool execution once
    - "always": Always allow this tool (remembered for session)
    - "reject": Reject this tool execution
    """
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get the input provider for this session
    input_provider = None
    if state.session_controller is not None:
        sp_session = state.session_controller.get_session(session_id)
        if sp_session is not None:
            input_provider = sp_session.input_provider
    if input_provider is None:
        raise HTTPException(status_code=404, detail="No input provider for session")

    # Resolve the permission
    resolved = input_provider.resolve_permission(permission_id, body.reply)
    if not resolved:
        raise HTTPException(status_code=404, detail="Permission not found or already resolved")
    event = PermissionResolvedEvent.create(
        session_id=session_id,
        request_id=permission_id,
        reply=body.reply,
    )
    await state.broadcast_event(event)
    return True


# OpenCode-style continuation prompt for summarization
SUMMARIZE_PROMPT = """Provide a detailed prompt for continuing our conversation above. Focus on information that would be helpful for continuing the conversation, including what we did, what we're doing, which files we're working on, and what we're going to do next considering new session will not have access to our conversation."""  # noqa: E501


@router.post("/{session_id}/summarize")
async def summarize_session(  # noqa: PLR0915
    session_id: str,
    state: StateDep,
    request: SummarizeRequest | None = None,
) -> bool:
    """Summarize the session conversation.

    First runs the compaction pipeline to condense older messages,
    then streams an LLM-generated summary/continuation prompt to the user.
    The summary message is marked with summary=true for UI display.

    Returns:
        True if summarization completed successfully.
    """
    from pydantic_ai.messages import (
        PartDeltaEvent as PydanticPartDeltaEvent,
        PartStartEvent,
        TextPart as PydanticTextPart,
        TextPartDelta,
    )

    from wolfharness.agents.events import StreamCompleteEvent, UserMessageInsertedEvent
    from wolfharness.messaging.compaction import compact_conversation, summarizing_context

    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if not await get_messages_for_session(state, session_id):
        raise HTTPException(status_code=400, detail="No messages to summarize")

    # Route-level lock: serialize summarization for this session.
    # Summarization has two phases (stream LLM + compaction) that must
    # not interleave with other operations on the same session.
    # Lock ordering: route-level lock first, then turn_lock.
    async with state.get_session_lock(session_id):
        # Determine model to use
        default_model_id, default_provider_id = state.resolve_default_model_info()
        model_id = request.model_id if request and request.model_id else default_model_id
        provider_id = (
            request.provider_id if request and request.provider_id else default_provider_id
        )

        now = now_ms()
        # Create assistant message for the summary (marked with summary=true)
        assistant_msg_id = identifier.ascending("message")
        assistant_message = AssistantMessage(
            id=assistant_msg_id,
            session_id=session_id,
            parent_id="",
            model_id=model_id,
            provider_id=provider_id,
            mode="summarize",
            agent="summarizer",
            path=MessagePath(cwd=state.working_dir, root=state.working_dir),
            time=MessageTime(created=now),
            summary=True,  # Mark as summary message
        )

        assistant_msg_with_parts = MessageWithParts(info=assistant_message, parts=[])
        await append_message_to_session(state, session_id, assistant_msg_with_parts)
        # Broadcast message created
        await state.broadcast_event(MessageUpdatedEvent.create(assistant_message))
        try:
            # Mark session as busy
            await set_session_status(state, session_id, SessionStatus(type="busy"))
            await state.broadcast_event(
                SessionStatusEvent.create(session_id, SessionStatus(type="busy"))
            )
            # Add step-start part
            part_id = identifier.ascending("part")
            step_start = StepStartPart(
                id=part_id,
                message_id=assistant_msg_id,
                session_id=session_id,
            )
            assistant_msg_with_parts.parts.append(step_start)
            await state.broadcast_event(PartUpdatedEvent.create(step_start))
            # Step 1: Stream LLM summary generation FIRST (while we have full history)
            # The LLM sees the complete conversation and generates a continuation prompt.
            response_text = ""
            usage = None
            cost = 0.0
            text_part: TextPart | None = None
            session_pool = state.pool.session_pool
            if session_pool is None:
                msg = "SessionPool is not available"
                raise RuntimeError(msg)
            try:
                stream = session_pool.run_stream(
                    session_id,
                    SUMMARIZE_PROMPT,
                    scope="session",
                    message_id=assistant_msg_id,
                )
                async for event in stream:
                    match event:
                        # Text streaming start
                        case PartStartEvent(part=PydanticTextPart(content=delta)):
                            response_text = delta
                            text_part = TextPart(
                                id=identifier.ascending("part"),
                                message_id=assistant_msg_id,
                                session_id=session_id,
                                text=delta,
                            )
                            assistant_msg_with_parts.parts.append(text_part)
                            await state.broadcast_event(PartUpdatedEvent.create(text_part))

                        # Text streaming delta
                        case PydanticPartDeltaEvent(delta=TextPartDelta(content_delta=delta)) if (
                            delta
                        ):
                            response_text += delta
                            if text_part is not None:
                                text_part = TextPart(
                                    id=text_part.id,
                                    message_id=assistant_msg_id,
                                    session_id=session_id,
                                    text=response_text,
                                )
                                # Update in parts list
                                for i, p in enumerate(assistant_msg_with_parts.parts):
                                    if isinstance(p, TextPart) and p.id == text_part.id:
                                        assistant_msg_with_parts.parts[i] = text_part
                                        break
                                await state.broadcast_event(
                                    PartDeltaEvent.create(
                                        session_id=session_id,
                                        message_id=assistant_msg_id,
                                        part_id=text_part.id,
                                        delta=delta,
                                    )
                                )

                        # Stream complete - extract token usage
                        case StreamCompleteEvent(message=complete_msg) if (
                            complete_msg and complete_msg.usage
                        ):
                            usage = complete_msg.usage
                            cost = (
                                float(complete_msg.cost_info.total_cost)
                                if complete_msg.cost_info
                                else 0
                            )

                        case UserMessageInsertedEvent():
                            pass  # User message insertions not relevant to summary generation

            except Exception as e:  # noqa: BLE001
                response_text = f"Error generating summary: {e}"
            finally:
                # Post-stream cleanup: compact conversation.
                # This runs in finally so compaction always occurs after streaming,
                # even if the stream raised an exception.
                try:
                    agent = state.agent
                    pipeline = None
                    if agent.host_context is not None:
                        pipeline = agent.host_context.manifest.get_compaction_pipeline()
                    if pipeline is None:
                        pipeline = summarizing_context()

                    await compact_conversation(pipeline, agent.conversation)
                    if state.storage is not None:
                        compacted_history = agent.conversation.get_history()
                        await state.storage.replace_conversation_messages(
                            session_id, compacted_history
                        )
                    await set_messages_for_session(state, session_id, [assistant_msg_with_parts])
                except Exception:  # noqa: BLE001
                    # Compaction failure is not fatal - we still have the summary
                    pass

            response_time = now_ms()
            # Create/update text part with final response
            if text_part is None:
                text_part = TextPart(
                    id=identifier.ascending("part"),
                    message_id=assistant_msg_id,
                    session_id=session_id,
                    text=response_text,
                )
                assistant_msg_with_parts.parts.append(text_part)
                await state.broadcast_event(PartUpdatedEvent.create(text_part))

            tokens = Tokens.from_pydantic_ai(usage) if usage else Tokens()
            # Add step-finish part
            step_finish = StepFinishPart(
                id=identifier.ascending("part"),
                message_id=assistant_msg_id,
                session_id=session_id,
                tokens=tokens,
                cost=cost,
            )
            assistant_msg_with_parts.parts.append(step_finish)
            await state.broadcast_event(PartUpdatedEvent.create(step_finish))
            # Update message with completion time and tokens
            msg_time = MessageTime(created=now, completed=response_time)
            update = {"time": msg_time, "tokens": tokens, "cost": cost}
            updated_assistant = assistant_message.model_copy(update=update)
            assistant_msg_with_parts.info = updated_assistant
            await state.broadcast_event(MessageUpdatedEvent.create(updated_assistant))

            # Broadcast session.diff event after summarization
            file_ops = state.pool.file_ops
            diffs = [FileDiff.from_file_change(change) for change in file_ops.changes]
            await state.broadcast_event(SessionDiffEvent.create(session_id, diffs))
        finally:
            await state.mark_session_idle(session_id)

        return True


@router.post("/{session_id}/share")
async def share_session(
    session_id: str,
    state: StateDep,
    num_messages: int | None = None,
) -> Session:
    """Share session conversation history via OpenCode's sharing service.

    Uses the OpenCode share API to create a shareable link with the full
    conversation including messages and parts.

    Returns the updated session with the share URL.
    """
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = await get_messages_for_session(state, session_id, prefer_in_memory=False)

    if not messages:
        raise HTTPException(status_code=400, detail="No messages to share")

    # Apply message limit if specified
    if num_messages is not None and num_messages > 0:
        messages = messages[-num_messages:]
    # Convert our messages to OpenCode Message format
    opencode_messages: list[Message] = []
    for msg_with_parts in messages:
        # Extract text parts
        parts = [
            MessagePart(type="text", text=part.text)
            for part in msg_with_parts.parts
            if isinstance(part, TextPart) and part.text
        ]
        if parts:
            opencode_messages.append(Message(role=msg_with_parts.info.role, parts=parts))
    if not opencode_messages:
        raise HTTPException(status_code=400, detail="No content to share")

    # Share via OpenCode API
    async with OpenCodeSharer() as sharer:
        result = await sharer.share_conversation(opencode_messages, title=session.title)
        share_url = result.url
    # Store the share URL in the session
    share_info = SessionShare(url=share_url)
    updated_session = session.model_copy(update={"share": share_info})
    state.sessions[session_id] = updated_session
    # Broadcast session update
    await state.broadcast_event(SessionUpdatedEvent.create(updated_session))
    return updated_session


_IDLE_WAIT_TIMEOUT: float = 10.0
"""Maximum seconds to wait for a running turn to finish after requesting cancel."""


async def _ensure_session_idle(state: ServerState, session_id: str) -> None:
    """Ensure a session is idle before proceeding with STAGE/CLEAR operations.

    If the session has an active run (``current_run_id`` is set), this helper:
    1. Cancels the running turn via ``cancel_run``.
    2. Waits up to :data:`_IDLE_WAIT_TIMEOUT` seconds for the run to complete.
    3. Broadcasts a ``SessionStatusEvent`` with status ``"idle"`` on success.
    4. On timeout or cancel failure, logs a warning and force-clears
       ``current_run_id`` so the caller can proceed.

    If the session is already idle (``current_run_id`` is ``None``), this
    is a no-op.
    """
    if state.session_controller is None:
        return

    session_state = state.session_controller.get_session(session_id)
    if session_state is None:
        return

    run_id = session_state.current_run_id
    if run_id is None:
        return

    session_pool = state.pool.session_pool

    # Step 1: Request cancellation of the running turn.
    cancel_succeeded = False
    if session_pool is not None:
        try:
            session_pool.cancel_run(run_id)
            cancel_succeeded = True
        except Exception:
            logger.warning(
                "cancel_run(%s) failed for session %s — force-clearing current_run_id",
                run_id,
                session_id,
                exc_info=True,
            )

    # Step 2: Wait for the run to complete (only if cancel was initiated).
    if cancel_succeeded and session_pool is not None:
        try:
            await asyncio.wait_for(
                session_pool.wait_for_completion(session_id),
                timeout=_IDLE_WAIT_TIMEOUT,
            )
        except TimeoutError:
            logger.warning(
                "Run %s for session %s did not complete within %.1fs"
                " — force-clearing current_run_id",
                run_id,
                session_id,
                _IDLE_WAIT_TIMEOUT,
            )
        except Exception:
            logger.warning(
                "wait_for_completion(%s) raised for session %s — force-clearing current_run_id",
                run_id,
                session_id,
                exc_info=True,
            )

    # Step 3: Broadcast idle status so clients update their UI.
    idle_status = SessionStatus(type="idle")
    await state.broadcast_event(SessionStatusEvent.create(session_id, idle_status))

    # Step 4: Always force-clear current_run_id so the caller can proceed.
    session_state.current_run_id = None


class RevertRequest(OpenCodeBaseModel):
    """Request body for reverting a message."""

    message_id: str
    part_id: str | None = None


@router.post("/{session_id}/revert")
async def revert_session(session_id: str, request: RevertRequest, state: StateDep) -> Session:  # noqa: PLR0915
    """STAGE: set a revert marker, roll back files, and broadcast hide events.

    This is the soft-marker model — messages are NOT deleted from the database
    or in-memory cache. Instead:

    1. Ensure the session is idle (auto-cancel any active run).
    2. Clear queued prompts and feedback so the reverted state is stable.
    3. Roll back file changes to before the specified message.
    4. Set a ``SessionRevert`` marker on the session (persisted via metadata).
    5. Broadcast ``message.removed`` / ``part.removed`` events so the frontend
       soft-hides the reverted messages without deleting them.

    If a previous STAGE is already in effect (``session.revert`` is set),
    the old ``FileOpsTracker.reverted_changes`` are cleared before the new
    STAGE overwrites the marker.
    """
    from wolfharness_server.opencode_server.models import MessageRemovedEvent, PartRemovedEvent

    # 3.1: Ensure session is idle before proceeding.
    await _ensure_session_idle(state, session_id)

    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # 3.2: Clear prompt and feedback queues AFTER idle check, BEFORE setting marker.
    if state.session_controller is not None:
        session_state = state.session_controller.get_session(session_id)
        if session_state is not None:
            while True:
                try:
                    session_state.prompt_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            while True:
                try:
                    session_state.feedback_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

    # Get ALL messages for this session, bypassing the revert filter.
    # We need the full unfiltered list so we can find any message ID, even
    # ones past the current revert point (for double-STAGE support).
    # state.messages is the in-memory cache that stores all messages without
    # any revert filtering.
    messages: list[MessageWithParts] = state.messages.get(session_id, [])
    if not messages:
        # Fall back to get_messages_for_session if state.messages is empty
        # (e.g., cold-start scenario where messages are only in the DB).
        messages = await get_messages_for_session(state, session_id)
    if not messages:
        raise HTTPException(status_code=400, detail="No messages to revert")

    # Find the revert message index
    revert_index = None
    for i, msg in enumerate(messages):
        if msg.info.id == request.message_id:
            revert_index = i
            break

    if revert_index is None:
        raise HTTPException(status_code=404, detail=f"Message {request.message_id} not found")

    # Messages from the revert point onwards are soft-hidden (not deleted).
    messages_to_hide = messages[revert_index:]

    if not messages_to_hide:
        raise HTTPException(status_code=400, detail="No messages to revert")

    # 3.9: Double-STAGE handling — if a previous STAGE is in effect, clear
    # the old reverted_changes before proceeding with the new STAGE.
    file_ops = state.pool.file_ops
    if session.revert is not None:
        file_ops.reverted_changes.clear()

    # 3.7: Broadcast message.removed and part.removed events for soft-hiding.
    # These tell the frontend to hide the messages, but they remain in the DB.
    for msg in messages_to_hide:
        await state.broadcast_event(MessageRemovedEvent.create(session_id, msg.info.id))
        for part in msg.parts:
            await state.broadcast_event(PartRemovedEvent.create(session_id, msg.info.id, part.id))

    # 3.5: Roll back file changes (kept from original implementation).
    if file_ops.changes and (
        revert_ops := file_ops.get_revert_operations(since_message_id=request.message_id)
    ):
        for path, content in revert_ops:
            try:
                if content is None:
                    await state.fs._rm_file(path)
                else:
                    content_bytes = content.encode("utf-8")
                    await state.fs._pipe_file(path, content_bytes)
            except Exception as e:
                detail = f"Failed to revert {path}: {e}"
                raise HTTPException(status_code=500, detail=detail) from e
        file_ops.remove_changes_since(request.message_id)

    # 3.6: Set the revert marker on the session and persist via metadata.
    session = state.sessions[session_id]
    revert_info = SessionRevert(message_id=request.message_id, part_id=request.part_id)
    updated_session = session.model_copy(update={"revert": revert_info})
    state.sessions[session_id] = updated_session

    # Persist the session (including the revert marker in metadata) via the
    # session store, if available.
    session_pool = state.pool.session_pool
    if session_pool is not None and session_pool.sessions.store is not None:
        pool_id = state.pool.manifest.config_file_path
        session_data = opencode_to_session_data(
            updated_session,
            agent_name=state.agent.name,
            pool_id=pool_id,
        )
        await session_pool.sessions.store.save_session(session_data)

    # Broadcast session update
    await state.broadcast_event(SessionUpdatedEvent.create(updated_session))

    # Broadcast session.diff event with current file diffs
    diffs = [FileDiff.from_file_change(change) for change in file_ops.changes]
    await state.broadcast_event(SessionDiffEvent.create(session_id, diffs))

    return updated_session


@router.post("/{session_id}/unrevert")
async def unrevert_session(session_id: str, state: StateDep) -> Session:
    """CLEAR: clear the revert marker and restore file changes.

    This is the soft-marker model — messages were never deleted, so CLEAR
    simply:

    1. Ensure the session is idle (auto-cancel any active run).
    2. Restore file changes that were rolled back during STAGE.
    3. Clear the ``SessionRevert`` marker from the session.
    4. Broadcast a ``SessionUpdatedEvent`` so the frontend unhides messages.

    No message restoration is needed — messages remained in ``state.messages``
    and the database throughout the STAGE.
    """
    # 4.1: Ensure session is idle before proceeding.
    await _ensure_session_idle(state, session_id)

    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # 4.2 / 4.8: If there's no revert marker, there's nothing to clear.
    if session.revert is None:
        raise HTTPException(status_code=400, detail="No reverted messages to restore")

    # 4.3: Restore file changes that were rolled back during STAGE.
    file_ops = state.pool.file_ops
    if file_ops.reverted_changes:
        unrevert_ops = file_ops.get_unrevert_operations()
        for path, content in unrevert_ops:
            try:
                if content is None:
                    await state.fs._rm_file(path)
                else:
                    content_bytes = content.encode("utf-8")
                    await state.fs._pipe_file(path, content_bytes)
            except Exception as e:
                detail = f"Failed to unrevert {path}: {e}"
                raise HTTPException(status_code=500, detail=detail) from e
        file_ops.restore_reverted_changes()

    # 4.4: Clear the revert marker and broadcast session update.
    updated_session = session.model_copy(update={"revert": None})
    state.sessions[session_id] = updated_session

    # Persist the cleared marker via the session store.
    session_pool = state.pool.session_pool
    if session_pool is not None and session_pool.sessions.store is not None:
        pool_id = state.pool.manifest.config_file_path
        session_data = opencode_to_session_data(
            updated_session,
            agent_name=state.agent.name,
            pool_id=pool_id,
        )
        await session_pool.sessions.store.save_session(session_data)

    await state.broadcast_event(SessionUpdatedEvent.create(updated_session))
    return updated_session


@router.delete("/{session_id}/share")
async def unshare_session(session_id: str, state: StateDep) -> bool:
    """Remove share link from a session.

    Note: This only removes the link from the session metadata.
    The shared content may still exist on the provider's servers.
    """
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.share is None:
        raise HTTPException(status_code=400, detail="Session is not shared")
    # Remove share info from session
    updated_session = session.model_copy(update={"share": None})
    state.sessions[session_id] = updated_session
    # Broadcast session update
    await state.broadcast_event(SessionUpdatedEvent.create(updated_session))
    return True


@router.post("/{session_id}/command")
async def execute_command(  # noqa: PLR0915
    session_id: str,
    request: CommandRequest,
    state: StateDep,
) -> MessageWithParts:
    """Execute a slash command (CommandStore or MCP prompt).

    Commands are resolved in order: first checked against CommandStore (for
    slashed/skill commands), then against MCP prompts. This provides unified
    command execution across both systems.
    """
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Route-level lock: serialize all command execution for this session.
    # Lock ordering: route-level lock first, then turn_lock (if acquired
    # internally by SessionPool). Always acquire in this order to prevent
    # deadlock.
    async with state.get_session_lock(session_id):
        # Check CommandStore first (slashed commands take priority)
        cmd = state.command_store.get_command(request.command) if state.command_store else None
        if cmd is not None:
            # Check for collision with MCP prompts
            session_agent = state.agent
            prompts = await session_agent.list_prompts()
            if any(p.name == request.command for p in prompts):
                logger.warning(
                    "Both slashed command and prompt exist for '%s'. Using slashed command.",
                    request.command,
                )
            # Route skill commands (category == "skill") through the normal
            # prompt lifecycle via _execute_skill_command (send_message →
            # EventBus-only). Non-skill commands continue to
            # _execute_slashed_command (existing behavior preserved).
            if cmd.category == "skill":
                return await _execute_skill_command(state, session_id, request)
            return await _execute_slashed_command(state, session_id, request)

        # Fall back to MCP prompts (existing code remains unchanged)
        session_agent = state.agent
        prompts = await session_agent.list_prompts()
        # Find matching prompt by name
        prompt = next((p for p in prompts if p.name == request.command), None)
        if prompt is None:
            detail = f"Command not found: {request.command}"
            raise HTTPException(status_code=404, detail=detail)

        # Parse arguments - OpenCode uses $1, $2 style, MCP uses named arguments
        # For simplicity, we'll pass the raw arguments string to the first argument
        # or parse space-separated args into a dict
        arguments: dict[str, str] = {}
        if request.arguments and prompt.arguments:
            # Split arguments and map to prompt argument names
            arg_values = request.arguments.split()
            for i, arg_def in enumerate(prompt.arguments):
                if i < len(arg_values):
                    arguments[arg_def["name"]] = arg_values[i]

        now = now_ms()
        # Create assistant message
        # D14: Use request.message_id if provided for end-to-end ID consistency.
        assistant_msg_id = identifier.ascending("message", request.message_id)
        assistant_message = AssistantMessage(
            id=assistant_msg_id,
            session_id=session_id,
            parent_id="",
            model_id=request.model or "default",
            provider_id="mcp",
            mode="command",
            agent=request.agent or "default",
            path=MessagePath(cwd=state.working_dir, root=state.working_dir),
            time=MessageTime(created=now),
        )
        assistant_msg_with_parts = MessageWithParts(info=assistant_message, parts=[])
        await append_message_to_session(state, session_id, assistant_msg_with_parts)
        await state.broadcast_event(MessageUpdatedEvent.create(assistant_message))
        try:
            # Mark session as busy
            await set_session_status(state, session_id, SessionStatus(type="busy"))
            await state.broadcast_event(
                SessionStatusEvent.create(session_id, SessionStatus(type="busy"))
            )
            # Add step-start part
            part_id = identifier.ascending("part")
            step_start = StepStartPart(
                id=part_id,
                message_id=assistant_msg_id,
                session_id=session_id,
            )
            assistant_msg_with_parts.parts.append(step_start)
            await state.broadcast_event(PartUpdatedEvent.create(step_start))

            # Get prompt content and execute through the agent
            try:
                prompt_parts = await prompt.get_components(arguments)
                # Extract text content from parts
                prompt_texts = []
                for part in prompt_parts:
                    if hasattr(part, "content"):
                        content = part.content
                        if isinstance(content, str):
                            prompt_texts.append(content)
                        elif isinstance(content, list):
                            # Handle Sequence[UserContent]
                            for item in content:
                                if isinstance(item, FileUrl):
                                    prompt_texts.append(item.url)
                                elif isinstance(item, str):
                                    prompt_texts.append(item)
                prompt_text = "\n".join(prompt_texts)

                session_pool = (
                    state.pool_or_none.session_pool if state.pool_or_none is not None else None
                )
                if session_pool is not None:
                    input_provider = state.ensure_input_provider(session_id)
                    message_id = await session_pool.send_message(
                        session_id=session_id,
                        content=prompt_text,
                        input_provider=input_provider,
                        message_id=assistant_msg_id,
                    )
                    if message_id is not None:
                        # Wait for the background run to complete before finalizing
                        try:
                            await session_pool.wait_for_completion(session_id, timeout=30.0)
                        except TimeoutError:
                            session_pool.sessions.cancel_run_for_session(session_id)
                            output_text = "Error: command execution timed out"
                        except asyncio.CancelledError:
                            session_pool.sessions.cancel_run_for_session(session_id)
                            raise
                        else:
                            output_text = ""
                    else:
                        output_text = ""
                else:
                    # Fallback to direct agent if SessionPool not available
                    result = await state.agent.run(prompt_text)
                    output_text = str(result.data)

            except Exception as e:  # noqa: BLE001
                output_text = f"Error executing command: {e}"

            response_time = now_ms()
            # Create text part with output
            text_part = TextPart(
                id=identifier.ascending("part"),
                message_id=assistant_msg_id,
                session_id=session_id,
                text=output_text,
            )
            assistant_msg_with_parts.parts.append(text_part)
            await state.broadcast_event(PartUpdatedEvent.create(text_part))
            step_finish = StepFinishPart(
                id=identifier.ascending("part"),
                message_id=assistant_msg_id,
                session_id=session_id,
            )
            assistant_msg_with_parts.parts.append(step_finish)
            await state.broadcast_event(PartUpdatedEvent.create(step_finish))
            # Update message with completion time
            time_ = MessageTime(created=now, completed=response_time)
            updated_assistant = assistant_message.model_copy(update={"time": time_})
            assistant_msg_with_parts.info = updated_assistant
            await state.broadcast_event(MessageUpdatedEvent.create(updated_assistant))
        finally:
            await state.mark_session_idle(session_id)

        # Broadcast command.executed event
        await state.broadcast_event(
            CommandExecutedEvent.create(
                name=request.command,
                session_id=session_id,
                arguments=request.arguments or "",
                message_id=assistant_msg_id,
            )
        )

        return assistant_msg_with_parts
