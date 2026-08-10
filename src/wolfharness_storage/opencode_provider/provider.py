"""OpenCode storage provider.

This module implements storage compatible with OpenCode's normalized JSON format,
storing conversations as relational data across multiple directories.

Key differences from Claude Code:
- Normalized structure (sessions → messages → parts)
- SHA1-based project IDs
- Timestamp-based message ordering (no parent links)
- In-place file updates (not append-only)

See ARCHITECTURE.md for detailed documentation of the storage format.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import anyenv
from pydantic import TypeAdapter
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from wolfharness.log import get_logger
from wolfharness.sessions.models import ProjectData, SessionData
from wolfharness.utils.pydantic_ai_helpers import safe_args_as_dict
from wolfharness.utils.thread_helpers import run_in_thread
from wolfharness.utils.time_utils import (
    datetime_to_ms,
    get_now,
    ms_to_datetime,
    parse_iso_timestamp,
)
from wolfharness_config.storage import OpenCodeStorageConfig
from wolfharness_server.opencode_server.models import (
    AssistantMessage,
    MessageInfo as OpenCodeMessage,
    Part as OpenCodePart,
    Session,
    SessionSummary,
    TimeCreatedUpdated,
)
from wolfharness_storage.base import StorageProvider
from wolfharness_storage.models import ConversationData as ConvData, TokenUsage
from wolfharness_storage.opencode_provider import helpers


if TYPE_CHECKING:
    from wolfharness.messaging import ChatMessage, TokenCost
    from wolfharness_config.session import SessionQuery
    from wolfharness_storage.models import QueryFilters, StatsFilters

logger = get_logger(__name__)

# OpenCode version we're emulating
OPENCODE_VERSION = "1.1.7"

# Type aliases - use server models directly
PartType = Literal[
    "text",
    "step-start",
    "step-finish",
    "reasoning",
    "tool",
    "patch",
    "compaction",
    "snapshot",
    "agent",
    "subtask",
    "retry",
]
ToolStatus = Literal["pending", "running", "completed", "error"]
FinishReason = Literal["stop", "tool-calls", "length", "error"]


@dataclass(slots=True)
class OpenCodeSessionMetadata:
    """Lightweight session metadata without loading messages/parts."""

    session_id: str
    path: Path
    title: str
    directory: str
    created_at: datetime
    updated_at: datetime
    project_id: str


def _read_session_metadata(session_path: Path) -> OpenCodeSessionMetadata | None:
    """Read minimal metadata from a session file without loading messages/parts.

    Args:
        session_path: Path to the session JSON file

    Returns:
        OpenCodeSessionMetadata or None if file is invalid
    """
    session = helpers.read_session(session_path)
    if session is None:
        return None

    return OpenCodeSessionMetadata(
        session_id=session.id,
        path=session_path,
        title=session.title,
        directory=session.directory,
        created_at=ms_to_datetime(session.time.created),
        updated_at=ms_to_datetime(session.time.updated),
        project_id=session.project_id,
    )


def _get_filtered_conversations_sync(
    provider: OpenCodeStorageProvider,
    *,
    agent_name: str | None,
    cutoff: datetime | None,
    query: str | None,
    model: str | None,
    limit: int | None,
    compact: bool,
    include_tokens: bool,
) -> list[ConvData]:
    """Run conversation filtering with filesystem-heavy reads off the event loop."""
    result: list[ConvData] = []
    for session_id, session_path in provider._list_sessions():
        session = helpers.read_session(session_path)
        if not session:
            continue

        session_created = ms_to_datetime(session.time.created)
        if cutoff and session_created < cutoff:
            continue

        oc_messages = provider._read_messages(session_id)
        if not oc_messages:
            continue

        msg_parts_map = {oc_msg.id: provider._read_parts(oc_msg.id) for oc_msg in oc_messages}

        chat_messages: list[ChatMessage[str]] = []
        total_tokens = 0
        for oc_msg in oc_messages:
            parts = msg_parts_map.get(oc_msg.id, [])
            chat_msg = helpers.to_chat_message(msg=oc_msg, parts=parts)
            chat_messages.append(chat_msg)

            if isinstance(oc_msg, AssistantMessage) and oc_msg.tokens:
                total_tokens += oc_msg.tokens.input + oc_msg.tokens.output

        if not chat_messages:
            continue

        if agent_name and not any(m.name == agent_name for m in chat_messages):
            continue

        if query and not any(query in m.content for m in chat_messages):
            continue

        if model and not any(
            isinstance(oc_msg, AssistantMessage) and oc_msg.model_id == model
            for oc_msg in oc_messages
        ):
            continue

        usage = TokenUsage(total=total_tokens, prompt=0, completion=0) if total_tokens else None
        filtered_messages = chat_messages
        _compact_min_messages = 2
        if compact and len(chat_messages) > _compact_min_messages:
            filtered_messages = [chat_messages[0], chat_messages[-1]]

        conv_data = ConvData(
            id=session_id,
            agent=chat_messages[0].name or "opencode",
            title=session.title,
            start_time=session_created.isoformat(),
            messages=filtered_messages,
            token_usage=usage if include_tokens else None,
        )
        result.append(conv_data)

        if limit and len(result) >= limit:
            break

    return result


class OpenCodeStorageProvider(StorageProvider):
    """Storage provider that reads/writes OpenCode's native format.

    OpenCode stores data in:
    - ~/.local/share/opencode/storage/session/{project_id}/ - Session JSON files
    - ~/.local/share/opencode/storage/message/{session_id}/ - Message JSON files
    - ~/.local/share/opencode/storage/part/{message_id}/ - Part JSON files

    Each file is a single JSON object (not JSONL).
    """

    can_load_history = True
    can_store_projects = True

    def __init__(self, config: OpenCodeStorageConfig | None = None) -> None:
        """Initialize OpenCode storage provider."""
        config = config or OpenCodeStorageConfig()
        super().__init__(config)
        path = Path(config.path).expanduser()
        # If path points to a .db file (legacy config), use the storage directory
        # instead which is where OpenCode actually stores message data
        if path.suffix == ".db":
            self.base_path = path.parent / "storage"
        else:
            self.base_path = path
        self.sessions_path = self.base_path / "session"
        self.messages_path = self.base_path / "message"
        self.parts_path = self.base_path / "part"
        self.projects_path = self.base_path / "project"
        self.projects_path.mkdir(parents=True, exist_ok=True)

    def _list_sessions(self, project_id: str | None = None) -> list[tuple[str, Path]]:
        """List all sessions, optionally filtered by project."""
        if not self.sessions_path.exists():
            return []
        sessions: list[tuple[str, Path]] = []
        if project_id:
            project_dir = self.sessions_path / project_id
            if project_dir.exists():
                sessions.extend((f.stem, f) for f in project_dir.glob("*.json"))
        else:
            for project_dir in self.sessions_path.iterdir():
                if project_dir.is_dir():
                    sessions.extend((f.stem, f) for f in project_dir.glob("*.json"))
        return sessions

    def list_session_metadata(
        self,
        project_id: str | None = None,
    ) -> list[OpenCodeSessionMetadata]:
        """List all sessions with lightweight metadata (no message/part loading).

        This is much faster than get_sessions() as it only reads session JSON files
        without loading messages or parts.

        Args:
            project_id: Optional project ID to filter by

        Returns:
            List of OpenCodeSessionMetadata objects
        """
        result: list[OpenCodeSessionMetadata] = []
        for _, session_path in self._list_sessions(project_id=project_id):
            metadata = _read_session_metadata(session_path)
            if metadata is not None:
                result.append(metadata)
        return result

    def _read_messages(self, session_id: str) -> list[OpenCodeMessage]:
        """Read all messages for a session."""
        msg_dir = self.messages_path / session_id
        if not msg_dir.exists():
            return []
        messages: list[OpenCodeMessage] = []
        adapter = TypeAdapter[OpenCodeMessage](OpenCodeMessage)
        for msg_file in sorted(msg_dir.glob("*.json")):
            try:
                content = msg_file.read_text(encoding="utf-8")
                data_dict = anyenv.load_json(content)
                data = adapter.validate_python(data_dict)
                messages.append(data)
            except anyenv.JsonLoadError as e:
                logger.warning("Failed to parse message", path=str(msg_file), error=str(e))
        return messages

    def _read_parts(self, message_id: str) -> list[OpenCodePart]:
        """Read all parts for a message."""
        parts_dir = self.parts_path / message_id
        if not parts_dir.exists():
            return []

        parts: list[OpenCodePart] = []
        adapter = TypeAdapter[Any](OpenCodePart)
        for part_file in sorted(parts_dir.glob("*.json")):
            try:
                content = part_file.read_text(encoding="utf-8")
                data = anyenv.load_json(content)
                parts.append(adapter.validate_python(data))
            except anyenv.JsonLoadError as e:
                logger.warning("Failed to parse part", path=str(part_file), error=str(e))
        return parts

    async def _write_message(  # noqa: PLR0915
        self,
        *,
        message_id: str,
        session_id: str,
        role: str,
        model_messages: list[ModelRequest | ModelResponse],
        parent_id: str | None = None,
        model: str | None = None,
        cost_info: TokenCost | None = None,
        finish_reason: Any | None = None,
    ) -> None:
        """Write a message in OpenCode format."""
        from wolfharness_server.opencode_server.models import (
            AssistantMessage,
            MessagePath,
            MessageTime,
            ModelRef,
            ReasoningPart as OpenCodeReasoningPart,
            TextPart as OpenCodeTextPart,
            TimeCreated,
            TimeStartEndCompacted,
            TimeStartEndOptional,
            TokenCache,
            Tokens,
            ToolPart as OpenCodeToolPart,
            ToolStateCompleted,
            ToolStateError,
            ToolStatePending,
            ToolStateRunning,
            UserMessage,
        )

        now_ms = int(get_now().timestamp() * 1000)
        # Ensure message directory exists
        msg_dir = self.messages_path / session_id
        msg_dir.mkdir(parents=True, exist_ok=True)
        # Create OpenCode message based on role
        oc_message: OpenCodeMessage
        if role == "assistant":
            oc_message = AssistantMessage(
                id=message_id,
                session_id=session_id,
                parent_id=parent_id or "",
                model_id=model or "",
                provider_id=model.split(":")[0] if model else "wolfharness",
                path=MessagePath(cwd=str(Path.cwd()), root=str(Path.cwd())),
                time=MessageTime(created=now_ms),
                tokens=Tokens(
                    input=cost_info.token_usage.input_tokens if cost_info else 0,
                    output=cost_info.token_usage.output_tokens if cost_info else 0,
                    cache=TokenCache(
                        read=cost_info.token_usage.cache_read_tokens if cost_info else 0,
                        write=cost_info.token_usage.cache_write_tokens if cost_info else 0,
                    ),
                )
                if cost_info
                else Tokens(),
                cost=float(cost_info.total_cost) if cost_info else 0.0,
                finish=finish_reason,
            )
        else:  # user message
            oc_message = UserMessage(
                id=message_id,
                session_id=session_id,
                time=TimeCreated(created=now_ms),
                model=ModelRef(provider_id="", model_id=model or "") if model else None,
            )

        # Write message file
        msg_file = msg_dir / f"{message_id}.json"
        dct = oc_message.model_dump(by_alias=True)
        msg_file.write_text(anyenv.dump_json(dct, indent=True), encoding="utf-8")
        # Convert model messages to OpenCode parts
        parts_dir = self.parts_path / message_id
        parts_dir.mkdir(parents=True, exist_ok=True)
        part_counter = 0
        for msg in model_messages:
            if isinstance(msg, ModelRequest):
                # User prompt parts
                for part in msg.parts:
                    match part:
                        case UserPromptPart(content=msg_content):
                            # Convert UserContent to OpenCode parts using helper
                            text_parts = helpers.convert_user_content_to_parts(
                                content=msg_content,
                                message_id=message_id,
                                session_id=session_id,
                                part_counter_start=part_counter,
                            )
                            part_counter += len(text_parts)
                            # Write each part to disk
                            for text_part in text_parts:
                                part_file = parts_dir / f"{text_part.id}.json"
                                data = text_part.model_dump(by_alias=True)
                                text = anyenv.dump_json(data, indent=True)
                                part_file.write_text(text, encoding="utf-8")
                        case ToolReturnPart(tool_call_id=tool_call_id):
                            # Tool return - update existing tool part with output
                            tool_part_file = None
                            # Find the tool part with matching call_id
                            for existing_file in parts_dir.glob("*.json"):
                                try:
                                    text = existing_file.read_text(encoding="utf-8")
                                    content = anyenv.load_json(text, return_type=dict)
                                    if (
                                        content.get("type") == "tool"
                                        and content.get("callID") == tool_call_id
                                    ):
                                        tool_part_file = existing_file
                                        break
                                except Exception:  # noqa: BLE001
                                    continue

                            if tool_part_file:
                                # Update the tool part with output - create new completed state
                                text = tool_part_file.read_text(encoding="utf-8")
                                tool_part = anyenv.load_json(text, return_type=OpenCodeToolPart)
                                # Create new ToolStateCompleted (states are immutable)
                                # All tool states have .input,
                                # but only Running/Completed/Error have .time
                                start_time = 0
                                if isinstance(
                                    tool_part.state,
                                    (ToolStateRunning, ToolStateCompleted, ToolStateError),
                                ):
                                    start_time = tool_part.state.time.start

                                completed_state = ToolStateCompleted(
                                    input=tool_part.state.input,
                                    output=str(part.content),
                                    title=tool_part.tool,
                                    time=TimeStartEndCompacted(
                                        start=start_time,
                                        end=int(get_now().timestamp() * 1000),
                                    ),
                                )
                                # Create new tool part with updated state
                                updated_tool_part = OpenCodeToolPart(
                                    id=tool_part.id,
                                    message_id=tool_part.message_id,
                                    session_id=tool_part.session_id,
                                    call_id=tool_part.call_id,
                                    tool=tool_part.tool,
                                    state=completed_state,
                                )
                                dct = updated_tool_part.model_dump(by_alias=True)
                                text = anyenv.dump_json(dct, indent=True)
                                tool_part_file.write_text(text, encoding="utf-8")

            elif isinstance(msg, ModelResponse):
                # Model response parts
                for part in msg.parts:  # type: ignore[assignment]
                    part_id = f"{message_id}-{part_counter}"
                    part_counter += 1

                    if isinstance(part, TextPart):
                        text_part = OpenCodeTextPart(
                            id=part_id,
                            session_id=session_id,
                            message_id=message_id,
                            text=part.content,
                            time=TimeStartEndOptional(start=now_ms),
                        )
                        part_file = parts_dir / f"{part_id}.json"
                        dct = text_part.model_dump(by_alias=True)
                        part_file.write_text(anyenv.dump_json(dct, indent=True), encoding="utf-8")

                    elif isinstance(part, ThinkingPart):
                        reasoning_metadata: dict[str, Any] = {}
                        if part.id is not None:
                            reasoning_metadata["thinking_id"] = part.id
                        if part.provider_name is not None:
                            reasoning_metadata["provider_name"] = part.provider_name
                        if part.signature is not None:
                            reasoning_metadata["signature"] = part.signature
                        if part.provider_details is not None:
                            reasoning_metadata["provider_details"] = part.provider_details
                        reasoning_part = OpenCodeReasoningPart(
                            id=part_id,
                            session_id=session_id,
                            message_id=message_id,
                            text=part.content,
                            metadata=reasoning_metadata or None,
                            time=TimeStartEndOptional(start=now_ms),
                        )
                        part_file = parts_dir / f"{part_id}.json"
                        dct = reasoning_part.model_dump(by_alias=True)
                        part_file.write_text(anyenv.dump_json(dct, indent=True), encoding="utf-8")

                    elif isinstance(part, ToolCallPart):
                        # Create tool part with pending status
                        tool_part = OpenCodeToolPart(
                            id=part_id,
                            session_id=session_id,
                            message_id=message_id,
                            call_id=part.tool_call_id,
                            tool=part.tool_name,
                            state=ToolStatePending(input=safe_args_as_dict(part)),
                        )
                        part_file = parts_dir / f"{part_id}.json"
                        dct = tool_part.model_dump(by_alias=True)
                        part_file.write_text(anyenv.dump_json(dct, indent=True), encoding="utf-8")

    async def filter_messages(self, query: SessionQuery) -> list[ChatMessage[str]]:
        """Filter messages based on query."""
        messages: list[ChatMessage[str]] = []
        # Narrow session list when a specific session is requested
        if query.name:
            sessions = [(sid, p) for sid, p in self._list_sessions() if sid == query.name]
        else:
            sessions = self._list_sessions()
        for session_id, session_path in sessions:
            if not session_path.exists():
                continue
            oc_messages = self._read_messages(session_id)
            # Read parts for all messages
            msg_parts_map = {oc_msg.id: self._read_parts(oc_msg.id) for oc_msg in oc_messages}
            for oc_msg in oc_messages:
                parts = msg_parts_map.get(oc_msg.id, [])
                chat_msg = helpers.to_chat_message(msg=oc_msg, parts=parts)

                # Apply filters
                if query.agents and chat_msg.name not in query.agents:
                    continue
                cutoff = query.get_time_cutoff()
                if query.since and cutoff and chat_msg.timestamp < cutoff:
                    continue
                if query.until:
                    until_dt = parse_iso_timestamp(query.until)
                    if chat_msg.timestamp > until_dt:
                        continue
                if query.contains and query.contains not in chat_msg.content:
                    continue
                if query.roles and chat_msg.role not in query.roles:
                    continue
                messages.append(chat_msg)

                if query.limit and len(messages) >= query.limit:
                    return messages

        return messages

    async def log_message(self, *, message: ChatMessage[Any]) -> None:
        """Log a message to OpenCode format."""
        if not message.messages:
            logger.debug("No structured messages to log, skipping")
            return

        try:
            await self._write_message(
                message_id=message.message_id,
                session_id=message.session_id or "",
                role=message.role,
                model_messages=message.messages,
                parent_id=message.parent_id,
                model=message.model_name,
                cost_info=message.cost_info,
                finish_reason=message.finish_reason,
            )
        except Exception as e:
            logger.exception("Failed to write OpenCode message", error=str(e))

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
        """Log a conversation start.

        Creates a new session file in OpenCode format.
        """
        # Check if session already exists
        existing_path = next(
            (p for sid, p in self._list_sessions() if sid == session_id),
            None,
        )
        if existing_path:
            return  # Session already exists

        # Create new session file
        now = datetime_to_ms(start_time or get_now())

        # Compute project_id from working directory
        project_id = helpers.compute_project_id(str(self.base_path))

        # Ensure project directory exists
        project_dir = self.sessions_path / project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        # Create new session
        new_session = Session(
            id=session_id,
            project_id=project_id,
            directory=str(self.base_path),
            title="New Session",  # Will be updated when title is generated
            version=OPENCODE_VERSION,
            time=TimeCreatedUpdated(created=now, updated=now),
            parent_id=parent_session_id,
        )

        # Write session file
        session_path = project_dir / f"{session_id}.json"
        dct = new_session.model_dump(by_alias=True)
        session_path.write_text(anyenv.dump_json(dct, indent=True), encoding="utf-8")

        logger.debug(
            "Created new session file",
            session_id=session_id,
            path=str(session_path),
        )

    async def get_sessions(self, filters: QueryFilters) -> list[ConvData]:
        """Get filtered conversations with their messages."""
        result: list[ConvData] = []
        for session_id, session_path in self._list_sessions():
            session = helpers.read_session(session_path)
            if not session:
                continue
            oc_messages = self._read_messages(session_id)
            if not oc_messages:
                continue
            # Read parts for all messages
            msg_parts_map = {oc_msg.id: self._read_parts(oc_msg.id) for oc_msg in oc_messages}
            # Convert messages
            chat_messages: list[ChatMessage[str]] = []
            total_tokens = 0
            for oc_msg in oc_messages:
                parts = msg_parts_map.get(oc_msg.id, [])
                chat_msg = helpers.to_chat_message(msg=oc_msg, parts=parts)
                chat_messages.append(chat_msg)

                # Only assistant messages have tokens and cost
                if isinstance(oc_msg, AssistantMessage) and oc_msg.tokens:
                    total_tokens += oc_msg.tokens.input + oc_msg.tokens.output

            if not chat_messages:
                continue
            first_timestamp = ms_to_datetime(session.time.created)
            # Apply filters
            if filters.agent_name and not any(m.name == filters.agent_name for m in chat_messages):
                continue
            if filters.since and first_timestamp < filters.since:
                continue
            if filters.query and not any(filters.query in m.content for m in chat_messages):
                continue

            usage = TokenUsage(total=total_tokens, prompt=0, completion=0) if total_tokens else None
            conv_data = ConvData(
                id=session_id,
                agent=chat_messages[0].name or "opencode",
                title=session.title,
                start_time=first_timestamp.isoformat(),
                messages=chat_messages,
                token_usage=usage,
            )
            result.append(conv_data)
            if filters.limit and len(result) >= filters.limit:
                break

        return result

    async def get_session_stats(self, filters: StatsFilters) -> dict[str, dict[str, Any]]:
        """Get conversation statistics."""
        stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"total_tokens": 0, "messages": 0, "models": set(), "total_cost": 0.0}
        )
        for _session_id, session_path in self._list_sessions():
            session = helpers.read_session(session_path)
            if not session:
                continue

            timestamp = ms_to_datetime(session.time.created)
            if timestamp < filters.cutoff:
                continue
            for oc_msg in self._read_messages(session.id):
                if not isinstance(oc_msg, AssistantMessage):
                    continue
                # AssistantMessage only has model_id
                model = oc_msg.model_id or "unknown"
                tokens = 0
                if oc_msg.tokens:
                    tokens = oc_msg.tokens.input + oc_msg.tokens.output

                msg_timestamp = ms_to_datetime(oc_msg.time.created)
                # Group by specified criterion
                match filters.group_by:
                    case "model":
                        key = model
                    case "hour":
                        key = msg_timestamp.strftime("%Y-%m-%d %H:00")
                    case "day":
                        key = msg_timestamp.strftime("%Y-%m-%d")
                    case _:
                        key = oc_msg.agent or "opencode"

                stats[key]["messages"] += 1
                stats[key]["total_tokens"] += tokens
                stats[key]["models"].add(model)
                stats[key]["total_cost"] += oc_msg.cost or 0.0

        # Convert sets to lists
        for value in stats.values():
            value["models"] = list(value["models"])

        return dict(stats)

    async def reset(self, *, agent_name: str | None = None, hard: bool = False) -> tuple[int, int]:
        """Reset storage.

        Warning: This would delete OpenCode data!
        """
        logger.warning("Reset not implemented for OpenCode storage (read-only)")
        return 0, 0

    async def get_session_counts(self, *, agent_name: str | None = None) -> tuple[int, int]:
        """Get counts of conversations and messages."""
        conv_count = 0
        msg_count = 0
        for session_id, _session_path in self._list_sessions():
            msg_dir = self.messages_path / session_id
            if not msg_dir.exists():
                continue
            # Count JSON files directly instead of parsing them
            if message_files := list(msg_dir.glob("*.json")):
                conv_count += 1
                msg_count += len(message_files)

        return conv_count, msg_count

    async def get_session_messages(
        self,
        session_id: str,
        *,
        include_ancestors: bool = False,
    ) -> list[ChatMessage[str]]:
        """Get all messages for a session."""
        # Read messages for this session
        messages: list[ChatMessage[str]] = []
        for oc_msg in self._read_messages(session_id):
            parts = self._read_parts(oc_msg.id)
            chat_msg = helpers.to_chat_message(msg=oc_msg, parts=parts)
            messages.append(chat_msg)

        # Sort by timestamp, then by message_id for deterministic ordering
        now = get_now()
        messages.sort(key=lambda m: (m.timestamp or now, m.message_id))
        if not include_ancestors or not messages:
            return messages
        # Get ancestor chain if first message has parent_id
        if parent_id := messages[0].parent_id:
            ancestors = await self.get_message_ancestry(parent_id, session_id=session_id)
            return ancestors + messages
        return messages

    async def get_message(
        self,
        message_id: str,
        *,
        session_id: str | None = None,
    ) -> ChatMessage[str] | None:
        """Get a single message by ID."""
        # If session_id is provided, search only that session
        sessions = [(session_id, None)] if session_id else self._list_sessions()
        for sid, _session_path in sessions:
            assert sid is not None
            for oc_msg in self._read_messages(sid):
                if oc_msg.id == message_id:
                    parts = self._read_parts(oc_msg.id)
                    return helpers.to_chat_message(msg=oc_msg, parts=parts)

        return None

    async def get_message_ancestry(
        self,
        message_id: str,
        *,
        session_id: str | None = None,
    ) -> list[ChatMessage[str]]:
        """Get the ancestry chain of a message.

        Traverses parent_id chain to build full history.
        When session_id is provided, loads messages once and traverses in-memory.

        Args:
            message_id: ID of the message
            session_id: Optional session ID hint for faster lookup

        Returns:
            List of messages from oldest ancestor to the specified message
        """
        # Fast path: if we know the session, load messages once and traverse in-memory
        ancestors: list[ChatMessage[str]] = []
        if session_id:
            # Build ID -> message index for O(1) lookups
            msg_by_id = {msg.id: msg for msg in self._read_messages(session_id)}
            current_id: str | None = message_id
            while current_id:
                oc_msg = msg_by_id.get(current_id)
                if not oc_msg:
                    break
                parts = self._read_parts(oc_msg.id)
                chat_msg = helpers.to_chat_message(msg=oc_msg, parts=parts)
                ancestors.append(chat_msg)
                current_id = chat_msg.parent_id
            ancestors.reverse()
            return ancestors
        # Slow path: search all sessions
        current_id = message_id
        while current_id:
            msg = await self.get_message(current_id)
            if not msg:
                break
            ancestors.append(msg)
            current_id = msg.parent_id
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

        Creates a new session directory. The fork point message_id is returned
        so callers can set it as parent_id for new messages.

        Args:
            source_session_id: Source session ID
            new_session_id: New session ID
            fork_from_message_id: Message ID to fork from. If None, forks from last
            new_agent_name: Not directly stored in OpenCode session format

        Returns:
            The ID of the fork point message
        """
        # Find source session
        source_path = next(
            (p for sid, p in self._list_sessions() if sid == source_session_id),
            None,
        )
        if not source_path:
            raise ValueError(f"Source conversation not found: {source_session_id}")
        source_session = helpers.read_session(source_path)
        assert source_session
        # Read source messages
        oc_messages = self._read_messages(source_session_id)
        # Find fork point
        fork_point_id: str | None = None
        if fork_from_message_id:
            # Verify message exists
            if not any(m.id == fork_from_message_id for m in oc_messages):
                raise ValueError(f"Message {fork_from_message_id} not found in conversation")
            fork_point_id = fork_from_message_id
        # Fork from last message
        elif oc_messages:
            # Messages are already in time order from _read_messages
            fork_point_id = oc_messages[-1].id
        # Create new session directory structure
        # Determine project from source path structure
        project_id = source_path.parent.name
        new_session_dir = self.sessions_path / project_id
        new_session_dir.mkdir(parents=True, exist_ok=True)
        # Create empty session file (will be populated when messages added)
        # Create new session metadata
        fork_title = f"{source_session.title} (fork)" if source_session.title else "Forked Session"
        now = datetime_to_ms(get_now())
        new_session = Session(
            id=new_session_id,
            project_id=project_id,
            directory=source_session.directory,  # Same project directory as source
            title=fork_title,
            version=OPENCODE_VERSION,
            time=TimeCreatedUpdated(created=now, updated=now),
            summary=SessionSummary(files=0, additions=0, deletions=0),
        )
        # Write session file
        dct = new_session.model_dump(by_alias=True)
        new_session_path = new_session_dir / f"{new_session_id}.json"
        new_session_path.write_text(anyenv.dump_json(dct, indent=True), encoding="utf-8")
        # Create message and part directories
        (self.messages_path / new_session_id).mkdir(parents=True, exist_ok=True)
        (self.parts_path / new_session_id).mkdir(parents=True, exist_ok=True)
        return fork_point_id

    # Project storage methods

    async def save_project(self, project: ProjectData) -> None:
        """Save or update a project.

        Writes project data as a JSON file at projects_path/{project_id}.json.

        Args:
            project: Project data to persist
        """
        await self._save_project_sync(project)
        logger.debug("Saved project", project_id=project.project_id)

    @run_in_thread
    def _save_project_sync(self, project: ProjectData) -> None:
        """Persist project metadata without blocking the event loop."""
        project_file = self.projects_path / f"{project.project_id}.json"
        data = project.model_dump(mode="json")
        project_file.write_text(anyenv.dump_json(data, indent=True), encoding="utf-8")

    async def get_project(self, project_id: str) -> ProjectData | None:
        """Get a project by ID.

        Args:
            project_id: Project identifier

        Returns:
            Project data if found, None otherwise
        """
        project_file = self.projects_path / f"{project_id}.json"
        if not project_file.exists():
            return None
        try:
            content = project_file.read_text(encoding="utf-8")
            data = anyenv.load_json(content, return_type=dict)
            return ProjectData.model_validate(data)
        except (anyenv.JsonLoadError, Exception) as e:  # noqa: BLE001
            logger.warning("Failed to read project file", path=str(project_file), error=str(e))
            return None

    async def get_project_by_worktree(self, worktree: str) -> ProjectData | None:
        """Get a project by worktree path.

        Resolves the worktree path before comparing to handle symlink differences.

        Args:
            worktree: Absolute path to the project worktree

        Returns:
            Project data if found, None otherwise
        """
        resolved_worktree = str(Path(worktree).resolve())
        for project_file in self.projects_path.glob("*.json"):
            try:
                content = project_file.read_text(encoding="utf-8")
                data = anyenv.load_json(content, return_type=dict)
                project = ProjectData.model_validate(data)
                if project.worktree and str(Path(project.worktree).resolve()) == resolved_worktree:
                    return project
            except (anyenv.JsonLoadError, Exception):  # noqa: BLE001
                continue
        return None

    async def get_project_by_name(self, name: str) -> ProjectData | None:
        """Get a project by friendly name.

        Args:
            name: Project name

        Returns:
            Project data if found, None otherwise
        """
        for project_file in self.projects_path.glob("*.json"):
            try:
                content = project_file.read_text(encoding="utf-8")
                data = anyenv.load_json(content, return_type=dict)
                project = ProjectData.model_validate(data)
                if project.name == name:
                    return project
            except (anyenv.JsonLoadError, Exception):  # noqa: BLE001
                continue
        return None

    async def list_projects(self, limit: int | None = None) -> list[ProjectData]:
        """List all projects, ordered by last_active descending.

        Args:
            limit: Maximum number of projects to return

        Returns:
            List of project data objects sorted by last_active descending
        """
        projects: list[ProjectData] = []
        for project_file in self.projects_path.glob("*.json"):
            try:
                content = project_file.read_text(encoding="utf-8")
                data = anyenv.load_json(content, return_type=dict)
                project = ProjectData.model_validate(data)
                projects.append(project)
            except (anyenv.JsonLoadError, Exception) as e:  # noqa: BLE001
                logger.warning(
                    "Skipping corrupted project file",
                    path=str(project_file),
                    error=str(e),
                )
                continue
        projects.sort(key=lambda p: p.last_active, reverse=True)
        if limit is not None:
            projects = projects[:limit]
        return projects

    async def delete_project(self, project_id: str) -> bool:
        """Delete a project.

        Args:
            project_id: Project identifier

        Returns:
            True if project was deleted, False if not found
        """
        project_file = self.projects_path / f"{project_id}.json"
        if not project_file.exists():
            return False
        project_file.unlink()
        logger.debug("Deleted project", project_id=project_id)
        return True

    async def touch_project(self, project_id: str) -> None:
        """Update project's last_active timestamp.

        Args:
            project_id: Project identifier
        """
        project = await self.get_project(project_id)
        if project is not None:
            updated = project.touch()
            await self.save_project(updated)

    # Session persistence methods (required by StorageProvider base class)

    async def load_session(self, session_id: str) -> SessionData | None:
        """Load session data by ID.

        Loads session metadata and all associated messages from OpenCode storage.

        Args:
            session_id: Session identifier

        Returns:
            SessionData if session was found and loaded, None otherwise
        """
        # Find session file
        session_path = next(
            (p for sid, p in self._list_sessions() if sid == session_id),
            None,
        )
        if not session_path:
            return None

        # Read session metadata
        oc_session = helpers.read_session(session_path)
        if not oc_session:
            return None

        # Load all messages for this session
        messages = await self.get_session_messages(session_id)

        # Get agent name from first message if available, otherwise use default
        agent_name = messages[0].name if messages else "default"

        return SessionData(
            session_id=session_id,
            agent_name=agent_name or "default",
            project_id=oc_session.project_id,
            parent_id=oc_session.parent_id,
            cwd=oc_session.directory,
            created_at=ms_to_datetime(oc_session.time.created),
            last_active=ms_to_datetime(oc_session.time.updated),
            metadata={
                "title": oc_session.title,
                "version": oc_session.version,
            },
        )

    async def save_session(self, data: SessionData) -> None:
        """Save or update session data.

        Creates a new session file if it doesn't exist, or updates the existing
        session metadata (title, etc.) if it does.

        Args:
            data: Session data to save
        """
        # Find existing session file
        session_path = next(
            (p for sid, p in self._list_sessions() if sid == data.session_id),
            None,
        )

        if session_path:
            # Update existing session
            oc_session = helpers.read_session(session_path)
            if not oc_session:
                return

            # Update metadata
            if data.metadata.get("title"):
                oc_session.title = data.metadata["title"]
            oc_session.time.updated = datetime_to_ms(get_now())

            # Write back
            dct = oc_session.model_dump(by_alias=True)
            session_path.write_text(anyenv.dump_json(dct, indent=True), encoding="utf-8")
        else:
            # Create new session file
            now = datetime_to_ms(get_now())

            # Compute project_id from cwd if not provided or is "default"
            if data.project_id and data.project_id != "default":
                project_id = data.project_id
            elif data.cwd:
                project_id = helpers.compute_project_id(data.cwd)
            else:
                project_id = "global"

            # Ensure project directory exists
            project_dir = self.sessions_path / project_id
            project_dir.mkdir(parents=True, exist_ok=True)

            # Create new session
            new_session = Session(
                id=data.session_id,
                project_id=project_id,
                directory=data.cwd or str(self.base_path),
                title=data.metadata.get("title") or data.title or "New Session",
                version=OPENCODE_VERSION,
                time=TimeCreatedUpdated(created=now, updated=now),
                parent_id=data.parent_id,
            )

            # Write session file
            session_path = project_dir / f"{data.session_id}.json"
            dct = new_session.model_dump(by_alias=True)
            session_path.write_text(anyenv.dump_json(dct, indent=True), encoding="utf-8")

            logger.debug(
                "Created new session file",
                session_id=data.session_id,
                path=str(session_path),
            )

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages.

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted, False if not found
        """
        import shutil

        # Find session file
        session_path = next(
            (p for sid, p in self._list_sessions() if sid == session_id),
            None,
        )
        if not session_path:
            return False

        # Delete session file
        session_path.unlink(missing_ok=True)

        # Delete message directory
        msg_dir = self.messages_path / session_id
        if msg_dir.exists():
            shutil.rmtree(msg_dir, ignore_errors=True)

        # Delete parts directory
        parts_dir = self.parts_path / session_id
        if parts_dir.exists():
            shutil.rmtree(parts_dir, ignore_errors=True)

        return True

    async def update_session_title(self, session_id: str, title: str) -> None:
        """Update the title of a conversation.

        Finds the session JSON file and updates its title field.

        Args:
            session_id: ID of the conversation to update
            title: New title for the conversation
        """
        session_path = next(
            (p for sid, p in self._list_sessions() if sid == session_id),
            None,
        )
        if not session_path:
            logger.warning("Session not found for title update", session_id=session_id)
            return

        oc_session = helpers.read_session(session_path)
        if not oc_session:
            logger.warning("Failed to read session for title update", session_id=session_id)
            return

        oc_session.title = title
        oc_session.time.updated = datetime_to_ms(get_now())
        dct = oc_session.model_dump(by_alias=True)
        session_path.write_text(anyenv.dump_json(dct, indent=True), encoding="utf-8")

    async def update_sdk_session_id(self, session_id: str, sdk_session_id: str) -> None:
        """Update the external SDK session ID for a session.

        Stores the SDK session ID in the session JSON's metadata.sdk_session_id field.
        Creates the metadata dict if it doesn't exist.

        Args:
            session_id: Internal session identifier
            sdk_session_id: External SDK session ID
        """
        session_path = next(
            (p for sid, p in self._list_sessions() if sid == session_id),
            None,
        )
        if not session_path:
            logger.warning("Session not found for SDK session ID update", session_id=session_id)
            return

        try:
            content = session_path.read_text(encoding="utf-8")
            data = anyenv.load_json(content, return_type=dict)
        except anyenv.JsonLoadError as e:
            logger.warning(
                "Failed to read session for SDK session ID update",
                session_id=session_id,
                error=str(e),
            )
            return

        metadata = data.get("metadata", {})
        metadata["sdk_session_id"] = sdk_session_id
        data["metadata"] = metadata
        data["time"] = data.get("time", {})
        data["time"]["updated"] = datetime_to_ms(get_now())
        session_path.write_text(anyenv.dump_json(data, indent=True), encoding="utf-8")

    async def delete_session_messages(self, session_id: str) -> int:
        """Delete all messages for a session.

        Removes message JSON files and their associated part files from disk.

        Args:
            session_id: ID of the conversation to clear

        Returns:
            Number of messages deleted
        """
        msg_dir = self.messages_path / session_id
        if not msg_dir.exists():
            return 0

        # Collect message IDs before deleting so we can clean up parts
        message_files = list(msg_dir.glob("*.json"))
        message_ids = [f.stem for f in message_files]

        # Delete part files for each message
        for message_id in message_ids:
            parts_dir = self.parts_path / message_id
            if parts_dir.exists():
                for part_file in parts_dir.glob("*.json"):
                    part_file.unlink(missing_ok=True)
                # Remove empty parts directory
                if not any(parts_dir.iterdir()):
                    parts_dir.rmdir()

        # Delete message files
        for msg_file in message_files:
            msg_file.unlink(missing_ok=True)

        # Remove empty message directory
        if msg_dir.exists() and not any(msg_dir.iterdir()):
            msg_dir.rmdir()

        return len(message_files)

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
    ) -> list[ConvData]:
        """Get filtered conversations with formatted output.

        Iterates all session JSON files, applies filters, and returns matching
        ConversationData objects.

        Args:
            agent_name: Filter by agent name
            period: Time period to include (e.g. "1h", "2d")
            since: Only show conversations after this time
            query: Search in message content
            model: Filter by model used
            limit: Maximum number of conversations
            compact: Only show first/last message of each conversation
            include_tokens: Include token usage statistics
        """
        from wolfharness.utils.parse_time import parse_time_period
        from wolfharness.utils.time_utils import get_now

        cutoff: datetime | None = None
        if period:
            cutoff = get_now() - parse_time_period(period)
        elif since:
            cutoff = since

        return await self._get_filtered_conversations_sync(
            agent_name=agent_name,
            cutoff=cutoff,
            query=query,
            model=model,
            limit=limit,
            compact=compact,
            include_tokens=include_tokens,
        )

    @run_in_thread
    def _get_filtered_conversations_sync(
        self,
        *,
        agent_name: str | None,
        cutoff: datetime | None,
        query: str | None,
        model: str | None,
        limit: int | None,
        compact: bool,
        include_tokens: bool,
    ) -> list[ConvData]:
        """Threaded wrapper for filesystem-heavy conversation filtering."""
        return _get_filtered_conversations_sync(
            self,
            agent_name=agent_name,
            cutoff=cutoff,
            query=query,
            model=model,
            limit=limit,
            compact=compact,
            include_tokens=include_tokens,
        )

    async def list_session_ids(
        self,
        *,
        pool_id: str | None = None,
        agent_name: str | None = None,
        cwd: str | None = None,
    ) -> list[str]:
        """List session IDs, optionally filtered.

        Args:
            pool_id: Filter by pool/manifest ID (not used in OpenCode storage)
            agent_name: Filter by agent name
            cwd: Filter by working directory. Uses compute_project_id() to narrow
                 to the project directory first, then verifies the session's
                 ``directory`` field matches.

        Returns:
            List of session IDs
        """
        # When cwd is provided, narrow search to the project directory for efficiency.
        # OpenCode organizes sessions under storage/session/{project_id}/, so
        # computing the project_id from cwd avoids scanning all project dirs.
        filter_project_id: str | None = None
        if cwd is not None:
            filter_project_id = helpers.compute_project_id(cwd)

        session_ids: list[str] = []
        for session_id, session_path in self._list_sessions(project_id=filter_project_id):
            # Check cwd filter by reading session's directory field
            if cwd is not None:
                session = helpers.read_session(session_path)
                if session is None or session.directory != cwd:
                    continue
            # Check agent filter if specified
            # Note: OpenCode session files don't store agent name directly,
            # so we need to check the first message's agent
            if agent_name:
                messages = await self.get_session_messages(session_id)
                if messages:
                    # Session has messages - check agent name
                    first_agent = messages[0].name
                    if first_agent != agent_name:
                        continue
                # If no messages, include the session anyway (don't filter out new sessions)
            session_ids.append(session_id)
        return session_ids


if __name__ == "__main__":
    import asyncio
    import datetime as dt

    from wolfharness_storage.models import QueryFilters, StatsFilters

    async def main() -> None:
        provider = OpenCodeStorageProvider()
        print(f"Base path: {provider.base_path}")
        print(f"Exists: {provider.base_path.exists()}")
        # List conversations
        filters = QueryFilters(limit=10)
        conversations = await provider.get_sessions(filters)
        print(f"\nFound {len(conversations)} conversations")
        for conv_data in conversations[:5]:
            print(f"  - {conv_data['id'][:8]}... | {conv_data['title'] or 'Untitled'}")
            print(f"    Messages: {len(conv_data['messages'])}, Updated: {conv_data['start_time']}")
        # Get counts
        conv_count, msg_count = await provider.get_session_counts()
        print(f"\nTotal: {conv_count} conversations, {msg_count} messages")
        # Get stats
        cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(days=30)
        stats_filters = StatsFilters(cutoff=cutoff, group_by="day")
        stats = await provider.get_session_stats(stats_filters)
        print(f"\nStats: {stats}")

    asyncio.run(main())
