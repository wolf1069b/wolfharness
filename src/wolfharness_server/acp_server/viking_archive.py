"""Viking archive for ACP session updates."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import os
import re
from typing import Any

from wolfharness.log import get_logger


logger = get_logger(__name__)

_ERROR_TYPES = (RuntimeError, OSError, TimeoutError, ValueError, TypeError, ImportError)


@dataclass
class ACPVikingEventArchive:
    """Asynchronous best-effort archive for ACP ``SessionUpdate`` payloads."""

    enabled: bool = False
    url: str | None = None
    api_key: str | None = None
    user: str | None = None
    session_prefix: str = "iroot-acp-session"
    batch_size: int = 25
    flush_on_turn_complete: bool = True
    transcript_enabled: bool = True
    transcript_keep_recent_count: int = 10
    _client: Any = field(default=None, init=False, repr=False)
    _pending: dict[str, list[dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
    _transcript_pending: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _chunk_buffers: dict[str, dict[tuple[str, str], dict[str, Any]]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _sequence: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _tasks: set[asyncio.Task[None]] = field(default_factory=set, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    @classmethod
    def from_config(cls, config: Any) -> ACPVikingEventArchive:
        """Build an archive from manifest config, expanding env vars lazily."""
        if config is None or not bool(getattr(config, "enabled", False)):
            return cls(enabled=False)
        api_key = _expand(getattr(config, "api_key", None) or os.getenv("VIKING_MCP_API_KEY"))
        url = _expand(
            getattr(config, "url", None)
            or os.getenv("VIKING_MCP_URL")
            or "http://viking.ai.rootcloud.info"
        )
        user = _expand(getattr(config, "user", None) or os.getenv("VIKING_MCP_USER"))
        return cls(
            enabled=True,
            url=url,
            api_key=api_key,
            user=user,
            session_prefix=str(getattr(config, "session_prefix", "iroot-acp-session")),
            batch_size=int(getattr(config, "batch_size", 25)),
            flush_on_turn_complete=bool(getattr(config, "flush_on_turn_complete", True)),
            transcript_enabled=bool(getattr(config, "transcript_enabled", True)),
            transcript_keep_recent_count=int(getattr(config, "transcript_keep_recent_count", 10)),
        )

    async def record_update(
        self,
        *,
        consumer_session_id: str,
        source_session_id: str,
        event_id: Any,
        update: Any,
    ) -> None:
        """Buffer an ACP update and schedule archive writes when needed."""
        if not self.enabled:
            return
        try:
            payload = _dump_update(update)
            event_type = str(payload.get("sessionUpdate") or payload.get("session_update") or "")
            session_id = source_session_id or consumer_session_id
            async with self._lock:
                seq = self._sequence.get(session_id, 0) + 1
                self._sequence[session_id] = seq
                record = {
                    "sequence": seq,
                    "created_at": datetime.now(UTC).isoformat(),
                    "session_id": session_id,
                    "consumer_session_id": consumer_session_id,
                    "source_session_id": source_session_id,
                    "event_id": str(event_id),
                    "event_type": event_type,
                    "message_id": payload.get("messageId") or payload.get("message_id"),
                    "tool_call_id": payload.get("toolCallId") or payload.get("tool_call_id"),
                    "update": payload,
                }
                pending = self._pending.setdefault(session_id, [])
                pending.append(record)
                if self.transcript_enabled:
                    self._buffer_transcript_update_locked(session_id, payload, event_type)
                should_flush = len(pending) >= self.batch_size or (
                    self.flush_on_turn_complete and event_type == "turn_complete"
                )
            if should_flush:
                self.schedule_flush(session_id)
        except _ERROR_TYPES:
            logger.warning("ACP Viking archive record failed", exc_info=True)

    async def record_protocol_frame(
        self,
        *,
        session_id: str,
        direction: str,
        message: dict[str, Any],
        request_method: str = "",
    ) -> None:
        """Buffer a raw ACP JSON-RPC frame for complete frontend interaction audit."""
        if not self.enabled:
            return
        try:
            rpc_id = message.get("id")
            method = str(message.get("method") or request_method or "")
            event_type = _protocol_event_type(message)
            effective_session_id = session_id or _extract_session_id(message) or "connection"
            async with self._lock:
                seq = self._sequence.get(effective_session_id, 0) + 1
                self._sequence[effective_session_id] = seq
                event_id = f"rpc:{direction}:{rpc_id}" if rpc_id is not None else f"rpc:{direction}"
                record = {
                    "sequence": seq,
                    "created_at": datetime.now(UTC).isoformat(),
                    "session_id": effective_session_id,
                    "event_id": event_id,
                    "event_type": event_type,
                    "direction": direction,
                    "rpc_id": rpc_id,
                    "method": method,
                    "message_id": _extract_message_id(message),
                    "tool_call_id": _extract_tool_call_id(message),
                    "protocol": message,
                }
                pending = self._pending.setdefault(effective_session_id, [])
                pending.append(record)
                should_flush = len(pending) >= self.batch_size or (
                    self.flush_on_turn_complete and _should_flush_protocol_event(method, event_type)
                )
            if should_flush:
                self.schedule_flush(effective_session_id)
        except _ERROR_TYPES:
            logger.warning("ACP Viking protocol archive record failed", exc_info=True)

    def schedule_protocol_frame(
        self,
        *,
        session_id: str,
        direction: str,
        message: dict[str, Any],
        request_method: str = "",
    ) -> None:
        """Schedule raw protocol frame recording without blocking JSON-RPC handling."""
        if not self.enabled:
            return
        task = asyncio.create_task(
            self.record_protocol_frame(
                session_id=session_id,
                direction=direction,
                message=message,
                request_method=request_method,
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def schedule_flush(self, session_id: str) -> None:
        """Schedule a background flush for one session."""
        if not self.enabled:
            return
        task = asyncio.create_task(self.flush_session(session_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def flush_session(self, session_id: str) -> None:
        """Flush buffered events for one session to Viking."""
        async with self._lock:
            records = self._pending.pop(session_id, [])
            transcript_messages = self._pop_transcript_locked(session_id)
        if not records and not transcript_messages:
            return
        try:
            client = await self._ensure_client()
            uri = self._events_uri(session_id)
            if records:
                content = "".join(
                    json.dumps(record, ensure_ascii=False) + "\n" for record in records
                )
                await client.write(uri, content, mode="append", wait=False, processing_mode="raw")
            transcript_commit = None
            if transcript_messages:
                transcript_commit = await self._write_transcript(
                    client,
                    session_id,
                    transcript_messages,
                )
            await client.write(
                self._memory_context_uri(session_id),
                json.dumps(
                    {
                        "session_id": session_id,
                        "archive_session_id": self._archive_session_id(session_id),
                        "events_uri": uri,
                        "last_event_sequence": records[-1]["sequence"] if records else None,
                        "last_event_type": records[-1]["event_type"] if records else None,
                        "transcript_message_count": len(transcript_messages),
                        "transcript_commit": transcript_commit,
                        "updated_at": datetime.now(UTC).isoformat(),
                        "memory_policy": "archive-all-events-and-readable-transcript",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                mode="replace",
                wait=False,
                processing_mode="raw",
            )
        except _ERROR_TYPES:
            async with self._lock:
                if records:
                    self._pending.setdefault(session_id, records[:0])[:0] = records
                if transcript_messages:
                    self._transcript_pending.setdefault(session_id, transcript_messages[:0])[:0] = (
                        transcript_messages
                    )
            logger.warning("ACP Viking archive flush failed", session_id=session_id, exc_info=True)

    async def flush_all(self) -> None:
        """Flush all buffered events and await in-flight writes."""
        session_ids = set(self._pending)
        await asyncio.gather(*(self.flush_session(session_id) for session_id in session_ids))
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def close(self) -> None:
        """Flush buffered events and close the Viking client."""
        await self.flush_all()
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if close is not None:
                await close()
            self._client = None

    async def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        from openviking_sdk import AsyncHTTPClient

        client = AsyncHTTPClient(url=self.url, api_key=self.api_key, user=self.user)
        await client.initialize()
        self._client = client
        if self.user is None:
            await self._resolve_user_from_health(client)
        return client

    async def _resolve_user_from_health(self, client: Any) -> None:
        """Best-effort user resolution for user-scoped archive URIs."""
        try:
            response = await client._request("GET", "/health")
            data = response.json() if hasattr(response, "json") else response
            if isinstance(data, dict):
                user_id = data.get("user_id")
                if isinstance(user_id, str) and user_id.strip():
                    self.user = user_id.strip()
        except _ERROR_TYPES:
            logger.debug("ACP Viking archive user resolution failed", exc_info=True)

    def _events_uri(self, session_id: str) -> str:
        return f"{self._archive_base_uri(session_id)}/events.jsonl"

    def _memory_context_uri(self, session_id: str) -> str:
        return f"{self._archive_base_uri(session_id)}/memory_context.json"

    def _archive_session_id(self, session_id: str) -> str:
        return f"{self.session_prefix}-{_safe_identifier(session_id)}"

    def _archive_base_uri(self, session_id: str) -> str:
        user_id = _safe_identifier(self.user or "default")
        return f"viking://user/{user_id}/sessions/{self._archive_session_id(session_id)}"

    def _buffer_transcript_update_locked(
        self,
        session_id: str,
        payload: dict[str, Any],
        event_type: str,
    ) -> None:
        text = _content_text(payload)
        message_id = str(
            payload.get("messageId")
            or payload.get("message_id")
            or f"{event_type}:{self._sequence.get(session_id, 0)}"
        )
        if event_type == "user_message_chunk":
            self._append_chunk_locked(session_id, "user", message_id, text)
            return
        if event_type in ("agent_message_chunk", "agent_thought_chunk"):
            self._append_chunk_locked(session_id, "assistant", message_id, text)
            return
        if event_type in ("tool_call", "tool_call_update"):
            self._flush_chunk_buffers_locked(session_id)
            tool_part = _tool_part_from_update(payload)
            if tool_part:
                self._append_transcript_locked(session_id, "assistant", [tool_part])
            return
        if event_type == "turn_complete":
            self._flush_chunk_buffers_locked(session_id)

    def _append_chunk_locked(
        self,
        session_id: str,
        role: str,
        message_id: str,
        text: str,
    ) -> None:
        if not text:
            return
        buffers = self._chunk_buffers.setdefault(session_id, {})
        key = (role, message_id)
        buffer = buffers.get(key)
        if buffer is None:
            buffer = {"role": role, "parts": [{"type": "text", "text": ""}]}
            buffers[key] = buffer
        buffer["parts"][0]["text"] += text

    def _flush_chunk_buffers_locked(self, session_id: str) -> None:
        buffers = self._chunk_buffers.pop(session_id, {})
        for message in buffers.values():
            parts = message.get("parts")
            if isinstance(parts, list) and parts:
                text = parts[0].get("text") if isinstance(parts[0], dict) else ""
                if isinstance(text, str) and text.strip():
                    self._transcript_pending.setdefault(session_id, []).append(message)

    def _append_transcript_locked(
        self,
        session_id: str,
        role: str,
        parts: list[dict[str, Any]],
    ) -> None:
        if parts:
            self._transcript_pending.setdefault(session_id, []).append({
                "role": role,
                "parts": parts,
            })

    def _pop_transcript_locked(self, session_id: str) -> list[dict[str, Any]]:
        self._flush_chunk_buffers_locked(session_id)
        return self._transcript_pending.pop(session_id, [])

    async def _write_transcript(
        self,
        client: Any,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> Any:
        archive_session_id = self._archive_session_id(session_id)
        try:
            await client.create_session(
                session_id=archive_session_id,
                source_type=self.session_prefix,
            )
        except TypeError:
            try:
                await client.create_session(session_id=archive_session_id)
            except _ERROR_TYPES:
                logger.debug(
                    "ACP Viking transcript create_session skipped",
                    session_id=archive_session_id,
                    exc_info=True,
                )
        except _ERROR_TYPES:
            logger.debug(
                "ACP Viking transcript create_session skipped",
                session_id=archive_session_id,
                exc_info=True,
            )

        for message in messages:
            if "parts" in message:
                await client.add_message(
                    archive_session_id,
                    message["role"],
                    parts=message["parts"],
                )
            else:
                await client.add_message(
                    archive_session_id,
                    message["role"],
                    content=str(message.get("content", "")),
                )

        commit_kwargs: dict[str, Any] = {}
        if self.transcript_keep_recent_count > 0:
            commit_kwargs["keep_recent_count"] = self.transcript_keep_recent_count
        return await client.commit_session(archive_session_id, **commit_kwargs)


def _expand(value: str | None) -> str | None:
    return os.path.expandvars(value) if isinstance(value, str) else value


def _safe_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-") or "unknown"


def _dump_update(update: Any) -> dict[str, Any]:
    if hasattr(update, "model_dump"):
        dumped = update.model_dump(mode="json", by_alias=True, exclude_none=True)
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    if isinstance(update, dict):
        return update
    return {"value": str(update)}


def _content_text(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    text = payload.get("text")
    return text if isinstance(text, str) else ""


def _tool_part_from_update(payload: dict[str, Any]) -> dict[str, Any]:
    tool_id = payload.get("toolCallId") or payload.get("tool_call_id")
    tool_name = payload.get("toolName") or payload.get("tool_name") or payload.get("title")
    if not isinstance(tool_name, str) or not tool_name.strip():
        kind = payload.get("kind")
        tool_name = kind if isinstance(kind, str) and kind.strip() else "tool"

    part: dict[str, Any] = {
        "type": "tool",
        "tool_name": tool_name,
        "tool_status": _tool_status(payload.get("status")),
    }
    if isinstance(tool_id, str) and tool_id.strip():
        part["tool_id"] = tool_id
    raw_input = payload.get("rawInput") or payload.get("raw_input")
    if raw_input is not None:
        part["tool_input"] = raw_input
    raw_output = payload.get("rawOutput") or payload.get("raw_output")
    if raw_output is not None:
        part["tool_output"] = raw_output
    return part


def _tool_status(status: Any) -> str:
    if status in ("completed", "success", "succeeded"):
        return "completed"
    if status in ("failed", "error"):
        return "error"
    return "running"


class ACPVikingProtocolObserver:
    """Connection stream observer that archives raw ACP JSON-RPC frames."""

    def __init__(self, archive: ACPVikingEventArchive) -> None:
        self.archive = archive
        self._pending_requests: dict[tuple[str, Any], tuple[str, str]] = {}

    def __call__(self, event: Any) -> None:
        """Record a raw incoming/outgoing JSON-RPC frame."""
        if not self.archive.enabled:
            return
        message = getattr(event, "message", None)
        direction = getattr(event, "direction", "")
        if not isinstance(message, dict) or direction not in ("incoming", "outgoing"):
            return

        rpc_id = message.get("id")
        method = message.get("method")
        if isinstance(method, str):
            session_id = _extract_session_id(message)
            if rpc_id is not None:
                self._pending_requests[(direction, rpc_id)] = (method, session_id)
            self.archive.schedule_protocol_frame(
                session_id=session_id,
                direction=direction,
                message=message,
                request_method=method,
            )
            return

        if rpc_id is None:
            self.archive.schedule_protocol_frame(
                session_id=_extract_session_id(message),
                direction=direction,
                message=message,
            )
            return

        request_direction = "outgoing" if direction == "incoming" else "incoming"
        request_method, request_session_id = self._pending_requests.pop(
            (request_direction, rpc_id),
            ("", ""),
        )
        session_id = request_session_id or _extract_session_id(message)
        self.archive.schedule_protocol_frame(
            session_id=session_id,
            direction=direction,
            message=message,
            request_method=request_method,
        )


def _protocol_event_type(message: dict[str, Any]) -> str:
    if "method" in message and "id" in message:
        return "rpc_request"
    if "method" in message:
        return "rpc_notification"
    if "error" in message:
        return "rpc_error"
    return "rpc_response"


def _should_flush_protocol_event(method: str, event_type: str) -> bool:
    """Flush when a frontend interaction reaches a natural protocol boundary."""
    if not method:
        return False
    if event_type in ("rpc_response", "rpc_error"):
        return method.startswith(("session/", "elicitation/", "terminal/", "fs/"))
    return False


def _extract_session_id(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    for key in ("sessionId", "session_id"):
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return item.strip()
    for key in ("params", "result"):
        nested = value.get(key)
        if isinstance(nested, dict):
            found = _extract_session_id(nested)
            if found:
                return found
    return ""


def _extract_message_id(value: Any) -> str | None:
    item = _extract_nested_value(value, ("messageId", "message_id"))
    return item if isinstance(item, str) else None


def _extract_tool_call_id(value: Any) -> str | None:
    item = _extract_nested_value(value, ("toolCallId", "tool_call_id"))
    return item if isinstance(item, str) else None


def _extract_nested_value(value: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        for key in keys:
            if key in value:
                return value[key]
        for item in value.values():
            found = _extract_nested_value(item, keys)
            if found is not None:
                return found
    if isinstance(value, list):
        for item in value:
            found = _extract_nested_value(item, keys)
            if found is not None:
                return found
    return None
