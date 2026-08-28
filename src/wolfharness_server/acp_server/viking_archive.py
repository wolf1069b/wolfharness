"""Viking archive for ACP session updates."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import os
import re
from typing import Any

import logfire

from wolfharness.log import get_logger


logger = get_logger(__name__)

_ERROR_TYPES = (RuntimeError, OSError, TimeoutError, ValueError, TypeError, ImportError)
_MIN_SIGNAL_CJK_CHARS = 4
_MIN_SIGNAL_ALNUM_CHARS = 6
_MIN_SIGNAL_TOTAL_CHARS = 12
_ACK_RE = re.compile(
    r"^(?:ok|okay|k|yes|yep|no|nope|thanks|thank you|thx|done|"
    r"收到|好的|好|嗯|可以|继续|不用|不需要|没了|好了)[.!?\u3002\uff01\uff1f\s]*$",
    re.IGNORECASE,
)


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
    transcript_keep_recent_count: int = 0
    _client: Any = field(default=None, init=False, repr=False)
    _pending_update_count: dict[str, int] = field(default_factory=dict, init=False, repr=False)
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
    _tasks: set[asyncio.Task[None]] = field(default_factory=set, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    @classmethod
    def from_config(cls, config: Any) -> ACPVikingEventArchive:
        """Build an archive from manifest config, expanding env vars lazily."""
        if config is None or getattr(config, "enabled", False) is not True:
            return cls(enabled=False)
        api_key = _expand(getattr(config, "api_key", None) or os.getenv("VIKING_MCP_API_KEY"))
        url = _expand(getattr(config, "url", None) or os.getenv("VIKING_MCP_URL"))
        if not url:
            logger.warning("ACP Viking archive disabled: missing Viking URL")
            return cls(enabled=False)
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
            transcript_keep_recent_count=int(getattr(config, "transcript_keep_recent_count", 0)),
        )

    async def record_update(
        self,
        *,
        consumer_session_id: str,
        source_session_id: str,
        event_id: Any,
        update: Any,
    ) -> None:
        """Buffer an ACP update for readable transcript archival."""
        if not self.enabled:
            return
        _ = event_id
        try:
            payload = _dump_update(update)
            event_type = str(payload.get("sessionUpdate") or payload.get("session_update") or "")
            session_id = source_session_id or consumer_session_id
            async with self._lock:
                if self.transcript_enabled:
                    self._buffer_transcript_update_locked(session_id, payload, event_type)
                pending_count = self._pending_update_count.get(session_id, 0) + 1
                self._pending_update_count[session_id] = pending_count
                should_flush = pending_count >= self.batch_size or (
                    self.flush_on_turn_complete and event_type == "turn_complete"
                )
            if should_flush:
                self.schedule_flush(session_id)
        except _ERROR_TYPES:
            logger.warning("ACP Viking archive record failed", exc_info=True)

    def schedule_flush(self, session_id: str) -> None:
        """Schedule a background flush for one session."""
        if not self.enabled:
            return
        task = asyncio.create_task(self._flush_session_with_span(session_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _flush_session_with_span(self, session_id: str) -> None:
        with logfire.span("acp_viking_archive.flush_session", session_id=session_id):
            await self.flush_session(session_id)

    async def flush_session(self, session_id: str) -> None:
        """Flush buffered transcript messages for one session to Viking."""
        async with self._lock:
            pending_count = self._pending_update_count.pop(session_id, 0)
            transcript_messages = self._pop_transcript_locked(session_id)
        if not transcript_messages:
            return
        try:
            client = await self._ensure_client()
            await self._write_transcript(
                client,
                session_id,
                transcript_messages,
            )
        except _ERROR_TYPES:
            async with self._lock:
                if pending_count:
                    self._pending_update_count[session_id] = (
                        self._pending_update_count.get(session_id, 0) + pending_count
                    )
                if transcript_messages:
                    self._transcript_pending.setdefault(session_id, transcript_messages[:0])[:0] = (
                        transcript_messages
                    )
            logger.warning("ACP Viking archive flush failed", session_id=session_id, exc_info=True)

    async def flush_all(self) -> None:
        """Flush all buffered events and await in-flight writes."""
        session_ids = set(self._pending_update_count) | set(self._transcript_pending)
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
            or f"{event_type}:{self._pending_update_count.get(session_id, 0)}"
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
                if isinstance(text, str) and _should_capture_text(text):
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


def _should_capture_text(text: str) -> bool:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact:
        return False
    if compact.lower().startswith("[openviking-memory]"):
        return False
    if _ACK_RE.match(compact):
        return False
    if not re.search(r"[a-z0-9\u3400-\u9fff]", compact, re.IGNORECASE):
        return False
    cjk = len(re.findall(r"[\u3400-\u9fff]", compact))
    alnum = len(re.findall(r"[a-z0-9]", compact, re.IGNORECASE))
    return (
        cjk >= _MIN_SIGNAL_CJK_CHARS
        or alnum >= _MIN_SIGNAL_ALNUM_CHARS
        or len(compact) >= _MIN_SIGNAL_TOTAL_CHARS
    )


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
