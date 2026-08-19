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
    _client: Any = field(default=None, init=False, repr=False)
    _pending: dict[str, list[dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
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
                event_id = (
                    f"rpc:{direction}:{rpc_id}" if rpc_id is not None else f"rpc:{direction}"
                )
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
                    self.flush_on_turn_complete
                    and _should_flush_protocol_event(method, event_type)
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
        if not records:
            return
        try:
            client = await self._ensure_client()
            uri = self._events_uri(session_id)
            content = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
            await client.write(uri, content, mode="append", wait=False, processing_mode="raw")
            await client.write(
                self._memory_context_uri(session_id),
                json.dumps(
                    {
                        "session_id": session_id,
                        "archive_session_id": self._archive_session_id(session_id),
                        "events_uri": uri,
                        "last_event_sequence": records[-1]["sequence"],
                        "last_event_type": records[-1]["event_type"],
                        "updated_at": datetime.now(UTC).isoformat(),
                        "memory_policy": "archive-all-events-memory-selected-transcript",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                mode="replace",
                wait=False,
                processing_mode="raw",
            )
        except _ERROR_TYPES:
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
        return client

    def _events_uri(self, session_id: str) -> str:
        return f"viking://sessions/{self._archive_session_id(session_id)}/events.jsonl"

    def _memory_context_uri(self, session_id: str) -> str:
        return f"viking://sessions/{self._archive_session_id(session_id)}/memory_context.json"

    def _archive_session_id(self, session_id: str) -> str:
        return f"{self.session_prefix}-{_safe_identifier(session_id)}"


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
    return event_type == "rpc_notification" and method.startswith("session/")


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
