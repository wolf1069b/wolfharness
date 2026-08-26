"""Per-session file logging event handler for wiki build team agents.

Each agent instance (identified by session_id) gets its own log file,
ensuring that parallel extraction_worker instances don't interleave output.

File naming: ./logs/{agent_name}_{session_id[:8]}.log

Ported from xeno_nmt_harness.event_handlers with wiki-specific adaptations:
- Tool result truncation limits tuned for wiki-build capability tools
- Handles wiki team tool names (write_entity, read_resource, etc.)
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic_ai import (
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)

from wolfharness.agents.events import (
    CompactionEvent,
    RunErrorEvent,
    RunFailedEvent,
    RunStartedEvent,
    SpawnSessionStart,
    StreamCompleteEvent,
    ToolCallCompleteEvent,
    ToolCallStartEvent,
    UserMessageInsertedEvent,
)


if TYPE_CHECKING:
    from wolfharness.agents.context import AgentContext
    from wolfharness.agents.events import RichAgentStreamEvent

LOG_DIR = Path("./logs")

# Module-level file handle cache: "{agent_name}_{session_id}" -> file handle
_file_handles: dict[str, Any] = {}

# Thinking buffer: "{agent_name}_{session_id}" -> accumulated thinking text.
# Thinking arrives as many small ThinkingPartDelta fragments; we buffer them
# and flush as a single [thinking] block when text output begins or the turn
# completes, producing readable consolidated thinking sections.
_thinking_buffers: dict[str, str] = {}

# Maximum characters of a tool result to write to the log file.  Longer
# results are truncated with a notice showing the original length.
_MAX_TOOL_RESULT_CHARS = 2000

# Per-tool overrides for the truncation limit.  Tools listed here use the
# specified limit instead of _MAX_TOOL_RESULT_CHARS.  Use math.inf for no
# truncation.
_TOOL_RESULT_LIMITS: dict[str, int | float] = {
    "read_resource": math.inf,
    "get_resource": math.inf,
    "read_blackboard": math.inf,
    "list_children": math.inf,
    "read_chapter": math.inf,
    "get_backlinks": math.inf,
}


def _timestamp() -> str:
    return datetime.datetime.now(datetime.UTC).astimezone().strftime("%H:%M:%S.%f")[:-3]


def _get_file_handle(agent_name: str, session_id: str) -> Any:
    key = f"{agent_name}_{session_id}"
    if key not in _file_handles:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = LOG_DIR / f"{agent_name}_{session_id}.log"
        _file_handles[key] = file_path.open("a", encoding="utf-8")
    return _file_handles[key]


def _close_file_handle(agent_name: str, session_id: str) -> None:
    key = f"{agent_name}_{session_id}"
    handle = _file_handles.pop(key, None)
    if handle is not None:
        handle.close()
    _thinking_buffers.pop(key, None)


def _flush_thinking(agent_name: str, session_id: str, file_handle: Any) -> None:
    """Flush accumulated thinking content to the log file as a single block."""
    key = f"{agent_name}_{session_id}"
    buffered = _thinking_buffers.pop(key, "")
    if buffered:
        file_handle.write(f"\n[{_timestamp()}] [thinking] {buffered}\n")
        file_handle.flush()


def _content_to_str(content: str | list[Any]) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text") or item.get("content") or ""
            if text:
                parts.append(str(text))
    return " ".join(parts)


def _truncate(content: Any, max_chars: float = _MAX_TOOL_RESULT_CHARS) -> str:
    """Truncate content to *max_chars*, appending a notice if truncated.

    ``max_chars`` accepts ``float('inf')`` to disable truncation entirely.
    """
    text = str(content)
    if math.isinf(max_chars) or len(text) <= max_chars:
        return text
    return f"{text[: int(max_chars)]}\n... (truncated, {len(text)} chars total)"


async def per_session_file_handler(
    ctx: AgentContext[Any],
    event: RichAgentStreamEvent[Any],
) -> None:
    """Event handler that writes agent output to per-session log files.

    Each agent session gets its own log file, so parallel instances
    (e.g., multiple extraction_worker instances processing different chapters)
    produce isolated logs without interleaving.

    Log file path: ./logs/{agent_name}_{session_id[:8]}.log

    Records:
    - Run start markers with run_id and agent name
    - Model text output (streaming)
    - Thinking content (buffered, flushed as a single block)
    - Tool calls and results (prefixed with [tool] / [result])
    - Team member spawns (SpawnSessionStart)
    - Team messages and user messages (UserMessageInsertedEvent)
    - Errors (prefixed with [error])
    - Turn-complete separators
    """
    agent_name = ctx.node_name
    session_id = ""
    if ctx.run_ctx is not None:
        session_id = ctx.run_ctx.session_id

    # Fallback: try to get session_id from the event
    if not session_id:
        session_id = getattr(event, "session_id", "") or ""

    if not session_id:
        return

    file_handle = _get_file_handle(agent_name, session_id)

    match event:
        case RunStartedEvent(run_id=run_id, agent_name=ev_agent_name):
            file_handle.write(
                f"\n{'=' * 70}\n[{_timestamp()}] [run-start] run_id={run_id} agent={ev_agent_name or agent_name}\n{'=' * 70}\n",
            )
            file_handle.flush()

        case (
            PartStartEvent(part=TextPart(content=delta))
            | PartDeltaEvent(delta=TextPartDelta(content_delta=delta))
        ):
            # Flush any pending thinking before writing text output
            _flush_thinking(agent_name, session_id, file_handle)
            if delta:
                file_handle.write(delta)
                file_handle.flush()

        case (
            PartStartEvent(part=ThinkingPart(content=delta))
            | PartDeltaEvent(delta=ThinkingPartDelta(content_delta=delta))
        ):
            # Buffer thinking fragments instead of writing each one separately
            if delta:
                key = f"{agent_name}_{session_id}"
                _thinking_buffers[key] = _thinking_buffers.get(key, "") + delta

        case ToolCallStartEvent() as ev:
            # Flush any pending thinking before writing tool call
            _flush_thinking(agent_name, session_id, file_handle)
            if ev.tool_name == "send_message":
                to = ev.raw_input.get("to", "?")
                msg_type = ev.raw_input.get("type", "private")
                content = ev.raw_input.get("content", "")
                file_handle.write(
                    f"\n[{_timestamp()}] [team-message-sent] to={to} type={msg_type}: {content}\n",
                )
            else:
                kwargs_str = ", ".join(f"{k}={v!r}" for k, v in ev.raw_input.items())
                file_handle.write(
                    f"\n[{_timestamp()}] [tool] {ev.tool_name}({kwargs_str})\n",
                )
            file_handle.flush()

        case ToolCallCompleteEvent() as ev:
            # If start event had empty raw_input (streaming timing issue),
            # log the complete tool_input here so params are visible.
            if ev.tool_input:
                kwargs_str = ", ".join(f"{k}={v!r}" for k, v in ev.tool_input.items())
                file_handle.write(
                    f"[{_timestamp()}] [tool-args] {ev.tool_name}({kwargs_str})\n",
                )
            limit = _TOOL_RESULT_LIMITS.get(ev.tool_name, _MAX_TOOL_RESULT_CHARS)
            truncated = _truncate(ev.tool_result, max_chars=limit)
            file_handle.write(f"[{_timestamp()}] [result] {truncated}\n")
            file_handle.flush()

        case RunErrorEvent(message=message):
            _flush_thinking(agent_name, session_id, file_handle)
            file_handle.write(f"\n[{_timestamp()}] [error] {message}\n")
            file_handle.flush()

        case SpawnSessionStart(
            child_session_id=child_sid,
            source_name=source,
            description=desc,
        ):
            short_child = child_sid[:8] if len(child_sid) >= 8 else child_sid
            file_handle.write(
                f"\n[{_timestamp()}] [spawn] {source} -> {short_child}: {desc}\n",
            )
            file_handle.flush()

        case UserMessageInsertedEvent(content=content, delivery=delivery) as ev:
            text = _content_to_str(content)
            if "<team-message" in text:
                file_handle.write(
                    f"\n[{_timestamp()}] [team-message] delivery={delivery}:\n{text}\n",
                )
            else:
                file_handle.write(
                    f"\n[{_timestamp()}] [user-message] delivery={delivery}:\n{text}\n",
                )
            file_handle.flush()

        case StreamCompleteEvent():
            _flush_thinking(agent_name, session_id, file_handle)
            file_handle.write(
                f"\n{'\u2500' * 70}\n[{_timestamp()}] [turn-complete]\n{'\u2500' * 70}\n",
            )
            file_handle.flush()
            _close_file_handle(agent_name, session_id)

        case RunFailedEvent(exception=exc):
            _flush_thinking(agent_name, session_id, file_handle)
            file_handle.write(
                f"\n[{_timestamp()}] [run-failed] {type(exc).__name__}: {exc}\n",
            )
            file_handle.flush()
            _close_file_handle(agent_name, session_id)

        case CompactionEvent(trigger=trig, phase=phase):
            _flush_thinking(agent_name, session_id, file_handle)
            file_handle.write(
                f"\n[{_timestamp()}] [compaction] trigger={trig} phase={phase}\n",
            )
            file_handle.flush()
