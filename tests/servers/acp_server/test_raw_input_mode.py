"""Tests for raw_input_mode config in ACPEventConverter.

Tests three modes:
- "dict": raw_input is a dict (default)
- "skip": raw_input is None in ToolCallStart, delivered via ToolCallProgress
- "json_str": raw_input is a JSON string
"""

from __future__ import annotations

from typing import Any

import pytest

from acp.schema import ToolCallProgress, ToolCallStart
from wolfharness.agents.events.events import (
    ToolCallProgressEvent,
    ToolCallStartEvent,
)
from wolfharness_server.acp_server.event_converter import ACPEventConverter


pytestmark = [pytest.mark.unit, pytest.mark.anyio]


async def _collect(converter: ACPEventConverter, event: Any) -> list[Any]:
    """Collect all notifications yielded by converter.convert(event)."""
    return [update async for update in converter.convert(event)]


def _make_start_event(
    tool_call_id: str = "tc-1",
    tool_name: str = "bash",
    raw_input: dict[str, Any] | None = None,
) -> ToolCallStartEvent:
    return ToolCallStartEvent(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        title=f"Executing: {tool_name}",
        kind="other",
        raw_input=raw_input or {"command": "ls -la"},
    )


async def test_dict_mode_emits_dict_raw_input() -> None:
    """In 'dict' mode, raw_input is the parsed dict."""
    converter = ACPEventConverter(raw_input_mode="dict")
    event = _make_start_event(raw_input={"command": "ls"})

    notifications = await _collect(converter, event)

    start = notifications[0]
    assert isinstance(start, ToolCallStart)
    assert start.raw_input == {"command": "ls"}


async def test_dict_mode_empty_raw_input_is_none() -> None:
    """In 'dict' mode, empty raw_input becomes None."""
    converter = ACPEventConverter(raw_input_mode="dict")
    event = ToolCallStartEvent(
        tool_call_id="tc-empty",
        tool_name="bash",
        title="Executing: bash",
        kind="other",
        raw_input={},
    )

    notifications = await _collect(converter, event)

    start = notifications[0]
    assert isinstance(start, ToolCallStart)
    assert start.raw_input is None


async def test_skip_mode_emits_none_in_tool_call_start() -> None:
    """In 'skip' mode, ToolCallStart has raw_input=None."""
    converter = ACPEventConverter(raw_input_mode="skip")
    event = _make_start_event(raw_input={"command": "ls"})

    notifications = await _collect(converter, event)

    start = notifications[0]
    assert isinstance(start, ToolCallStart)
    assert start.raw_input is None


async def test_skip_mode_delivers_raw_input_in_progress() -> None:
    """In 'skip' mode, raw_input is delivered via ToolCallProgressEvent."""
    converter = ACPEventConverter(raw_input_mode="skip")

    start_event = _make_start_event(raw_input={"command": "ls"})
    await _collect(converter, start_event)

    progress_event = ToolCallProgressEvent(
        tool_call_id="tc-1",
        tool_name="bash",
        tool_input={"command": "ls -la /tmp"},
        status="in_progress",
    )
    notifications = await _collect(converter, progress_event)

    progress = notifications[-1]
    assert isinstance(progress, ToolCallProgress)
    assert progress.raw_input is None


async def test_json_str_mode_emits_json_string() -> None:
    """In 'json_str' mode, raw_input is a JSON string."""
    import json

    converter = ACPEventConverter(raw_input_mode="json_str")
    event = _make_start_event(raw_input={"command": "ls"})

    notifications = await _collect(converter, event)

    start = notifications[0]
    assert isinstance(start, ToolCallStart)
    assert isinstance(start.raw_input, str)
    assert json.loads(start.raw_input) == {"command": "ls"}


async def test_json_str_mode_empty_raw_input_is_none() -> None:
    """In 'json_str' mode, empty raw_input becomes None."""
    converter = ACPEventConverter(raw_input_mode="json_str")
    event = ToolCallStartEvent(
        tool_call_id="tc-empty",
        tool_name="bash",
        title="Executing: bash",
        kind="other",
        raw_input={},
    )

    notifications = await _collect(converter, event)

    start = notifications[0]
    assert isinstance(start, ToolCallStart)
    assert start.raw_input is None


async def test_json_str_mode_progress_emits_json_string() -> None:
    """In 'json_str' mode, ToolCallProgress also emits JSON string."""
    import json

    converter = ACPEventConverter(raw_input_mode="json_str")

    start_event = _make_start_event(raw_input={"command": "ls"})
    await _collect(converter, start_event)

    progress_event = ToolCallProgressEvent(
        tool_call_id="tc-1",
        tool_name="bash",
        tool_input={"command": "ls -la /tmp"},
        status="in_progress",
    )
    notifications = await _collect(converter, progress_event)

    progress = notifications[-1]
    assert isinstance(progress, ToolCallProgress)
    assert isinstance(progress.raw_input, str)
    assert json.loads(progress.raw_input) == {"command": "ls -la /tmp"}


async def test_json_str_mode_preserves_unicode() -> None:
    """In 'json_str' mode, unicode chars are preserved (ensure_ascii=False)."""
    converter = ACPEventConverter(raw_input_mode="json_str")
    event = _make_start_event(raw_input={"path": "/tmp/中文文件.txt"})

    notifications = await _collect(converter, event)

    start = notifications[0]
    assert isinstance(start, ToolCallStart)
    assert isinstance(start.raw_input, str)
    assert "中文" in start.raw_input
