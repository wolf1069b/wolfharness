"""Unit tests for ACPEventConverter tool call event handling fixes.

Tests three specific bug fixes in event_converter.py:
- Bug 1: PartDeltaEvent with ToolCallPartDelta yields no notifications
- Bug 2: FunctionToolCallEvent handler removed (dead code)
- Bug 3: ToolCallProgressEvent extracts tool_input/tool_name and emits raw_input
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import PartDeltaEvent, ToolCallPartDelta
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
    results: list[Any] = []
    results.extend([update async for update in converter.convert(event)])
    return results


# ---------------------------------------------------------------------------
# Bug 1: PartDeltaEvent with ToolCallPartDelta is a no-op
# ---------------------------------------------------------------------------


async def test_part_delta_event_with_none_tool_call_id_yields_nothing() -> None:
    """PartDeltaEvent with tool_call_id=None yields no notifications.

    Previously, delta.as_part() generated a random tool_call_id, creating
    spurious _ToolState entries and ToolCallStart notifications.
    """
    converter = ACPEventConverter()
    event = PartDeltaEvent(
        index=0,
        delta=ToolCallPartDelta(tool_name_delta="bash"),
    )
    # tool_call_id is None on this delta
    delta = event.delta
    assert isinstance(delta, ToolCallPartDelta)
    assert delta.tool_call_id is None

    notifications = await _collect(converter, event)

    assert notifications == [], "PartDeltaEvent should yield no notifications"
    # No tool state should have been created
    assert converter._tool_states == {}, "No tool state should be created"


async def test_part_delta_event_with_known_tool_call_id_no_new_state_or_start() -> None:
    """PartDeltaEvent with a known tool_call_id doesn't create new state or emit ToolCallStart.

    If a tool state already exists (created by ToolCallStartEvent), PartDeltaEvent
    must not create a duplicate state or emit another ToolCallStart.
    """
    converter = ACPEventConverter()

    # Pre-populate tool state via ToolCallStartEvent (the correct path)
    start_event = ToolCallStartEvent(
        tool_call_id="tc-existing",
        tool_name="bash",
        title="Running bash",
        kind="execute",
        locations=[],
        raw_input={"command": "echo hello"},
    )
    start_notifications = await _collect(converter, start_event)
    assert len(start_notifications) == 1
    assert isinstance(start_notifications[0], ToolCallStart)

    # Now send a PartDeltaEvent with the same tool_call_id
    delta_event = PartDeltaEvent(
        index=0,
        delta=ToolCallPartDelta(
            tool_name_delta="bash",
            tool_call_id="tc-existing",
        ),
    )
    delta_notifications = await _collect(converter, delta_event)

    assert delta_notifications == [], "PartDeltaEvent should yield no notifications"
    # Tool state should still be the one from ToolCallStartEvent, not duplicated
    assert len(converter._tool_states) == 1
    state = converter._tool_states["tc-existing"]
    assert state.tool_name == "bash"
    assert state.started is True


# ---------------------------------------------------------------------------
# Bug 2: FunctionToolCallEvent handler is removed (dead code)
# ---------------------------------------------------------------------------


async def test_function_tool_call_event_not_handled() -> None:
    """FunctionToolCallEvent is not handled by ACPEventConverter.

    EventMapper intercepts FunctionToolCallEvent and converts it to
    ToolCallStartEvent/ToolCallProgressEvent before it reaches the converter.
    The handler was dead code and has been removed.
    """
    from pydantic_ai import FunctionToolCallEvent, ToolCallPart

    converter = ACPEventConverter()
    part = ToolCallPart(
        tool_name="bash",
        args='{"command": "echo test"}',
        tool_call_id="tc-func-1",
    )
    event = FunctionToolCallEvent(part=part)

    notifications = await _collect(converter, event)

    # Should fall through to the `case _` default handler (no notifications)
    assert notifications == [], (
        "FunctionToolCallEvent should not produce notifications — "
        "it is intercepted by EventMapper before reaching the converter"
    )
    # No tool state should have been created
    assert converter._tool_states == {}


# ---------------------------------------------------------------------------
# Bug 3: ToolCallProgressEvent extracts tool_input/tool_name
# ---------------------------------------------------------------------------


async def test_tool_call_progress_event_with_tool_input_updates_state_and_emits_raw_input() -> None:
    """ToolCallProgressEvent with tool_input updates state and emits raw_input.

    When ToolCallProgressEvent carries tool_input and tool_name, the converter
    should:
    1. Use tool_name instead of "unknown" when creating state
    2. Store tool_input in state.raw_input
    3. Include raw_input in the yielded ToolCallProgress
    """
    converter = ACPEventConverter()
    event = ToolCallProgressEvent(
        tool_call_id="tc-progress-1",
        tool_name="read",
        tool_input={"path": "/tmp/test.txt"},
        title="Reading file",
        status="in_progress",
    )

    notifications = await _collect(converter, event)

    # First notification should be ToolCallStart (since state didn't exist)
    assert len(notifications) >= 1
    start = notifications[0]
    assert isinstance(start, ToolCallStart)
    assert start.tool_call_id == "tc-progress-1"
    assert start.kind == "read"

    # Second notification should be ToolCallProgress with raw_input
    assert len(notifications) == 2
    progress = notifications[1]
    assert isinstance(progress, ToolCallProgress)
    assert progress.tool_call_id == "tc-progress-1"
    assert progress.status == "in_progress"
    assert progress.raw_input == {"path": "/tmp/test.txt"}

    # State should have tool_name and raw_input properly set
    state = converter._tool_states["tc-progress-1"]
    assert state.tool_name == "read"
    assert state.raw_input == {"path": "/tmp/test.txt"}


async def test_tool_call_progress_event_with_none_tool_input_preserves_existing_state() -> None:
    """ToolCallProgressEvent with tool_input=None preserves existing state.

    When ToolCallProgressEvent has tool_input=None (e.g., a progress update
    without input data), the converter should not overwrite existing state
    with empty data.
    """
    converter = ACPEventConverter()

    # First, create state via ToolCallStartEvent with raw_input
    start_event = ToolCallStartEvent(
        tool_call_id="tc-progress-2",
        tool_name="bash",
        title="Running command",
        kind="execute",
        locations=[],
        raw_input={"command": "ls -la"},
    )
    await _collect(converter, start_event)

    # Verify state was created correctly
    state = converter._tool_states["tc-progress-2"]
    assert state.tool_name == "bash"
    assert state.raw_input == {"command": "ls -la"}

    # Now send a progress event without tool_input or tool_name
    progress_event = ToolCallProgressEvent(
        tool_call_id="tc-progress-2",
        title="Still running...",
        status="in_progress",
    )

    notifications = await _collect(converter, progress_event)

    # Should yield a ToolCallProgress (update, not start, since state.started=True)
    assert len(notifications) == 1
    progress = notifications[0]
    assert isinstance(progress, ToolCallProgress)
    assert progress.tool_call_id == "tc-progress-2"
    assert progress.status == "in_progress"
    # raw_input should still be the original value from ToolCallStartEvent
    assert progress.raw_input == {"command": "ls -la"}

    # State should be unchanged
    state = converter._tool_states["tc-progress-2"]
    assert state.tool_name == "bash"
    assert state.raw_input == {"command": "ls -la"}


if __name__ == "__main__":
    pytest.main(["-v", __file__])
