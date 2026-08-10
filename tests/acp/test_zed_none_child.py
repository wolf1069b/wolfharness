"""Tests for ``child_session_id=None`` graceful fallback in zed mode.

Verifies that ``SubAgentEvent`` with ``child_session_id=None`` (or other
unknown/missing values) in zed mode does not raise ``KeyError`` or crash,
matching the plan requirement in ``tasks.md 6.8``.

Ref: ``src/wolfharness_server/acp_server/event_converter.py:690-693``
"""

from __future__ import annotations

from pydantic_ai import PartStartEvent, TextPart, TextPartDelta, ThinkingPart
from pydantic_ai.usage import RequestUsage
import pytest

from wolfharness.agents.events import (
    PartDeltaEvent,
    RunErrorEvent,
    SpawnSessionStart,
    StreamCompleteEvent,
    SubAgentEvent,
)
from wolfharness.messaging.messages import ChatMessage
from wolfharness_server.acp_server.event_converter import ACPEventConverter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def zed_converter() -> ACPEventConverter:
    """Converter configured for zed subagent display mode."""
    c = ACPEventConverter(subagent_display_mode="zed")
    c._current_message_id = "test-msg-id"
    return c


# ---------------------------------------------------------------------------
# child_session_id=None with various inner event types
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_none_child_id_text_part_does_not_crash(
    zed_converter: ACPEventConverter,
) -> None:
    """SubAgentEvent with child_session_id=None and TextPart must not crash."""
    inner_event = PartStartEvent(index=0, part=TextPart(content="Hello"))
    sub_event = SubAgentEvent(
        source_name="helper",
        source_type="agent",
        event=inner_event,
        depth=1,
        child_session_id=None,
    )
    updates = [u async for u in zed_converter.convert(sub_event)]
    assert len(updates) == 0


@pytest.mark.unit
async def test_none_child_id_text_delta_does_not_crash(
    zed_converter: ACPEventConverter,
) -> None:
    """SubAgentEvent with child_session_id=None and TextPartDelta must not crash."""
    inner_event = PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=" more text"))
    sub_event = SubAgentEvent(
        source_name="helper",
        source_type="agent",
        event=inner_event,
        depth=1,
        child_session_id=None,
    )
    updates = [u async for u in zed_converter.convert(sub_event)]
    assert len(updates) == 0


@pytest.mark.unit
async def test_none_child_id_thinking_part_does_not_crash(
    zed_converter: ACPEventConverter,
) -> None:
    """SubAgentEvent with child_session_id=None and ThinkingPart must not crash."""
    inner_event = PartStartEvent(index=0, part=ThinkingPart(content="thinking..."))
    sub_event = SubAgentEvent(
        source_name="helper",
        source_type="agent",
        event=inner_event,
        depth=1,
        child_session_id=None,
    )
    updates = [u async for u in zed_converter.convert(sub_event)]
    assert len(updates) == 0


@pytest.mark.unit
async def test_none_child_id_stream_complete_does_not_crash(
    zed_converter: ACPEventConverter,
) -> None:
    """SubAgentEvent with child_session_id=None and StreamCompleteEvent must not crash."""
    message = ChatMessage(
        content="Done",
        role="assistant",  # type: ignore[arg-type]
        usage=RequestUsage(),
    )
    inner_event = StreamCompleteEvent(message=message)
    sub_event = SubAgentEvent(
        source_name="helper",
        source_type="agent",
        event=inner_event,
        depth=1,
        child_session_id=None,
    )
    updates = [u async for u in zed_converter.convert(sub_event)]
    assert len(updates) == 0


@pytest.mark.unit
async def test_none_child_id_run_error_does_not_crash(
    zed_converter: ACPEventConverter,
) -> None:
    """SubAgentEvent with child_session_id=None and RunErrorEvent must not crash."""
    inner_event = RunErrorEvent(message="Something went wrong", agent_name="helper")
    sub_event = SubAgentEvent(
        source_name="helper",
        source_type="agent",
        event=inner_event,
        depth=1,
        child_session_id=None,
    )
    updates = [u async for u in zed_converter.convert(sub_event)]
    assert len(updates) == 0


# ---------------------------------------------------------------------------
# Empty string and unknown session ID
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_empty_child_id_does_not_crash(
    zed_converter: ACPEventConverter,
) -> None:
    """SubAgentEvent with child_session_id='' in zed mode must not crash."""
    inner_event = PartStartEvent(index=0, part=TextPart(content="Hello"))
    sub_event = SubAgentEvent(
        source_name="helper",
        source_type="agent",
        event=inner_event,
        depth=1,
        child_session_id="",
    )
    updates = [u async for u in zed_converter.convert(sub_event)]
    assert len(updates) == 0


@pytest.mark.unit
async def test_unknown_child_id_does_not_crash(
    zed_converter: ACPEventConverter,
) -> None:
    """SubAgentEvent with unknown child_session_id in zed mode must not crash."""
    inner_event = PartStartEvent(index=0, part=TextPart(content="Hello"))
    sub_event = SubAgentEvent(
        source_name="helper",
        source_type="agent",
        event=inner_event,
        depth=1,
        child_session_id="nonexistent_session",
    )
    updates = [u async for u in zed_converter.convert(sub_event)]
    assert len(updates) == 0


# ---------------------------------------------------------------------------
# Mixing valid spawns with None child_session_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_none_child_after_valid_spawn_does_not_crash(
    zed_converter: ACPEventConverter,
) -> None:
    """SubAgentEvent with child_session_id=None after a valid spawn must not crash.

    This validates the scenario where a subagent was properly spawned (so the
    converter has state), but the SubAgentEvent arrives with ``None`` instead
    of the real child_session_id.
    """
    # First, register a child session via SpawnSessionStart
    spawn_event = SpawnSessionStart(
        child_session_id="real_child",
        parent_session_id="parent",
        tool_call_id="tc_001",
        spawn_mechanism="spawn",
        source_name="coder",
        source_type="agent",
        description="Coding",
    )
    _ = [u async for u in zed_converter.convert(spawn_event)]

    # Now send SubAgentEvent with None child_session_id
    inner_event = PartStartEvent(index=0, part=TextPart(content="Hello"))
    sub_event = SubAgentEvent(
        source_name="helper",
        source_type="agent",
        event=inner_event,
        depth=1,
        child_session_id=None,
    )
    updates = [u async for u in zed_converter.convert(sub_event)]
    assert len(updates) == 0


# ---------------------------------------------------------------------------
# Multiple inner event types with None child_session_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_multiple_none_child_events_do_not_crash(
    zed_converter: ACPEventConverter,
) -> None:
    """Multiple SubAgentEvents with None child_session_id and various inner events.

    Must not crash.
    """
    events = [
        SubAgentEvent(
            source_name="helper",
            source_type="agent",
            event=PartStartEvent(index=0, part=TextPart(content="Hello")),
            depth=1,
            child_session_id=None,
        ),
        SubAgentEvent(
            source_name="helper",
            source_type="agent",
            event=PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=" world")),
            depth=1,
            child_session_id=None,
        ),
        SubAgentEvent(
            source_name="helper",
            source_type="agent",
            event=PartStartEvent(index=2, part=ThinkingPart(content="thinking...")),
            depth=1,
            child_session_id=None,
        ),
        SubAgentEvent(
            source_name="helper",
            source_type="agent",
            event=StreamCompleteEvent(
                message=ChatMessage(
                    content="Done",
                    role="assistant",  # type: ignore[arg-type]
                    usage=RequestUsage(),
                )
            ),
            depth=1,
            child_session_id=None,
        ),
    ]

    all_updates: list[object] = []
    for event in events:
        all_updates.extend([update async for update in zed_converter.convert(event)])

    # All None child_session_id events must silently skip — no updates
    assert len(all_updates) == 0
