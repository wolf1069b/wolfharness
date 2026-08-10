"""Tests for ToolCallDeferredEvent handling in OpenCode server.

Tests:
- ToolCallDeferredEvent → ToolPart with state: ToolStateRunning and _meta.deferred: true
- _meta.deferred_handle contains correlation ID
- Replayed completed events are deduplicated
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from wolfharness.agents.events.events import ToolCallDeferredEvent
from wolfharness_server.opencode_server.event_processor import EventProcessor
from wolfharness_server.opencode_server.event_processor_context import (
    EventProcessorContext,
)
from wolfharness_server.opencode_server.models import (
    MessagePath,
    MessageTime,
    MessageWithParts,
    PartUpdatedEvent,
)
from wolfharness_server.opencode_server.models.parts import (
    TimeStartEndCompacted,
    ToolPart,
    ToolStateCompleted,
    ToolStateRunning,
)


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from wolfharness_server.opencode_server.state import ServerState


# =============================================================================
# ToolCallDeferredEvent → ToolPart with deferred metadata
# =============================================================================


@pytest.mark.asyncio
async def test_deferred_event_creates_tool_part_running(server_state: ServerState) -> None:
    """ToolCallDeferredEvent→ToolPart with state:ToolStateRunning, metadata.deferred=true.

    Verifies:
    - EventProcessor yields PartUpdatedEvent
    - ToolPart has state=ToolStateRunning
    - ToolPart metadata contains deferred=true
    - ToolPart metadata contains deferred_handle
    - tool_call_id matches
    - tool_name matches
    """
    # GIVEN: empty context with assistant message
    processor = EventProcessor()
    assistant_msg = MessageWithParts.assistant(
        message_id="msg-1",
        session_id="test-session",
        time=MessageTime(created=0),
        agent_name="test-agent",
        model_id="test-model",
        parent_id="parent-1",
        provider_id="wolfharness",
        path=MessagePath(cwd="/tmp", root="/tmp"),
    )
    ctx = EventProcessorContext(
        session_id="test-session",
        assistant_msg_id="msg-1",
        assistant_msg=assistant_msg,
        state=server_state,
        working_dir="/tmp",
    )

    # WHEN: ToolCallDeferredEvent with status='pending' received
    event = ToolCallDeferredEvent(
        tool_call_id="tc-1",
        tool_name="long_running_task",
        deferred_handle="corr-xyz-123",
        deferred_strategy="block",
        status="pending",
        session_id="test-session",
    )
    events = [e async for e in processor.process(event, ctx)]

    # THEN: PartUpdatedEvent is yielded
    assert len(events) == 1
    assert isinstance(events[0], PartUpdatedEvent)

    # AND: ToolPart exists in context
    assert ctx.has_tool_part("tc-1")
    tool_part = ctx.get_tool_part("tc-1")
    assert tool_part is not None
    assert isinstance(tool_part, ToolPart)

    # AND: state is ToolStateRunning
    assert isinstance(tool_part.state, ToolStateRunning)

    # AND: tool_name and call_id match event
    assert tool_part.tool == "long_running_task"
    assert tool_part.call_id == "tc-1"

    # AND: metadata contains deferred=true
    assert tool_part.metadata is not None
    assert tool_part.metadata.get("deferred") is True

    # AND: metadata contains deferred_handle correlation ID
    assert tool_part.metadata.get("deferred_handle") == "corr-xyz-123"


@pytest.mark.asyncio
async def test_deferred_event_sets_title_with_deferred_strategy(
    server_state: ServerState,
) -> None:
    """ToolCallDeferredEvent sets a descriptive title indicating the deferral strategy.

    Verifies:
    - ToolStateRunning title reflects that the tool is deferred
    - Different strategies produce appropriate titles
    """
    # GIVEN: empty context
    processor = EventProcessor()
    assistant_msg = MessageWithParts.assistant(
        message_id="msg-2",
        session_id="test-session",
        time=MessageTime(created=0),
        agent_name="test-agent",
        model_id="test-model",
        parent_id="parent-1",
        provider_id="wolfharness",
        path=MessagePath(cwd="/tmp", root="/tmp"),
    )
    ctx = EventProcessorContext(
        session_id="test-session",
        assistant_msg_id="msg-2",
        assistant_msg=assistant_msg,
        state=server_state,
        working_dir="/tmp",
    )

    # WHEN: ToolCallDeferredEvent with deferred_strategy='block'
    event = ToolCallDeferredEvent(
        tool_call_id="tc-block",
        tool_name="blocked_tool",
        deferred_handle="h-block",
        deferred_strategy="block",
        status="pending",
        session_id="test-session",
    )
    events = [e async for e in processor.process(event, ctx)]

    # THEN: title reflects deferral
    assert len(events) == 1
    tool_part = ctx.get_tool_part("tc-block")
    assert tool_part is not None
    assert isinstance(tool_part.state, ToolStateRunning)
    title = tool_part.state.title
    assert title is not None
    assert "blocked_tool" in title
    assert "deferred" in title.lower()


# =============================================================================
# Deduplication: replayed completed events are skipped
# =============================================================================


@pytest.mark.asyncio
async def test_deferred_event_skipped_when_already_completed(
    server_state: ServerState,
) -> None:
    """ToolCallDeferredEvent is skipped when ToolPart is already completed.

    During session resume, completed deferred events may be replayed.
    The deduplication guard should skip processing when the ToolPart
    for the given tool_call_id already has a completed/error state.

    Verifies:
    - No new events are yielded when ToolPart is already ToolStateCompleted
    - The existing ToolPart is not modified
    """
    # GIVEN: context with an already-completed ToolPart
    processor = EventProcessor()
    assistant_msg = MessageWithParts.assistant(
        message_id="msg-3",
        session_id="test-session",
        time=MessageTime(created=0),
        agent_name="test-agent",
        model_id="test-model",
        parent_id="parent-1",
        provider_id="wolfharness",
        path=MessagePath(cwd="/tmp", root="/tmp"),
    )
    ctx = EventProcessorContext(
        session_id="test-session",
        assistant_msg_id="msg-3",
        assistant_msg=assistant_msg,
        state=server_state,
        working_dir="/tmp",
    )

    # Pre-populate with an already-completed ToolPart
    completed_state = ToolStateCompleted(
        title="Completed blocked_tool",
        input={},
        output="done",
        metadata={"deferred": True, "deferred_handle": "corr-already"},
        time=TimeStartEndCompacted(start=0, end=100),
    )
    existing_tool_part = ToolPart(
        id="part-existing",
        message_id="msg-3",
        session_id="test-session",
        tool="blocked_tool",
        call_id="tc-completed",
        state=completed_state,
        metadata={"deferred": True, "deferred_handle": "corr-already"},
    )
    ctx.add_tool_part("tc-completed", existing_tool_part)
    assistant_msg.parts.append(existing_tool_part)

    # WHEN: A replayed ToolCallDeferredEvent for the same tool_call_id
    event = ToolCallDeferredEvent(
        tool_call_id="tc-completed",
        tool_name="blocked_tool",
        deferred_handle="corr-already",
        deferred_strategy="block",
        status="pending",
        session_id="test-session",
    )
    events = [e async for e in processor.process(event, ctx)]

    # THEN: No new PartUpdatedEvent is yielded (deduplicated)
    assert len(events) == 0

    # AND: The existing ToolPart is unchanged (still completed)
    tp = ctx.get_tool_part("tc-completed")
    assert tp is not None
    assert isinstance(tp.state, ToolStateCompleted)


@pytest.mark.asyncio
async def test_deferred_event_skipped_when_already_errored(
    server_state: ServerState,
) -> None:
    """ToolCallDeferredEvent is skipped when ToolPart is already in error state.

    Similar to the completed dedup test, but for ToolStateError.
    """
    from wolfharness_server.opencode_server.models.parts import (
        TimeStartEnd,
        ToolStateError,
    )

    # GIVEN: context with an already-errored ToolPart
    processor = EventProcessor()
    assistant_msg = MessageWithParts.assistant(
        message_id="msg-4",
        session_id="test-session",
        time=MessageTime(created=0),
        agent_name="test-agent",
        model_id="test-model",
        parent_id="parent-1",
        provider_id="wolfharness",
        path=MessagePath(cwd="/tmp", root="/tmp"),
    )
    ctx = EventProcessorContext(
        session_id="test-session",
        assistant_msg_id="msg-4",
        assistant_msg=assistant_msg,
        state=server_state,
        working_dir="/tmp",
    )

    error_state = ToolStateError(
        error="something went wrong",
        input={},
        time=TimeStartEnd(start=0, end=100),
        metadata={"deferred": True, "deferred_handle": "corr-error"},
    )
    existing_tool_part = ToolPart(
        id="part-error",
        message_id="msg-4",
        session_id="test-session",
        tool="failing_tool",
        call_id="tc-error",
        state=error_state,
        metadata={"deferred": True, "deferred_handle": "corr-error"},
    )
    ctx.add_tool_part("tc-error", existing_tool_part)
    assistant_msg.parts.append(existing_tool_part)

    # WHEN: A replayed ToolCallDeferredEvent for the same tool_call_id
    event = ToolCallDeferredEvent(
        tool_call_id="tc-error",
        tool_name="failing_tool",
        deferred_handle="corr-error",
        deferred_strategy="continue",
        status="pending",
        session_id="test-session",
    )
    events = [e async for e in processor.process(event, ctx)]

    # THEN: No new PartUpdatedEvent is yielded (deduplicated)
    assert len(events) == 0

    # AND: The existing ToolPart remains errored
    tp = ctx.get_tool_part("tc-error")
    assert tp is not None
    assert isinstance(tp.state, ToolStateError)
