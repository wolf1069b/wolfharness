"""Tests for reasoning/thinking part behavior in OpenCode event processor."""

from unittest.mock import MagicMock

from pydantic_ai.messages import (
    PartDeltaEvent as PydanticPartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)
import pytest

from wolfharness_server.opencode_server.event_processor import EventProcessor
from wolfharness_server.opencode_server.event_processor_context import (
    EventProcessorContext,
)
from wolfharness_server.opencode_server.models import PartUpdatedEvent
from wolfharness_server.opencode_server.models.events import (
    PartUpdatedEventProperties,
)
from wolfharness_server.opencode_server.models.parts import (
    ReasoningPart,
    TextPart as OpenCodeTextPart,
)


pytestmark = pytest.mark.integration


def _make_processor_and_ctx(
    mock_msg: MagicMock | None = None,
) -> tuple[EventProcessor, EventProcessorContext]:
    """Create an EventProcessor and EventProcessorContext for testing."""
    if mock_msg is None:
        mock_msg = MagicMock()
        mock_msg.parts = []
    mock_state = MagicMock()
    processor = EventProcessor()
    ctx = EventProcessorContext(
        session_id="test-session",
        assistant_msg_id="msg-1",
        assistant_msg=mock_msg,
        state=mock_state,
        working_dir=".",
    )
    return processor, ctx


async def _process_event(
    processor: EventProcessor,
    ctx: EventProcessorContext,
    event: object,
) -> list[object]:
    """Process a single event and return the resulting events."""
    return [e async for e in processor.process(event, ctx)]  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_thinking_events_create_reasoning_part():
    """Verify ThinkingPart/ThinkingPartDelta events create ReasoningPart."""
    mock_msg = MagicMock()
    mock_msg.parts = []

    processor, ctx = _make_processor_and_ctx(mock_msg)

    events = await _process_event(
        processor, ctx, PartStartEvent(index=0, part=ThinkingPart(content="Thinking..."))
    )
    events.extend(
        await _process_event(
            processor,
            ctx,
            PydanticPartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=" more...")),
        )
    )

    # Assert reasoning part was created and content accumulated
    reasoning_parts = []
    for e in events:
        if isinstance(e, PartUpdatedEvent):
            props = e.properties
            if isinstance(props, PartUpdatedEventProperties) and isinstance(
                props.part, ReasoningPart
            ):
                reasoning_parts.append(props.part)

    # Verify accumulation via context
    assert ctx.reasoning_part is not None, "ReasoningPart should be created"
    assert "Thinking..." in ctx.reasoning_part.text
    assert " more..." in ctx.reasoning_part.text


@pytest.mark.asyncio
async def test_multi_turn_thinking_creates_separate_parts():
    """Verify that multiple thinking phases create separate ReasoningParts.

    This tests the fix for: "Multi-turn conversation thinking displayed in single block"
    Each thinking phase should be its own Part with its own ID.
    """
    mock_msg = MagicMock()
    mock_msg.parts = []

    processor, ctx = _make_processor_and_ctx(mock_msg)

    events: list[object] = []

    # Simulate multi-turn conversation with thinking in each turn:
    # Turn 1: Thinking -> Text
    events.extend(
        await _process_event(
            processor, ctx, PartStartEvent(index=0, part=ThinkingPart(content="First thinking..."))
        )
    )
    events.extend(
        await _process_event(
            processor,
            ctx,
            PydanticPartDeltaEvent(
                index=0, delta=ThinkingPartDelta(content_delta=" more thinking")
            ),
        )
    )
    # End of thinking - text response starts
    events.extend(
        await _process_event(
            processor, ctx, PartStartEvent(index=1, part=TextPart(content="First response"))
        )
    )

    # Turn 2: Thinking -> Text (new turn, should be separate Part)
    events.extend(
        await _process_event(
            processor,
            ctx,
            PartStartEvent(index=2, part=ThinkingPart(content="Second turn thinking...")),
        )
    )
    events.extend(
        await _process_event(
            processor,
            ctx,
            PydanticPartDeltaEvent(index=2, delta=ThinkingPartDelta(content_delta=" more")),
        )
    )
    # End of thinking - text response starts
    events.extend(
        await _process_event(
            processor,
            ctx,
            PydanticPartDeltaEvent(index=3, delta=TextPartDelta(content_delta="Second response")),
        )
    )

    # Turn 3: Thinking (should be third separate Part)
    events.extend(
        await _process_event(
            processor,
            ctx,
            PartStartEvent(index=4, part=ThinkingPart(content="Third turn thinking...")),
        )
    )

    # Extract ReasoningParts from events
    reasoning_parts = []
    for e in events:
        if isinstance(e, PartUpdatedEvent):
            props = e.properties
            if isinstance(props, PartUpdatedEventProperties) and isinstance(
                props.part, ReasoningPart
            ):
                reasoning_parts.append(props.part)

    # Extract TextParts to verify they were created correctly
    text_parts = []
    for e in events:
        if isinstance(e, PartUpdatedEvent):
            props = e.properties
            if isinstance(props, PartUpdatedEventProperties) and isinstance(
                props.part, OpenCodeTextPart
            ):
                text_parts.append(props.part)

    # Assertions
    # We need to check that there are 3 unique reasoning phases (unique Part IDs)
    # Each thinking start creates a new Part, and deltas update the same Part
    unique_reasoning_parts: dict[str, ReasoningPart] = {}
    for p in reasoning_parts:
        unique_reasoning_parts[p.id] = p

    # We should have 3 separate ReasoningParts (one for each thinking phase)
    assert len(unique_reasoning_parts) >= 3, (
        f"Expected at least 3 unique ReasoningParts (one per thinking phase), "
        f"got {len(unique_reasoning_parts)} unique IDs from {len(reasoning_parts)} events"
    )

    # Get the unique parts (one per thinking phase) sorted by creation order
    unique_parts_list = list(unique_reasoning_parts.values())

    # Verify the content is not accumulated across turns
    # Each unique part should represent one thinking phase
    first_thinking = unique_parts_list[0].text
    second_thinking = unique_parts_list[1].text if len(unique_parts_list) > 1 else ""
    third_thinking = unique_parts_list[2].text if len(unique_parts_list) > 2 else ""

    # Each thinking should only have that turn's content
    assert "First thinking..." in first_thinking, (
        f"First thinking content missing: {first_thinking}"
    )
    assert "Second turn thinking..." in second_thinking, (
        f"Second thinking content missing: {second_thinking}"
    )
    assert "Third turn thinking..." in third_thinking, (
        f"Third thinking content missing: {third_thinking}"
    )

    # Verify no cross-contamination - second thinking shouldn't have first thinking's content
    assert "First thinking" not in second_thinking, (
        f"Second thinking has first turn's content: {second_thinking}"
    )
    assert "First thinking" not in third_thinking, (
        f"Third thinking has first turn's content: {third_thinking}"
    )

    # Verify text parts were created correctly
    assert len(text_parts) >= 1, f"Expected at least 1 TextPart, got {len(text_parts)}"


@pytest.mark.asyncio
async def test_single_thinking_phase_accumulates_correctly():
    """Verify that a single thinking phase still accumulates correctly."""
    mock_msg = MagicMock()
    mock_msg.parts = []

    processor, ctx = _make_processor_and_ctx(mock_msg)

    events: list[object] = []

    # Single thinking phase with multiple deltas
    events.extend(
        await _process_event(
            processor, ctx, PartStartEvent(index=0, part=ThinkingPart(content="Start "))
        )
    )
    events.extend(
        await _process_event(
            processor,
            ctx,
            PydanticPartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="middle ")),
        )
    )
    events.extend(
        await _process_event(
            processor,
            ctx,
            PydanticPartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="end")),
        )
    )

    # Verify accumulation via context (not just events)
    # PartDeltaEvent yields deltas, not full parts, so check context directly
    assert ctx.reasoning_part is not None, "ReasoningPart should be created"

    # The content should be accumulated in the context
    final_content = ctx.reasoning_part.text
    expected = "Start middle end"
    assert final_content == expected, f"Expected '{expected}', got '{final_content}'"


@pytest.mark.asyncio
async def test_reasoning_part_gets_end_time_when_text_starts():
    """Verify that ReasoningPart gets time.end set when text starts (thinking ends)."""
    mock_msg = MagicMock()
    mock_msg.parts = []
    mock_msg.update_part = MagicMock()

    processor, ctx = _make_processor_and_ctx(mock_msg)

    events: list[object] = []

    # Thinking phase
    events.extend(
        await _process_event(
            processor, ctx, PartStartEvent(index=0, part=ThinkingPart(content="Thinking..."))
        )
    )

    # Text starts - this should close out the reasoning part
    events.extend(
        await _process_event(
            processor, ctx, PartStartEvent(index=1, part=TextPart(content="Response"))
        )
    )

    # The reasoning part should have been updated with an end time
    reasoning_final_events = [
        e
        for e in events
        if isinstance(e, PartUpdatedEvent)
        and isinstance(e.properties, PartUpdatedEventProperties)
        and isinstance(e.properties.part, ReasoningPart)
        and e.properties.part.time is not None
        and e.properties.part.time.end is not None
    ]

    assert len(reasoning_final_events) >= 1, (
        "Expected at least one ReasoningPart with time.end set when text starts"
    )

    # The context should have cleared the reasoning_part reference
    assert ctx.reasoning_part is None, (
        "reasoning_part should be cleared from context after text starts"
    )


@pytest.mark.asyncio
async def test_reasoning_part_gets_end_time_on_stream_complete():
    """Verify that ReasoningPart gets time.end set on stream complete if still active."""
    mock_msg = MagicMock()
    mock_msg.parts = []
    mock_msg.update_part = MagicMock()

    processor, ctx = _make_processor_and_ctx(mock_msg)

    events: list[object] = []

    # Thinking phase only (no text follows)
    events.extend(
        await _process_event(
            processor, ctx, PartStartEvent(index=0, part=ThinkingPart(content="Thinking..."))
        )
    )

    # Stream completes without any text starting
    from wolfharness.messaging import ChatMessage

    chat_msg = ChatMessage(content="", role="assistant")
    events.extend(list(processor._process_stream_complete(ctx, chat_msg)))

    # The reasoning part should have been finalized with an end time
    reasoning_final_events = [
        e
        for e in events
        if isinstance(e, PartUpdatedEvent)
        and isinstance(e.properties, PartUpdatedEventProperties)
        and isinstance(e.properties.part, ReasoningPart)
        and e.properties.part.time is not None
        and e.properties.part.time.end is not None
    ]

    assert len(reasoning_final_events) >= 1, (
        "Expected ReasoningPart with time.end set on stream complete"
    )

    # The context should have cleared the reasoning_part reference
    assert ctx.reasoning_part is None, (
        "reasoning_part should be cleared from context after stream complete"
    )
