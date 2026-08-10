"""Test that native agent streaming emits events in correct order.

Verifies: (intermediate events) -> StreamCompleteEvent.
RunStartedEvent is published by RunHandle.start() to EventBus,
not yielded in the standalone stream.
"""

from __future__ import annotations

from pydantic_ai.models.test import TestModel
import pytest

from wolfharness import Agent
from wolfharness.agents.events import StreamCompleteEvent


TEST_RESPONSE = "I am a test response"


@pytest.fixture
def ordering_agent() -> Agent[None]:
    """Agent with instant TestModel for event ordering testing."""
    model = TestModel(custom_output_text=TEST_RESPONSE)
    return Agent(name="ordering-test-agent", model=model)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_streaming_event_ordering(ordering_agent: Agent[None]) -> None:
    """Last event must be StreamCompleteEvent with correct content.

    RunStartedEvent is published by RunHandle.start() to EventBus,
    not yielded in the standalone stream path.
    """
    events = []

    events.extend([event async for event in ordering_agent.run_stream("Hello")])

    assert len(events) >= 1, "Expected at least StreamCompleteEvent"

    last_event = events[-1]

    assert isinstance(last_event, StreamCompleteEvent), (
        f"Last event must be StreamCompleteEvent, got {type(last_event).__name__}"
    )
    assert last_event.message is not None, "StreamCompleteEvent.message must not be None"
    assert last_event.message.content == TEST_RESPONSE

    # Ensure StreamCompleteEvent is strictly the final event
    stream_complete_count = sum(1 for e in events if isinstance(e, StreamCompleteEvent))
    assert stream_complete_count == 1, (
        f"Expected exactly one StreamCompleteEvent, got {stream_complete_count}"
    )
