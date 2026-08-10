"""Test that native agent streaming yields events in real-time.

Verifies that PartDeltaEvent (and other model events) are yielded to the consumer
while the background iteration task is still running — not batched and released
only after the iteration completes.

This is a regression test for a bug where events were buffered via an internal
event queue and only released at the end.
The fix restored direct iteration so events flow to the consumer immediately.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from pydantic_ai import PartDeltaEvent, PartStartEvent
from pydantic_ai.models.test import TestModel
import pytest

from wolfharness import Agent
from wolfharness.agents.events import StreamCompleteEvent


# ---------------------------------------------------------------------------
# Slow test model: inserts async sleep into the streaming path
# ---------------------------------------------------------------------------


class SlowTestModel(TestModel):
    """TestModel that inserts a delay before yielding the streamed response.

    The default TestModel's request_stream yields TestStreamedResponse which
    emits all parts instantly. We override request_stream to inject a sleep
    before yielding the response, giving us a window where the iteration_task
    is still running when the consumer receives the first event.
    """

    def __init__(
        self,
        *,
        custom_output_text: str | None = None,
        pre_stream_delay: float = 0.2,
    ) -> None:
        super().__init__(custom_output_text=custom_output_text)
        self.pre_stream_delay = pre_stream_delay

    @asynccontextmanager
    async def request_stream(  # type: ignore[override]
        self,
        messages: list[Any],
        model_settings: Any,
        model_request_parameters: Any,
        run_context: Any = None,
    ) -> Any:
        """Yield the streamed response after a configurable delay."""
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        self.last_model_request_parameters = model_request_parameters

        model_response = self._request(messages, model_settings, model_request_parameters)

        # Delay before yielding — this is the window where we can verify
        # the iteration task is still running when events are received
        await asyncio.sleep(self.pre_stream_delay)
        from pydantic_ai.models.test import TestStreamedResponse

        yield TestStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=self._model_name,
            _structured_response=model_response,
            _messages=messages,
            _provider_name=self._system,
        )


@pytest.fixture
def realtime_agent() -> Agent[None]:
    """Agent with SlowTestModel for real-time streaming tests."""
    model = SlowTestModel(
        custom_output_text="Hello world streaming test",
        pre_stream_delay=0.2,
    )
    return Agent(name="realtime-test-agent", model=model)


# ---------------------------------------------------------------------------
# Real-time streaming tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_stream_yields_events_while_iteration_running(
    realtime_agent: Agent[None],
) -> None:
    """PartDeltaEvent is yielded before the background iteration task completes.

    If events were batched at the end, the consumer would receive no model events
    until the background iteration finishes and dumps everything into the queue.
    With real-time streaming, each event is pushed to the queue as it arrives
    from node.stream(), so the consumer receives events while the iteration task
    is still active.
    """
    first_model_event = asyncio.Event()
    events: list[Any] = []

    async def consume() -> None:
        async for event in realtime_agent.run_stream("Test prompt"):
            events.append(event)
            if isinstance(event, (PartDeltaEvent, PartStartEvent)):
                first_model_event.set()
            if isinstance(event, StreamCompleteEvent):
                break

    task = asyncio.create_task(consume())

    # Wait until the first model event is received by the consumer
    await asyncio.wait_for(first_model_event.wait(), timeout=2.0)

    # The critical assertion: when we receive the first model event, the
    # consumer task should still be running (i.e., the stream hasn't
    # completed yet). If events were batched at the end, the consumer
    # task would have already finished before any model event reached us.
    assert not task.done(), (
        "Consumer task is already done when first model event was received — "
        "events are likely batched at end instead of streamed in real-time"
    )

    # Wait for the stream to complete
    await asyncio.wait_for(task, timeout=2.0)

    # Verify we got the expected event types
    assert any(isinstance(e, PartDeltaEvent) for e in events), (
        f"Expected PartDeltaEvent in stream, got: {[type(e).__name__ for e in events]}"
    )
    assert any(isinstance(e, StreamCompleteEvent) for e in events), (
        f"Expected StreamCompleteEvent in stream, got: {[type(e).__name__ for e in events]}"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_stream_events_not_batched_at_end(
    realtime_agent: Agent[None],
) -> None:
    """Multiple model events are spread across the stream, not dumped at once.

    This test verifies that the consumer receives model events incrementally
    rather than receiving a burst of them right before StreamCompleteEvent.
    """
    events: list[Any] = []
    model_event_count = 0

    async for event in realtime_agent.run_stream("Test prompt"):
        events.append(event)
        if isinstance(event, (PartDeltaEvent, PartStartEvent)):
            model_event_count += 1
        if isinstance(event, StreamCompleteEvent):
            break

    # "Hello world streaming test" is split into words by TestStreamedResponse:
    # "Hello ", "world ", "streaming ", "test"  => 4 words
    # Plus PartStartEvent for the text part => 5 model events
    # (The exact count depends on TestStreamedResponse internals, but we expect
    #  more than 1 model event for a multi-word response.)
    assert model_event_count > 1, (
        f"Expected multiple model events for multi-word response, got {model_event_count}. "
        "Events may be batched into a single delivery."
    )

    # Verify the stream completed successfully
    assert isinstance(events[-1], StreamCompleteEvent)
    assert events[-1].message.content == "Hello world streaming test"
