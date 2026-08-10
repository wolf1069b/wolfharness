"""Red flag test: verify reasoning events flow through SessionPool to EventBus.

This test verifies the end-to-end event flow when model produces reasoning output
through the SessionPool orchestration layer.
"""

import asyncio
from typing import Any

from pydantic_ai.messages import (
    PartDeltaEvent,
    PartStartEvent,
    ThinkingPart,
    ThinkingPartDelta,
)
import pytest

from wolfharness.agents.events import (
    RunStartedEvent,
)
from wolfharness.orchestrator.core import EventBus


pytestmark = pytest.mark.unit


async def _drain_queue(queue: asyncio.Queue[Any]) -> list[Any]:
    """Drain all items from an asyncio.Queue until QueueShutDown."""
    items: list[Any] = []
    while True:
        try:
            items.append(await queue.get())
        except asyncio.QueueShutDown:
            break
    return items


@pytest.mark.asyncio
async def test_reasoning_events_published_to_eventbus():
    """Red-flag: Verify that reasoning/thinking events are published to EventBus.

    And can be consumed by subscribers.
    """
    event_bus = EventBus()
    session_id = "test_session"

    # Subscribe to events
    queue = await event_bus.subscribe(session_id, scope="session")

    # Simulate publishing thinking events (as would happen in agent stream)
    thinking_start = PartStartEvent(index=0, part=ThinkingPart(content="Let me analyze"))
    thinking_delta = PartDeltaEvent(
        index=0, delta=ThinkingPartDelta(content_delta=" this problem...")
    )
    run_started = RunStartedEvent(session_id=session_id, run_id="run1")

    await event_bus.publish(session_id, run_started)
    await event_bus.publish(session_id, thinking_start)
    await event_bus.publish(session_id, thinking_delta)
    await event_bus.close_session(session_id)  # shuts down queues

    collected = await _drain_queue(queue)

    await event_bus.unsubscribe(session_id, queue)

    # Verify thinking events are received
    thinking_events = [
        e for e in collected if isinstance(e.event, (PartStartEvent, PartDeltaEvent))
    ]
    assert len(thinking_events) == 2, f"Expected 2 thinking events, got: {thinking_events}"
    assert isinstance(thinking_events[0].event.part, ThinkingPart)
    assert thinking_events[0].event.part.content == "Let me analyze"
    assert thinking_events[1].event.delta.content_delta == " this problem..."


@pytest.mark.asyncio
async def test_eventbus_preserves_event_types_after_copy():
    """Red-flag: EventBus uses copy.copy() before publishing to each subscriber.

    Verify that copied thinking events maintain their type and content.
    """
    event_bus = EventBus()
    session_id = "test_session"

    # Multiple subscribers to trigger copy.copy path
    queue1 = await event_bus.subscribe(session_id, scope="session")
    queue2 = await event_bus.subscribe(session_id, scope="session")

    thinking_start = PartStartEvent(index=0, part=ThinkingPart(content="Deep thinking..."))
    await event_bus.publish(session_id, thinking_start)
    await event_bus.close_session(session_id)

    # Verify both subscribers got the event with correct type
    for queue in [queue1, queue2]:
        collected = await _drain_queue(queue)

        assert len(collected) == 1
        event = collected[0]
        assert isinstance(event.event, PartStartEvent)
        assert isinstance(event.event.part, ThinkingPart)
        assert event.event.part.content == "Deep thinking..."

    await event_bus.unsubscribe(session_id, queue1)
    await event_bus.unsubscribe(session_id, queue2)


@pytest.mark.asyncio
async def test_multiple_subscribers_receive_reasoning():
    """Red-flag: Verify all subscribers receive reasoning events.

    This simulates the scenario where both the adapter_task and _event_consumer_loop
    subscribe to the same EventBus.
    """
    event_bus = EventBus()
    session_id = "test_session"

    # Simulate adapter subscriber (like in message_routes)
    adapter_queue = await event_bus.subscribe(session_id, scope="session")

    # Simulate consumer subscriber (like _event_consumer_loop)
    consumer_queue = await event_bus.subscribe(session_id, scope="session")

    # Publish thinking events
    for i in range(3):
        await event_bus.publish(
            session_id,
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=f"chunk{i}")),
        )
    await event_bus.close_session(session_id)

    # Both queues should receive all events
    adapter_events, consumer_events = await asyncio.gather(
        _drain_queue(adapter_queue),
        _drain_queue(consumer_queue),
    )

    assert len(adapter_events) == 3
    assert len(consumer_events) == 3
    for e in adapter_events:
        assert isinstance(e.event.delta, ThinkingPartDelta)

    await event_bus.unsubscribe(session_id, adapter_queue)
    await event_bus.unsubscribe(session_id, consumer_queue)


@pytest.mark.asyncio
async def test_eventbus_with_subagent_wrapping():
    """Red-flag: Verify that events wrapped in SubAgentEvent still contain.

    reasoning events that can be extracted.
    """
    from wolfharness.agents.events import SubAgentEvent

    event_bus = EventBus()
    parent_session = "parent"
    child_session = "child"

    # Set up parent-child relationship in session tree
    event_bus._session_tree[parent_session] = [child_session]

    # Subscribe to parent with descendants scope (like _event_consumer_loop)
    queue = await event_bus.subscribe(parent_session, scope="descendants")

    # Create a reasoning event wrapped in SubAgentEvent
    thinking_event = PartStartEvent(index=0, part=ThinkingPart(content="Subagent thinking..."))
    subagent_event = SubAgentEvent(
        source_name="subagent",
        source_type="agent",
        event=thinking_event,
        child_session_id=child_session,
        parent_session_id=parent_session,
    )

    # Publish to child session
    await event_bus.publish(child_session, subagent_event)
    # Sentinel must be sent to parent session since that's what the queue is subscribed to
    await event_bus.close_session(parent_session)

    # Parent subscriber should receive it
    collected = await _drain_queue(queue)

    assert len(collected) == 1
    assert isinstance(collected[0].event, SubAgentEvent)
    assert isinstance(collected[0].event.event, PartStartEvent)
    assert isinstance(collected[0].event.event.part, ThinkingPart)

    await event_bus.unsubscribe(parent_session, queue)
