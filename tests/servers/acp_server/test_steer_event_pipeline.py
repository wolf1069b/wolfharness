"""L2 integration test: steer → EventBus → ACPEventConverter → UserMessageChunk.

Verifies the complete UserMessageInsertedEvent pipeline:
1. RunHandle.steer() publishes UserMessageInsertedEvent to EventBus
2. EventBus subscriber receives the event
3. ACPEventConverter converts the event to UserMessageChunk (ACP v1)
4. UserMessageChunk has correct message_id and content

This test uses a real EventBus and real ACPEventConverter — no mocking of
the event pipeline. Only the agent_run is mocked to control steer timing.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from acp.schema import UserMessageChunk
from wolfharness.agents.events.events import UserMessageInsertedEvent
from wolfharness.lifecycle.comm_channel import DirectChannel
from wolfharness.lifecycle.journal import MemoryJournal
from wolfharness.messaging import MessageHistory
from wolfharness.orchestrator.core import EventBus, SessionState
from wolfharness.orchestrator.run import RunHandle
from wolfharness_server.acp_server.event_converter import ACPEventConverter


pytestmark = [pytest.mark.integration, pytest.mark.anyio]


async def test_steer_to_acp_converter_pipeline() -> None:
    """Full pipeline: steer() → EventBus → ACPEventConverter → UserMessageChunk.

    Given: A RunHandle with a real EventBus and a mocked active agent_run.
    When: steer() is called with emit_user_message=True.
    Then: UserMessageInsertedEvent appears on the EventBus.
    And: ACPEventConverter converts it to a UserMessageChunk.
    And: The UserMessageChunk has the steer content and a unique message_id.
    """
    import asyncio

    bus = EventBus()
    session = SessionState(session_id="integ-session", agent_name="test-agent")
    session._comm_channel = DirectChannel(MemoryJournal())

    agent = MagicMock()
    agent.name = "test-agent"
    agent.conversation = MessageHistory()

    handle = RunHandle(
        run_id="integ-run",
        session_id="integ-session",
        agent_type="native",
        agent=agent,
        event_bus=bus,
        session=session,
    )

    mock_agent_run = MagicMock()
    handle.active_agent_run = mock_agent_run
    # Force fire-and-forget emission path — the test uses a mock agent_run
    # that won't trigger EnqueuedMessagesEvent, so the skip-emission logic
    # would suppress the event.
    handle._enqueued_messages_available = False

    # Subscribe to EventBus (returns asyncio.Queue[EventEnvelope])
    queue = await bus.subscribe("integ-session", scope="session")

    # Act: steer with user message emission
    handle.steer("additional context", emit_user_message=True)

    # Wait for fire-and-forget task
    if handle._emission_tasks:
        await asyncio.gather(*handle._emission_tasks)

    # Drain the queue to collect events
    received_events: list[Any] = []
    while not queue.empty():
        envelope = queue.get_nowait()
        received_events.append(envelope.event)

    # Assert 1: UserMessageInsertedEvent published to EventBus
    user_msg_events = [e for e in received_events if isinstance(e, UserMessageInsertedEvent)]
    assert len(user_msg_events) == 1, (
        f"Expected 1 UserMessageInsertedEvent, got {len(user_msg_events)}"
    )
    event = user_msg_events[0]
    assert event.delivery == "steer"
    assert event.source == "accepted"
    assert event.content == "additional context"
    assert event.message_id  # non-empty

    # Assert 2: ACPEventConverter converts to UserMessageChunk (v1)
    converter = ACPEventConverter()
    updates = [u async for u in converter.convert(event)]

    assert len(updates) == 1
    chunk = updates[0]
    assert isinstance(chunk, UserMessageChunk)
    assert chunk.message_id == event.message_id
    assert chunk.content.text == "additional context"
