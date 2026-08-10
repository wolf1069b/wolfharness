"""L2 integration test: steer_from_background_task → EventBus → ACPEventConverter.

Verifies the full chain from background task steer to ACP UserMessageChunk output:
1. ``steer_from_background_task()`` publishes ``UserMessageInsertedEvent`` to EventBus
2. ACPEventConverter receives the event and yields ``UserMessageChunk``
3. The ``UserMessageChunk`` contains the steer message text
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from wolfharness.agents.events.events import UserMessageInsertedEvent
from wolfharness_server.acp_server.event_converter import ACPEventConverter


if TYPE_CHECKING:
    from wolfharness.delegation.pool import AgentPool


pytestmark = [pytest.mark.integration]


def _unwrap(raw: Any) -> Any:
    """Unwrap an EventBus envelope to get the event."""
    if hasattr(raw, "event"):
        return raw.event
    return raw


async def test_steer_from_background_task_acp_converter(
    minimal_pool: AgentPool,
) -> None:
    """Verify ACPEventConverter converts the background task steer event to UserMessageChunk.

    Full chain: steer_from_background_task() → EventBus → ACPEventConverter → UserMessageChunk
    """
    from acp.schema import UserMessageChunk

    session_pool = minimal_pool.session_pool
    assert session_pool is not None

    session_id = "test-bg-steer-acp-converter"
    await session_pool.create_session(session_id, agent_name="test_agent")

    # Patch agent create_turn to block so we have an active run handle
    release = asyncio.Event()

    async def _blocking_create_turn(*args: Any, **kwargs: Any) -> Any:
        await release.wait()
        from wolfharness.agents.events.events import StreamCompleteEvent
        from wolfharness.messaging.messagenode import ChatMessage

        return StreamCompleteEvent(message=ChatMessage(role="assistant", content="done"))

    agent = await session_pool.sessions.get_or_create_session_agent(session_id)
    agent.create_turn = _blocking_create_turn  # type: ignore[method-assign]

    queue = await session_pool.event_bus.subscribe(session_id, scope="session")

    # Start a run — _consume_run launches as a background task.
    msg_id = await session_pool.send_message(session_id, "initial prompt")
    assert msg_id is not None
    await asyncio.sleep(0.1)  # Let the turn start and block.

    # Call steer_from_background_task
    steer_text = "ACP_STEER_TEST_MARKER"
    result = await session_pool.steer_from_background_task(session_id, steer_text)
    assert result is not None, "steer_from_background_task() returned None"

    # Collect events from the queue
    all_events: list[Any] = []
    try:
        async with asyncio.timeout(3.0):
            while True:
                try:
                    raw = await asyncio.wait_for(queue.get(), timeout=1.0)
                except TimeoutError:
                    break
                all_events.append(_unwrap(raw))
    except TimeoutError:
        pass

    # Release the blocking turn
    release.set()

    # Find UserMessageInsertedEvent with source="accepted" and delivery="steer"
    # (filter by delivery to exclude the initial prompt event which also has
    # source="accepted").
    bg_events = [
        e
        for e in all_events
        if isinstance(e, UserMessageInsertedEvent)
        and e.source == "accepted"
        and e.delivery == "steer"
    ]
    assert len(bg_events) >= 1, (
        f"Expected UserMessageInsertedEvent(source='accepted', delivery='steer'); "
        f"got {len(bg_events)} from {len(all_events)} total events. "
        f"Event types: {[type(e).__name__ for e in all_events]}"
    )

    # Feed the event through ACPEventConverter
    converter = ACPEventConverter()
    chunks = [
        update
        async for update in converter.convert(bg_events[0])
        if isinstance(update, UserMessageChunk)
    ]

    assert len(chunks) >= 1, (
        f"Expected at least one UserMessageChunk from ACPEventConverter; got {len(chunks)}"
    )

    chunk = chunks[0]
    assert chunk.content is not None
    assert steer_text in chunk.content.text  # type: ignore[union-attr]
