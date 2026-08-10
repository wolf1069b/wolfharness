"""Integration tests for user message display in the OpenCode TUI.

Reproduces and verifies fixes for bugs identified in PR #289/#298:
- Bug 1 (FIXED): ``handle_enqueued_messages()`` FIFO queue reference broken
  by ``or []`` in EventMapper.__init__ — empty list is falsy, so ``[] or []``
  creates a new list instead of preserving the reference.
- Bug 2 (FIXED): ``message_routes.py`` persists to storage AND EventProcessor
  creates message → duplicates.
- Bug 3 (FIXED): ``_route_message()`` used ``source="protocol"`` which doesn't
  trigger steer split (split only accepts ``"processed"`` and ``"accepted"``).
  Fixed by changing default to ``source="accepted"``.
- Bug 4 (FIXED): ``_route_message()`` didn't generate ``message_id`` before
  calling both ``_emit_user_message_inserted()`` and ``steer()``, causing
  each to generate its own ID — dedup couldn't work. Fixed by generating
  ``message_id`` once at the top of ``_route_message()``.

These tests use a **real AgentPool** with **FunctionModel** (deterministic,
no real model calls) and the **full event pipeline** (EventBus → consumer).
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from pydantic_ai.models.function import AgentInfo, FunctionModel
import pytest

from wolfharness import AgentPool, AgentsManifest, NativeAgentConfig
from wolfharness.agents.events.events import (
    StreamCompleteEvent,
    UserMessageInsertedEvent,
)
from wolfharness.lifecycle.types import DeliveryMode
from wolfharness.orchestrator.event_bus import EventEnvelope


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


pytestmark = pytest.mark.integration


def _is_timestamp_id(message_id: str) -> bool:
    """Check if message_id is timestamp-encoded (from ``ascending()``)."""
    return message_id.startswith("msg_")


async def _drain_events(
    queue: asyncio.Queue[EventEnvelope],
    *,
    timeout: float = 10.0,
    until: type | None = None,
) -> list[Any]:
    """Drain events from an EventBus subscription queue."""
    events: list[Any] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            async with asyncio.timeout(remaining):
                envelope = await queue.get()
        except TimeoutError:
            break
        event = envelope.event if isinstance(envelope, EventEnvelope) else envelope
        events.append(event)
        if until is not None and isinstance(event, until):
            break
    return events


def _make_blocking_model(
    release_event: asyncio.Event,
    *,
    response_text: str = "Done processing",
) -> FunctionModel:
    """Create a FunctionModel that blocks until release_event is set."""

    async def _stream_fn(messages: list[Any], info: AgentInfo) -> Any:
        await release_event.wait()
        yield response_text

    return FunctionModel(stream_function=_stream_fn)


@pytest.fixture
async def blocking_pool() -> AsyncIterator[tuple[AgentPool, asyncio.Event, Any]]:
    """AgentPool with a FunctionModel that blocks on an asyncio.Event."""
    release_event = asyncio.Event()
    config = AgentsManifest(
        agents={
            "conductor": NativeAgentConfig(
                name="conductor",
                model="test",
                system_prompt="You are a conductor agent.",
            ),
        },
    )
    async with AgentPool(config) as pool:
        sp = pool.session_pool
        assert sp is not None

        async def setup_session(session_id: str) -> None:
            await sp.sessions.get_or_create_session(session_id, agent_name="conductor")
            agent = await sp.sessions.get_or_create_session_agent(
                session_id, agent_name="conductor"
            )
            agent._model = _make_blocking_model(release_event)

        yield pool, release_event, setup_session


# ---------------------------------------------------------------------------
# Scenario 1: Initial prompt display
# ---------------------------------------------------------------------------


async def test_initial_prompt_display() -> None:
    """Initial prompt produces exactly one UserMessageInsertedEvent."""
    config = AgentsManifest(
        agents={
            "conductor": NativeAgentConfig(name="conductor", model="test", system_prompt="test"),
        },
    )
    async with AgentPool(config) as pool:
        sp = pool.session_pool
        assert sp is not None
        bus = sp.event_bus
        session_id = "test-initial"
        await sp.sessions.get_or_create_session(session_id, agent_name="conductor")
        queue = await bus.subscribe(session_id, scope="session")
        await sp.send_message(session_id, "Hello", mode=DeliveryMode.QUEUE)
        events = await _drain_events(queue, timeout=10.0, until=StreamCompleteEvent)
        user_msgs = [e for e in events if isinstance(e, UserMessageInsertedEvent)]
        assert len(user_msgs) == 1
        assert user_msgs[0].delivery == "initial"
        assert user_msgs[0].content
        assert _is_timestamp_id(user_msgs[0].message_id)


# ---------------------------------------------------------------------------
# Scenario 2: Steer mid-turn — dual event with shared message_id
# ---------------------------------------------------------------------------


async def test_steer_mid_turn_dual_event(
    blocking_pool: tuple[AgentPool, asyncio.Event, Any],
) -> None:
    """Steer mid-turn produces TWO events with the SAME message_id.

    Event 1: source="accepted" (routing time) — immediate display
    Event 2: source="processed" (processing time) — steer split trigger

    Both share the same message_id so EventProcessor dedup catches the second.

    Bug 1 (FIXED): EventMapper.__init__ used ``or []`` which broke FIFO
    reference when list was empty — handle_enqueued_messages() always saw
    empty FIFO and returned None.

    Bug 3 (FIXED): send_message() defaulted to source="protocol" which
    doesn't trigger steer split. Now uses source="accepted".

    Bug 4 (FIXED): _route_message() didn't generate message_id before
    calling both _emit_user_message_inserted() and steer() — each
    generated its own ID, dedup couldn't work.
    """
    pool, release_event, setup_session = blocking_pool
    sp = pool.session_pool
    assert sp is not None
    bus = sp.event_bus

    session_id = "test-steer-dual"
    await setup_session(session_id)
    queue = await bus.subscribe(session_id, scope="session")

    await sp.send_message(session_id, "Start", mode=DeliveryMode.QUEUE)
    await asyncio.sleep(0.3)
    await sp.send_message(session_id, "Steer this", mode=DeliveryMode.STEER)
    await asyncio.sleep(0.1)
    release_event.set()

    events = await _drain_events(queue, timeout=10.0, until=StreamCompleteEvent)
    steer_events = [
        e for e in events if isinstance(e, UserMessageInsertedEvent) and e.delivery == "steer"
    ]

    assert len(steer_events) == 2, (
        f"Expected 2 steer events (internal + enqueued), got {len(steer_events)}. "
        f"Events: {[(e.source, e.message_id[:20]) for e in steer_events]}"
    )

    # Event 1: routing-time (source="accepted")
    routing_event = next(e for e in steer_events if e.source == "accepted")
    assert routing_event.content == "Steer this"
    assert _is_timestamp_id(routing_event.message_id)

    # Event 2: processing-time (source="processed")
    enqueued_event = next(e for e in steer_events if e.source == "processed")
    assert enqueued_event.content == "Steer this"

    # Both events share the same message_id — dedup works
    assert routing_event.message_id == enqueued_event.message_id, (
        f"Message IDs differ: routing={routing_event.message_id}, "
        f"enqueued={enqueued_event.message_id}"
    )

    # Enqueued event arrives BEFORE StreamCompleteEvent
    enqueued_idx = events.index(enqueued_event)
    complete_idx = next(i for i, e in enumerate(events) if isinstance(e, StreamCompleteEvent))
    assert enqueued_idx < complete_idx


# ---------------------------------------------------------------------------
# Scenario 3: Background task steer
# ---------------------------------------------------------------------------


async def test_background_task_steer(
    blocking_pool: tuple[AgentPool, asyncio.Event, Any],
) -> None:
    """steer_from_background_task() produces a display event.

    The event should have source="accepted" (fire-and-forget from
    steer_from_background_task) and content matching the steer message.
    """
    pool, release_event, setup_session = blocking_pool
    sp = pool.session_pool
    assert sp is not None
    bus = sp.event_bus

    session_id = "test-bg-steer"
    await setup_session(session_id)
    queue = await bus.subscribe(session_id, scope="session")

    await sp.send_message(session_id, "Start", mode=DeliveryMode.QUEUE)
    await asyncio.sleep(0.3)
    await sp.steer_from_background_task(session_id, "BG task result: 42")
    await asyncio.sleep(0.1)
    release_event.set()

    events = await _drain_events(queue, timeout=10.0, until=StreamCompleteEvent)
    steer_events = [
        e for e in events if isinstance(e, UserMessageInsertedEvent) and e.delivery == "steer"
    ]

    assert len(steer_events) >= 1, (
        f"Expected at least 1 steer event, got {len(steer_events)}. "
        f"Events: {[(e.source, e.message_id[:20]) for e in steer_events]}"
    )

    # At least one event should have the correct content
    bg_event = next(e for e in steer_events if "BG task result" in str(e.content))
    assert bg_event.content


# ---------------------------------------------------------------------------
# Scenario 4: FIFO reference bug (Bug 1 regression test)
# ---------------------------------------------------------------------------


async def test_fifo_reference_preserved(
    blocking_pool: tuple[AgentPool, asyncio.Event, Any],
) -> None:
    """EventMapper._enqueue_message_ids is the SAME list as run_ctx._pending_enqueue_message_ids.

    Bug 1: ``self._enqueue_message_ids = _enqueue_message_ids or []`` breaks
    reference when the list is empty (``[] or []`` creates a new list).
    Fix: use ``if _enqueue_message_ids is not None else []`` instead.

    This test verifies that after steer() appends to the FIFO,
    handle_enqueued_messages() can pop from it (same list object).
    """
    pool, release_event, setup_session = blocking_pool
    sp = pool.session_pool
    assert sp is not None
    bus = sp.event_bus

    session_id = "test-fifo-ref"
    await setup_session(session_id)
    queue = await bus.subscribe(session_id, scope="session")

    await sp.send_message(session_id, "Start", mode=DeliveryMode.QUEUE)
    await asyncio.sleep(0.3)
    await sp.send_message(session_id, "Steer for FIFO test", mode=DeliveryMode.STEER)
    await asyncio.sleep(0.1)
    release_event.set()

    events = await _drain_events(queue, timeout=10.0, until=StreamCompleteEvent)

    # If FIFO reference is broken, handle_enqueued_messages() returns None
    # and no source="processed" event is produced.
    enqueued_events = [
        e for e in events if isinstance(e, UserMessageInsertedEvent) and e.source == "processed"
    ]
    assert len(enqueued_events) >= 1, (
        "No source='processed' event — FIFO reference is broken. "
        "Check EventMapper.__init__: `or []` creates a new list when empty."
    )


# ---------------------------------------------------------------------------
# Scenario 5: Steer split trigger (Bug 3 regression test)
# ---------------------------------------------------------------------------


async def test_steer_split_triggers_on_internal_source(
    blocking_pool: tuple[AgentPool, asyncio.Event, Any],
) -> None:
    """Routing-time event with source="accepted" triggers steer split.

    Bug 3: send_message() defaulted to source="protocol" which doesn't
    trigger steer split (split only accepts "processed" and "accepted").
    Fix: change default to source="accepted".
    """
    pool, release_event, setup_session = blocking_pool
    sp = pool.session_pool
    assert sp is not None
    bus = sp.event_bus

    session_id = "test-split-trigger"
    await setup_session(session_id)
    queue = await bus.subscribe(session_id, scope="session")

    await sp.send_message(session_id, "Start", mode=DeliveryMode.QUEUE)
    await asyncio.sleep(0.3)
    await sp.send_message(session_id, "Steer for split", mode=DeliveryMode.STEER)
    await asyncio.sleep(0.1)
    release_event.set()

    events = await _drain_events(queue, timeout=10.0, until=StreamCompleteEvent)

    # The routing-time event should have source="accepted" (not "protocol")
    routing_events = [
        e
        for e in events
        if isinstance(e, UserMessageInsertedEvent)
        and e.delivery == "steer"
        and e.source == "accepted"
    ]
    assert len(routing_events) >= 1, (
        "No source='accepted' steer event — send_message() may still default "
        "to source='protocol'. Check session_pool_messaging.py."
    )


# ---------------------------------------------------------------------------
# Scenario 6: Degradation — no EnqueuedMessagesEvent (no active run)
# ---------------------------------------------------------------------------


async def test_degradation_no_enqueued_event() -> None:
    """When no active run, accepted event displays but no processed event.

    Given: A session with NO active run (idle).
    When: A message is sent via send_message with steer mode.
    Then: Only source="accepted" event is produced (routing-time display).
        No source="processed" event is produced because there is no
        EnqueuedMessagesEvent (no active model drain).
    """
    config = AgentsManifest(
        agents={
            "conductor": NativeAgentConfig(name="conductor", model="test", system_prompt="test"),
        },
    )
    async with AgentPool(config) as pool:
        sp = pool.session_pool
        assert sp is not None
        bus = sp.event_bus
        session_id = "test-degradation"
        await sp.sessions.get_or_create_session(session_id, agent_name="conductor")
        queue = await bus.subscribe(session_id, scope="session")

        # Send a message — no active run, so no EnqueuedMessagesEvent
        await sp.send_message(session_id, "No active run steer", mode=DeliveryMode.STEER)

        events = await _drain_events(queue, timeout=5.0, until=StreamCompleteEvent)

        # Should have at least one accepted event
        accepted_events = [
            e for e in events if isinstance(e, UserMessageInsertedEvent) and e.source == "accepted"
        ]
        assert len(accepted_events) >= 1, (
            f"Expected at least 1 source='accepted' event, got {len(accepted_events)}"
        )

        # Should NOT have any processed events (no active run = no model drain)
        processed_events = [
            e for e in events if isinstance(e, UserMessageInsertedEvent) and e.source == "processed"
        ]
        assert len(processed_events) == 0, (
            f"Expected 0 source='processed' events (no active run), got {len(processed_events)}"
        )
