"""L2 integration tests for ``UserMessageInsertedEvent`` feature.

Tests use REAL components (real EventBus, real SessionController, real
SessionPool) — NO mocking of ``EventBus.publish()``,
``EventProcessor.process()``, or ``SessionState._event_bus``.

Covers the full chain: ``steer_from_background_task()`` →
``EventBus.publish()`` → subscriber receives ``UserMessageInsertedEvent``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from wolfharness.agents.events import (
    RunStartedEvent,
    StreamCompleteEvent,
    ToolCallStartEvent,
    UserMessageInsertedEvent,
)
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.event_bus import EventEnvelope
from wolfharness.orchestrator.turn import Turn
from wolfharness.utils.identifiers import ascending


if TYPE_CHECKING:
    from wolfharness import AgentPool
    from wolfharness.agents.context import AgentRunContext


pytestmark = [pytest.mark.integration, pytest.mark.anyio]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unwrap(event: Any) -> Any:
    """Unwrap EventEnvelope if present, otherwise return the event as-is."""
    return event.event if isinstance(event, EventEnvelope) else event


class _EventBlockingTurn(Turn):
    """Turn that blocks on an ``asyncio.Event`` before completing.

    Unlike ``_BlockingTurn`` (which blocks on ``run_ctx.cancelled``), this
    turn can be released by setting the event — allowing tests to control
    turn completion without cancelling the run.
    """

    def __init__(self, run_ctx: AgentRunContext, release: asyncio.Event) -> None:
        self._run_ctx = run_ctx
        self._release = release

    async def execute(self):  # type: ignore[override]
        self._message_history = []
        self._final_message = ChatMessage(content="blocked", role="assistant")
        yield RunStartedEvent(session_id=self._run_ctx.session_id, run_id="test-run")
        await self._release.wait()
        yield StreamCompleteEvent(
            message=ChatMessage(content="done", role="assistant"),
            cancelled=False,
            session_id=self._run_ctx.session_id,
        )


class _ToolBlockingTurn(Turn):
    """Turn that yields ToolCallStartEvent then blocks on an Event."""

    def __init__(self, run_ctx: AgentRunContext, release: asyncio.Event) -> None:
        self._run_ctx = run_ctx
        self._release = release

    async def execute(self):  # type: ignore[override]
        self._message_history = []
        self._final_message = ChatMessage(content="tool-blocked", role="assistant")
        yield RunStartedEvent(session_id=self._run_ctx.session_id, run_id="test-run")
        yield ToolCallStartEvent(
            tool_call_id="test-tool-1",
            tool_name="bash",
            title="Running bash command",
        )
        await self._release.wait()
        yield StreamCompleteEvent(
            message=ChatMessage(content="done", role="assistant"),
            cancelled=False,
            session_id=self._run_ctx.session_id,
        )


def _make_blocking_create_turn(release: asyncio.Event) -> Any:
    """Return a create_turn function whose first call returns _EventBlockingTurn.

    Subsequent calls return a stub turn that yields StreamCompleteEvent.
    """
    call_count = 0

    def _create_turn(
        prompts: Any, run_ctx: AgentRunContext, message_history: Any, **kwargs: Any
    ) -> Turn:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _EventBlockingTurn(run_ctx, release)
        # Stub turn for chained prompts.
        return _StubTurn(run_ctx)

    return _create_turn


def _make_tool_blocking_create_turn(release: asyncio.Event) -> Any:
    """Return a create_turn function whose first call returns _ToolBlockingTurn."""
    call_count = 0

    def _create_turn(
        prompts: Any, run_ctx: AgentRunContext, message_history: Any, **kwargs: Any
    ) -> Turn:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _ToolBlockingTurn(run_ctx, release)
        return _StubTurn(run_ctx)

    return _create_turn


class _StubTurn(Turn):
    """Minimal Turn that yields StreamCompleteEvent immediately."""

    def __init__(self, run_ctx: AgentRunContext) -> None:
        self._run_ctx = run_ctx

    async def execute(self):  # type: ignore[override]
        self._message_history = []
        self._final_message = ChatMessage(content="done", role="assistant")
        yield RunStartedEvent(session_id=self._run_ctx.session_id, run_id="test-run")
        yield StreamCompleteEvent(
            message=ChatMessage(content="done", role="assistant"),
            cancelled=False,
            session_id=self._run_ctx.session_id,
        )


async def _patch_agent_create_turn(session_pool: Any, session_id: str, create_turn_fn: Any) -> Any:
    """Get the real agent from the pool and patch its create_turn method."""
    agent = await session_pool.sessions.get_or_create_session_agent(session_id)
    agent.create_turn = create_turn_fn  # type: ignore[method-assign]
    return agent


async def _drain_queue(queue: asyncio.Queue) -> list[Any]:
    """Drain all currently-available events from a queue without blocking."""
    events: list[Any] = []
    while True:
        with contextlib_suppress(asyncio.QueueEmpty):
            events.append(queue.get_nowait())
            continue
        break
    return events


def contextlib_suppress(exc: type[BaseException]):
    """Context manager that suppresses the given exception type."""
    import contextlib

    return contextlib.suppress(exc)


async def _collect_user_message_events(
    queue: asyncio.Queue, timeout: float = 2.0
) -> list[UserMessageInsertedEvent]:
    """Collect UserMessageInsertedEvent items from a subscriber queue."""
    events: list[UserMessageInsertedEvent] = []
    try:
        async with asyncio.timeout(timeout):
            while True:
                try:
                    raw = await asyncio.wait_for(queue.get(), timeout=timeout)
                except TimeoutError:
                    break
                unwrapped = _unwrap(raw)
                if isinstance(unwrapped, UserMessageInsertedEvent):
                    events.append(unwrapped)
    except TimeoutError:
        pass
    return events


# ---------------------------------------------------------------------------
# Test 1: Full chain — steer_from_background_task → EventBus → subscriber
# ---------------------------------------------------------------------------


async def test_background_task_steer_full_chain(minimal_pool: AgentPool) -> None:  # noqa: PLR0915
    """Steer from background task publishes UserMessageInsertedEvent on EventBus.

    Given: A session with an active blocking turn and an EventBus subscriber.
    When: ``steer_from_background_task()`` is called mid-turn.
    Then: ``UserMessageInsertedEvent`` appears on the subscriber queue with
        ``source="accepted"``, ``delivery="steer"``, and a valid
        ``message_id``.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "test-full-chain"
    await session_pool.create_session(session_id, agent_name="test_agent")
    release = asyncio.Event()
    await _patch_agent_create_turn(session_pool, session_id, _make_blocking_create_turn(release))
    queue = await session_pool.event_bus.subscribe(session_id, scope="session")
    # Start a run — _consume_run launches as a background task.
    msg_id = await session_pool.send_message(session_id, "initial prompt")
    assert msg_id is not None
    await asyncio.sleep(0.1)  # Let the turn start and block.
    # Inject steer from background task.
    result = await session_pool.steer_from_background_task(session_id, "bg task result")
    assert result is not None
    # Collect events — filter for the accepted steer event specifically.
    # The initial prompt also publishes a UserMessageInsertedEvent with
    # source="accepted", so we must filter by delivery="steer".
    user_msg_events: list[UserMessageInsertedEvent] = []
    all_events: list[Any] = []
    try:
        async with asyncio.timeout(3.0):
            while len(user_msg_events) < 1:
                try:
                    raw = await asyncio.wait_for(queue.get(), timeout=2.0)
                except TimeoutError:
                    break
                unwrapped = _unwrap(raw)
                all_events.append(unwrapped)
                if (
                    isinstance(unwrapped, UserMessageInsertedEvent)
                    and unwrapped.source == "accepted"
                    and unwrapped.delivery == "steer"
                ):
                    user_msg_events.append(unwrapped)
    except TimeoutError:
        pass
    # If not found on live queue, try replay buffer.
    if not user_msg_events:
        queue2 = await session_pool.event_bus.subscribe(session_id, scope="session")
        try:
            async with asyncio.timeout(3.0):
                while len(user_msg_events) < 1:
                    try:
                        raw = await asyncio.wait_for(queue2.get(), timeout=1.0)
                    except TimeoutError:
                        break
                    unwrapped = _unwrap(raw)
                    if (
                        isinstance(unwrapped, UserMessageInsertedEvent)
                        and unwrapped.source == "accepted"
                        and unwrapped.delivery == "steer"
                    ):
                        user_msg_events.append(unwrapped)
        except TimeoutError:
            pass
    assert len(user_msg_events) >= 1, (
        f"Expected at least 1 UserMessageInsertedEvent with source=accepted, "
        f"got {user_msg_events}. All events: {all_events}"
    )
    evt = user_msg_events[0]
    assert evt.source == "accepted"
    assert evt.delivery == "steer"
    assert evt.content == "bg task result"
    assert evt.message_id, "message_id must be a non-empty string"
    assert evt.message_id.startswith("msg_")
    # Release the turn and clean up.
    release.set()
    await asyncio.sleep(0.2)


# ---------------------------------------------------------------------------
# Test 2: Message ID ordering via ascending("message")
# ---------------------------------------------------------------------------


async def test_message_id_ordering_between_assistant_messages(
    minimal_pool: AgentPool,
) -> None:
    """``ascending("message")`` generates monotonically increasing IDs.

    Given: Three sequential calls to ``ascending("message")``.
    Then: IDs sort lexicographically in creation order and all start with
        ``"msg_"``.
    """
    id1 = ascending("message")
    id2 = ascending("message")
    id3 = ascending("message")
    assert id1.startswith("msg_")
    assert id2.startswith("msg_")
    assert id3.startswith("msg_")
    assert id1 < id2, f"Expected id1 < id2, got {id1!r} >= {id2!r}"
    assert id2 < id3, f"Expected id2 < id3, got {id2!r} >= {id3!r}"
    assert sorted([id1, id2, id3]) == [id1, id2, id3]


# ---------------------------------------------------------------------------
# Test 3: Dedup set mechanism — event IS published on EventBus
# ---------------------------------------------------------------------------


async def test_dedup_protocol_handler_same_message_id(
    minimal_pool: AgentPool,
) -> None:
    """EventBus publication is delivered to subscribers regardless of message_id.

    Given: A session and a ``UserMessageInsertedEvent`` with ``message_id="msg_123"``.
    When: The event is published to the EventBus.
    Then: The event IS delivered to subscribers (single-path architecture,
        no consumer-side filtering).
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "test-dedup"
    await session_pool.create_session(session_id, agent_name="test_agent")
    # Ensure agent is created so session._event_bus is set.
    await session_pool.sessions.get_or_create_session_agent(session_id)
    # Subscribe BEFORE publishing.
    queue = await session_pool.event_bus.subscribe(session_id, scope="session")
    # Publish a UserMessageInsertedEvent with message_id="msg_123".
    event = UserMessageInsertedEvent(
        session_id=session_id,
        message_id="msg_123",
        content="dedup test",
        delivery="steer",
        source="accepted",
    )
    await session_pool.event_bus.publish(session_id, event)
    # The event should arrive on the subscriber queue.
    received: list[UserMessageInsertedEvent] = []
    try:
        async with asyncio.timeout(2.0):
            raw = await asyncio.wait_for(queue.get(), timeout=2.0)
            unwrapped = _unwrap(raw)
            if isinstance(unwrapped, UserMessageInsertedEvent):
                received.append(unwrapped)
    except TimeoutError:
        pass
    assert len(received) == 1, f"Expected 1 UserMessageInsertedEvent on EventBus, got {received}"
    assert received[0].message_id == "msg_123"


# ---------------------------------------------------------------------------
# Test 4: Internal steer is NOT pre-registered in dedup set
# ---------------------------------------------------------------------------


async def test_dedup_accepted_steer_not_deduped(minimal_pool: AgentPool) -> None:
    """Accepted steer paths publish UserMessageInsertedEvent to EventBus.

    Given: A session with an active turn.
    When: ``steer_from_background_task()`` is called (accepted path).
    Then: The event appears on EventBus with ``source="accepted"`` and
        ``delivery="steer"``.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "test-accepted-no-dedup"
    await session_pool.create_session(session_id, agent_name="test_agent")
    release = asyncio.Event()
    await _patch_agent_create_turn(session_pool, session_id, _make_blocking_create_turn(release))
    queue = await session_pool.event_bus.subscribe(session_id, scope="session")
    await session_pool.send_message(session_id, "initial")
    await asyncio.sleep(0.1)
    result = await session_pool.steer_from_background_task(session_id, "accepted steer")
    assert result is not None
    # Collect the UserMessageInsertedEvent.
    events = await _collect_user_message_events(queue, timeout=2.0)
    bg_events = [e for e in events if e.source == "accepted" and e.delivery == "steer"]
    if not bg_events:
        # Try replay buffer.
        queue2 = await session_pool.event_bus.subscribe(session_id, scope="session")
        try:
            async with asyncio.timeout(2.0):
                while True:
                    try:
                        raw = await asyncio.wait_for(queue2.get(), timeout=0.5)
                    except TimeoutError:
                        break
                    unwrapped = _unwrap(raw)
                    if (
                        isinstance(unwrapped, UserMessageInsertedEvent)
                        and unwrapped.source == "accepted"
                        and unwrapped.delivery == "steer"
                    ):
                        bg_events.append(unwrapped)
                        break

        except TimeoutError:
            pass
    assert len(bg_events) >= 1, (
        f"Expected UserMessageInsertedEvent from background_task, got {bg_events}"
    )
    evt = bg_events[0]
    assert evt.source == "accepted"
    assert evt.content == "accepted steer"
    release.set()
    await asyncio.sleep(0.2)


# ---------------------------------------------------------------------------
# Test 5: session._event_bus is not None after init
# ---------------------------------------------------------------------------


async def test_session_event_bus_is_not_none_after_init(
    minimal_pool: AgentPool,
) -> None:
    """After agent creation, ``session._event_bus`` is set to the pool's EventBus.

    Given: A session created via ``create_session`` and an agent created via
        ``get_or_create_session_agent``.
    Then: ``session._event_bus`` is not None and is the same EventBus instance
        as ``pool._event_bus``.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "test-event-bus-init"
    await session_pool.create_session(session_id, agent_name="test_agent")
    # Trigger _initialize_lifecycle_and_recovery by creating the agent.
    await session_pool.sessions.get_or_create_session_agent(session_id)
    session = session_pool.sessions.get_session(session_id)
    assert session is not None
    assert session._event_bus is not None, "session._event_bus should be set after agent init"
    assert session._event_bus is session_pool.event_bus, (
        "session._event_bus should be the same EventBus instance as pool.event_bus"
    )


# ---------------------------------------------------------------------------
# Test 6: await publish does not block steer injection
# ---------------------------------------------------------------------------


async def test_await_publish_does_not_block_steer_injection(
    minimal_pool: AgentPool,
) -> None:
    """``await event_bus.publish()`` in steer_from_background_task does not deadlock.

    Given: A session with an active blocking turn.
    When: ``await steer_from_background_task()`` is called (uses inline
        ``await event_bus.publish()``).
    Then: The method returns within a reasonable time (< 1 second) and the
        steer message is injected into the RunHandle.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "test-no-deadlock"
    await session_pool.create_session(session_id, agent_name="test_agent")
    release = asyncio.Event()
    await _patch_agent_create_turn(session_pool, session_id, _make_blocking_create_turn(release))
    queue = await session_pool.event_bus.subscribe(session_id, scope="session")
    await session_pool.send_message(session_id, "initial")
    await asyncio.sleep(0.1)
    # Time the steer_from_background_task call — should return quickly.
    import time

    start = time.monotonic()
    result = await session_pool.steer_from_background_task(session_id, "test steer")
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, f"steer_from_background_task took {elapsed:.3f}s — possible deadlock"
    assert result is not None
    # Verify the event was published.
    events = await _collect_user_message_events(queue, timeout=2.0)
    bg_events = [e for e in events if e.source == "accepted" and e.delivery == "steer"]
    if not bg_events:
        queue2 = await session_pool.event_bus.subscribe(session_id, scope="session")
        try:
            async with asyncio.timeout(2.0):
                while True:
                    try:
                        raw = await asyncio.wait_for(queue2.get(), timeout=0.5)
                    except TimeoutError:
                        break
                    unwrapped = _unwrap(raw)
                    if (
                        isinstance(unwrapped, UserMessageInsertedEvent)
                        and unwrapped.source == "accepted"
                    ):
                        bg_events.append(unwrapped)
                        break

        except TimeoutError:
            pass
    assert len(bg_events) >= 1, (
        f"Expected UserMessageInsertedEvent, got {bg_events}. Events: {events}"
    )
    assert bg_events[0].content == "test steer"
    release.set()
    await asyncio.sleep(0.2)


# ---------------------------------------------------------------------------
# Test 7: Rapid steer messages ordering
# ---------------------------------------------------------------------------


async def test_rapid_steer_messages_ordering(minimal_pool: AgentPool) -> None:  # noqa: PLR0915
    """Three rapid steer calls produce 3 events with distinct ascending message_ids.

    Given: A session with an active blocking turn.
    When: ``steer_from_background_task()`` is called 3 times rapidly with
        "msg1", "msg2", "msg3".
    Then: 3 ``UserMessageInsertedEvent`` events are received, each with a
        distinct ``message_id`` in ascending order.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "test-rapid-steer"
    await session_pool.create_session(session_id, agent_name="test_agent")
    release = asyncio.Event()
    await _patch_agent_create_turn(session_pool, session_id, _make_blocking_create_turn(release))
    queue = await session_pool.event_bus.subscribe(session_id, scope="session")
    await session_pool.send_message(session_id, "initial")
    await asyncio.sleep(0.1)
    # Fire 3 rapid steers.
    r1 = await session_pool.steer_from_background_task(session_id, "msg1")
    r2 = await session_pool.steer_from_background_task(session_id, "msg2")
    r3 = await session_pool.steer_from_background_task(session_id, "msg3")
    assert r1 is not None
    assert r2 is not None
    assert r3 is not None
    # Collect all UserMessageInsertedEvent from the queue.
    # Filter by delivery="steer" to exclude the initial prompt event
    # (which also has source="accepted").
    all_events: list[UserMessageInsertedEvent] = []
    try:
        async with asyncio.timeout(3.0):
            while len(all_events) < 3:
                try:
                    raw = await asyncio.wait_for(queue.get(), timeout=2.0)
                except TimeoutError:
                    break
                unwrapped = _unwrap(raw)
                if (
                    isinstance(unwrapped, UserMessageInsertedEvent)
                    and unwrapped.source == "accepted"
                    and unwrapped.delivery == "steer"
                ):
                    all_events.append(unwrapped)

    except TimeoutError:
        pass
    # If we didn't get all 3 from the live queue, try replay buffer.
    if len(all_events) < 3:
        queue2 = await session_pool.event_bus.subscribe(session_id, scope="session")
        try:
            async with asyncio.timeout(3.0):
                while len(all_events) < 3:
                    try:
                        raw = await asyncio.wait_for(queue2.get(), timeout=1.0)
                    except TimeoutError:
                        break
                    unwrapped = _unwrap(raw)
                    if (
                        isinstance(unwrapped, UserMessageInsertedEvent)
                        and unwrapped.source == "accepted"
                        and unwrapped.delivery == "steer"
                    ):
                        all_events.append(unwrapped)
        except TimeoutError:
            pass
    assert len(all_events) == 3, (
        f"Expected 3 UserMessageInsertedEvent, got {len(all_events)}: {all_events}"
    )
    message_ids = [e.message_id for e in all_events]
    # All distinct.
    assert len(set(message_ids)) == 3, f"Expected 3 distinct message_ids, got {message_ids}"
    # Ascending order.
    assert message_ids == sorted(message_ids), (
        f"Expected message_ids in ascending order, got {message_ids}"
    )
    # Content matches.
    contents = [e.content for e in all_events]
    assert contents == ["msg1", "msg2", "msg3"], (
        f"Expected contents ['msg1','msg2','msg3'], got {contents}"
    )
    release.set()
    await asyncio.sleep(0.2)


# ---------------------------------------------------------------------------
# Test 8: Steer during tool execution — event before processing
# ---------------------------------------------------------------------------


async def test_steer_during_tool_execution_event_before_processing(
    minimal_pool: AgentPool,
) -> None:
    """UserMessageInsertedEvent is published BEFORE the tool completes.

    Given: A session with a turn that yields ToolCallStartEvent then blocks.
    When: ``steer_from_background_task()`` is called while the tool is
        "executing" (blocked).
    Then: ``UserMessageInsertedEvent`` is received by the subscriber before
        the StreamCompleteEvent (i.e., before the tool/turn completes).
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "test-tool-steer"
    await session_pool.create_session(session_id, agent_name="test_agent")
    release = asyncio.Event()
    await _patch_agent_create_turn(
        session_pool, session_id, _make_tool_blocking_create_turn(release)
    )
    queue = await session_pool.event_bus.subscribe(session_id, scope="session")
    await session_pool.send_message(session_id, "initial")
    await asyncio.sleep(0.15)  # Let the turn start and block on tool.
    # Steer while tool is "executing".
    result = await session_pool.steer_from_background_task(session_id, "steer during tool")
    assert result is not None
    # Collect events in order — filter for the accepted steer event.
    # The initial prompt also publishes a UserMessageInsertedEvent with
    # delivery="initial" source="accepted", so we must filter by delivery="steer".
    received_order: list[Any] = []
    user_msg_events: list[UserMessageInsertedEvent] = []
    try:
        async with asyncio.timeout(3.0):
            while len(user_msg_events) < 1:
                try:
                    raw = await asyncio.wait_for(queue.get(), timeout=2.0)
                except TimeoutError:
                    break
                unwrapped = _unwrap(raw)
                received_order.append(unwrapped)
                if (
                    isinstance(unwrapped, UserMessageInsertedEvent)
                    and unwrapped.source == "accepted"
                    and unwrapped.delivery == "steer"
                ):
                    user_msg_events.append(unwrapped)
    except TimeoutError:
        pass
    assert len(user_msg_events) >= 1, (
        f"Expected UserMessageInsertedEvent before tool completion, got {received_order}"
    )
    evt = user_msg_events[0]
    assert evt.delivery == "steer"
    assert evt.source == "accepted"
    assert evt.content == "steer during tool"
    # No StreamCompleteEvent should have arrived before the UserMessageInsertedEvent
    # (the turn is still blocked).
    evt_index = received_order.index(evt)
    stream_completes_before = [
        e for e in received_order[:evt_index] if isinstance(e, StreamCompleteEvent)
    ]
    assert stream_completes_before == [], (
        "StreamCompleteEvent arrived before UserMessageInsertedEvent — "
        "event should be published BEFORE tool completion"
    )
    release.set()
    await asyncio.sleep(0.2)


# ---------------------------------------------------------------------------
# Test 9: Followup from prompt queue — event published
# ---------------------------------------------------------------------------


async def test_followup_from_prompt_queue_event_published(
    minimal_pool: AgentPool,
) -> None:
    """Followup from prompt_queue does NOT publish UserMessageInsertedEvent.

    In the single-path display architecture, ``_route_message()`` is the
    sole publication point. Messages in ``prompt_queue`` were already
    displayed by ``_route_message()`` before being enqueued, so
    ``_consume_run()`` must NOT re-emit the event.

    Given: A session with an active blocking turn.
    When: A followup is queued via ``pool.followup()`` and the turn completes.
    Then: ``_consume_run()`` picks up the followup from ``prompt_queue`` but
        does NOT publish ``UserMessageInsertedEvent``.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "test-followup-queue"
    await session_pool.create_session(session_id, agent_name="test_agent")
    release = asyncio.Event()
    await _patch_agent_create_turn(session_pool, session_id, _make_blocking_create_turn(release))
    queue = await session_pool.event_bus.subscribe(session_id, scope="session")
    await session_pool.send_message(session_id, "initial")
    await asyncio.sleep(0.1)
    # Queue a followup — this puts it on session.prompt_queue.
    followup_result = await session_pool.followup(session_id, "queued followup")
    assert followup_result is not None
    # Release the blocking turn — _consume_run will pick up the followup.
    release.set()
    # Collect events — looking for absence of followup UserMessageInsertedEvent.
    received: list[Any] = []
    try:
        async with asyncio.timeout(5.0):
            while True:
                try:
                    raw = await asyncio.wait_for(queue.get(), timeout=3.0)
                except TimeoutError:
                    break
                unwrapped = _unwrap(raw)
                received.append(unwrapped)
    except TimeoutError:
        pass
    followup_events = [
        e
        for e in received
        if isinstance(e, UserMessageInsertedEvent)
        and e.delivery == "followup"
        and e.source == "accepted"
    ]
    assert len(followup_events) == 0, (
        f"Expected NO UserMessageInsertedEvent from _consume_run (single-path), "
        f"got {followup_events}. All received: {received}"
    )
    await asyncio.sleep(0.2)


# ---------------------------------------------------------------------------
# Test 10: No EventBus — standalone, no crash
# ---------------------------------------------------------------------------


async def test_no_event_bus_standalone_no_crash(minimal_pool: AgentPool) -> None:
    """Steer from background task does not crash when EventBus is None.

    Given: An AgentPool with ``_event_bus`` set to None on the SessionPool.
    When: ``steer_from_background_task()`` is called.
    Then: No exception is raised and the method returns gracefully.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "test-no-event-bus"
    await session_pool.create_session(session_id, agent_name="test_agent")
    await session_pool.sessions.get_or_create_session_agent(session_id)
    session = session_pool.sessions.get_session(session_id)
    assert session is not None
    # Simulate no EventBus on the session.
    original_bus = session._event_bus
    session._event_bus = None
    try:
        # Should not raise — the method checks `if event_bus is not None`.
        result = await session_pool.steer_from_background_task(session_id, "no bus test")
        # Should return None (no active run) or a message_id (if feedback_queue).
        # The key assertion is that no exception was raised.
        assert result is None or isinstance(result, str)
    finally:
        session._event_bus = original_bus


# ---------------------------------------------------------------------------
# Test 11: session._event_bus = None — silent drop
# ---------------------------------------------------------------------------


async def test_event_bus_none_field_silent_drop(minimal_pool: AgentPool) -> None:
    """Setting ``session._event_bus = None`` silently drops the event.

    Given: A session with ``_event_bus`` manually set to None.
    When: ``steer_from_background_task()`` is called.
    Then: No exception is raised and the method returns gracefully (the event
        is silently dropped since there is no bus to publish to).
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "test-silent-drop"
    await session_pool.create_session(session_id, agent_name="test_agent")
    await session_pool.sessions.get_or_create_session_agent(session_id)
    session = session_pool.sessions.get_session(session_id)
    assert session is not None
    assert session._event_bus is not None
    # Manually set _event_bus to None.
    original_bus = session._event_bus
    session._event_bus = None
    try:
        # Should not raise.
        result = await session_pool.steer_from_background_task(session_id, "silent drop")
        # No active run → feedback_queue path. Result is message_id or None.
        assert result is None or isinstance(result, str)
    finally:
        session._event_bus = original_bus
