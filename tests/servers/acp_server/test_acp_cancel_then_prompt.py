"""Integration test: ACP cancel-then-prompt does not hang.

Tests the SessionPool-level behavior that the ACP handler relies on
when a client sends session/cancel followed immediately by session/prompt.

The ACP handler's ``cancel_session()`` delegates to
``SessionPool.sessions.cancel_run_for_session()``, and its ``handle_prompt()``
delegates to ``SessionPool.receive_request()``. This test verifies that the
underlying SessionPool correctly handles the cancel-then-prompt sequence
without hanging — the same sequence that occurs when an ACP client cancels
a run and immediately sends a new prompt.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from wolfharness.agents.events import RunFailedEvent, RunStartedEvent, StreamCompleteEvent
from wolfharness.lifecycle import DirectChannel, MemoryJournal
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.core import EventEnvelope
from wolfharness.orchestrator.turn import Turn


if TYPE_CHECKING:
    from wolfharness import AgentPool
    from wolfharness.agents.context import AgentRunContext


pytestmark = [pytest.mark.unit, pytest.mark.anyio]


def _unwrap_event(event: Any) -> Any:
    """Unwrap EventEnvelope if present, otherwise return the event as-is."""
    return event.event if isinstance(event, EventEnvelope) else event


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _BlockingTurn(Turn):
    """Turn that blocks until run_ctx.cancelled, then returns without StreamCompleteEvent."""

    def __init__(self, run_ctx: AgentRunContext) -> None:
        self._run_ctx = run_ctx

    async def execute(self):  # type: ignore[override]
        self._message_history = []
        self._final_message = ChatMessage(content="blocked", role="assistant")
        while not self._run_ctx.cancelled:
            await asyncio.sleep(0.01)
        return
        yield  # makes this an async generator


class _StubTurn(Turn):
    """Minimal Turn that yields events from a list and sets message history."""

    def __init__(
        self,
        *,
        events: list[Any] | None = None,
        message_history: list[Any] | None = None,
    ) -> None:
        self._events = events or []
        self._history = message_history or []

    async def execute(self):  # type: ignore[override]
        self._message_history = self._history
        self._final_message = ChatMessage(content="done", role="assistant")
        for event in self._events:
            yield event


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _attach_agent(
    pool: object,
    session_id: str,
    agent: MagicMock,
    real_pool: AgentPool,
) -> None:
    """Attach a mock agent to an existing session."""
    state, _ = await pool.sessions.get_or_create_session(session_id)  # type: ignore[union-attr]
    state.agent = agent
    # Initialize comm_channel on session (normally done by
    # _initialize_lifecycle_and_recovery, but we pre-cache the agent
    # so that path is skipped).
    state._comm_channel = DirectChannel(MemoryJournal())
    pool.sessions._session_agents[session_id] = agent  # type: ignore[union-attr]
    real_pool.get_agent = MagicMock(return_value=agent)  # type: ignore[assignment]


def _make_cancel_aware_agent() -> MagicMock:
    """Create a mock agent whose first create_turn returns _BlockingTurn.

    Subsequent calls return _StubTurn instances that yield RunStartedEvent
    followed by StreamCompleteEvent.
    """
    agent = MagicMock()
    agent.AGENT_TYPE = "native"

    call_count = 0

    def _create_turn(
        prompts: Any,
        run_ctx: AgentRunContext,
        message_history: Any,
    ) -> Turn:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _BlockingTurn(run_ctx)
        return _StubTurn(
            events=[
                StreamCompleteEvent(
                    message=ChatMessage(content="response", role="assistant"),
                ),
            ],
            message_history=["msg"],
        )

    agent.create_turn = _create_turn
    return agent


async def _drain_queue(queue: asyncio.Queue[Any]) -> list[Any]:
    """Drain all currently-available events from a queue without blocking."""
    events: list[Any] = []
    while True:
        try:
            events.append(queue.get_nowait())
        except (asyncio.QueueEmpty, asyncio.QueueShutDown):
            break
    return events


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_acp_cancel_then_prompt_no_hang(
    minimal_pool: AgentPool,
) -> None:
    """ACP cancel-then-prompt sequence does not hang at the SessionPool level.

    Simulates the ACP handler flow:
        1. ``handle_prompt()`` → ``receive_request()`` starts a blocking run.
        2. ``cancel_session()`` → ``cancel_run_for_session()`` cancels it.
        3. ``handle_prompt()`` → ``receive_request()`` sends a new prompt.

    The second ``receive_request()`` must return within 30s (no hang),
    and the new prompt must be processed (RunStartedEvent + StreamCompleteEvent).

    Uses ``asyncio.wait_for()`` with a 30s timeout to catch hangs.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None

    session_id = "sess-acp-cancel-prompt"
    await session_pool.create_session(session_id, agent_name="test-agent")

    agent = _make_cancel_aware_agent()
    await _attach_agent(session_pool, session_id, agent, minimal_pool)

    # Subscribe to events BEFORE sending the first prompt
    queue = await session_pool.event_bus.subscribe(session_id)

    # --- Step 1: Start a run with the blocking agent (simulates handle_prompt) ---
    await session_pool.send_message(session_id, "first prompt")
    session_state = session_pool.sessions.get_session(session_id)
    assert session_state is not None
    first_handle = session_pool.sessions._runs.get(session_state.current_run_id)  # type: ignore[union-attr]
    assert first_handle is not None, "Should have a RunHandle after receive_request"

    # Wait for the blocking turn to start
    await asyncio.sleep(0.1)

    # --- Step 2: Cancel the active run (simulates cancel_session) ---
    session_pool.sessions.cancel_run_for_session(session_id)

    # Wait for cancellation to propagate: the start() loop should
    # publish RunFailedEvent, set _turn_complete_event, clear the
    # message queue, and continue.
    await asyncio.sleep(0.2)

    # Drain events published so far
    pre_events = await _drain_queue(queue)
    pre_event_types = [type(_unwrap_event(e)) for e in pre_events]

    # RunFailedEvent must have been published as a result of the cancel
    assert RunFailedEvent in pre_event_types, (
        f"Expected RunFailedEvent from cancelled turn, got: {pre_event_types}"
    )

    # --- Step 3: Send a new prompt (simulates second handle_prompt) ---
    # Use asyncio.wait_for to catch hangs.
    await asyncio.wait_for(
        session_pool.send_message(session_id, "second prompt"),
        timeout=30.0,
    )
    session2 = session_pool.sessions.get_session(session_id)
    assert session2 is not None
    session_pool.sessions._runs.get(session2.current_run_id)  # type: ignore[union-attr]

    # --- Step 4: Verify new prompt is processed (events published, no hang) ---
    post_events: list[Any] = []
    try:
        async with asyncio.timeout(30.0):
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=5.0)
                    post_events.append(event)
                    unwrapped = _unwrap_event(event)
                    if isinstance(unwrapped, StreamCompleteEvent):
                        break
                except TimeoutError:
                    break
    except TimeoutError:
        pytest.fail("Timed out waiting for events after cancel-then-prompt")

    post_event_types = [type(_unwrap_event(e)) for e in post_events]

    assert RunStartedEvent in post_event_types, (
        f"Expected RunStartedEvent for new prompt, got: {post_event_types}"
    )
    assert StreamCompleteEvent in post_event_types, (
        f"Expected StreamCompleteEvent for new prompt, got: {post_event_types}"
    )

    # --- Step 5: Verify RunHandle state ---
    # After cancel + followup, the RunHandle may be the same instance (reused)
    # or a new one. Either is valid — what matters is the state.
    # In the per-prompt model, RunHandle has no _run_state. Use complete_event.
    assert first_handle.complete_event.is_set(), (
        "First RunHandle should be complete (complete_event set) after cancel"
    )

    # Cleanup: close the RunHandle first so the start() loop exits and
    # releases turn_lock. Otherwise close_session waits 30s for the lock.
    first_handle.close()
    await asyncio.sleep(0.1)


@pytest.mark.anyio
async def test_cancel_does_not_start_spontaneous_turn(
    minimal_pool: AgentPool,
) -> None:
    """After cancel, RunHandle.start() must enter idle — not re-execute the cancelled prompt.

    Regression test for the ``current_prompts`` reuse bug:
    - ``start()`` loop had ``continue`` in the cancel path without clearing
      ``current_prompts``.
    - This caused the loop to skip the idle phase and immediately create a
      new turn with the SAME prompts that were just cancelled.
    - The agent would then see an empty/stale user message and produce
      unexpected output (e.g. "The user hasn't said anything yet").

    This test verifies:
    1. ``create_turn`` is called exactly ONCE (for the initial prompt).
    2. After cancel propagation, ``_status`` is ``idle``.
    3. No new events are published between cancel and the idle check.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None

    session_id = "sess-cancel-no-spontaneous"
    await session_pool.create_session(session_id, agent_name="test-agent")

    # Agent whose create_turn tracks call count.
    # First call: _BlockingTurn (blocks until cancelled).
    # Subsequent calls: _StubTurn (yields StreamCompleteEvent only;
    # RunStartedEvent is published by RunHandle.start()).
    # If the bug exists (current_prompts not cleared), the spontaneous turn
    # would call create_turn a second time and yield events we can detect.
    agent = MagicMock()
    agent.AGENT_TYPE = "native"
    create_turn_calls: list[tuple[Any, ...]] = []

    def _create_turn(
        prompts: Any,
        run_ctx: AgentRunContext,
        message_history: Any,
    ) -> Turn:
        create_turn_calls.append((prompts, run_ctx, message_history))
        if len(create_turn_calls) == 1:
            return _BlockingTurn(run_ctx)
        return _StubTurn(
            events=[
                StreamCompleteEvent(
                    message=ChatMessage(content="response", role="assistant"),
                ),
            ],
            message_history=["msg"],
        )

    agent.create_turn = _create_turn
    await _attach_agent(session_pool, session_id, agent, minimal_pool)

    queue = await session_pool.event_bus.subscribe(session_id)

    # --- Start a blocking run ---
    await session_pool.send_message(session_id, "first prompt")
    session_state = session_pool.sessions.get_session(session_id)
    assert session_state is not None
    first_handle = session_pool.sessions._runs.get(session_state.current_run_id)  # type: ignore[union-attr]
    assert first_handle is not None

    # Wait for the blocking turn to start
    await asyncio.sleep(0.1)
    assert len(create_turn_calls) == 1, (
        f"Expected 1 create_turn call before cancel, got {len(create_turn_calls)}"
    )

    # Drain initial events (RunStartedEvent from RunHandle.start())
    # so they don't contaminate the post-cancel event check below.
    _ = await _drain_queue(queue)

    # --- Cancel the active run ---
    session_pool.sessions.cancel_run_for_session(session_id)

    # Wait for cancellation to propagate through the start() loop:
    # RunFailedEvent published, _turn_complete_event set, current_prompts
    # cleared (the fix), continue to idle phase.
    await asyncio.sleep(0.3)

    # --- Assert: no spontaneous second turn ---
    # With the bug, current_prompts was not cleared, so the loop would
    # immediately create a second turn (call_count == 2).
    # With the fix, current_prompts = [] forces the loop into idle.
    assert len(create_turn_calls) == 1, (
        f"Expected exactly 1 create_turn call after cancel (no spontaneous "
        f"turn), got {len(create_turn_calls)}. The cancelled prompt was "
        f"re-executed — current_prompts was not cleared in the cancel path."
    )

    # RunHandle should be complete (complete_event set) after cancel
    assert first_handle.complete_event.is_set(), (
        "Expected RunHandle complete_event set after cancel"
    )

    # No new events should have been published between cancel and idle
    # (only RunFailedEvent from the cancel itself)
    events_after_cancel = await _drain_queue(queue)
    event_types = [type(_unwrap_event(e)) for e in events_after_cancel]
    assert RunFailedEvent in event_types, f"Expected RunFailedEvent from cancel, got: {event_types}"
    # No RunStartedEvent or StreamCompleteEvent — those would indicate a
    # spontaneous turn was started
    assert RunStartedEvent not in event_types, (
        f"RunStartedEvent found after cancel — spontaneous turn was started! Events: {event_types}"
    )
    assert StreamCompleteEvent not in event_types, (
        f"StreamCompleteEvent found after cancel — spontaneous turn completed! "
        f"Events: {event_types}"
    )

    # --- Verify the session accepts a new prompt normally ---
    # receive_request() will call followup() on the existing (idle) RunHandle
    # rather than creating a new one. This returns None but queues the message.
    await asyncio.wait_for(
        session_pool.send_message(session_id, "second prompt"),
        timeout=10.0,
    )
    # second_handle is None because the existing RunHandle is still alive (idle)
    # and receive_request routes to followup() instead of _start_run_handle().

    # The second prompt should trigger a second create_turn call
    await asyncio.sleep(0.1)
    assert len(create_turn_calls) == 2, (
        f"Expected 2 create_turn calls after second prompt, got {len(create_turn_calls)}"
    )

    # Cleanup
    first_handle.close()
    await asyncio.sleep(0.1)
