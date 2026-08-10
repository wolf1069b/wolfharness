"""Steer/queue delivery semantics tests (Group 6).

Verifies behavioral semantics of steer() through the RunHandle — NOT event
type assertions, since AgentPool does not emit steer-specific events by design.

Tests use mock agents with stub turns to control timing and inspect
the prompts passed to create_turn, following the pattern in
tests/orchestrator/test_run_handle.py.

In the per-prompt RunHandle model, each RunHandle executes exactly one turn
and terminates. Tests that verify steer during an ACTIVE turn (in-flight)
are preserved. Tests for the old idle-loop / followup / multi-turn patterns
have been removed since those APIs are no longer on RunHandle.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import anyio
import pytest

from wolfharness.agents.context import AgentRunContext
from wolfharness.agents.events import RunErrorEvent, RunStartedEvent, StreamCompleteEvent
from wolfharness.lifecycle.comm_channel import DirectChannel
from wolfharness.lifecycle.journal import MemoryJournal
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.core import EventBus, SessionState
from wolfharness.orchestrator.run import RunHandle


pytestmark = [pytest.mark.integration, pytest.mark.anyio]


def _make_handle(
    *,
    agent: Any | None = None,
    event_bus: EventBus | None = None,
) -> tuple[RunHandle, EventBus, list[list[Any]]]:
    """Create a RunHandle with a mock agent that captures prompts.

    Returns (handle, event_bus, captured_prompts).
    """
    bus = event_bus or EventBus()
    captured_prompts: list[list[Any]] = []

    mock_agent = agent or MagicMock()
    # Capture prompts passed to create_turn.
    original_create_turn = mock_agent.create_turn

    def _capturing_create_turn(*, prompts: list[Any], **kwargs: Any) -> Any:
        captured_prompts.append(list(prompts))
        return original_create_turn(prompts=prompts, **kwargs)

    mock_agent.create_turn = _capturing_create_turn

    session = SessionState(session_id="test-sess", agent_name="test-agent")
    # In the per-prompt model, _execute_turn reads session._comm_channel to
    # publish events. DirectChannel provides the minimal impl for tests.
    session._comm_channel = DirectChannel(MemoryJournal())
    run_ctx = AgentRunContext(session_id="test-sess", event_bus=bus)
    handle = RunHandle(
        run_id="test-run",
        session_id="test-sess",
        agent_type="native",
        agent=mock_agent,
        event_bus=bus,
        session=session,
        run_ctx=run_ctx,
    )
    return handle, bus, captured_prompts


def _stub_turn(
    *,
    output: str = "done",
    fail: bool = False,
) -> Any:
    """Create a stub Turn that yields minimal events."""
    turn = MagicMock()

    async def _execute():
        yield RunStartedEvent(session_id="test-sess", run_id="test-run")
        if fail:
            yield RunErrorEvent(
                session_id="test-sess",
                run_id="test-run",
                error_type="TestError",
                error_message="Turn failed",
            )
            return
        yield StreamCompleteEvent(
            message=ChatMessage(content=output, role="assistant"),
            cancelled=False,
            session_id="test-sess",
        )

    turn.execute = _execute
    return turn


# ---------------------------------------------------------------------------
# In-flight steer: steer during an ACTIVE turn
# ---------------------------------------------------------------------------


def _blocking_stub_turn(
    *,
    block_event: anyio.Event,
    output: str = "done",
) -> Any:
    """Create a stub Turn that blocks on an event before completing.

    This allows testing steer during an active turn.
    """
    turn = MagicMock()

    async def _execute() -> Any:
        yield RunStartedEvent(session_id="test-sess", run_id="test-run")
        await block_event.wait()
        yield StreamCompleteEvent(
            message=ChatMessage(content=output, role="assistant"),
            cancelled=False,
            session_id="test-sess",
        )

    turn.execute = _execute
    # Simulate message_history accumulation (NativeTurn sets this).
    turn.message_history = []
    return turn


async def test_inflight_steer_single_message() -> None:
    """Steer during an active turn is enqueued to agent_run with priority='asap'.

    Given: A RunHandle with a turn actively running.
    When: handle.steer("in-flight msg") is called.
    Then: agent_run.enqueue("in-flight msg", priority="asap") is called.
    """
    block_event = anyio.Event()
    mock_agent_run = MagicMock()
    mock_agent_run.enqueue = MagicMock()

    def create_turn(**kwargs: Any) -> Any:
        return _blocking_stub_turn(block_event=block_event)

    mock_agent = MagicMock()
    mock_agent.create_turn = create_turn

    handle, _bus, _captured = _make_handle(agent=mock_agent)

    events: list[Any] = []
    gen = handle.start("initial")

    async with anyio.create_task_group() as tg:

        async def _consume() -> None:
            async for event in gen:
                events.append(event)  # noqa: PERF401

        tg.start_soon(_consume)
        await anyio.sleep(0.05)

        # Turn is now running and blocked on block_event.
        assert handle.is_running
        # Simulate NativeTurn setting active_agent_run during turn execution.
        handle.active_agent_run = mock_agent_run

        handle.steer("in-flight msg")
        await anyio.sleep(0.02)

        # Verify enqueue was called with the steer message and asap priority.
        mock_agent_run.enqueue.assert_called_once()
        call_args = mock_agent_run.enqueue.call_args
        assert call_args.args[0] == "in-flight msg"
        assert call_args.kwargs.get("priority") == "asap"

        # Release the turn to complete.
        block_event.set()
        await anyio.sleep(0.05)

        handle.close()
        await anyio.sleep(0.05)


async def test_inflight_steer_multiple_messages() -> None:
    """Multiple steers during an active turn are all enqueued to agent_run.

    Checks:
    - handle.steer() calls agent_run.enqueue for each steer message.
    - All enqueued messages have asap priority.
    """
    block_event = anyio.Event()
    mock_agent_run = MagicMock()
    mock_agent_run.enqueue = MagicMock()

    def create_turn(**kwargs: Any) -> Any:
        return _blocking_stub_turn(block_event=block_event)

    mock_agent = MagicMock()
    mock_agent.create_turn = create_turn

    handle, _bus, _captured = _make_handle(agent=mock_agent)

    gen = handle.start("tasks")

    async with anyio.create_task_group() as tg:

        async def _consume() -> None:
            async for _event in gen:
                pass

        tg.start_soon(_consume)
        await anyio.sleep(0.05)

        handle.active_agent_run = mock_agent_run

        handle.steer("msg1")
        handle.steer("msg2")
        handle.steer("msg3")
        await anyio.sleep(0.02)

        # All 3 steers should be enqueued.
        assert mock_agent_run.enqueue.call_count == 3
        enqueued_msgs = [call.args[0] for call in mock_agent_run.enqueue.call_args_list]
        assert enqueued_msgs == ["msg1", "msg2", "msg3"]
        # All with asap priority.
        for call in mock_agent_run.enqueue.call_args_list:
            assert call.kwargs.get("priority") == "asap"

        block_event.set()
        await anyio.sleep(0.05)

        handle.close()
        await anyio.sleep(0.05)


# ---------------------------------------------------------------------------
# Queued steer: steer during RUNNING with no active_agent_run
# ---------------------------------------------------------------------------


async def test_queued_steer_delivered_to_next_turn() -> None:
    """Steer during a running turn with no active_agent_run is re-enqueued to feedback_queue.

    Checks:
    - steer() with no active_agent_run falls back to session.feedback_queue
    - Message survives across RunHandle boundaries via feedback_queue
    """
    block_event = anyio.Event()
    turn_count = 0

    def create_turn(**kwargs: Any) -> Any:
        nonlocal turn_count
        turn_count += 1
        if turn_count == 1:
            return _blocking_stub_turn(block_event=block_event)
        return _stub_turn()

    mock_agent = MagicMock()
    mock_agent.create_turn = create_turn

    handle, _bus, _captured = _make_handle(agent=mock_agent)

    gen = handle.start("tasks")

    async with anyio.create_task_group() as tg:

        async def _consume() -> None:
            async for _event in gen:
                pass

        tg.start_soon(_consume)
        await anyio.sleep(0.05)

        # Turn 1 is active and blocked.
        assert handle.is_running
        # Simulate no active agent_run available for enqueue.
        handle.active_agent_run = None

        handle.steer("queued steer")
        await anyio.sleep(0.02)

        # With fix A, the steer message should be in session.feedback_queue
        # (not queued_steer_messages, which is never drained).
        assert handle.session is not None
        assert not handle.session.feedback_queue.empty(), (
            "steer message should be re-enqueued to session.feedback_queue"
        )
        fb = handle.session.feedback_queue.get_nowait()
        assert fb.content == "queued steer"

        # Release turn 1 to complete.
        block_event.set()
        await anyio.sleep(0.05)

        handle.close()
        await anyio.sleep(0.05)

    # In the per-prompt model, start() executes exactly one turn and
    # terminates. The steer message survives in feedback_queue and will
    # be drained by the next RunHandle's start() into queued_steer_messages,
    # then delivered to agent_run via drain_queued_steer_messages().
    assert turn_count == 1, (
        "Only one turn executes in the per-prompt model; queued steer "
        "survives via feedback_queue and is delivered by the next RunHandle."
    )
