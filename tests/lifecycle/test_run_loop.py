"""Tests for RunHandle lifecycle in the per-prompt model.

In the per-prompt RunHandle model, each RunHandle executes exactly one
turn and terminates naturally. Session-level state (lifecycle dimensions,
conversation history, message routing) is owned by ``SessionState``.

The following test categories were REMOVED because they tested RunHandle
features that no longer exist in the per-prompt model:

- Constructor dimension defaults (_trigger_source, _journal,
  _snapshot_store, _comm_channel, _event_transport) — moved to SessionState
- State machine (_run_state, RunState transitions, _transition()) — removed
- Idle loop (_idle_event, _idle_loop, _message_queue) — removed
- followup() — removed (routing moves to SessionState.prompt_queue)
- _handle_recovery() — moved to session init (_initialize_lifecycle_and_recovery)
- Dimension closing in close() — close() only sets complete_event
- Event journaling in RunHandle — moved to CommChannel
- Snapshot saving at turn boundaries — moved to session init
- Steer routing via CommChannel.deliver_feedback — steer() now uses
  agent_run.enqueue() or run_ctx.queued_steer_messages
- _turn_complete_event — replaced by complete_event
- _closing / _closed flags — replaced by complete_event.is_set()
- _state_lock — removed (no state transitions)
- _force_cancelling — removed

The preserved tests below verify the remaining RunHandle API surface.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from wolfharness.agents.events import StreamCompleteEvent
from wolfharness.lifecycle import DirectChannel, MemoryJournal
from wolfharness.messaging import ChatMessage, MessageHistory
from wolfharness.orchestrator.run import RunHandle
from wolfharness.orchestrator.session_controller import SessionState
from wolfharness.orchestrator.turn import Turn


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubTurn(Turn):
    """Minimal Turn implementation for testing."""

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


def _stream_complete_event() -> StreamCompleteEvent[Any]:
    return StreamCompleteEvent(message=ChatMessage(content="done", role="assistant"))


def _make_run_handle(
    *,
    agent: Any | None = None,
    event_bus: Any | None = None,
    session: Any | None = None,
    run_id: str = "test-run",
    session_id: str = "test-session",
    agent_type: str = "native",
) -> RunHandle:
    """Create a RunHandle with real SessionState and DirectChannel."""
    if agent is None:
        agent = MagicMock()
        agent.create_turn = MagicMock(return_value=_StubTurn())
        agent.name = "test-agent"
        agent.conversation = MessageHistory()
    if event_bus is None:
        event_bus = AsyncMock()
    if session is None:
        session = SessionState(session_id=session_id, agent_name="test-agent")
        session._comm_channel = DirectChannel(MemoryJournal())
    return RunHandle(
        run_id=run_id,
        session_id=session_id,
        agent_type=agent_type,
        agent=agent,
        event_bus=event_bus,
        session=session,
    )


# ---------------------------------------------------------------------------
# Preserved: is_running property
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_is_running_property() -> None:
    """is_running returns True when complete_event is not set."""
    handle = _make_run_handle()
    # Fresh handle: complete_event not set → is_running is True.
    assert handle.is_running is True
    assert handle.complete_event.is_set() is False

    # After close(): complete_event set → is_running is False.
    handle.close()
    assert handle.is_running is False
    assert handle.complete_event.is_set() is True


# ---------------------------------------------------------------------------
# Preserved: close() idempotency
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_close_twice_is_noop() -> None:
    """close() called twice: second call is a no-op."""
    handle = _make_run_handle()

    handle.close()
    assert handle.complete_event.is_set() is True

    # Second close should not raise and complete_event stays set.
    handle.close()
    assert handle.complete_event.is_set() is True


# ---------------------------------------------------------------------------
# Preserved: close() while running lets turn finish
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_close_while_running_lets_turn_finish() -> None:
    """close() during a turn sets complete_event after the turn completes.

    In the per-prompt model, close() sets complete_event immediately.
    The turn finishes naturally because start() runs the turn to
    completion in its try/finally block.
    """
    turn = _StubTurn(events=[_stream_complete_event()], message_history=["m1"])
    agent = MagicMock()
    agent.create_turn = MagicMock(return_value=turn)
    agent.name = "test-agent"
    agent.conversation = MessageHistory()

    handle = _make_run_handle(agent=agent)

    # Start the turn in a background task.
    async def _consume() -> None:
        async for _ in handle.start("test"):
            pass

    consumer = asyncio.create_task(_consume())
    await asyncio.sleep(0.05)

    # Close while running.
    handle.close()
    await asyncio.sleep(0.05)
    await consumer

    # complete_event should be set.
    assert handle.complete_event.is_set() is True


# ---------------------------------------------------------------------------
# Preserved: steer() returns message_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_steer_returns_message_id() -> None:
    """steer() returns the provided message_id."""
    handle = _make_run_handle()

    msg_id = handle.steer("test message", message_id="custom-id-001")
    assert msg_id == "custom-id-001"


@pytest.mark.unit
async def test_steer_without_message_id_generates_one() -> None:
    """steer() without message_id auto-generates a UUID."""
    handle = _make_run_handle()

    msg_id = handle.steer("test message")
    assert msg_id is not None
    assert isinstance(msg_id, str)
    assert len(msg_id) > 0
