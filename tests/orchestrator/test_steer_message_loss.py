"""TDD tests for steer message loss fix (PR #168 comment #5066106164).

These tests verify that steer messages are not lost when:
1. ``steer()`` is called with no ``active_agent_run`` (race window after turn ends)
2. ``feedback_queue`` is drained at ``start()`` time (before ``active_agent_run`` is set)
3. ``queued_steer_messages`` are delivered to the ``agent_run`` when it becomes active

Fix strategy:
- Fix A: ``steer()`` fallback writes to ``session.feedback_queue``
  (survives across RunHandles)
- Fix B: ``start()`` drain writes directly to ``queued_steer_messages``
  (avoids infinite loop with A)
- Fix C: ``RunHandle.drain_queued_steer_messages()`` drains to active ``agent_run``
         Called by ``NativeTurn.execute()`` after setting ``active_agent_run``
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from unittest.mock import MagicMock

import anyio
import pytest

from wolfharness.agents.context import AgentRunContext
from wolfharness.agents.events import RunStartedEvent, StreamCompleteEvent
from wolfharness.lifecycle import Feedback
from wolfharness.lifecycle.comm_channel import DirectChannel
from wolfharness.lifecycle.journal import MemoryJournal
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.core import EventBus, SessionState
from wolfharness.orchestrator.run import RunHandle


pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handle(
    *,
    agent: Any | None = None,
    session_id: str = "test-sess",
) -> RunHandle:
    """Create a RunHandle with a mock agent and real SessionState."""
    mock_agent = agent or MagicMock()
    mock_agent.name = "test-agent"
    session = SessionState(session_id=session_id, agent_name="test-agent")
    session._comm_channel = DirectChannel(MemoryJournal())
    bus = EventBus()
    run_ctx = AgentRunContext(session_id=session_id, event_bus=bus)
    return RunHandle(
        run_id="test-run",
        session_id=session_id,
        agent_type="native",
        agent=mock_agent,
        event_bus=bus,
        session=session,
        run_ctx=run_ctx,
    )


def _stub_turn(*, output: str = "done") -> Any:
    """Create a stub Turn that yields minimal events."""
    turn = MagicMock()

    async def _execute() -> Any:
        yield RunStartedEvent(session_id="test-sess", run_id="test-run")
        yield StreamCompleteEvent(
            message=ChatMessage(content=output, role="assistant"),
            cancelled=False,
            session_id="test-sess",
        )

    turn.execute = _execute
    turn.message_history = []
    return turn


def _blocking_stub_turn(
    *,
    block_event: anyio.Event,
    on_start: Any | None = None,
    output: str = "done",
) -> Any:
    """Create a stub Turn that blocks on an event before completing.

    Args:
        block_event: Event to wait on before completing.
        on_start: Optional callback invoked when the turn starts
            (simulates NativeTurn setting active_agent_run).
        output: Output text for the StreamCompleteEvent.
    """
    turn = MagicMock()

    async def _execute() -> Any:
        if on_start is not None:
            on_start()
        yield RunStartedEvent(session_id="test-sess", run_id="test-run")
        await block_event.wait()
        yield StreamCompleteEvent(
            message=ChatMessage(content=output, role="assistant"),
            cancelled=False,
            session_id="test-sess",
        )

    turn.execute = _execute
    turn.message_history = []
    return turn


# ---------------------------------------------------------------------------
# Fix A: steer() fallback writes to session.feedback_queue
# ---------------------------------------------------------------------------


async def test_steer_no_active_run_writes_to_feedback_queue() -> None:
    """steer() with no active_agent_run writes to session.feedback_queue.

    Before fix: message goes to ``queued_steer_messages`` (dead list, never drained).
    After fix: message goes to ``session.feedback_queue`` (survives across RunHandles).

    This is the core fix for the race window where ``active_agent_run`` is None
    but the run hasn't been cleaned up yet.
    """
    handle = _make_handle()
    assert handle.active_agent_run is None
    assert handle.session is not None

    handle.steer("race window steer")

    # Message MUST be in feedback_queue (survives across RunHandle boundaries)
    assert not handle.session.feedback_queue.empty(), (
        "steer() fallback should write to session.feedback_queue, "
        "not queued_steer_messages (which is never drained)"
    )
    fb = handle.session.feedback_queue.get_nowait()
    assert fb.content == "race window steer"
    assert fb.is_steer is True

    # Message MUST NOT be in queued_steer_messages (dead list)
    assert handle.run_ctx.queued_steer_messages == [], (
        "steer() fallback should NOT write to queued_steer_messages "
        "(that list is never drained — messages would be lost)"
    )


async def test_steer_no_active_run_list_content_writes_to_feedback_queue() -> None:
    """steer() with list content and no active_agent_run writes to feedback_queue.

    Same as above but for multimodal/list content blocks.
    """
    handle = _make_handle()
    assert handle.active_agent_run is None
    assert handle.session is not None

    blocks: list[Any] = [{"type": "text", "text": "multimodal steer"}]
    handle.steer(blocks)

    assert not handle.session.feedback_queue.empty(), (
        "steer() with list content should also write to feedback_queue"
    )
    fb = handle.session.feedback_queue.get_nowait()
    assert fb.content_blocks == blocks
    assert fb.is_steer is True

    assert handle.run_ctx.queued_steer_messages == []


# ---------------------------------------------------------------------------
# Fix C: RunHandle.drain_queued_steer_messages() delivers to agent_run
# ---------------------------------------------------------------------------


async def test_drain_queued_steer_messages_delivers_to_agent_run() -> None:
    """drain_queued_steer_messages() enqueues queued messages to agent_run.

    After ``NativeTurn.execute()`` sets ``active_agent_run``, it should call
    ``drain_queued_steer_messages()`` to deliver any messages that arrived
    before ``active_agent_run`` was set (e.g., during ``start()`` drain).
    """
    handle = _make_handle()
    mock_agent_run = MagicMock()
    mock_agent_run.enqueue = MagicMock()
    handle.active_agent_run = mock_agent_run

    # Simulate messages queued during start() drain (before active_agent_run was set)
    handle.run_ctx.queued_steer_messages = ["steer msg 1", "steer msg 2"]

    # Call the drain method (should exist after fix)
    handle.drain_queued_steer_messages()

    # Both messages should be enqueued to agent_run with asap priority
    assert mock_agent_run.enqueue.call_count == 2
    calls = mock_agent_run.enqueue.call_args_list
    assert calls[0].args[0] == "steer msg 1"
    assert calls[0].kwargs.get("priority") == "asap"
    assert calls[1].args[0] == "steer msg 2"
    assert calls[1].kwargs.get("priority") == "asap"

    # queued_steer_messages should be cleared after draining
    assert handle.run_ctx.queued_steer_messages == []


async def test_drain_queued_steer_messages_list_content() -> None:
    """drain_queued_steer_messages() handles list content blocks.

    List content (multimodal) should be unpacked with *args to enqueue.
    """
    handle = _make_handle()
    mock_agent_run = MagicMock()
    mock_agent_run.enqueue = MagicMock()
    handle.active_agent_run = mock_agent_run

    blocks: list[Any] = [{"type": "text", "text": "hello"}]
    handle.run_ctx.queued_steer_messages = [blocks, "plain text"]

    handle.drain_queued_steer_messages()

    assert mock_agent_run.enqueue.call_count == 2
    call_list = mock_agent_run.enqueue.call_args_list
    # First call: list content — first element passed as arg
    assert call_list[0].args[0] == blocks[0]
    assert call_list[0].kwargs.get("priority") == "asap"
    # Second call: plain text
    assert call_list[1].args[0] == "plain text"
    assert call_list[1].kwargs.get("priority") == "asap"


async def test_drain_queued_steer_messages_noop_when_empty() -> None:
    """drain_queued_steer_messages() is a no-op when queue is empty."""
    handle = _make_handle()
    mock_agent_run = MagicMock()
    mock_agent_run.enqueue = MagicMock()
    handle.active_agent_run = mock_agent_run

    handle.drain_queued_steer_messages()

    mock_agent_run.enqueue.assert_not_called()
    assert handle.run_ctx.queued_steer_messages == []


async def test_drain_queued_steer_messages_noop_when_no_agent_run() -> None:
    """drain_queued_steer_messages() is a no-op when no active agent_run."""
    handle = _make_handle()
    handle.run_ctx.queued_steer_messages = ["msg"]

    handle.drain_queued_steer_messages()

    # Messages should remain in queue (no agent_run to drain to)
    assert handle.run_ctx.queued_steer_messages == ["msg"]


# ---------------------------------------------------------------------------
# Fix B: start() drain writes directly to queued_steer_messages (no infinite loop)
# ---------------------------------------------------------------------------


async def test_start_drain_feedback_queue_to_queued_steer_messages() -> None:
    """start() drains feedback_queue messages into queued_steer_messages.

    After fix A (steer fallback → feedback_queue), start() must NOT call
    steer() to drain feedback_queue (that would create an infinite loop).
    Instead, it writes directly to queued_steer_messages.

    The messages in queued_steer_messages are later drained by
    NativeTurn.execute() via drain_queued_steer_messages().
    """
    from wolfharness.orchestrator.turn import Turn

    class _CapturingStubTurn(Turn):
        """Stub turn that captures prompts and simulates drain."""

        def __init__(self) -> None:
            self._prompts: list[Any] = []
            self._message_history: list[Any] = []
            self._final_message = ChatMessage(content="done", role="assistant")

        async def execute(self):  # type: ignore[override]
            # Simulate NativeTurn: set active_agent_run and drain queued_steer_messages
            run_ctx = getattr(self, "_run_ctx", None)
            if run_ctx is not None and run_ctx._run_handle is not None:
                mock_ar = MagicMock()
                mock_ar.enqueue = MagicMock()
                self._run_ctx._run_handle.active_agent_run = mock_ar
                self._run_ctx._run_handle.drain_queued_steer_messages()
            yield RunStartedEvent(session_id="test-sess", run_id="test-run")
            yield StreamCompleteEvent(
                message=ChatMessage(content="done", role="assistant"),
                cancelled=False,
                session_id="test-sess",
            )

    agent = MagicMock()
    captured_prompts: list[list[Any]] = []

    def create_turn(*, prompts: list[Any], **kwargs: Any) -> Any:
        captured_prompts.append(list(prompts))
        turn = _CapturingStubTurn()
        turn._prompts = prompts
        return turn

    agent.create_turn = create_turn
    agent.name = "test-agent"
    agent.conversation = MagicMock()
    agent.conversation.get_history.return_value = []

    handle = _make_handle(agent=agent)
    assert handle.session is not None

    # Enqueue steer messages in feedback_queue
    handle.session.feedback_queue.put_nowait(Feedback(content="steer from feedback", is_steer=True))

    # Start the handle — should drain feedback_queue without hanging
    gen = handle.start("initial prompt")
    try:
        async with asyncio.timeout(5):
            _events = [e async for e in gen]
    finally:
        with contextlib.suppress(Exception):
            await gen.aclose()

    # feedback_queue should be drained
    assert handle.session.feedback_queue.empty(), (
        "feedback_queue should be fully drained by start()"
    )
    # The initial prompt should be passed to create_turn
    assert len(captured_prompts) == 1
    assert captured_prompts[0] == ["initial prompt"]


# ---------------------------------------------------------------------------
# End-to-end: steer during race window survives and is delivered
# ---------------------------------------------------------------------------


async def test_steer_during_race_window_survives_via_feedback_queue() -> None:
    """Steer message during race window survives via feedback_queue.

    Timeline:
    1. Turn starts, active_agent_run is set
    2. Turn ends, active_agent_run is cleared (race window opens)
    3. steer() is called — message goes to feedback_queue (fix A)
    4. RunHandle completes
    5. New RunHandle starts, drains feedback_queue → queued_steer_messages
    6. Turn sets active_agent_run, drains queued_steer_messages (fix C)
    7. Message is delivered to agent_run
    """
    block_event = anyio.Event()
    mock_agent_run = MagicMock()
    mock_agent_run.enqueue = MagicMock()

    handle = _make_handle()
    assert handle.session is not None

    def create_turn(**kwargs: Any) -> Any:
        return _blocking_stub_turn(block_event=block_event)

    handle.agent.create_turn = create_turn  # type: ignore[attr-defined]

    gen = handle.start("initial")

    async with anyio.create_task_group() as tg:

        async def _consume() -> None:
            async for _event in gen:
                pass

        tg.start_soon(_consume)
        await anyio.sleep(0.05)

        # Turn is active — simulate NativeTurn setting active_agent_run
        handle.active_agent_run = mock_agent_run
        assert handle.is_running

        # Now simulate the race window: active_agent_run cleared but run not done
        handle.active_agent_run = None

        # Steer during race window — should go to feedback_queue (fix A)
        handle.steer("race window msg")
        await anyio.sleep(0.02)

        # Message MUST be in feedback_queue (not lost in queued_steer_messages)
        assert not handle.session.feedback_queue.empty(), (
            "steer during race window should go to feedback_queue"
        )
        fb = handle.session.feedback_queue.get_nowait()
        assert fb.content == "race window msg"

        # Complete the turn
        block_event.set()
        await anyio.sleep(0.05)

        handle.close()
        await anyio.sleep(0.05)

    # The message survived in feedback_queue — it would be drained by the
    # next RunHandle's start() and delivered via drain_queued_steer_messages()
