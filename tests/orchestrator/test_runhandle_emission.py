"""Tests for RunHandle steer/followup emission behavior.

Verifies that:
- ``steer()`` SKIPS ``_schedule_user_message_emission()`` when
  ``emit_user_message=True`` AND ``_enqueued_messages_available=True``
  AND ``active_agent_run`` is not ``None`` (EnqueuedMessagesEvent path
  handles display instead).
- ``steer()`` CALLS ``_schedule_user_message_emission()`` when
  ``emit_user_message=True`` AND either ``_enqueued_messages_available``
  is ``False`` OR ``active_agent_run`` is ``None`` (fallback path).
- ``followup()`` follows the same conditional skip logic.
- ``steer()`` appends ``message_id`` to ``_pending_enqueue_message_ids``
  before calling ``agent_run.enqueue()``.
- ``followup()`` appends ``message_id`` to ``_pending_enqueue_message_ids``
  before calling ``agent_run.enqueue()``.
- ``followup()`` uses ``agent_run.enqueue(content, priority='when_idle')``
  when ``active_agent_run`` is not ``None``.
- ``followup()`` falls back to ``session.prompt_queue`` when
  ``active_agent_run`` is ``None``.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from wolfharness.agents.context import AgentRunContext
from wolfharness.agents.events import UserMessageInsertedEvent
from wolfharness.lifecycle.comm_channel import DirectChannel
from wolfharness.lifecycle.journal import MemoryJournal
from wolfharness.orchestrator.core import SessionState
from wolfharness.orchestrator.run import RunHandle


pytestmark = pytest.mark.unit


_UNSET = object()


def _make_handle(
    *,
    event_bus: Any | None = _UNSET,
    session: SessionState | None = None,
    enqueued_available: bool = True,
    active_agent_run: Any | None = None,
) -> RunHandle:
    """Create a RunHandle with mocked deps for emission tests.

    Args:
        event_bus: EventBus mock or ``None``. Defaults to ``AsyncMock()``.
        session: Real or mock SessionState. If ``None``, a real one is
            created with a ``DirectChannel``.
        enqueued_available: Value to set on
            ``_enqueued_messages_available``.
        active_agent_run: Mock AgentRun or ``None``.
    """
    if event_bus is _UNSET:  # type: ignore[comparison-overlap]
        event_bus = AsyncMock()
    agent = MagicMock()
    agent.name = "test-agent"
    agent.conversation = MagicMock()
    if session is None:
        session = SessionState(
            session_id="test-session",
            agent_name="test-agent",
        )
        session._comm_channel = DirectChannel(MemoryJournal())
    handle = RunHandle(
        run_id="test-run",
        session_id="test-session",
        agent_type="test",
        agent=agent,
        event_bus=event_bus,
        session=session,
        run_ctx=AgentRunContext(),
    )
    # Override the computed field for test control.
    handle._enqueued_messages_available = enqueued_available
    handle.active_agent_run = active_agent_run
    return handle


async def _drain_tasks() -> None:
    """Yield control to let pending ``create_task`` coroutines run."""
    for _ in range(5):
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# steer() emission tests — always emit when emit_user_message=True
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_steer_emits_when_enqueued_available_and_agent_run_active() -> None:
    """When EnqueuedMessagesEvent is available and active_agent_run is set,
    steer() SKIPS _schedule_user_message_emission() — the display event
    comes from handle_enqueued_messages() with source="processed" instead.
    """  # noqa: D205
    agent_run = MagicMock()
    event_bus = AsyncMock()
    handle = _make_handle(
        event_bus=event_bus,
        enqueued_available=True,
        active_agent_run=agent_run,
    )

    handle.steer("steer content")

    await _drain_tasks()

    user_msg_events = [
        call
        for call in event_bus.publish.call_args_list
        if isinstance(call.args[1], UserMessageInsertedEvent)
    ]
    assert len(user_msg_events) == 0
    # Verify the actual enqueue happened.
    agent_run.enqueue.assert_called_once_with("steer content", priority="asap")


@pytest.mark.unit
async def test_steer_emits_when_agent_run_is_none() -> None:
    """When active_agent_run is None (inter-turn), steer() calls
    _schedule_user_message_emission().
    """  # noqa: D205
    event_bus = AsyncMock()
    handle = _make_handle(
        event_bus=event_bus,
        enqueued_available=True,
        active_agent_run=None,
    )

    handle.steer("steer content")

    await _drain_tasks()

    user_msg_events = [
        call
        for call in event_bus.publish.call_args_list
        if isinstance(call.args[1], UserMessageInsertedEvent)
    ]
    assert len(user_msg_events) == 1
    assert user_msg_events[0].args[1].delivery == "steer"


@pytest.mark.unit
async def test_steer_emits_when_enqueued_unavailable() -> None:
    """When EnqueuedMessagesEvent is NOT available, steer() calls
    _schedule_user_message_emission().
    """  # noqa: D205
    agent_run = MagicMock()
    event_bus = AsyncMock()
    handle = _make_handle(
        event_bus=event_bus,
        enqueued_available=False,
        active_agent_run=agent_run,
    )

    handle.steer("steer content")

    await _drain_tasks()

    user_msg_events = [
        call
        for call in event_bus.publish.call_args_list
        if isinstance(call.args[1], UserMessageInsertedEvent)
    ]
    assert len(user_msg_events) == 1


@pytest.mark.unit
async def test_steer_appends_message_id_to_pending_queue() -> None:
    """steer() appends message_id to _pending_enqueue_message_ids before
    calling agent_run.enqueue().
    """  # noqa: D205
    agent_run = MagicMock()
    handle = _make_handle(
        enqueued_available=True,
        active_agent_run=agent_run,
    )

    result = handle.steer("steer content")

    assert result is not None
    assert result in handle.run_ctx._pending_enqueue_message_ids
    assert len(handle.run_ctx._pending_enqueue_message_ids) == 1


# ---------------------------------------------------------------------------
# followup() emission tests — always emit when emit_user_message=True
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_followup_emits_when_enqueued_available_and_agent_run_active() -> None:
    """When EnqueuedMessagesEvent is available and active_agent_run is set,
    followup() SKIPS _schedule_user_message_emission() when
    emit_user_message=True — the display event comes from
    handle_enqueued_messages() with source="processed" instead.
    """  # noqa: D205
    agent_run = MagicMock()
    event_bus = AsyncMock()
    handle = _make_handle(
        event_bus=event_bus,
        enqueued_available=True,
        active_agent_run=agent_run,
    )

    handle.followup("followup content", emit_user_message=True)

    await _drain_tasks()

    user_msg_events = [
        call
        for call in event_bus.publish.call_args_list
        if isinstance(call.args[1], UserMessageInsertedEvent)
    ]
    assert len(user_msg_events) == 0
    # Verify the enqueue happened with when_idle priority.
    agent_run.enqueue.assert_called_once_with("followup content", priority="when_idle")


@pytest.mark.unit
async def test_followup_emits_when_agent_run_is_none() -> None:
    """When active_agent_run is None (inter-turn), followup() calls
    _schedule_user_message_emission().
    """  # noqa: D205
    event_bus = AsyncMock()
    handle = _make_handle(
        event_bus=event_bus,
        enqueued_available=True,
        active_agent_run=None,
    )

    handle.followup("followup content", emit_user_message=True)

    await _drain_tasks()

    user_msg_events = [
        call
        for call in event_bus.publish.call_args_list
        if isinstance(call.args[1], UserMessageInsertedEvent)
    ]
    assert len(user_msg_events) == 1
    assert user_msg_events[0].args[1].delivery == "followup"


@pytest.mark.unit
async def test_followup_emits_when_enqueued_unavailable() -> None:
    """When EnqueuedMessagesEvent is NOT available, followup() calls
    _schedule_user_message_emission().
    """  # noqa: D205
    agent_run = MagicMock()
    event_bus = AsyncMock()
    handle = _make_handle(
        event_bus=event_bus,
        enqueued_available=False,
        active_agent_run=agent_run,
    )

    handle.followup("followup content", emit_user_message=True)

    await _drain_tasks()

    user_msg_events = [
        call
        for call in event_bus.publish.call_args_list
        if isinstance(call.args[1], UserMessageInsertedEvent)
    ]
    assert len(user_msg_events) == 1


@pytest.mark.unit
async def test_followup_appends_message_id_to_pending_queue() -> None:
    """followup() appends message_id to _pending_enqueue_message_ids before
    calling agent_run.enqueue().
    """  # noqa: D205
    agent_run = MagicMock()
    handle = _make_handle(
        enqueued_available=True,
        active_agent_run=agent_run,
    )

    result = handle.followup("followup content", emit_user_message=True)

    assert result is not None
    assert result in handle.run_ctx._pending_enqueue_message_ids
    assert len(handle.run_ctx._pending_enqueue_message_ids) == 1


# ---------------------------------------------------------------------------
# followup() queue migration tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_followup_uses_agent_run_enqueue_when_active() -> None:
    """followup() must use agent_run.enqueue(priority='when_idle') when
    active_agent_run is not None, instead of session.prompt_queue.
    """  # noqa: D205
    agent_run = MagicMock()
    session = SessionState(
        session_id="test-session",
        agent_name="test-agent",
    )
    session._comm_channel = DirectChannel(MemoryJournal())
    handle = _make_handle(
        session=session,
        enqueued_available=True,
        active_agent_run=agent_run,
    )

    handle.followup("followup content")

    agent_run.enqueue.assert_called_once_with("followup content", priority="when_idle")
    # prompt_queue should NOT have been used.
    assert session.prompt_queue.empty()


@pytest.mark.unit
async def test_followup_falls_back_to_prompt_queue_when_agent_run_none() -> None:
    """followup() must fall back to session.prompt_queue when
    active_agent_run is None.
    """  # noqa: D205
    session = SessionState(
        session_id="test-session",
        agent_name="test-agent",
    )
    session._comm_channel = DirectChannel(MemoryJournal())
    handle = _make_handle(
        session=session,
        enqueued_available=True,
        active_agent_run=None,
    )

    result = handle.followup("followup content")

    assert result is not None
    assert not session.prompt_queue.empty()
    queued = session.prompt_queue.get_nowait()
    assert queued == "followup content"


# ---------------------------------------------------------------------------
# Inter-turn fallback tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_steer_emission_called_in_inter_turn_even_when_enqueued_available() -> None:
    """When active_agent_run is None (inter-turn window), _schedule_user_message_emission()
    IS called even when _enqueued_messages_available=True, because there is no
    active stream to emit EnqueuedMessagesEvent.
    """  # noqa: D205
    event_bus = AsyncMock()
    handle = _make_handle(
        event_bus=event_bus,
        enqueued_available=True,
        active_agent_run=None,
    )

    handle.steer("inter-turn steer")

    await _drain_tasks()

    user_msg_events = [
        call
        for call in event_bus.publish.call_args_list
        if isinstance(call.args[1], UserMessageInsertedEvent)
    ]
    assert len(user_msg_events) == 1
    assert user_msg_events[0].args[1].content == "inter-turn steer"


@pytest.mark.unit
async def test_followup_emission_called_in_inter_turn_even_when_enqueued_available() -> None:
    """When active_agent_run is None (inter-turn window), _schedule_user_message_emission()
    IS called for followup() even when _enqueued_messages_available=True.
    """  # noqa: D205
    event_bus = AsyncMock()
    handle = _make_handle(
        event_bus=event_bus,
        enqueued_available=True,
        active_agent_run=None,
    )

    handle.followup("inter-turn followup", emit_user_message=True)

    await _drain_tasks()

    user_msg_events = [
        call
        for call in event_bus.publish.call_args_list
        if isinstance(call.args[1], UserMessageInsertedEvent)
    ]
    assert len(user_msg_events) == 1
    assert user_msg_events[0].args[1].content == "inter-turn followup"
