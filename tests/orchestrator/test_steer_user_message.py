"""Tests for emit_user_message parameter on RunHandle.steer/followup.

Verifies that:
- ``steer(emit_user_message=True)`` (default) schedules a
  ``UserMessageInsertedEvent`` publication via ``create_task``.
- ``steer(emit_user_message=False)`` suppresses emission.
- ``followup(emit_user_message=True)`` schedules emission.
- ``followup(emit_user_message=False)`` (default) suppresses emission.
- No-running-loop scenario: ``steer()`` called outside an async context
  catches ``RuntimeError`` and steer still proceeds.
- ``event_bus=None`` guard: emission helper does not crash.
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
) -> RunHandle:
    """Create a RunHandle with mocked deps for steer/followup tests.

    Args:
        event_bus: EventBus mock or ``None``. Defaults to ``AsyncMock()``.
        session: Real or mock SessionState. If ``None``, a real one is
            created with a ``DirectChannel``.
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
    return RunHandle(
        run_id="test-run",
        session_id="test-session",
        agent_type="test",
        agent=agent,
        event_bus=event_bus,
        session=session,
        run_ctx=AgentRunContext(),
    )


async def _drain_tasks() -> None:
    """Yield control to let pending ``create_task`` coroutines run."""
    # Multiple sleep(0) rounds to ensure background tasks complete.
    for _ in range(5):
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# steer() emit_user_message tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_steer_default_emits_user_message() -> None:
    """Given ``steer()`` with default params, a ``UserMessageInsertedEvent``
    is published to the EventBus.
    """  # noqa: D205
    event_bus = AsyncMock()
    handle = _make_handle(event_bus=event_bus)

    handle.steer("steer content")

    await _drain_tasks()

    # Verify UserMessageInsertedEvent was published.
    publish_calls = event_bus.publish.call_args_list
    assert len(publish_calls) >= 1
    # Find the UserMessageInsertedEvent call.
    user_msg_events = [
        call for call in publish_calls if isinstance(call.args[1], UserMessageInsertedEvent)
    ]
    assert len(user_msg_events) == 1
    event = user_msg_events[0].args[1]
    assert isinstance(event, UserMessageInsertedEvent)
    assert event.session_id == "test-session"
    assert event.content == "steer content"
    assert event.delivery == "steer"
    assert event.source == "accepted"


@pytest.mark.unit
async def test_steer_emit_false_suppresses_emission() -> None:
    """Given ``steer(emit_user_message=False)``, no
    ``UserMessageInsertedEvent`` is published.
    """  # noqa: D205
    event_bus = AsyncMock()
    handle = _make_handle(event_bus=event_bus)

    handle.steer("steer content", emit_user_message=False)

    await _drain_tasks()

    publish_calls = event_bus.publish.call_args_list
    user_msg_events = [
        call for call in publish_calls if isinstance(call.args[1], UserMessageInsertedEvent)
    ]
    assert len(user_msg_events) == 0


@pytest.mark.unit
async def test_steer_multimodal_content_emits() -> None:
    """Given ``steer()`` with list content (multi-modal), the event carries
    the list content.
    """  # noqa: D205
    event_bus = AsyncMock()
    handle = _make_handle(event_bus=event_bus)

    multimodal: list[Any] = [{"type": "text", "text": "hello"}]
    handle.steer(multimodal)

    await _drain_tasks()

    user_msg_events = [
        call
        for call in event_bus.publish.call_args_list
        if isinstance(call.args[1], UserMessageInsertedEvent)
    ]
    assert len(user_msg_events) == 1
    event = user_msg_events[0].args[1]
    assert event.content == multimodal
    assert event.delivery == "steer"


# ---------------------------------------------------------------------------
# followup() emit_user_message tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_followup_default_does_not_emit() -> None:
    """Given ``followup()`` with default params, no
    ``UserMessageInsertedEvent`` is published (default is ``False``).
    """  # noqa: D205
    event_bus = AsyncMock()
    handle = _make_handle(event_bus=event_bus)

    result = handle.followup("followup content")
    assert result is not None

    await _drain_tasks()

    user_msg_events = [
        call
        for call in event_bus.publish.call_args_list
        if isinstance(call.args[1], UserMessageInsertedEvent)
    ]
    assert len(user_msg_events) == 0


@pytest.mark.unit
async def test_followup_emit_true_emits_user_message() -> None:
    """Given ``followup(emit_user_message=True)``, a
    ``UserMessageInsertedEvent`` is published with ``delivery="followup"``.
    """  # noqa: D205
    event_bus = AsyncMock()
    handle = _make_handle(event_bus=event_bus)

    handle.followup("followup content", emit_user_message=True)

    await _drain_tasks()

    user_msg_events = [
        call
        for call in event_bus.publish.call_args_list
        if isinstance(call.args[1], UserMessageInsertedEvent)
    ]
    assert len(user_msg_events) == 1
    event = user_msg_events[0].args[1]
    assert isinstance(event, UserMessageInsertedEvent)
    assert event.content == "followup content"
    assert event.delivery == "followup"
    assert event.source == "accepted"


@pytest.mark.unit
async def test_followup_multimodal_content_emits() -> None:
    """Given ``followup()`` with list content and ``emit_user_message=True``,
    the event carries the list content.
    """  # noqa: D205
    event_bus = AsyncMock()
    handle = _make_handle(event_bus=event_bus)

    multimodal: list[Any] = [{"type": "image", "url": "data:..."}]
    handle.followup(multimodal, emit_user_message=True)

    await _drain_tasks()

    user_msg_events = [
        call
        for call in event_bus.publish.call_args_list
        if isinstance(call.args[1], UserMessageInsertedEvent)
    ]
    assert len(user_msg_events) == 1
    event = user_msg_events[0].args[1]
    assert event.content == multimodal


# ---------------------------------------------------------------------------
# No-running-loop scenario
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_steer_no_running_loop_does_not_crash() -> None:
    """Given ``steer()`` called outside an async context (no running loop),
    the ``RuntimeError`` is caught and steer still proceeds.
    """  # noqa: D205
    event_bus = AsyncMock()
    handle = _make_handle(event_bus=event_bus)

    # Called from sync context — no running event loop.
    # Should not raise RuntimeError.
    result = handle.steer("steer without loop")

    # Steer still returns a message_id (it queued the message).
    assert result is not None


@pytest.mark.unit
def test_followup_no_running_loop_does_not_crash() -> None:
    """Given ``followup(emit_user_message=True)`` called outside an async
    context, the ``RuntimeError`` is caught and followup still proceeds.
    """  # noqa: D205
    event_bus = AsyncMock()
    handle = _make_handle(event_bus=event_bus)

    # Called from sync context — no running event loop.
    # Should not raise RuntimeError.
    result = handle.followup("followup without loop", emit_user_message=True)

    # Followup still returns a message_id.
    assert result is not None


# ---------------------------------------------------------------------------
# event_bus=None guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_steer_with_none_event_bus_does_not_crash() -> None:
    """Given ``event_bus=None``, ``steer(emit_user_message=True)`` does not
    crash — the emission helper checks ``self.event_bus`` before publishing.
    """  # noqa: D205
    handle = _make_handle(event_bus=None)

    # Should not raise.
    handle.steer("steer content")

    await _drain_tasks()


@pytest.mark.unit
async def test_followup_with_none_event_bus_does_not_crash() -> None:
    """Given ``event_bus=None``, ``followup(emit_user_message=True)`` does
    not crash.
    """  # noqa: D205
    handle = _make_handle(event_bus=None)

    handle.followup("followup content", emit_user_message=True)

    await _drain_tasks()


# ---------------------------------------------------------------------------
# Emission helper exception handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_steer_emission_exception_does_not_break_steer() -> None:
    """Given the EventBus raises during publish, the emission helper catches
    the exception and steer still succeeds.
    """  # noqa: D205
    event_bus = AsyncMock()
    event_bus.publish = AsyncMock(side_effect=RuntimeError("bus broken"))
    handle = _make_handle(event_bus=event_bus)

    # Should not raise — the exception is caught in _emit_user_message_inserted.
    handle.steer("steer content")

    await _drain_tasks()

    # The publish was attempted (and failed), but steer completed.
    assert event_bus.publish.call_count >= 1
