"""Tests for RunStartedEvent/RunErrorEvent agent_name field.

Verifies that RunStartedEvent.agent_name and RunErrorEvent.agent_name carry
the agent's NAME (e.g. "librarian"), not the agent's TYPE (e.g. "native").

This is a producer-side test — it checks the events emitted by RunHandle.start(),
not just the event consumers.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from wolfharness.agents.events import (
    RunErrorEvent,
    RunStartedEvent,
    StreamCompleteEvent,
)
from wolfharness.lifecycle.comm_channel import DirectChannel
from wolfharness.lifecycle.journal import MemoryJournal
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.run import RunHandle
from wolfharness.orchestrator.turn import Turn


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers (mirrors test_run_handle.py but with agent.name set)
# ---------------------------------------------------------------------------


class _StubTurn(Turn):
    """Minimal Turn that yields events then completes."""

    def __init__(
        self,
        *,
        events: list[Any] | None = None,
        raise_exc: BaseException | None = None,
    ) -> None:
        self._events = events or []
        self._raise = raise_exc

    async def execute(self):  # type: ignore[override]
        if self._raise is not None:
            raise self._raise
        self._message_history: list[Any] = []
        self._final_message = ChatMessage(content="done", role="assistant")
        for event in self._events:
            yield event


def _stream_complete() -> StreamCompleteEvent[Any]:
    return StreamCompleteEvent(message=ChatMessage(content="done", role="assistant"))


def _make_handle(
    *,
    agent_name: str = "librarian",
    agent_type: str = "native",
    raise_exc: BaseException | None = None,
) -> tuple[RunHandle, AsyncMock]:
    """Create a RunHandle with a named mock agent and AsyncMock event_bus.

    Returns (handle, event_bus) so the caller can inspect published events.
    """
    from wolfharness.orchestrator.core import SessionState

    agent = MagicMock()
    agent.name = agent_name
    agent.AGENT_TYPE = agent_type
    agent.create_turn = MagicMock(
        return_value=_StubTurn(
            events=[_stream_complete()],
            raise_exc=raise_exc,
        ),
    )
    event_bus = AsyncMock()
    session = SessionState(session_id="test-session", agent_name=agent_name)
    session._comm_channel = DirectChannel(MemoryJournal())
    handle = RunHandle(
        run_id="test-run",
        session_id="test-session",
        agent_type=agent_type,
        agent=agent,
        event_bus=event_bus,
        session=session,
    )
    return handle, event_bus


async def _run_and_collect(handle: RunHandle, prompt: str = "hello") -> list[Any]:
    """Start the handle, consume events, close, and return them."""
    gen = handle.start(prompt)
    events: list[Any] = []

    async def _consume() -> None:
        events.extend([event async for event in gen])

    task = asyncio.create_task(_consume())
    await asyncio.sleep(0.05)
    handle.close()
    await asyncio.sleep(0.05)
    await task
    return events


def _get_published_events(event_bus: AsyncMock) -> list[Any]:
    """Extract events from event_bus.publish call_args_list."""
    return [call.args[1] for call in event_bus.publish.call_args_list]


# ---------------------------------------------------------------------------
# RunStartedEvent.agent_name
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_run_started_event_uses_agent_name_not_type() -> None:
    """RunStartedEvent.agent_name should be the agent's name, not its type.

    Given an agent named "librarian" with type "native", the RunStartedEvent
    should carry agent_name="librarian", not agent_name="native".
    """
    handle, event_bus = _make_handle(agent_name="librarian", agent_type="native")
    await _run_and_collect(handle)

    published = _get_published_events(event_bus)
    run_started = [e for e in published if isinstance(e, RunStartedEvent)]
    assert len(run_started) >= 1
    assert run_started[0].agent_name == "librarian"


@pytest.mark.unit
async def test_run_started_event_agent_name_different_per_agent() -> None:
    """RunStartedEvent.agent_name should reflect each agent's unique name."""
    for name in ("engineer", "historian", "visionary"):
        handle, event_bus = _make_handle(agent_name=name, agent_type="native")
        await _run_and_collect(handle)

        published = _get_published_events(event_bus)
        run_started = [e for e in published if isinstance(e, RunStartedEvent)]
        assert run_started[0].agent_name == name


# ---------------------------------------------------------------------------
# RunErrorEvent.agent_name
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_run_error_event_uses_agent_name_not_type() -> None:
    """RunErrorEvent.agent_name should be the agent's name, not its type.

    Given an agent named "librarian" with type "native", if the turn raises
    an exception, the RunErrorEvent should carry agent_name="librarian".
    """
    handle, event_bus = _make_handle(
        agent_name="librarian",
        agent_type="native",
        raise_exc=RuntimeError("test error"),
    )
    gen = handle.start("hello")
    events: list[Any] = []

    async def _consume() -> None:
        with contextlib.suppress(Exception):
            events.extend([event async for event in gen])

    task = asyncio.create_task(_consume())
    await asyncio.sleep(0.05)
    handle.close()
    await asyncio.sleep(0.05)
    await task

    published = _get_published_events(event_bus)
    error_events = [e for e in published if isinstance(e, RunErrorEvent)]
    assert len(error_events) >= 1
    assert error_events[0].agent_name == "librarian"
