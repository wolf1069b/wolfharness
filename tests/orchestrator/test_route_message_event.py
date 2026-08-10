"""Unit tests for ``UserMessageInsertedEvent`` publication from ``_route_message()``.

Tests all three delivery paths (initial, steer, followup), the
EventBus=None guard, and ``message_id`` pass-through.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from wolfharness.agents.events import UserMessageInsertedEvent
from wolfharness.orchestrator.core import EventBus, SessionController, SessionState


if TYPE_CHECKING:
    from wolfharness import AgentPool


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def controller(minimal_pool: AgentPool) -> SessionController:
    """Return a SessionController backed by the real pool."""
    assert minimal_pool.session_pool is not None
    return minimal_pool.session_pool.sessions


@pytest.fixture
def mock_event_bus() -> EventBus:
    """Return a real EventBus for testing."""
    return EventBus()


@pytest.fixture
def mock_agent() -> MagicMock:
    """Return a MagicMock simulating a native agent."""
    agent = MagicMock()
    agent.AGENT_TYPE = "native"
    agent.name = "test-agent"
    agent.conversation = MagicMock()
    agent.conversation.get_history = MagicMock(return_value=[])
    agent._cancelled = False
    return agent


def _make_session(session_id: str = "sess-1") -> SessionState:
    """Create a real SessionState for testing."""
    return SessionState(session_id=session_id, agent_name="test-agent")


class _EventRecorder:
    """Records events published to the EventBus."""

    def __init__(self) -> None:
        self.published: list[tuple[str, Any]] = []

    async def publish(self, session_id: str, event: Any) -> None:
        self.published.append((session_id, event))

    async def close_session(self, session_id: str) -> None:
        """No-op close for test compatibility."""


# ---------------------------------------------------------------------------
# Tests: delivery paths
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_route_message_initial_delivery_publishes_event(
    controller: SessionController,
    mock_agent: MagicMock,
) -> None:
    """Given an idle session, _route_message publishes event with delivery='initial'."""
    recorder = _EventRecorder()
    controller._event_bus = recorder  # type: ignore[assignment]
    session = _make_session()
    controller._sessions["sess-1"] = session
    controller._consume_run = AsyncMock(return_value=None)  # type: ignore[method-assign]

    result = await controller._route_message(
        session,
        mock_agent,
        "sess-1",
        "hello world",
        priority="when_idle",
    )

    # Event was published before routing action.
    assert len(recorder.published) == 1
    sid, event = recorder.published[0]
    assert sid == "sess-1"
    assert isinstance(event, UserMessageInsertedEvent)
    assert event.delivery == "initial"
    assert event.source == "accepted"
    assert event.content == "hello world"
    assert result is not None


@pytest.mark.anyio
async def test_route_message_steer_delivery_publishes_event(
    controller: SessionController,
    mock_agent: MagicMock,
) -> None:
    """Given a busy session + asap, _route_message publishes event with delivery='steer'."""
    recorder = _EventRecorder()
    controller._event_bus = recorder  # type: ignore[assignment]
    session = _make_session()
    controller._sessions["sess-1"] = session

    # Simulate an active run.
    mock_run = MagicMock()
    mock_run.steer = MagicMock(return_value="steer-mid-123")
    mock_run.complete_event = asyncio.Event()
    run_id = "active-run-1"
    session.set_current_run_id(run_id)
    controller._runs[run_id] = mock_run

    result = await controller._route_message(
        session,
        mock_agent,
        "sess-1",
        "steer me",
        priority="asap",
    )

    assert len(recorder.published) == 1
    _, event = recorder.published[0]
    assert isinstance(event, UserMessageInsertedEvent)
    assert event.delivery == "steer"
    assert event.source == "accepted"
    assert event.content == "steer me"
    assert result == "steer-mid-123"
    mock_run.steer.assert_called_once()


@pytest.mark.anyio
async def test_route_message_followup_delivery_publishes_event(
    controller: SessionController,
    mock_agent: MagicMock,
) -> None:
    """Given a busy session + when_idle, _route_message publishes event with delivery='followup'."""
    recorder = _EventRecorder()
    controller._event_bus = recorder  # type: ignore[assignment]
    session = _make_session()
    controller._sessions["sess-1"] = session

    # Simulate an active run.
    mock_run = MagicMock()
    mock_run.complete_event = asyncio.Event()
    run_id = "active-run-2"
    session.set_current_run_id(run_id)
    controller._runs[run_id] = mock_run

    result = await controller._route_message(
        session,
        mock_agent,
        "sess-1",
        "followup text",
        priority="when_idle",
    )

    assert len(recorder.published) == 1
    _, event = recorder.published[0]
    assert isinstance(event, UserMessageInsertedEvent)
    assert event.delivery == "followup"
    assert event.source == "accepted"
    assert event.content == "followup text"
    assert result is None  # Queued — _wait_and_finalize will early-return
    # Prompt should be enqueued.
    assert not session.prompt_queue.empty()


# ---------------------------------------------------------------------------
# Tests: EventBus=None guard
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_route_message_no_event_bus_does_not_crash(
    controller: SessionController,
    mock_agent: MagicMock,
) -> None:
    """Given EventBus=None, _route_message does not crash and does not publish."""
    controller._event_bus = None
    session = _make_session()
    controller._sessions["sess-1"] = session
    controller._consume_run = AsyncMock(return_value=None)  # type: ignore[method-assign]

    result = await controller._route_message(
        session,
        mock_agent,
        "sess-1",
        "hello",
        priority="when_idle",
    )

    # Routing still succeeds.
    assert result is not None


# ---------------------------------------------------------------------------
# Tests: message_id pass-through
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_route_message_passes_message_id_through(
    controller: SessionController,
    mock_agent: MagicMock,
) -> None:
    """Given an explicit message_id, _route_message publishes event with that ID."""
    recorder = _EventRecorder()
    controller._event_bus = recorder  # type: ignore[assignment]
    session = _make_session()
    controller._sessions["sess-1"] = session
    controller._consume_run = AsyncMock(return_value=None)  # type: ignore[method-assign]

    await controller._route_message(
        session,
        mock_agent,
        "sess-1",
        "hello",
        priority="when_idle",
        message_id="custom-mid-999",
    )

    assert len(recorder.published) == 1
    _, event = recorder.published[0]
    assert event.message_id == "custom-mid-999"


@pytest.mark.anyio
async def test_route_message_generates_message_id_when_none(
    controller: SessionController,
    mock_agent: MagicMock,
) -> None:
    """Given message_id=None, _route_message generates a UUID for the event."""
    recorder = _EventRecorder()
    controller._event_bus = recorder  # type: ignore[assignment]
    session = _make_session()
    controller._sessions["sess-1"] = session
    controller._consume_run = AsyncMock(return_value=None)  # type: ignore[method-assign]

    await controller._route_message(
        session,
        mock_agent,
        "sess-1",
        "hello",
        priority="when_idle",
    )

    assert len(recorder.published) == 1
    _, event = recorder.published[0]
    assert event.message_id  # Non-empty
    assert len(event.message_id) > 0


# ---------------------------------------------------------------------------
# Tests: explicit delivery parameter
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_route_message_explicit_delivery_overrides_inference(
    controller: SessionController,
    mock_agent: MagicMock,
) -> None:
    """Given an explicit delivery, _route_message uses it instead of inferring."""
    recorder = _EventRecorder()
    controller._event_bus = recorder  # type: ignore[assignment]
    session = _make_session()
    controller._sessions["sess-1"] = session
    controller._consume_run = AsyncMock(return_value=None)  # type: ignore[method-assign]

    await controller._route_message(
        session,
        mock_agent,
        "sess-1",
        "hello",
        priority="when_idle",
        delivery="steer",
    )

    assert len(recorder.published) == 1
    _, event = recorder.published[0]
    assert event.delivery == "steer"
