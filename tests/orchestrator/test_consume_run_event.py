"""Unit tests for ``_consume_run()`` followup-from-queue behavior.

Verifies that when ``_consume_run()`` picks up a followup prompt from
``prompt_queue``, it does NOT publish ``UserMessageInsertedEvent``
(since ``_route_message()`` already displayed the message before
enqueuing).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from wolfharness.agents.events import StreamCompleteEvent, UserMessageInsertedEvent
from wolfharness.lifecycle.comm_channel import DirectChannel
from wolfharness.lifecycle.journal import MemoryJournal
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.core import SessionController, SessionState


if TYPE_CHECKING:
    from wolfharness import AgentPool
    from wolfharness.orchestrator.run import RunHandle


pytestmark = pytest.mark.unit


class _EventRecorder:
    """Records events published to a mock EventBus."""

    def __init__(self) -> None:
        self.published: list[tuple[str, Any]] = []

    async def publish(self, session_id: str, event: Any) -> None:
        self.published.append((session_id, event))

    async def close_session(self, session_id: str) -> None:
        """No-op close for test compatibility."""


def _make_session(session_id: str = "cr-sess-1") -> SessionState:
    """Create a real SessionState with a DirectChannel for testing."""
    session = SessionState(session_id=session_id, agent_name="test-agent")
    session._comm_channel = DirectChannel(MemoryJournal())
    return session


def _make_stub_turn_events() -> list[Any]:
    """Return a minimal list of events that _consume_run can drain."""
    msg = ChatMessage(content="done", role="assistant")
    return [StreamCompleteEvent(message=msg)]


def _make_mock_agent() -> MagicMock:
    """Create a mock agent for testing."""
    mock_agent = MagicMock()
    mock_agent.AGENT_TYPE = "native"
    mock_agent.name = "test-agent"
    mock_agent.conversation = MagicMock()
    mock_agent.conversation.get_history = MagicMock(return_value=[])
    mock_agent._cancelled = False
    return mock_agent


def _stub_start_factory() -> Any:
    """Return a stub ``start`` method that yields minimal events."""

    def _stub_start(prompt: str | list[Any] | None = None) -> Any:
        async def _gen() -> Any:
            for event in _make_stub_turn_events():
                yield event

        return _gen()

    return _stub_start


@pytest.fixture
def controller(minimal_pool: AgentPool) -> SessionController:
    """Return a SessionController backed by the real pool."""
    assert minimal_pool.session_pool is not None
    return minimal_pool.session_pool.sessions


# ---------------------------------------------------------------------------
# Tests: followup-from-queue — NO event published (single-path architecture)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_consume_run_does_not_publish_event_for_followup_from_queue(
    controller: SessionController,
) -> None:
    """Given a prompt in prompt_queue, _consume_run does NOT publish UserMessageInsertedEvent.

    In the single-path display architecture, ``_route_message()`` is the
    sole publication point. Messages in ``prompt_queue`` were already
    displayed by ``_route_message()`` before being enqueued, so
    ``_consume_run()`` must NOT re-emit the event.
    """
    recorder = _EventRecorder()
    controller._event_bus = recorder  # type: ignore[assignment]

    session = _make_session("cr-sess-1")
    controller._sessions["cr-sess-1"] = session

    # Enqueue a followup prompt.
    session.prompt_queue.put_nowait("followup prompt text")

    mock_agent = _make_mock_agent()

    # Create the initial RunHandle via _create_per_prompt_handle.
    initial_handle = controller._create_per_prompt_handle(
        session,
        mock_agent,
        "initial prompt",
    )
    initial_handle.start = _stub_start_factory()  # type: ignore[method-assign]

    # Intercept _create_per_prompt_handle for the second call (followup).
    original_create = controller._create_per_prompt_handle

    def _stub_create(
        sess: SessionState,
        agt: Any,
        prompt: str | list[Any],
    ) -> RunHandle:
        handle = original_create(sess, agt, prompt)
        handle.start = _stub_start_factory()  # type: ignore[method-assign]
        return handle

    controller._create_per_prompt_handle = _stub_create  # type: ignore[method-assign]

    # Drive _consume_run to completion.
    await controller._consume_run(initial_handle, "initial prompt")

    # No UserMessageInsertedEvent should have been published —
    # _route_message() already displayed the message before enqueuing.
    user_msg_events = [
        (sid, ev) for sid, ev in recorder.published if isinstance(ev, UserMessageInsertedEvent)
    ]
    assert len(user_msg_events) == 0


# ---------------------------------------------------------------------------
# Tests: no followup in queue — no event published
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_consume_run_no_followup_no_event(
    controller: SessionController,
) -> None:
    """Given an empty prompt_queue, _consume_run does not publish UserMessageInsertedEvent."""
    recorder = _EventRecorder()
    controller._event_bus = recorder  # type: ignore[assignment]

    session = _make_session("cr-sess-2")
    controller._sessions["cr-sess-2"] = session

    mock_agent = _make_mock_agent()

    initial_handle = controller._create_per_prompt_handle(
        session,
        mock_agent,
        "only prompt",
    )
    initial_handle.start = _stub_start_factory()  # type: ignore[method-assign]

    await controller._consume_run(initial_handle, "only prompt")

    # No UserMessageInsertedEvent should have been published.
    user_msg_events = [
        ev for _, ev in recorder.published if isinstance(ev, UserMessageInsertedEvent)
    ]
    assert len(user_msg_events) == 0


# ---------------------------------------------------------------------------
# Tests: EventBus=None guard
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_consume_run_no_event_bus_does_not_crash(
    controller: SessionController,
) -> None:
    """Given EventBus=None, _consume_run does not crash on followup pickup."""
    controller._event_bus = None

    session = _make_session("cr-sess-3")
    controller._sessions["cr-sess-3"] = session

    # Enqueue a followup prompt.
    session.prompt_queue.put_nowait("followup with no bus")

    mock_agent = _make_mock_agent()

    initial_handle = controller._create_per_prompt_handle(
        session,
        mock_agent,
        "initial",
    )
    initial_handle.start = _stub_start_factory()  # type: ignore[method-assign]

    original_create = controller._create_per_prompt_handle

    def _stub_create(
        sess: SessionState,
        agt: Any,
        prompt: str | list[Any],
    ) -> RunHandle:
        handle = original_create(sess, agt, prompt)
        handle.start = _stub_start_factory()  # type: ignore[method-assign]
        return handle

    controller._create_per_prompt_handle = _stub_create  # type: ignore[method-assign]

    # Should not crash.
    await controller._consume_run(initial_handle, "initial")
