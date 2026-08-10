"""Tests for SessionController.receive_request() RunHandle path.

Covers five scenarios:
1. Idle session -> creates RunHandle, returns message_id.
2. Busy session + asap -> calls RunHandle.steer(), returns message_id.
3. Busy session + when_idle -> calls RunHandle.followup(), returns message_id.
4. Session not found -> returns None.
5. Session closing -> returns None.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests._controller_helpers import send_via_controller
from wolfharness.lifecycle.types import DeliveryMode
from wolfharness.orchestrator.core import EventBus, SessionController
from wolfharness.orchestrator.run import RunHandle


if TYPE_CHECKING:
    from wolfharness import AgentPool


pytestmark = pytest.mark.unit

_L2_SKIP = pytest.mark.skip(reason="L2 migration: requires mock internals — remains L1 unit test")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def controller(minimal_pool: AgentPool) -> SessionController:
    """Return a SessionController backed by the real pool."""
    assert minimal_pool.session_pool is not None
    return minimal_pool.session_pool.sessions


@pytest.fixture
def event_bus() -> EventBus:
    """Return a real EventBus for testing."""
    return EventBus()


@pytest.fixture
def mock_agent() -> MagicMock:
    """Return a MagicMock simulating a native Agent (AGENT_TYPE = 'native')."""
    agent = MagicMock()
    agent.AGENT_TYPE = "native"
    return agent


def _setup_session(
    controller: SessionController,
    session_id: str,
    agent: MagicMock,
) -> None:
    """Create a session and register an agent for it."""
    controller._sessions[session_id] = MagicMock()
    controller._sessions[session_id].session_id = session_id
    controller._sessions[session_id].current_run_id = None
    controller._sessions[session_id].closing = False
    controller._sessions[session_id].is_closing = False
    controller._sessions[session_id]._request_lock = asyncio.Lock()
    controller._sessions[session_id].turn_lock = asyncio.Lock()
    controller._sessions[session_id].input_provider = None
    controller._session_agents[session_id] = agent


# ---------------------------------------------------------------------------
# Test 1: Idle -> creates RunHandle, returns message_id
# ---------------------------------------------------------------------------


@_L2_SKIP
@pytest.mark.anyio
async def test_idle_creates_run_handle_and_returns_message_id(
    controller: SessionController,
    event_bus: EventBus,
    mock_agent: MagicMock,
) -> None:
    """When session is idle, a RunHandle is created and message_id is returned."""
    controller._event_bus = event_bus
    _setup_session(controller, "sess-1", mock_agent)

    # Patch _consume_run so asyncio.create_task doesn't block
    controller._consume_run = AsyncMock(return_value=None)  # type: ignore[method-assign]

    result = await send_via_controller(controller, "sess-1", "hello")

    # receive_request() now returns str | None (message_id)
    assert result is not None
    assert isinstance(result, str)
    # A RunHandle should have been created and registered
    session = controller.get_session("sess-1")
    assert session is not None
    assert session.current_run_id is not None
    run_handle = controller._runs.get(session.current_run_id)
    assert run_handle is not None
    assert run_handle.agent is mock_agent
    assert run_handle.event_bus is event_bus


# ---------------------------------------------------------------------------
# Test 2: Busy + asap -> calls steer(), returns message_id
# ---------------------------------------------------------------------------


@_L2_SKIP
@pytest.mark.anyio
async def test_busy_asap_calls_steer(
    controller: SessionController,
    event_bus: EventBus,
    mock_agent: MagicMock,
) -> None:
    """When session is busy with asap, RunHandle.steer() is called."""
    controller._event_bus = event_bus
    _setup_session(controller, "sess-2", mock_agent)

    # Simulate an active run
    existing_run = MagicMock(spec=RunHandle)
    existing_run.steer = MagicMock(return_value="msg-steer-123")
    existing_run.followup = MagicMock(return_value="msg-followup-123")
    existing_run._run_state = MagicMock()  # Not RunState.DONE
    existing_run.run_id = "existing-run-id"
    controller._runs["existing-run-id"] = existing_run
    controller.get_session("sess-2").current_run_id = "existing-run-id"  # type: ignore[union-attr]

    result = await send_via_controller(controller, "sess-2", "urgent", mode=DeliveryMode.STEER)

    assert result == "msg-steer-123"
    existing_run.steer.assert_called_once_with("urgent", message_id=None)
    existing_run.followup.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: Busy + when_idle -> calls followup(), returns message_id
# ---------------------------------------------------------------------------


@_L2_SKIP
@pytest.mark.anyio
async def test_busy_when_idle_calls_followup(
    controller: SessionController,
    event_bus: EventBus,
    mock_agent: MagicMock,
) -> None:
    """When session is busy with when_idle, RunHandle.followup() is called."""
    controller._event_bus = event_bus
    _setup_session(controller, "sess-3", mock_agent)

    existing_run = MagicMock(spec=RunHandle)
    existing_run.steer = MagicMock(return_value="msg-steer-123")
    existing_run.followup = MagicMock(return_value="msg-followup-123")
    existing_run._run_state = MagicMock()  # Not RunState.DONE
    existing_run.run_id = "existing-run-id"
    controller._runs["existing-run-id"] = existing_run
    controller.get_session("sess-3").current_run_id = "existing-run-id"  # type: ignore[union-attr]

    result = await send_via_controller(controller, "sess-3", "later", mode=DeliveryMode.QUEUE)

    assert result == "msg-followup-123"
    existing_run.followup.assert_called_once_with("later", message_id=None)
    existing_run.steer.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: Session not found -> returns None
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_session_not_found_returns_none(
    controller: SessionController,
    event_bus: EventBus,
) -> None:
    """When the session does not exist, receive_request returns None."""
    controller._event_bus = event_bus

    result = await send_via_controller(controller, "nonexistent-session", "hello")

    assert result is None


# ---------------------------------------------------------------------------
# Test 5: Session closing -> returns None
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_session_closing_returns_none(
    controller: SessionController,
    event_bus: EventBus,
) -> None:
    """When the session is closing, receive_request returns None."""
    controller._event_bus = event_bus
    # Use real session from the real pool
    await controller.get_or_create_session("sess-closing", agent_name="test_agent")
    session = controller.get_session("sess-closing")
    assert session is not None
    session.is_closing = True

    result = await send_via_controller(controller, "sess-closing", "hello")

    assert result is None


# ---------------------------------------------------------------------------
# Test 6: message_id parameter is passed through
# ---------------------------------------------------------------------------


@_L2_SKIP
@pytest.mark.anyio
async def test_message_id_passed_to_steer(
    controller: SessionController,
    event_bus: EventBus,
    mock_agent: MagicMock,
) -> None:
    """message_id keyword argument is forwarded to steer()."""
    controller._event_bus = event_bus
    _setup_session(controller, "sess-mid", mock_agent)

    existing_run = MagicMock(spec=RunHandle)
    existing_run.steer = MagicMock(return_value="custom-mid")
    existing_run.followup = MagicMock(return_value="custom-mid")
    existing_run._run_state = MagicMock()
    existing_run.run_id = "existing-run-id"
    controller._runs["existing-run-id"] = existing_run
    controller.get_session("sess-mid").current_run_id = "existing-run-id"  # type: ignore[union-attr]

    result = await send_via_controller(
        controller,
        "sess-mid",
        "steer me",
        mode=DeliveryMode.STEER,
        message_id="custom-mid",
    )

    assert result == "custom-mid"
    existing_run.steer.assert_called_once_with("steer me", message_id="custom-mid")


# ---------------------------------------------------------------------------
# Test 7: List content is passed directly without stringification
# ---------------------------------------------------------------------------


@_L2_SKIP
@pytest.mark.anyio
async def test_list_content_not_stringified_for_steer(
    controller: SessionController,
    event_bus: EventBus,
    mock_agent: MagicMock,
) -> None:
    """List content is passed directly to steer() without stringification (D9)."""
    controller._event_bus = event_bus
    _setup_session(controller, "sess-list", mock_agent)

    existing_run = MagicMock(spec=RunHandle)
    existing_run.steer = MagicMock(return_value="msg-id")
    existing_run.followup = MagicMock(return_value="msg-id")
    existing_run._run_state = MagicMock()
    existing_run.run_id = "existing-run-id"
    controller._runs["existing-run-id"] = existing_run
    controller.get_session("sess-list").current_run_id = "existing-run-id"  # type: ignore[union-attr]

    content_list: list[Any] = ["hello", "world"]
    result = await send_via_controller(
        controller, "sess-list", content_list, mode=DeliveryMode.STEER
    )

    assert result == "msg-id"
    # List should be passed directly, not joined into a string
    existing_run.steer.assert_called_once_with(content_list, message_id=None)


# ---------------------------------------------------------------------------
# Test 8: receive_request uses get_or_create_session_agent
# ---------------------------------------------------------------------------


@_L2_SKIP
@pytest.mark.asyncio
async def test_receive_request_uses_get_or_create_session_agent(minimal_pool: AgentPool) -> None:
    """receive_request should use get_or_create_session_agent, not .get().

    When agent is not yet cached (new top-level sessions), .get()
    returns None and receive_request silently does nothing.
    """
    from wolfharness.orchestrator.core import SessionController

    controller = SessionController(pool=minimal_pool)
    event_bus = EventBus()
    controller._event_bus = event_bus

    mock_agent = MagicMock()
    mock_agent.AGENT_TYPE = "native"

    session_id = "sess-lazy"
    controller._sessions[session_id] = MagicMock()
    controller._sessions[session_id].session_id = session_id
    controller._sessions[session_id].current_run_id = None
    controller._sessions[session_id].closing = False
    controller._sessions[session_id].is_closing = False
    controller._sessions[session_id]._request_lock = asyncio.Lock()
    controller._sessions[session_id].turn_lock = asyncio.Lock()
    controller._sessions[session_id].input_provider = None
    # Deliberately do NOT pre-register agent in _session_agents

    # Mock get_or_create_session_agent to return the agent
    controller.get_or_create_session_agent = AsyncMock(return_value=mock_agent)  # type: ignore[method-assign]
    controller._consume_run = AsyncMock(return_value=None)  # type: ignore[method-assign]

    result = await send_via_controller(controller, session_id, "hello")

    # get_or_create_session_agent should have been called
    controller.get_or_create_session_agent.assert_called_once_with(session_id, input_provider=None)
    assert result is not None, (
        "receive_request returned None because agent was not in _session_agents cache"
    )


# ---------------------------------------------------------------------------
# Test 9: revoke_inject()
# ---------------------------------------------------------------------------


@_L2_SKIP
@pytest.mark.anyio
async def test_revoke_inject_calls_run_handle_revoke(
    controller: SessionController,
    event_bus: EventBus,
    mock_agent: MagicMock,
) -> None:
    """revoke_inject() delegates to RunHandle.revoke() on the active run."""
    controller._event_bus = event_bus
    _setup_session(controller, "sess-revoke", mock_agent)

    existing_run = MagicMock(spec=RunHandle)
    existing_run.revoke = MagicMock(return_value=True)
    existing_run.run_id = "existing-run-id"
    controller._runs["existing-run-id"] = existing_run
    controller.get_session("sess-revoke").current_run_id = "existing-run-id"  # type: ignore[union-attr]

    result = controller.revoke_inject("sess-revoke", "msg-123")

    assert result is True
    existing_run.revoke.assert_called_once_with("msg-123")


@pytest.mark.anyio
async def test_revoke_inject_no_session_returns_false(
    controller: SessionController,
) -> None:
    """revoke_inject() returns False when session doesn't exist."""
    result = controller.revoke_inject("nonexistent", "msg-123")
    assert result is False


@pytest.mark.anyio
async def test_revoke_inject_no_run_returns_false(
    controller: SessionController,
    event_bus: EventBus,
) -> None:
    """revoke_inject() returns False when session has no active run."""
    controller._event_bus = event_bus
    # Use real session from the real pool — no active run
    await controller.get_or_create_session("sess-no-run", agent_name="test_agent")

    result = controller.revoke_inject("sess-no-run", "msg-123")
    assert result is False


# ---------------------------------------------------------------------------
# Test 10: wait_for_completion()
# ---------------------------------------------------------------------------


@_L2_SKIP
@pytest.mark.anyio
async def test_wait_for_completion_waits_for_complete_event(
    controller: SessionController,
    event_bus: EventBus,
    mock_agent: MagicMock,
) -> None:
    """wait_for_completion() awaits RunHandle.complete_event."""
    controller._event_bus = event_bus
    _setup_session(controller, "sess-wait", mock_agent)

    existing_run = MagicMock(spec=RunHandle)
    existing_run.complete_event = asyncio.Event()
    existing_run.run_id = "existing-run-id"
    controller._runs["existing-run-id"] = existing_run
    controller.get_session("sess-wait").current_run_id = "existing-run-id"  # type: ignore[union-attr]

    # Set the event after a short delay
    async def _set_event() -> None:
        await asyncio.sleep(0.05)
        existing_run.complete_event.set()

    task = asyncio.create_task(_set_event())

    result = await controller.wait_for_completion("sess-wait", timeout=2.0)

    assert result == "sess-wait"
    await task


@pytest.mark.anyio
async def test_wait_for_completion_no_run_returns_immediately(
    controller: SessionController,
    event_bus: EventBus,
) -> None:
    """wait_for_completion() returns immediately when no active run exists."""
    controller._event_bus = event_bus
    # Use real session from the real pool — no active run
    await controller.get_or_create_session("sess-idle", agent_name="test_agent")

    result = await controller.wait_for_completion("sess-idle", timeout=2.0)

    assert result == "sess-idle"


@pytest.mark.anyio
async def test_wait_for_completion_session_not_found_raises(
    controller: SessionController,
) -> None:
    """wait_for_completion() raises SessionNotFoundError for unknown session."""
    from wolfharness.orchestrator.session_controller import SessionNotFoundError

    with pytest.raises(SessionNotFoundError, match="Session not found"):
        await controller.wait_for_completion("nonexistent", timeout=1.0)


@_L2_SKIP
@pytest.mark.anyio
async def test_wait_for_completion_timeout(
    controller: SessionController,
    event_bus: EventBus,
    mock_agent: MagicMock,
) -> None:
    """wait_for_completion() raises TimeoutError when run doesn't complete in time."""
    controller._event_bus = event_bus
    _setup_session(controller, "sess-slow", mock_agent)

    existing_run = MagicMock(spec=RunHandle)
    existing_run.complete_event = asyncio.Event()  # Never set
    existing_run.run_id = "existing-run-id"
    controller._runs["existing-run-id"] = existing_run
    controller.get_session("sess-slow").current_run_id = "existing-run-id"  # type: ignore[union-attr]

    with pytest.raises(TimeoutError):
        await controller.wait_for_completion("sess-slow", timeout=0.05)
