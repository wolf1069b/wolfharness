"""Tests for SessionController.receive_request() ACP RunHandle path.

Covers three scenarios:
1. ACPAgent + idle -> creates RunHandle, returns message_id.
2. ACPAgent + busy + asap -> calls RunHandle.steer().
3. ACPAgent + busy + when_idle -> calls RunHandle.followup().
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests._controller_helpers import send_via_controller
from wolfharness.agents.acp_agent import ACPAgent
from wolfharness.lifecycle import RunState
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
def mock_acp_agent() -> MagicMock:
    """Return a MagicMock that isinstance-checks as ACPAgent."""
    agent = MagicMock(spec=ACPAgent)
    agent.AGENT_TYPE = "acp"
    agent.conversation = None  # No conversation history in mock
    return agent


def _setup_session(
    controller: SessionController,
    session_id: str,
    agent: MagicMock,
) -> None:
    """Create a session and register an agent for it."""
    import asyncio

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
# Test 1: ACPAgent + idle -> creates RunHandle, returns message_id
# ---------------------------------------------------------------------------


@_L2_SKIP
@pytest.mark.anyio
async def test_acp_idle_creates_run_handle_and_returns_message_id(
    controller: SessionController,
    event_bus: EventBus,
    mock_acp_agent: MagicMock,
) -> None:
    """When session is idle, a RunHandle is created and message_id returned."""
    controller._event_bus = event_bus
    _setup_session(controller, "sess-1", mock_acp_agent)

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
    assert run_handle.agent is mock_acp_agent
    assert run_handle.event_bus is event_bus


# ---------------------------------------------------------------------------
# Test 2: ACPAgent + busy + asap -> calls steer()
# ---------------------------------------------------------------------------


@_L2_SKIP
@pytest.mark.anyio
async def test_acp_busy_asap_calls_steer(
    controller: SessionController,
    event_bus: EventBus,
    mock_acp_agent: MagicMock,
) -> None:
    """When busy with asap, RunHandle.steer() is called."""
    controller._event_bus = event_bus
    _setup_session(controller, "sess-2", mock_acp_agent)

    existing_run = MagicMock(spec=RunHandle)
    existing_run.steer = MagicMock(return_value="msg-steer-id")
    existing_run.followup = MagicMock(return_value="msg-followup-id")
    existing_run._run_state = MagicMock()  # Not RunState.DONE
    existing_run.run_id = "existing-run-id"
    controller._runs["existing-run-id"] = existing_run
    controller.get_session("sess-2").current_run_id = "existing-run-id"  # type: ignore[union-attr]

    result = await send_via_controller(controller, "sess-2", "urgent", mode=DeliveryMode.STEER)

    assert result == "msg-steer-id"
    existing_run.steer.assert_called_once_with("urgent", message_id=None)
    existing_run.followup.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: ACPAgent + busy + when_idle -> calls followup()
# ---------------------------------------------------------------------------


@_L2_SKIP
@pytest.mark.anyio
async def test_acp_busy_when_idle_calls_followup(
    controller: SessionController,
    event_bus: EventBus,
    mock_acp_agent: MagicMock,
) -> None:
    """When busy with when_idle, followup() is called."""
    controller._event_bus = event_bus
    _setup_session(controller, "sess-3", mock_acp_agent)

    existing_run = MagicMock(spec=RunHandle)
    existing_run.steer = MagicMock(return_value="msg-steer-id")
    existing_run.followup = MagicMock(return_value="msg-followup-id")
    existing_run._run_state = MagicMock()  # Not RunState.DONE
    existing_run.run_id = "existing-run-id"
    controller._runs["existing-run-id"] = existing_run
    controller.get_session("sess-3").current_run_id = "existing-run-id"  # type: ignore[union-attr]

    result = await send_via_controller(controller, "sess-3", "later", mode=DeliveryMode.QUEUE)

    assert result == "msg-followup-id"
    existing_run.followup.assert_called_once_with("later", message_id=None)
    existing_run.steer.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: Stale current_run_id (missing run) -> clears and starts new run
# ---------------------------------------------------------------------------


@_L2_SKIP
@pytest.mark.anyio
async def test_stale_current_run_id_detected(
    controller: SessionController,
    event_bus: EventBus,
    mock_acp_agent: MagicMock,
) -> None:
    """A stale current_run_id pointing to a missing run is cleared and a new run starts."""
    controller._event_bus = event_bus
    _setup_session(controller, "sess-stale", mock_acp_agent)

    # Set a stale run_id that doesn't exist in _runs
    controller.get_session("sess-stale").current_run_id = "nonexistent-run-id"  # type: ignore[union-attr]

    # Patch _consume_run so asyncio.create_task doesn't block
    controller._consume_run = AsyncMock(return_value=None)  # type: ignore[method-assign]

    result = await send_via_controller(controller, "sess-stale", "test prompt")

    assert result is not None
    assert isinstance(result, str)
    session = controller.get_session("sess-stale")
    assert session is not None
    assert session.current_run_id != "nonexistent-run-id"
    # A new RunHandle should have been created
    assert session.current_run_id in controller._runs


# ---------------------------------------------------------------------------
# Test 5: Failed run in _runs -> stale detection starts new run
# ---------------------------------------------------------------------------


@_L2_SKIP
@pytest.mark.anyio
async def test_cancel_then_receive_request_starts_new_run(
    controller: SessionController,
    event_bus: EventBus,
    mock_acp_agent: MagicMock,
) -> None:
    """A failed run in _runs triggers stale detection and starts a new run."""
    controller._event_bus = event_bus
    _setup_session(controller, "sess-cancel", mock_acp_agent)

    # Create an existing run that has failed
    existing_run = RunHandle(
        run_id="failed-run-id",
        session_id="sess-cancel",
        agent_type="acp",
        event_bus=event_bus,
    )
    existing_run._run_state = RunState.DONE
    controller._runs["failed-run-id"] = existing_run
    controller.get_session("sess-cancel").current_run_id = "failed-run-id"  # type: ignore[union-attr]

    # Patch _consume_run so asyncio.create_task doesn't block
    controller._consume_run = AsyncMock(return_value=None)  # type: ignore[method-assign]

    result = await send_via_controller(controller, "sess-cancel", "new prompt")

    assert result is not None
    assert isinstance(result, str)
    session = controller.get_session("sess-cancel")
    assert session is not None
    assert session.current_run_id != "failed-run-id"
    assert session.current_run_id in controller._runs
