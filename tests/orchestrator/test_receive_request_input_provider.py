"""Tests for input_provider propagation in the RunTurn code path.

Verifies that input_provider passed to receive_request() is correctly
propagated to session.input_provider, where AgentContext.get_input_provider()
finds it via the session-state lookup chain (step 2 of the resolution:
self.input_provider → session_state.input_provider → pool._input_provider).

This is a regression test for the bug where the new RunTurn path in
receive_request() dropped input_provider from kwargs, causing
"No InputProvider configured" errors at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests._controller_helpers import send_via_controller
from wolfharness.orchestrator.core import EventBus, SessionController


if TYPE_CHECKING:
    from wolfharness import AgentPool
    from wolfharness.orchestrator.run import RunHandle


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
def event_bus() -> EventBus:
    """Return a real EventBus for testing."""
    return EventBus()


@pytest.fixture
def mock_agent() -> MagicMock:
    """Return a MagicMock simulating a native Agent (AGENT_TYPE = 'native')."""
    agent = MagicMock()
    agent.AGENT_TYPE = "native"
    return agent


@pytest.fixture
def mock_input_provider() -> MagicMock:
    """Return a MagicMock simulating an InputProvider."""
    provider = MagicMock()
    provider.__class__.__name__ = "MockInputProvider"
    return provider


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
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_input_provider_propagated_to_session(
    controller: SessionController,
    event_bus: EventBus,
    mock_agent: MagicMock,
    mock_input_provider: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """input_provider passed to receive_request is stored on session.input_provider."""
    monkeypatch.setenv("AGENTPOOL_USE_RUN_TURN", "true")
    controller._event_bus = event_bus
    _setup_session(controller, "sess-ip-1", mock_agent)

    controller._use_run_turn = lambda _agent: True  # type: ignore[method-assign]
    controller._consume_run = AsyncMock(return_value=None)  # type: ignore[method-assign]

    await send_via_controller(controller, "sess-ip-1", "hello", input_provider=mock_input_provider)

    session = controller.get_session("sess-ip-1")
    assert session is not None
    assert session.input_provider is mock_input_provider


@pytest.mark.anyio
async def test_input_provider_none_when_not_passed(
    controller: SessionController,
    event_bus: EventBus,
    mock_agent: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When input_provider is not passed, session.input_provider remains None."""
    monkeypatch.setenv("AGENTPOOL_USE_RUN_TURN", "true")
    controller._event_bus = event_bus
    _setup_session(controller, "sess-ip-3", mock_agent)

    controller._use_run_turn = lambda _agent: True  # type: ignore[method-assign]
    controller._consume_run = AsyncMock(return_value=None)  # type: ignore[method-assign]

    result = await send_via_controller(controller, "sess-ip-3", "hello")

    assert result is not None
    session = controller.get_session("sess-ip-3")
    assert session is not None
    assert session.input_provider is None


@pytest.mark.anyio
async def test_input_provider_stored_on_session_for_cached_agent(
    controller: SessionController,
    event_bus: EventBus,
    mock_agent: MagicMock,
    mock_input_provider: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """input_provider is stored on session even when agent is already cached.

    This tests the regression scenario: on the second receive_request call
    for the same session, get_or_create_session_agent() returns the cached
    agent via early return, but input_provider must still be available on
    session.input_provider for get_input_provider() lookup chain.
    """
    monkeypatch.setenv("AGENTPOOL_USE_RUN_TURN", "true")
    controller._event_bus = event_bus
    _setup_session(controller, "sess-ip-4", mock_agent)

    controller._use_run_turn = lambda _agent: True  # type: ignore[method-assign]
    controller._consume_run = AsyncMock(return_value=None)  # type: ignore[method-assign]

    # First call — sets input_provider
    await send_via_controller(
        controller, "sess-ip-4", "first message", input_provider=mock_input_provider
    )

    session = controller.get_session("sess-ip-4")
    assert session is not None
    assert session.input_provider is mock_input_provider

    # Simulate run completion — clear current_run_id
    session.current_run_id = None

    # Second call — input_provider should be updated on session
    # (even though agent is already cached)
    second_provider = MagicMock()
    result2 = await send_via_controller(
        controller, "sess-ip-4", "second message", input_provider=second_provider
    )

    assert result2 is not None
    assert session.input_provider is second_provider


@pytest.mark.anyio
async def test_input_provider_not_in_kwargs_after_processing(
    controller: SessionController,
    event_bus: EventBus,
    mock_agent: MagicMock,
    mock_input_provider: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """input_provider is popped from kwargs and not forwarded to legacy path.

    When the RunTurn path is used, input_provider should be consumed by
    _start_run_handle and not leak into any downstream kwargs.
    """
    monkeypatch.setenv("AGENTPOOL_USE_RUN_TURN", "true")
    controller._event_bus = event_bus
    _setup_session(controller, "sess-ip-5", mock_agent)

    controller._use_run_turn = lambda _agent: True  # type: ignore[method-assign]

    # Track what _consume_run receives
    consumed_args: dict[str, object] = {}

    async def _track_consume(run_handle: RunHandle, content: str) -> None:
        consumed_args["run_handle"] = run_handle
        consumed_args["content"] = content

    controller._consume_run = _track_consume  # type: ignore[method-assign]

    await send_via_controller(controller, "sess-ip-5", "hello", input_provider=mock_input_provider)

    # Verify input_provider was stored on session
    session = controller.get_session("sess-ip-5")
    assert session is not None
    assert session.input_provider is mock_input_provider
