"""Test that get_or_load_session() broadcasts SessionUpdatedEvent when loading from storage.

When a session is not cached and must be loaded from storage (the "cold load" path),
the TUI never learns about the session via SSE unless SessionUpdatedEvent is broadcast.
This test verifies that the cold load path broadcasts the event.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

import pytest

from wolfharness.sessions.models import SessionData
from wolfharness_server.opencode_server.models import Session, SessionUpdatedEvent
from wolfharness_server.opencode_server.routes.session_routes import get_or_load_session


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from pathlib import Path

    from wolfharness_server.opencode_server.state import ServerState


@pytest.fixture
def session_data() -> SessionData:
    """Create a SessionData that simulates data loaded from storage."""
    return SessionData(
        session_id="cold-load-session-001",
        agent_name="test-agent",
        cwd="/tmp/test-project",
        version="1",
        created_at=datetime(2025, 1, 1, 12, 0, 0),
        last_active=datetime(2025, 1, 1, 12, 0, 0),
        metadata={"title": "Cold Load Session"},
    )


@pytest.fixture
def mock_state_and_broadcast(
    tmp_path: Path, session_data: SessionData
) -> tuple[ServerState, AsyncMock]:
    """Create a ServerState with a mock agent and capture the broadcast_event mock.

    Returns a tuple of (state, broadcast_mock) so the test can inspect
    broadcast calls without fighting pyright's type narrowing on overridden methods.
    """
    from wolfharness_server.opencode_server.state import ServerState

    agent = Mock()

    agent.model_name = None  # resolve_default_model_info() fallback
    agent.name = "test-agent"
    agent.session_id = None  # No session currently loaded — forces cold load
    agent.load_session = AsyncMock(return_value=session_data)
    # Explicitly set agent_pool so the SessionPool cold-load path is skipped
    agent.host_context = Mock()
    agent._agent_pool = agent.host_context  # state.py resolves _pool via agent._agent_pool
    agent.host_context.session_pool = None

    # agent.conversation.chat_messages must be iterable (empty for cold load test)
    conversation = Mock()
    conversation.chat_messages = []
    agent.conversation = conversation

    state = ServerState(working_dir=str(tmp_path), agent=agent)
    # Initialize backward-compat dicts removed from ServerState dataclass
    # so tests and helper fallbacks can access them.
    state.messages = {}  # type: ignore[attr-defined]
    # Override broadcast_event with an AsyncMock so we can assert calls
    broadcast_mock = AsyncMock()
    state.broadcast_event = broadcast_mock  # type: ignore[method-assign]
    return state, broadcast_mock


async def test_cold_load_broadcasts_session_updated_event(
    mock_state_and_broadcast: tuple[ServerState, AsyncMock],
    session_data: SessionData,
) -> None:
    """Verify that loading a session from storage broadcasts SessionUpdatedEvent.

    When get_or_load_session() encounters a session that is not in the cache
    (state.sessions) and loads it from storage via agent.load_session(), it
    must broadcast SessionUpdatedEvent so the TUI learns about the session.
    """
    mock_state, broadcast_mock = mock_state_and_broadcast
    session_id = session_data.session_id

    # Pre-condition: session is NOT in cache, so cold load path is taken
    assert session_id not in mock_state.sessions
    assert session_id not in mock_state.messages  # type: ignore[attr-defined]

    result = await get_or_load_session(mock_state, session_id)

    # The session should have been loaded successfully
    assert result is not None
    assert isinstance(result, Session)
    assert result.id == session_id

    # The session must now be cached
    assert mock_state.sessions[session_id] is result

    # broadcast_event should have been called with SessionUpdatedEvent
    broadcast_mock.assert_any_call(
        SessionUpdatedEvent.create(result),
    )

    # Verify that the broadcasted event contains the loaded session
    updated_calls = [
        call
        for call in broadcast_mock.call_args_list
        if isinstance(call[0][0], SessionUpdatedEvent)
    ]
    assert len(updated_calls) >= 1, (
        "Expected at least one SessionUpdatedEvent broadcast during cold load"
    )
    event = updated_calls[0][0][0]
    assert isinstance(event, SessionUpdatedEvent)
    assert event.properties.info.id == session_id
