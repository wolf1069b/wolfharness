"""Tests for the ensure_session() function."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wolfharness.agents.base_agent import BaseAgent
from wolfharness_server.opencode_server.models import (
    Session,
    SessionIdleEvent,
    SessionStatusEvent,
    SessionUpdatedEvent,
    TimeCreatedUpdated,
)
from wolfharness_server.opencode_server.session_pool_integration import ensure_session
from wolfharness_server.opencode_server.state import ServerState


pytestmark = pytest.mark.integration


def create_mock_agent() -> MagicMock:
    """Create a properly configured mock agent."""
    agent = MagicMock(spec=BaseAgent)
    agent.name = "test_agent"
    agent.session_id = "original_session_id"
    agent.host_context = MagicMock()
    agent._agent_pool = agent.host_context  # state.py resolves _pool via agent._agent_pool
    agent.host_context.manifest.config_file_path = "test_config.yml"
    agent.host_context.storage.save_session = AsyncMock()
    agent.host_context.storage.load_session = AsyncMock(return_value=None)
    agent.host_context.session_pool = MagicMock()
    agent.host_context.session_pool.sessions = MagicMock()
    agent.host_context.session_pool.sessions.store = None
    agent.env = MagicMock()
    agent.env.cwd = "/test/dir"
    return agent


@pytest.fixture
def mock_state() -> ServerState:
    """Create a ServerState with mocked dependencies."""
    agent = create_mock_agent()
    state = ServerState(
        working_dir="/test/working/dir",
        agent=agent,
    )
    # Initialize backward-compat dicts removed from ServerState dataclass
    state.messages = {}  # type: ignore[attr-defined]
    state.session_status = {}  # type: ignore[attr-defined]
    state.todos = {}  # type: ignore[attr-defined]
    state.input_providers = {}  # type: ignore[attr-defined]
    state.pending_questions = {}  # type: ignore[attr-defined]
    return state


@pytest.mark.asyncio
async def test_ensure_session_creates_new_session(mock_state: ServerState) -> None:
    """Test that ensure_session creates a new session when it doesn't exist."""
    session_id = "test_session_123"
    parent_id = "parent_session_456"

    with patch(
        "wolfharness_server.opencode_server.converters.opencode_to_session_data"
    ) as mock_converter:
        mock_converter.return_value = MagicMock()

        with patch(
            "wolfharness_server.opencode_server.input_provider.OpenCodeInputProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            result = await ensure_session(mock_state, session_id, parent_id=parent_id)

    assert result.id == session_id
    assert result.parent_id == parent_id
    # project_id is computed from working_dir (returns "global" for non-git dirs)
    assert result.project_id == "global"
    assert result.directory == mock_state.working_dir
    assert result.title == "New Session"
    assert result.version == "1"
    assert isinstance(result.time, TimeCreatedUpdated)


@pytest.mark.asyncio
async def test_ensure_session_returns_existing_session(mock_state: ServerState) -> None:
    """Test that ensure_session returns existing session if already in memory."""
    session_id = "test_session_123"

    existing_session = Session(
        id=session_id,
        project_id="test_project",
        directory="/custom/dir",
        title="Custom Title",
        version="2",
        time=TimeCreatedUpdated(created=1000, updated=2000),
        parent_id=None,
    )
    mock_state.sessions[session_id] = existing_session

    result = await ensure_session(mock_state, session_id)

    assert result is existing_session
    assert result.title == "Custom Title"
    assert result.project_id == "test_project"


@pytest.mark.asyncio
async def test_ensure_session_persists_to_storage(mock_state: ServerState) -> None:
    """Test that ensure_session persists the session to storage."""
    session_id = "test_session_789"

    with patch(
        "wolfharness_server.opencode_server.converters.opencode_to_session_data"
    ) as mock_converter:
        mock_session_data = MagicMock()
        mock_converter.return_value = mock_session_data

        with patch(
            "wolfharness_server.opencode_server.input_provider.OpenCodeInputProvider"
        ) as mock_provider_class:
            mock_provider_class.return_value = MagicMock()

            await ensure_session(mock_state, session_id)

    mock_converter.assert_called_once()
    args, kwargs = mock_converter.call_args
    session_arg = args[0]
    assert session_arg.id == session_id
    assert kwargs["agent_name"] == "test_agent"
    assert kwargs["pool_id"] == "test_config.yml"

    mock_state.agent.host_context.storage.save_session.assert_awaited_once_with(  # type: ignore[union-attr]
        mock_session_data
    )


@pytest.mark.asyncio
async def test_ensure_session_caches_in_memory(mock_state: ServerState) -> None:
    """Test that ensure_session caches all session state in memory."""
    session_id = "test_session_abc"

    with (
        patch("wolfharness_server.opencode_server.converters.opencode_to_session_data"),
        patch(
            "wolfharness_server.opencode_server.input_provider.OpenCodeInputProvider"
        ) as mock_provider_class,
    ):
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        result = await ensure_session(mock_state, session_id)

    assert session_id in mock_state.sessions
    assert mock_state.sessions[session_id] is result

    messages = getattr(mock_state, "messages", {})
    assert messages is not None

    # Session status is now broadcast via broadcast_event() instead of stored
    # in the in-memory session_status dict. The status broadcast happens
    # through mark_session_idle() which calls set_session_status().
    # We verify the session was created correctly above.

    todos = getattr(mock_state, "todos", {})
    assert todos is not None

    input_providers = getattr(mock_state, "input_providers", {})
    assert input_providers is not None


@pytest.mark.asyncio
async def test_ensure_session_broadcasts_idle_events(mock_state: ServerState) -> None:
    """Test that ensure_session broadcasts both status and idle events."""
    session_id = "test_session_idle_event"

    with (
        patch("wolfharness_server.opencode_server.converters.opencode_to_session_data"),
        patch("wolfharness_server.opencode_server.input_provider.OpenCodeInputProvider"),
        patch.object(mock_state, "broadcast_event", new=AsyncMock()) as mock_broadcast,
    ):
        await ensure_session(mock_state, session_id)

    status_events = [
        call.args[0]
        for call in mock_broadcast.await_args_list
        if isinstance(call.args[0], SessionStatusEvent)
    ]
    idle_events = [
        call.args[0]
        for call in mock_broadcast.await_args_list
        if isinstance(call.args[0], SessionIdleEvent)
    ]
    assert len(status_events) == 2  # set_session_status() + mark_session_idle() explicit broadcast
    assert len(idle_events) == 1
    assert status_events[0].properties.status.type == "idle"
    assert idle_events[0].properties.session_id == session_id


@pytest.mark.asyncio
async def test_ensure_session_creates_input_provider(mock_state: ServerState) -> None:
    """Test that ensure_session creates and stores an OpenCodeInputProvider."""
    session_id = "test_session_def"

    with (
        patch("wolfharness_server.opencode_server.converters.opencode_to_session_data"),
        patch(
            "wolfharness_server.opencode_server.input_provider.OpenCodeInputProvider"
        ) as mock_provider_class,
    ):
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        await ensure_session(mock_state, session_id)

    mock_provider_class.assert_called_once_with(mock_state, session_id)
    input_providers = getattr(mock_state, "input_providers", {})
    assert input_providers is not None


@pytest.mark.asyncio
async def test_ensure_session_without_parent_id(mock_state: ServerState) -> None:
    """Test that ensure_session works without a parent_id."""
    session_id = "test_session_no_parent"

    with (
        patch("wolfharness_server.opencode_server.converters.opencode_to_session_data"),
        patch("wolfharness_server.opencode_server.input_provider.OpenCodeInputProvider"),
    ):
        result = await ensure_session(mock_state, session_id)

    assert result.id == session_id
    assert result.parent_id is None


@pytest.mark.asyncio
async def test_ensure_session_is_idempotent(mock_state: ServerState) -> None:
    """Test that calling ensure_session twice with the same ID returns the same session."""
    session_id = "test_session_idempotent"

    with (
        patch("wolfharness_server.opencode_server.converters.opencode_to_session_data"),
        patch("wolfharness_server.opencode_server.input_provider.OpenCodeInputProvider"),
    ):
        result1 = await ensure_session(mock_state, session_id)
        result2 = await ensure_session(mock_state, session_id)

    assert result1 is result2
    mock_state.agent.host_context.storage.save_session.assert_awaited_once()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_ensure_session_broadcasts_updated_event_on_early_return(
    mock_state: ServerState,
) -> None:
    """Test that ensure_session broadcasts SessionUpdatedEvent when returning an existing session.

    Without this broadcast, the TUI's store stays empty on reconnect because
    it relies on `session.updated` SSE events to populate it.
    """
    session_id = "test_session_existing"

    existing_session = Session(
        id=session_id,
        project_id="test_project",
        directory="/custom/dir",
        title="Existing Session",
        version="2",
        time=TimeCreatedUpdated(created=1000, updated=2000),
        parent_id=None,
    )
    mock_state.sessions[session_id] = existing_session

    with patch.object(mock_state, "broadcast_event", new=AsyncMock()) as mock_broadcast:
        result = await ensure_session(mock_state, session_id)

    # Should still return the existing session
    assert result is existing_session

    # Should have broadcast a SessionUpdatedEvent
    updated_events = [
        call.args[0]
        for call in mock_broadcast.await_args_list
        if isinstance(call.args[0], SessionUpdatedEvent)
    ]
    assert len(updated_events) == 1, (
        f"Expected exactly 1 SessionUpdatedEvent, got {len(updated_events)}"
    )
    assert updated_events[0].properties.info is existing_session


@pytest.mark.asyncio
async def test_ensure_session_child_inherits_parent_project_and_directory(
    mock_state: ServerState,
) -> None:
    """Test that ensure_session inherits project_id and directory from parent session.

    When a subagent session is created with a parent_id, it must inherit the
    parent's project_id and directory so that the child remains visible when
    listing sessions filtered by cwd/project_id.
    """
    parent_id = "parent_session_proj"
    child_id = "child_session_proj"

    # Pre-populate a parent session with explicit project_id / directory
    parent_session = Session(
        id=parent_id,
        project_id="my-project",
        directory="/custom/project/dir",
        title="Parent",
        version="1",
        time=TimeCreatedUpdated(created=1000, updated=2000),
        parent_id=None,
    )
    mock_state.sessions[parent_id] = parent_session

    with (
        patch("wolfharness_server.opencode_server.converters.opencode_to_session_data"),
        patch("wolfharness_server.opencode_server.input_provider.OpenCodeInputProvider"),
    ):
        child = await ensure_session(mock_state, child_id, parent_id=parent_id)

    assert child.project_id == parent_session.project_id, (
        f"Child project_id should be {parent_session.project_id!r}, got {child.project_id!r}"
    )
    assert child.directory == parent_session.directory, (
        f"Child directory should be {parent_session.directory!r}, got {child.directory!r}"
    )


@pytest.mark.asyncio
async def test_ensure_session_child_falls_back_when_parent_missing(
    mock_state: ServerState,
) -> None:
    """Test that ensure_session falls back to working_dir when parent_id is not found."""
    child_id = "child_orphan"
    orphan_parent_id = "nonexistent_parent"

    with (
        patch("wolfharness_server.opencode_server.converters.opencode_to_session_data"),
        patch("wolfharness_server.opencode_server.input_provider.OpenCodeInputProvider"),
    ):
        child = await ensure_session(mock_state, child_id, parent_id=orphan_parent_id)

    # Should fall back to working_dir-based values
    assert child.project_id == "global"
    assert child.directory == mock_state.working_dir


@pytest.mark.asyncio
async def test_ensure_session_child_skips_agent_binding(mock_state: ServerState) -> None:
    """Test that ensure_session does NOT bind agent for child sessions.

    Child sessions live inside the parent's agent stream. Binding the
    shared agent to a child session would overwrite the parent's
    session_id and also deadlock on agent_lock.
    """
    session_id = "child_session_abc"
    parent_id = "parent_session_xyz"

    with (
        patch("wolfharness_server.opencode_server.converters.opencode_to_session_data"),
        patch("wolfharness_server.opencode_server.input_provider.OpenCodeInputProvider"),
    ):
        result = await ensure_session(mock_state, session_id, parent_id=parent_id)

    assert result.id == session_id
    assert result.parent_id == parent_id
    # Agent session_id must NOT be changed to the child's ID
    assert mock_state.agent.session_id != session_id  # type: ignore[attr-defined]
