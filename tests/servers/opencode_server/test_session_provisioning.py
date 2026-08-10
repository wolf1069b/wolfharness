"""Merged tests for session provisioning in the OpenCode server.

Combines tests from:
- test_ensure_session_durable.py
- test_ensure_session_store_first.py
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wolfharness.agents.base_agent import BaseAgent
from wolfharness.agents.native_agent.checkpoint import CheckpointData
from wolfharness.lifecycle.types import DeliveryMode
from wolfharness.sessions.models import PendingDeferredCall, SessionData
from wolfharness_server.opencode_server.converters import session_data_to_opencode
from wolfharness_server.opencode_server.models import (
    MessageWithParts,
    Session,
    SessionCreatedEvent,
    SessionIdleEvent,
    SessionStatusEvent,
    SessionUpdatedEvent,
    TimeCreatedUpdated,
)
from wolfharness_server.opencode_server.models.parts import (
    ToolPart,
    ToolStateRunning,
)
from wolfharness_server.opencode_server.session_pool_integration import (
    OpenCodeSessionPoolIntegration,
    ensure_session,
)
from wolfharness_server.opencode_server.state import ServerState


pytestmark = pytest.mark.integration


# =============================================================================
# --- Merged from test_ensure_session_durable.py ---
# =============================================================================


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
    agent.host_context.session_pool.send_message = AsyncMock()
    agent.host_context.session_pool.resume_session = AsyncMock()
    agent.host_context.session_pool.close_session = AsyncMock()
    agent.host_context.session_pool.sessions.get_or_create_session_agent = AsyncMock()
    agent.host_context.session_pool.sessions.get_session = MagicMock(return_value=None)
    agent.host_context.session_pool.event_bus = MagicMock()
    from tests._helpers.mock_stream import EmptyReceiveStream

    agent.host_context.session_pool.event_bus.subscribe = AsyncMock(
        return_value=EmptyReceiveStream()
    )
    agent.host_context.session_pool.event_bus.unsubscribe = AsyncMock()
    agent.env = MagicMock()
    agent.env.cwd = "/test/dir"
    return agent


def _make_session_data(
    session_id: str = "stored-session",
    *,
    agent_name: str = "stored_agent",
    agent_type: str = "native",
    pool_id: str = "stored-pool",
    project_id: str = "stored-project",
    cwd: str = "/stored/dir",
    parent_id: str | None = None,
    status: str = "active",
    pending_calls: list[PendingDeferredCall] | None = None,
    metadata: dict[str, Any] | None = None,
) -> SessionData:
    """Create a SessionData instance with configurable fields."""
    meta = metadata or {}
    if "title" not in meta:
        meta["title"] = "Stored Session Title"
    return SessionData(
        session_id=session_id,
        agent_name=agent_name,
        agent_type=agent_type,
        pool_id=pool_id,
        project_id=project_id,
        cwd=cwd,
        parent_id=parent_id,
        version="1",
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
        last_active=datetime(2025, 6, 1, tzinfo=UTC),
        metadata=meta,
        status=status,
        pending_deferred_calls=pending_calls or [],
    )


def _make_pending_call(
    tool_call_id: str = "call_001",
    tool_name: str = "bash",
    deferred_kind: str = "external",
    deferred_strategy: str = "block",
) -> PendingDeferredCall:
    """Create a PendingDeferredCall."""
    return PendingDeferredCall(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        deferred_kind=deferred_kind,  # type: ignore[arg-type]
        deferred_strategy=deferred_strategy,  # type: ignore[arg-type]
    )


def _make_checkpoint_data(
    pending_calls: list[PendingDeferredCall] | None = None,
) -> CheckpointData:
    """Create CheckpointData with optional pending calls."""
    return CheckpointData(
        message_history=[],
        pending_calls=pending_calls or [],
    )


@pytest.fixture
def mock_state() -> ServerState:
    """Create a ServerState with mocked dependencies."""
    agent = create_mock_agent()
    state = ServerState(
        working_dir="/test/working/dir",
        agent=agent,
    )
    # Initialize backward-compat dicts
    state.messages = {}  # type: ignore[attr-defined]
    state.session_status = {}  # type: ignore[attr-defined]
    state.todos = {}  # type: ignore[attr-defined]
    state.input_providers = {}  # type: ignore[attr-defined]
    state.pending_questions = {}  # type: ignore[attr-defined]
    state.reverted_messages = {}  # type: ignore[attr-defined]
    return state


# ---------------------------------------------------------------------------
# Task 27.1: ensure_session detects checkpointed sessions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_session_detects_checkpointed_status(mock_state: ServerState) -> None:
    """ensure_session loads a session with status='checkpointed' successfully.

    The session should still be loaded from store, with runtime state
    initialised, even when status is 'checkpointed'.
    """
    session_id = "checkpointed-session"
    sd = _make_session_data(
        session_id,
        status="checkpointed",
        pending_calls=[_make_pending_call("call_001", "bash")],
    )

    mock_store = MagicMock()
    mock_store.load_session = AsyncMock(return_value=sd)
    mock_store.save_session = AsyncMock()
    mock_state.pool.session_pool.sessions.store = mock_store

    with patch.object(mock_state, "broadcast_event", new=AsyncMock()):
        session = await ensure_session(mock_state, session_id)

    # Session should be loaded and registered in memory
    assert session.id == session_id
    assert session_id in mock_state.sessions
    assert session.title == "Stored Session Title"

    # Runtime state should be initialised
    assert session_id in mock_state.reverted_messages
    assert mock_state.reverted_messages[session_id] == []
    assert session_id in mock_state.messages

    # store.save should NOT be called (data already persisted)
    mock_store.save_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_ensure_session_checkpointed_marks_idle(mock_state: ServerState) -> None:
    """Checkpointed session loaded from store is marked as idle."""
    session_id = "checkpointed-idle"
    sd = _make_session_data(session_id, status="checkpointed")

    mock_store = MagicMock()
    mock_store.load_session = AsyncMock(return_value=sd)
    mock_state.pool.session_pool.sessions.store = mock_store

    with patch.object(mock_state, "broadcast_event", new=AsyncMock()) as mock_broadcast:
        await ensure_session(mock_state, session_id)

    # Should be idle even though status is checkpointed in storage.
    # Status is broadcast via set_session_status() + mark_session_idle().
    status_events = [
        call.args[0]
        for call in mock_broadcast.await_args_list
        if isinstance(call.args[0], SessionStatusEvent)
    ]
    assert len(status_events) >= 1
    assert status_events[0].properties.status.type == "idle"


# ---------------------------------------------------------------------------
# Task 27.2: Running ToolParts re-inserted into in-memory message list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_session_reconstructs_tool_parts(mock_state: ServerState) -> None:
    """ensure_session creates running ToolParts for pending deferred calls.

    When a checkpointed session has pending_deferred_calls, the in-memory
    message list should contain an assistant message with ToolStateRunning
    ToolParts for each pending call.
    """
    session_id = "checkpointed-with-pending"
    sd = _make_session_data(
        session_id,
        status="checkpointed",
        pending_calls=[
            _make_pending_call("call_a", "bash"),
            _make_pending_call("call_b", "subagent"),
        ],
    )

    mock_store = MagicMock()
    mock_store.load_session = AsyncMock(return_value=sd)
    mock_store.save_session = AsyncMock()
    mock_state.pool.session_pool.sessions.store = mock_store

    with patch.object(mock_state, "broadcast_event", new=AsyncMock()):
        await ensure_session(mock_state, session_id)

    # In-memory messages should contain an assistant message with ToolParts
    messages: list[MessageWithParts] = mock_state.messages.get(session_id, [])
    assert len(messages) >= 1, f"Expected at least 1 message, got {len(messages)}"

    # Find the assistant message (should be the last one or the only one)
    assistant_msgs = [m for m in messages if m.info.role == "assistant"]
    assert len(assistant_msgs) == 1, f"Expected 1 assistant message, got {len(assistant_msgs)}"

    assistant_msg = assistant_msgs[0]
    # Find ToolParts
    tool_parts = [p for p in assistant_msg.parts if isinstance(p, ToolPart)]
    assert len(tool_parts) == 2, f"Expected 2 ToolParts, got {len(tool_parts)}"

    for tp in tool_parts:
        assert isinstance(tp, ToolPart)
        assert isinstance(tp.state, ToolStateRunning), (
            f"ToolPart state should be ToolStateRunning, got {type(tp.state).__name__}"
        )
        # Each ToolPart should have a call_id from pending calls
        call_ids = {call.tool_call_id for call in sd.pending_deferred_calls}
        assert tp.call_id in call_ids, (
            f"ToolPart call_id {tp.call_id!r} not in pending calls {call_ids}"
        )


@pytest.mark.asyncio
async def test_ensure_session_no_tool_parts_for_no_pending_calls(
    mock_state: ServerState,
) -> None:
    """ensure_session does NOT create ToolParts when no pending deferred calls."""
    session_id = "checkpointed-no-pending"
    sd = _make_session_data(session_id, status="checkpointed", pending_calls=[])

    mock_store = MagicMock()
    mock_store.load_session = AsyncMock(return_value=sd)
    mock_state.pool.session_pool.sessions.store = mock_store

    with patch.object(mock_state, "broadcast_event", new=AsyncMock()):
        await ensure_session(mock_state, session_id)

    # Still creates an assistant message (placeholder), but no ToolParts
    messages: list[MessageWithParts] = mock_state.messages.get(session_id, [])
    assistant_msgs = [m for m in messages if m.info.role == "assistant"]
    if assistant_msgs:
        tool_parts = [p for p in assistant_msgs[0].parts if isinstance(p, ToolPart)]
        assert len(tool_parts) == 0, (
            f"Expected 0 ToolParts when no pending calls, got {len(tool_parts)}"
        )


# ---------------------------------------------------------------------------
# Task 27.3: Parent/child spawn graph restored from checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_session_restores_child_topology(mock_state: ServerState) -> None:
    """ensure_session restores parent/child spawn graph from checkpoint.

    When a session was checkpointed with spawn_children stored in metadata,
    the spawn topology should be restored on the ServerState.
    """
    parent_id = "parent-checkpointed"
    child_id_1 = "child-1"
    child_id_2 = "child-2"

    # Parent is checkpointed with spawn_children in metadata
    parent_sd = _make_session_data(
        parent_id,
        status="checkpointed",
        pending_calls=[
            _make_pending_call("call_x", "task"),
        ],
        metadata={
            "title": "Stored Session Title",
            "spawn_children": [child_id_1, child_id_2],
        },
    )

    mock_store = MagicMock()
    mock_store.load_session = AsyncMock(return_value=parent_sd)
    mock_store.save_session = AsyncMock()
    mock_state.pool.session_pool.sessions.store = mock_store

    with patch.object(mock_state, "broadcast_event", new=AsyncMock()):
        session = await ensure_session(mock_state, parent_id)

    assert session.id == parent_id

    # The spawn topology should be stored on state for the integration to pick up.
    spawn_graph: dict[str, list[str]] = getattr(mock_state, "checkpoint_spawn_graph", None) or {}
    children = spawn_graph.get(parent_id, [])
    assert child_id_1 in children, f"Child {child_id_1!r} not found in spawn graph"
    assert child_id_2 in children, f"Child {child_id_2!r} not found in spawn graph"


@pytest.mark.asyncio
async def test_ensure_session_no_spawn_graph_for_no_children(
    mock_state: ServerState,
) -> None:
    """ensure_session does not create spawn graph entries for sessions with no children."""
    session_id = "no-children-session"
    sd = _make_session_data(session_id, status="checkpointed")

    mock_store = MagicMock()
    mock_store.load_session = AsyncMock(return_value=sd)
    mock_store.save_session = AsyncMock()
    mock_state.pool.session_pool.sessions.store = mock_store

    with patch.object(mock_state, "broadcast_event", new=AsyncMock()):
        await ensure_session(mock_state, session_id)

    spawn_graph: dict[str, list[str]] = getattr(mock_state, "checkpoint_spawn_graph", {})
    assert session_id in spawn_graph, "Spawn graph entry should exist (empty children list)"
    assert spawn_graph[session_id] == []


# ---------------------------------------------------------------------------
# Task 27.4: route_message() replays deferred results before new input
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_message_replays_deferred_results(mock_state: ServerState) -> None:
    """route_message calls resume_session when session is checkpointed with results."""
    session_id = "checkpointed-resume"
    sd = _make_session_data(
        session_id,
        status="checkpointed",
        pending_calls=[_make_pending_call("call_r", "bash")],
    )

    mock_store = MagicMock()
    mock_store.load_session = AsyncMock(return_value=sd)
    mock_store.save_session = AsyncMock()
    mock_state.pool.session_pool.sessions.store = mock_store

    # Wire up SessionPool
    sp = mock_state.pool.session_pool
    sp.sessions.store = mock_store
    sp.sessions.get_session = MagicMock(return_value=MagicMock(current_run_id=None))
    sp.sessions.get_or_create_session = AsyncMock(return_value=(MagicMock(), True))

    # Mock deferred tool results
    deferred_results = MagicMock()
    deferred_results.calls = {"call_r": MagicMock()}

    integration = OpenCodeSessionPoolIntegration(sp, mock_state)

    # Patch internal methods that spawn background tasks
    with (
        patch.object(integration, "_start_event_consumer", new=AsyncMock()),
        patch.object(mock_state, "broadcast_event", new=AsyncMock()),
    ):
        await integration.route_message(
            session_id,
            content="resume with results",
            mode=DeliveryMode.QUEUE,
            deferred_tool_results=deferred_results,
        )

    # resume_session should have been called because session is checkpointed
    # and deferred_tool_results were provided
    sp.resume_session.assert_awaited_once()

    # receive_request should still be called after resume
    sp.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_route_message_skips_resume_when_not_checkpointed(
    mock_state: ServerState,
) -> None:
    """route_message does NOT call resume_session when session is not checkpointed."""
    session_id = "active-session"

    sp = mock_state.pool.session_pool
    sp.sessions.store = None  # No store → no data → normal path
    sp.sessions.get_session = MagicMock(return_value=MagicMock(current_run_id=None))
    sp.sessions.get_or_create_session = AsyncMock(return_value=(MagicMock(), True))

    integration = OpenCodeSessionPoolIntegration(sp, mock_state)

    with (
        patch.object(integration, "_start_event_consumer", new=AsyncMock()),
        patch.object(mock_state, "broadcast_event", new=AsyncMock()),
    ):
        await integration.route_message(
            session_id,
            content="hello",
            mode=DeliveryMode.QUEUE,
        )

    # resume_session should NOT be called for non-checkpointed sessions
    sp.resume_session.assert_not_awaited()
    # receive_request should be called
    sp.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_session_checkpointed_restores_input_provider(
    mock_state: ServerState,
) -> None:
    """Checkpointed session restores input provider correctly."""
    session_id = "checkpointed-input-provider"
    sd = _make_session_data(session_id, status="checkpointed")

    mock_store = MagicMock()
    mock_store.load_session = AsyncMock(return_value=sd)
    mock_state.pool.session_pool.sessions.store = mock_store

    with (
        patch.object(mock_state, "broadcast_event", new=AsyncMock()),
        patch(
            "wolfharness_server.opencode_server.input_provider.OpenCodeInputProvider"
        ) as mock_prov_cls,
    ):
        mock_prov = MagicMock()
        mock_prov_cls.return_value = mock_prov
        await ensure_session(mock_state, session_id)

    mock_prov_cls.assert_called()  # Called at least once (idempotent ensure)
    # Input provider is created (the mock tracks calls to OpenCodeInputProvider)
    assert mock_prov_cls.call_count >= 1


# ---------------------------------------------------------------------------
# Edge case: checkpointed session without store
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_session_checkpointed_without_store(
    mock_state: ServerState,
) -> None:
    """ensure_session handles checkpointed session when store is None (falls back)."""
    session_id = "checkpointed-no-store"
    sd = _make_session_data(session_id, status="checkpointed")

    # Set store to None (simulates no checkpoint storage)
    mock_state.pool.session_pool.sessions.store = None
    # But storage.load_session still returns the data (via pool.storage)
    mock_state.pool.storage.load_session = AsyncMock(return_value=sd)

    with (
        patch.object(mock_state, "broadcast_event", new=AsyncMock()),
        patch(
            "wolfharness_server.opencode_server.input_provider.OpenCodeInputProvider"
        ) as mock_prov_cls,
    ):
        mock_prov_cls.return_value = MagicMock()
        session = await ensure_session(mock_state, session_id)

    assert session.id == session_id
    assert session_id in mock_state.sessions


# =============================================================================
# --- Merged from test_ensure_session_store_first.py ---
# =============================================================================


def create_mock_agent_store_first() -> MagicMock:
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
    agent.host_context.session_pool.create_session = AsyncMock()
    agent.host_context.session_pool.sessions = MagicMock()
    agent.host_context.session_pool.sessions.store = None
    agent.env = MagicMock()
    agent.env.cwd = "/test/dir"
    return agent


def _make_session_data_store_first(
    session_id: str = "stored-session",
    *,
    agent_name: str = "stored_agent",
    agent_type: str = "acp",
    pool_id: str = "stored-pool",
    project_id: str = "stored-project",
    cwd: str = "/stored/dir",
    parent_id: str | None = None,
) -> SessionData:
    """Create a SessionData instance that simulates persisted data."""
    return SessionData(
        session_id=session_id,
        agent_name=agent_name,
        agent_type=agent_type,
        pool_id=pool_id,
        project_id=project_id,
        cwd=cwd,
        parent_id=parent_id,
        version="1",
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
        last_active=datetime(2025, 6, 1, tzinfo=UTC),
        metadata={"title": "Stored Session Title"},
    )


@pytest.fixture
def mock_state_store_first() -> ServerState:
    """Create a ServerState with mocked dependencies."""
    agent = create_mock_agent_store_first()
    state = ServerState(
        working_dir="/test/working/dir",
        agent=agent,
    )
    # Initialize backward-compat dicts removed from ServerState dataclass
    # so tests and helper fallbacks can access them.
    state.messages = {}
    state.session_status = {}
    state.todos = {}
    state.input_providers = {}
    state.pending_questions = {}
    return state


# ---------------------------------------------------------------------------
# TG-2: ensure_session preserves already-persisted agent_type/pool_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_first_preserves_agent_type_and_pool_id(
    mock_state_store_first: ServerState,
) -> None:
    """TG-2: Store-first path preserves persisted agent_type and pool_id.

    When a session was previously persisted with agent_type='acp' and
    pool_id='stored-pool', ensure_session must NOT overwrite those values
    with defaults from the current agent config.
    """
    session_id = "stored-session"
    sd = _make_session_data_store_first(session_id, agent_type="acp", pool_id="stored-pool")

    # Wire store.load to return the persisted data
    mock_store = MagicMock()
    mock_store.load_session = AsyncMock(return_value=sd)
    mock_state_store_first.pool.session_pool.sessions.store = mock_store

    # Also mock save so we can verify it's NOT called
    mock_store.save_session = AsyncMock()

    with patch.object(mock_state_store_first, "broadcast_event", new=AsyncMock()):
        session = await ensure_session(mock_state_store_first, session_id)

    # The session should have the stored title/directory
    assert session.title == "Stored Session Title"
    assert session.directory == "/stored/dir"
    assert session.project_id == "stored-project"

    # store.save must NOT be called — data is already persisted
    mock_store.save_session.assert_not_awaited()

    # Also verify pool.storage.save_session was NOT called
    mock_state_store_first.pool.storage.save_session.assert_not_awaited()


# ---------------------------------------------------------------------------
# TG-5: store-first child session is not overwritten by fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_first_child_not_overwritten(mock_state_store_first: ServerState) -> None:
    """TG-5: A child session loaded from store is not overwritten by fallback.

    If a child session (parent_id set) was persisted by
    create_child_session() and then ensure_session() is called for that
    session_id, the store-first path must restore it as-is rather than
    creating a fresh session with default values.
    """
    child_id = "child-session-stored"
    parent_id = "parent-session-stored"
    sd = _make_session_data_store_first(
        child_id,
        agent_name="child_agent",
        agent_type="native",
        pool_id="child-pool",
        parent_id=parent_id,
        cwd="/child/dir",
        project_id="child-project",
    )

    mock_store = MagicMock()
    mock_store.load_session = AsyncMock(return_value=sd)
    mock_store.save_session = AsyncMock()
    mock_state_store_first.pool.session_pool.sessions.store = mock_store

    with patch.object(mock_state_store_first, "broadcast_event", new=AsyncMock()):
        session = await ensure_session(mock_state_store_first, child_id, parent_id=parent_id)

    # Child fields from store must be preserved
    assert session.id == child_id
    assert session.parent_id == parent_id
    assert session.directory == "/child/dir"
    assert session.project_id == "child-project"

    # Must NOT have overwritten by creating a fresh "New Session"
    assert session.title != "New Session"

    # Must NOT have called save
    mock_store.save_session.assert_not_awaited()


# ---------------------------------------------------------------------------
# TG-11: concurrent calls produce one Session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_calls_produce_one_session(
    mock_state_store_first: ServerState,
) -> None:
    """TG-11: Concurrent ensure_session calls for same ID produce one Session.

    Multiple coroutines calling ensure_session with the same session_id
    concurrently must result in exactly one in-memory Session object.
    """
    session_id = "concurrent-session"

    # Use the conftest-style real store (pool.sessions.store is already
    # wired to storage_manager via the mock_pool fixture in conftest.py,
    # but mock_state_store_first uses a simpler mock).  Set store to None so the
    # store-first path yields None and falls through to creation.
    mock_state_store_first.pool.session_pool.sessions.store = None
    mock_state_store_first.pool.storage.load_session = AsyncMock(return_value=None)

    with (
        patch("wolfharness_server.opencode_server.converters.opencode_to_session_data"),
        patch("wolfharness_server.opencode_server.input_provider.OpenCodeInputProvider"),
        patch.object(mock_state_store_first, "broadcast_event", new=AsyncMock()),
    ):
        results = await asyncio.gather(
            ensure_session(mock_state_store_first, session_id),
            ensure_session(mock_state_store_first, session_id),
            ensure_session(mock_state_store_first, session_id),
        )

    # All calls should return the same Session object
    assert results[0] is results[1]
    assert results[1] is results[2]

    # Only one Session should exist in memory
    assert len([s for s in mock_state_store_first.sessions.values() if s.id == session_id]) == 1


@pytest.mark.asyncio
async def test_concurrent_store_first_produces_one_session(
    mock_state_store_first: ServerState,
) -> None:
    """TG-11 (store variant): Concurrent calls when data is in store.

    When multiple coroutines race to ensure_session for an ID that exists
    in the store, only one should trigger the store.load + conversion and
    the others should find it in memory via double-check locking.
    """
    session_id = "concurrent-store-session"
    sd = _make_session_data_store_first(session_id)

    mock_store = MagicMock()
    mock_store.load_session = AsyncMock(return_value=sd)
    mock_store.save_session = AsyncMock()
    mock_state_store_first.pool.session_pool.sessions.store = mock_store

    with patch.object(mock_state_store_first, "broadcast_event", new=AsyncMock()):
        results = await asyncio.gather(
            ensure_session(mock_state_store_first, session_id),
            ensure_session(mock_state_store_first, session_id),
        )

    # Both should return the same Session object
    assert results[0] is results[1]
    assert mock_state_store_first.sessions[session_id] is results[0]


# ---------------------------------------------------------------------------
# TG-17: In-memory session not overwritten by store data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_in_memory_session_not_overwritten_by_store(
    mock_state_store_first: ServerState,
) -> None:
    """TG-17: An in-memory session is not overwritten by store data.

    If a session is already in memory (from a prior ensure_session or
    create_session call), ensure_session must return the in-memory version
    even if the store has different data for that session_id.
    """
    session_id = "in-mem-session"

    existing_session = Session(
        id=session_id,
        project_id="in-mem-project",
        directory="/in-mem/dir",
        title="In-Memory Title",
        version="1",
        time=TimeCreatedUpdated(created=1000, updated=2000),
        parent_id=None,
    )
    mock_state_store_first.sessions[session_id] = existing_session

    # Store has different data
    sd = _make_session_data_store_first(session_id, cwd="/store/dir", project_id="store-project")
    mock_store = MagicMock()
    mock_store.load_session = AsyncMock(return_value=sd)
    mock_state_store_first.pool.session_pool.sessions.store = mock_store

    with patch.object(mock_state_store_first, "broadcast_event", new=AsyncMock()) as mock_broadcast:
        result = await ensure_session(mock_state_store_first, session_id)

    # Must return the in-memory session, not the store version
    assert result is existing_session
    assert result.title == "In-Memory Title"
    assert result.project_id == "in-mem-project"

    # Store.load should NOT have been called
    mock_store.load_session.assert_not_awaited()

    # Only SessionUpdatedEvent should be broadcast (not Created)
    created_events = [
        e for e in mock_broadcast.await_args_list if isinstance(e.args[0], SessionCreatedEvent)
    ]
    assert len(created_events) == 0


# ---------------------------------------------------------------------------
# TG-19: Store-first path does NOT call bind_agent_to_session for children
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_first_child_skips_agent_binding(
    mock_state_store_first: ServerState,
) -> None:
    """TG-19: Store-first child session does NOT call bind_agent_to_session.

    When a child session (parent_id is set) is loaded from the store,
    ensure_session must NOT call bind_agent_to_session — that would
    overwrite the parent's session_id and deadlock on agent_lock.
    """
    child_id = "child-no-bind"
    parent_id = "parent-no-bind"
    sd = _make_session_data_store_first(child_id, parent_id=parent_id)

    mock_store = MagicMock()
    mock_store.load_session = AsyncMock(return_value=sd)
    mock_state_store_first.pool.session_pool.sessions.store = mock_store

    original_session_id = mock_state_store_first.agent.session_id

    with patch.object(mock_state_store_first, "broadcast_event", new=AsyncMock()):
        session = await ensure_session(mock_state_store_first, child_id)

    assert session.id == child_id

    # Agent's session_id must NOT have been changed to the child's ID
    assert mock_state_store_first.agent.session_id == original_session_id
    assert mock_state_store_first.agent.session_id != child_id


# ---------------------------------------------------------------------------
# TG-32: Store-miss fallback still creates and persists new session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_miss_fallback_creates_and_persists(
    mock_state_store_first: ServerState,
) -> None:
    """TG-32: Store-miss fallback creates and persists a new session.

    When a session_id is absent from both memory and the store,
    ensure_session must fall back to creating a new session and
    persisting it (original behaviour).
    """
    session_id = "new-session-fallback"

    # Store returns None (session not found)
    mock_store = MagicMock()
    mock_store.load_session = AsyncMock(return_value=None)
    mock_store.save_session = AsyncMock()
    mock_state_store_first.pool.session_pool.sessions.store = mock_store

    with (
        patch(
            "wolfharness_server.opencode_server.converters.opencode_to_session_data"
        ) as mock_conv,
        patch(
            "wolfharness_server.opencode_server.input_provider.OpenCodeInputProvider"
        ) as mock_prov_cls,
        patch.object(mock_state_store_first, "broadcast_event", new=AsyncMock()),
    ):
        mock_conv.return_value = MagicMock()
        mock_prov_cls.return_value = MagicMock()
        result = await ensure_session(mock_state_store_first, session_id)

    assert result.id == session_id
    assert result.title == "New Session"

    # Must have persisted via SessionPool.create_session (unified creation path)
    mock_state_store_first.pool.session_pool.create_session.assert_awaited_once()

    # Must be in memory
    assert session_id in mock_state_store_first.sessions
    assert mock_state_store_first.sessions[session_id] is result


# ---------------------------------------------------------------------------
# Additional: Store-first path broadcasts created + updated events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_first_broadcasts_created_and_updated(
    mock_state_store_first: ServerState,
) -> None:
    """Store-first path broadcasts session.created and session.updated events."""
    session_id = "broadcast-test-session"
    sd = _make_session_data_store_first(session_id)

    mock_store = MagicMock()
    mock_store.load_session = AsyncMock(return_value=sd)
    mock_state_store_first.pool.session_pool.sessions.store = mock_store

    with patch.object(mock_state_store_first, "broadcast_event", new=AsyncMock()) as mock_broadcast:
        await ensure_session(mock_state_store_first, session_id)

    broadcast_events = [call.args[0] for call in mock_broadcast.await_args_list]

    created_events = [e for e in broadcast_events if isinstance(e, SessionCreatedEvent)]
    updated_events = [e for e in broadcast_events if isinstance(e, SessionUpdatedEvent)]

    assert len(created_events) == 1
    assert len(updated_events) >= 1  # at least the session.updated

    # The session from created event should match what we loaded
    assert created_events[0].properties.info.id == session_id


# ---------------------------------------------------------------------------
# Additional: Store-first path marks session idle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_first_marks_session_idle(mock_state_store_first: ServerState) -> None:
    """Store-first path marks the session as idle."""
    session_id = "idle-test-session"
    sd = _make_session_data_store_first(session_id)

    mock_store = MagicMock()
    mock_store.load_session = AsyncMock(return_value=sd)
    mock_state_store_first.pool.session_pool.sessions.store = mock_store

    with patch.object(mock_state_store_first, "broadcast_event", new=AsyncMock()) as mock_broadcast:
        await ensure_session(mock_state_store_first, session_id)

    # Should have idle events.
    # Status is now broadcast via set_session_status() + mark_session_idle()
    # instead of stored in the in-memory session_status dict.
    broadcast_events = [call.args[0] for call in mock_broadcast.await_args_list]
    status_events = [e for e in broadcast_events if isinstance(e, SessionStatusEvent)]
    idle_events = [e for e in broadcast_events if isinstance(e, SessionIdleEvent)]
    assert len(status_events) == 2  # set_session_status() + mark_session_idle() explicit broadcast
    assert len(idle_events) == 1


# ---------------------------------------------------------------------------
# Additional: session_data_to_opencode converter
# ---------------------------------------------------------------------------


def test_session_from_session_data_uses_converter() -> None:
    """session_data_to_opencode converts SessionData to OpenCode Session correctly."""
    sd = _make_session_data_store_first("converter-test")

    result = session_data_to_opencode(sd)

    assert result.id == "converter-test"
    assert result.title == "Stored Session Title"
    assert result.directory == "/stored/dir"
    assert result.project_id == "stored-project"


# ---------------------------------------------------------------------------
# Additional: Store-first creates runtime state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_first_creates_runtime_state(mock_state_store_first: ServerState) -> None:
    """Store-first path initializes runtime state for the session."""
    session_id = "runtime-state-session"
    sd = _make_session_data_store_first(session_id)

    mock_store = MagicMock()
    mock_store.load_session = AsyncMock(return_value=sd)
    mock_state_store_first.pool.session_pool.sessions.store = mock_store

    with patch.object(mock_state_store_first, "broadcast_event", new=AsyncMock()):
        await ensure_session(mock_state_store_first, session_id)

    # Session should be registered in memory
    assert session_id in mock_state_store_first.sessions
    # ensure_runtime_session_state initializes reverted_messages
    assert session_id in mock_state_store_first.reverted_messages
    assert mock_state_store_first.reverted_messages[session_id] == []


# ---------------------------------------------------------------------------
# Additional: Store-first top-level session does not bind agent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_first_top_level_session_does_not_bind_agent(
    mock_state_store_first: ServerState,
) -> None:
    """Store-first path does not bind agent for top-level sessions.

    Agent binding was removed from ensure_session; sessions are now
    managed by the SessionPool orchestration layer.
    """
    session_id = "top-level-session"
    sd = _make_session_data_store_first(session_id, parent_id=None)

    mock_store = MagicMock()
    mock_store.load_session = AsyncMock(return_value=sd)
    mock_state_store_first.pool.session_pool.sessions.store = mock_store

    original_session_id = mock_state_store_first.agent.session_id

    with patch.object(mock_state_store_first, "broadcast_event", new=AsyncMock()):
        await ensure_session(mock_state_store_first, session_id)

    # Session should be created
    assert session_id in mock_state_store_first.sessions
    # Agent should NOT be bound to this session
    assert mock_state_store_first.agent.session_id == original_session_id
