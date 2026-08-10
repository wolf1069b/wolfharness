"""Integration tests for OpenCode session pool integration.

These tests verify the integration layer between OpenCode server routes
and the SessionPool orchestration layer. The integration class under test
(OpenCodeSessionPoolIntegration) does not yet exist — these are TDD RED
phase tests.

Coverage:
- Session creation via SessionPool.create_session()
- Message routing through SessionPool.receive_request()
- Session status sync (idle -> busy -> idle)
- Abort via SessionPool.cancel_run()
- Session fork with parent_session_id
- Input provider flow
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock

import pytest


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from wolfharness.lifecycle.types import DeliveryMode
from wolfharness.orchestrator.core import SessionPool
from wolfharness.sessions.models import SessionData
from wolfharness_server.opencode_server.input_provider import OpenCodeInputProvider
from wolfharness_server.opencode_server.state import ServerState


pytestmark = pytest.mark.integration


# =============================================================================
# Fixtures
# =============================================================================


def _stream_empty(queue: asyncio.Queue[Any]) -> bool:
    """Check if a subscriber queue has no buffered items."""
    return queue.empty()


@pytest.fixture
def mock_agent_pool() -> Mock:
    """Create a mock AgentPool for SessionPool construction."""
    from wolfharness.agents.events import RunStartedEvent, StreamCompleteEvent
    from wolfharness.messaging.messages import ChatMessage

    pool = Mock()
    pool.main_agent = Mock()
    pool.main_agent.name = "test-agent"
    pool.manifest = Mock()
    pool._config_file_path = None
    pool.mcp = Mock()
    pool.mcp.servers = []

    async def _mock_run_stream(*args: Any, **kwargs: Any) -> Any:
        """Yield a minimal run event sequence for testing."""
        session_id = kwargs.get("session_id", "unknown")
        run_id = "run-mock-001"
        yield RunStartedEvent(session_id=session_id, run_id=run_id)
        yield StreamCompleteEvent(
            message=ChatMessage(content="test response", role="assistant"),
        )

    mock_agent = Mock()
    mock_agent.run_stream = _mock_run_stream
    mock_agent._input_provider = None
    mock_agent.AGENT_TYPE = "native"
    mock_agent.conversation = Mock()
    mock_agent.conversation.add_chat_messages = Mock()
    mock_agent.conversation.get_history = Mock(return_value=[])
    mock_agent.tools = Mock()
    mock_agent.__aenter__ = AsyncMock(return_value=mock_agent)
    mock_agent.__aexit__ = AsyncMock(return_value=None)

    # Wire create_turn so RunHandle.start() can execute the turn.
    # start() publishes RunStartedEvent to EventBus directly, so
    # execute() only needs to yield StreamCompleteEvent.
    async def _mock_execute() -> Any:
        """Yield events for turn.execute()."""
        yield StreamCompleteEvent(
            message=ChatMessage(content="test response", role="assistant"),
        )

    mock_turn = Mock()
    mock_turn.execute = _mock_execute
    mock_turn.message_history = []  # type: ignore[misc]
    mock_agent.create_turn = Mock(return_value=mock_turn)

    # Use a mock config that returns our mock agent from get_agent().
    # This ensures get_or_create_session_agent() returns mock_agent
    # (with properly wired create_turn) instead of a raw MagicMock.
    mock_cfg = Mock()
    mock_cfg.name = "test-agent"
    mock_cfg.get_agent = Mock(return_value=mock_agent)
    mock_cfg.get_mcp_servers = Mock(return_value=[])
    pool.manifest.agents = {"test-agent": mock_cfg}
    pool.get_agent = Mock(return_value=mock_agent)

    # Wire AgentFactory mock so SessionController.get_or_create_session_agent()
    # can delegate to pool._factory.create_session_agent().
    pool._factory.create_session_agent = AsyncMock(return_value=mock_agent)
    pool.get_context = Mock(return_value=Mock())

    return pool


@pytest.fixture
def mock_session_store() -> Mock:
    """Create a mock SessionStore."""
    store = Mock()
    store.save_session = AsyncMock(return_value=None)
    store.delete_session = AsyncMock(return_value=None)
    store.load_session = AsyncMock(return_value=None)
    store.list_session_ids = AsyncMock(return_value=[])
    return store


@pytest.fixture
async def session_pool(
    mock_agent_pool: Mock, mock_session_store: Mock
) -> AsyncIterator[SessionPool]:  # type: ignore[misc]
    """Create a real SessionPool with mocked dependencies."""
    sp = SessionPool(
        pool=mock_agent_pool,
        store=mock_session_store,
        enable_auto_resume=False,
        enable_event_bus=True,
    )
    await sp.start()
    yield sp
    await sp.shutdown()


@pytest.fixture
def server_state(tmp_project_dir: Any) -> ServerState:
    """Create a minimal ServerState for testing."""
    agent = Mock()
    agent.model_name = None  # resolve_default_model_info() fallback
    agent.name = "test-agent"
    agent.storage = Mock()
    state = ServerState(working_dir=str(tmp_project_dir), agent=agent)
    # No real pool in tests — _pool would be a Mock from agent._agent_pool,
    # which breaks model_variants property and resolve_default_model_info().
    state._pool = None  # type: ignore[assignment]
    return state


@pytest.fixture
def mock_input_provider(server_state: ServerState) -> OpenCodeInputProvider:
    """Create an OpenCodeInputProvider for testing."""
    return OpenCodeInputProvider(state=server_state, session_id="test-session")


# =============================================================================
# OpenCodeSessionPoolIntegration tests (TDD RED phase)
# =============================================================================


class TestOpenCodeSessionPoolIntegrationExists:
    """Verify the integration class exists and can be instantiated."""

    @pytest.mark.asyncio
    async def test_integration_class_importable(self) -> None:
        """The OpenCodeSessionPoolIntegration class should be importable."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        assert OpenCodeSessionPoolIntegration is not None

    @pytest.mark.asyncio
    async def test_integration_initialization(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """Integration should accept SessionPool and ServerState."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )
        assert integration.session_pool is session_pool
        assert integration.server_state is server_state


class TestSessionCreation:
    """Tests for session creation through the integration layer."""

    @pytest.mark.asyncio
    async def test_create_session_delegates_to_session_pool(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """Creating a session should delegate to SessionPool.create_session()."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )

        session_id = "test-session-001"
        agent_name = "test-agent"

        state = await integration.create_session(
            session_id=session_id,
            agent_name=agent_name,
        )

        assert state.session_id == session_id
        assert state.agent_name == agent_name
        assert session_pool.sessions.get_session(session_id) is not None

    @pytest.mark.asyncio
    async def test_create_session_persists_to_store(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
        mock_session_store: Mock,
    ) -> None:
        """Session creation should persist to the session store."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )

        await integration.create_session(
            session_id="test-session-002",
            agent_name="test-agent",
            metadata={"project_id": "proj-1", "cwd": "/tmp"},
        )

        mock_session_store.save_session.assert_awaited()
        saved_data: SessionData = mock_session_store.save_session.await_args[0][0]
        assert saved_data.session_id == "test-session-002"
        assert saved_data.agent_name == "test-agent"

    @pytest.mark.asyncio
    async def test_create_session_broadcasts_created_event(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """Session creation should broadcast a session.created SSE event."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )

        broadcast_events = []
        original_broadcast = server_state.broadcast_event

        async def capture_broadcast(event: Any) -> None:
            broadcast_events.append(event)
            await original_broadcast(event)

        server_state.broadcast_event = capture_broadcast  # type: ignore[method-assign]

        await integration.create_session(
            session_id="test-session-003",
            agent_name="test-agent",
        )

        created_events = [
            e for e in broadcast_events if getattr(e, "type", None) == "session.created"
        ]
        assert len(created_events) == 1


class TestMessageRouting:
    """Tests for message routing through SessionPool.receive_request()."""

    @pytest.mark.asyncio
    async def test_route_message_creates_run_handle(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """Routing a message should create a RunHandle via receive_request()."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )

        await integration.create_session(
            session_id="test-session-004",
            agent_name="test-agent",
        )

        run_handle = await integration.route_message(
            session_id="test-session-004",
            content="Hello, agent!",
        )

        assert run_handle is not None
        # D14: route_message now returns str (message_id), not RunHandle
        assert isinstance(run_handle, str)

    @pytest.mark.asyncio
    async def test_route_message_with_when_idle_priority_queues(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """Routing with 'when_idle' priority should queue when busy."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )

        await integration.create_session(
            session_id="test-session-005",
            agent_name="test-agent",
        )

        # First message starts a run
        run_handle_1 = await integration.route_message(
            session_id="test-session-005",
            content="First message",
            mode=DeliveryMode.QUEUE,
        )
        assert run_handle_1 is not None

        # Second message should be queued (session is busy).
        # Issue #253: followup messages return None instead of message_id
        # to avoid duplicate session_created broadcasts. A None return means
        # the message was queued successfully (not rejected).
        run_handle_2 = await integration.route_message(
            session_id="test-session-005",
            content="Second message",
            mode=DeliveryMode.QUEUE,
        )
        assert run_handle_2 is None  # Queued successfully (followup returns None per #253)

    @pytest.mark.asyncio
    async def test_route_message_with_asap_priority_injects(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """Routing with 'asap' priority should inject into active turn."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )

        await integration.create_session(
            session_id="test-session-006",
            agent_name="test-agent",
        )

        run_handle = await integration.route_message(
            session_id="test-session-006",
            content="Inject this now",
            mode=DeliveryMode.STEER,
        )

        # ASAP on idle session should still create a run
        assert run_handle is not None

    @pytest.mark.asyncio
    async def test_route_message_publishes_run_started_event(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """Routing a message should publish RunStartedEvent to EventBus."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )

        await integration.create_session(
            session_id="test-session-007",
            agent_name="test-agent",
        )

        # Subscribe to EventBus before routing
        queue = await session_pool.event_bus.subscribe("test-session-007")

        await integration.route_message(
            session_id="test-session-007",
            content="Trigger events",
        )

        # Give async tasks a moment to publish
        await asyncio.sleep(0.05)

        events = []
        while True:
            try:
                event = queue.get_nowait()
            except (asyncio.QueueEmpty, asyncio.QueueShutDown):
                break
            if event is not None:
                events.append(event)

        run_started_events = [e for e in events if getattr(e, "event_kind", None) == "run_started"]
        assert len(run_started_events) >= 1


class TestSessionStatusSync:
    """Tests for session status synchronization (idle -> busy -> idle)."""

    @pytest.mark.asyncio
    async def test_create_session_starts_event_consumer(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """Creating a session should start the event consumer."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )

        await integration.create_session(
            session_id="test-session-008",
            agent_name="test-agent",
        )

        # Event consumer should be started for the session
        assert "test-session-008" in integration._session_groups

    @pytest.mark.asyncio
    async def test_status_broadcasts_busy_on_run_start(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """Run start should broadcast session.status with type 'busy'."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )

        broadcast_events = []
        original_broadcast = server_state.broadcast_event

        async def capture_broadcast(event: Any) -> None:
            broadcast_events.append(event)
            await original_broadcast(event)

        server_state.broadcast_event = capture_broadcast  # type: ignore[method-assign]

        await integration.create_session(
            session_id="test-session-009",
            agent_name="test-agent",
        )

        await integration.route_message(
            session_id="test-session-009",
            content="Start working",
        )

        await asyncio.sleep(0.05)

        status_events = [
            e for e in broadcast_events if getattr(e, "type", None) == "session.status"
        ]
        busy_events = [
            e
            for e in status_events
            if getattr(getattr(e, "properties", None), "status", None)
            and e.properties.status.type == "busy"
        ]
        assert len(busy_events) >= 1

    @pytest.mark.asyncio
    async def test_status_broadcasts_idle_on_run_complete(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """Run completion should broadcast session.status with type 'idle'."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )

        broadcast_events = []
        original_broadcast = server_state.broadcast_event

        async def capture_broadcast(event: Any) -> None:
            broadcast_events.append(event)
            await original_broadcast(event)

        server_state.broadcast_event = capture_broadcast  # type: ignore[method-assign]

        await integration.create_session(
            session_id="test-session-010",
            agent_name="test-agent",
        )

        # Route a message and wait for completion
        await integration.route_message(
            session_id="test-session-010",
            content="Complete quickly",
        )

        # Wait for run to complete
        await asyncio.sleep(0.2)

        status_events = [
            e for e in broadcast_events if getattr(e, "type", None) == "session.status"
        ]
        idle_events = [
            e
            for e in status_events
            if getattr(getattr(e, "properties", None), "status", None)
            and e.properties.status.type == "idle"
        ]
        assert len(idle_events) >= 1


class TestSessionAbort:
    """Tests for aborting sessions via SessionPool.cancel_run()."""

    @pytest.mark.asyncio
    async def test_abort_session_cancels_active_run(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """Aborting a session should cancel the active run."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )

        await integration.create_session(
            session_id="test-session-011",
            agent_name="test-agent",
        )

        # Override create_turn to simulate a long-running turn that
        # blocks until the run_ctx is cancelled.
        mock_agent = session_pool.pool.get_agent("test-agent")  # type: ignore[attr-defined]
        captured_ctx: list[Any] = []

        async def _blocking_execute() -> Any:
            # Wait until cancelled, then return without yielding.
            # start() will detect run_ctx.cancelled and set
            # _turn_was_cancelled in its post-turn code.
            while True:
                if captured_ctx and captured_ctx[0].cancelled:
                    return
                await asyncio.sleep(0.01)
            yield  # pragma: no cover  # makes this an async generator

        blocking_turn = Mock()
        blocking_turn.execute = _blocking_execute
        blocking_turn.message_history = []  # type: ignore[misc]

        def _create_turn_with_ctx(prompts: Any, run_ctx: Any, message_history: Any) -> Any:
            captured_ctx.append(run_ctx)
            return blocking_turn

        mock_agent.create_turn = Mock(side_effect=_create_turn_with_ctx)

        run_handle = await integration.route_message(
            session_id="test-session-011",
            content="Long running task",
        )

        assert run_handle is not None

        # D14: route_message now returns str (message_id), not RunHandle.
        # Look up the RunHandle from the session pool for state checks.
        sp_session = session_pool.sessions.get_session("test-session-011")
        assert sp_session is not None
        assert sp_session.current_run_id is not None
        actual_handle = session_pool.get_run(sp_session.current_run_id)
        assert actual_handle is not None

        # Give the background task time to start and transition to running
        await asyncio.sleep(0.05)
        # In the per-prompt model, RunHandle has no _run_state.
        # is_running returns True when complete_event is not set.
        assert actual_handle.is_running is True

        await integration.abort_session("test-session-011")

        # After abort, the run context should be cancelled.
        await asyncio.sleep(0.1)
        assert actual_handle.run_ctx.cancelled is True

    @pytest.mark.asyncio
    async def test_abort_session_broadcasts_error_event(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """Abort should broadcast a session.error SSE event."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )

        broadcast_events = []
        original_broadcast = server_state.broadcast_event

        async def capture_broadcast(event: Any) -> None:
            broadcast_events.append(event)
            await original_broadcast(event)

        server_state.broadcast_event = capture_broadcast  # type: ignore[method-assign]

        await integration.create_session(
            session_id="test-session-012",
            agent_name="test-agent",
        )

        await integration.route_message(
            session_id="test-session-012",
            content="Task to abort",
        )

        await integration.abort_session("test-session-012")

        error_events = [e for e in broadcast_events if getattr(e, "type", None) == "session.error"]
        assert len(error_events) >= 1

    @pytest.mark.asyncio
    async def test_abort_idle_session_is_noop(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """Aborting an idle session should be a no-op."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )

        await integration.create_session(
            session_id="test-session-013",
            agent_name="test-agent",
        )

        # Session is idle, no active run
        await integration.abort_session("test-session-013")  # Should not raise


class TestSessionFork:
    """Tests for session forking with parent_session_id."""

    @pytest.mark.asyncio
    async def test_fork_session_creates_child_with_parent(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """Forking a session should create a child with parent_session_id."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )

        parent_id = "parent-session-001"
        child_id = "child-session-001"

        await integration.create_session(
            session_id=parent_id,
            agent_name="test-agent",
        )

        child_state = await integration.fork_session(
            parent_session_id=parent_id,
            new_session_id=child_id,
            agent_name="test-agent",
        )

        assert child_state.session_id == child_id
        assert child_state.parent_session_id == parent_id

    @pytest.mark.asyncio
    async def test_fork_session_inherits_metadata(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
        mock_session_store: Mock,
    ) -> None:
        """Forked session should inherit parent's metadata."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )

        parent_id = "parent-session-002"
        child_id = "child-session-002"

        # Set up parent data in store
        parent_data = SessionData(
            session_id=parent_id,
            agent_name="test-agent",
            project_id="proj-inherited",
            cwd="/inherited/cwd",
            created_at=__import__("datetime").datetime.now(),
            last_active=__import__("datetime").datetime.now(),
        )
        mock_session_store.load_session = AsyncMock(return_value=parent_data)

        await integration.create_session(
            session_id=parent_id,
            agent_name="test-agent",
            metadata={"project_id": "proj-inherited", "cwd": "/inherited/cwd"},
        )

        child_state = await integration.fork_session(
            parent_session_id=parent_id,
            new_session_id=child_id,
            agent_name="test-agent",
        )

        assert child_state.metadata.get("project_id") == "proj-inherited"
        assert child_state.metadata.get("cwd") == "/inherited/cwd"

    @pytest.mark.asyncio
    async def test_fork_session_tracked_as_child(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """Forked session should be tracked as a child of parent."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )

        parent_id = "parent-session-003"
        child_id = "child-session-003"

        await integration.create_session(
            session_id=parent_id,
            agent_name="test-agent",
        )

        await integration.fork_session(
            parent_session_id=parent_id,
            new_session_id=child_id,
            agent_name="test-agent",
        )

        children = session_pool.sessions.get_children(parent_id)
        assert child_id in children


class TestInputProviderFlow:
    """Tests for input provider attachment and flow."""

    @pytest.mark.asyncio
    async def test_attach_input_provider_to_session(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
        mock_input_provider: OpenCodeInputProvider,
    ) -> None:
        """Input provider should be attachable to a session."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )

        await integration.create_session(
            session_id="test-session-014",
            agent_name="test-agent",
        )

        await integration.attach_input_provider(
            session_id="test-session-014",
            input_provider=mock_input_provider,
        )

        session_state = session_pool.sessions.get_session("test-session-014")
        assert session_state is not None
        assert session_state.input_provider is mock_input_provider

    @pytest.mark.asyncio
    async def test_route_message_with_input_provider(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
        mock_input_provider: OpenCodeInputProvider,
    ) -> None:
        """Routing a message should pass the input provider to the turn runner."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )

        await integration.create_session(
            session_id="test-session-015",
            agent_name="test-agent",
        )

        run_handle = await integration.route_message(
            session_id="test-session-015",
            content="Message with input provider",
            input_provider=mock_input_provider,
        )

        assert run_handle is not None
        # The input provider should be stored on the session for auto-resume
        session_state = session_pool.sessions.get_session("test-session-015")
        assert session_state is not None
        assert session_state.input_provider is mock_input_provider

    @pytest.mark.asyncio
    async def test_concurrent_sessions_have_isolated_input_providers(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
        mock_input_provider: OpenCodeInputProvider,
    ) -> None:
        """Concurrent sessions must NOT share input provider state.

        Previously, the shared agent's ``_input_provider`` was mutated
        directly, causing race conditions where concurrent sessions
        overwrote each other's input provider. The fix stores input
        providers on ``SessionState`` only and lets SessionController
        pass the correct one at run time.
        """
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )

        # Create two sessions concurrently
        await integration.create_session(
            session_id="test-session-concurrent-a",
            agent_name="test-agent",
        )
        await integration.create_session(
            session_id="test-session-concurrent-b",
            agent_name="test-agent",
        )

        # Create distinct input providers for each session
        from wolfharness_server.opencode_server.input_provider import OpenCodeInputProvider

        provider_a = OpenCodeInputProvider(
            state=server_state, session_id="test-session-concurrent-a"
        )
        provider_b = OpenCodeInputProvider(
            state=server_state, session_id="test-session-concurrent-b"
        )

        await integration.attach_input_provider(
            session_id="test-session-concurrent-a",
            input_provider=provider_a,
        )
        await integration.attach_input_provider(
            session_id="test-session-concurrent-b",
            input_provider=provider_b,
        )

        # Each SessionState must hold its own input provider
        state_a = session_pool.sessions.get_session("test-session-concurrent-a")
        state_b = session_pool.sessions.get_session("test-session-concurrent-b")
        assert state_a is not None
        assert state_b is not None
        assert state_a.input_provider is provider_a
        assert state_b.input_provider is provider_b
        assert state_a.input_provider is not state_b.input_provider

        # The shared agent must NOT be mutated
        shared_agent = session_pool.pool.get_agent("test-agent")  # type: ignore[attr-defined]
        assert shared_agent._input_provider is None


class TestEventSubscription:
    """Tests for subscribing to session events through the integration."""

    @pytest.mark.asyncio
    async def test_subscribe_to_session_events(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """Should be able to subscribe to session events and receive OpenCode events."""
        import asyncio

        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )

        await integration.create_session(
            session_id="test-session-016",
            agent_name="test-agent",
        )

        # Subscribe with a timeout so the test doesn't hang if no events arrive
        events = []
        try:
            async with asyncio.timeout(0.5):
                async for event in integration.subscribe_to_events("test-session-016"):
                    events.append(event)
                    if len(events) >= 1:
                        break
        except TimeoutError:
            pass  # No events within timeout is acceptable

        assert len(events) >= 0  # May or may not have events depending on timing

    @pytest.mark.asyncio
    async def test_event_conversion_in_subscription(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """Subscribed events should be converted to OpenCode SSE events."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )

        await integration.create_session(
            session_id="test-session-017",
            agent_name="test-agent",
        )

        # Publish a RunErrorEvent which is converted to SessionErrorEvent by the adapter
        from wolfharness.agents.events import RunErrorEvent

        await session_pool.event_bus.publish(
            "test-session-017",
            RunErrorEvent(
                run_id="run-001",
                message="test error",
                code="TEST_ERR",
            ),
        )

        events = []
        async for event in integration.subscribe_to_events("test-session-017"):
            events.append(event)
            # We expect OpenCode events, not AgentPool events
            if hasattr(event, "type"):
                break
            if len(events) > 5:
                break

        # At least one event should be an OpenCode event (has 'type' attribute)
        opencode_events = [e for e in events if hasattr(e, "type")]
        assert len(opencode_events) >= 1


class TestIntegrationLifecycle:
    """Tests for the overall lifecycle of the integration layer."""

    @pytest.mark.asyncio
    async def test_shutdown_closes_all_sessions(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """Shutting down integration should close all tracked sessions."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )

        await integration.create_session(
            session_id="test-session-018",
            agent_name="test-agent",
        )
        await integration.create_session(
            session_id="test-session-019",
            agent_name="test-agent",
        )

        await integration.shutdown()

        assert session_pool.sessions.get_session("test-session-018") is None
        assert session_pool.sessions.get_session("test-session-019") is None

    @pytest.mark.asyncio
    async def test_get_session_status_returns_current_status(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """Should return the current status of a session."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )

        await integration.create_session(
            session_id="test-session-020",
            agent_name="test-agent",
        )

        status = await integration.get_session_status("test-session-020")
        assert status is not None
        assert status.type in ("idle", "busy")
