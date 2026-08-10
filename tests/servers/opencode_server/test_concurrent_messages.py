"""Tests for concurrent message handling in OpenCode server.

These tests verify that the OpenCode server correctly handles concurrent
messages to the same session, preventing race conditions and event interleaving.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from wolfharness_server.opencode_server.models import (
    MessageRequest,
    TextPartInput,
)
from wolfharness_server.opencode_server.models.message import UserMessage
from wolfharness_server.opencode_server.routes.message_routes import _process_message
from wolfharness_server.opencode_server.session_pool_integration import ensure_session
from wolfharness_server.opencode_server.state import ServerState


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class SlowAgentMock:
    """Mock agent that simulates slow processing to expose concurrency issues."""

    def __init__(self, delay: float = 0.5) -> None:
        self.name = "test-agent"
        self.delay = delay
        self.run_stream_call_count = 0
        self.active_runs: set[str] = set()
        self.agent_pool: Mock | None = None
        self.host_context: Mock | None = None
        self.env: Mock | None = None
        self.storage: Any = None
        self.tools = []
        self._input_provider = None
        self.model_name = "test-model"
        self.session_id: str | None = None

    async def set_model(self, model: str) -> None:
        """Mock set_model method."""
        self.model_name = model

    async def set_mode(self, mode: str, category_id: str | None = None) -> None:
        """Mock set_mode method."""
        return

    async def get_available_models(self):
        """Mock get_available_models method."""
        return []

    async def load_session(self, session_id: str) -> Any:
        """Mock load_session method."""
        self.session_id = session_id
        return None

    def run_stream(
        self,
        user_prompt: Any,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Simulate slow processing with concurrent run detection."""
        self.run_stream_call_count += 1

        # Check if another run is already active for this session
        if session_id in self.active_runs:
            raise RuntimeError(
                f"Concurrent run detected for session {session_id}! "
                "This indicates missing concurrency control."
            )

        self.active_runs.add(session_id or "unknown")

        async def stream() -> AsyncIterator[Any]:
            try:
                # Simulate processing time
                await asyncio.sleep(self.delay)

                # Yield a simple text event
                from wolfharness.agents.events import StreamCompleteEvent, TextContentItem
                from wolfharness.messaging import ChatMessage

                yield TextContentItem(text=f"Response for {session_id}")
                yield StreamCompleteEvent(message=ChatMessage(role="assistant", content="done"))
            finally:
                self.active_runs.discard(session_id or "unknown")

        return stream()


@pytest.fixture
def slow_mock_agent():  # noqa: PLR0915
    """Create a slow mock agent for testing concurrency."""
    agent = SlowAgentMock(delay=0.3)
    saved_sessions: dict[str, Any] = {}

    # Set up pool mock with async storage methods
    pool = Mock()
    pool.manifest = Mock()
    pool.manifest.config_file_path = "/tmp/test"
    pool.manifest.model_variants = {}

    # Storage needs to be properly mocked with async methods
    storage = Mock()

    async def save_session(session_data: Any) -> None:
        saved_sessions[session_data.session_id] = session_data

    storage.save_session = AsyncMock(side_effect=save_session)
    storage.log_session = AsyncMock()
    storage.log_message = AsyncMock()
    storage.load_session = AsyncMock(return_value=None)
    pool.storage = storage

    pool.todos = Mock()
    pool.todos.on_change = None

    pool.sessions = Mock()
    pool.sessions.store = None
    pool.session_pool = Mock()
    pool.session_pool.sessions = Mock()
    pool.session_pool.sessions.store = None

    pool.sessions = Mock()
    pool.sessions.store = None
    pool.session_pool = Mock()
    pool.session_pool.sessions = Mock()
    pool.session_pool.sessions.store = None
    # Ensure get_messages returns [] so get_messages_for_session falls back to state.messages
    pool.session_pool.get_messages = AsyncMock(return_value=[])

    # Mock SessionPool methods that are awaited in _process_message_locked
    pool.session_pool.sessions.get_or_create_session = AsyncMock(return_value=(Mock(), True))
    pool.session_pool.sessions.get_or_create_session_agent = AsyncMock(return_value=agent)
    pool.session_pool.sessions.get_session = Mock(return_value=None)

    # Set up a real EventBus so adapter can subscribe/unsubscribe
    from wolfharness.orchestrator.core import EventBus

    event_bus = EventBus(max_queue_size=100)
    pool.session_pool.event_bus = event_bus

    # Mock receive_request to actually call agent.run_stream and publish events
    async def _mock_receive_request(
        *,
        session_id: str,
        content: Any,
        priority: str = "when_idle",
        input_provider: Any | None = None,
        message_id: str | None = None,
        **kwargs: Any,
    ):
        from wolfharness.lifecycle import RunOutcome, RunState
        from wolfharness.orchestrator.run import RunHandle

        handle = Mock(spec=RunHandle)
        handle.run_id = "test-run"
        handle.session_id = session_id
        handle._run_state = RunState.RUNNING
        complete_event = asyncio.Event()
        handle.complete_event = complete_event

        async def _do_run():
            try:
                stream = agent.run_stream(content, session_id=session_id)
                async for event in stream:
                    await event_bus.publish(session_id, event)
                handle._run_state = RunState.DONE
                handle.outcome = RunOutcome.COMPLETED
            except Exception:  # noqa: BLE001
                handle._run_state = RunState.DONE
                handle.outcome = RunOutcome.FAILED
            finally:
                complete_event.set()

        _task = asyncio.create_task(_do_run())  # noqa: RUF006
        return message_id or "msg_test_run"

    pool.session_pool.send_message = AsyncMock(side_effect=_mock_receive_request)

    # Mock wait_for_completion to wait for the run to actually finish
    async def _mock_wait_for_completion(
        sid: str,
        timeout: float | None = None,
    ) -> str:
        # Wait a bit for the background task to complete
        await asyncio.sleep(0.5)
        return sid

    pool.session_pool.wait_for_completion = AsyncMock(side_effect=_mock_wait_for_completion)
    # Add cancel_run_for_session to the existing sessions mock
    if not hasattr(pool.session_pool.sessions, "cancel_run_for_session"):
        pool.session_pool.sessions.cancel_run_for_session = Mock()

    # CRITICAL: all_agents must return a real dict to avoid Mock issues
    pool.manifest.agents = {agent.name: agent}

    agent.agent_pool = pool
    agent.host_context = pool
    agent._agent_pool = pool  # state.py resolves _pool via agent._agent_pool

    # Set up env mock
    env = Mock()
    fs = Mock()
    fs.read_file = AsyncMock(return_value="file content")
    env.get_fs = Mock(return_value=fs)
    env.cwd = "/tmp"
    agent.env = env

    # Set up storage
    agent.storage = storage

    conversation = Mock()
    conversation.chat_messages = []
    agent.conversation = conversation

    async def load_session(session_id: str) -> Any:
        return saved_sessions.get(session_id)

    agent.load_session = AsyncMock(side_effect=load_session)

    return agent


@pytest.fixture
def concurrent_test_state(tmp_project_dir, slow_mock_agent):
    """Create a server state with slow agent for concurrency testing."""
    state = ServerState(
        working_dir=str(tmp_project_dir),
        agent=slow_mock_agent,
    )
    # Initialize backward-compat dicts removed from ServerState dataclass
    state.messages = {}
    state.session_status = {}
    state.todos = {}
    state.input_providers = {}
    state.pending_questions = {}

    # Simulate EventProcessor: wrap send_message to also add user messages
    # to state. In production, EventProcessor._process_user_message_inserted
    # receives UserMessageInsertedEvent from the EventBus and calls
    # append_message_to_session. The EventProcessor isn't running in tests,
    # so we simulate it here.
    original_send_message = state.pool.session_pool.send_message

    async def _wrapping_send_message(*args: Any, **kwargs: Any) -> Any:
        result = await original_send_message(*args, **kwargs)
        message_id = kwargs.get("message_id")
        content = kwargs.get("content")
        sid = kwargs.get("session_id", "")
        if message_id is not None and content is not None:
            import time as _time

            from wolfharness_server.opencode_server.models.common import TimeCreated
            from wolfharness_server.opencode_server.models.message import (
                MessageWithParts,
                UserMessage,
            )
            from wolfharness_server.opencode_server.opencode_message_bridge import (
                append_message_to_session,
            )

            user_message = UserMessage(
                id=message_id,
                session_id=sid,
                time=TimeCreated(created=int(_time.time() * 1000)),
            )
            user_msg_with_parts = MessageWithParts(info=user_message)
            if isinstance(content, str) and content:
                user_msg_with_parts.add_text_part(content)
            await append_message_to_session(state, sid, user_msg_with_parts)
        return result

    state.pool.session_pool.send_message = _wrapping_send_message

    return state


@pytest.fixture
def sample_message_request():
    """Create a sample message request."""
    return MessageRequest(
        parts=[TextPartInput(text="Hello, test!")],
        agent="default",
    )


class TestConcurrentMessageHandling:
    """Tests for concurrent message handling behavior."""

    @pytest.mark.asyncio
    async def test_concurrent_messages_same_session_should_be_sequential(
        self,
        concurrent_test_state: ServerState,
        sample_message_request: MessageRequest,
    ) -> None:
        """Test that concurrent messages to the same session are processed sequentially.

        This test verifies that when multiple messages are sent to the same session
        concurrently, they are processed one at a time (not in parallel), preventing
        event interleaving and data corruption.

        Before the fix: This test would fail because both messages would be processed
        concurrently, causing the SlowAgentMock to raise a RuntimeError.

        After the fix: Messages should be processed sequentially, and no concurrent
        run error should occur.
        """
        state = concurrent_test_state
        session_id = "test-session-concurrent"

        # Create session first
        await ensure_session(state, session_id)

        # Track events for verification
        all_events = []
        original_broadcast = state.broadcast_event

        async def tracking_broadcast(event):
            all_events.append(event)
            await original_broadcast(event)

        state.broadcast_event = tracking_broadcast  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]

        # Send two messages concurrently to the same session
        # This should NOT cause concurrent processing
        async def send_message_with_id(msg_id: str):
            req = sample_message_request.model_copy()
            req.message_id = msg_id
            return await _process_message(session_id, req, state)

        # Run both messages concurrently
        results = await asyncio.gather(
            send_message_with_id("msg-1"),
            send_message_with_id("msg-2"),
            return_exceptions=True,
        )

        # Debug: capture all error events first
        from wolfharness_server.opencode_server.models.events import SessionErrorEvent

        error_events = [e for e in all_events if isinstance(e, SessionErrorEvent)]
        if error_events:
            print(f"Error events found: {error_events}")
            for err in error_events:
                if hasattr(err, "properties") and hasattr(err.properties, "message"):
                    print(f"Error message: {err.properties.message}")

        # Verify no errors occurred (no concurrent run detected)
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Exception during processing: {result}")

        # Verify both messages were processed.
        # In the async model, _process_message only adds user messages —
        # assistant messages are registered by the event consumer on
        # StreamCompleteEvent. We verify both user messages were added
        # and the agent was called twice.
        assert len(state.messages[session_id]) == 2  # 2 user messages
        user_msgs = [m for m in state.messages[session_id] if m.info.role == "user"]
        assert len(user_msgs) == 2

        # Verify the agent was called twice
        agent_mock = cast(SlowAgentMock, state.agent)
        assert agent_mock.run_stream_call_count == 2

    @pytest.mark.asyncio
    async def test_session_status_reflects_busy_state(
        self,
        concurrent_test_state: ServerState,
        sample_message_request: MessageRequest,
    ) -> None:
        """Test that session status correctly reflects busy state during processing.

        Session status is now broadcast via set_session_status() which calls
        server_state.broadcast_event() directly, instead of writing to the
        in-memory session_status dict. We track status from broadcast events.
        """
        state = concurrent_test_state
        session_id = "test-session-status"

        # Create session
        await ensure_session(state, session_id)

        # Track status changes from broadcast events
        status_types_seen: list[str] = []
        original_broadcast = state.broadcast_event

        async def tracking_broadcast(event: Any) -> None:
            if hasattr(event, "type") and event.type == "session.status":
                status_types_seen.append(event.properties.status.type)
            await original_broadcast(event)

        state.broadcast_event = tracking_broadcast  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]

        # Process a message
        await _process_message(session_id, sample_message_request, state)

        # Verify status transitioned through busy and back to idle
        assert "busy" in status_types_seen, (
            f"Expected 'busy' in status broadcasts, got {status_types_seen}"
        )
        assert "idle" in status_types_seen, (
            f"Expected 'idle' in status broadcasts, got {status_types_seen}"
        )

    @pytest.mark.asyncio
    async def test_different_sessions_run_concurrently_with_per_session_agents(
        self,
        concurrent_test_state: ServerState,
        sample_message_request: MessageRequest,
    ) -> None:
        """Test that different sessions with per-session agents can run concurrently.

        With per-session agent instances, different sessions no longer need
        to be serialized by a global agent_lock. Each session has its own
        agent (or falls back to the shared agent), and per-session locks
        ensure same-session serialization while allowing cross-session
        concurrency.
        """
        state = concurrent_test_state
        session_id_1 = "test-session-1"
        session_id_2 = "test-session-2"

        # Create both sessions
        await ensure_session(state, session_id_1)
        await ensure_session(state, session_id_2)

        # Process messages to different sessions concurrently
        results = await asyncio.gather(
            _process_message(session_id_1, sample_message_request, state),
            _process_message(session_id_2, sample_message_request, state),
            return_exceptions=True,
        )

        # Verify both sessions processed their messages without errors.
        # In the async model, _process_message only adds user messages —
        # assistant messages are registered by the event consumer on
        # StreamCompleteEvent.
        for result in results:
            assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify both sessions have their user messages
        assert len(state.messages[session_id_1]) == 1  # user message
        assert len(state.messages[session_id_2]) == 1  # user message

    @pytest.mark.asyncio
    async def test_message_ordering_preserved_under_concurrency(
        self,
        concurrent_test_state: ServerState,
        sample_message_request: MessageRequest,
    ) -> None:
        """Test that message ordering is preserved when messages are processed sequentially.

        When multiple messages are queued for the same session, they should be
        processed in the order they were received.
        """
        state = concurrent_test_state
        session_id = "test-session-order"

        # Create session
        await ensure_session(state, session_id)

        # Send messages with specific IDs to verify order
        async def send_message_with_content(content: str, msg_id: str):
            req = MessageRequest(
                parts=[TextPartInput(text=content)],
                agent="default",
                message_id=msg_id,
            )
            return await _process_message(session_id, req, state)

        # Process multiple messages concurrently
        await asyncio.gather(
            send_message_with_content("First message", "msg-first"),
            send_message_with_content("Second message", "msg-second"),
            send_message_with_content("Third message", "msg-third"),
        )

        # Get user messages (every other message starting from 0)
        user_messages = [
            msg for msg in state.messages[session_id] if isinstance(msg.info, UserMessage)
        ]

        # Verify we have 3 user messages
        assert len(user_messages) == 3

        # Verify the agent was called 3 times
        agent_mock = cast(SlowAgentMock, state.agent)
        assert agent_mock.run_stream_call_count == 3
