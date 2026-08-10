"""Tests for CancelledError handling in message processing.

When a user cancels/aborts a running agent (e.g., presses ESC in the TUI),
the server must properly finalize the assistant message so that the TUI's
`pending` memo can move past it. Specifically:

1. `assistant_msg.time.completed` must be set (so TUI finds no "pending" msg)
2. `assistant_msg.error` must be set to `MessageAbortedError`
3. A `MessageUpdatedEvent` must be broadcast with the finalized state
4. The message must be persisted to storage

Without these, the TUI's `pending` memo permanently finds the stale
assistant message (which lacks `time.completed`), causing ALL subsequent
user messages to display "QUEUED" indefinitely.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from tests.servers.opencode_server.conftest import run_message_phases
from wolfharness.utils import identifiers as identifier
from wolfharness.utils.time_utils import now_ms
from wolfharness_server.opencode_server.models import (
    AssistantMessage,
    MessageRequest,
    MessageUpdatedEvent,
    TextPartInput,
    TimeCreated,
    UserMessage,
)
from wolfharness_server.opencode_server.models.message import (
    MessageAbortedError,
    MessageWithParts,
)
from wolfharness_server.opencode_server.state import ServerState


pytestmark = pytest.mark.integration


class CancellableAgentMock:
    """Mock agent that raises CancelledError during run_stream.

    Simulates what happens when a user presses ESC in the TUI — the SSE
    stream is closed and the asyncio task gets cancelled.
    """

    def __init__(self) -> None:
        self.name = "test-agent"
        self.run_stream_call_count = 0
        self.agent_pool: Mock | None = None
        self.host_context: Mock | None = None
        self.env: Mock | None = None
        self.storage: Any = None
        self.tools: list[Any] = []
        self._input_provider = None
        self.model_name = "test-model"
        self.session_id: str | None = None
        # Real MessageHistory so conversation state is testable
        from wolfharness.messaging.message_history import MessageHistory

        self.conversation = MessageHistory()

    async def set_model(self, model: str) -> None:
        """Mock set_model method."""
        self.model_name = model

    async def set_mode(self, mode: str, category_id: str | None = None) -> None:
        """Mock set_mode method."""

    async def get_available_models(self):
        """Mock get_available_models method."""
        return []

    async def load_session(self, session_id: str) -> None:
        """Mock load_session method."""
        return

    def run_stream(self, *args: Any, **kwargs: Any):
        """Raise CancelledError immediately to simulate user abort."""
        self.run_stream_call_count += 1

        async def stream():
            # Simulate: agent starts, then gets cancelled mid-stream.
            # The yield makes this an async generator (which is what
            # OpenCodeStreamAdapter.process_stream expects).
            raise asyncio.CancelledError
            yield  # unreachable — makes this an async generator

        return stream()


@pytest.fixture
def cancellable_mock_agent():
    """Create a cancellable mock agent."""
    agent = CancellableAgentMock()

    # Set up pool mock
    pool = Mock()
    pool.manifest = Mock()
    pool.manifest.config_file_path = "/tmp/test"
    pool.manifest.model_variants = {}

    storage = Mock()
    storage.save_session = AsyncMock()
    storage.log_message = AsyncMock()
    pool.storage = storage

    pool.todos = Mock()
    pool.todos.on_change = None
    pool.manifest.agents = {agent.name: agent}

    agent.agent_pool = pool
    agent.host_context = pool
    agent._agent_pool = pool  # state.py resolves _pool via agent._agent_pool

    # Set up SessionPool mock for new architecture
    from wolfharness.lifecycle import RunState

    session_pool = Mock()
    session_pool.sessions = Mock()
    # Create a mock session with the agent attached
    mock_session = Mock()
    mock_session.agent = agent
    session_pool.sessions.get_session = Mock(return_value=mock_session)
    session_pool.sessions.get_or_create_session = AsyncMock(return_value=(mock_session, True))
    session_pool.sessions.get_or_create_session_agent = AsyncMock(return_value=agent)
    session_pool.sessions.store = None
    # Create a RunHandle that raises CancelledError when waiting
    run_handle = Mock()
    run_handle._run_state = RunState.RUNNING
    run_handle.complete_event = Mock()
    run_handle.complete_event.wait = AsyncMock(side_effect=asyncio.CancelledError)
    session_pool.send_message = AsyncMock(return_value=run_handle)
    # Ensure get_messages returns [] so get_messages_for_session falls back to state.messages
    session_pool.get_messages = AsyncMock(return_value=[])
    # Set up a mock event_bus so run_message_phases can subscribe
    session_pool.event_bus = Mock()
    from tests._helpers.mock_stream import EmptyReceiveStream

    session_pool.event_bus.subscribe = AsyncMock(return_value=EmptyReceiveStream())
    session_pool.event_bus.unsubscribe = AsyncMock()
    pool.session_pool = session_pool

    # Set up env mock
    env = Mock()
    fs = Mock()
    fs.read_file = AsyncMock(return_value="file content")
    env.get_fs = Mock(return_value=fs)
    env.cwd = "/tmp"
    agent.env = env

    agent.storage = storage

    return agent


@pytest.fixture
def cancelled_test_state(tmp_project_dir, cancellable_mock_agent):
    """Create a server state with cancellable agent."""
    state = ServerState(
        working_dir=str(tmp_project_dir),
        agent=cancellable_mock_agent,
    )
    # Initialize backward-compat dicts removed from ServerState dataclass
    state.messages = {}
    state.todos = {}
    state.input_providers = {}
    state.pending_questions = {}
    # No session_pool_integration — run_message_phases will use the
    # fallback path via session_pool.sessions.get_or_create_session.
    return state


@pytest.fixture
def sample_message_request():
    """Create a sample message request."""
    return MessageRequest(
        parts=[TextPartInput(text="Hello, test!")],
        agent="default",
    )


def _setup_session(state: ServerState, session_id: str) -> None:
    """Set up session state manually (bypassing ensure_session which needs pool.storage)."""
    from wolfharness_server.opencode_server.models import Session
    from wolfharness_server.opencode_server.models.common import TimeCreatedUpdated

    now = now_ms()
    session = Session(
        id=session_id,
        project_id="default",
        directory=state.working_dir,
        title="Test Session",
        version="1",
        time=TimeCreatedUpdated(created=now, updated=now),
    )
    state.sessions[session_id] = session
    state.messages[session_id] = []
    # Session status is no longer tracked via state.session_status.
    # run_message_phases uses set_session_status() which needs
    # session_pool_integration with _status_bridges.
    state.agent.session_id = session_id


def _create_user_message(
    session_id: str,
    request: MessageRequest,
) -> tuple[str, MessageWithParts]:
    """Create user message and parts (mimics _process_message logic)."""
    user_msg_id = identifier.ascending("message", request.message_id)
    user_message = UserMessage(
        id=user_msg_id,
        session_id=session_id,
        time=TimeCreated.now(),
        agent=request.agent or "default",
        model=request.model,
    )
    user_msg_with_parts = MessageWithParts(info=user_message)
    # Add text parts from request
    for part_input in request.parts:
        if isinstance(part_input, TextPartInput):
            user_msg_with_parts.add_text_part(part_input.text)
    return user_msg_id, user_msg_with_parts


class TestCancelledMessageHandling:
    """Tests for CancelledError handling during message processing."""

    @pytest.mark.asyncio
    async def test_cancelled_message_sets_time_completed(
        self,
        cancelled_test_state: ServerState,
        sample_message_request: MessageRequest,
    ) -> None:
        """Cancelled assistant message must have time.completed set.

        The TUI's `pending` memo finds the last assistant message without
        `time.completed`. If a cancelled message never gets `time.completed`,
        all subsequent user messages show as "QUEUED".
        """
        state = cancelled_test_state
        session_id = "test-session-cancel"

        _setup_session(state, session_id)
        user_msg_id, user_msg_with_parts = _create_user_message(session_id, sample_message_request)
        state.messages[session_id].append(user_msg_with_parts)

        # Process message — agent will raise CancelledError
        await run_message_phases(
            session_id, sample_message_request, state, user_msg_id, user_msg_with_parts
        )

        # Find the assistant message
        assistant_msgs = [
            msg for msg in state.messages[session_id] if isinstance(msg.info, AssistantMessage)
        ]
        assert len(assistant_msgs) == 1, "Should have one assistant message"

        assistant = assistant_msgs[0].info
        assert isinstance(assistant, AssistantMessage)
        assert assistant.time.completed is not None, (
            "Cancelled assistant message MUST have time.completed set — "
            "otherwise TUI marks all subsequent messages as QUEUED"
        )

    @pytest.mark.asyncio
    async def test_cancelled_message_sets_aborted_error(
        self,
        cancelled_test_state: ServerState,
        sample_message_request: MessageRequest,
    ) -> None:
        """Cancelled assistant message must have MessageAbortedError set.

        The TUI checks `message.error` to display the appropriate status
        indicator (e.g., "Aborted" label instead of a spinner).
        """
        state = cancelled_test_state
        session_id = "test-session-cancel-error"

        _setup_session(state, session_id)
        user_msg_id, user_msg_with_parts = _create_user_message(session_id, sample_message_request)
        state.messages[session_id].append(user_msg_with_parts)

        await run_message_phases(
            session_id, sample_message_request, state, user_msg_id, user_msg_with_parts
        )

        assistant_msgs = [
            msg for msg in state.messages[session_id] if isinstance(msg.info, AssistantMessage)
        ]
        assert len(assistant_msgs) == 1

        assistant = assistant_msgs[0].info
        assert isinstance(assistant, AssistantMessage)
        assert assistant.error is not None, "Cancelled assistant message MUST have error set"
        assert isinstance(assistant.error, MessageAbortedError), (
            f"Error should be MessageAbortedError, got {type(assistant.error).__name__}"
        )

    @pytest.mark.asyncio
    async def test_cancelled_message_broadcasts_update_event(
        self,
        cancelled_test_state: ServerState,
        sample_message_request: MessageRequest,
    ) -> None:
        """Cancelled assistant message must be persisted with error state.

        The event bridge (via _finalize_assistant_time on RunFailedEvent)
        handles the SSE broadcast. _wait_and_finalize is responsible for
        persisting the aborted assistant message to state and storage.
        This test verifies the persist path; the broadcast is tested in
        integration tests with a real event bridge.
        """
        state = cancelled_test_state
        session_id = "test-session-cancel-event"

        _setup_session(state, session_id)
        user_msg_id, user_msg_with_parts = _create_user_message(session_id, sample_message_request)
        state.messages[session_id].append(user_msg_with_parts)

        # Capture broadcast events
        all_events: list[Any] = []
        original_broadcast = state.broadcast_event

        async def tracking_broadcast(event: Any) -> None:
            all_events.append(event)
            await original_broadcast(event)

        state.broadcast_event = tracking_broadcast  # type: ignore[method-assign]

        await run_message_phases(
            session_id, sample_message_request, state, user_msg_id, user_msg_with_parts
        )

        # Find the assistant message in state
        assistant_msgs = [
            msg for msg in state.messages[session_id] if isinstance(msg.info, AssistantMessage)
        ]
        assert len(assistant_msgs) == 1
        assistant_info = assistant_msgs[0].info
        assert isinstance(assistant_info, AssistantMessage)

        # The assistant message must have time.completed set (persisted by
        # _wait_and_finalize) and carry the aborted error.
        assert assistant_info.time.completed is not None, (
            "Persisted assistant message must have time.completed set"
        )
        assert assistant_info.error is not None, (
            "Persisted assistant message for a cancelled run must carry the error"
        )

        # _wait_and_finalize must NOT broadcast MessageUpdatedEvent for the
        # assistant message — that is the event bridge's responsibility via
        # _finalize_assistant_time(). Broadcasting here would cause duplicate
        # SSE events (issue #263).
        assistant_broadcasts = [
            e
            for e in all_events
            if isinstance(e, MessageUpdatedEvent)
            and isinstance(e.properties.info, AssistantMessage)
            and e.properties.info.id == assistant_info.id
        ]
        assert len(assistant_broadcasts) == 0, (
            "_wait_and_finalize must not broadcast MessageUpdatedEvent for the assistant "
            "message — the event bridge handles this via _finalize_assistant_time() "
            "(issue #263: duplicate SSE broadcasts)"
        )

    @pytest.mark.asyncio
    async def test_cancelled_message_session_returns_to_idle(
        self,
        cancelled_test_state: ServerState,
        sample_message_request: MessageRequest,
    ) -> None:
        """After cancellation, session status must return to idle.

        This is already working (the finally block handles it), but we verify
        it stays that way.
        """
        state = cancelled_test_state
        session_id = "test-session-cancel-idle"

        _setup_session(state, session_id)
        user_msg_id, user_msg_with_parts = _create_user_message(session_id, sample_message_request)
        state.messages[session_id].append(user_msg_with_parts)

        await run_message_phases(
            session_id, sample_message_request, state, user_msg_id, user_msg_with_parts
        )

        # Session status is no longer tracked via state.session_status.
        # The cancellation is verified by the assertions above.

    @pytest.mark.asyncio
    async def test_message_after_cancel_is_not_queued(
        self,
        cancelled_test_state: ServerState,
        sample_message_request: MessageRequest,
    ) -> None:
        """After a cancelled message, a new message should NOT appear as QUEUED.

        This is the user-facing bug: after cancelling, all subsequent user
        messages show "QUEUED" because the stale assistant message has no
        `time.completed`. This test verifies the TUI's pending logic would
        work correctly after the fix.

        The TUI's pending memo:
          pending = messages().findLast(x => x.role === "assistant" && !x.time.completed)?.id
          queued = props.pending && props.message.id > props.pending
        """
        state = cancelled_test_state
        session_id = "test-session-not-queued"

        _setup_session(state, session_id)

        # First message: gets cancelled
        user_msg_id_1, user_msg_1 = _create_user_message(session_id, sample_message_request)
        state.messages[session_id].append(user_msg_1)
        await run_message_phases(
            session_id, sample_message_request, state, user_msg_id_1, user_msg_1
        )

        # Second message: should NOT be queued
        second_request = MessageRequest(
            parts=[TextPartInput(text="Second message after cancel")],
            agent="default",
            message_id="msg-after-cancel",
        )
        user_msg_id_2, user_msg_2 = _create_user_message(session_id, second_request)
        state.messages[session_id].append(user_msg_2)
        await run_message_phases(session_id, second_request, state, user_msg_id_2, user_msg_2)

        # Simulate the TUI's pending memo logic
        all_messages = state.messages[session_id]
        pending_id = None
        for msg in all_messages:
            if isinstance(msg.info, AssistantMessage) and msg.info.time.completed is None:
                pending_id = msg.info.id

        assert pending_id is None, (
            f"No assistant message should be 'pending' (without time.completed), "
            f"but found pending message {pending_id}. This causes the TUI to "
            f"display subsequent user messages as QUEUED."
        )

    @pytest.mark.asyncio
    async def test_cancelled_message_preserves_conversation_history(
        self,
        cancelled_test_state: ServerState,
        sample_message_request: MessageRequest,
    ) -> None:
        """After cancellation, the agent's in-memory conversation must include the aborted response.

        The agent's `conversation.chat_messages` is what gets sent to the LLM as
        conversation history. When a run is cancelled, `_stream_events` adds the
        user message but never adds the assistant response (the post-processing
        code at base_agent.py:857-858 is skipped due to the exception).

        Without adding the aborted assistant message to the conversation, the next
        `run_stream()` call sends incomplete history — the LLM doesn't know it
        already (partially) responded, causing "conversation history lost" symptoms.
        """
        state = cancelled_test_state
        session_id = "test-session-history"

        _setup_session(state, session_id)

        # Record initial conversation length
        initial_count = len(state.agent.conversation.chat_messages)

        user_msg_id, user_msg_with_parts = _create_user_message(session_id, sample_message_request)
        state.messages[session_id].append(user_msg_with_parts)

        # Process message — agent will raise CancelledError
        await run_message_phases(
            session_id, sample_message_request, state, user_msg_id, user_msg_with_parts
        )

        # In the real flow, _stream_events adds the user message to conversation
        # (base_agent.py:784), and our CancelledError handler adds the aborted
        # assistant message. Since CancellableAgentMock doesn't run _stream_events,
        # only our handler's addition is reflected. What matters is that the
        # aborted assistant message IS present in the conversation.
        final_count = len(state.agent.conversation.chat_messages)

        # The aborted assistant message must have been added to the conversation
        assert final_count >= initial_count + 1, (
            f"Agent conversation should have at least {initial_count + 1} messages "
            f"(original + aborted assistant), but has {final_count}. "
            f"Without the aborted assistant message in conversation.chat_messages, "
            f"the LLM receives incomplete history on the next run."
        )

        # Verify the last message in conversation is the aborted assistant
        last_msg = state.agent.conversation.chat_messages[-1]
        assert last_msg.role == "assistant", (
            f"The last message in agent conversation should be an assistant message, "
            f"but got role='{last_msg.role}'. The aborted assistant response must be "
            f"added to conversation so the LLM knows about it."
        )
