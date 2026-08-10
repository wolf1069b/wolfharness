"""Merged tests for question handling in the OpenCode server.

Combines tests from:
- test_question_abort_regression.py
- test_question_integration.py
- test_question_session_controller.py
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock

from mcp import types
import pytest

from tests.servers.opencode_server.conftest import run_message_phases
from wolfharness.orchestrator.core import SessionController, SessionState
from wolfharness.tasks.exceptions import RunAbortedError
from wolfharness.utils import identifiers as identifier
from wolfharness.utils.time_utils import now_ms
from wolfharness_server.opencode_server.input_provider import (
    OpenCodeInputProvider,
    PendingPermission,
)
from wolfharness_server.opencode_server.models import (
    AssistantMessage,
    MessageRequest,
    PermissionRequestEvent,
    PermissionResolvedEvent,
    QuestionReply,
    TextPartInput,
    TimeCreated,
    UserMessage,
)
from wolfharness_server.opencode_server.models.message import (
    MessageAbortedError,
    MessageWithParts,
)
from wolfharness_server.opencode_server.routes.question_routes import (
    list_questions,
    reject_question,
    reply_to_question,
)
from wolfharness_server.opencode_server.session_pool_integration import (
    append_message_to_session,
    get_messages_for_session,
    get_session_status,
)
from wolfharness_server.opencode_server.state import PendingQuestion, ServerState


pytestmark = pytest.mark.integration


# =============================================================================
# --- Merged from test_question_abort_regression.py ---
# =============================================================================

# ---------------------------------------------------------------------------
# Mock agents that simulate the question abort scenarios
# ---------------------------------------------------------------------------


class RunAbortedAgentMock:
    """Mock agent that raises RunAbortedError during run_stream.

    Simulates: question_for_user tool raises RunAbortedError when user
    cancels the questionnaire (ESC in TUI).
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
        from wolfharness.messaging.message_history import MessageHistory

        self.conversation = MessageHistory()

    async def set_model(self, model: str) -> None:
        self.model_name = model

    async def set_mode(self, mode: str, category_id: str | None = None) -> None:
        pass

    async def get_available_models(self):
        return []

    async def load_session(self, session_id: str) -> None:
        return None

    def run_stream(self, *args: Any, **kwargs: Any):
        self.run_stream_call_count += 1

        async def stream():
            # Simulate: agent starts streaming, calls question_for_user,
            # which raises RunAbortedError when user cancels.
            raise RunAbortedError("User cancelled the questionnaire")
            yield

        return stream()


class BlockingOnQuestionAgentMock:
    """Mock agent that blocks forever waiting for a question answer.

    Simulates: agent calls question_for_user, which creates a Future
    and awaits it. The Future is never resolved, so agent_lock stays held.
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
        self.block_forever_event: asyncio.Event = asyncio.Event()
        from wolfharness.messaging.message_history import MessageHistory

        self.conversation = MessageHistory()

    async def set_model(self, model: str) -> None:
        self.model_name = model

    async def set_mode(self, mode: str, category_id: str | None = None) -> None:
        pass

    async def get_available_models(self):
        return []

    async def load_session(self, session_id: str) -> None:
        return None

    def run_stream(self, *args: Any, **kwargs: Any):
        self.run_stream_call_count += 1

        async def stream():
            # Simulate: agent blocks waiting for question answer (Future never resolves).
            # In real code this is: answers = await future  (in input_provider.py:344)
            # Here we just wait on an Event that nobody will set.
            await self.block_forever_event.wait()
            if False:
                yield None

        return stream()


class BlockingOnRealQuestionAgentMock:
    """Mock agent that creates a real PendingQuestion and blocks on its Future.

    Unlike BlockingOnQuestionAgentMock which blocks on an Event, this creates
    an actual PendingQuestion in session.pending_questions and awaits the Future.
    This simulates the real question_for_user flow more accurately, allowing
    tests to verify that cancel_all_pending_questions() releases agent_lock.
    """

    def __init__(self, state: ServerState) -> None:
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
        self._state = state
        from wolfharness.messaging.message_history import MessageHistory

        self.conversation = MessageHistory()

    async def set_model(self, model: str) -> None:
        self.model_name = model

    async def set_mode(self, mode: str, category_id: str | None = None) -> None:
        pass

    async def get_available_models(self):
        return []

    async def load_session(self, session_id: str) -> None:
        return None

    def run_stream(self, *args: Any, session_id: str | None = None, **kwargs: Any):
        self.run_stream_call_count += 1
        state = self._state
        _session_id = session_id or "unknown"

        async def stream():
            # Simulate: agent calls question_for_user → input_provider.get_elicitation()
            # creates a PendingQuestion and awaits the Future.
            yield None  # Ensure async for starts executing the generator body
            question_id = f"que_test_{id(self)}"
            future: asyncio.Future[list[list[str]]] = asyncio.get_event_loop().create_future()
            # Store on SessionState via session_controller if available
            pending_questions_dict: dict[str, Any] | None = None
            if state.session_controller is not None:
                session = state.session_controller.get_session(_session_id)
                if session is not None:
                    pending_questions_dict = session.pending_questions
            if pending_questions_dict is None:
                # Fallback: use a local dict (won't be visible to cancel_all)
                pending_questions_dict = {}
            pending_questions_dict[question_id] = PendingQuestion(
                session_id=_session_id,
                questions=[],
                future=future,
            )
            try:
                # This blocks until the Future is resolved or cancelled
                await future
            except asyncio.CancelledError:
                # Same path as input_provider.py:354-356
                raise RunAbortedError("User cancelled the questionnaire") from None
            finally:
                pending_questions_dict.pop(question_id, None)

        return stream()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pool_mock(agent: Any) -> Mock:  # noqa: PLR0915
    """Create a mock pool wired to the given agent."""
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

    # Set up SessionPool mock for new architecture
    session_pool = Mock()
    session_pool.sessions = Mock()
    session_pool.sessions.get_or_create_session = AsyncMock(return_value=(Mock(), True))
    session_pool.sessions.get_or_create_session_agent = AsyncMock(return_value=agent)
    session_pool.sessions.store = None
    sp_session = Mock()
    sp_session.agent = agent
    sp_session.current_run_id = None
    session_pool.sessions.get_session = Mock(return_value=sp_session)
    # Use a functional event bus that routes publish→subscribe
    from tests.servers.opencode_server.conftest import _make_functional_event_bus

    session_pool.event_bus = _make_functional_event_bus()

    # Override subscribe to return a real queue-based stream
    _event_queues: dict[str, list[Any]] = {}

    async def _subscribe(sid: str, scope: str = "session") -> Any:
        from asyncio import Queue

        q: Any = Queue(maxsize=1024)
        _event_queues.setdefault(sid, []).append(q)
        return q

    async def _unsubscribe(sid: str, q: Any) -> None:
        if sid in _event_queues:
            _event_queues[sid] = [x for x in _event_queues[sid] if x is not q]

    async def _publish(sid: str, event: Any) -> None:
        for subscriber_sid, queues in _event_queues.items():
            if subscriber_sid == sid:
                for q in queues:
                    with contextlib.suppress(Exception):
                        q.put_nowait(event)

    session_pool.event_bus.subscribe = AsyncMock(side_effect=_subscribe)
    session_pool.event_bus.unsubscribe = AsyncMock(side_effect=_unsubscribe)
    session_pool.event_bus.publish = AsyncMock(side_effect=_publish)

    # Shared completion tracking across receive_request and wait_for_completion
    _completion_events: dict[str, asyncio.Event] = {}

    async def _mock_receive_request(
        session_id: str,
        content: str,
        priority: str = "when_idle",
        input_provider: Any = None,
        message_id: str | None = None,
        **kwargs: Any,
    ) -> str | None:
        from wolfharness.lifecycle import RunOutcome, RunState

        complete_event = asyncio.Event()
        _completion_events[session_id] = complete_event
        run_handle = Mock()
        run_handle._run_state = RunState.RUNNING
        run_handle.complete_event = complete_event

        async def _background_run():
            try:
                stream = agent.run_stream(content, session_id=session_id)
                async for event in stream:
                    await session_pool.event_bus.publish(session_id, event)
                run_handle._run_state = RunState.DONE
                run_handle.outcome = RunOutcome.COMPLETED
            except Exception as exc:  # noqa: BLE001
                run_handle._run_state = RunState.DONE
                run_handle.outcome = RunOutcome.FAILED
                # Publish RunFailedEvent so the message_routes error path fires
                from wolfharness.agents.events import RunFailedEvent

                await session_pool.event_bus.publish(
                    session_id,
                    RunFailedEvent(
                        run_id="test-run",
                        session_id=session_id,
                        exception=exc,
                    ),
                )
            finally:
                complete_event.set()

        _task = asyncio.create_task(_background_run())  # noqa: RUF006
        return message_id or "msg_test_run"

    session_pool.send_message = _mock_receive_request

    # Mock wait_for_completion to actually wait for the background run
    async def _mock_wait_for_completion(
        sid: str,
        timeout: float | None = None,
    ) -> str:
        ev = _completion_events.get(sid)
        if ev is not None:
            await asyncio.wait_for(ev.wait(), timeout=timeout or 30.0)
        return sid

    session_pool.wait_for_completion = _mock_wait_for_completion
    session_pool.sessions.cancel_run_for_session = Mock()
    pool.session_pool = session_pool

    return pool


def _make_env_mock(tmp_dir: str) -> Mock:
    """Create a mock environment."""
    env = Mock()
    fs = Mock()
    fs.read_file = AsyncMock(return_value="file content")
    env.get_fs = Mock(return_value=fs)
    env.cwd = tmp_dir
    return env


@pytest.fixture
def aborted_mock_agent(tmp_project_dir):
    """Create a RunAbortedAgentMock with pool and env."""
    agent = RunAbortedAgentMock()
    agent.agent_pool = _make_pool_mock(agent)
    agent.host_context = agent.agent_pool
    agent._agent_pool = agent.agent_pool  # state.py resolves _pool via agent._agent_pool
    agent.env = _make_env_mock(str(tmp_project_dir))
    agent.storage = agent.agent_pool.storage
    return agent


@pytest.fixture
def blocking_mock_agent(tmp_project_dir):
    """Create a BlockingOnQuestionAgentMock with pool and env."""
    agent = BlockingOnQuestionAgentMock()
    agent.agent_pool = _make_pool_mock(agent)
    agent.host_context = agent.agent_pool
    agent._agent_pool = agent.agent_pool  # state.py resolves _pool via agent._agent_pool
    agent.env = _make_env_mock(str(tmp_project_dir))
    agent.storage = agent.agent_pool.storage
    return agent


@pytest.fixture
def aborted_test_state(aborted_mock_agent, tmp_project_dir):
    return ServerState(working_dir=str(tmp_project_dir), agent=aborted_mock_agent)


@pytest.fixture
def blocking_test_state(blocking_mock_agent, tmp_project_dir):
    return ServerState(working_dir=str(tmp_project_dir), agent=blocking_mock_agent)


@pytest.fixture
def blocking_real_question_state(tmp_project_dir):
    """Create a ServerState with an agent that creates real PendingQuestions."""
    # Need to create state first so the agent can reference it
    placeholder_agent = RunAbortedAgentMock()
    placeholder_agent.agent_pool = _make_pool_mock(placeholder_agent)
    placeholder_agent.host_context = placeholder_agent.agent_pool
    placeholder_agent._agent_pool = (
        placeholder_agent.agent_pool
    )  # state.py resolves _pool via agent._agent_pool
    placeholder_agent.env = _make_env_mock(str(tmp_project_dir))
    placeholder_agent.storage = placeholder_agent.agent_pool.storage
    state = ServerState(working_dir=str(tmp_project_dir), agent=placeholder_agent)
    # Set up a mock session_controller for the BlockingOnRealQuestionAgentMock
    from wolfharness.orchestrator.core import SessionState as SPSessionState

    sp_session = SPSessionState(session_id="test-session", agent_name="test-agent")
    controller = Mock()
    controller.get_session = Mock(return_value=sp_session)
    controller._sessions = {"test-session": sp_session}

    def _cancel_all():
        cancelled = []
        for session in controller._sessions.values():
            for qid, pending in list(session.pending_questions.items()):
                if not pending.future.done():
                    pending.future.cancel()
                    cancelled.append(qid)
        return cancelled

    controller.cancel_all_pending_questions = Mock(side_effect=_cancel_all)
    state.session_controller = controller
    # Now create the real blocking agent with state reference
    real_agent = BlockingOnRealQuestionAgentMock(state)
    real_agent.agent_pool = _make_pool_mock(real_agent)
    real_agent.host_context = real_agent.agent_pool
    real_agent._agent_pool = real_agent.agent_pool  # state.py resolves _pool via agent._agent_pool
    real_agent.env = _make_env_mock(str(tmp_project_dir))
    real_agent.storage = real_agent.agent_pool.storage
    state.agent = real_agent
    # Update the pool reference on the state to use the real agent's pool
    state._pool = real_agent.host_context
    return state


@pytest.fixture
def sample_message_request():
    return MessageRequest(
        parts=[TextPartInput(text="Hello, test!")],
        agent="default",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_session(state: ServerState, session_id: str) -> None:
    """Set up session state manually."""
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
    # Dynamically add fallback dicts for helpers that use getattr
    if not hasattr(state, "messages"):
        state.messages = {}
    state.messages[session_id] = []
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
    for part_input in request.parts:
        if isinstance(part_input, TextPartInput):
            user_msg_with_parts.add_text_part(part_input.text)
    return user_msg_id, user_msg_with_parts


# ---------------------------------------------------------------------------
# Red Flag Test #1: RunAbortedError corrupts agent conversation history
# ---------------------------------------------------------------------------


class TestRunAbortedErrorCorruptsConversation:
    """BUG: RunAbortedError is not caught by the CancelledError handler.

    When question_for_user raises RunAbortedError, _run_message_locked
    does NOT add the aborted assistant message to the agent's conversation.
    This corrupts the LLM's context for subsequent messages.

    Compare with test_cancelled_message.py which tests the CancelledError path
    (which IS properly handled). These tests should pass once RunAbortedError
    is added to the except clause at message_routes.py:480.
    """

    @pytest.mark.asyncio
    async def test_run_aborted_error_message_has_time_completed(
        self,
        aborted_test_state: ServerState,
        sample_message_request: MessageRequest,
    ) -> None:
        """RunAbortedError assistant message MUST have time.completed set.

        Without this, the TUI's `pending` memo permanently finds the stale
        assistant message, causing all subsequent user messages to display
        as "QUEUED" — same bug as CancelledError but through a different
        exception path.
        """
        state = aborted_test_state
        session_id = "test-abort-completed"

        _setup_session(state, session_id)
        user_msg_id, user_msg_with_parts = _create_user_message(session_id, sample_message_request)
        await append_message_to_session(state, session_id, user_msg_with_parts)

        await run_message_phases(
            session_id, sample_message_request, state, user_msg_id, user_msg_with_parts
        )

        assistant_msgs = [
            msg
            for msg in await get_messages_for_session(state, session_id)
            if isinstance(msg.info, AssistantMessage)
        ]
        assert len(assistant_msgs) == 1, "Should have one assistant message"

        assistant = assistant_msgs[0].info
        assert isinstance(assistant, AssistantMessage)
        assert assistant.time.completed is not None, (
            "RunAbortedError assistant message MUST have time.completed set — "
            "otherwise TUI marks all subsequent messages as QUEUED. "
            "This is the same invariant as CancelledError (see test_cancelled_message.py)."
        )

    @pytest.mark.asyncio
    async def test_run_aborted_error_message_has_aborted_error(
        self,
        aborted_test_state: ServerState,
        sample_message_request: MessageRequest,
    ) -> None:
        """RunAbortedError assistant message MUST have MessageAbortedError set."""
        state = aborted_test_state
        session_id = "test-abort-error"

        _setup_session(state, session_id)
        user_msg_id, user_msg_with_parts = _create_user_message(session_id, sample_message_request)
        await append_message_to_session(state, session_id, user_msg_with_parts)

        await run_message_phases(
            session_id, sample_message_request, state, user_msg_id, user_msg_with_parts
        )

        assistant_msgs = [
            msg
            for msg in await get_messages_for_session(state, session_id)
            if isinstance(msg.info, AssistantMessage)
        ]
        assert len(assistant_msgs) == 1

        assistant = assistant_msgs[0].info
        assert isinstance(assistant, AssistantMessage)
        assert assistant.error is not None, (
            "RunAbortedError assistant message MUST have error set — same as CancelledError path."
        )
        assert isinstance(assistant.error, MessageAbortedError), (
            f"Error should be MessageAbortedError, got {type(assistant.error).__name__}"
        )

    @pytest.mark.asyncio
    async def test_run_aborted_error_preserves_conversation_history(
        self,
        aborted_test_state: ServerState,
        sample_message_request: MessageRequest,
    ) -> None:
        """CRITICAL: After RunAbortedError, conversation MUST include aborted message.

        This is the root cause of the TUI black screen bug:
        - RunAbortedError is NOT caught by `except (CancelledError, TimeoutError)`
        - The aborted assistant message is NOT added to agent.conversation
        - On the next message, the LLM sees corrupted history:
          it has the user message but no assistant response
        - The LLM may behave unpredictably (repeat tool calls, hallucinate, etc.)

        Compare: test_cancelled_message.py::test_cancelled_message_preserves_conversation_history
        which tests the CancelledError path (which IS properly handled).
        """
        state = aborted_test_state
        session_id = "test-abort-history"

        _setup_session(state, session_id)
        initial_count = len(state.agent.conversation.chat_messages)

        user_msg_id, user_msg_with_parts = _create_user_message(session_id, sample_message_request)
        await append_message_to_session(state, session_id, user_msg_with_parts)

        await run_message_phases(
            session_id, sample_message_request, state, user_msg_id, user_msg_with_parts
        )

        final_count = len(state.agent.conversation.chat_messages)

        # The aborted assistant message MUST have been added to conversation
        assert final_count >= initial_count + 1, (
            f"Agent conversation should have at least {initial_count + 1} messages "
            f"(original + aborted assistant), but has {final_count}. "
            f"RunAbortedError is not handled like CancelledError — the aborted "
            f"assistant message is never added to agent.conversation.chat_messages. "
            f"This corrupts the LLM's context for the next message."
        )

        # The last message should be the aborted assistant
        last_msg = state.agent.conversation.chat_messages[-1]
        assert last_msg.role == "assistant", (
            f"The last message in agent conversation should be assistant, "
            f"but got role='{last_msg.role}'. The aborted assistant response "
            f"must be added so the LLM knows about it."
        )

    @pytest.mark.asyncio
    async def test_run_aborted_error_session_returns_to_idle(
        self,
        aborted_test_state: ServerState,
        sample_message_request: MessageRequest,
    ) -> None:
        """After RunAbortedError, session status must return to idle.

        This currently works because process_stream catches the error and
        yields SessionErrorEvent, then the iterator completes normally,
        allowing agent_lock to be released and mark_session_idle to fire.
        But we verify it stays that way.
        """
        state = aborted_test_state
        session_id = "test-abort-idle"

        # Set up session_pool_integration mock so set_session_status /
        # get_session_status work. The old state.session_status fallback
        # was removed in the SessionPool single-path cleanup.
        _session_statuses: dict[str, Any] = {}

        async def _mock_get_status(sid: str) -> Any:
            return _session_statuses.get(sid)

        # Capture broadcasted SessionStatusEvents to populate _session_statuses
        _original_broadcast = state.broadcast_event

        async def _capturing_broadcast(event: Any) -> None:
            from wolfharness_server.opencode_server.models import SessionStatusEvent

            if isinstance(event, SessionStatusEvent):
                _session_statuses[event.properties.session_id] = event.properties.status
            await _original_broadcast(event)

        state.broadcast_event = _capturing_broadcast  # type: ignore[method-assign]

        integration = AsyncMock()
        integration.create_session = AsyncMock(return_value=Mock())
        integration.get_session_status = AsyncMock(side_effect=_mock_get_status)

        async def _mock_create_session(sid: str, *args: Any, **kw: Any) -> Any:
            return Mock()

        integration.create_session = AsyncMock(side_effect=_mock_create_session)
        state.session_pool_integration = integration

        _setup_session(state, session_id)
        user_msg_id, user_msg_with_parts = _create_user_message(session_id, sample_message_request)
        await append_message_to_session(state, session_id, user_msg_with_parts)

        ctx = await run_message_phases(
            session_id, sample_message_request, state, user_msg_id, user_msg_with_parts
        )
        # Phase 3: mark session idle (not included in run_message_phases)
        from wolfharness_server.opencode_server.routes.message_routes import (
            _mark_session_idle_safe,
        )

        await _mark_session_idle_safe(state, session_id, ctx)

        status = await get_session_status(state, session_id)
        assert status is not None
        assert status.type == "idle", "Session must be idle after RunAbortedError"

    @pytest.mark.asyncio
    async def test_message_after_run_aborted_is_not_queued(
        self,
        aborted_test_state: ServerState,
        sample_message_request: MessageRequest,
    ) -> None:
        """After RunAbortedError, a new message should NOT appear as QUEUED.

        This is the user-facing symptom: after aborting a question and sending
        a new message, the TUI shows "QUEUED" because the stale assistant
        message lacks `time.completed`.
        """
        state = aborted_test_state
        session_id = "test-abort-not-queued"

        _setup_session(state, session_id)

        # First message: RunAbortedError
        user_msg_id_1, user_msg_1 = _create_user_message(session_id, sample_message_request)
        await append_message_to_session(state, session_id, user_msg_1)
        await run_message_phases(
            session_id, sample_message_request, state, user_msg_id_1, user_msg_1
        )

        # Second message: should NOT be queued
        second_request = MessageRequest(
            parts=[TextPartInput(text="Second message after abort")],
            agent="default",
            message_id="msg-after-abort",
        )
        user_msg_id_2, user_msg_2 = _create_user_message(session_id, second_request)
        await append_message_to_session(state, session_id, user_msg_2)
        await run_message_phases(session_id, second_request, state, user_msg_id_2, user_msg_2)

        # Simulate the TUI's pending memo logic
        all_messages = await get_messages_for_session(state, session_id)
        pending_id = None
        for msg in all_messages:
            if isinstance(msg.info, AssistantMessage) and msg.info.time.completed is None:
                pending_id = msg.info.id

        assert pending_id is None, (
            f"No assistant message should be 'pending' (without time.completed), "
            f"but found pending message {pending_id}. This causes the TUI to "
            f"display subsequent user messages as QUEUED after question abort."
        )


# ---------------------------------------------------------------------------
# Red Flag Test #2: agent_lock deadlock when question Future is never resolved
# ---------------------------------------------------------------------------


class TestAgentLockDeadlockOnUnresolvedQuestion:
    """Per-session agents resolve the agent_lock deadlock.

    With per-session agents, each session has its own agent instance.
    There is no global agent_lock that could deadlock when one session
    blocks on a question. The old deadlock scenario (agent_lock held
    while agent blocks on question, preventing ALL other sessions from
    processing) is resolved by the per-session agent architecture.

    These tests verify that the per-session model prevents the deadlock
    that existed in the shared-agent model.
    """

    @pytest.mark.asyncio
    async def test_per_session_agents_no_agent_lock_deadlock(
        self,
        blocking_test_state: ServerState,
        sample_message_request: MessageRequest,
    ) -> None:
        """With per-session agents, a blocking question in one session doesn't block another.

        In the old shared-agent model, agent_lock was held while the agent
        blocked on a question, preventing get_or_load_session from working
        for ANY session. With per-session agents, get_or_load_session no
        longer uses agent_lock.
        """
        state = blocking_test_state
        session_id = "test-no-deadlock"

        _setup_session(state, session_id)
        user_msg_id, user_msg_with_parts = _create_user_message(session_id, sample_message_request)
        await append_message_to_session(state, session_id, user_msg_with_parts)

        # Start message processing in background (it will block on the question)
        process_task = asyncio.create_task(
            run_message_phases(
                session_id, sample_message_request, state, user_msg_id, user_msg_with_parts
            )
        )

        # Give the task time to start
        await asyncio.sleep(0.2)

        # Verify agent_lock is NOT held (per-session agents don't need it)
        # In the old model, this would timeout because agent_lock was held.
        # In the new model, agent_lock should be available.
        try:
            await asyncio.wait_for(state.agent_lock.acquire(), timeout=0.5)
            state.agent_lock.release()
            # agent_lock is available — no deadlock
        except TimeoutError:
            pytest.fail(
                "agent_lock should NOT be held while agent blocks on question "
                "in the per-session agent model. The deadlock bug is back!"
            )

        # Clean up
        process_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await process_task

    @pytest.mark.asyncio
    async def test_no_deadlock_different_session_after_blocking_question(
        self,
        blocking_test_state: ServerState,
        sample_message_request: MessageRequest,
    ) -> None:
        """With per-session agents, a blocking question doesn't prevent loading another session.

        This verifies that the "关了 opencode 重新启动, 无法在新 session 中发送
        user message" bug is resolved by per-session agents.
        """
        state = blocking_test_state
        session_id = "test-no-deadlock-session"

        _setup_session(state, session_id)
        user_msg_id, user_msg_with_parts = _create_user_message(session_id, sample_message_request)
        await append_message_to_session(state, session_id, user_msg_with_parts)

        # Start message processing in background (blocks on question)
        process_task = asyncio.create_task(
            run_message_phases(
                session_id, sample_message_request, state, user_msg_id, user_msg_with_parts
            )
        )

        # Give it time to start
        await asyncio.sleep(0.2)

        # Now try to get_or_load_session for a DIFFERENT session
        # In the old model, this would deadlock because get_or_load_session
        # needs agent_lock which is held by the blocking task.
        # In the new model, get_or_load_session doesn't use agent_lock.
        new_session_id = "test-no-deadlock-new-session"
        _setup_session(state, new_session_id)

        # This MUST NOT deadlock — per-session agents resolve the issue
        try:
            from wolfharness_server.opencode_server.routes.session_routes import get_or_load_session

            await asyncio.wait_for(get_or_load_session(state, new_session_id), timeout=1.0)
            # Either gets the session or None — both are fine, no deadlock
        except TimeoutError:
            pytest.fail(
                "get_or_load_session should NOT deadlock when another session "
                "blocks on a question. The per-session agent model should prevent this."
            )

        # Clean up
        process_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await process_task

    @pytest.mark.asyncio
    async def test_cancelling_pending_question_releases_resources(
        self,
        blocking_test_state: ServerState,
        sample_message_request: MessageRequest,
    ) -> None:
        """If we cancel the pending question's Future, resources are released.

        This test verifies that cancelling the Future properly propagates
        through the agent stack and the task completes.

        In production, this would happen when:
        - TUI sends question reject (ESC)
        - Server detects SSE disconnect and cancels pending questions
        """
        state = blocking_test_state
        session_id = "test-cancel-releases"

        _setup_session(state, session_id)
        user_msg_id, user_msg_with_parts = _create_user_message(session_id, sample_message_request)
        await append_message_to_session(state, session_id, user_msg_with_parts)

        # Start message processing in background
        process_task = asyncio.create_task(
            run_message_phases(
                session_id, sample_message_request, state, user_msg_id, user_msg_with_parts
            )
        )

        # Wait for the question to be created
        await asyncio.sleep(0.3)

        # Cancel the task directly (simulates question cancellation)
        process_task.cancel()

        # Wait for the task to be cancelled
        with contextlib.suppress(asyncio.CancelledError):
            await process_task

        # Verify agent_lock is available after cancellation
        try:
            await asyncio.wait_for(state.agent_lock.acquire(), timeout=0.5)
            state.agent_lock.release()
        except TimeoutError:
            pytest.fail(
                "agent_lock should be available after cancelling the blocked task, "
                "but it's still held. This indicates a lock leak on CancelledError."
            )


# ---------------------------------------------------------------------------
# Red Flag Test #4: SSE disconnect cancels pending questions → releases agent_lock
# ---------------------------------------------------------------------------


class TestSSEDisconnectReleasesAgentLock:
    """Per-session agents resolve the agent_lock deadlock on SSE disconnect.

    With per-session agents, the agent_lock is no longer used by
    get_or_load_session, so the deadlock scenario where an unresolved
    question blocks ALL session access is resolved.

    These tests verify that cancel_all_pending_questions still works
    correctly and that new sessions can be accessed after disconnect.
    """

    @pytest.mark.asyncio
    async def test_cancel_all_pending_questions_cancels_futures(
        self,
        blocking_real_question_state: ServerState,
        sample_message_request: MessageRequest,
    ) -> None:
        """cancel_all_pending_questions() must cancel all pending question Futures."""
        state = blocking_real_question_state
        session_id = "test-cancel-questions"

        _setup_session(state, session_id)
        user_msg_id, user_msg_with_parts = _create_user_message(session_id, sample_message_request)
        await append_message_to_session(state, session_id, user_msg_with_parts)

        # Start message processing in background (will create PendingQuestion)
        process_task = asyncio.create_task(
            run_message_phases(
                session_id, sample_message_request, state, user_msg_id, user_msg_with_parts
            )
        )

        # Allow event loop to start process_task and background_run
        await asyncio.sleep(0)

        # Wait for the question to be created in session controller
        session = (
            state.session_controller.get_session(session_id) if state.session_controller else None
        )
        for _ in range(40):
            if session and session.pending_questions:
                break
            await asyncio.sleep(0.05)

        assert session is not None, "Session should exist"
        assert session.pending_questions, (
            "Agent should have created a pending question, but no pending questions found."
        )

        # Simulate SSE disconnect: call cancel_all_pending_questions
        cancelled_ids = state.cancel_all_pending_questions()

        assert len(cancelled_ids) > 0, "cancel_all_pending_questions should return cancelled IDs"

        # The process task should complete (no longer blocked)
        try:
            await asyncio.wait_for(process_task, timeout=2.0)
        except TimeoutError:
            pytest.fail(
                "process_task should complete after cancel_all_pending_questions, "
                "but it's still running."
            )

        # Verify agent_lock is available
        try:
            await asyncio.wait_for(state.agent_lock.acquire(), timeout=0.5)
            state.agent_lock.release()
        except TimeoutError:
            pytest.fail(
                "agent_lock should be available after cancelling pending questions, "
                "but it's still held."
            )

    @pytest.mark.asyncio
    async def test_cancel_all_pending_questions_allows_new_session_access_after_sse_disconnect(
        self,
        blocking_real_question_state: ServerState,
        sample_message_request: MessageRequest,
    ) -> None:
        """After SSE disconnect, a new session can be accessed without deadlock.

        This reproduces the exact user scenario: "关了 opencode 重新启动, 无法在新
        session 中发送 user message". With per-session agents, get_or_load_session
        no longer uses agent_lock, so this scenario cannot deadlock.
        """
        state = blocking_real_question_state
        session_id = "test-sse-disconnect"

        _setup_session(state, session_id)
        user_msg_id, user_msg_with_parts = _create_user_message(session_id, sample_message_request)
        await append_message_to_session(state, session_id, user_msg_with_parts)

        # Start message processing in background (will block on question)
        process_task = asyncio.create_task(
            run_message_phases(
                session_id, sample_message_request, state, user_msg_id, user_msg_with_parts
            )
        )

        # Wait for the question to be created
        session = (
            state.session_controller.get_session(session_id) if state.session_controller else None
        )
        for _ in range(20):
            if session and session.pending_questions:
                break
            await asyncio.sleep(0.05)

        # Simulate SSE disconnect
        state.cancel_all_pending_questions()

        # Wait for the process task to complete
        try:
            await asyncio.wait_for(process_task, timeout=2.0)
        except TimeoutError:
            pytest.fail("process_task should complete after cancelling questions")

        # Now try accessing a NEW session via get_or_load_session
        new_session_id = "test-sse-reconnect"
        _setup_session(state, new_session_id)

        # This MUST NOT deadlock — per-session agents resolve the issue
        from wolfharness_server.opencode_server.routes.session_routes import get_or_load_session

        try:
            await asyncio.wait_for(get_or_load_session(state, new_session_id), timeout=2.0)
        except TimeoutError:
            pytest.fail(
                "get_or_load_session for new session should succeed after SSE disconnect + "
                "cancel_all_pending_questions, but it timed out (deadlock)."
            )


# ---------------------------------------------------------------------------
# Red Flag Test #3: UnboundLocalError when CancelledError before agent assignment
# ---------------------------------------------------------------------------


class CancelBeforeAgentAssignmentMock:
    """Mock agent whose bind_agent_to_session raises CancelledError.

    Simulates: CancelledError arrives before `agent` is assigned in the
    `async with state.agent_lock:` block. The except clause at line 480
    references `agent` at line 518, which is UnboundLocalError if CancelledError
    fires before line 376 (`agent = state.agent`).

    We trigger this by making bind_agent_to_session raise CancelledError,
    which occurs at line 379 — after agent assignment but before run_stream.
    This is a realistic scenario (task cancellation during session binding).
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
        from wolfharness.messaging.message_history import MessageHistory

        self.conversation = MessageHistory()

    async def set_model(self, model: str) -> None:
        self.model_name = model

    async def set_mode(self, mode: str, category_id: str | None = None) -> None:
        pass

    async def get_available_models(self):
        return []

    async def load_session(self, session_id: str) -> None:
        return None

    def run_stream(self, *args: Any, **kwargs: Any):
        self.run_stream_call_count += 1

        async def stream():
            if False:
                yield None

        return stream()


class TestUnboundLocalErrorInExceptHandler:
    """BUG: UnboundLocalError when CancelledError occurs during agent binding.

    When CancelledError occurs during bind_agent_to_session (line 379), the
    except handler at line 480 tries to access `agent` at line 518, but `agent`
    is a local variable defined inside `async with state.agent_lock:` (line 376).
    If CancelledError arrives before that assignment, `agent` is unbound.

    Even in the case where agent IS assigned (line 376 runs before CancelledError),
    the `agent` variable is scoped inside the `async with` block. Python's scoping
    rules mean that if the assignment never executes (e.g., CancelledError at
    `state.agent_lock.acquire()`), the except handler crashes with UnboundLocalError.

    Fix: Hoist `agent = state.agent` before the `async with state.agent_lock:` block.
    """

    @pytest.mark.asyncio
    async def test_cancelled_error_during_agent_binding_no_crash(
        self,
        aborted_test_state: ServerState,
        sample_message_request: MessageRequest,
    ) -> None:
        """CancelledError during agent binding must NOT crash with UnboundLocalError.

        We simulate this by making the agent_lock's __aenter__ raise CancelledError,
        which prevents `agent = state.agent` (line 376) from executing. The except
        handler at line 480 references `agent` at line 518, so UnboundLocalError occurs.
        """
        state = aborted_test_state
        session_id = "test-unbound-agent"

        _setup_session(state, session_id)

        # Replace agent_lock with one that raises CancelledError on acquire
        original_lock = state.agent_lock
        lock_that_cancels = asyncio.Lock()

        # Make the lock raise CancelledError on first acquire
        original_acquire = lock_that_cancels.acquire

        _acquire_count = 0

        async def acquire_raising_cancel() -> bool:
            nonlocal _acquire_count
            _acquire_count += 1
            if _acquire_count == 1:
                raise asyncio.CancelledError("Simulated cancellation during lock acquire")
            return await original_acquire()

        lock_that_cancels.acquire = acquire_raising_cancel  # type: ignore[assignment]
        state.agent_lock = lock_that_cancels

        user_msg_id, user_msg_with_parts = _create_user_message(session_id, sample_message_request)
        await append_message_to_session(state, session_id, user_msg_with_parts)

        # This should NOT raise UnboundLocalError — it should handle CancelledError gracefully
        try:
            await run_message_phases(
                session_id, sample_message_request, state, user_msg_id, user_msg_with_parts
            )
        except asyncio.CancelledError:
            # CancelledError may propagate if not caught internally — that's fine
            pass
        except UnboundLocalError:
            pytest.fail(
                "UnboundLocalError in except handler: `agent` is referenced at line 518 "
                "but defined inside `async with state.agent_lock:` at line 376. "
                "When CancelledError occurs before agent assignment, the except handler "
                "crashes. Fix: hoist `agent = state.agent` before the agent_lock block."
            )
        finally:
            state.agent_lock = original_lock


# ---------------------------------------------------------------------------
# Import for contextlib (used in cleanup)
# ---------------------------------------------------------------------------

import contextlib  # noqa: E402


pytestmark = pytest.mark.integration


# =============================================================================
# --- Merged from test_question_integration.py ---
# =============================================================================


def _make_mock_session_controller(session_id: str) -> Mock:
    """Create a mock SessionController with a SessionState for the given session."""
    from wolfharness.orchestrator.core import SessionState

    session = SessionState(session_id=session_id, agent_name="test-agent")
    controller = Mock()
    controller.get_session = Mock(return_value=session)
    controller._sessions = {session_id: session}
    return controller


async def test_question_elicitation_single_select():
    """Test single-select question via elicitation."""
    # This is a basic unit test without full server
    # Create minimal mock agent (pool not needed for this test)
    mock_agent = Mock()
    mock_agent.agent_pool = None
    mock_agent.host_context = None
    # Create minimal state
    state = ServerState(working_dir="/tmp", agent=mock_agent)
    state.session_controller = _make_mock_session_controller("test_session")
    # Create provider
    provider = OpenCodeInputProvider(state=state, session_id="test_session")
    # Create elicitation params with enum
    schema = {"type": "string", "enum": ["PostgreSQL", "MySQL", "SQLite"]}
    params = types.ElicitRequestFormParams(message="Which database?", requestedSchema=schema)

    # Start elicitation in background
    async def get_answer():
        return await provider.get_elicitation(params)

    task = asyncio.create_task(get_answer())
    # Wait a bit for question to be created
    await asyncio.sleep(0.1)
    # Verify question was created
    session = state.session_controller.get_session("test_session")
    assert len(session.pending_questions) == 1
    question_id = next(iter(session.pending_questions.keys()))
    pending = session.pending_questions[question_id]
    # Verify question structure
    assert pending.session_id == "test_session"
    assert len(pending.questions) == 1
    question_info = pending.questions[0]
    assert question_info.question == "Which database?"
    assert question_info.multiple is None
    assert len(question_info.options) == 3
    # Simulate user reply
    success = provider.resolve_question(question_id, [["PostgreSQL"]])
    assert success
    # Wait for result
    result = await task
    # Verify result
    assert isinstance(result, types.ElicitResult)
    assert result.action == "accept"
    assert result.content == {"value": "PostgreSQL"}
    # Verify cleanup
    assert question_id not in session.pending_questions


async def test_question_elicitation_multi_select():
    """Test multi-select question via elicitation."""
    mock_agent = Mock()
    mock_agent.agent_pool = None
    mock_agent.host_context = None
    state = ServerState(working_dir="/tmp", agent=mock_agent)
    state.session_controller = _make_mock_session_controller("test_session")
    provider = OpenCodeInputProvider(state=state, session_id="test_session")
    # Multi-select schema
    schema = {"type": "array", "items": {"type": "string", "enum": ["Auth", "API", "Admin"]}}
    params = types.ElicitRequestFormParams(message="Which features?", requestedSchema=schema)
    task = asyncio.create_task(provider.get_elicitation(params))
    await asyncio.sleep(0.1)
    # Get question
    session = state.session_controller.get_session("test_session")
    question_id = next(iter(session.pending_questions.keys()))
    pending = session.pending_questions[question_id]
    question_info = pending.questions[0]
    # Verify multi-select flag
    assert question_info.multiple is True
    # Reply with multiple selections
    provider.resolve_question(question_id, [["Auth", "Admin"]])
    result = await task
    # Multi-select returns list in dict
    assert isinstance(result, types.ElicitResult)
    assert result.action == "accept"
    assert result.content == {"value": ["Auth", "Admin"]}


async def test_question_cancellation():
    """Test question cancellation."""
    mock_agent = Mock()
    mock_agent.agent_pool = None
    mock_agent.host_context = None
    state = ServerState(working_dir="/tmp", agent=mock_agent)
    state.session_controller = _make_mock_session_controller("test_session")
    provider = OpenCodeInputProvider(state=state, session_id="test_session")
    schema = {"type": "string", "enum": ["PostgreSQL", "MySQL"]}
    params = types.ElicitRequestFormParams(message="Which database?", requestedSchema=schema)
    task = asyncio.create_task(provider.get_elicitation(params))
    await asyncio.sleep(0.1)
    # Get question and cancel it
    session = state.session_controller.get_session("test_session")
    question_id = next(iter(session.pending_questions.keys()))
    future = session.pending_questions[question_id].future
    future.cancel()
    result = await task
    # Should return cancel action
    assert isinstance(result, types.ElicitResult)
    assert result.action == "cancel"


async def test_question_with_descriptions():
    """Test question with option descriptions."""
    mock_agent = Mock()
    mock_agent.agent_pool = None
    mock_agent.host_context = None
    state = ServerState(working_dir="/tmp", agent=mock_agent)
    state.session_controller = _make_mock_session_controller("test_session")
    provider = OpenCodeInputProvider(state=state, session_id="test_session")
    # Schema with custom descriptions
    schema = {
        "type": "string",
        "enum": ["PostgreSQL", "MySQL", "SQLite"],
        "x-option-descriptions": {
            "PostgreSQL": "Best for production",
            "MySQL": "Compatible with many tools",
            "SQLite": "Lightweight, file-based",
        },
    }
    params = types.ElicitRequestFormParams(message="Which database?", requestedSchema=schema)
    task = asyncio.create_task(provider.get_elicitation(params))
    await asyncio.sleep(0.1)
    # Verify descriptions were included
    session = state.session_controller.get_session("test_session")
    question_id = next(iter(session.pending_questions.keys()))
    question_info = session.pending_questions[question_id].questions[0]
    options = question_info.options
    assert options[0].label == "PostgreSQL"
    assert options[0].description == "Best for production"
    assert options[1].description == "Compatible with many tools"
    # Clean up
    future = session.pending_questions[question_id].future
    future.cancel()
    await task


async def test_multi_question_rfc0010_example():
    """Test multi-question with RFC-0010 schema format (q0, q1, etc.).

    RFC-0010 example schema format:
    {
        "type": "object",
        "properties": {
            "q0": {"type": "string", "enum": ["opt1", "opt2"]},
            "q1": {"type": "array", "items": {"enum": ["val1", "val2"]}}
        }
    }
    """
    mock_agent = Mock()
    mock_agent.agent_pool = None
    mock_agent.host_context = None
    state = ServerState(working_dir="/tmp", agent=mock_agent)
    state.session_controller = _make_mock_session_controller("test_session")
    provider = OpenCodeInputProvider(state=state, session_id="test_session")

    # RFC-0010 example schema with q0, q1 format
    schema = {
        "type": "object",
        "properties": {
            "q0": {
                "type": "string",
                "enum": ["opt1", "opt2"],
                "title": "First Choice",
                "description": "Select your first option",
            },
            "q1": {
                "type": "array",
                "items": {"enum": ["val1", "val2"]},
                "title": "Features",
                "description": "Select multiple features",
            },
        },
    }
    params = types.ElicitRequestFormParams(
        message="Configuration questions", requestedSchema=schema
    )

    # Start elicitation in background
    task = asyncio.create_task(provider.get_elicitation(params))
    await asyncio.sleep(0.1)

    # Verify question was created with multiple questions
    session = state.session_controller.get_session("test_session")
    assert len(session.pending_questions) == 1
    question_id = next(iter(session.pending_questions.keys()))
    pending = session.pending_questions[question_id]

    # Verify 2 questions created
    assert len(pending.questions) == 2

    # First question (q0) - single-select enum
    question1 = pending.questions[0]
    assert question1.question == "Select your first option"
    assert question1.header == "First Choice"[:12]  # Truncated title
    assert question1.multiple is None  # Single-select
    assert len(question1.options) == 2
    assert question1.options[0].label == "opt1"
    assert question1.options[1].label == "opt2"

    # Second question (q1) - multi-select array
    question2 = pending.questions[1]
    assert question2.question == "Select multiple features"
    assert question2.header == "Features"[:12]  # Truncated title
    assert question2.multiple is True  # Multi-select
    assert len(question2.options) == 2
    assert question2.options[0].label == "val1"
    assert question2.options[1].label == "val2"

    # Simulate user answers (answering both questions)
    success = provider.resolve_question(question_id, [["opt1"], ["val1", "val2"]])
    assert success

    # Wait for result
    result = await task

    # Verify result preserves original property keys (q0, q1)
    assert isinstance(result, types.ElicitResult)
    assert result.action == "accept"
    assert result.content == {"q0": "opt1", "q1": ["val1", "val2"]}

    assert question_id not in session.pending_questions


async def test_multi_question_cancellation():
    """Test cancellation during multi-question flow."""
    mock_agent = Mock()
    mock_agent.agent_pool = None
    mock_agent.host_context = None
    state = ServerState(working_dir="/tmp", agent=mock_agent)
    state.session_controller = _make_mock_session_controller("test_session")
    provider = OpenCodeInputProvider(state=state, session_id="test_session")

    # Multi-question schema with 3 questions
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "title": "Name", "description": "Your name"},
            "role": {
                "type": "string",
                "enum": ["admin", "user"],
                "title": "Role",
                "description": "Select role",
            },
            "features": {
                "type": "array",
                "items": {"enum": ["a", "b"]},
                "title": "Features",
                "description": "Select features",
            },
        },
    }
    params = types.ElicitRequestFormParams(message="User details", requestedSchema=schema)

    task = asyncio.create_task(provider.get_elicitation(params))
    await asyncio.sleep(0.1)

    # Get question and cancel it
    session = state.session_controller.get_session("test_session")
    question_id = next(iter(session.pending_questions.keys()))
    future = session.pending_questions[question_id].future
    future.cancel()

    result = await task

    # Should return cancel action
    assert isinstance(result, types.ElicitResult)
    assert result.action == "cancel"


async def test_question_reply_can_resolve_permission_request():
    """Permission replies routed through /question should still resolve."""
    mock_agent = Mock()
    mock_agent.agent_pool = None
    mock_agent.host_context = None
    state = ServerState(working_dir="/tmp", agent=mock_agent)
    state.session_controller = _make_mock_session_controller("test_session")
    provider = OpenCodeInputProvider(state=state, session_id="test_session")
    session = state.session_controller.get_session("test_session")
    session.input_provider = provider
    broadcast_calls = []

    async def _mock_broadcast(event):
        broadcast_calls.append(event)

    state.broadcast_event = _mock_broadcast

    permission_id = "perm_1_1776434635956"
    future = asyncio.get_running_loop().create_future()
    provider._pending_permissions[permission_id] = PendingPermission(
        permission_id=permission_id,
        tool_name="bash",
        args={"command": "echo test"},
        future=future,
    )

    result = await reply_to_question(
        permission_id,
        QuestionReply(answers=[["once"]]),
        state,
    )

    assert result is True
    assert future.done()
    assert future.result() == "once"
    assert len(broadcast_calls) == 1
    event = broadcast_calls[0]
    assert isinstance(event, PermissionResolvedEvent)
    assert event.properties.request_id == permission_id
    assert event.properties.reply == "once"


async def test_question_reject_can_resolve_permission_request():
    """Permission rejects routed through /question should still resolve."""
    mock_agent = Mock()
    mock_agent.agent_pool = None
    mock_agent.host_context = None
    state = ServerState(working_dir="/tmp", agent=mock_agent)
    state.session_controller = _make_mock_session_controller("test_session")
    provider = OpenCodeInputProvider(state=state, session_id="test_session")
    session = state.session_controller.get_session("test_session")
    session.input_provider = provider
    broadcast_calls = []

    async def _mock_broadcast(event):
        broadcast_calls.append(event)

    state.broadcast_event = _mock_broadcast

    permission_id = "perm_2_1776434635957"
    future = asyncio.get_running_loop().create_future()
    provider._pending_permissions[permission_id] = PendingPermission(
        permission_id=permission_id,
        tool_name="bash",
        args={"command": "echo reject"},
        future=future,
    )

    result = await reject_question(permission_id, state)

    assert result is True
    assert future.done()
    assert future.result() == "reject"
    assert len(broadcast_calls) == 1
    event = broadcast_calls[0]
    assert isinstance(event, PermissionResolvedEvent)
    assert event.properties.request_id == permission_id
    assert event.properties.reply == "reject"


async def test_permission_request_uses_permission_prefix():
    """Permission requests should keep the permission ID namespace."""
    mock_agent = Mock()
    mock_agent.agent_pool = None
    mock_agent.host_context = None
    state = ServerState(working_dir="/tmp", agent=mock_agent)
    state.session_controller = _make_mock_session_controller("test_session")
    provider = OpenCodeInputProvider(state=state, session_id="test_session")
    broadcast_calls = []

    async def _mock_broadcast(event):
        broadcast_calls.append(event)

    state.broadcast_event = _mock_broadcast

    context = Mock()
    context.tool_name = "bash"
    context.tool_input = {"command": "echo test"}
    context.tool_call_id = "call-123"

    task = asyncio.create_task(provider.get_tool_confirmation(context))
    await asyncio.sleep(0.1)

    assert len(broadcast_calls) == 1
    event = broadcast_calls[0]
    assert isinstance(event, PermissionRequestEvent)
    assert event.properties.id.startswith("perm_")

    resolved = provider.resolve_permission(event.properties.id, "once")
    assert resolved is True

    result = await task
    assert result == "allow"


async def test_multi_question_partial_answers():
    """Test multi-question with partial answers (fewer than questions)."""
    mock_agent = Mock()
    mock_agent.agent_pool = None
    mock_agent.host_context = None
    state = ServerState(working_dir="/tmp", agent=mock_agent)
    state.session_controller = _make_mock_session_controller("test_session")
    provider = OpenCodeInputProvider(state=state, session_id="test_session")

    # Schema with 3 questions
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "string", "enum": ["x", "y"], "title": "A", "description": "Select A"},
            "b": {"type": "string", "enum": ["m", "n"], "title": "B", "description": "Select B"},
            "c": {"type": "string", "enum": ["p", "q"], "title": "C", "description": "Select C"},
        },
    }
    params = types.ElicitRequestFormParams(message="Selections", requestedSchema=schema)

    task = asyncio.create_task(provider.get_elicitation(params))
    await asyncio.sleep(0.1)

    session = state.session_controller.get_session("test_session")
    question_id = next(iter(session.pending_questions.keys()))

    # Provide only 2 answers for 3 questions
    success = provider.resolve_question(question_id, [["x"], ["m"]])
    assert success

    result = await task

    assert isinstance(result, types.ElicitResult)
    assert result.action == "accept"
    # Only first 2 properties should have answers
    assert result.content == {"a": "x", "b": "m"}
    assert question_id not in session.pending_questions


async def test_multi_question_empty_object_declines():
    """Test that empty object schema returns decline."""
    mock_agent = Mock()
    mock_agent.agent_pool = None
    mock_agent.host_context = None
    state = ServerState(working_dir="/tmp", agent=mock_agent)
    provider = OpenCodeInputProvider(state=state, session_id="test_session")

    # Empty object schema (no properties)
    schema = {"type": "object", "properties": {}}
    params = types.ElicitRequestFormParams(message="Empty config", requestedSchema=schema)

    result = await provider.get_elicitation(params)

    # Empty object schema doesn't match len(props) >= 1, goes to fallback case
    # which returns decline
    assert isinstance(result, types.ElicitResult)
    assert result.action == "decline"


async def test_multi_question_rfc0010_backward_compat():
    """Test RFC-0010 schema maintains backward compatibility with single questions."""
    mock_agent = Mock()
    mock_agent.agent_pool = None
    mock_agent.host_context = None
    state = ServerState(working_dir="/tmp", agent=mock_agent)
    state.session_controller = _make_mock_session_controller("test_session")
    provider = OpenCodeInputProvider(state=state, session_id="test_session")

    # Single property schema (should still use multi-question handler per Task 4)
    schema = {
        "type": "object",
        "properties": {
            "q0": {
                "type": "string",
                "enum": ["yes", "no"],
                "title": "Confirm",
                "description": "Proceed?",
            },
        },
    }
    params = types.ElicitRequestFormParams(message="Confirm action", requestedSchema=schema)

    task = asyncio.create_task(provider.get_elicitation(params))
    await asyncio.sleep(0.1)

    session = state.session_controller.get_session("test_session")
    assert len(session.pending_questions) == 1
    question_id = next(iter(session.pending_questions.keys()))
    pending = session.pending_questions[question_id]
    assert question_id.startswith("que_")

    # Single question in multi-question format
    assert len(pending.questions) == 1
    assert pending.questions[0].question == "Proceed?"

    # Resolve
    provider.resolve_question(question_id, [["yes"]])
    result = await task

    assert isinstance(result, types.ElicitResult)
    assert result.action == "accept"
    assert result.content == {"q0": "yes"}


async def test_multi_question_event_structure():
    """Test that SSE QuestionAskedEvent has correct structure for multi-questions."""
    from wolfharness_server.opencode_server.models.events import QuestionAskedEvent
    from wolfharness_server.opencode_server.models.question import QuestionInfo, QuestionOption

    # Create a QuestionsAskedEvent with multiple questions
    questions = [
        QuestionInfo(
            question="Select your first option",
            header="First Choice",
            options=[
                QuestionOption(label="opt1", description=""),
                QuestionOption(label="opt2", description=""),
            ],
            multiple=None,
        ),
        QuestionInfo(
            question="Select features",
            header="Features",
            options=[
                QuestionOption(label="val1", description=""),
                QuestionOption(label="val2", description=""),
            ],
            multiple=True,
        ),
    ]

    event = QuestionAskedEvent.create(
        request_id="test-req-123",
        session_id="test-session",
        questions=questions,
    )

    # Verify event structure
    assert event.type == "question.asked"
    assert event.properties.id == "test-req-123"
    assert event.properties.session_id == "test-session"

    # Verify questions array
    assert len(event.properties.questions) == 2

    # First question
    q1 = event.properties.questions[0]
    assert q1.question == "Select your first option"
    assert q1.header == "First Choice"
    assert q1.multiple is None
    assert len(q1.options) == 2
    assert q1.options[0].label == "opt1"

    # Second question
    q2 = event.properties.questions[1]
    assert q2.question == "Select features"
    assert q2.header == "Features"
    assert q2.multiple is True
    assert len(q2.options) == 2
    assert q2.options[0].label == "val1"

    # Verify tool is None (not passed)
    assert event.properties.tool is None


async def test_multi_question_max_limit():
    """Test that multi-questions are capped at 10."""
    mock_agent = Mock()
    mock_agent.agent_pool = None
    mock_agent.host_context = None
    state = ServerState(working_dir="/tmp", agent=mock_agent)
    state.session_controller = _make_mock_session_controller("test_session")
    provider = OpenCodeInputProvider(state=state, session_id="test_session")

    # Create schema with 12 properties (exceeds max)
    properties = {
        f"q{i}": {"type": "string", "enum": ["a", "b"], "title": f"Q{i}"} for i in range(12)
    }
    schema = {"type": "object", "properties": properties}
    params = types.ElicitRequestFormParams(message="Many questions", requestedSchema=schema)

    task = asyncio.create_task(provider.get_elicitation(params))
    await asyncio.sleep(0.1)

    session = state.session_controller.get_session("test_session")
    question_id = next(iter(session.pending_questions.keys()))
    pending = session.pending_questions[question_id]

    # Should be limited to 10 questions
    assert len(pending.questions) == 10

    # Clean up
    pending.future.cancel()
    await task


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# =============================================================================
# --- Merged from test_question_session_controller.py ---
# =============================================================================


@pytest.fixture
def mock_pool():
    """Create a mock AgentPool."""
    pool = Mock()
    pool.main_agent = Mock()
    pool.main_agent.name = "test_agent"
    pool.manifest = Mock()
    pool.manifest.agents = {}
    pool.mcp = Mock()
    pool.mcp.get_aggregating_provider = Mock(return_value=Mock())
    pool.skill_capabilities = []
    pool._config_file_path = None
    return pool


@pytest.fixture
def session_controller(mock_pool):
    """Create a SessionController with a mock pool."""
    return SessionController(pool=mock_pool)


async def _create_session_state(
    controller: SessionController,
    session_id: str,
) -> SessionState:
    """Helper to create a session in the controller."""
    session, _was_created = await controller.get_or_create_session(session_id)
    return session


def _make_pending_question(
    session_id: str,
    question_id: str,
    future: asyncio.Future[list[list[str]]] | None = None,
) -> PendingQuestion:
    """Create a PendingQuestion for testing."""
    from wolfharness_server.opencode_server.models.question import QuestionInfo, QuestionOption

    if future is None:
        future = asyncio.get_event_loop().create_future()
    return PendingQuestion(
        session_id=session_id,
        questions=[
            QuestionInfo(
                question="Test question?",
                header="Test",
                options=[QuestionOption(label="yes", description="")],
            )
        ],
        future=future,
    )


class TestSessionControllerPendingQuestions:
    """Tests for SessionController question management."""

    @pytest.mark.asyncio
    async def test_list_pending_questions_aggregates_across_sessions(
        self,
        session_controller: SessionController,
    ) -> None:
        """list_pending_questions should aggregate from all sessions."""
        session_a = await _create_session_state(session_controller, "session_a")
        session_b = await _create_session_state(session_controller, "session_b")

        future_a = asyncio.get_event_loop().create_future()
        future_b = asyncio.get_event_loop().create_future()
        session_a.pending_questions["q1"] = _make_pending_question("session_a", "q1", future_a)
        session_b.pending_questions["q2"] = _make_pending_question("session_b", "q2", future_b)

        result = session_controller.list_pending_questions()

        assert len(result) == 2
        ids = {getattr(q, "session_id", None) for q in result}
        assert ids == {"session_a", "session_b"}

    @pytest.mark.asyncio
    async def test_list_pending_questions_returns_empty_when_none(
        self,
        session_controller: SessionController,
    ) -> None:
        """list_pending_questions should return empty list when no questions."""
        result = session_controller.list_pending_questions()
        assert result == []

    @pytest.mark.asyncio
    async def test_cancel_all_pending_questions_cancels_across_sessions(
        self,
        session_controller: SessionController,
    ) -> None:
        """cancel_all_pending_questions should cancel all pending question futures."""
        session_a = await _create_session_state(session_controller, "session_a")
        session_b = await _create_session_state(session_controller, "session_b")

        future_a = asyncio.get_event_loop().create_future()
        future_b = asyncio.get_event_loop().create_future()
        session_a.pending_questions["q1"] = _make_pending_question("session_a", "q1", future_a)
        session_b.pending_questions["q2"] = _make_pending_question("session_b", "q2", future_b)

        cancelled = session_controller.cancel_all_pending_questions()

        assert sorted(cancelled) == ["q1", "q2"]
        assert future_a.cancelled()
        assert future_b.cancelled()

    @pytest.mark.asyncio
    async def test_cancel_all_pending_questions_skips_done_futures(
        self,
        session_controller: SessionController,
    ) -> None:
        """cancel_all_pending_questions should skip futures that are already done."""
        session_a = await _create_session_state(session_controller, "session_a")

        future_done = asyncio.get_event_loop().create_future()
        future_done.set_result([["yes"]])
        future_pending = asyncio.get_event_loop().create_future()

        session_a.pending_questions["q_done"] = _make_pending_question(
            "session_a", "q_done", future_done
        )
        session_a.pending_questions["q_pending"] = _make_pending_question(
            "session_a", "q_pending", future_pending
        )

        cancelled = session_controller.cancel_all_pending_questions()

        assert cancelled == ["q_pending"]
        assert not future_done.cancelled()
        assert future_pending.cancelled()

    @pytest.mark.asyncio
    async def test_cancel_session_pending_questions_targets_one_session(
        self,
        session_controller: SessionController,
    ) -> None:
        """cancel_session_pending_questions should only cancel for the specified session."""
        session_a = await _create_session_state(session_controller, "session_a")
        session_b = await _create_session_state(session_controller, "session_b")

        future_a = asyncio.get_event_loop().create_future()
        future_b = asyncio.get_event_loop().create_future()
        session_a.pending_questions["q1"] = _make_pending_question("session_a", "q1", future_a)
        session_b.pending_questions["q2"] = _make_pending_question("session_b", "q2", future_b)

        cancelled = session_controller.cancel_session_pending_questions("session_a")

        assert cancelled == ["q1"]
        assert future_a.cancelled()
        assert not future_b.cancelled()

    @pytest.mark.asyncio
    async def test_cancel_session_pending_questions_returns_empty_for_missing_session(
        self,
        session_controller: SessionController,
    ) -> None:
        """cancel_session_pending_questions should return empty for unknown session."""
        cancelled = session_controller.cancel_session_pending_questions("nonexistent")
        assert cancelled == []


class TestQuestionRoutesViaSessionController:
    """Tests for question routes reading from SessionState via SessionController."""

    @pytest.mark.asyncio
    async def test_list_questions_via_session_controller(
        self,
        session_controller: SessionController,
    ) -> None:
        """list_questions should read from SessionState when session_controller is set."""
        session = await _create_session_state(session_controller, "test_session")
        future = asyncio.get_event_loop().create_future()
        session.pending_questions["q1"] = _make_pending_question("test_session", "q1", future)

        mock_agent = Mock()
        mock_agent.agent_pool = None
        mock_agent.host_context = None
        state = ServerState(working_dir="/tmp", agent=mock_agent)
        state.session_controller = session_controller

        result = await list_questions(state)

        assert len(result) == 1
        assert result[0].id == "q1"
        assert result[0].session_id == "test_session"

    @pytest.mark.asyncio
    async def test_list_questions_no_session_controller_returns_empty(
        self,
        session_controller: SessionController,
    ) -> None:
        """list_questions should return empty list when no session_controller."""
        mock_agent = Mock()
        mock_agent.agent_pool = None
        mock_agent.host_context = None
        state = ServerState(working_dir="/tmp", agent=mock_agent)
        # No session_controller set, so list_questions should return empty

        result = await list_questions(state)

        assert result == []

    @pytest.mark.asyncio
    async def test_reply_to_question_via_session_controller(
        self,
        session_controller: SessionController,
    ) -> None:
        """reply_to_question should resolve questions stored on SessionState."""
        session = await _create_session_state(session_controller, "test_session")
        future = asyncio.get_event_loop().create_future()
        session.pending_questions["q1"] = _make_pending_question("test_session", "q1", future)

        mock_agent = Mock()
        mock_agent.agent_pool = None
        mock_agent.host_context = None
        state = ServerState(working_dir="/tmp", agent=mock_agent)
        state.session_controller = session_controller
        session.input_provider = OpenCodeInputProvider(state, "test_session")
        broadcast_calls = []

        async def _mock_broadcast(event):
            broadcast_calls.append(event)

        state.broadcast_event = _mock_broadcast

        reply = QuestionReply(answers=[["yes"]])
        result = await reply_to_question("q1", reply, state)

        assert result is True
        assert future.done()
        assert future.result() == [["yes"]]

    @pytest.mark.asyncio
    async def test_reject_question_via_session_controller(
        self,
        session_controller: SessionController,
    ) -> None:
        """reject_question should cancel questions stored on SessionState."""
        session = await _create_session_state(session_controller, "test_session")
        future = asyncio.get_event_loop().create_future()
        session.pending_questions["q1"] = _make_pending_question("test_session", "q1", future)

        mock_agent = Mock()
        mock_agent.agent_pool = None
        mock_agent.host_context = None
        state = ServerState(working_dir="/tmp", agent=mock_agent)
        state.session_controller = session_controller
        broadcast_calls = []

        async def _mock_broadcast(event):
            broadcast_calls.append(event)

        state.broadcast_event = _mock_broadcast

        result = await reject_question("q1", state)

        assert result is True
        assert future.cancelled()
        assert "q1" not in session.pending_questions


class TestInputProviderStoresQuestionsOnSessionState:
    """Tests that OpenCodeInputProvider stores questions on SessionState."""

    @pytest.mark.asyncio
    async def test_input_provider_stores_question_on_session_state(
        self,
        session_controller: SessionController,
    ) -> None:
        """When session_controller is available, questions go to SessionState."""
        await _create_session_state(session_controller, "test_session")

        mock_agent = Mock()
        mock_agent.agent_pool = None
        mock_agent.host_context = None
        state = ServerState(working_dir="/tmp", agent=mock_agent)
        state.session_controller = session_controller
        broadcast_calls = []

        async def _mock_broadcast(event):
            broadcast_calls.append(event)

        state.broadcast_event = _mock_broadcast

        provider = OpenCodeInputProvider(state, "test_session")

        from mcp import types

        schema = {"type": "string", "enum": ["a", "b"]}
        params = types.ElicitRequestFormParams(message="Pick one?", requestedSchema=schema)

        task = asyncio.create_task(provider.get_elicitation(params))
        await asyncio.sleep(0.1)

        # Question should be on SessionState
        session = session_controller.get_session("test_session")
        assert session is not None
        assert len(session.pending_questions) == 1

        # Clean up
        question_id = next(iter(session.pending_questions.keys()))
        provider.resolve_question(question_id, [["a"]])
        await task

    @pytest.mark.asyncio
    async def test_input_provider_no_fallback_to_server_state(
        self,
        session_controller: SessionController,
    ) -> None:
        """When no session_controller, questions use instance-level fallback dict."""
        mock_agent = Mock()
        mock_agent.agent_pool = None
        mock_agent.host_context = None
        state = ServerState(working_dir="/tmp", agent=mock_agent)
        broadcast_calls = []

        async def _mock_broadcast(event):
            broadcast_calls.append(event)

        state.broadcast_event = _mock_broadcast

        provider = OpenCodeInputProvider(state, "test_session")

        from mcp import types

        schema = {"type": "string", "enum": ["a", "b"]}
        params = types.ElicitRequestFormParams(message="Pick one?", requestedSchema=schema)

        task = asyncio.create_task(provider.get_elicitation(params))
        await asyncio.sleep(0.1)

        # Question is stored on the provider's fallback dict, not on ServerState
        assert len(provider.get_pending_questions()) == 1
        # ServerState itself should not have a session_controller set
        assert state.session_controller is None

        # Clean up via the provider's fallback dict
        question_id = next(iter(provider._fallback_pending_questions.keys()))
        provider.resolve_question(question_id, [["a"]])
        await task


class TestSSEDisconnectViaSessionController:
    """Tests that SSE disconnect cancels questions via SessionController."""

    @pytest.mark.asyncio
    async def test_cancel_all_pending_questions_delegates_to_session_controller(
        self,
        session_controller: SessionController,
    ) -> None:
        """ServerState.cancel_all_pending_questions delegates to SessionController."""
        session = await _create_session_state(session_controller, "test_session")
        future = asyncio.get_event_loop().create_future()
        session.pending_questions["q1"] = _make_pending_question("test_session", "q1", future)

        mock_agent = Mock()
        mock_agent.agent_pool = None
        mock_agent.host_context = None
        state = ServerState(working_dir="/tmp", agent=mock_agent)
        state.session_controller = session_controller

        cancelled = state.cancel_all_pending_questions()

        assert cancelled == ["q1"]
        assert future.cancelled()

    @pytest.mark.asyncio
    async def test_cancel_session_pending_questions_delegates_to_session_controller(
        self,
        session_controller: SessionController,
    ) -> None:
        """ServerState.cancel_session_pending_questions delegates to SessionController."""
        session = await _create_session_state(session_controller, "test_session")
        future = asyncio.get_event_loop().create_future()
        session.pending_questions["q1"] = _make_pending_question("test_session", "q1", future)

        mock_agent = Mock()
        mock_agent.agent_pool = None
        mock_agent.host_context = None
        state = ServerState(working_dir="/tmp", agent=mock_agent)
        state.session_controller = session_controller

        cancelled = state.cancel_session_pending_questions("test_session")

        assert cancelled == ["q1"]
        assert future.cancelled()
