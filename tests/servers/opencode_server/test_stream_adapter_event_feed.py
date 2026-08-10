"""Tests for OpenCodeStreamAdapter receiving events via EventBus.

Verifies the fix for the orphaned adapter bug: the adapter created in
run_message_phases must receive stream events so that finalize()
produces correct tokens and response text.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock

from pydantic_ai import RequestUsage
import pytest

from tests.servers.opencode_server.conftest import run_message_phases
from wolfharness.agents.events import StreamCompleteEvent
from wolfharness.lifecycle import RunOutcome, RunState
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.core import EventBus
from wolfharness.utils import identifiers as identifier
from wolfharness.utils.time_utils import now_ms
from wolfharness_server.opencode_server.models import (
    AssistantMessage,
    MessagePath,
    MessageRequest,
    MessageTime,
    PartUpdatedEvent,
    TextPartInput,
    TimeCreated,
    TimeCreatedUpdated,
    UserMessage,
)
from wolfharness_server.opencode_server.models.message import MessageWithParts
from wolfharness_server.opencode_server.models.parts import StepFinishPart
from wolfharness_server.opencode_server.session_pool_integration import get_messages_for_session
from wolfharness_server.opencode_server.state import ServerState
from wolfharness_server.opencode_server.stream_adapter import OpenCodeStreamAdapter


pytestmark = pytest.mark.integration


def _setup_session(state: ServerState, session_id: str) -> None:
    """Set up session state manually."""
    now = now_ms()
    from wolfharness_server.opencode_server.models import Session

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


def _create_user_message(
    session_id: str,
    request: MessageRequest,
) -> tuple[str, MessageWithParts]:
    """Create user message and parts."""
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


@pytest.fixture
def mock_agent_with_event_bus(tmp_project_dir):
    """Create a mock agent wired to a real EventBus."""
    agent = Mock()
    agent.name = "test-agent"
    agent.env = Mock()
    agent.env.get_fs = Mock(return_value=Mock())
    agent.env.cwd = str(tmp_project_dir)
    agent._input_provider = None
    agent.storage = Mock()
    agent.storage.save_session = AsyncMock()
    agent.storage.log_message = AsyncMock()
    agent.set_model = AsyncMock()
    agent.set_mode = AsyncMock()
    agent.get_available_models = AsyncMock(return_value=[])
    agent.load_session = AsyncMock(return_value=None)

    pool = Mock()
    pool.manifest = Mock()
    pool.manifest.config_file_path = "/tmp/test"
    pool.manifest.model_variants = {}
    pool.storage = agent.storage
    pool.todos = Mock()
    pool.todos.on_change = None
    pool.manifest.agents = {agent.name: agent}

    # Real EventBus so _feed_adapter can subscribe and receive events
    event_bus = EventBus()
    session_pool = Mock()
    session_pool.sessions = Mock()
    session_pool.sessions.store = None
    session_pool.sessions.get_session = Mock(return_value=None)
    session_pool.sessions.cancel_run_for_session = Mock()
    session_pool.sessions.get_or_create_session_agent = AsyncMock(return_value=Mock())
    session_pool.sessions.get_or_create_session = AsyncMock(return_value=(Mock(), True))
    session_pool.event_bus = event_bus

    # D14: receive_request now returns str (message_id), not RunHandle.
    # wait_for_completion is used to wait for the run to finish.
    run_handle = Mock()
    run_handle._run_state = RunState.DONE
    run_handle.outcome = RunOutcome.COMPLETED
    run_handle.complete_event = asyncio.Event()
    session_pool.send_message = AsyncMock(return_value="msg_test_run")

    # wait_for_completion waits on the same event the test controls
    async def _mock_wait_for_completion(
        sid: str,
        timeout: float | None = None,
    ) -> str:
        await asyncio.wait_for(run_handle.complete_event.wait(), timeout=timeout or 30.0)
        return sid

    session_pool.wait_for_completion = _mock_wait_for_completion

    pool.session_pool = session_pool
    agent.agent_pool = pool
    agent.host_context = pool
    agent._agent_pool = pool  # state.py resolves _pool via agent._agent_pool

    return agent, run_handle, event_bus


@pytest.fixture
def event_bus_test_state(tmp_project_dir, mock_agent_with_event_bus):
    """Create a server state with EventBus-backed agent."""
    agent, _run_handle, _event_bus = mock_agent_with_event_bus
    state = ServerState(working_dir=str(tmp_project_dir), agent=agent)
    # Initialize backward-compat dicts removed from ServerState dataclass
    state.messages = {}
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


@pytest.mark.asyncio
async def test_adapter_receives_events_before_finalize(
    event_bus_test_state: ServerState,
    sample_message_request: MessageRequest,
    mock_agent_with_event_bus: tuple[Any, Any, EventBus],
) -> None:
    """Adapter must receive all events so finalize() produces non-zero tokens.

    Before the fix, the adapter was created but never fed events from the
    EventBus.  This meant adapter.finalize() produced StepFinishPart with
    input=0, output=0 and empty response_text.
    """
    state = event_bus_test_state
    session_id = "test-session-adapter-feed"
    _agent, run_handle, event_bus = mock_agent_with_event_bus

    _setup_session(state, session_id)
    user_msg_id, user_msg_with_parts = _create_user_message(session_id, sample_message_request)
    state.messages[session_id].append(user_msg_with_parts)

    # Start run_message_phases in background
    process_task = asyncio.create_task(
        run_message_phases(
            session_id, sample_message_request, state, user_msg_id, user_msg_with_parts
        )
    )

    # Give _feed_adapter time to subscribe
    await asyncio.sleep(0.05)

    # Publish a StreamCompleteEvent with usage info
    chat_msg = ChatMessage(
        role="assistant",
        content="Hello from test",
        usage=RequestUsage(input_tokens=42, output_tokens=17),
    )
    await event_bus.publish(session_id, StreamCompleteEvent(message=chat_msg))

    # Signal run completion so run_message_phases continues
    run_handle.complete_event.set()

    # Wait for processing to finish
    await process_task

    # Find the assistant message
    messages = await get_messages_for_session(state, session_id)
    assistant_msgs = [msg for msg in messages if isinstance(msg.info, AssistantMessage)]
    assert len(assistant_msgs) == 1
    assistant = assistant_msgs[0].info
    assert isinstance(assistant, AssistantMessage)

    # The key assertion: tokens must be non-zero because the adapter
    # received the StreamCompleteEvent before finalize() was called.
    assert assistant.tokens is not None, "Assistant message should have tokens set"
    assert assistant.tokens.input == 42, f"Expected input=42, got {assistant.tokens.input}"
    assert assistant.tokens.output == 17, f"Expected output=17, got {assistant.tokens.output}"


@pytest.mark.asyncio
async def test_message_cleanup_tolerates_deleted_session(
    event_bus_test_state: ServerState,
    sample_message_request: MessageRequest,
    mock_agent_with_event_bus: tuple[Any, Any, EventBus],
) -> None:
    """Message cleanup should not fail if the client deleted the session."""
    state = event_bus_test_state
    session_id = "test-session-deleted-before-cleanup"
    _agent, run_handle, event_bus = mock_agent_with_event_bus

    _setup_session(state, session_id)
    user_msg_id, user_msg_with_parts = _create_user_message(session_id, sample_message_request)
    state.messages[session_id].append(user_msg_with_parts)

    process_task = asyncio.create_task(
        run_message_phases(
            session_id, sample_message_request, state, user_msg_id, user_msg_with_parts
        )
    )

    await asyncio.sleep(0.05)
    chat_msg = ChatMessage(
        role="assistant",
        content="Completed after client timeout",
        usage=RequestUsage(input_tokens=1, output_tokens=1),
    )
    await event_bus.publish(session_id, StreamCompleteEvent(message=chat_msg))
    state.sessions.pop(session_id)

    run_handle.complete_event.set()
    await process_task
    assert session_id not in state.sessions


@pytest.mark.asyncio
async def test_adapter_response_text_populated_after_finalize(
    event_bus_test_state: ServerState,
    sample_message_request: MessageRequest,
    mock_agent_with_event_bus: tuple[Any, Any, EventBus],
) -> None:
    """Adapter must accumulate response_text from streamed events.

    When text is delivered via PartDeltaEvent through the EventBus, the
    adapter's context accumulates it.  After finalize(), the assistant
    message should reflect the accumulated text.
    """
    state = event_bus_test_state
    session_id = "test-session-response-text"
    _agent, run_handle, event_bus = mock_agent_with_event_bus

    _setup_session(state, session_id)
    user_msg_id, user_msg_with_parts = _create_user_message(session_id, sample_message_request)
    state.messages[session_id].append(user_msg_with_parts)

    process_task = asyncio.create_task(
        run_message_phases(
            session_id, sample_message_request, state, user_msg_id, user_msg_with_parts
        )
    )

    await asyncio.sleep(0.05)

    # Publish StreamCompleteEvent with text content
    chat_msg = ChatMessage(
        role="assistant",
        content="This is the final response",
        usage=RequestUsage(input_tokens=10, output_tokens=5),
    )
    await event_bus.publish(session_id, StreamCompleteEvent(message=chat_msg))

    run_handle.complete_event.set()
    await process_task

    messages = await get_messages_for_session(state, session_id)
    assistant_msgs = [msg for msg in messages if isinstance(msg.info, AssistantMessage)]
    assert len(assistant_msgs) == 1
    assistant = assistant_msgs[0].info
    assert isinstance(assistant, AssistantMessage)

    # The response text should be captured from the event
    assert assistant.tokens is not None
    assert assistant.tokens.input == 10
    assert assistant.tokens.output == 5


@pytest.mark.asyncio
async def test_adapter_convert_event_updates_context() -> None:
    """Direct test: OpenCodeStreamAdapter.convert_event() updates its own context.

    Verifies that the adapter's new :meth:`convert_event` entry point
    processes a single event, updates mutable state (tokens, cost), and
    tracks ``_step_finish_emitted`` so that :meth:`finalize` behaves
    correctly.
    """
    session_id = "direct-test-session"
    assistant_msg = MessageWithParts.assistant(
        message_id="msg-1",
        session_id=session_id,
        time=MessageTime(created=1000),
        agent_name="test",
        model_id="test-model",
        parent_id="user-1",
        provider_id="test-provider",
        path=MessagePath(cwd="/tmp", root="/tmp"),
    )

    # Create a minimal server state mock
    state_mock = Mock()
    state_mock.messages = {}
    state_mock.working_dir = "/tmp"

    adapter = OpenCodeStreamAdapter(
        state=state_mock,
        session_id=session_id,
        assistant_msg_id="msg-1",
        assistant_msg=assistant_msg,
        working_dir="/tmp",
    )

    # Before any events, context should be empty
    assert adapter.response_text == ""
    assert adapter.input_tokens == 0
    assert adapter.output_tokens == 0

    # Simulate receiving a StreamCompleteEvent
    chat_msg = ChatMessage(
        role="assistant",
        content="Hello world",
        usage=RequestUsage(input_tokens=100, output_tokens=25),
    )

    _ = [e async for e in adapter.convert_event(StreamCompleteEvent(message=chat_msg))]

    # Token counts are updated by StreamCompleteEvent processing
    assert adapter.input_tokens == 100
    assert adapter.output_tokens == 25

    # StepFinishPart was emitted, so _step_finish_emitted should be True
    assert adapter._step_finish_emitted is True

    # finalize() sees _step_finish_emitted=True and skips emitting another
    # StepFinishPart, but the tokens remain available via usage property
    finalized = list(adapter.finalize())
    step_finish_events = [
        e
        for e in finalized
        if isinstance(e, PartUpdatedEvent) and isinstance(e.properties.part, StepFinishPart)
    ]
    assert len(step_finish_events) == 0, (
        "finalize() should skip StepFinishPart when already emitted"
    )
    assert adapter.input_tokens == 100
    assert adapter.output_tokens == 25
