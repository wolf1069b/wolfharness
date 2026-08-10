"""Red flag test: auto-resume events must create message in state.messages.

When a background task completes and inject_prompt triggers auto-resume,
the agent processes the injected message and generates stream events.
These events are consumed by _event_consumer_loop and converted to
OpenCode SSE events. But if the assistant_msg is not registered in
state.messages, the TUI cannot display the parts because the message
store lacks the corresponding message entry.

REGRESSION TEST:
  Previously, _event_consumer_loop created an assistant_msg in its
  EventProcessorContext but NEVER added it to state.messages or broadcast
  a MessageUpdatedEvent. The TUI received PartUpdatedEvents but could
  not display them because the message was missing from the message store.

EXPECTED BEHAVIOR:
  After _event_consumer_loop processes auto-resume events, the
  assistant_msg should exist in state.messages[session_id] and a
  MessageUpdatedEvent should have been broadcast so the TUI can
  render the message and its parts.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock

from pydantic_ai.messages import (
    PartDeltaEvent as PydanticPartDeltaEvent,
    PartStartEvent,
    TextPart as PydanticTextPart,
    TextPartDelta,
)
import pytest

from wolfharness.agents.events import StreamCompleteEvent
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.core import SessionPool
from wolfharness_server.opencode_server.models import (
    AssistantMessage,
    MessageUpdatedEvent,
    PartUpdatedEvent,
)
from wolfharness_server.opencode_server.session_pool_integration import (
    OpenCodeSessionPoolIntegration,
    get_messages_for_session,
)
from wolfharness_server.opencode_server.state import ServerState


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.fixture
def mock_agent_pool() -> Mock:
    """Create a mock AgentPool for SessionPool construction."""
    from wolfharness.agents.events import RunStartedEvent, StreamCompleteEvent
    from wolfharness.messaging.messages import ChatMessage

    pool = Mock()
    pool.main_agent = Mock()
    pool.main_agent.name = "test-agent"
    pool.manifest = Mock()
    pool.manifest.agents = {}
    pool._config_file_path = None

    async def _mock_stream_events(*args: Any, **kwargs: Any) -> Any:
        """Yield a minimal run event sequence for testing."""
        session_id = kwargs.get("session_id", "unknown")
        run_id = "run-mock-001"
        yield RunStartedEvent(session_id=session_id, run_id=run_id)
        yield StreamCompleteEvent(
            message=ChatMessage(content="test response", role="assistant"),
        )

    mock_agent = Mock()
    mock_agent._stream_events = _mock_stream_events
    mock_agent._input_provider = None
    mock_agent.conversation = Mock()
    mock_agent.conversation.add_chat_messages = Mock()
    pool.get_agent = Mock(return_value=mock_agent)

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
) -> AsyncIterator[SessionPool]:
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
def server_state(tmp_path: Any) -> ServerState:
    """Create a minimal ServerState for testing."""
    agent = Mock()
    agent.model_name = None  # resolve_default_model_info() fallback
    agent.name = "test-agent"
    agent.storage = Mock()
    state = ServerState(working_dir=str(tmp_path), agent=agent)
    # Initialize backward-compat dicts removed from ServerState dataclass
    state.messages = {}
    # No session_pool_integration — this test uses OpenCodeSessionPoolIntegration
    # directly and creates its own integration instance.
    return state


@pytest.mark.asyncio
async def test_auto_resume_events_create_message_in_state(
    session_pool: SessionPool,
    server_state: ServerState,
) -> None:
    """Auto-resume events must create message in state.messages so TUI can display.

    This test simulates the exact scenario described in the bug report:
    1. Background task completes
    2. inject_prompt triggers auto-resume
    3. Agent processes injected message and generates stream events
    4. _event_consumer_loop converts events to OpenCode SSE events
    5. TUI should be able to display the response

    FAILURE MODE:
      If _event_consumer_loop does not register assistant_msg in
      state.messages or broadcast MessageUpdatedEvent, the TUI
      receives PartUpdatedEvents but cannot render them because
      the message store lacks the message entry.
    """
    integration = OpenCodeSessionPoolIntegration(
        session_pool=session_pool,
        server_state=server_state,
    )

    session_id = "test-autoresume-session"

    # Capture broadcast events
    broadcast_events: list[Any] = []
    original_broadcast = server_state.broadcast_event

    async def capture_broadcast(event: Any) -> None:
        broadcast_events.append(event)
        await original_broadcast(event)

    server_state.broadcast_event = capture_broadcast  # type: ignore[method-assign]

    # Create session (starts _event_consumer_loop)
    await integration.create_session(
        session_id=session_id,
        agent_name="test-agent",
    )

    # Give consumer time to start
    await asyncio.sleep(0.05)

    # Simulate auto-resume event sequence (what happens after inject_prompt)
    # Step 1: PartStartEvent (text starts)
    await session_pool.event_bus.publish(
        session_id,
        PartStartEvent(index=0, part=PydanticTextPart(content="Background task")),
    )

    # Step 2: PartDeltaEvent (text continues)
    await session_pool.event_bus.publish(
        session_id,
        PydanticPartDeltaEvent(index=0, delta=TextPartDelta(content_delta=" completed!")),
    )

    # Step 3: StreamCompleteEvent (turn finishes)
    await session_pool.event_bus.publish(
        session_id,
        StreamCompleteEvent(
            message=ChatMessage(content="Background task completed!", role="assistant"),
        ),
    )

    # Wait for consumer to process all events
    await asyncio.sleep(0.1)

    # ASSERTION 1: MessageUpdatedEvent must be broadcast
    # so the TUI knows the message exists
    message_updated_events = [e for e in broadcast_events if isinstance(e, MessageUpdatedEvent)]
    assert len(message_updated_events) > 0, (
        "No MessageUpdatedEvent was broadcast by _event_consumer_loop. "
        "The TUI cannot display the auto-resume response because it "
        "does not know the message exists. "
        f"Events broadcast: {[type(e).__name__ for e in broadcast_events]}"
    )

    # ASSERTION 2: The message must exist in state.messages
    session_messages = await get_messages_for_session(server_state, session_id)
    auto_resume_messages = [
        msg for msg in session_messages if isinstance(msg.info, AssistantMessage)
    ]
    assert len(auto_resume_messages) > 0, (
        "No AssistantMessage was added to state.messages for the auto-resume turn. "
        "The TUI's message store lacks the message entry, so parts cannot be rendered. "
        f"Messages in state: {[type(m.info).__name__ for m in session_messages]}"
    )

    # ASSERTION 3: PartUpdatedEvent must be broadcast
    # (for the text content to be displayed)
    part_updated_events = [e for e in broadcast_events if isinstance(e, PartUpdatedEvent)]
    assert len(part_updated_events) > 0, (
        "No PartUpdatedEvent was broadcast for the auto-resume text content. "
        "The TUI has no parts to render even if the message exists."
    )

    # Clean up
    await integration._stop_event_consumer(session_id)
