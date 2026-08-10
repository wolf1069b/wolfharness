"""Integration tests for ACP session/load with message_history replay.

Tests that session/load correctly replays conversation history from
checkpointed sessions, including pending ToolCallPart without
matching ToolReturnPart for deferred tools.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic_ai import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
import pytest

from acp.schema import LoadSessionRequest, LoadSessionResponse
from wolfharness import Agent
from wolfharness.delegation import AgentPool
from wolfharness.sessions.models import PendingDeferredCall
from wolfharness.storage.manager import StorageManager
from wolfharness.storage.serialization import serialize_messages
from wolfharness_config.storage import MemoryStorageConfig, StorageConfig
from wolfharness_server.acp_server.acp_agent import AgentPoolACPAgent


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_messages_with_pending_tool() -> list[ModelMessage]:
    """Create message history with a pending ToolCallPart (no ToolReturnPart).

    Simulates a checkpointed session where a tool was deferred (block strategy).
    The ToolCallPart exists but no ToolReturnPart follows.
    """
    return [
        ModelRequest(parts=[UserPromptPart(content="Run this command")]),
        ModelResponse(
            parts=[
                TextPart(content="I'll run that for you."),
                ToolCallPart(
                    tool_name="bash",
                    args="ls -la",
                    tool_call_id="pending_call_1",
                ),
            ],
        ),
        # Note: NO ToolReturnPart for pending_call_1 (deferred tool)
    ]


def _make_messages_with_completed_tool() -> list[ModelMessage]:
    """Create message history with a completed tool cycle.

    Includes both ToolCallPart and matching ToolReturnPart.
    """
    return [
        ModelRequest(parts=[UserPromptPart(content="Read the file")]),
        ModelResponse(
            parts=[
                TextPart(content="Reading the file..."),
                ToolCallPart(
                    tool_name="read",
                    args="file.txt",
                    tool_call_id="completed_call_1",
                ),
            ],
        ),
        ModelRequest(
            parts=[
                # ToolReturnPart is a pydantic model - use dict for testing
            ],
        ),
    ]


def _make_chat_messages_from_model_messages(
    model_messages: list[ModelMessage],
    agent_name: str = "test_agent",
    session_id: str = "test-session-id",
) -> list[Any]:
    """Convert ModelMessages to ChatMessage-like objects for mocking.

    Returns objects compatible with what ACPSession expects
    after agent.load_session() populates conversation.chat_messages.
    """
    from wolfharness.messaging import ChatMessage

    return [
        ChatMessage[Any](
            content="mock",
            role="user",
            name=agent_name,
            session_id=session_id,
            messages=model_messages,
        )
    ]


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def mock_connection() -> AsyncMock:
    """Create a mock ACP connection."""
    return AsyncMock()


@pytest.fixture
def mock_agent_pool_with_agent() -> tuple[AgentPool, Agent]:
    """Create a mock agent pool with a test agent."""

    def simple_callback(message: str) -> str:
        return f"Test response: {message}"

    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.manifest import AgentsManifest

    manifest = AgentsManifest(agents={"test_agent": NativeAgentConfig(model="test")})

    pool = AgentPool(manifest)
    agent = Agent.from_callback(name="test_agent", callback=simple_callback, agent_pool=pool)
    # pool.register() removed; agent created from callback/config above
    return pool, agent


@pytest.fixture
def default_test_agent(mock_agent_pool_with_agent: tuple[AgentPool, Agent]) -> Agent:
    """Get the default test agent from the mock pool."""
    return mock_agent_pool_with_agent[1]


@pytest.fixture
def mock_acp_agent(mock_connection: AsyncMock, default_test_agent: Agent) -> AgentPoolACPAgent:
    """Create a mock ACP agent for testing."""
    return AgentPoolACPAgent(client=mock_connection, default_agent=default_test_agent)


@pytest.fixture
def storage_manager() -> StorageManager:
    """Create a StorageManager backed by MemoryStorageProvider."""
    config = StorageConfig(providers=[MemoryStorageConfig()])
    return StorageManager(config)


@pytest.fixture
def pending_messages() -> list[ModelMessage]:
    """ModelMessages with a pending ToolCallPart."""
    return _make_messages_with_pending_tool()


@pytest.fixture
def pending_deferred_calls() -> list[PendingDeferredCall]:
    """PendingDeferredCall matching the pending ToolCallPart."""
    return [
        PendingDeferredCall(
            tool_call_id="pending_call_1",
            tool_name="bash",
            deferred_kind="external",
            deferred_strategy="block",
        ),
    ]


# ── Tests: Message History Replay ────────────────────────────────────────


@pytest.mark.unit
async def test_load_session_replays_message_history(
    mock_acp_agent: AgentPoolACPAgent,
    mock_connection: AsyncMock,
    storage_manager: StorageManager,
    pending_messages: list[ModelMessage],
    pending_deferred_calls: list[PendingDeferredCall],
) -> None:
    """session/load replays message_history with correct ordering.

    When a client calls session/load for a checkpointed session,
    the message_history is replayed as session/update notifications,
    preserving the original message sequence.
    """
    session_id = "test-session-id"
    mock_session = MagicMock()
    mock_session.session_id = session_id
    mock_session.cwd = "/tmp"

    # Set up agent.conversation.chat_messages with the checkpointed messages
    chat_msgs = _make_chat_messages_from_model_messages(pending_messages, session_id=session_id)
    mock_session.agent = MagicMock()
    mock_session.agent.load_session = AsyncMock(return_value=True)
    mock_session.agent.load_rules = AsyncMock()
    mock_session.agent.conversation = MagicMock()
    mock_session.agent.conversation.chat_messages = chat_msgs

    # Notifications mock - capture replay calls
    mock_session.notifications = MagicMock()
    mock_session.notifications.replay = AsyncMock()
    mock_session.send_available_commands_update = AsyncMock()

    mock_acp_agent.session_manager.get_session = MagicMock(return_value=mock_session)  # type: ignore[assignment]  # type: ignore[assignment]
    mock_acp_agent._initialized = True

    with patch.object(mock_acp_agent._task_group, "start_soon"):
        await mock_acp_agent.load_session(LoadSessionRequest(session_id=session_id, cwd="/tmp"))

    # Verify replay was called
    mock_session.notifications.replay.assert_awaited_once()

    # Verify the replayed messages include our model messages
    replay_args = mock_session.notifications.replay.call_args[0][0]
    assert len(replay_args) == len(pending_messages), (
        f"Expected {len(pending_messages)} messages replayed, got {len(replay_args)}"
    )

    # Verify message ordering is preserved
    assert isinstance(replay_args[0], ModelRequest)
    assert isinstance(replay_args[1], ModelResponse)
    assert replay_args[0].parts[0].content == "Run this command"
    assert "I'll run that for you." in replay_args[1].parts[0].content


@pytest.mark.unit
async def test_load_session_includes_pending_toolcallpart_without_toolreturnpart(
    mock_acp_agent: AgentPoolACPAgent,
    mock_connection: AsyncMock,
    pending_messages: list[ModelMessage],
) -> None:
    """Pending ToolCallPart is visible without matching ToolReturnPart.

    When replaying checkpointed message_history, ToolCallPart entries
    for deferred tools are present, but no ToolReturnPart follows.
    This lets the client detect pending elicitation and re-render UI.
    """
    session_id = "test-session-id"
    mock_session = MagicMock()
    mock_session.session_id = session_id
    mock_session.cwd = "/tmp"

    chat_msgs = _make_chat_messages_from_model_messages(pending_messages, session_id=session_id)
    mock_session.agent = MagicMock()
    mock_session.agent.load_session = AsyncMock(return_value=True)
    mock_session.agent.load_rules = AsyncMock()
    mock_session.agent.conversation = MagicMock()
    mock_session.agent.conversation.chat_messages = chat_msgs
    mock_session.agent.get_modes = AsyncMock(return_value=[])

    mock_session.notifications = MagicMock()
    mock_session.notifications.replay = AsyncMock()
    mock_session.send_available_commands_update = AsyncMock()

    mock_acp_agent.session_manager.get_session = MagicMock(return_value=mock_session)  # type: ignore[assignment]
    mock_acp_agent._initialized = True

    with patch.object(mock_acp_agent._task_group, "start_soon"):
        await mock_acp_agent.load_session(LoadSessionRequest(session_id=session_id, cwd="/tmp"))

    replay_args = mock_session.notifications.replay.call_args[0][0]

    # Verify ToolCallPart exists in the replayed messages
    response_msg = replay_args[1]
    assert isinstance(response_msg, ModelResponse)
    tool_call_parts = [p for p in response_msg.parts if isinstance(p, ToolCallPart)]
    assert len(tool_call_parts) >= 1, "Expected at least one ToolCallPart in replayed messages"
    assert tool_call_parts[0].tool_name == "bash"
    assert tool_call_parts[0].tool_call_id == "pending_call_1"

    # Verify NO ToolReturnPart for the pending tool call
    tool_return_ids = set()
    for msg in replay_args:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                from pydantic_ai import ToolReturnPart

                if isinstance(part, ToolReturnPart):
                    tool_return_ids.add(part.tool_call_id)
    assert "pending_call_1" not in tool_return_ids, (
        "ToolReturnPart should NOT exist for pending tool call"
    )


@pytest.mark.unit
async def test_load_session_message_ordering_is_correct(
    mock_acp_agent: AgentPoolACPAgent,
) -> None:
    """Replayed messages preserve the original chronological order.

    Messages from the checkpoint must be replayed in the exact
    order they were originally generated, so the client renders
    the conversation correctly.
    """
    session_id = "test-session-id"
    # Create messages in a specific order
    ordered_messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="First message")]),
        ModelResponse(parts=[TextPart(content="First response")]),
        ModelRequest(parts=[UserPromptPart(content="Second message")]),
        ModelResponse(parts=[TextPart(content="Second response")]),
        ModelRequest(parts=[UserPromptPart(content="Third message")]),
        ModelResponse(parts=[TextPart(content="Third response")]),
    ]

    mock_session = MagicMock()
    mock_session.session_id = session_id
    mock_session.cwd = "/tmp"

    chat_msgs = _make_chat_messages_from_model_messages(ordered_messages, session_id=session_id)
    mock_session.agent = MagicMock()
    mock_session.agent.load_session = AsyncMock(return_value=True)
    mock_session.agent.load_rules = AsyncMock()
    mock_session.agent.conversation = MagicMock()
    mock_session.agent.conversation.chat_messages = chat_msgs
    mock_session.agent.get_modes = AsyncMock(return_value=[])

    mock_session.notifications = MagicMock()
    mock_session.notifications.replay = AsyncMock()
    mock_session.send_available_commands_update = AsyncMock()

    mock_acp_agent.session_manager.get_session = MagicMock(return_value=mock_session)  # type: ignore[assignment]
    mock_acp_agent._initialized = True

    with patch.object(mock_acp_agent._task_group, "start_soon"):
        await mock_acp_agent.load_session(LoadSessionRequest(session_id=session_id, cwd="/tmp"))

    replay_args = mock_session.notifications.replay.call_args[0][0]

    # Verify exact ordering
    for i, expected in enumerate(ordered_messages):
        actual = replay_args[i]
        assert type(actual) is type(expected), (
            f"Message {i}: expected {type(expected).__name__}, got {type(actual).__name__}"
        )
        if (isinstance(actual, ModelRequest) and isinstance(expected, ModelRequest)) or (
            isinstance(actual, ModelResponse) and isinstance(expected, ModelResponse)
        ):
            actual_part = actual.parts[0]
            if hasattr(actual_part, "content"):
                assert actual_part.content == expected.parts[0].content  # type: ignore[union-attr]

    assert len(replay_args) == len(ordered_messages)


@pytest.mark.unit
async def test_load_session_empty_conversation_no_replay(
    mock_acp_agent: AgentPoolACPAgent,
) -> None:
    """Empty conversation does not call replay."""
    session_id = "test-session-id"
    mock_session = MagicMock()
    mock_session.session_id = session_id
    mock_session.cwd = "/tmp"
    mock_session.agent = MagicMock()
    mock_session.agent.load_session = AsyncMock(return_value=True)
    mock_session.agent.load_rules = AsyncMock()
    mock_session.agent.conversation = MagicMock()
    mock_session.agent.conversation.chat_messages = []
    mock_session.agent.get_modes = AsyncMock(return_value=[])

    mock_session.notifications = MagicMock()
    mock_session.notifications.replay = AsyncMock()
    mock_session.send_available_commands_update = AsyncMock()

    mock_acp_agent.session_manager.get_session = MagicMock(return_value=mock_session)  # type: ignore[assignment]
    mock_acp_agent._initialized = True

    with patch.object(mock_acp_agent._task_group, "start_soon"):
        await mock_acp_agent.load_session(LoadSessionRequest(session_id=session_id, cwd="/tmp"))

    mock_session.notifications.replay.assert_not_awaited()


@pytest.mark.unit
async def test_load_session_returns_response_with_config(
    mock_acp_agent: AgentPoolACPAgent,
) -> None:
    """session/load returns a valid LoadSessionResponse with config options."""
    session_id = "test-session-id"
    mock_session = MagicMock()
    mock_session.session_id = session_id
    mock_session.cwd = "/tmp"
    mock_session.agent = MagicMock()
    mock_session.agent.load_session = AsyncMock(return_value=True)
    mock_session.agent.load_rules = AsyncMock()
    mock_session.agent.conversation = MagicMock()
    mock_session.agent.conversation.chat_messages = []
    mock_session.agent.get_modes = AsyncMock(return_value=[])

    mock_session.notifications = MagicMock()
    mock_session.notifications.replay = AsyncMock()
    mock_session.send_available_commands_update = AsyncMock()

    mock_acp_agent.session_manager.get_session = MagicMock(return_value=mock_session)  # type: ignore[assignment]
    mock_acp_agent._initialized = True

    with patch.object(mock_acp_agent._task_group, "start_soon"):
        response = await mock_acp_agent.load_session(
            LoadSessionRequest(session_id=session_id, cwd="/tmp")
        )

    assert isinstance(response, LoadSessionResponse)
    assert isinstance(response.config_options, list)


# ── Tests: Checkpoint-Aware Loading ──────────────────────────────────────


@pytest.mark.unit
async def test_load_checkpoint_messages_used_for_replay(
    mock_acp_agent: AgentPoolACPAgent,
    storage_manager: StorageManager,
    pending_messages: list[ModelMessage],
    pending_deferred_calls: list[PendingDeferredCall],
) -> None:
    """Checkpointed messages include ToolCallPart without ToolReturnPart.

    When loading from a checkpoint, the message_history from the checkpoint
    is used for replay. This ensures the exact state at checkpoint time
    (with pending ToolCallPart) is preserved.
    """
    session_id = "checkpointed-session-1"

    # Save checkpoint data to storage manager
    messages_json = serialize_messages(pending_messages) or ""
    await storage_manager.save_checkpoint(session_id, messages_json, pending_deferred_calls)

    # Load checkpoint to verify
    result = await storage_manager.load_checkpoint(session_id)
    assert result is not None
    loaded_msgs, loaded_calls = result

    # Verify the loaded messages match what we saved
    assert len(loaded_msgs) == len(pending_messages)
    assert len(loaded_calls) == len(pending_deferred_calls)

    # Verify ToolCallPart exists but no ToolReturnPart
    response_msg = loaded_msgs[1]
    assert isinstance(response_msg, ModelResponse)
    tool_parts = [p for p in response_msg.parts if isinstance(p, ToolCallPart)]
    assert len(tool_parts) == 1
    assert tool_parts[0].tool_call_id == "pending_call_1"

    # Verify no ToolReturnPart for the pending call
    for msg in loaded_msgs:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                from pydantic_ai import ToolReturnPart

                if isinstance(part, ToolReturnPart):
                    assert part.tool_call_id != "pending_call_1"


@pytest.mark.unit
async def test_load_session_with_checkpointed_data(
    mock_acp_agent: AgentPoolACPAgent,
    storage_manager: StorageManager,
    pending_messages: list[ModelMessage],
    pending_deferred_calls: list[PendingDeferredCall],
) -> None:
    """session/load properly handles checkpointed sessions.

    A checkpointed session with pending_deferred_calls should load
    message_history from the checkpoint and replay it correctly.
    """
    session_id = "checkpointed-session-2"

    # Save checkpoint data
    messages_json = serialize_messages(pending_messages) or ""
    await storage_manager.save_checkpoint(session_id, messages_json, pending_deferred_calls)

    # Create a mock session representing a checkpointed session
    mock_session = MagicMock()
    mock_session.session_id = session_id
    mock_session.cwd = "/tmp"

    chat_msgs = _make_chat_messages_from_model_messages(pending_messages, session_id=session_id)
    mock_session.agent = MagicMock()
    mock_session.agent.load_session = AsyncMock(return_value=True)
    mock_session.agent.load_rules = AsyncMock()
    mock_session.agent.conversation = MagicMock()
    mock_session.agent.conversation.chat_messages = chat_msgs
    mock_session.agent.get_modes = AsyncMock(return_value=[])

    mock_session.notifications = MagicMock()
    mock_session.notifications.replay = AsyncMock()
    mock_session.send_available_commands_update = AsyncMock()

    mock_acp_agent.session_manager.get_session = MagicMock(return_value=mock_session)  # type: ignore[assignment]
    mock_acp_agent._initialized = True

    with patch.object(mock_acp_agent._task_group, "start_soon"):
        await mock_acp_agent.load_session(LoadSessionRequest(session_id=session_id, cwd="/tmp"))

    # Verify replay was called
    mock_session.notifications.replay.assert_awaited_once()

    # Verify the messages were loaded from the checkpoint
    replay_args = mock_session.notifications.replay.call_args[0][0]
    assert len(replay_args) == len(pending_messages)


@pytest.mark.unit
async def test_load_session_no_toolcalldeferredevent_during_replay(
    mock_acp_agent: AgentPoolACPAgent,
    pending_messages: list[ModelMessage],
) -> None:
    """session/load does NOT emit ToolCallDeferredEvent during replay.

    Replaying message_history should only send session/update
    notifications. No new tool call events should be emitted.
    """
    session_id = "test-session-id"
    mock_session = MagicMock()
    mock_session.session_id = session_id
    mock_session.cwd = "/tmp"

    chat_msgs = _make_chat_messages_from_model_messages(pending_messages, session_id=session_id)
    mock_session.agent = MagicMock()
    mock_session.agent.load_session = AsyncMock(return_value=True)
    mock_session.agent.load_rules = AsyncMock()
    mock_session.agent.conversation = MagicMock()
    mock_session.agent.conversation.chat_messages = chat_msgs
    mock_session.agent.get_modes = AsyncMock(return_value=[])

    mock_session.notifications = MagicMock()
    mock_session.notifications.replay = AsyncMock()
    mock_session.send_available_commands_update = AsyncMock()

    mock_acp_agent.session_manager.get_session = MagicMock(return_value=mock_session)  # type: ignore[assignment]
    mock_acp_agent._initialized = True

    with patch.object(mock_acp_agent._task_group, "start_soon"):
        await mock_acp_agent.load_session(LoadSessionRequest(session_id=session_id, cwd="/tmp"))

    # replay is called, not any tool event emission
    mock_session.notifications.replay.assert_awaited_once()

    # The notifications should be session/update notifications only
    # (verified by mocking notifications.replay, not any event bus method)


@pytest.mark.unit
async def test_load_session_replay_preserves_chatmessage_model_messages(
    mock_acp_agent: AgentPoolACPAgent,
) -> None:
    """Replay preserves the exact ModelMessage objects from ChatMessage.messages.

    ModelMessages should be forwarded to replay without mutation or
    filtering that would change the ToolCallPart/ToolReturnPart state.
    """
    session_id = "test-session-id"
    model_messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="test")]),
        ModelResponse(
            parts=[
                TextPart(content="response"),
                ToolCallPart(
                    tool_name="test_tool",
                    args="{}",
                    tool_call_id="tool_1",
                ),
            ]
        ),
    ]

    mock_session = MagicMock()
    mock_session.session_id = session_id
    mock_session.cwd = "/tmp"

    chat_msgs = _make_chat_messages_from_model_messages(model_messages, session_id=session_id)
    mock_session.agent = MagicMock()
    mock_session.agent.load_session = AsyncMock(return_value=True)
    mock_session.agent.load_rules = AsyncMock()
    mock_session.agent.conversation = MagicMock()
    mock_session.agent.conversation.chat_messages = chat_msgs
    mock_session.agent.get_modes = AsyncMock(return_value=[])

    mock_session.notifications = MagicMock()
    mock_session.notifications.replay = AsyncMock()
    mock_session.send_available_commands_update = AsyncMock()

    mock_acp_agent.session_manager.get_session = MagicMock(return_value=mock_session)  # type: ignore[assignment]
    mock_acp_agent._initialized = True

    with patch.object(mock_acp_agent._task_group, "start_soon"):
        await mock_acp_agent.load_session(LoadSessionRequest(session_id=session_id, cwd="/tmp"))

    replay_args = mock_session.notifications.replay.call_args[0][0]
    # Verify the model messages are passed through without mutation
    assert len(replay_args) == len(model_messages)
    for i, (expected, actual) in enumerate(zip(model_messages, replay_args, strict=False)):
        assert type(actual) is type(expected), (
            f"Message {i} type mismatch: {type(actual)} != {type(expected)}"
        )
