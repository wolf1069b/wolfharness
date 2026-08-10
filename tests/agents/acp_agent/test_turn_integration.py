"""Integration tests for ACPTurn with RunHandle and ACPAgent.

Covers end-to-end integration between ACPTurn, RunHandle, ACPAgent,
and PromptInjectionManager using mocked ACP client interactions.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from acp import InitializeRequest
from acp.schema import AgentMessageChunk, TextContentBlock
from wolfharness.agents.acp_agent import ACPAgent
from wolfharness.agents.acp_agent.turn import ACPTurn
from wolfharness.agents.context import AgentRunContext
from wolfharness.agents.events import (
    PartDeltaEvent,
    StreamCompleteEvent,
)
from wolfharness.lifecycle import DirectChannel, MemoryJournal
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.run import RunHandle
from wolfharness.orchestrator.turn import Turn


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Shared helpers (adapted from test_turn.py and test_run_handle.py)
# ---------------------------------------------------------------------------


class MockACPClient:
    """Mock ACP client implementing ACPClientProtocol for testing."""

    def __init__(
        self,
        *,
        updates: list[Any] | None = None,
        messages: list[Any] | None = None,
        prompt_error: Exception | None = None,
    ) -> None:
        self._updates = updates or []
        self._messages = messages or []
        self._prompt_error = prompt_error
        self.prompt_calls: list[tuple[str, list[Any]]] = []

    async def prompt(self, session_id: str, content: list[Any]) -> Any:
        self.prompt_calls.append((session_id, content))
        if self._prompt_error:
            raise self._prompt_error
        return MagicMock(name="PromptResponse")

    async def stream_events(self, response: Any) -> AsyncIterator[Any]:
        for update in self._updates:
            yield update

    async def get_messages(self, session_id: str) -> list[Any]:
        return list(self._messages)


def _make_run_ctx(session_id: str = "test-session") -> AgentRunContext:
    """Create an AgentRunContext for testing."""
    return AgentRunContext(session_id=session_id)


def _text_update(text: str) -> AgentMessageChunk:
    """Create an AgentMessageChunk with a text content block."""
    return AgentMessageChunk(content=TextContentBlock(text=text))


# ---------------------------------------------------------------------------
# Test 1: ACPTurn full prompt→stream→complete cycle with mock client
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_acp_turn_full_cycle_with_mock_client() -> None:
    """Given a mock ACP client, ACPTurn yields the full event sequence.

    Verifies RunStarted, PartDelta, StreamComplete events are yielded, and
    that message_history and final_message are populated after execute().
    """
    updates = [_text_update("Hello"), _text_update(" world")]
    messages = [_text_update("Hello world")]
    client = MockACPClient(updates=updates, messages=messages)
    run_ctx = _make_run_ctx()

    turn = ACPTurn(
        acp_client=client,
        prompts=["Say hello"],
        run_ctx=run_ctx,
        session_id="test-session",
    )

    events = [event async for event in turn.execute()]

    # Verify prompt was called with correct session
    assert len(client.prompt_calls) == 1
    assert client.prompt_calls[0][0] == "test-session"

    # Verify event sequence: PartDelta → PartDelta → StreamComplete
    # (RunStartedEvent is published by RunHandle.start(), not Turn.execute())
    assert len(events) == 3

    assert isinstance(events[0], PartDeltaEvent)
    assert isinstance(events[1], PartDeltaEvent)

    assert isinstance(events[2], StreamCompleteEvent)
    assert events[2].message.content == "Hello world"

    # Verify message_history is populated after execute()
    history = turn.message_history
    assert len(history) > 0

    # Verify final_message is populated and matches StreamComplete payload
    final = turn.final_message
    assert final.role == "assistant"
    assert final.content == "Hello world"
    assert events[2].message is final


# ---------------------------------------------------------------------------
# Test 2: RunHandle.steer() for ACP path
# ---------------------------------------------------------------------------


class _BlockingTurn(Turn):
    """Stub Turn that blocks on a release event before completing.

    Used to keep _status == RunState.RUNNING long enough to call steer().
    """

    def __init__(self, *, release_event: asyncio.Event) -> None:
        super().__init__()
        self._release_event = release_event

    async def execute(self):  # type: ignore[override]
        await self._release_event.wait()
        self._message_history: list[Any] = ["msg1"]
        self._final_message = ChatMessage(content="done", role="assistant")
        yield StreamCompleteEvent(message=self._final_message)


@pytest.mark.unit
async def test_run_handle_steer_for_acp_path() -> None:
    """Given a RunHandle with a mocked ACPAgent, steer() queues to run_ctx.

    While the turn is running, steer() queues the message to
    queued_steer_messages (ACP path does not set active_agent_run).
    """
    release_event = asyncio.Event()

    # Create a mocked ACPAgent (avoid subprocess/ACP initialization)
    init_request = MagicMock(spec=InitializeRequest)
    agent = ACPAgent(command="test-cmd", init_request=init_request)
    agent.create_turn = MagicMock(return_value=_BlockingTurn(release_event=release_event))

    event_bus = AsyncMock()
    session = MagicMock()
    session.turn_lock = asyncio.Lock()
    session.parent_session_id = None
    session.input_provider = None
    # Per-prompt model: CommChannel lives on SessionState
    journal = MemoryJournal()
    comm = DirectChannel(journal=journal)
    session._comm_channel = comm
    session._active_steer_callback = None
    # Provide a real feedback_queue for steer() fallback (fix A)
    session.feedback_queue = asyncio.Queue()

    handle = RunHandle(
        run_id="test-run",
        session_id="test-session",
        agent_type="acp",
        agent=agent,
        event_bus=event_bus,
        session=session,
    )

    events: list[Any] = []
    gen = handle.start("hello")

    async def _consume() -> None:
        async for event in gen:
            events.append(event)  # noqa: PERF401

    consumer_task = asyncio.create_task(_consume())
    await asyncio.sleep(0.05)

    # Turn should be running (blocked on release_event inside _BlockingTurn)
    assert not handle.complete_event.is_set()
    # ACP path does not set active_agent_run (only NativeTurn does)
    assert handle.active_agent_run is None

    # Steer while running — should re-enqueue to session.feedback_queue (fix A)
    result = handle.steer("steered message")
    assert result is not None
    # With fix A, steer() fallback writes to session.feedback_queue (not queued_steer_messages)
    assert not session.feedback_queue.empty()
    fb = session.feedback_queue.get_nowait()
    assert fb.content == "steered message"
    assert fb.is_steer is True

    # Release the turn so it can complete
    release_event.set()
    await asyncio.sleep(0.05)
    await consumer_task

    assert handle.complete_event.is_set()


# ---------------------------------------------------------------------------
# Test 3: Tool-result augmentation via PromptInjectionManager.inject()/consume()
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Test 4: ACPAgent.create_turn() returns ACPTurn with correct fields
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_acp_agent_create_turn_returns_acp_turn_with_correct_fields() -> None:
    """Given an ACPAgent with mocked _api, create_turn() returns ACPTurn.

    Verifies the returned ACPTurn has correct acp_client, prompts, run_ctx,
    message_history, and session_id fields.
    """
    init_request = MagicMock(spec=InitializeRequest)
    agent = ACPAgent(command="test-cmd", init_request=init_request)

    mock_api = MagicMock(name="ACPAgentAPI")
    agent._api = mock_api
    agent._sdk_session_id = "acp-session-123"

    run_ctx = _make_run_ctx(session_id="run-ctx-session")
    message_history: list[Any] = []

    turn = agent.create_turn(
        prompts=["hello world"],
        run_ctx=run_ctx,
        message_history=message_history,
    )

    # Verify return type
    assert isinstance(turn, ACPTurn)

    # Verify fields are correctly wired
    assert turn._acp_client is mock_api
    assert turn._prompts == ["hello world"]
    assert turn._run_ctx is run_ctx
    # session_id should use _sdk_session_id when available
    assert turn._session_id == "acp-session-123"

    # Verify session_id falls back to run_ctx.session_id when _sdk_session_id is None
    agent._sdk_session_id = None
    turn2 = agent.create_turn(
        prompts=["second prompt"],
        run_ctx=run_ctx,
        message_history=message_history,
    )
    assert turn2._session_id == "run-ctx-session"
