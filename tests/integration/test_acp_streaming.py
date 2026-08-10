"""ACP streaming snapshot baseline (V10).

Captures the event sequence from ACP standalone streaming via
``ACPAgent._stream_events()``, including ``ToolCallCompleteEvent``
enriched with ``ToolResultMetadataEvent`` metadata.

This is a regression baseline for the pre-M4 protocol cleanup.  The test
uses **syrupy** for snapshot comparison and is marked ``snapshot``
so it is excluded from the default test run (run explicitly with
``-m snapshot`` or ``--snapshot-update``).

Key event sequence captured:
    1. ``RunStartedEvent``
    2. ``PartDeltaEvent`` (text chunk "Hello")
    3. ``ToolCallStartEvent`` (tool call "Read file")
    4. ``ToolCallCompleteEvent`` (enriched with metadata from
       ``ToolResultMetadataEvent``)
    5. ``PartDeltaEvent`` (text chunk " world")
    6. ``StreamCompleteEvent``
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from acp import InitializeRequest
from acp.schema import (
    AgentMessageChunk,
    PromptResponse,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
)
from wolfharness.agents.acp_agent import ACPAgent
from wolfharness.agents.acp_agent.session_state import ACPSessionState
from wolfharness.agents.context import AgentRunContext
from wolfharness.agents.events import (
    PartDeltaEvent,
    RunStartedEvent,
    StreamCompleteEvent,
    ToolCallCompleteEvent,
    ToolCallStartEvent,
    ToolResultMetadataEvent,
)
from wolfharness.messaging import ChatMessage


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from syrupy import SnapshotAssertion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize_event(event: Any) -> dict[str, Any]:  # noqa: PLR0911
    """Serialize a RichAgentStreamEvent to a dict suitable for snapshot.

    Strips non-deterministic fields (uuids, timestamps) and converts
    nested dataclass/Pydantic objects to plain dicts.
    """
    if isinstance(event, RunStartedEvent):
        return {
            "event_kind": event.event_kind,
            "agent_name": event.agent_name,
            "parent_session_id": event.parent_session_id,
        }
    if isinstance(event, PartDeltaEvent):
        delta = event.delta
        # TextPartDelta or ThinkingPartDelta
        delta_type = type(delta).__name__
        content = ""
        if hasattr(delta, "content_delta"):
            content = delta.content_delta
        return {
            "event_kind": "part_delta",
            "delta_type": delta_type,
            "content": content,
            "index": event.index,
        }
    if isinstance(event, ToolCallStartEvent):
        return {
            "event_kind": event.event_kind,
            "tool_call_id": event.tool_call_id,
            "tool_name": event.tool_name,
            "title": event.title,
            "kind": event.kind,
            "raw_input": event.raw_input,
        }
    if isinstance(event, ToolCallCompleteEvent):
        return {
            "event_kind": event.event_kind,
            "tool_call_id": event.tool_call_id,
            "tool_name": event.tool_name,
            "tool_input": event.tool_input,
            "tool_result": event.tool_result,
            "agent_name": event.agent_name,
            "metadata": event.metadata,
        }
    if isinstance(event, ToolResultMetadataEvent):
        return {
            "event_kind": event.event_kind,
            "tool_call_id": event.tool_call_id,
            "metadata": event.metadata,
        }
    if isinstance(event, StreamCompleteEvent):
        msg = event.message
        return {
            "event_kind": event.event_kind,
            "cancelled": event.cancelled,
            "content": msg.content,
            "role": msg.role,
            "name": msg.name,
            "finish_reason": msg.finish_reason,
        }
    # Fallback: type name + str
    return {"event_kind": getattr(event, "event_kind", "unknown"), "type": type(event).__name__}


def _text_chunk(text: str) -> AgentMessageChunk:
    """Create an ACP AgentMessageChunk with text content."""
    return AgentMessageChunk(content=TextContentBlock(text=text))


def _tool_call_start(
    tool_call_id: str,
    title: str,
    raw_input: dict[str, Any] | None = None,
) -> ToolCallStart:
    """Create an ACP ToolCallStart update."""
    return ToolCallStart(
        tool_call_id=tool_call_id,
        title=title,
        raw_input=raw_input or {},
    )


def _tool_call_progress_completed(
    tool_call_id: str,
    title: str,
    raw_output: str,
) -> ToolCallProgress:
    """Create an ACP ToolCallProgress with status=completed."""
    return ToolCallProgress(
        tool_call_id=tool_call_id,
        status="completed",
        title=title,
        raw_output=raw_output,
    )


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------


class _MockUpdateEvent:
    """Minimal mock of TimeoutableEvent for polling.

    Yields control to the event loop so that the prompt_task can execute
    concurrently with the polling loop.
    """

    def __init__(self) -> None:
        self._set = False

    def set(self) -> None:
        self._set = True

    def clear(self) -> None:
        self._set = False

    async def wait_with_timeout(self, timeout: float | None = None) -> bool:
        # Yield control to the event loop so prompt_task can run.
        # A tiny sleep prevents busy-looping while allowing concurrent task execution.
        await asyncio.sleep(0.001)
        return True


class _MockToolBridge:
    """Mock ToolManagerBridge that does nothing in set_run_context."""

    @asynccontextmanager
    async def set_run_context(
        self,
        context: Any,
        prompt: Any = None,
    ) -> AsyncIterator[_MockToolBridge]:
        yield self


class _MockACPClientHandler:
    """Mock ACPClientHandler with just the _update_event needed."""

    def __init__(self) -> None:
        self._update_event = _MockUpdateEvent()
        self._input_provider = None


class _MockAPI:
    """Mock ACPAgentAPI that returns a PromptResponse after a short delay.

    Implements stream_events() and get_messages() with the same polling
    logic as ACPAgentAPI so ACPTurn.execute() can use it as an
    ACPClientProtocol.
    """

    def __init__(
        self,
        delay: float = 0.01,
        state: Any | None = None,
        update_event: Any | None = None,
    ) -> None:
        self._delay = delay
        self._state = state
        self._update_event = update_event
        self._consumed_updates: list[Any] = []

    async def prompt(self, session_id: str, content: list[Any]) -> PromptResponse:
        await asyncio.sleep(self._delay)
        return PromptResponse(stop_reason="end_turn")

    async def fork_session(self, session_id: str, cwd: str) -> Any:
        raise NotImplementedError

    async def stream_events(self, response: Any) -> AsyncIterator[Any]:
        """Poll state queue for updates, same as ACPAgentAPI.stream_events()."""
        self._consumed_updates.clear()
        if self._state is None or self._update_event is None:
            return
        while True:
            try:
                await self._update_event.wait_with_timeout(0.05)
                self._update_event.clear()
            except TimeoutError:
                pass
            drained_any = False
            while (update := self._state.pop_update()) is not None:
                self._consumed_updates.append(update)
                yield update
                drained_any = True
            if not drained_any:
                break

    async def get_messages(self, session_id: str) -> list[Any]:
        return list(self._consumed_updates)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _PrePopulatedSessionState(ACPSessionState):
    """Session state that re-populates updates after clear().

    ``_stream_events`` calls ``self._state.clear()`` at the start of each
    turn, which wipes the updates deque.  This subclass overrides
    ``clear()`` to re-add the pre-populated updates so they survive the
    clear and are available for the polling loop.
    """

    def __init__(self, updates: list[Any], session_id: str = "test-session-id") -> None:
        super().__init__(session_id=session_id)
        self._initial_updates = list(updates)

    def clear(self) -> None:
        """Clear and re-populate with the pre-populated updates."""
        self.updates.clear()
        for update in self._initial_updates:
            self.updates.append(update)


def _make_acp_agent_with_mocks(
    updates: list[Any],
) -> ACPAgent[Any]:
    """Create an ACPAgent with mocked internals for _stream_events testing.

    Pre-populates the session state with the given updates (ACP SessionUpdate
    objects and optionally ToolResultMetadataEvent for metadata enrichment).
    """
    init_request = MagicMock(spec=InitializeRequest)
    agent = ACPAgent(command="test-cmd", init_request=init_request, name="test-acp-agent")

    # Set up mocked state with pre-populated updates that survive clear()
    state = _PrePopulatedSessionState(updates=updates)
    agent._state = state

    # Set up mocked client handler
    agent._client_handler = _MockACPClientHandler()

    # Set up mocked API with state/event for stream_events() and get_messages()
    agent._api = _MockAPI(
        state=state,
        update_event=agent._client_handler._update_event,
    )

    # Set up mocked session ID
    agent._sdk_session_id = "test-session-id"

    # Set up mocked tool bridge
    agent._tool_bridge = _MockToolBridge()

    return agent


def _make_run_ctx() -> AgentRunContext:
    """Create a minimal AgentRunContext for testing."""
    return AgentRunContext(session_id="test-session-id")


# ---------------------------------------------------------------------------
# Patched acp_to_native_event
# ---------------------------------------------------------------------------


_original_acp_to_native_event: Any = None


def _patched_acp_to_native_event(
    update: Any, *, step_index: int = 0, cumulative_usage: Any = None
) -> Any:
    """Patched converter that passes through ToolResultMetadataEvent.

    The real ``acp_to_native_event`` only handles ACP ``SessionUpdate``
    types.  This wrapper allows ``ToolResultMetadataEvent`` instances
    (which are native events, not ACP updates) to pass through the
    polling loop so that ``_stream_events`` can exercise the metadata
    enrichment code path.
    """
    if isinstance(update, ToolResultMetadataEvent):
        return update
    return _original_acp_to_native_event(update)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.snapshot
async def test_acp_streaming_event_sequence_with_tool_metadata(
    snapshot: SnapshotAssertion,
) -> None:
    """Capture full event sequence from ACP standalone streaming.

    The sequence includes:
    - RunStartedEvent
    - PartDeltaEvent (text "Hello")
    - ToolCallStartEvent (read file)
    - ToolResultMetadataEvent (sidechannel, consumed for enrichment)
    - ToolCallCompleteEvent (enriched with metadata)
    - PartDeltaEvent (text " world")
    - StreamCompleteEvent
    """
    tool_call_id = "tc-read-001"
    tool_metadata = {"diff": {"path": "/test/file.py", "old": "a", "new": "b"}}

    updates: list[Any] = [
        _text_chunk("Hello"),
        _tool_call_start(
            tool_call_id=tool_call_id,
            title="Read file",
            raw_input={"path": "/test/file.py"},
        ),
        # ToolResultMetadataEvent is a native event, not an ACP update.
        # It passes through the patched converter and gets consumed by
        # _stream_events for metadata enrichment.
        ToolResultMetadataEvent(
            tool_call_id=tool_call_id,
            metadata=tool_metadata,
        ),
        _tool_call_progress_completed(
            tool_call_id=tool_call_id,
            title="Read file",
            raw_output="file contents here",
        ),
        _text_chunk(" world"),
    ]

    agent = _make_acp_agent_with_mocks(updates)
    run_ctx = _make_run_ctx()

    user_msg = ChatMessage[str](
        content="test prompt",
        role="user",
        message_id="msg-001",
        session_id="test-session-id",
    )

    # Patch acp_to_native_event to allow ToolResultMetadataEvent passthrough
    from wolfharness.agents.acp_agent import acp_converters as converters_mod

    global _original_acp_to_native_event  # noqa: PLW0603
    _original_acp_to_native_event = converters_mod.acp_to_native_event

    with patch.object(converters_mod, "acp_to_native_event", _patched_acp_to_native_event):
        events: list[Any] = [
            event
            async for event in agent._stream_events(
                run_ctx,
                ["test prompt"],
                user_msg=user_msg,
                message_history=[],
                effective_parent_id=None,
                session_id="test-session-id",
            )
        ]

    # Serialize events for snapshot comparison
    serialized = [_serialize_event(e) for e in events]

    # Verify the event sequence matches the snapshot
    assert serialized == snapshot

    # Verify key properties of the event sequence
    # 1. First event is RunStartedEvent
    assert isinstance(events[0], RunStartedEvent)
    assert events[0].agent_name == "test-acp-agent"

    # 2. ToolResultMetadataEvent is NOT in the output (consumed for enrichment)
    metadata_events = [e for e in events if isinstance(e, ToolResultMetadataEvent)]
    assert len(metadata_events) == 0

    # 3. ToolCallCompleteEvent is enriched with metadata
    complete_events = [e for e in events if isinstance(e, ToolCallCompleteEvent)]
    assert len(complete_events) == 1
    assert complete_events[0].metadata == tool_metadata
    assert complete_events[0].agent_name == "test-acp-agent"

    # 4. Last event is StreamCompleteEvent
    assert isinstance(events[-1], StreamCompleteEvent)
    assert events[-1].message.content == "Hello world"
    assert events[-1].message.finish_reason == "stop"


@pytest.mark.anyio
@pytest.mark.snapshot
async def test_acp_streaming_text_only_sequence(
    snapshot: SnapshotAssertion,
) -> None:
    """Capture event sequence for a simple text-only ACP stream.

    No tool calls, just text chunks followed by StreamCompleteEvent.
    """
    updates: list[Any] = [
        _text_chunk("Hello"),
        _text_chunk(" "),
        _text_chunk("world"),
    ]

    agent = _make_acp_agent_with_mocks(updates)
    run_ctx = _make_run_ctx()

    user_msg = ChatMessage[str](
        content="test prompt",
        role="user",
        message_id="msg-002",
        session_id="test-session-id",
    )

    events: list[Any] = []
    async for event in agent._stream_events(
        run_ctx,
        ["test prompt"],
        user_msg=user_msg,
        message_history=[],
        effective_parent_id=None,
        session_id="test-session-id",
    ):
        events.append(event)  # noqa: PERF401

    serialized = [_serialize_event(e) for e in events]
    assert serialized == snapshot

    # Verify basic structure
    assert isinstance(events[0], RunStartedEvent)
    assert isinstance(events[-1], StreamCompleteEvent)
    assert events[-1].message.content == "Hello world"
