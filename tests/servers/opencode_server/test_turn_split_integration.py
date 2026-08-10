"""L2 integration tests for logical turn split in the OpenCode event bridge.

These tests use a REAL ``OpenCodeEventAdapter`` (wrapping a real
``EventProcessor``) and a REAL ``ServerState`` to verify the turn split
behavior when a ``UserMessageInsertedEvent(delivery="steer")`` arrives
during an active turn.

No mocking of ``EventBus.publish``, ``EventProcessor.process``, or
``ServerState.broadcast_event`` — the full event → SSE conversion pipeline
runs end-to-end. Tests verify ACTUAL message state in the session
(``server_state.messages``), not just event emission.

See ``test_event_bridge.py`` for the L1 unit tests of the split logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai import TextPart, TextPartDelta
import pytest

from wolfharness.agents.events.events import (
    PartDeltaEvent,
    PartStartEvent,
    RunStartedEvent,
    SpawnSessionStart,
    StreamCompleteEvent,
    ToolCallCompleteEvent,
    ToolCallStartEvent,
    UserMessageInsertedEvent,
)
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.event_bus import EventEnvelope
from wolfharness.utils import identifiers as identifier
from wolfharness_server.opencode_server.models import (
    AssistantMessage,
    MessageWithParts,
    TimeCreated,
    UserMessage,
)
from wolfharness_server.opencode_server.opencode_event_bridge import (
    OpenCodeEventBridgeMixin,
)


if TYPE_CHECKING:
    from wolfharness_server.opencode_server.event_adapter import OpenCodeEventAdapter
    from wolfharness_server.opencode_server.event_processor_context import (
        EventProcessorContext,
    )
    from wolfharness_server.opencode_server.state import ServerState


def _part_start(index: int, session_id: str, content: str = "") -> PartStartEvent:
    """Create a PartStartEvent with session_id (the .text() classmethod lacks it)."""
    return PartStartEvent(index=index, part=TextPart(content=content), session_id=session_id)


def _part_delta(index: int, session_id: str, content: str) -> PartDeltaEvent:
    """Create a PartDeltaEvent with session_id (the .text() classmethod lacks it)."""
    return PartDeltaEvent(
        index=index,
        delta=TextPartDelta(content_delta=content),
        session_id=session_id,
    )


pytestmark = [pytest.mark.integration, pytest.mark.anyio]


# =============================================================================
# Test infrastructure
# =============================================================================


class _FakeBridge(OpenCodeEventBridgeMixin):
    """Minimal concrete subclass wired to a real ServerState.

    Unlike the ``_FakeBridge`` in ``test_event_bridge.py`` which mocks
    ``server_state`` with ``MagicMock``, this version uses the real
    ``ServerState`` fixture so that ``append_message_to_session``,
    ``set_session_status``, and ``broadcast_event`` all run for real.
    """

    def __init__(self, server_state: ServerState, session_pool: Any) -> None:
        self.server_state = server_state
        self.session_pool = session_pool
        self._contexts: dict[str, EventProcessorContext] = {}
        self._adapters: dict[str, OpenCodeEventAdapter] = {}
        self._message_registered: dict[str, bool] = {}
        self._steer_split_ids: dict[str, set[str]] = {}
        self._child_to_parent: dict[str, str] = {}
        self._child_spawns: dict[str, SpawnSessionStart] = {}
        self._children_of: dict[str, set[str]] = {}
        self._resume_contexts: dict[str, dict[str, Any]] = {}
        self._pending_message_ids: dict[str, str] = {}
        self._pending_message_metadata: dict[str, dict[str, str | None]] = {}

    def get_session_context_data(self, session_id: str) -> dict[str, Any] | None:
        """No resume context for fresh test sessions."""
        return None


def _make_envelope(session_id: str, event: Any) -> EventEnvelope:
    """Create an EventEnvelope for the given session and event."""
    return EventEnvelope(source_session_id=session_id, event=event)


async def _feed_events(bridge: _FakeBridge, session_id: str, events: list[Any]) -> None:
    """Feed a sequence of events through _handle_event one by one."""
    for event in events:
        envelope = _make_envelope(session_id, event)
        await bridge._handle_event(session_id, envelope)


def _assistant_messages(state: ServerState, session_id: str) -> list[MessageWithParts]:
    """Return only assistant messages from the session state."""
    messages = state.messages.get(session_id, [])
    return [m for m in messages if isinstance(m.info, AssistantMessage)]


def _user_messages(state: ServerState, session_id: str) -> list[MessageWithParts]:
    """Return only user messages from the session state."""
    messages = state.messages.get(session_id, [])
    return [m for m in messages if isinstance(m.info, UserMessage)]


def _all_message_ids(state: ServerState, session_id: str) -> list[str]:
    """Return all message IDs in insertion order."""
    messages = state.messages.get(session_id, [])
    return [m.info.id for m in messages]


# =============================================================================
# TEST 1: Single steer creates two assistant messages
# =============================================================================


@pytest.mark.asyncio
async def test_single_steer_creates_two_assistant_messages(
    server_state: ServerState,
) -> None:
    """Single steer during active turn splits assistant message into two.

    Given: A real EventProcessor with an active turn (RunStarted → PartStart
        → PartDelta) that receives a steer UserMessageInsertedEvent followed
        by another PartStart (post-steer content).
    When: Events are fed through _handle_event in sequence.
    Then: Session has 2 assistant messages with different IDs, A1 has
        time.completed set (finalized during split), A2 has time.completed
        set by StreamCompleteEvent.
    """
    session_id = "test-split-1"
    bridge = _FakeBridge(server_state, server_state.pool.session_pool)
    await bridge._before_consumer_loop(session_id)

    steer_msg_id = identifier.ascending("message")

    events: list[Any] = [
        RunStartedEvent(run_id="run-1", agent_name="test-agent", session_id=session_id),
        _part_start(0, session_id),
        _part_delta(0, session_id, "hello"),
        UserMessageInsertedEvent(
            session_id=session_id,
            message_id=steer_msg_id,
            content="steer text",
            delivery="steer",
            source="processed",
        ),
        _part_start(1, session_id),
        _part_delta(1, session_id, "world"),
        StreamCompleteEvent(
            message=ChatMessage(content="done", role="assistant"),
            session_id=session_id,
        ),
    ]
    await _feed_events(bridge, session_id, events)

    assistant_msgs = _assistant_messages(server_state, session_id)
    assert len(assistant_msgs) == 2, f"Expected 2 assistant messages, got {len(assistant_msgs)}"

    a1, a2 = assistant_msgs
    assert a1.info.id != a2.info.id, "A1 and A2 must have different message IDs"

    # A1 should have time.completed set (finalized during split)
    assert isinstance(a1.info, AssistantMessage)
    assert a1.info.time.completed is not None, "A1 time.completed should be set during split"

    # A2 should have time.completed set by StreamCompleteEvent
    assert isinstance(a2.info, AssistantMessage)
    assert a2.info.time.completed is not None, (
        "A2 time.completed should be set by StreamCompleteEvent"
    )


# =============================================================================
# TEST 2: Message ID ordering after split
# =============================================================================


@pytest.mark.asyncio
async def test_message_id_ordering_after_split(
    server_state: ServerState,
) -> None:
    """Message IDs sort lexicographically: msg_U < msg_A1 < msg_steer < msg_A2.

    Given: A user message (msg_U) created before the turn, then A1 from
        _before_consumer_loop, then a steer user message (msg_steer), then
        A2 from the split.
    When: Events are fed through _handle_event with a steer split.
    Then: All four message IDs are in ascending lexicographic order when
        compared as strings, so the TUI sorts them correctly.
    """
    session_id = "test-split-2"
    bridge = _FakeBridge(server_state, server_state.pool.session_pool)

    # Create a user message BEFORE _before_consumer_loop so msg_U < msg_A1
    msg_u_id = identifier.ascending("message")
    user_msg = MessageWithParts(
        info=UserMessage(
            id=msg_u_id,
            session_id=session_id,
            time=TimeCreated(created=0),
        ),
        parts=[],
    )
    server_state.messages.setdefault(session_id, []).append(user_msg)

    await bridge._before_consumer_loop(session_id)

    # A1 ID is assigned by _before_consumer_loop
    ctx = bridge._contexts[session_id]
    msg_a1_id = ctx.assistant_msg_id

    msg_steer_id = identifier.ascending("message")

    events: list[Any] = [
        RunStartedEvent(run_id="run-1", agent_name="test-agent", session_id=session_id),
        _part_start(0, session_id),
        _part_delta(0, session_id, "hello"),
        UserMessageInsertedEvent(
            session_id=session_id,
            message_id=msg_steer_id,
            content="steer text",
            delivery="steer",
            source="processed",
        ),
        _part_start(1, session_id),
        _part_delta(1, session_id, "world"),
        StreamCompleteEvent(
            message=ChatMessage(content="done", role="assistant"),
            session_id=session_id,
        ),
    ]
    await _feed_events(bridge, session_id, events)

    # A2 ID is the current assistant_msg_id after the split
    msg_a2_id = ctx.assistant_msg_id

    # Verify lexicographic ordering: msg_U < msg_A1 < msg_steer < msg_A2
    assert msg_u_id < msg_a1_id, f"msg_U ({msg_u_id}) must sort before msg_A1 ({msg_a1_id})"
    assert msg_a1_id < msg_steer_id, (
        f"msg_A1 ({msg_a1_id}) must sort before msg_steer ({msg_steer_id})"
    )
    assert msg_steer_id < msg_a2_id, (
        f"msg_steer ({msg_steer_id}) must sort before msg_A2 ({msg_a2_id})"
    )

    # Also verify the actual messages in session state match.
    # Note: msg_steer_id is NOT in all_ids because source="processed" does
    # NOT create a UserMessage (only source="accepted" creates UserMessages).
    # The split still happens, creating A2 with a new ID.
    all_ids = _all_message_ids(server_state, session_id)
    assert msg_u_id in all_ids
    assert msg_a1_id in all_ids
    assert msg_a2_id in all_ids


# =============================================================================
# TEST 3: Multiple steers create three splits
# =============================================================================


@pytest.mark.asyncio
async def test_multiple_steers_create_three_splits(
    server_state: ServerState,
) -> None:
    """Two consecutive steers during a turn create three assistant messages.

    Given: A single turn with two steer injections, each followed by a
        PartStartEvent triggering a split.
    When: Events are fed through _handle_event.
    Then: Session has 3 assistant messages (A1, A2, A3) with different IDs,
        and A1 and A2 both have time.completed set (finalized during splits).
    """
    session_id = "test-split-3"
    bridge = _FakeBridge(server_state, server_state.pool.session_pool)
    await bridge._before_consumer_loop(session_id)

    steer_1_id = identifier.ascending("message")
    steer_2_id = identifier.ascending("message")

    events: list[Any] = [
        RunStartedEvent(run_id="run-1", agent_name="test-agent", session_id=session_id),
        _part_start(0, session_id),
        _part_delta(0, session_id, "part1"),
        UserMessageInsertedEvent(
            session_id=session_id,
            message_id=steer_1_id,
            content="steer 1",
            delivery="steer",
            source="processed",
        ),
        _part_start(1, session_id),
        _part_delta(1, session_id, "part2"),
        UserMessageInsertedEvent(
            session_id=session_id,
            message_id=steer_2_id,
            content="steer 2",
            delivery="steer",
            source="processed",
        ),
        _part_start(2, session_id),
        _part_delta(2, session_id, "part3"),
        StreamCompleteEvent(
            message=ChatMessage(content="done", role="assistant"),
            session_id=session_id,
        ),
    ]
    await _feed_events(bridge, session_id, events)

    assistant_msgs = _assistant_messages(server_state, session_id)
    assert len(assistant_msgs) == 3, f"Expected 3 assistant messages, got {len(assistant_msgs)}"

    a1, a2, a3 = assistant_msgs
    ids = {a1.info.id, a2.info.id, a3.info.id}
    assert len(ids) == 3, "All three assistant message IDs must be different"

    # A1 and A2 should have time.completed set (finalized during splits)
    assert isinstance(a1.info, AssistantMessage)
    assert a1.info.time.completed is not None, "A1 time.completed should be set"
    assert isinstance(a2.info, AssistantMessage)
    assert a2.info.time.completed is not None, "A2 time.completed should be set"

    # A3 should have time.completed set by StreamCompleteEvent
    assert isinstance(a3.info, AssistantMessage)
    assert a3.info.time.completed is not None, "A3 time.completed should be set"


# =============================================================================
# TEST 4: Followup does not trigger split
# =============================================================================


@pytest.mark.asyncio
async def test_followup_does_not_trigger_split(
    server_state: ServerState,
) -> None:
    """UserMessageInsertedEvent with delivery="followup" does NOT trigger split.

    Given: A real EventProcessor with an active turn that receives a
        followup UserMessageInsertedEvent (not steer).
    When: Events are fed through _handle_event.
    Then: Session has only 1 assistant message (no split).
    """
    session_id = "test-split-4"
    bridge = _FakeBridge(server_state, server_state.pool.session_pool)
    await bridge._before_consumer_loop(session_id)

    followup_msg_id = identifier.ascending("message")

    events: list[Any] = [
        RunStartedEvent(run_id="run-1", agent_name="test-agent", session_id=session_id),
        _part_start(0, session_id),
        UserMessageInsertedEvent(
            session_id=session_id,
            message_id=followup_msg_id,
            content="followup text",
            delivery="followup",
            source="accepted",
        ),
        _part_delta(0, session_id, "hello"),
        StreamCompleteEvent(
            message=ChatMessage(content="done", role="assistant"),
            session_id=session_id,
        ),
    ]
    await _feed_events(bridge, session_id, events)

    assistant_msgs = _assistant_messages(server_state, session_id)
    assert len(assistant_msgs) == 1, (
        f"Expected 1 assistant message (no split), got {len(assistant_msgs)}"
    )


# =============================================================================
# TEST 5: Split preserves tool parts in A1 (cleared on split)
# =============================================================================


@pytest.mark.asyncio
async def test_split_preserves_tool_parts_in_a1(
    server_state: ServerState,
) -> None:
    """Tool parts in A1 are preserved after split; A2 has no tool parts.

    Given: A turn with tool calls in A1, then a steer that triggers a split
        to A2.
    When: Events are fed through _handle_event.
    Then: A1's parts list contains tool-related parts (from the tool calls),
        and A2's parts list does not contain those tool parts (they were
        cleared during the split reset).
    """
    session_id = "test-split-5"
    bridge = _FakeBridge(server_state, server_state.pool.session_pool)
    await bridge._before_consumer_loop(session_id)

    steer_msg_id = identifier.ascending("message")
    tool_call_id = "tc_001"

    events: list[Any] = [
        RunStartedEvent(run_id="run-1", agent_name="test-agent", session_id=session_id),
        _part_start(0, session_id),
        ToolCallStartEvent(
            tool_call_id=tool_call_id,
            tool_name="read",
            title="Reading file",
            session_id=session_id,
        ),
        ToolCallCompleteEvent(
            tool_name="read",
            tool_call_id=tool_call_id,
            tool_input={"path": "/tmp/test.txt"},
            tool_result="file contents",
            agent_name="test-agent",
            message_id="",
            session_id=session_id,
        ),
        UserMessageInsertedEvent(
            session_id=session_id,
            message_id=steer_msg_id,
            content="steer text",
            delivery="steer",
            source="processed",
        ),
        _part_start(1, session_id),
        _part_delta(1, session_id, "after steer"),
        StreamCompleteEvent(
            message=ChatMessage(content="done", role="assistant"),
            session_id=session_id,
        ),
    ]
    await _feed_events(bridge, session_id, events)

    assistant_msgs = _assistant_messages(server_state, session_id)
    assert len(assistant_msgs) == 2, f"Expected 2 assistant messages, got {len(assistant_msgs)}"

    a1, a2 = assistant_msgs

    # A1 should have tool-related parts (ToolPart from ToolCallCompleteEvent)
    # The EventProcessor creates ToolPart entries in ctx.tool_parts and
    # appends them to assistant_msg.parts. After split, ctx.tool_parts is
    # cleared, but A1's parts list already has them.
    from wolfharness_server.opencode_server.models import ToolPart

    a1_tool_parts = [p for p in a1.parts if isinstance(p, ToolPart)]
    assert len(a1_tool_parts) >= 1, "A1 should contain tool parts from the tool call before split"

    # A2 should NOT have tool parts from A1 (ctx.tool_parts was cleared)
    a2_tool_parts = [p for p in a2.parts if isinstance(p, ToolPart)]
    assert len(a2_tool_parts) == 0, "A2 should not contain tool parts from before the split"

    # Also verify ctx state was cleared
    ctx = bridge._contexts[session_id]
    assert len(ctx.tool_parts) == 0, "ctx.tool_parts should be cleared after split"


# =============================================================================
# TEST 6: D1 reset still works after split
# =============================================================================


@pytest.mark.asyncio
async def test_d1_reset_still_works_after_split(
    server_state: ServerState,
) -> None:
    """D1 reset (RunStartedEvent for subsequent turn) works after a split.

    Given: Turn 1 has a steer split (producing A1, A2), then Turn 2 starts
        with a RunStartedEvent that triggers D1 reset (producing A3).
    When: Events for both turns are fed through _handle_event.
    Then: Session has 3 assistant messages (A1, A2 from turn 1 split, A3
        from turn 2 D1 reset), and A1 and A2 both have time.completed set.
    """
    session_id = "test-split-6"
    bridge = _FakeBridge(server_state, server_state.pool.session_pool)
    await bridge._before_consumer_loop(session_id)

    steer_msg_id = identifier.ascending("message")

    # Turn 1: RunStarted → PartStart(A1) → steer → PartStart(A2) → StreamComplete
    turn_1_events: list[Any] = [
        RunStartedEvent(run_id="run-1", agent_name="test-agent", session_id=session_id),
        _part_start(0, session_id),
        _part_delta(0, session_id, "turn1-before-steer"),
        UserMessageInsertedEvent(
            session_id=session_id,
            message_id=steer_msg_id,
            content="steer text",
            delivery="steer",
            source="processed",
        ),
        _part_start(1, session_id),
        _part_delta(1, session_id, "turn1-after-steer"),
        StreamCompleteEvent(
            message=ChatMessage(content="done", role="assistant"),
            session_id=session_id,
        ),
    ]
    await _feed_events(bridge, session_id, turn_1_events)

    # Verify turn 1 produced 2 assistant messages
    assistant_after_t1 = _assistant_messages(server_state, session_id)
    assert len(assistant_after_t1) == 2, (
        f"Expected 2 assistant messages after turn 1, got {len(assistant_after_t1)}"
    )

    # Turn 2: RunStarted → PartStart(A3) → PartDelta(A3) → StreamComplete
    # The RunStartedEvent triggers D1 reset (creates A3)
    turn_2_events: list[Any] = [
        RunStartedEvent(run_id="run-2", agent_name="test-agent", session_id=session_id),
        _part_start(0, session_id),
        _part_delta(0, session_id, "turn2 content"),
        StreamCompleteEvent(
            message=ChatMessage(content="done", role="assistant"),
            session_id=session_id,
        ),
    ]
    await _feed_events(bridge, session_id, turn_2_events)

    # Verify total 3 assistant messages
    assistant_msgs = _assistant_messages(server_state, session_id)
    assert len(assistant_msgs) == 3, (
        f"Expected 3 assistant messages (A1, A2, A3), got {len(assistant_msgs)}"
    )

    a1, a2, a3 = assistant_msgs

    # All three must have different IDs
    ids = {a1.info.id, a2.info.id, a3.info.id}
    assert len(ids) == 3, "All three assistant message IDs must be different"

    # A1 and A2 should have time.completed set
    assert isinstance(a1.info, AssistantMessage)
    assert a1.info.time.completed is not None, "A1 time.completed should be set"
    assert isinstance(a2.info, AssistantMessage)
    assert a2.info.time.completed is not None, "A2 time.completed should be set"

    # A3 should also have time.completed set (by StreamCompleteEvent)
    assert isinstance(a3.info, AssistantMessage)
    assert a3.info.time.completed is not None, "A3 time.completed should be set"


# =============================================================================
# TEST 7: Split triggers on processed steer event (precise trigger)
# =============================================================================


@pytest.mark.asyncio
async def test_split_triggers_on_processed_steer_event_precise(
    server_state: ServerState,
) -> None:
    """UserMessageInsertedEvent(source="processed", delivery="steer") triggers split.

    Given: A real EventProcessor with an active turn that receives a processed
        steer UserMessageInsertedEvent (from EnqueuedMessagesEvent drain time).
    When: Events are fed through _handle_event.
    Then: Session has 2 assistant messages (split happened immediately on the
        processed steer event, no PartStartEvent heuristic needed).
    """
    session_id = "test-split-7"
    bridge = _FakeBridge(server_state, server_state.pool.session_pool)
    await bridge._before_consumer_loop(session_id)

    steer_msg_id = identifier.ascending("message")

    events: list[Any] = [
        RunStartedEvent(run_id="run-1", agent_name="test-agent", session_id=session_id),
        _part_start(0, session_id),
        _part_delta(0, session_id, "hello"),
        UserMessageInsertedEvent(
            session_id=session_id,
            message_id=steer_msg_id,
            content="steer text",
            delivery="steer",
            source="processed",
        ),
        _part_delta(0, session_id, "world"),
        StreamCompleteEvent(
            message=ChatMessage(content="done", role="assistant"),
            session_id=session_id,
        ),
    ]
    await _feed_events(bridge, session_id, events)

    assistant_msgs = _assistant_messages(server_state, session_id)
    assert len(assistant_msgs) == 2, f"Expected 2 assistant messages, got {len(assistant_msgs)}"


# =============================================================================
# TEST 8: Split DOES trigger on internal steer event (fire-and-forget fallback)
# =============================================================================


@pytest.mark.asyncio
async def test_split_triggers_on_processed_steer_event(
    server_state: ServerState,
) -> None:
    """UserMessageInsertedEvent(source="processed", delivery="steer") DOES split.

    source="processed" (fire-and-forget) is a fallback when
    EnqueuedMessagesEvent is not available. The split triggers on
    processed source to ensure the turn boundary is created.
    """
    session_id = "test-split-8"
    bridge = _FakeBridge(server_state, server_state.pool.session_pool)
    await bridge._before_consumer_loop(session_id)

    steer_msg_id = identifier.ascending("message")

    events: list[Any] = [
        RunStartedEvent(run_id="run-1", agent_name="test-agent", session_id=session_id),
        _part_start(0, session_id),
        _part_delta(0, session_id, "hello"),
        UserMessageInsertedEvent(
            session_id=session_id,
            message_id=steer_msg_id,
            content="steer text",
            delivery="steer",
            source="processed",
        ),
        _part_delta(0, session_id, "world"),
        StreamCompleteEvent(
            message=ChatMessage(content="done", role="assistant"),
            session_id=session_id,
        ),
    ]
    await _feed_events(bridge, session_id, events)

    assistant_msgs = _assistant_messages(server_state, session_id)
    assert len(assistant_msgs) == 2, (
        f"Expected 2 assistant messages (split on processed source), got {len(assistant_msgs)}"
    )


# =============================================================================
# TEST 9: Split does NOT trigger on internal followup event
# =============================================================================


@pytest.mark.asyncio
async def test_split_does_not_trigger_on_accepted_followup_event(
    server_state: ServerState,
) -> None:
    """UserMessageInsertedEvent(source="accepted", delivery="followup") does NOT split.

    Given: A real EventProcessor with an active turn that receives an accepted
        followup UserMessageInsertedEvent (not steer delivery).
    When: Events are fed through _handle_event.
    Then: Session has only 1 assistant message (no split).
    """
    session_id = "test-split-9"
    bridge = _FakeBridge(server_state, server_state.pool.session_pool)
    await bridge._before_consumer_loop(session_id)

    followup_msg_id = identifier.ascending("message")

    events: list[Any] = [
        RunStartedEvent(run_id="run-1", agent_name="test-agent", session_id=session_id),
        _part_start(0, session_id),
        _part_delta(0, session_id, "hello"),
        UserMessageInsertedEvent(
            session_id=session_id,
            message_id=followup_msg_id,
            content="followup text",
            delivery="followup",
            source="accepted",
        ),
        _part_delta(0, session_id, "world"),
        StreamCompleteEvent(
            message=ChatMessage(content="done", role="assistant"),
            session_id=session_id,
        ),
    ]
    await _feed_events(bridge, session_id, events)

    assistant_msgs = _assistant_messages(server_state, session_id)
    assert len(assistant_msgs) == 1, (
        f"Expected 1 assistant message (no split for followup), got {len(assistant_msgs)}"
    )


# =============================================================================
# TEST 10: Split does NOT trigger when no message registered
# =============================================================================


@pytest.mark.asyncio
async def test_split_does_not_trigger_when_no_message_registered(
    server_state: ServerState,
) -> None:
    """Accepted steer event with no registered message does NOT split.

    Given: A real EventProcessor where _message_registered is False (no
        assistant message registered yet) receives a processed steer event.
    When: Events are fed through _handle_event.
    Then: Session has only 1 assistant message (no split), because the
        _message_registered guard prevents splitting before any content.
    """
    session_id = "test-split-10"
    bridge = _FakeBridge(server_state, server_state.pool.session_pool)
    await bridge._before_consumer_loop(session_id)

    steer_msg_id = identifier.ascending("message")

    # Feed the internal steer event as the VERY FIRST event, before
    # RunStartedEvent. At this point _message_registered is False (default),
    # so the split guard should prevent the split.
    events: list[Any] = [
        UserMessageInsertedEvent(
            session_id=session_id,
            message_id=steer_msg_id,
            content="steer text",
            delivery="steer",
            source="processed",
        ),
        RunStartedEvent(run_id="run-1", agent_name="test-agent", session_id=session_id),
        _part_start(0, session_id),
        _part_delta(0, session_id, "hello"),
        StreamCompleteEvent(
            message=ChatMessage(content="done", role="assistant"),
            session_id=session_id,
        ),
    ]
    await _feed_events(bridge, session_id, events)

    assistant_msgs = _assistant_messages(server_state, session_id)
    assert len(assistant_msgs) == 1, (
        f"Expected 1 assistant message (no split when unregistered), got {len(assistant_msgs)}"
    )


# =============================================================================
# TEST 11: Split creates new assistant message with fresh ID
# =============================================================================


@pytest.mark.asyncio
async def test_split_creates_new_assistant_message(
    server_state: ServerState,
) -> None:
    """Split creates a new assistant message with a fresh message ID.

    Given: A real EventProcessor with an active turn that receives a processed
        steer event.
    When: Events are fed through _handle_event.
    Then: The split creates a new assistant message (A2) whose ID is different
        from the original (A1), and A2 is registered in the session.
    """
    session_id = "test-split-11"
    bridge = _FakeBridge(server_state, server_state.pool.session_pool)
    await bridge._before_consumer_loop(session_id)

    ctx = bridge._contexts[session_id]
    original_msg_id = ctx.assistant_msg_id

    steer_msg_id = identifier.ascending("message")

    events: list[Any] = [
        RunStartedEvent(run_id="run-1", agent_name="test-agent", session_id=session_id),
        _part_start(0, session_id),
        _part_delta(0, session_id, "hello"),
        UserMessageInsertedEvent(
            session_id=session_id,
            message_id=steer_msg_id,
            content="steer text",
            delivery="steer",
            source="processed",
        ),
        _part_delta(0, session_id, "world"),
        StreamCompleteEvent(
            message=ChatMessage(content="done", role="assistant"),
            session_id=session_id,
        ),
    ]
    await _feed_events(bridge, session_id, events)

    # After split, ctx.assistant_msg_id should be different (A2's ID)
    assert ctx.assistant_msg_id != original_msg_id, (
        "Split should have created a new assistant message with a fresh ID"
    )

    # Both messages should be in session state
    all_ids = _all_message_ids(server_state, session_id)
    assert original_msg_id in all_ids, "A1 (original) should be in session messages"
    assert ctx.assistant_msg_id in all_ids, "A2 (new) should be in session messages"


# =============================================================================
# TEST 12: Split triggers on processed source (5.17)
# =============================================================================


@pytest.mark.asyncio
async def test_split_triggers_on_processed_source(
    server_state: ServerState,
) -> None:
    """UserMessageInsertedEvent(source="processed", delivery="steer") triggers split.

    Given: A real EventProcessor with an active turn that receives a
        processed steer UserMessageInsertedEvent.
    When: Events are fed through _handle_event.
    Then: Session has 2 assistant messages (split triggered by processed source).
    """
    session_id = "test-split-12"
    bridge = _FakeBridge(server_state, server_state.pool.session_pool)
    await bridge._before_consumer_loop(session_id)

    steer_msg_id = identifier.ascending("message")

    events: list[Any] = [
        RunStartedEvent(run_id="run-1", agent_name="test-agent", session_id=session_id),
        _part_start(0, session_id),
        _part_delta(0, session_id, "hello"),
        UserMessageInsertedEvent(
            session_id=session_id,
            message_id=steer_msg_id,
            content="steer text",
            delivery="steer",
            source="processed",
        ),
        _part_delta(0, session_id, "world"),
        StreamCompleteEvent(
            message=ChatMessage(content="done", role="assistant"),
            session_id=session_id,
        ),
    ]
    await _feed_events(bridge, session_id, events)

    assistant_msgs = _assistant_messages(server_state, session_id)
    assert len(assistant_msgs) == 2, (
        f"Expected 2 assistant messages (split on processed source), got {len(assistant_msgs)}"
    )


# =============================================================================
# TEST 13: Split does NOT trigger on accepted source (5.18)
# =============================================================================


@pytest.mark.asyncio
async def test_split_does_not_trigger_on_accepted_source(
    server_state: ServerState,
) -> None:
    """UserMessageInsertedEvent(source="accepted", delivery="steer") does NOT split.

    Given: A real EventProcessor with an active turn that receives an
        accepted steer UserMessageInsertedEvent.
    When: Events are fed through _handle_event.
    Then: Session has only 1 assistant message (accepted source does NOT
        trigger split — only processed source does).
    """
    session_id = "test-split-13"
    bridge = _FakeBridge(server_state, server_state.pool.session_pool)
    await bridge._before_consumer_loop(session_id)

    steer_msg_id = identifier.ascending("message")

    events: list[Any] = [
        RunStartedEvent(run_id="run-1", agent_name="test-agent", session_id=session_id),
        _part_start(0, session_id),
        _part_delta(0, session_id, "hello"),
        UserMessageInsertedEvent(
            session_id=session_id,
            message_id=steer_msg_id,
            content="steer text",
            delivery="steer",
            source="accepted",
        ),
        _part_delta(0, session_id, "world"),
        StreamCompleteEvent(
            message=ChatMessage(content="done", role="assistant"),
            session_id=session_id,
        ),
    ]
    await _feed_events(bridge, session_id, events)

    assistant_msgs = _assistant_messages(server_state, session_id)
    assert len(assistant_msgs) == 1, (
        f"Expected 1 assistant message (accepted source does NOT split), got {len(assistant_msgs)}"
    )


# =============================================================================
# TEST 14: Split skipped during replay (5.22)
# =============================================================================


@pytest.mark.asyncio
async def test_split_skipped_during_replay(
    server_state: ServerState,
) -> None:
    """EventBridge does NOT split when _replaying=True even for processed+steer.

    Given: A real EventProcessor with an active turn and _replaying=True
        that receives a processed steer UserMessageInsertedEvent.
    When: Events are fed through _handle_event with _replaying set.
    Then: Session has only 1 assistant message (split is skipped during
        replay to avoid duplicate splits from journaled events).
    """
    session_id = "test-split-14"
    bridge = _FakeBridge(server_state, server_state.pool.session_pool)
    await bridge._before_consumer_loop(session_id)

    # Set _replaying flag to simulate crash-recovery replay.
    bridge._replaying = True

    steer_msg_id = identifier.ascending("message")

    events: list[Any] = [
        RunStartedEvent(run_id="run-1", agent_name="test-agent", session_id=session_id),
        _part_start(0, session_id),
        _part_delta(0, session_id, "hello"),
        UserMessageInsertedEvent(
            session_id=session_id,
            message_id=steer_msg_id,
            content="steer text",
            delivery="steer",
            source="processed",
        ),
        _part_delta(0, session_id, "world"),
        StreamCompleteEvent(
            message=ChatMessage(content="done", role="assistant"),
            session_id=session_id,
        ),
    ]
    await _feed_events(bridge, session_id, events)

    assistant_msgs = _assistant_messages(server_state, session_id)
    assert len(assistant_msgs) == 1, (
        f"Expected 1 assistant message (split skipped during replay), got {len(assistant_msgs)}"
    )
