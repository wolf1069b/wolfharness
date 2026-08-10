"""Tests for ACP event converter error event handling.

Covers RunErrorEvent and RunFailedEvent conversion to ACP session updates.
"""

from __future__ import annotations

from pydantic_ai import RequestUsage
import pytest

from wolfharness.agents.events import (
    RunErrorEvent,
    RunFailedEvent,
    StreamCompleteEvent,
    ToolCallStartEvent,
)
from wolfharness.messaging.messages import ChatMessage
from wolfharness_server.acp_server.event_converter import ACPEventConverter


@pytest.fixture
def converter() -> ACPEventConverter:
    """Create a converter instance for testing."""
    return ACPEventConverter()


@pytest.fixture
def converter_with_turn_complete() -> ACPEventConverter:
    """Create a converter with client_supports_turn_complete enabled."""
    return ACPEventConverter(client_supports_turn_complete=True)


def _dump(update: object) -> dict[str, object]:
    """Convert an ACP update to a dict for assertion, using model_dump if available."""
    if hasattr(update, "model_dump"):
        return update.model_dump(exclude_none=True)  # type: ignore[union-attr]
    return {"_str": str(update)}


# ---------------------------------------------------------------------------
# RunErrorEvent
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_run_error_event_produces_text_message(converter: ACPEventConverter):
    """RunErrorEvent should produce an AgentMessageChunk with error text."""
    event = RunErrorEvent(message="Model API returned 500", agent_name="engineer")
    updates = [u async for u in converter.convert(event)]

    assert len(updates) == 1
    d = _dump(updates[0])
    assert d["session_update"] == "agent_message_chunk"
    assert "Error" in str(d["content"])
    assert "engineer" in str(d["content"])
    assert "Model API returned 500" in str(d["content"])


@pytest.mark.unit
async def test_run_error_event_without_agent_name(converter: ACPEventConverter):
    """RunErrorEvent without agent_name should still produce error text."""
    event = RunErrorEvent(message="Something went wrong")
    updates = [u async for u in converter.convert(event)]

    assert len(updates) == 1
    d = _dump(updates[0])
    assert "Error" in str(d["content"])


@pytest.mark.unit
async def test_run_error_event_yields_turn_complete_when_supported(
    converter_with_turn_complete: ACPEventConverter,
):
    """RunErrorEvent should yield error text + TurnCompleteUpdate when supported."""
    event = RunErrorEvent(message="Model API returned 500", agent_name="engineer")
    updates = [u async for u in converter_with_turn_complete.convert(event)]

    assert len(updates) == 2

    d_text = _dump(updates[0])
    assert d_text["session_update"] == "agent_message_chunk"
    assert "Error" in str(d_text["content"])
    assert "engineer" in str(d_text["content"])
    assert "Model API returned 500" in str(d_text["content"])

    d_turn = _dump(updates[1])
    assert d_turn["session_update"] == "turn_complete"
    assert d_turn["stop_reason"] == "refusal"


@pytest.mark.unit
async def test_run_error_event_resets_converter_state(
    converter_with_turn_complete: ACPEventConverter,
):
    """RunErrorEvent should reset converter state after emitting updates."""
    tc_event = ToolCallStartEvent(
        tool_call_id="tc_001",
        tool_name="bash",
        title="Running command",
    )
    _ = [u async for u in converter_with_turn_complete.convert(tc_event)]
    assert "tc_001" in converter_with_turn_complete._tool_states

    event = RunErrorEvent(message="Something went wrong", agent_name="engineer")
    _ = [u async for u in converter_with_turn_complete.convert(event)]

    assert "tc_001" not in converter_with_turn_complete._tool_states


@pytest.mark.unit
async def test_run_error_event_cancels_pending_tools(
    converter_with_turn_complete: ACPEventConverter,
):
    """RunErrorEvent should emit ToolCallProgress for pending tools before resetting."""
    tc_event = ToolCallStartEvent(
        tool_call_id="tc_001",
        tool_name="bash",
        title="Running command",
    )
    _ = [u async for u in converter_with_turn_complete.convert(tc_event)]

    event = RunErrorEvent(message="Something went wrong", agent_name="engineer")
    updates = [u async for u in converter_with_turn_complete.convert(event)]

    progress_dumps = [
        _dump(u) for u in updates if _dump(u).get("session_update") == "tool_call_update"
    ]
    assert len(progress_dumps) == 1
    assert progress_dumps[0]["tool_call_id"] == "tc_001"
    assert progress_dumps[0]["status"] == "completed"


# ---------------------------------------------------------------------------
# RunFailedEvent
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_run_failed_event_produces_text_and_turn_complete(
    converter_with_turn_complete: ACPEventConverter,
):
    """RunFailedEvent should produce error text + TurnCompleteUpdate."""
    event = RunFailedEvent(
        run_id="run_abc123",
        session_id="ses_xyz",
        exception=RuntimeError("Agent crashed"),
    )
    updates = [u async for u in converter_with_turn_complete.convert(event)]

    assert len(updates) >= 2

    d_text = _dump(updates[0])
    assert d_text["session_update"] == "agent_message_chunk"
    text_str = str(d_text["content"])
    assert "Run Failed" in text_str
    assert "run_abc123" in text_str
    assert "Agent crashed" in text_str

    d_turn = _dump(updates[1])
    assert d_turn["session_update"] == "turn_complete"
    assert d_turn["stop_reason"] == "end_turn"


@pytest.mark.unit
async def test_run_failed_event_without_turn_complete_support(converter: ACPEventConverter):
    """RunFailedEvent should only produce error text when client lacks turn_complete support."""
    event = RunFailedEvent(
        run_id="run_abc123",
        session_id="ses_xyz",
        exception=RuntimeError("Agent crashed"),
    )
    updates = [u async for u in converter.convert(event)]

    assert len(updates) == 1
    d = _dump(updates[0])
    assert "Run Failed" in str(d["content"])
    assert "Agent crashed" in str(d["content"])


@pytest.mark.unit
async def test_run_failed_event_resets_converter_state(
    converter_with_turn_complete: ACPEventConverter,
):
    """RunFailedEvent should reset converter state after emitting updates."""
    tc_event = ToolCallStartEvent(
        tool_call_id="tc_001",
        tool_name="bash",
        title="Running command",
    )
    _ = [u async for u in converter_with_turn_complete.convert(tc_event)]
    assert "tc_001" in converter_with_turn_complete._tool_states

    event = RunFailedEvent(
        run_id="run_abc123",
        session_id="ses_xyz",
        exception=RuntimeError("Agent crashed"),
    )
    _ = [u async for u in converter_with_turn_complete.convert(event)]

    assert "tc_001" not in converter_with_turn_complete._tool_states


@pytest.mark.unit
async def test_run_failed_event_cancels_pending_tools_before_turn_complete(
    converter_with_turn_complete: ACPEventConverter,
):
    """RunFailedEvent should cancel tools BEFORE yielding TurnCompleteUpdate."""
    tc_event = ToolCallStartEvent(
        tool_call_id="tc_001",
        tool_name="bash",
        title="Running command",
    )
    _ = [u async for u in converter_with_turn_complete.convert(tc_event)]

    event = RunFailedEvent(
        run_id="run_abc123",
        session_id="ses_xyz",
        exception=RuntimeError("Agent crashed"),
    )
    updates = [u async for u in converter_with_turn_complete.convert(event)]

    update_types = [_dump(u).get("session_update") for u in updates]
    tool_idx = update_types.index("tool_call_update")
    turn_idx = update_types.index("turn_complete")
    assert tool_idx < turn_idx, "tool cleanup must come before turn_complete"


# ---------------------------------------------------------------------------
# Reset idempotency
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_reset_idempotent(converter: ACPEventConverter):
    """reset() can be called multiple times without error - verifies idempotent behavior."""
    # First call
    converter.reset()
    # Second call should not raise
    converter.reset()
    # State should be clean after multiple resets
    assert converter._tool_states == {}
    assert converter._current_tool_inputs == {}
    assert converter._subagent_headers == set()
    assert converter._subagent_content == {}
    assert converter._child_sessions == set()
    assert converter.last_usage is None


@pytest.mark.unit
async def test_stream_complete_single_reset(converter_with_turn_complete: ACPEventConverter):
    """StreamCompleteEvent triggers reset exactly once, after emitting all updates."""
    # Set up tool state to verify cleanup after StreamCompleteEvent
    tc_event = ToolCallStartEvent(
        tool_call_id="tc_001",
        tool_name="bash",
        title="Running command",
    )
    _ = [u async for u in converter_with_turn_complete.convert(tc_event)]
    assert "tc_001" in converter_with_turn_complete._tool_states

    msg = ChatMessage[str](
        content="Final response",
        role="assistant",
        usage=RequestUsage(input_tokens=40, output_tokens=60, details={}),
    )
    event = StreamCompleteEvent(message=msg)
    updates = [u async for u in converter_with_turn_complete.convert(event)]

    # ACP spec: all session/update notifications must precede the turn_complete barrier
    update_types = [_dump(u).get("session_update") for u in updates]
    assert "usage_update" in update_types
    assert "turn_complete" in update_types
    if "tool_call_update" in update_types:
        assert update_types.index("tool_call_update") < update_types.index("turn_complete")

    # Verify state is reset
    assert converter_with_turn_complete._tool_states == {}
    assert converter_with_turn_complete._current_tool_inputs == {}
    assert converter_with_turn_complete._subagent_headers == set()
    assert converter_with_turn_complete.last_usage is None
