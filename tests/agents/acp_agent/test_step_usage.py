"""Tests for per-step usage capture in ACP turn.

Tests that ``acp_to_native_event()`` correctly converts ACP ``UsageUpdate``
session updates to native ``StepUsageEvent`` objects, and that non-usage
session updates do not produce ``StepUsageEvent`` objects.

Also tests that ``ACPTurn.execute()`` correctly maintains ``cumulative_usage``
and ``step_index`` across multiple ``UsageUpdate`` events from the ACP stream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai import RunUsage
import pytest

from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    PromptResponse,
    ToolCallProgress,
    ToolCallStart,
    UsageUpdate,
)
from wolfharness.agents.acp_agent.acp_converters import acp_to_native_event
from wolfharness.agents.acp_agent.turn import ACPTurn
from wolfharness.agents.context import AgentRunContext
from wolfharness.agents.events import StepUsageEvent


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from acp.schema import ContentBlock, SessionUpdate


pytestmark = pytest.mark.unit


def test_usage_update_produces_step_usage_event() -> None:
    """UsageUpdate converts to StepUsageEvent with output_tokens mapped from used."""
    update = UsageUpdate(used=500, size=4096)
    event = acp_to_native_event(update)

    assert event is not None
    assert isinstance(event, StepUsageEvent)
    assert event.step_usage.output_tokens == 500
    assert event.step_usage.input_tokens == 0


def test_usage_update_with_step_index_and_cumulative() -> None:
    """UsageUpdate respects step_index and cumulative_usage parameters."""
    cumulative = RunUsage(input_tokens=100, output_tokens=200)
    update = UsageUpdate(used=50, size=4096)
    event = acp_to_native_event(update, step_index=3, cumulative_usage=cumulative)

    assert event is not None
    assert isinstance(event, StepUsageEvent)
    assert event.step_index == 3
    assert event.step_usage.output_tokens == 50
    assert event.cumulative_usage.input_tokens == 100
    assert event.cumulative_usage.output_tokens == 200


def test_usage_update_zero_tokens() -> None:
    """UsageUpdate with used=0 produces StepUsageEvent with zero output_tokens."""
    update = UsageUpdate(used=0, size=4096)
    event = acp_to_native_event(update)

    assert event is not None
    assert isinstance(event, StepUsageEvent)
    assert event.step_usage.output_tokens == 0


def test_usage_update_defaults() -> None:
    """UsageUpdate with no step_index/cumulative_usage uses sensible defaults."""
    update = UsageUpdate(used=42, size=8192)
    event = acp_to_native_event(update)

    assert event is not None
    assert isinstance(event, StepUsageEvent)
    assert event.step_index == 0
    assert event.cumulative_usage.output_tokens == 0
    assert event.cumulative_usage.input_tokens == 0


def test_non_usage_updates_no_step_usage() -> None:
    """Non-usage SessionUpdate variants do not produce StepUsageEvent."""
    updates = [
        AgentMessageChunk.text("hello"),
        AgentThoughtChunk.text("thinking"),
        AgentPlanUpdate(entries=[]),
        ToolCallStart(tool_call_id="tc1", title="bash", kind="execute"),
        ToolCallProgress(tool_call_id="tc1", status="completed"),
    ]

    for update in updates:
        event = acp_to_native_event(update)
        # Event may be None or a non-StepUsageEvent, but never StepUsageEvent
        assert not isinstance(event, StepUsageEvent), (
            f"Unexpected StepUsageEvent from {type(update).__name__}"
        )


def test_mixed_stream_no_false_step_usage() -> None:
    """A mixed stream of updates produces StepUsageEvent only for UsageUpdate."""
    updates = [
        AgentMessageChunk.text("hello"),
        UsageUpdate(used=100, size=4096),
        AgentThoughtChunk.text("thinking"),
        UsageUpdate(used=200, size=4096),
        AgentMessageChunk.text("world"),
    ]

    step_usage_count = 0
    for update in updates:
        event = acp_to_native_event(update)
        if isinstance(event, StepUsageEvent):
            step_usage_count += 1
            assert event.step_usage.output_tokens in (100, 200)

    assert step_usage_count == 2


def test_mixed_stream_with_step_index_progression() -> None:
    """Step index increments correctly across UsageUpdate events in a mixed stream."""
    updates = [
        AgentMessageChunk.text("hello"),
        UsageUpdate(used=100, size=4096),
        AgentThoughtChunk.text("thinking"),
        UsageUpdate(used=150, size=4096),
    ]

    step_index = 0
    cumulative = RunUsage()
    usage_events: list[StepUsageEvent] = []

    for update in updates:
        event = acp_to_native_event(
            update,
            step_index=step_index,
            cumulative_usage=cumulative,
        )
        if isinstance(event, StepUsageEvent):
            usage_events.append(event)
            cumulative = RunUsage(
                input_tokens=cumulative.input_tokens + event.step_usage.input_tokens,
                output_tokens=cumulative.output_tokens + event.step_usage.output_tokens,
            )
            step_index += 1

    assert len(usage_events) == 2
    assert usage_events[0].step_index == 0
    assert usage_events[1].step_index == 1
    assert usage_events[0].step_usage.output_tokens == 100
    assert usage_events[1].step_usage.output_tokens == 150
    # Cumulative after first event (passed to second event's converter)
    assert usage_events[1].cumulative_usage.output_tokens == 100


# ============================================================================
# ACPTurn.execute() cumulative_usage tracking
# ============================================================================


class _MockACPClient:
    """Mock ACP client that yields a predetermined sequence of session updates.

    Implements the :class:`~wolfharness.agents.acp_agent.turn.ACPClientProtocol`
    with a fixed list of ``SessionUpdate`` objects for ``stream_events()`` and
    an empty list for ``get_messages()``.
    """

    def __init__(self, updates: list[Any]) -> None:
        self._updates = list(updates)

    async def prompt(self, session_id: str, content: list[ContentBlock]) -> PromptResponse:
        return PromptResponse(stop_reason="end_turn")

    async def stream_events(self, response: PromptResponse) -> AsyncIterator[SessionUpdate]:
        for update in self._updates:
            yield update

    async def get_messages(self, session_id: str) -> list[SessionUpdate]:
        return []


@pytest.mark.asyncio
async def test_acp_turn_cumulative_usage_tracking() -> None:
    """ACPTurn.execute() maintains cumulative_usage and step_index across UsageUpdates.

    Feeds a mixed stream of session updates through ACPTurn.execute() and
    verifies that:
    - 2 StepUsageEvent instances are emitted (one per UsageUpdate).
    - step_index values are 0 and 1.
    - cumulative_usage on the second event reflects accumulation from the
      first step (output_tokens == 100).
    - step_usage.output_tokens on each event matches the UsageUpdate.used value.
    """
    updates: list[Any] = [
        AgentMessageChunk.text("hello"),
        UsageUpdate(used=100, size=4096),
        AgentThoughtChunk.text("thinking"),
        UsageUpdate(used=150, size=4096),
        AgentMessageChunk.text("world"),
    ]

    client = _MockACPClient(updates)
    run_ctx = AgentRunContext(session_id="test-acp-turn")
    turn = ACPTurn(
        acp_client=client,
        prompts=["test prompt"],
        run_ctx=run_ctx,
        session_id="test-acp-turn",
        agent_name="test-acp-agent",
    )

    events: list[Any] = [event async for event in turn.execute()]

    step_events = [e for e in events if isinstance(e, StepUsageEvent)]
    assert len(step_events) == 2, f"Expected 2 StepUsageEvent, got {len(step_events)}"

    # Step 0: first UsageUpdate(used=100)
    assert step_events[0].step_index == 0
    assert step_events[0].step_usage.output_tokens == 100
    # cumulative_usage is the accumulator BEFORE this step (empty)
    assert step_events[0].cumulative_usage.output_tokens == 0

    # Step 1: second UsageUpdate(used=150)
    assert step_events[1].step_index == 1
    assert step_events[1].step_usage.output_tokens == 150
    # cumulative_usage reflects accumulation from the first step (100)
    assert step_events[1].cumulative_usage.output_tokens == 100
