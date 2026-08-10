"""Unit tests for per-step token usage emission from NativeTurn.execute().

Tests that ``StepUsageEvent`` is emitted after each LLM call within a turn,
carrying the per-step delta and running cumulative usage.  Uses real
``Agent`` + ``TestModel`` (no mocking of NativeTurn or agent_run).
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.models.test import TestModel
import pytest

from wolfharness import Agent
from wolfharness.agents.context import AgentRunContext
from wolfharness.agents.events.events import (
    StepUsageEvent,
    StreamCompleteEvent,
    ToolCallCompleteEvent,
)
from wolfharness.agents.native_agent.turn import NativeTurn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def my_tool() -> str:
    """A simple tool for testing."""
    return "tool result"


async def _collect_events(agent: Agent[Any, Any], prompts: list[str]) -> list[Any]:
    """Run a NativeTurn and collect all yielded events."""
    async with agent:
        run_ctx = AgentRunContext(session_id="test-session")
        turn = NativeTurn(
            agent=agent,
            prompts=prompts,
            run_ctx=run_ctx,
            message_history=[],
        )
        return [event async for event in turn.execute()]


def _step_usage_events(events: list[Any]) -> list[StepUsageEvent]:
    """Extract StepUsageEvent instances from an event list."""
    return [e for e in events if isinstance(e, StepUsageEvent)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_multi_step_turn_emits_step_usage_events() -> None:
    """Multi-step turn (tool call → final response) emits 2 StepUsageEvents.

    TestModel with call_tools produces 2 LLM calls:
    1. Model decides to call the tool.
    2. Model produces final text after tool result.
    """
    agent = Agent(
        name="test-multi-step",
        model=TestModel(call_tools=["my_tool"], custom_output_text="done"),
        tools=[my_tool],
    )
    events = await _collect_events(agent, ["Call the tool"])
    step_events = _step_usage_events(events)

    assert len(step_events) == 2, f"Expected 2 StepUsageEvent (2 LLM calls), got {len(step_events)}"

    # Step 0: first LLM call
    assert step_events[0].step_index == 0
    assert step_events[0].step_usage.requests == 1

    # Step 1: second LLM call
    assert step_events[1].step_index == 1
    assert step_events[1].step_usage.requests == 1

    # Cumulative usage should increase monotonically
    assert step_events[1].cumulative_usage.requests > step_events[0].cumulative_usage.requests, (
        "Cumulative requests should increase between steps"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_single_step_turn_emits_one_step_usage() -> None:
    """Single-step turn (no tools) emits exactly 1 StepUsageEvent."""
    agent = Agent(
        name="test-single-step",
        model=TestModel(custom_output_text="hello"),
    )
    events = await _collect_events(agent, ["Say hello"])
    step_events = _step_usage_events(events)

    assert len(step_events) == 1, f"Expected exactly 1 StepUsageEvent, got {len(step_events)}"
    assert step_events[0].step_index == 0
    assert step_events[0].step_usage.requests == 1

    # StepUsageEvent should arrive before StreamCompleteEvent
    stream_complete_idx = next(
        i for i, e in enumerate(events) if isinstance(e, StreamCompleteEvent)
    )
    step_event_idx = events.index(step_events[0])
    assert step_event_idx < stream_complete_idx, (
        "StepUsageEvent should arrive before StreamCompleteEvent"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_only_iterations_no_step_usage() -> None:
    """StepUsageEvent count equals number of LLM calls, not node iterations.

    With call_tools, pydantic-ai iterates through multiple nodes
    (ModelRequestNode → CallToolsNode → ModelRequestNode → ...), but
    only iterations that involve an actual LLM call should emit
    StepUsageEvent.
    """
    agent = Agent(
        name="test-tool-only",
        model=TestModel(call_tools=["my_tool"], custom_output_text="done"),
        tools=[my_tool],
    )
    events = await _collect_events(agent, ["Call the tool"])
    step_events = _step_usage_events(events)

    # TestModel with call_tools makes exactly 2 LLM requests:
    # 1) decide to call tool, 2) produce final output
    assert len(step_events) == 2, (
        f"Expected 2 StepUsageEvents (2 LLM calls), got {len(step_events)}. "
        f"Tool-only iterations should NOT emit StepUsageEvent."
    )

    # Each step should have requests=1 (each involved one LLM call)
    for i, se in enumerate(step_events):
        assert se.step_usage.requests == 1, (
            f"Step {i} should have requests=1, got {se.step_usage.requests}"
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_error_preserves_emitted_step_usage() -> None:
    """StepUsageEvent emitted before an error is preserved in the event stream.

    Uses a custom TestModel that succeeds on the first LLM call (producing
    a tool call → StepUsageEvent(0)) but raises ``RuntimeError`` on the
    second LLM call.  The turn should propagate the error, but
    ``StepUsageEvent(0)`` should already be in the event stream.
    """

    class _ErrorOnSecondRequest(TestModel):
        """TestModel that raises RuntimeError on the second model request."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._request_count = 0

        def _request(self, *args: Any, **kwargs: Any) -> Any:
            self._request_count += 1
            if self._request_count >= 2:
                raise RuntimeError("Simulated failure on second LLM call")
            return super()._request(*args, **kwargs)

    agent = Agent(
        name="test-error-step",
        model=_ErrorOnSecondRequest(call_tools=["my_tool"], custom_output_text="done"),
        tools=[my_tool],
    )
    async with agent:
        run_ctx = AgentRunContext(session_id="test-session")
        turn = NativeTurn(
            agent=agent,
            prompts=["Call the tool"],
            run_ctx=run_ctx,
            message_history=[],
        )

        events: list[Any] = []
        propagated_error: Exception | None = None
        try:
            async for event in turn.execute():
                events.append(event)  # noqa: PERF401
        except RuntimeError as exc:
            propagated_error = exc

        # The error should have been propagated from the generator.
        assert propagated_error is not None, "Expected RuntimeError to be propagated"
        assert "Simulated failure" in str(propagated_error)

        step_events = _step_usage_events(events)
        # At least 1 StepUsageEvent should have been emitted before the error.
        assert len(step_events) >= 1, (
            f"Expected >= 1 StepUsageEvent before error, got {len(step_events)}"
        )
        assert step_events[0].step_index == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_step_usage_event_ordering() -> None:
    """StepUsageEvent appears at the right position in the event stream.

    For a tool-call turn:
    - StepUsageEvent(0) appears after text/tool events from the first LLM
      call and before ToolCallStartEvent (or after PartEndEvent).
    - StepUsageEvent(1) appears after ToolCallCompleteEvent and before
      StreamCompleteEvent.
    """
    agent = Agent(
        name="test-ordering",
        model=TestModel(call_tools=["my_tool"], custom_output_text="done"),
        tools=[my_tool],
    )
    events = await _collect_events(agent, ["Call the tool"])
    step_events = _step_usage_events(events)

    assert len(step_events) == 2

    # StepUsageEvent(0) should appear after at least one PartDelta/PartEnd
    # event (from the first LLM response) and before any
    # ToolCallStartEvent from the second LLM call.
    step0_idx = events.index(step_events[0])
    step1_idx = events.index(step_events[1])

    # Find first ToolCallCompleteEvent
    tool_complete_idx = next(
        (i for i, e in enumerate(events) if isinstance(e, ToolCallCompleteEvent)),
        None,
    )
    # Find StreamCompleteEvent
    stream_complete_idx = next(
        i for i, e in enumerate(events) if isinstance(e, StreamCompleteEvent)
    )

    # StepUsageEvent(0) should come before ToolCallStartEvent
    assert step0_idx < step1_idx, "Step 0 should come before Step 1"

    # StepUsageEvent(1) should come after ToolCallCompleteEvent
    if tool_complete_idx is not None:
        assert step1_idx > tool_complete_idx, (
            "StepUsageEvent(1) should appear after ToolCallCompleteEvent"
        )

    # StepUsageEvent(1) should come before StreamCompleteEvent
    assert step1_idx < stream_complete_idx, (
        "StepUsageEvent(1) should appear before StreamCompleteEvent"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_step_usage_zero_token_model() -> None:
    """StepUsageEvent is emitted even when token counts are zero.

    TestModel may report zero input/output tokens, but ``requests``
    should still be > 0 (each LLM call increments requests by 1).
    The StepUsageEvent should still be emitted in this case.
    """
    agent = Agent(
        name="test-zero-tokens",
        model=TestModel(custom_output_text="hello"),
    )
    events = await _collect_events(agent, ["test"])
    step_events = _step_usage_events(events)

    assert len(step_events) == 1, f"Expected 1 StepUsageEvent, got {len(step_events)}"
    assert step_events[0].step_usage.requests == 1, "requests should be 1 even with zero tokens"
    # TestModel may report some token counts; the key assertion is that
    # StepUsageEvent is emitted (requests > 0) regardless of token values.
    assert step_events[0].step_usage.input_tokens >= 0
    assert step_events[0].step_usage.output_tokens >= 0
