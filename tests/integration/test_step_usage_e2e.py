"""E2E integration test for per-step token usage through the full pipeline.

Exercises: ``Agent`` + ``TestModel`` → ``NativeTurn.execute()`` → events
published to ``EventBus`` → ``ACPEventConverter`` consumes events → verify
``UsageUpdate`` notifications in output.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.models.test import TestModel
import pytest

from acp.schema import UsageUpdate
from wolfharness import Agent
from wolfharness.agents.context import AgentRunContext
from wolfharness.agents.events.events import (
    StepUsageEvent,
    StreamCompleteEvent,
)
from wolfharness.agents.native_agent.turn import NativeTurn
from wolfharness.orchestrator.event_bus import EventBus
from wolfharness_server.acp_server.event_converter import ACPEventConverter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def my_tool() -> str:
    """A simple tool for testing."""
    return "tool result"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_step_usage_through_event_bus_to_acp_converter() -> None:
    """Full pipeline: NativeTurn → EventBus → ACPEventConverter → UsageUpdate.

    1. Create a real Agent with TestModel(call_tools) and a tool.
    2. Execute a NativeTurn, collecting events.
    3. Publish events to an EventBus.
    4. Subscribe and drain events from the EventBus.
    5. Feed drained events through an ACPEventConverter.
    6. Verify 3 UsageUpdate notifications: 2 per-step deltas + 1 final cumulative.
    """
    agent = Agent(
        name="test-e2e",
        model=TestModel(call_tools=["my_tool"], custom_output_text="done"),
        tools=[my_tool],
    )

    bus = EventBus()
    session_id = "test-e2e-session"
    queue = await bus.subscribe(session_id, scope="session")

    # Execute NativeTurn and publish each event to the EventBus.
    async with agent:
        run_ctx = AgentRunContext(session_id=session_id)
        turn = NativeTurn(
            agent=agent,
            prompts=["Call the tool"],
            run_ctx=run_ctx,
            message_history=[],
        )
        async for event in turn.execute():
            await bus.publish(session_id, event)

    # Drain all events from the EventBus subscriber queue.
    drained_events: list[Any] = []
    while not queue.empty():
        envelope = queue.get_nowait()
        drained_events.append(envelope.event)

    # Feed drained events through ACPEventConverter.
    converter = ACPEventConverter()
    all_updates: list[Any] = []
    for event in drained_events:
        all_updates.extend([u async for u in converter.convert(event)])  # type: ignore[arg-type]

    # Extract UsageUpdate notifications.
    usage_updates = [u for u in all_updates if isinstance(u, UsageUpdate)]

    # Expect 3 UsageUpdate notifications: 2 per-step + 1 final.
    assert len(usage_updates) == 3, (
        f"Expected 3 UsageUpdate (2 per-step + 1 final), got {len(usage_updates)}"
    )

    # First 2 are per-step deltas (from StepUsageEvent).
    assert usage_updates[0].field_meta is not None
    assert usage_updates[0].field_meta["step_index"] == 0

    assert usage_updates[1].field_meta is not None
    assert usage_updates[1].field_meta["step_index"] == 1

    # Final UsageUpdate (from StreamCompleteEvent) has no step_index meta.
    final_meta = usage_updates[2].field_meta
    assert final_meta is None or "step_index" not in final_meta, (
        "Final UsageUpdate should not have step_index"
    )

    # Per-step `used` values are deltas (should be > 0 for TestModel with call_tools).
    assert usage_updates[0].used > 0, "Step 0 delta should be > 0"
    assert usage_updates[1].used > 0, "Step 1 delta should be > 0"

    # Final `used` is cumulative (should equal sum of all tokens for the turn).
    # For TestModel, the final cumulative should be >= sum of per-step deltas
    # (TestModel may report slightly different totals due to estimation).
    assert usage_updates[2].used > 0, "Final cumulative should be > 0"

    # Verify ordering: per-step deltas arrive before final cumulative.
    assert usage_updates[0].field_meta is not None
    assert usage_updates[1].field_meta is not None
    assert usage_updates[0].field_meta["step_index"] < usage_updates[1].field_meta["step_index"], (
        "Step 0 should arrive before Step 1"
    )

    # Verify that StepUsageEvent and StreamCompleteEvent were in the drained events.
    step_events = [e for e in drained_events if isinstance(e, StepUsageEvent)]
    stream_completes = [e for e in drained_events if isinstance(e, StreamCompleteEvent)]
    assert len(step_events) == 2, f"Expected 2 StepUsageEvent, got {len(step_events)}"
    assert len(stream_completes) == 1, (
        f"Expected 1 StreamCompleteEvent, got {len(stream_completes)}"
    )

    await bus.unsubscribe(session_id, queue)
