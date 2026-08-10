"""Tests for StepUsageEvent → ACP UsageUpdate conversion.

Verifies that the ACPEventConverter correctly converts StepUsageEvent
into per-step UsageUpdate notifications with delta token values, and
that the final StreamCompleteEvent UsageUpdate carries cumulative totals.
"""

from __future__ import annotations

from pydantic_ai import RequestUsage, RunUsage
import pytest

from acp.schema import UsageUpdate
from wolfharness.agents.events.events import (
    StepUsageEvent,
    StreamCompleteEvent,
)
from wolfharness.messaging import ChatMessage
from wolfharness_server.acp_server.event_converter import ACPEventConverter


pytestmark = [pytest.mark.integration, pytest.mark.anyio]


# ============================================================================
# Helpers
# ============================================================================


async def _convert(event: object) -> list[object]:
    """Run a single event through a fresh ACPEventConverter and collect outputs."""
    converter = ACPEventConverter()
    return [u async for u in converter.convert(event)]  # type: ignore[arg-type]


async def _convert_many(events: list[object]) -> list[object]:
    """Run multiple events through a single ACPEventConverter sequentially."""
    converter = ACPEventConverter()
    all_updates: list[object] = []
    for event in events:
        async for update in converter.convert(event):  # type: ignore[arg-type]
            all_updates.append(update)  # noqa: PERF401
    return all_updates


# ============================================================================
# Test 1: Single StepUsageEvent produces a UsageUpdate
# ============================================================================


async def test_step_usage_produces_usage_update() -> None:
    """A single StepUsageEvent yields one UsageUpdate with delta and cumulative.

    The ``used`` field carries the per-step delta tokens, ``size`` carries
    the cumulative total, and ``field_meta`` includes ``step_index`` plus
    serialized ``step_usage`` and ``cumulative_usage``.
    """
    step_usage = RunUsage(input_tokens=50, output_tokens=30, requests=1)
    cumulative_usage = RunUsage(input_tokens=50, output_tokens=30, requests=1)
    event = StepUsageEvent(
        step_index=0,
        step_usage=step_usage,
        cumulative_usage=cumulative_usage,
    )

    updates = await _convert(event)
    assert len(updates) == 1

    update = updates[0]
    assert isinstance(update, UsageUpdate)
    # Delta (per-step) tokens in "used"
    assert update.used == 80  # 50 + 30
    # Cumulative tokens in "size"
    assert update.size == 80
    assert update.cost is None
    # Extension fields in field_meta
    assert update.field_meta is not None
    assert update.field_meta["step_index"] == 0
    assert "step_usage" in update.field_meta
    assert "cumulative_usage" in update.field_meta
    # step_usage in meta should have the delta values
    step_meta = update.field_meta["step_usage"]
    assert step_meta["input_tokens"] == 50
    assert step_meta["output_tokens"] == 30
    assert step_meta["total_tokens"] == 80


# ============================================================================
# Test 2: Multiple StepUsageEvent + StreamCompleteEvent ordering
# ============================================================================


async def test_multiple_step_usage_events_ordered() -> None:
    """StepUsageEvent(0) → StepUsageEvent(1) → StreamCompleteEvent ordering.

    Per-step UsageUpdate notifications arrive in order, with the final
    cumulative UsageUpdate from StreamCompleteEvent arriving LAST.
    """
    events: list[object] = [
        StepUsageEvent(
            step_index=0,
            step_usage=RunUsage(input_tokens=100, output_tokens=20, requests=1),
            cumulative_usage=RunUsage(input_tokens=100, output_tokens=20, requests=1),
        ),
        StepUsageEvent(
            step_index=1,
            step_usage=RunUsage(input_tokens=50, output_tokens=10, requests=1),
            cumulative_usage=RunUsage(input_tokens=150, output_tokens=30, requests=2),
        ),
        StreamCompleteEvent(
            message=ChatMessage(
                role="assistant",
                content="done",
                usage=RequestUsage(input_tokens=150, output_tokens=30),
            )
        ),
    ]

    all_updates = await _convert_many(events)

    # Should have 3 UsageUpdate notifications: 2 per-step + 1 final
    usage_updates = [u for u in all_updates if isinstance(u, UsageUpdate)]
    assert len(usage_updates) == 3

    # Step 0: delta = 120 tokens
    assert usage_updates[0].used == 120  # 100 + 20
    assert usage_updates[0].field_meta is not None
    assert usage_updates[0].field_meta["step_index"] == 0

    # Step 1: delta = 60 tokens
    assert usage_updates[1].used == 60  # 50 + 10
    assert usage_updates[1].field_meta is not None
    assert usage_updates[1].field_meta["step_index"] == 1

    # Final: cumulative = 180 tokens (from StreamCompleteEvent)
    assert usage_updates[2].used == 180  # 150 + 30
    # Final UsageUpdate from StreamCompleteEvent has no step_index meta
    assert usage_updates[2].field_meta is None or "step_index" not in (
        usage_updates[2].field_meta or {}
    )


async def test_only_stream_complete_backward_compat() -> None:
    """Only StreamCompleteEvent (no StepUsageEvent) yields exactly 1 UsageUpdate.

    Backward compatibility with agents that don't emit per-step usage.
    """
    converter = ACPEventConverter()
    event = StreamCompleteEvent(
        message=ChatMessage(
            role="assistant",
            content="done",
            usage=RequestUsage(input_tokens=100, output_tokens=50),
        )
    )

    updates = [u async for u in converter.convert(event)]
    usage_updates = [u for u in updates if isinstance(u, UsageUpdate)]
    assert len(usage_updates) == 1
    assert usage_updates[0].used == 150  # 100 + 50


# ============================================================================
# Test 3: Final usage not double-counted
# ============================================================================


async def test_final_usage_not_double_counted() -> None:
    """2 StepUsageEvent + StreamCompleteEvent: 3 UsageUpdate, final is cumulative.

    The final ``used`` equals the cumulative total from StreamCompleteEvent,
    not the sum of per-step ``used`` values.
    """
    events: list[object] = [
        StepUsageEvent(
            step_index=0,
            step_usage=RunUsage(input_tokens=40, output_tokens=10, requests=1),
            cumulative_usage=RunUsage(input_tokens=40, output_tokens=10, requests=1),
        ),
        StepUsageEvent(
            step_index=1,
            step_usage=RunUsage(input_tokens=60, output_tokens=20, requests=1),
            cumulative_usage=RunUsage(input_tokens=100, output_tokens=30, requests=2),
        ),
        StreamCompleteEvent(
            message=ChatMessage(
                role="assistant",
                content="done",
                usage=RequestUsage(input_tokens=100, output_tokens=30),
            )
        ),
    ]

    all_updates = await _convert_many(events)

    usage_updates = [u for u in all_updates if isinstance(u, UsageUpdate)]
    assert len(usage_updates) == 3

    # Per-step deltas: 50 and 80
    assert usage_updates[0].used == 50  # 40 + 10
    assert usage_updates[1].used == 80  # 60 + 20

    # Final cumulative: 130 (from RequestUsage directly, not sum of deltas)
    assert usage_updates[2].used == 130  # 100 + 30


async def test_zero_token_step_final_still_correct() -> None:
    """Per-step with zero tokens, final still carries correct cumulative.

    Non-happy-path: a step that used zero tokens should not corrupt the
    final cumulative UsageUpdate from StreamCompleteEvent.
    """
    events: list[object] = [
        StepUsageEvent(
            step_index=0,
            step_usage=RunUsage(input_tokens=0, output_tokens=0, requests=0),
            cumulative_usage=RunUsage(input_tokens=0, output_tokens=0, requests=0),
        ),
        StreamCompleteEvent(
            message=ChatMessage(
                role="assistant",
                content="done",
                usage=RequestUsage(input_tokens=200, output_tokens=100),
            )
        ),
    ]

    all_updates = await _convert_many(events)

    usage_updates = [u for u in all_updates if isinstance(u, UsageUpdate)]
    assert len(usage_updates) == 2

    # Step 0: zero tokens
    assert usage_updates[0].used == 0

    # Final: cumulative = 300
    assert usage_updates[1].used == 300  # 200 + 100


# ============================================================================
# Test 4: Cache and reasoning tokens in field_meta
# ============================================================================


async def test_step_usage_cache_and_reasoning_tokens() -> None:
    """StepUsageEvent with cache_read, cache_write, and reasoning tokens.

    Verifies that ``cache_read_tokens``, ``cache_write_tokens``, and
    ``details={"reasoning_tokens": N}`` are correctly propagated into
    the ``field_meta["step_usage"]`` dict, and that ``used`` reflects
    only the basic input + output total.
    """
    step_usage = RunUsage(
        input_tokens=50,
        output_tokens=30,
        cache_read_tokens=10,
        cache_write_tokens=5,
        details={"reasoning_tokens": 15},
    )
    cumulative_usage = RunUsage(
        input_tokens=50,
        output_tokens=30,
        cache_read_tokens=10,
        cache_write_tokens=5,
        details={"reasoning_tokens": 15},
    )
    event = StepUsageEvent(
        step_index=0,
        step_usage=step_usage,
        cumulative_usage=cumulative_usage,
    )

    updates = await _convert(event)
    assert len(updates) == 1

    update = updates[0]
    assert isinstance(update, UsageUpdate)
    # used = input + output (basic total, excludes cache/reasoning)
    assert update.used == 80  # 50 + 30
    # size = cumulative total_tokens (same as used here, single step)
    assert update.size == 80

    # field_meta should contain step_usage with cache and reasoning tokens.
    # The ACP Usage model maps reasoning_tokens → thought_tokens and
    # cache_read/write_tokens → cached_read/write_tokens.
    assert update.field_meta is not None
    assert update.field_meta["step_index"] == 0
    step_meta = update.field_meta["step_usage"]
    assert step_meta["input_tokens"] == 50
    assert step_meta["output_tokens"] == 30
    assert step_meta["cached_read_tokens"] == 10
    assert step_meta["cached_write_tokens"] == 5
    assert step_meta["thought_tokens"] == 15
