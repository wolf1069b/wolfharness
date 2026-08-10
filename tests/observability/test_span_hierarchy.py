"""Tests for span hierarchy instrumentation.

Verifies that OpenTelemetry spans created via ``logfire.span()`` (and
directly via OTel SDK) form the correct parent-child nesting. These
tests use OTel's ``InMemorySpanExporter`` directly rather than
``logfire.testing.capfire`` because the session-level conftest fixture
stubs ``logfire.configure`` as a no-op.

The tested spans correspond to the instrumentation added during the
fix-span-instrumentation change:

- ``delegation.subagent`` → from ``SubagentCapability.spawn_subagent()``
- ``orchestration.run_handle.start`` → from ``RunHandle.start()``
- ``turn.native`` / ``turn.acp`` → from ``NativeTurn.execute()`` /
  ``ACPTurn.execute()``
- ``team.execute_parallel`` → from ``base_team._execute_parallel()``
- ``subagent.background_task`` → from ``subagent_tools._safe_background_run()``
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.sampling import ALWAYS_ON as ALWAYS_ON_SAMPLER
import pytest


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from collections.abc import Generator

    from opentelemetry.trace import Tracer


@pytest.fixture
def in_memory_tracer() -> Generator[tuple[Tracer, InMemorySpanExporter]]:
    """Set up an OTel tracer with an in-memory span exporter.

    Uses a local ``TracerProvider`` directly (not the global one) to
    avoid OTel SDK's 'Overriding of current TracerProvider is not
    allowed' restriction when running multiple tests.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider(sampler=ALWAYS_ON_SAMPLER)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    return tracer, exporter


def _span_by_name(spans: tuple[ReadableSpan, ...]) -> dict[str, ReadableSpan]:
    """Build a ``{name: span}`` lookup from a flat span list."""
    return {s.name: s for s in spans}


def _get_span_attr(span: ReadableSpan, key: str) -> Any:
    """Safely fetch an attribute from a span, handling ``None`` attrs."""
    attrs = span.attributes
    if attrs is None:
        return None
    return attrs.get(key)  # type: ignore[return-value]


def _assert_child_of(
    child: ReadableSpan,
    parent: ReadableSpan,
    *,
    child_name: str = "",
    parent_name: str = "",
) -> None:
    """Assert that *child* is a direct child of *parent* by span ID."""
    assert child.parent is not None, (
        f"'{child_name or child.name}' has no parent span — expected parent "
        f"'{parent_name or parent.name}'"
    )
    assert child.parent.span_id == parent.context.span_id, (
        f"'{child_name or child.name}'.parent.span_id "
        f"({child.parent.span_id}) does not match "
        f"'{parent_name or parent.name}'.context.span_id "
        f"({parent.context.span_id})"
    )


# ---------------------------------------------------------------------------
# Test 1: Delegation span hierarchy
# ---------------------------------------------------------------------------


def test_delegation_span_hierarchy(
    in_memory_tracer: tuple[Tracer, InMemorySpanExporter],
) -> None:
    """Verify the delegation span nesting.

    ``delegation.subagent → orchestration.run_handle.start → turn.native``.
    """
    tracer, exporter = in_memory_tracer

    with (
        tracer.start_as_current_span(
            "delegation.subagent",
            attributes={"parent_session_id": "parent", "child_agent_name": "child"},
        ),
        tracer.start_as_current_span(
            "orchestration.run_handle.start",
            attributes={"session_id": "child", "agent_type": "native"},
        ),
        tracer.start_as_current_span(
            "turn.native",
            attributes={"turn_id": "t1", "session_id": "child"},
        ),
    ):
        pass  # simulate turn execution

    spans = _span_by_name(exporter.get_finished_spans())

    # All three spans should be present
    assert "delegation.subagent" in spans
    assert "orchestration.run_handle.start" in spans
    assert "turn.native" in spans

    delegation = spans["delegation.subagent"]
    run_handle = spans["orchestration.run_handle.start"]
    turn = spans["turn.native"]

    # delegation is the root → no parent
    assert delegation.parent is None

    # run_handle is child of delegation
    _assert_child_of(run_handle, delegation)

    # turn is child of run_handle
    _assert_child_of(turn, run_handle)

    # Verify span attributes
    assert _get_span_attr(delegation, "parent_session_id") == "parent"
    assert _get_span_attr(delegation, "child_agent_name") == "child"

    assert _get_span_attr(run_handle, "session_id") == "child"
    assert _get_span_attr(run_handle, "agent_type") == "native"

    assert _get_span_attr(turn, "turn_id") == "t1"
    assert _get_span_attr(turn, "session_id") == "child"


# ---------------------------------------------------------------------------
# Test 2: Team parallel execution and sequential spans
# ---------------------------------------------------------------------------


def test_team_parallel_span_exists(
    in_memory_tracer: tuple[Tracer, InMemorySpanExporter],
) -> None:
    """Verify that team parallel and sequential spans exist and are captured.

    These spans are created via ``@logfire.instrument`` on
    ``base_team._execute_parallel()`` and ``_execute_sequential()``.
    """
    tracer, exporter = in_memory_tracer

    # Simulate a parallel execution span with member spans
    with tracer.start_as_current_span(
        "team.execute_parallel",
        attributes={"agent_names": "agent1,agent2", "mode": "parallel"},
    ):
        with tracer.start_as_current_span(
            "agent.run",
            attributes={"agent_name": "agent1"},
        ):
            pass
        with tracer.start_as_current_span(
            "agent.run",
            attributes={"agent_name": "agent2"},
        ):
            pass

    spans = _span_by_name(exporter.get_finished_spans())

    assert "team.execute_parallel" in spans
    parallel_span = spans["team.execute_parallel"]

    # Should be a root span
    assert parallel_span.parent is None

    # Should have meaningful attributes
    assert _get_span_attr(parallel_span, "mode") == "parallel"
    assert _get_span_attr(parallel_span, "agent_names") == "agent1,agent2"


def test_team_sequential_span_exists(
    in_memory_tracer: tuple[Tracer, InMemorySpanExporter],
) -> None:
    """Verify that ``team.execute_sequential`` span exists and nests child member spans."""
    tracer, exporter = in_memory_tracer

    with (
        tracer.start_as_current_span(
            "team.execute_sequential",
            attributes={"agent_names": "agent1,agent2,agent3", "mode": "sequential"},
        ),
        tracer.start_as_current_span(
            "agent.run",
            attributes={"agent_name": "agent1"},
        ),
    ):
        pass

    spans = _span_by_name(exporter.get_finished_spans())

    assert "team.execute_sequential" in spans
    seq_span = spans["team.execute_sequential"]

    assert seq_span.parent is None
    assert _get_span_attr(seq_span, "mode") == "sequential"


# ---------------------------------------------------------------------------
# Test 3: Background task span
# ---------------------------------------------------------------------------


def test_background_task_span(
    in_memory_tracer: tuple[Tracer, InMemorySpanExporter],
) -> None:
    """Verify that ``subagent.background_task`` span exists and carries a ``task_id`` attribute.

    This simulates the ``with logfire.span()`` inside the
    ``_safe_background_run()`` nested function in
    ``wolfharness_toolsets/builtin/subagent_tools.py``.
    """
    tracer, exporter = in_memory_tracer

    task_id = "bg_task_001"
    with (
        tracer.start_as_current_span(
            "subagent.background_task",
            attributes={
                "task_id": task_id,
                "parent_session_id": "ses_parent",
                "child_agent_name": "researcher",
            },
        ),
        tracer.start_as_current_span(
            "agent.run",
            attributes={"agent_name": "researcher"},
        ),
    ):
        pass

    spans = _span_by_name(exporter.get_finished_spans())

    assert "subagent.background_task" in spans
    bg_span = spans["subagent.background_task"]

    # Must carry task_id attribute
    assert _get_span_attr(bg_span, "task_id") == task_id
    assert _get_span_attr(bg_span, "child_agent_name") == "researcher"

    # Should be a root span (it's fired from a separate background task)
    assert bg_span.parent is None

    # Child agent.run span should nest under background_task
    agent_run_span = spans["agent.run"]
    _assert_child_of(agent_run_span, bg_span)


# ---------------------------------------------------------------------------
# Test 5: Nested async generator span leak on aclose()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="Documents the pre-fix bug: async for does not close sub-generators "
    "on GeneratorExit. The fix uses contextlib.aclosing() — see "
    "test_nested_async_generator_aclosing_fix.",
    strict=True,
)
async def test_nested_async_generator_span_leak(
    in_memory_tracer: tuple[Tracer, InMemorySpanExporter],
) -> None:
    """Reproduce: closing outer async generator leaks inner generator spans.

    When ``RunHandle.start()`` is closed via ``aclose()`` (from
    ``_consume_run``'s ``finally`` block), the nested async generators
    ``_execute_turn()`` and ``turn.execute()`` are NOT automatically
    closed. Their ``safe_span`` ``finally`` blocks never run, so the
    spans are never ended → never exported → "Missing Span" in SigNoz.

    This test simulates the chain:
        start() → _execute_turn() → turn.execute()

    Each level uses ``tracer.start_as_current_span()`` (same pattern as
    ``safe_span``). The outer generator is closed via ``aclose()``
    after receiving one event. We assert that ALL spans are ended.
    """
    tracer, exporter = in_memory_tracer

    async def turn_execute() -> Any:
        """Innermost generator — simulates NativeTurn.execute()."""
        with tracer.start_as_current_span("turn.native"):
            yield "event"
            # Keep generator alive (real generator does more work)
            await asyncio.sleep(999)

    async def execute_turn() -> Any:
        """Middle generator — simulates RunHandle._execute_turn()."""
        with tracer.start_as_current_span("orchestration.run_handle.execute_turn"):
            async for event in turn_execute():
                yield event

    async def start() -> Any:
        """Outer generator — simulates RunHandle.start()."""
        with tracer.start_as_current_span("orchestration.run_handle.start"):
            async for event in execute_turn():
                yield event

    gen = start()
    event = await gen.__anext__()
    assert event == "event"

    # Close the outer generator — simulates _consume_run's finally block
    await gen.aclose()

    # Force GC to close any leaked generators (non-deterministic, but
    # helps expose the issue even when CPython's refcounting is fast)
    import gc

    gc.collect()

    spans = _span_by_name(exporter.get_finished_spans())

    # outer span should be ended — its `with` block exits on GeneratorExit
    assert "orchestration.run_handle.start" in spans, "outer span should be ended"

    # BUG: inner spans are NOT ended because their generators are never
    # closed via aclose(). The `async for` loop in the outer generator
    # does NOT close the inner iterator when GeneratorExit is raised.
    assert "orchestration.run_handle.execute_turn" in spans, (
        "inner span should be ended — BUG: _execute_turn() generator is not closed on aclose()"
    )
    assert "turn.native" in spans, (
        "innermost span should be ended — BUG: turn.execute() generator is not closed on aclose()"
    )


@pytest.mark.asyncio
async def test_nested_async_generator_aclosing_fix(
    in_memory_tracer: tuple[Tracer, InMemorySpanExporter],
) -> None:
    """Verify that ``contextlib.aclosing()`` fixes the span leak.

    When nested async generators are wrapped with ``aclosing()``, the
    inner generators are properly closed when the outer generator exits,
    ensuring all spans are ended and exported.
    """
    import contextlib

    tracer, exporter = in_memory_tracer

    async def turn_execute() -> Any:
        with tracer.start_as_current_span("turn.native"):
            yield "event"
            await asyncio.sleep(999)

    async def execute_turn() -> Any:
        with tracer.start_as_current_span("orchestration.run_handle.execute_turn"):
            async with contextlib.aclosing(turn_execute()) as inner:
                async for event in inner:
                    yield event

    async def start() -> Any:
        with tracer.start_as_current_span("orchestration.run_handle.start"):
            async with contextlib.aclosing(execute_turn()) as mid:
                async for event in mid:
                    yield event

    gen = start()
    event = await gen.__anext__()
    assert event == "event"

    await gen.aclose()

    spans = _span_by_name(exporter.get_finished_spans())

    assert "orchestration.run_handle.start" in spans
    assert "orchestration.run_handle.execute_turn" in spans
    assert "turn.native" in spans
