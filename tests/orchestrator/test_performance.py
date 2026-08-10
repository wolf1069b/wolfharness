"""Performance and stress tests for SessionPool orchestration.

Consolidated from:
- test_benchmark.py (latency, throughput, scaling benchmarks)
- test_stress.py (1000+ sessions, rapid cycles, queue overflow)
"""

from __future__ import annotations

import asyncio
import gc
import time
from typing import TYPE_CHECKING, Any, ClassVar
from unittest.mock import MagicMock

import pytest

from wolfharness.agents.events import RunStartedEvent, StreamCompleteEvent
from wolfharness.lifecycle import DirectChannel, MemoryJournal
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.core import EventBus, SessionPool
from wolfharness.orchestrator.metrics import MetricsCollector


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from wolfharness import AgentPool


# ============================================================================
# Fixtures
# ============================================================================


def _stream_empty(queue: asyncio.Queue[Any]) -> bool:
    """Check if a subscriber queue has no buffered items."""
    return queue.empty()


class _MockTurn:
    """Minimal turn that yields RunStartedEvent + StreamCompleteEvent."""

    message_history: ClassVar[list[Any]] = []

    def __init__(self, *, delay: float = 0.0) -> None:
        self._delay = delay

    async def execute(self) -> AsyncIterator[Any]:
        if self._delay:
            await asyncio.sleep(self._delay)
        yield RunStartedEvent(session_id="default", run_id="run-1")
        yield StreamCompleteEvent(
            message=ChatMessage(content="ok", role="assistant"),
            session_id="default",
        )


@pytest.fixture
def mock_agent() -> MagicMock:
    """Return a mocked BaseAgent that yields a single event instantly."""
    agent = MagicMock()
    agent.create_turn = MagicMock(return_value=_MockTurn())
    return agent


@pytest.fixture
def mock_agent_with_delay() -> MagicMock:
    """Return a mocked BaseAgent with a small per-event delay."""
    agent = MagicMock()
    agent.create_turn = MagicMock(return_value=_MockTurn(delay=0.001))
    return agent


async def _attach_agent(
    pool: SessionPool,
    session_id: str,
    agent: MagicMock,
    real_pool: AgentPool,
) -> None:
    """Attach a mock agent to an existing session."""
    state, _ = await pool.sessions.get_or_create_session(session_id)
    state.agent = agent  # type: ignore[assignment]
    pool.sessions._session_agents[session_id] = agent  # type: ignore[assignment]
    real_pool.get_agent = MagicMock(return_value=agent)  # type: ignore[assignment]
    state._comm_channel = DirectChannel(MemoryJournal())


# ============================================================================
# Benchmark: session lifecycle
# ============================================================================


@pytest.mark.benchmark
async def test_benchmark_session_creation_latency(minimal_pool: AgentPool) -> None:
    """Measure time to create and close sessions at varying scales."""
    session_pool = minimal_pool.session_pool
    assert session_pool is not None

    results: dict[str, dict[str, float]] = {}

    for count in (1, 10, 100, 500):
        # Measure creation
        start = time.perf_counter()
        sids = [f"bench-create-{i}" for i in range(count)]
        await asyncio.gather(*[session_pool.create_session(sid) for sid in sids])
        create_time = time.perf_counter() - start

        # Measure close
        start = time.perf_counter()
        await asyncio.gather(*[session_pool.close_session(sid) for sid in sids])
        close_time = time.perf_counter() - start

        per_session_create = create_time / count * 1000  # ms
        per_session_close = close_time / count * 1000  # ms

        results[f"{count}_sessions"] = {
            "total_create_ms": create_time * 1000,
            "total_close_ms": close_time * 1000,
            "per_session_create_ms": per_session_create,
            "per_session_close_ms": per_session_close,
        }

        assert len(session_pool.sessions._sessions) == 0

    # Print benchmark results
    print("\n=== Session Lifecycle Benchmark ===")
    for label, metrics in results.items():
        print(
            f"{label}: create={metrics['per_session_create_ms']:.3f}ms/ea, "
            f"close={metrics['per_session_close_ms']:.3f}ms/ea"
        )

    # Sanity checks: should be reasonably fast (generous thresholds for CI variance)
    assert results["1_sessions"]["per_session_create_ms"] < 200
    assert results["500_sessions"]["per_session_create_ms"] < 50

    # session_pool lifecycle managed by minimal_pool fixture


@pytest.mark.benchmark
@pytest.mark.flaky(reruns=2, reruns_delay=1)
async def test_benchmark_session_lifecycle_memory(minimal_pool: AgentPool) -> None:
    """Verify session creation/close does not leak memory under sustained load."""
    session_pool = minimal_pool.session_pool
    assert session_pool is not None

    iterations = 10
    batch_size = 100

    times: list[float] = []
    for _ in range(iterations):
        sids = [f"mem-{i}" for i in range(batch_size)]
        start = time.perf_counter()
        await asyncio.gather(*[session_pool.create_session(sid) for sid in sids])
        await asyncio.gather(*[session_pool.close_session(sid) for sid in sids])
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    print("\n=== Session Lifecycle Memory Benchmark ===")
    print(f"Average batch ({batch_size} sessions): {avg_time * 1000:.2f}ms")

    # Time should be stable (last 3 iterations within 5x of first 3 —
    # generous for CI runner variance)
    first_avg = sum(times[:3]) / 3
    last_avg = sum(times[-3:]) / 3
    assert last_avg < first_avg * 5.0, f"Time grew from {first_avg:.3f}s to {last_avg:.3f}s"

    # session_pool lifecycle managed by minimal_pool fixture


# ============================================================================
# Benchmark: turn latency
# ============================================================================


@pytest.mark.benchmark
async def test_benchmark_turn_latency_under_load(
    minimal_pool: AgentPool,
    mock_agent_with_delay: MagicMock,
) -> None:
    """Measure turn latency with increasing concurrent session load."""
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    collector = MetricsCollector(session_pool)

    results: dict[str, dict[str, float]] = {}

    for session_count in (1, 10, 50, 100):
        # Setup sessions with agents
        sids = [f"latency-{i}" for i in range(session_count)]
        for sid in sids:
            await _attach_agent(session_pool, sid, mock_agent_with_delay, minimal_pool)

        # Subscribe to events
        queues = {sid: await session_pool.event_bus.subscribe(sid) for sid in sids}

        # Run turns concurrently
        start = time.perf_counter()
        await asyncio.gather(*[session_pool.process_prompt(sid, "hello") for sid in sids])
        total_time = time.perf_counter() - start

        # Collect events to ensure completion
        for sid in sids:
            await asyncio.wait_for(queues[sid].get(), timeout=5.0)

        metrics = await collector.get_metrics()
        avg_latency = metrics.turn_latency_ms

        results[f"{session_count}_sessions"] = {
            "total_time_ms": total_time * 1000,
            "avg_turn_latency_ms": avg_latency,
            "throughput_turns_per_sec": session_count / total_time,
        }

        # Cleanup
        await asyncio.gather(*[session_pool.close_session(sid) for sid in sids])

    print("\n=== Turn Latency Benchmark ===")
    for label, metrics in results.items():  # type: ignore[assignment]
        print(
            f"{label}: total={metrics['total_time_ms']:.1f}ms, "  # type: ignore[index]
            f"avg_latency={metrics['avg_turn_latency_ms']:.2f}ms, "  # type: ignore[index]
            f"throughput={metrics['throughput_turns_per_sec']:.1f} turns/s"  # type: ignore[index]
        )

    # Sanity: 100 sessions should complete in under 15 seconds (generous for CI)
    assert results["100_sessions"]["total_time_ms"] < 15000

    # session_pool lifecycle managed by minimal_pool fixture


@pytest.mark.benchmark
async def test_benchmark_turn_latency_serial_vs_concurrent(
    minimal_pool: AgentPool,
    mock_agent_with_delay: MagicMock,
) -> None:
    """Concurrent turns should be faster than serial for multiple sessions."""
    session_pool = minimal_pool.session_pool
    assert session_pool is not None

    session_count = 20
    sids = [f"cmp-{i}" for i in range(session_count)]
    for sid in sids:
        await _attach_agent(session_pool, sid, mock_agent_with_delay, minimal_pool)
        _ = await session_pool.event_bus.subscribe(sid)

    # Serial execution
    serial_start = time.monotonic()
    for sid in sids:
        await session_pool.process_prompt(sid, "hello")
    serial_time = time.monotonic() - serial_start

    # Reset for concurrent test
    await asyncio.gather(*[session_pool.close_session(sid) for sid in sids])
    for sid in sids:
        await session_pool.create_session(sid)
        await _attach_agent(session_pool, sid, mock_agent_with_delay, minimal_pool)

    # Concurrent execution
    concurrent_start = time.monotonic()
    await asyncio.gather(*[session_pool.process_prompt(sid, "hello") for sid in sids])
    concurrent_time = time.monotonic() - concurrent_start

    speedup = serial_time / concurrent_time
    print("\n=== Serial vs Concurrent ===")
    print(f"Serial: {serial_time * 1000:.1f}ms, Concurrent: {concurrent_time * 1000:.1f}ms")
    print(f"Speedup: {speedup:.2f}x")

    # Concurrent should be faster than serial (generous threshold for CI scheduling variance)
    assert speedup > 1.1, f"Concurrent not faster: {speedup:.2f}x"

    await asyncio.gather(*[session_pool.close_session(sid) for sid in sids])
    # session_pool lifecycle managed by minimal_pool fixture


# ============================================================================
# Benchmark: event throughput
# ============================================================================


@pytest.mark.benchmark
async def test_benchmark_event_throughput_single_subscriber() -> None:
    """Measure raw event publish throughput with one subscriber."""
    event_bus = EventBus(max_queue_size=10000)
    session_id = "throughput-1"
    queue = await event_bus.subscribe(session_id)

    event_count = 10000
    start = time.perf_counter()

    for i in range(event_count):
        await event_bus.publish(
            session_id, RunStartedEvent(session_id=session_id, run_id=f"run-{i}")
        )

    publish_time = time.perf_counter() - start
    throughput = event_count / publish_time

    print("\n=== Event Throughput (1 subscriber) ===")
    print(f"Published {event_count} events in {publish_time * 1000:.1f}ms")
    print(f"Throughput: {throughput:.0f} events/second")

    # Generous threshold for CI variance — just verify the bus is functional
    assert throughput > 100, f"Throughput too low: {throughput:.0f} events/s"

    # Verify all events reached subscriber
    received = 0
    while True:
        try:
            queue.get_nowait()
            received += 1
        except asyncio.QueueEmpty:
            break
    assert received == event_count, f"Expected {event_count} events, got {received}"
    await event_bus.close_session(session_id)


@pytest.mark.benchmark
async def test_benchmark_event_throughput_many_subscribers() -> None:
    """Measure event throughput with many subscribers."""
    event_bus = EventBus(max_queue_size=1000)
    session_id = "throughput-n"
    subscriber_count = 100

    queues = [await event_bus.subscribe(session_id) for _ in range(subscriber_count)]

    event_count = 1000
    start = time.perf_counter()

    for i in range(event_count):
        await event_bus.publish(
            session_id, RunStartedEvent(session_id=session_id, run_id=f"run-{i}")
        )

    publish_time = time.perf_counter() - start
    total_events_delivered = event_count * subscriber_count
    throughput = total_events_delivered / publish_time

    print(f"\n=== Event Throughput ({subscriber_count} subscribers) ===")
    print(f"Published {event_count} events to {subscriber_count} subscribers")
    print(f"Total deliveries: {total_events_delivered}")
    print(f"Time: {publish_time * 1000:.1f}ms")
    print(f"Effective throughput: {throughput:.0f} events/second")

    # Verify each subscriber received events
    for queue in queues:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            raise AssertionError("Subscriber did not receive any events") from None

    await event_bus.close_session(session_id)


@pytest.mark.benchmark
@pytest.mark.slow
async def test_benchmark_event_throughput_scaling() -> None:
    """Measure how throughput scales with subscriber count."""
    event_bus = EventBus(max_queue_size=500)
    session_id = "scale"
    event_count = 500

    results: dict[str, dict[str, float]] = {}

    for subscriber_count in (1, 10, 50):
        queues = [await event_bus.subscribe(session_id) for _ in range(subscriber_count)]

        start = time.perf_counter()
        for i in range(event_count):
            await event_bus.publish(
                session_id, RunStartedEvent(session_id=session_id, run_id=f"run-{i}")
            )
        elapsed = time.perf_counter() - start

        total_deliveries = event_count * subscriber_count
        results[f"{subscriber_count}_subscribers"] = {
            "publish_time_ms": elapsed * 1000,
            "events_per_second": event_count / elapsed,
            "total_deliveries_per_second": total_deliveries / elapsed,
        }

        # Drain and unsubscribe for next iteration
        await event_bus.close_session(session_id)
        for q in queues:
            while True:
                try:
                    q.get_nowait()
                except (asyncio.QueueEmpty, asyncio.QueueShutDown):
                    break

    print("\n=== Event Throughput Scaling ===")
    for label, metrics in results.items():
        print(
            f"{label}: {metrics['events_per_second']:.0f} publishes/s, "
            f"{metrics['total_deliveries_per_second']:.0f} total deliveries/s"
        )

    # With many subscribers, total deliveries should remain healthy (generous for CI)
    fifty_total = results["50_subscribers"]["total_deliveries_per_second"]
    assert fifty_total > 5000, f"Total throughput with 50 subscribers too low: {fifty_total:.0f}"


# ============================================================================
# Stress: 1000+ concurrent sessions
# ============================================================================


@pytest.mark.slow
async def test_1000_concurrent_sessions(minimal_pool: AgentPool) -> None:
    """Create 1000 sessions concurrently and verify no resource leaks."""
    session_pool = minimal_pool.session_pool
    assert session_pool is not None

    session_count = 1000

    async def create_session(i: int) -> str:
        sid = f"sess-{i}"
        await session_pool.create_session(sid, agent_name="agent-a")
        return sid

    # Concurrent creation
    created = await asyncio.gather(*[create_session(i) for i in range(session_count)])
    assert len(created) == session_count
    assert len(session_pool.sessions._sessions) == session_count

    # Verify each session is individually accessible
    for sid in created:
        state = session_pool.sessions.get_session(sid)
        assert state is not None
        assert state.session_id == sid

    # Close all concurrently
    await asyncio.gather(*[session_pool.close_session(sid) for sid in created])
    assert len(session_pool.sessions._sessions) == 0

    # Verify event bus has no lingering subscribers
    counts = await session_pool.event_bus.get_subscriber_counts()
    assert counts == {}

    # session_pool lifecycle managed by minimal_pool fixture


@pytest.mark.slow
async def test_1000_concurrent_sessions_with_agents(
    minimal_pool: AgentPool,
    mock_agent: MagicMock,
) -> None:
    """Create 1000 sessions with attached agents and run a turn on each."""
    session_pool = minimal_pool.session_pool
    assert session_pool is not None

    session_count = 1000

    # Create and attach agents
    for i in range(session_count):
        sid = f"sess-{i}"
        await _attach_agent(session_pool, sid, mock_agent, minimal_pool)

    # Subscribe to all sessions
    queues: dict[str, asyncio.Queue[Any]] = {}
    for i in range(session_count):
        sid = f"sess-{i}"
        queues[sid] = await session_pool.event_bus.subscribe(sid)

    # Run a turn on each session concurrently
    async def run_turn(i: int) -> None:
        sid = f"sess-{i}"
        await session_pool.process_prompt(sid, "hello")

    await asyncio.gather(*[run_turn(i) for i in range(session_count)])

    # Verify each session received exactly one event
    for i in range(session_count):
        sid = f"sess-{i}"
        queue = queues[sid]
        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event is not None
        assert isinstance(event, RunStartedEvent)

    # Close all
    await asyncio.gather(*[session_pool.close_session(f"sess-{i}") for i in range(session_count)])
    assert len(session_pool.sessions._sessions) == 0

    # session_pool lifecycle managed by minimal_pool fixture


# ============================================================================
# Stress: rapid create/close cycles
# ============================================================================


@pytest.mark.slow
async def test_rapid_create_close_cycles(minimal_pool: AgentPool) -> None:
    """Repeatedly create and close sessions to verify stability."""
    session_pool = minimal_pool.session_pool
    assert session_pool is not None

    cycles = 200
    for i in range(cycles):
        sid = f"cycle-{i}"
        await session_pool.create_session(sid)
        assert session_pool.sessions.get_session(sid) is not None
        await session_pool.close_session(sid)
        assert session_pool.sessions.get_session(sid) is None

    # After all cycles, state should be clean
    assert len(session_pool.sessions._sessions) == 0
    counts = await session_pool.event_bus.get_subscriber_counts()
    assert counts == {}

    # session_pool lifecycle managed by minimal_pool fixture


@pytest.mark.slow
async def test_rapid_create_close_cycles_with_turns(
    minimal_pool: AgentPool,
    mock_agent: MagicMock,
) -> None:
    """Create, run a turn, and close sessions in rapid succession."""
    session_pool = minimal_pool.session_pool
    assert session_pool is not None

    cycles = 100
    for i in range(cycles):
        sid = f"cycle-{i}"
        await _attach_agent(session_pool, sid, mock_agent, minimal_pool)
        queue = await session_pool.event_bus.subscribe(sid)
        await session_pool.process_prompt(sid, "hello")
        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event is not None
        await session_pool.close_session(sid)
        assert session_pool.sessions.get_session(sid) is None

    assert len(session_pool.sessions._sessions) == 0
    # session_pool lifecycle managed by minimal_pool fixture


@pytest.mark.slow
async def test_rapid_create_close_memory_stable(minimal_pool: AgentPool) -> None:
    """Memory usage should remain stable across many create/close cycles."""
    session_pool = minimal_pool.session_pool
    assert session_pool is not None

    gc.collect()
    initial_objects = len(gc.get_objects())

    cycles = 200
    for i in range(cycles):
        sid = f"cycle-{i}"
        await session_pool.create_session(sid)
        await session_pool.close_session(sid)

    gc.collect()
    final_objects = len(gc.get_objects())

    # Object count should not grow unboundedly (allow some tolerance)
    growth = final_objects - initial_objects
    assert growth <= cycles * 2, f"Object growth ({growth}) suggests leak"

    # session_pool lifecycle managed by minimal_pool fixture


# ============================================================================
# Stress: EventBus at capacity
# ============================================================================


@pytest.mark.slow
async def test_event_bus_capacity_many_subscribers() -> None:
    """EventBus handles many subscribers with bounded queues under load."""
    event_bus = EventBus(max_queue_size=100)
    session_id = "stress-session"
    subscriber_count = 500

    # Create many subscribers
    queues = [await event_bus.subscribe(session_id) for _ in range(subscriber_count)]
    counts = await event_bus.get_subscriber_counts()
    assert counts[session_id] == subscriber_count

    # Publish more events than queue capacity
    events_to_publish = 300
    for i in range(events_to_publish):
        await event_bus.publish(
            session_id, RunStartedEvent(session_id=session_id, run_id=f"run-{i}")
        )

    # Each queue should have at most max_queue_size items
    for queue in queues:
        assert queue.qsize() <= 100

    # Close session and verify cleanup
    await event_bus.close_session(session_id)
    counts = await event_bus.get_subscriber_counts()
    assert session_id not in counts


@pytest.mark.slow
async def test_event_bus_drop_oldest_under_load() -> None:
    """Under heavy load, EventBus drops oldest events correctly."""
    event_bus = EventBus(max_queue_size=10)
    session_id = "drop-session"

    queue = await event_bus.subscribe(session_id)

    # Publish 1000 events to a queue of size 10
    publish_count = 1000
    for i in range(publish_count):
        await event_bus.publish(
            session_id, RunStartedEvent(session_id=session_id, run_id=f"run-{i}")
        )

    # Queue should be at max capacity
    assert queue.qsize() == 10

    # Drain and verify oldest events were dropped
    items: list[Any] = []
    while not _stream_empty(queue):
        items.append(queue.get_nowait())

    run_ids = [e.run_id for e in items if isinstance(e, RunStartedEvent)]
    # Oldest events (run-0 through run-989) should have been dropped
    assert "run-0" not in run_ids
    assert "run-500" not in run_ids
    # Most recent 10 should remain
    assert run_ids == [f"run-{publish_count - 10 + i}" for i in range(10)]

    await event_bus.close_session(session_id)


@pytest.mark.slow
async def test_event_bus_high_throughput_publish() -> None:
    """EventBus can sustain high publish rate without deadlocking."""
    event_bus = EventBus(max_queue_size=1000)
    session_id = "throughput-session"
    subscriber_count = 50

    queues = [await event_bus.subscribe(session_id) for _ in range(subscriber_count)]

    publish_count = 5000
    start = time.monotonic()

    for i in range(publish_count):
        await event_bus.publish(
            session_id, RunStartedEvent(session_id=session_id, run_id=f"run-{i}")
        )

    elapsed = time.monotonic() - start
    # Should complete reasonably fast (< 5 seconds for 5000 publishes)
    assert elapsed < 5.0, f"Publish took too long: {elapsed:.2f}s"

    # All queues should have received events (up to capacity)
    for queue in queues:
        assert queue.qsize() > 0

    await event_bus.close_session(session_id)
    for q in queues:
        assert q.qsize() <= 1000
