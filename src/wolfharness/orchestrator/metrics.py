"""Metrics and observability for the SessionPool orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING

from wolfharness.log import get_logger


if TYPE_CHECKING:
    from wolfharness.orchestrator.core import SessionPool


logger = get_logger(__name__)


@dataclass
class SessionPoolMetrics:
    """Metrics snapshot for SessionPool observability.

    Attributes:
        active_sessions: Number of currently active sessions.
        active_turns: Number of sessions with a turn currently in progress.
        auto_resume_count: Total number of auto-resume occurrences recorded.
        event_bus_queue_depth: Mapping of session IDs to subscriber counts.
        session_lifetime_seconds: Average lifetime of closed sessions.
        turn_latency_ms: Average turn latency in milliseconds.
    """

    active_sessions: int
    active_turns: int
    auto_resume_count: int
    event_bus_queue_depth: dict[str, int]
    session_lifetime_seconds: float
    turn_latency_ms: float
    turn_latency_p99: float = 0.0
    active_runs_by_agent_type: dict[str, int] = field(default_factory=dict)

    def to_prometheus(self) -> str:
        """Return metrics in Prometheus exposition format.

        All metric names use the ``wolfharness_`` prefix and the same
        label names as the existing Grafana dashboards so no dashboard
        changes are required.

        Returns:
            Multi-line string in Prometheus text format.
        """
        lines: list[str] = []

        lines.append("# TYPE wolfharness_sessions_total gauge")
        lines.append(f"wolfharness_sessions_total {self.active_sessions}")

        lines.append("# TYPE wolfharness_active_turns_total gauge")
        lines.append(f"wolfharness_active_turns_total {self.active_turns}")

        lines.append("# TYPE wolfharness_auto_resume_total counter")
        lines.append(f"wolfharness_auto_resume_total {self.auto_resume_count}")

        lines.append("# TYPE wolfharness_event_bus_subscribers gauge")
        for session_id, count in self.event_bus_queue_depth.items():
            sid = session_id.replace('"', '\\"')
            lines.append(f'wolfharness_event_bus_subscribers{{session_id="{sid}"}} {count}')

        lines.append("# TYPE wolfharness_session_lifetime_seconds gauge")
        lines.append(f"wolfharness_session_lifetime_seconds {self.session_lifetime_seconds:.3f}")

        lines.append("# TYPE wolfharness_turn_latency_ms summary")
        lines.append(f'wolfharness_turn_latency_ms{{quantile="0.99"}} {self.turn_latency_p99:.3f}')

        lines.append("# TYPE wolfharness_active_runs_by_agent_type gauge")
        for agent_type, count in self.active_runs_by_agent_type.items():
            at = agent_type.replace('"', '\\"')
            lines.append(f'wolfharness_active_runs_by_agent_type{{agent_type="{at}"}} {count}')

        return "\n".join(lines)


class MetricsCollector:
    """Collects metrics from a SessionPool instance.

    Provides lock-protected access to session state and turn timings
    to produce a SessionPoolMetrics snapshot.
    """

    def __init__(self, session_pool: SessionPool) -> None:
        """Initialize the metrics collector.

        Args:
            session_pool: The session pool to collect metrics from.
        """
        self.session_pool = session_pool
        self._auto_resume_counter: int = 0

    def record_auto_resume(self) -> None:
        """Record an auto-resume occurrence.

        Called when an auto-resume iteration is triggered.
        """
        self._auto_resume_counter += 1

    async def get_metrics(self) -> SessionPoolMetrics:
        """Collect a metrics snapshot from the session pool.

        Returns:
            A SessionPoolMetrics instance with current values.
        """
        async with self.session_pool.sessions._lock:
            sessions = dict(self.session_pool.sessions._sessions)

        closed_sessions = [s for s in sessions.values() if s.closed_at is not None]
        if closed_sessions:
            total_lifetime = 0.0
            for s in closed_sessions:
                assert s.closed_at is not None
                total_lifetime += s.closed_at - s.created_at
            avg_session_lifetime = total_lifetime / len(closed_sessions)
        else:
            avg_session_lifetime = 0.0

        # Turn timing data is no longer available (old turn lifecycle removed).
        turn_timings: list[tuple[float, float]] = []
        if turn_timings:
            latencies_ms = [(end - start) * 1000 for start, end in turn_timings]
            avg_turn_latency_ms = sum(latencies_ms) / len(latencies_ms)
            sorted_latencies = sorted(latencies_ms)
            p99_index = math.ceil(0.99 * len(sorted_latencies)) - 1
            p99_turn_latency_ms = sorted_latencies[max(0, p99_index)]
        else:
            avg_turn_latency_ms = 0.0
            p99_turn_latency_ms = 0.0

        subscriber_counts = await self.session_pool.event_bus.get_subscriber_counts()

        active_runs = self.session_pool.active_runs
        active_turns = len(active_runs)
        active_runs_by_agent_type: dict[str, int] = {}
        for run in active_runs:
            active_runs_by_agent_type[run.agent_type] = (
                active_runs_by_agent_type.get(run.agent_type, 0) + 1
            )

        return SessionPoolMetrics(
            active_sessions=len(sessions),
            active_turns=active_turns,
            auto_resume_count=self._auto_resume_counter,
            event_bus_queue_depth=subscriber_counts,
            session_lifetime_seconds=avg_session_lifetime,
            turn_latency_ms=avg_turn_latency_ms,
            turn_latency_p99=p99_turn_latency_ms,
            active_runs_by_agent_type=active_runs_by_agent_type,
        )
