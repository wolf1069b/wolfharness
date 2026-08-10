# SessionPool Ops Playbook

Operational runbook for the `SessionPool` orchestration layer.

## Dashboard Compatibility

All Prometheus metric names use the `wolfharness_` prefix and identical label names to the pre-SessionPool Grafana dashboards. **No dashboard changes are required.**

| Metric | Type | Description |
|--------|------|-------------|
| `wolfharness_sessions_total` | gauge | Active sessions |
| `wolfharness_active_turns_total` | gauge | Turns in progress |
| `wolfharness_auto_resume_total` | counter | Auto-resume iterations |
| `wolfharness_event_bus_subscribers` | gauge | Subscribers per session (`session_id` label) |
| `wolfharness_session_lifetime_seconds` | gauge | Average closed-session lifetime |
| `wolfharness_turn_latency_ms` | summary | Turn latency p99 (`quantile="0.99"`) |

Call `metrics.to_prometheus()` on a `SessionPoolMetrics` snapshot to emit these lines for scraping.

## Startup

```python
from wolfharness.orchestrator import SessionPool

session_pool = SessionPool(agent_pool)
await session_pool.start()
```

- `start()` launches the background cleanup task (`SessionController._cleanup_loop`).
- The cleanup task scans for expired sessions every `session_ttl_seconds / 2` (default 30 min).
- No events are processed until `process_prompt()` is called.

## Shutdown

```python
await session_pool.shutdown()
```

- Cancels the background cleanup task.
- Iterates over all active sessions and calls `close_session()` for each.
- Per-session agents receive `__aexit__()` if the turn completed within the 30-second timeout.
- EventBus queues receive the `None` sentinel, unblocking any waiting consumers.

## Health Checks

### Minimal Liveness

```python
# Liveness: SessionPool object exists and event_bus is reachable
assert session_pool.event_bus is not None
```

### Readiness (Recommended)

```python
from wolfharness.orchestrator.metrics import MetricsCollector

collector = MetricsCollector(session_pool)
metrics = await collector.get_metrics()

# Flag if queue depth is backing up
max_subscribers = max(metrics.event_bus_queue_depth.values(), default=0)
assert max_subscribers < 100, "EventBus subscriber backlog detected"

# Flag if turns are stalling
assert metrics.turn_latency_p99 < 30_000, "Turn latency p99 > 30s"
```

## Memory Troubleshooting

### Symptom: Memory grows linearly with session count

**Likely cause:** Sessions are not being closed or the cleanup loop is not running.

**Diagnosis:**

```python
metrics = await collector.get_metrics()
print(f"Active sessions: {metrics.active_sessions}")
print(f"Auto-resume count: {metrics.auto_resume_count}")
print(f"Event bus subscribers: {metrics.event_bus_queue_depth}")
```

**Remediation:**

1. Verify `session_pool.start()` was called (cleanup task must be running).
2. Check logs for "Closing expired session" — if absent, TTL may be too long.
3. If sessions are intentionally long-lived, ensure `close_session()` is called by protocol handlers on client disconnect.
4. Inspect `gc.get_objects()` for unbounded `SessionState` or `asyncio.Lock` growth.

### Symptom: EventBus queues growing unbounded

**Likely cause:** Dead subscribers are not being cleaned up, or consumers are slower than producers.

**Diagnosis:**

- `wolfharness_event_bus_subscribers` gauge will show high counts for specific sessions.
- Check consumer task logs — look for "Event consumer cancelled" or uncaught exceptions.

**Remediation:**

1. EventBus already drops oldest events when a queue is full (bounded queue with dropping strategy).
2. Ensure protocol handler consumer loops call `unsubscribe()` in their `finally` block.
3. If a single session has abnormally high subscriber count, restart the protocol handler for that session.

## TTL Tuning

The default session TTL is **3600 seconds** (1 hour). Tune via `SessionController._session_ttl_seconds`:

```python
# Short TTL for high-churn workloads (e.g., webhooks)
session_pool.sessions._session_ttl_seconds = 300.0  # 5 minutes

# Long TTL for persistent IDE sessions
session_pool.sessions._session_ttl_seconds = 7200.0  # 2 hours
```

**Guidelines:**

| Workload | Recommended TTL | Rationale |
|----------|----------------|-----------|
| ACP IDE (Zed) | 3600s | Users keep sessions open for hours |
| OpenCode TUI | 1800s | TUI reconnects frequently |
| Web/API | 300s | Stateless, high churn |
| Batch jobs | 60s | Short-lived, deterministic |

After changing TTL, the cleanup loop automatically picks up the new value on its next iteration (no restart required).

## Metrics Scraping

```python
collector = MetricsCollector(session_pool)
metrics = await collector.get_metrics()
print(metrics.to_prometheus())
```

Example output:

```text
# TYPE wolfharness_sessions_total gauge
wolfharness_sessions_total 42
# TYPE wolfharness_active_turns_total gauge
wolfharness_active_turns_total 3
# TYPE wolfharness_auto_resume_total counter
wolfharness_auto_resume_total 17
# TYPE wolfharness_event_bus_subscribers gauge
wolfharness_event_bus_subscribers{session_id="sess_abc"} 2
# TYPE wolfharness_session_lifetime_seconds gauge
wolfharness_session_lifetime_seconds 1245.300
# TYPE wolfharness_turn_latency_ms summary
wolfharness_turn_latency_ms{quantile="0.99"} 150.000
```

## Emergency Procedures

### Force-close all sessions

```python
for sid in list(session_pool.sessions._sessions.keys()):
    await session_pool.close_session(sid)
```

### Disable auto-resume globally

```python
session_pool.turns._enable_auto_resume = False
```

This stops the automatic processing of queued injections/prompts without affecting in-flight turns.

### Drain turns before deploy

```python
# Wait up to 60s for all active turns to complete
import asyncio
for sid, state in list(session_pool.sessions._sessions.items()):
    if state.turn_lock.locked():
        try:
            async with asyncio.timeout(60.0):
                async with state.turn_lock:
                    pass
        except TimeoutError:
            logger.warning("Turn did not drain in time", session_id=sid)
```
