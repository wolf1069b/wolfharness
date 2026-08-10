# OpenCode Handler Ops Playbook

Operational runbook for the `OpenCodeProtocolHandler` (`wolfharness_server.opencode_server.handler`).

## SSE Connection Troubleshooting

### Symptom: Client not receiving events

**Diagnosis pipeline:**

1. **Verify the event consumer is running:**
   ```python
   task = handler._consumer_tasks.get(session_id)
   assert task is not None and not task.done(), "Consumer not running"
   ```

2. **Verify the EventBus subscription exists:**
   ```python
   queue = handler._event_bus_subscriptions.get(session_id)
   assert queue is not None, "Not subscribed to EventBus"
   ```

3. **Verify events are being published:**
   ```python
   from wolfharness.orchestrator.metrics import MetricsCollector
   collector = MetricsCollector(session_pool)
   metrics = await collector.get_metrics()
   print(metrics.event_bus_queue_depth.get(session_id, 0))
   ```

4. **Verify SSE broadcast is working:**
   ```python
   from wolfharness_server.opencode_server.models.events import SessionIdleEvent
   await state.broadcast_event(SessionIdleEvent.create(session_id))
   # Client should receive this immediately
   ```

### Common causes

| Cause | Indicator | Fix |
|-------|-----------|-----|
| SessionPool disabled | `RuntimeError: OpenCode use_session_pool is disabled` | Set `opencode.use_session_pool: true` in manifest or `metadata.use_session_pool: true` on agent |
| Consumer task crashed | `task.done()` is `True` | Check logs for "Event consumer loop failed" |
| `state` is `None` | Events converted but not broadcast | Pass `state` to handler constructor |
| SSE subscriber queue full | Log line "SSE subscriber queue full, dropping event" | Increase subscriber queue size or reduce event rate |

### Per-agent canary resolution

The handler checks flags in this order:

1. `agent.metadata.use_session_pool` (bool) — if set, wins.
2. `manifest.opencode.use_session_pool` (bool) — global fallback.

```python
# Check resolution for a specific agent
enabled = handler._agent_uses_session_pool(agent_name="my_agent")
```

## Event Bus Queue Monitoring

### Monitor queue depth per session

```python
collector = MetricsCollector(session_pool)
metrics = await collector.get_metrics()
for sid, count in metrics.event_bus_queue_depth.items():
    print(f"Session {sid}: {count} subscribers")
```

### Alert thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| EventBus subscribers per session | > 10 | > 50 |
| Queue fullness (drops) | > 0 events dropped/min | > 100 events dropped/min |
| Consumer task restarts | > 1/min | > 5/min |

### Inspect a specific queue

```python
queue = handler._event_bus_subscriptions.get(session_id)
if queue is not None:
    print(f"Queue size: {queue.qsize()}")
    print(f"Queue maxsize: {queue.maxsize}")
    print(f"Queue full: {queue.full()}")
```

## Graceful Shutdown

### Normal shutdown sequence

```python
# 1. Cancel and await all consumer tasks
async with handler._lock:
    for sid, task in list(handler._consumer_tasks.items()):
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

# 2. Unsubscribe from EventBus
for sid, queue in list(handler._event_bus_subscriptions.items()):
    await session_pool.event_bus.unsubscribe(sid, queue)

# 3. Close all sessions in SessionPool
for sid in list(session_pool.sessions._sessions.keys()):
    await session_pool.close_session(sid)
```

### Shutdown with timeout

```python
import asyncio

async def shutdown_with_timeout(handler, timeout_seconds: float = 30.0) -> None:
    async with asyncio.timeout(timeout_seconds):
        # Close each session sequentially to avoid thundering herd
        for sid in list(handler._consumer_tasks.keys()):
            await handler.close_session(sid)
```

### Emergency shutdown

If graceful shutdown hangs (e.g., agent turn is stuck):

```python
# Force-cancel all consumer tasks without waiting
for sid, task in list(handler._consumer_tasks.items()):
    task.cancel()
handler._consumer_tasks.clear()
handler._event_bus_subscriptions.clear()

# Force-close SessionPool sessions
for sid in list(session_pool.sessions._sessions.keys()):
    session = session_pool.sessions._sessions.get(sid)
    if session is not None:
        session.is_closing = True
    session_pool.sessions._sessions.pop(sid, None)
```

## Handler State Inspection

```python
def inspect_handler(handler) -> dict:
    return {
        "session_pool_available": handler._session_pool is not None,
        "consumer_tasks": {
            sid: {
                "running": not task.done(),
                "name": task.get_name(),
            }
            for sid, task in handler._consumer_tasks.items()
        },
        "subscriptions": {
            sid: {
                "queue_size": queue.qsize(),
                "queue_maxsize": queue.maxsize,
            }
            for sid, queue in handler._event_bus_subscriptions.items()
        },
    }
```

## Recovery Procedures

### Restart event consumer for a session

```python
# Close first (cleans up old task and subscription)
await handler.close_session(session_id)

# Re-create session and consumer
await handler._ensure_event_consumer(session_id, agent_name="my_agent")
await session_pool.create_session(session_id)
```

### Switch agent for an existing session

1. Close the session.
2. Re-create with the new agent name.
3. The new agent's canary flag will be evaluated on re-creation.

```python
await handler.close_session(session_id)
await handler.handle_message(session_id, message, agent_name="new_agent")
```

## Debugging Checklist

1. **Is SessionPool enabled for this agent?**
   ```python
   assert handler._agent_uses_session_pool(agent_name) is True
   ```

2. **Is the SessionPool initialized?**
   ```python
   assert handler._session_pool is not None
   ```

3. **Is the event consumer active?**
   ```python
   task = handler._consumer_tasks.get(session_id)
   assert task is not None and not task.done()
   ```

4. **Are events flowing through the EventBus?**
   ```python
   queue = handler._event_bus_subscriptions[session_id]
   assert queue.qsize() > 0 or not queue.empty()
   ```

5. **Is the OpenCode state broadcasting?**
   ```python
   assert handler._state is not None
   assert len(handler._state.event_subscribers) > 0
   ```
