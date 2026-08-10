# ACP Handler Ops Playbook

Operational runbook for the `ACPProtocolHandler` (`wolfharness_server.acp_server.handler`).

## Feature Flag Toggle

SessionPool integration is controlled by a **per-agent canary flag** in the agent metadata.

### Enable for a single agent

```yaml
agents:
  my_agent:
    type: native
    model: openai:gpt-4o
    metadata:
      use_session_pool: true
```

### Disable for a single agent

```yaml
    metadata:
      use_session_pool: false
```

### Check at runtime

```python
from wolfharness_server.acp_server.handler import ACPProtocolHandler

handler = ACPProtocolHandler(agent_pool, event_converter, client)
enabled = handler._should_use_session_pool()
```

Resolution order:
1. If `agent_pool.main_agent.metadata["use_session_pool"]` is set, that value wins.
2. Otherwise falls back to `False` (legacy path).

## Fallback to Legacy Mode

When the canary flag is disabled (or `SessionPool` is not initialized), the handler returns `None` from `handle_prompt()` and returns early from `close_session()`. The caller (legacy `ACPSessionManager`) must handle the fallback.

### Force fallback for debugging

1. Set `use_session_pool: false` in the agent metadata.
2. Restart the ACP server (metadata is read at handler initialization).

### Verify which path is active

Watch for these log lines:

| Path | Log message |
|------|-------------|
| SessionPool | `"Started event consumer"` |
| Legacy | `"Per-agent canary flag off, skipping SessionPool"` |

## Session Drain

### Graceful session close

```python
await handler.close_session(session_id)
```

Steps performed:
1. Sends `None` sentinel via `EventBus.close_session()` to stop the consumer loop.
2. Waits up to **5 seconds** for the consumer task to finish.
3. Cancels the consumer task if it does not finish in time.
4. Delegates to `SessionPool.close_session()` for final cleanup.

### Bulk drain (deploy / maintenance)

```python
# Close all sessions known to the handler
for session_id in list(handler._consumer_tasks.keys()):
    await handler.close_session(session_id)
```

### Emergency session termination

If `close_session()` hangs (e.g., agent `__aexit__` is deadlocked):

```python
import asyncio

task = handler._consumer_tasks.pop(session_id, None)
if task is not None and not task.done():
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=2.0)
    except (asyncio.CancelledError, TimeoutError):
        pass

# Force SessionPool cleanup
session_pool = agent_pool.session_pool
if session_pool is not None:
    await session_pool.event_bus.close_session(session_id)
    await session_pool.close_session(session_id)
```

## Event Consumer Monitoring

### Check consumer health

```python
task = handler._consumer_tasks.get(session_id)
if task is None:
    status = "not_started"
elif task.done():
    status = f"exited: {task.exception()!r}"
else:
    status = "running"
```

### Common consumer failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| Consumer task done with exception | Event conversion failed | Check `ACPEventConverter` logs |
| Consumer task cancelled | `close_session()` was called | Expected during normal cleanup |
| No events reaching client | EventBus not subscribed | Verify `_ensure_event_consumer()` was called |

## Debugging Checklist

1. **Is SessionPool initialized?**
   ```python
   assert agent_pool.session_pool is not None
   ```

2. **Is the canary flag set?**
   ```python
   assert agent_pool.main_agent.metadata.get("use_session_pool") is True
   ```

3. **Is the event consumer running?**
   ```python
   task = handler._consumer_tasks.get(session_id)
   assert task is not None and not task.done()
   ```

4. **Is the EventBus queue receiving events?**
   ```python
   queue = handler._consumer_queues.get(session_id)
   assert queue is not None and not queue.empty()
   ```

5. **Are ACP session updates being sent?**
   - Check ACP client logs for `session_update` calls.
   - Verify the client connection is still open.
