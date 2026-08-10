# EventBus Replay Buffer Design

> **Status**: Accepted — this design has been implemented in the AgentPool codebase.
> See `docs/explanation/` for the current architecture documentation.

## Overview

This document specifies the replay buffer for the EventBus, enabling new SSE subscribers to receive historical events before receiving live events. This is required for Migration B when SSE endpoints migrate from `state.event_subscribers` to EventBus-only subscription.

## Problem Statement

Currently:
- Events are published to `asyncio.Queue` subscribers
- No historical events are retained after delivery
- New subscribers only receive events published after their subscription

For SSE migration (B3):
- Client reconnects → needs last N events before live stream
- Without replay, client sees a gap in event history

## Requirements

1. **Bounded memory**: Buffer size is capped to prevent unbounded growth
2. **Per-session isolation**: Each session has its own replay buffer
3. **Subscriber replay**: New subscribers receive historical events before live events
4. **Non-blocking**: Replay must not block live event publishing
5. **Configurable**: Buffer size configurable via `OpenCodeConfig`

## Design Decisions

### Buffer Data Structure: Ring Buffer (Circular Array)

**Chosen over linked list** because:
- O(1) append (overwrite oldest when full)
- O(k) replay where k = number of events to replay
- Cache-friendly contiguous memory
- No allocation during steady state

```python
from collections import deque

class ReplayBuffer:
    """Fixed-size ring buffer for event replay."""
    
    def __init__(self, max_size: int = 100):
        self._buffer: deque[Any] = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
    
    async def append(self, event: Any) -> None:
        """Append an event. Oldest event is dropped when full."""
        async with self._lock:
            self._buffer.append(event)
    
    async def replay(self, limit: int | None = None) -> list[Any]:
        """Get events for replay.
        
        Args:
            limit: Maximum events to return. If None, returns all.
        
        Returns:
            Events ordered from oldest to newest.
        """
        async with self._lock:
            events = list(self._buffer)
            if limit is not None:
                events = events[-limit:]
            return events
```

### Event Retention Policy: Count-Based

**Chosen over time-based** because:
- Simpler to reason about ("last 100 events" vs "events from last 5 minutes")
- Deterministic memory usage (max_size × avg_event_size)
- No background cleanup task needed

Trade-off: Bursty traffic might lose events faster than steady traffic. Mitigation: set `max_size` generously (default 100).

### Subscriber Replay Protocol

When a new subscriber joins:

```python
async def subscribe(self, session_id: str, scope: str = "session") -> asyncio.Queue[Any]:
    queue = asyncio.Queue(maxsize=self._max_queue_size)
    
    # 1. Register subscriber FIRST (before replay to avoid missing live events)
    async with self._lock:
        self._subscribers.setdefault(session_id, []).append((queue, scope))
    
    # 2. Get replay buffer snapshot
    buffer = self._get_replay_buffer(session_id)
    historical_events = await buffer.replay()
    
    # 3. Drain any live events that arrived during replay
    # (these are already in the queue from publish())
    live_events_during_replay: list[Any] = []
    while not queue.empty():
        try:
            live_events_during_replay.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    
    # 4. Replay historical events first (with is_replay flag)
    # This ensures ordering: historical → live
    import copy
    for event in historical_events:
        try:
            # Copy before modifying to avoid affecting other subscribers
            event_copy = copy.copy(event)
            event_copy.is_replay = True  # type: ignore
            queue.put_nowait(event_copy)
        except asyncio.QueueFull:
            break  # Skip remaining if queue is full
    
    # 5. Re-insert live events that arrived during replay
    for event in live_events_during_replay:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            break
    
    return queue
```

**Key properties**:
- Subscriber is registered **before** replay starts, so no live events are lost
- Live events that arrive during replay are temporarily drained
- Historical events are replayed first, then live events re-inserted
- Ordering guarantee: historical events always precede live events in the queue
- If subscriber queue is full during replay, remaining historical events are skipped

**Race condition handled**: If `publish()` is called between step 1 (register) and step 2 (replay), the event goes into the subscriber's queue. Step 3 drains it, and step 5 re-inserts it after historical events.

## Integration with EventBus

### Current EventBus

```python
class EventBus:
    def __init__(self, max_queue_size: int = 1000, session_controller=None):
        self._subscribers: dict[str, list[tuple[Queue, str]]] = {}
        self._session_tree: dict[str, list[str]] = {}
        self._lock = asyncio.Lock()
        self._max_queue_size = max_queue_size
        self._session_controller = session_controller
```

### Enhanced EventBus

```python
class EventBus:
    def __init__(
        self,
        max_queue_size: int = 1000,
        replay_buffer_size: int = 100,
        session_controller=None,
    ):
        self._subscribers: dict[str, list[tuple[Queue, str]]] = {}
        self._replay_buffers: dict[str, ReplayBuffer] = {}
        self._max_queue_size = max_queue_size
        self._replay_buffer_size = replay_buffer_size
        self._session_controller = session_controller
    
    def _get_replay_buffer(self, session_id: str) -> ReplayBuffer:
        """Get or create replay buffer for a session."""
        if session_id not in self._replay_buffers:
            self._replay_buffers[session_id] = ReplayBuffer(max_size=self._replay_buffer_size)
        return self._replay_buffers[session_id]
    
    async def publish(self, session_id: str, event: Any) -> None:
        # 1. Store in replay buffer
        buffer = self._get_replay_buffer(session_id)
        await buffer.append(copy.copy(event))
        
        # 2. Publish to live subscribers (existing logic)
        # ... (existing publish logic)
```

### Memory Bounds

Per-session memory usage:
```
max_memory_per_session = replay_buffer_size × avg_event_size
```

With default values:
- `replay_buffer_size = 100`
- `avg_event_size ≈ 2 KB` (typical ChatMessage with text content)
- `max_memory_per_session ≈ 200 KB`

For 1000 active sessions: ~200 MB total (acceptable for most deployments).

## Configuration

```python
# wolfharness_config/session_pool.py
@dataclass
class OpenCodeConfig:
    use_session_pool: bool = True
    # ... other fields ...
    eventbus_replay_buffer_size: int = 100
    """Maximum number of events retained per session for replay."""
```

Environment variable override:
```bash
AGENTPOOL_EVENTBUS_REPLAY_BUFFER_SIZE=200
```

## Migration Path

### Phase 1: Add Replay Buffer to EventBus
- Add `ReplayBuffer` class
- Enhance `EventBus` with `_replay_buffers`
- Update `subscribe()` to replay historical events
- Add configuration field to `OpenCodeConfig`
- Unit tests for buffer behavior

### Phase 2: SSE Endpoint Migration (B3)
- SSE endpoint creates EventBus subscriber (instead of `state.event_subscribers`)
- New subscribers automatically receive replay + live events
- Verify `OpenCodeEventAdapter` converts `RichAgentStreamEvent` correctly

### Phase 3: Cleanup
- Remove `state.event_subscribers` after SSE migration is complete
- Remove manual `broadcast_event()` path where redundant

## Edge Cases

### Subscriber Queue Full During Replay

If a subscriber's queue is full while replaying historical events:
- **Behavior**: Skip remaining historical events, continue with live events
- **Rationale**: Live events are more important than historical ones for UX
- **Mitigation**: Increase queue size or buffer size

### Session Closed

When a session is closed:
- **Behavior**: Replay buffer is cleared to free memory
- **Implementation**: `EventBus.close_session()` removes the buffer

```python
async def close_session(self, session_id: str) -> None:
    """Close a session and clean up its replay buffer."""
    self._replay_buffers.pop(session_id, None)
    # ... existing close logic ...
```

### Child Session Events

With `scope="descendants"`, parent subscribers should receive child session events:
- **Behavior**: Child events are stored in child's replay buffer, not parent's
- **Rationale**: Each session's buffer is independent
- **For replay**: Parent subscriber replaying gets parent's history; child events arrive live via scope matching

## Testing Strategy

1. **Buffer bounds**: Append 150 events to buffer of size 100, verify only last 100 are replayed
2. **Replay ordering**: Verify events are replayed oldest-to-newest
3. **Concurrent access**: Multiple subscribers join while events are being published
4. **Memory cleanup**: Verify buffer is removed when session is closed
5. **Scope behavior**: Parent subscriber with `scope="descendants"` receives child events

## Open Questions

1. **Should we support event filtering in replay?**
   - E.g., only replay `PartDeltaEvent`, skip `ToolCallStartEvent`
   - *Decision*: Defer to post-Migration B. Current design replays all events.

2. **Persistent replay buffer?**
   - Should replay buffer survive server restart?
   - *Decision*: No, in-memory only. Persistence is StorageProvider's responsibility.
