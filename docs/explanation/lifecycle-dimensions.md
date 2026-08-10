# Lifecycle Dimensions (M2)

The M2 lifecycle subsystem introduces six pluggable dimensions that decouple the RunLoop from its infrastructure. Each dimension has a `@runtime_checkable` Protocol and default in-memory implementations.

## Dimension Reference

| Dimension | Protocol | Default | Durable Alternative | Purpose |
|---|---|---|---|---|
| `TriggerSource` | `TriggerSource` | `ImmediateTrigger` | `ProtocolTrigger` | How prompts arrive at the RunLoop |
| `Journal` | `Journal` | `MemoryJournal` | `DurableJournal` (SQLite WAL) | Event-layer persistence (append + upsert) |
| `SnapshotStore` | `SnapshotStore` | `MemorySnapshotStore` | `DurableSnapshotStore` (SQLite) | Loop-layer state persistence at Turn boundaries |
| `CommChannel` | `CommChannel` | `DirectChannel` | `ProtocolChannel` | Event delivery + feedback reception (owns Journal) |
| `EventTransport` | `EventTransport` | `InProcessTransport` | MQ/gRPC (future) | Wire protocol abstraction for external consumers |
| `session_id` | n/a | `"default"` | n/a | Logical session identifier |

## Default Implementations

- **`ImmediateTrigger`** — Delivers a single prompt on the first `poll()` call, then returns `None`. Used for standalone `agent.run()` execution.
- **`ProtocolTrigger`** — Bridges protocol handlers to the RunLoop via an `asyncio.Queue`. Callers use `trigger.deliver(content)` to enqueue prompts.
- **`MemoryJournal`** — In-process journal using Python lists/dicts. Data is lost on process exit.
- **`DurableJournal`** — SQLite-backed journal with WAL mode and `synschronous=NORMAL` for crash-safe writes. Schema: `lifecycle_journal` (seq, entry_type, upsert_key, event_json) and `lifecycle_tool_log` (turn_id, tool_name, args, result, status).
- **`MemorySnapshotStore`** — In-memory snapshot store using plain dicts.
- **`DurableSnapshotStore`** — SQLite-backed snapshot store with WAL mode and `synschronous=FULL`. Schema: `snapshots` (seq, state_blob) and `turn_results` (turn_id, result_blob).
- **`DirectChannel`** — Unidirectional; publishes events to an internal `asyncio.Queue` that `start()` drains via `get_nowait()`. `recv()` always returns `None`.
- **`ProtocolChannel`** — Bidirectional; publishes events to the `EventBus` and maintains a feedback queue for steer/followup. `StateUpdate` events are journaled but not published to the EventBus. `recv()` dequeues from the feedback queue.
- **`InProcessTransport`** — In-process `EventTransport` using per-topic `asyncio.Queue` with optional replay buffer (disabled by default).

## Ownership Topology

```
RunLoop (RunHandle)
  +-- _trigger_source: TriggerSource   (owned by RunLoop)
  +-- _snapshot_store: SnapshotStore   (owned by RunLoop)
  +-- _event_transport: EventTransport (owned by RunLoop)
  +-- _comm_channel: CommChannel       (owned by RunLoop, but OWNS Journal)
  |     +-- _journal: Journal          (owned by CommChannel)
  |           +-- append/upsert        (delta / entity-state write semantics)
  |           +-- log_tool_execution   (tool execution log for idempotency)
```

The Journal is owned by the CommChannel so that every event is persisted before delivery. The SnapshotStore sits beside the CommChannel (not behind it) because snapshot writes are batch operations at Turn boundaries, not event-by-event.

## Crash Recovery

Recovery is triggered in `start()` when a durable journal and snapshot store are configured:

1. `journal.resume(snapshot_store)` loads the latest snapshot, replays journal entries since the snapshot, and detects in-flight Turns via `_detect_inflight_turn()`.
2. If a Turn was in-flight (turn appeared in journal but has no completed result in the snapshot store), the result determines behavior:
   - `recover_strategy: "mark_interrupted"` — preserves partial output in the journal but continues from idle.
   - `recover_strategy: "retry"` — checks `journal.get_tool_executions(turn_id)` to skip already-completed tools during re-execution.
3. Events since the last snapshot are replayed through the CommChannel with `_replaying = True` (journaling skipped).

## Tool Execution Log

The Journal maintains a tool execution log for idempotent crash recovery. Each `ToolExecutionRecord` stores `(turn_id, tool_name, args, result, status)`. The log is populated by `HookAwareTurn._fire_post_tool_hooks()`, which calls `_log_tool_execution()` after every tool completes. This is independent of the hooks system and always fires (even when `hooks:` is not configured).

## `lifecycle:` YAML Config Section

```yaml
agents:
  my_agent:
    type: native
    model: openai:gpt-4o
    lifecycle:
      journal: durable           # "memory" (default) or "durable"
      snapshot: durable          # "memory" (default) or "durable"
      recover_strategy: retry    # "mark_interrupted" (default) or "retry"
```

When the `lifecycle:` section is omitted or all fields are at default, `create_dimensions()` returns `None` for all dimensions, and `RunHandle.__post_init__()` creates in-memory defaults.

## Factory Function

```python
from wolfharness.lifecycle.factory import create_dimensions

trigger, journal, snapshot, comm, transport = create_dimensions(
    lifecycle_config, session_id="my_session",
)
run_handle = RunHandle(
    run_id="run1",
    session_id="my_session",
    agent_type="native",
    _trigger_source=trigger,
    _journal=journal,
    _snapshot_store=snapshot,
    _comm_channel=comm,
    _event_transport=transport,
)
```

## Lifecycle Package Structure

```
src/wolfharness/lifecycle/
  __init__.py         — Public exports for all types and implementations
  types.py            — RunState, Prompt, Feedback, ResumeResult,
                        ToolExecutionRecord, EventEnvelope (plain dataclasses;
                        M6 upgrades to Pydantic)
  protocols.py        — TriggerSource, Journal, SnapshotStore, CommChannel,
                        EventTransport (@runtime_checkable Protocols)
  triggers.py         — ImmediateTrigger, ProtocolTrigger, ScheduledTrigger
                        (stub), ChannelTrigger (stub)
  journal.py          — MemoryJournal, DurableJournal (SQLite WAL)
  snapshot_store.py   — MemorySnapshotStore, DurableSnapshotStore (SQLite)
  comm_channel.py     — DirectChannel, ProtocolChannel
  event_transport.py  — InProcessTransport
  factory.py          — create_dimensions() from LifecycleConfig
```

## Conventions

- **M2 uses dataclasses; M6 upgrades to Pydantic**: All types in `types.py` are plain dataclasses with the same field names and types that the M6 Pydantic models will use.
- **`lifecycle.EventEnvelope` is separate** from `orchestrator.event_bus.EventEnvelope`. The lifecycle envelope (`EventEnvelope`) is the language-agnostic serialization format for event transport. The orchestrator envelope (`orchestrator.event_bus.EventEnvelope`) is the internal EventBus delivery envelope. These are distinct types with different responsibilities.
- **CommChannel owns the Journal**: The Journal is injected into the CommChannel constructor. `CommChannel.publish()` journals (append or upsert) before delivery.
- **`_replaying` flag**: Set to `True` during crash recovery replay to skip journaling and prevent duplicate entries.
- **`ProtocolChannel` filters `StateUpdate`**: `StateUpdate` events are journaled but NOT published to the EventBus. They are internal lifecycle signals that protocol servers do not need to receive.
- **`turn_id` on `AgentRunContext`**: Generated as `str(uuid.uuid4())` and stored on `AgentRunContext.turn_id` for tool execution log correlation.
- **Recovery metadata preserved on `RunHandle`**: `_recovered_inflight_turn_id` and `recovered_tool_executions` property give downstream code (re-engagement flows) access to the interrupted state.
