# lifecycle — RunLoop Dimension Protocols

## Where to Look

| Task | File |
|---|---|
| Five Protocols (TriggerSource, Journal, SnapshotStore, CommChannel, EventTransport) | `protocols.py` |
| RunState, Prompt, Feedback, ResumeResult, ToolExecutionRecord, EventEnvelope | `types.py` |
| ImmediateTrigger, ProtocolTrigger | `triggers.py` |
| MemoryJournal, DurableJournal (SQLite WAL) | `journal.py` |
| MemorySnapshotStore, DurableSnapshotStore (SQLite) | `snapshot_store.py` |
| DirectChannel, ProtocolChannel | `comm_channel.py` |
| InProcessTransport | `event_transport.py` |
| create_dimensions() from LifecycleConfig | `factory.py` |

## Conventions

- **CommChannel owns the Journal**: Every `CommChannel` has a `_journal` reference. `publish()` journals (append/upsert) before delivery. Journal is injected into CommChannel constructor.
- **`lifecycle.EventEnvelope` != `orchestrator.event_bus.EventEnvelope`**: Different types. Lifecycle envelope is for language-agnostic transport serialization. Orchestrator envelope is for internal EventBus delivery.
- **`_replaying` flag**: Set to `True` during crash recovery replay to skip journaling and prevent duplicate entries.
- **`ProtocolChannel` filters `StateUpdate`**: `StateUpdate` events are journaled but NOT published to EventBus — they are internal lifecycle signals.
- **`turn_id` on `AgentRunContext`**: Generated as `str(uuid.uuid4())` for tool execution log correlation.
- **M2 uses dataclasses; M6 upgrades to Pydantic**: All types in `types.py` are plain dataclasses matching future Pydantic model fields.
- **SnapshotStore sits beside CommChannel**: Not behind it. Snapshot writes are batch operations at Turn boundaries, not event-by-event.

## Anti-Patterns

- **Mixing `lifecycle.EventEnvelope` with `orchestrator.event_bus.EventEnvelope`**: Import from `wolfharness.lifecycle.types` for lifecycle transport; from `wolfharness.orchestrator.event_bus` for internal EventBus.
- **Do NOT instrument MemoryJournal / MemorySnapshotStore**: Pure in-memory implementations, no telemetry needed.
