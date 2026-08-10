---
rfc_id: RFC-0042
title: "Unified Lifecycle Architecture: RunLoop, TriggerSource, Journal, SnapshotStore, CommChannel, EventTransport"
status: DRAFT
author: yuchen.liu
reviewers: []
created: 2026-07-08
last_updated: 2026-07-08 (revision 4.4: Oracle VERIFIED — glossary dedup, RunState enum, ProtocolBridge None guard)
decision_date:
related_rfcs:
  - RFC-0041 (Run vs Turn Separation — prerequisite Phase 1)
  - RFC-0037 (Unify Steer and Followup — subsumed by TriggerSource)
  - RFC-0029 (Agent Reactivation via Pending Prompt Queue — legacy mechanism)
  - RFC-0024 (Agent Stateless Refactor — related architectural direction)
  - RFC-0035 (MCP over ACP Complete Connection Chain — protocol bridging precedent)
related_specs:
  - openspec/changes/introduce-anyio-structured-concurrency/ (CancelScope hierarchy)
  - docs/design/lifecycle-analysis.md (cross-framework research basis)
---

# RFC-0042: Unified Lifecycle Architecture — RunLoop, TriggerSource, Journal, SnapshotStore, CommChannel, EventTransport

## Overview

AgentPool currently supports multiple execution modes — standalone single-turn, protocol-backed session (ACP/OpenCode/AG-UI/OpenAI API), and background workers — but each mode is implemented as a special case with its own entry point, lifecycle management, and communication pattern. Cross-framework research (see `docs/design/lifecycle-analysis.md`) reveals that other agent frameworks face the same fragmentation: hermes-agent has a 7000-line god-object loop to handle gateway adapters, opencode built a separate durable event store, and pi implemented a custom steering/followup queue.

This RFC proposes a **unified lifecycle architecture** built on six orthogonal, pluggable dimensions (Revision 4: StateStore split into Journal + SnapshotStore):

| Dimension | Layer | Responsibility | Interface |
|-----------|-------|---------------|-----------|
| **RunLoop** | Loop | Drives the idle → running → idle \| done cycle. Executes Turns. Owns SnapshotStore. | `RunLoop.start()`, `RunLoop.steer()`, `RunLoop.followup()`, `RunLoop.close()` |
| **TriggerSource** | Input | Abstracts how prompts arrive at the RunLoop. Bridges external stimulus to internal message queue. | `TriggerSource.subscribe(run_loop)`, `TriggerSource.poll() → Prompt` |
| **Journal** | Event | Event persistence (append + upsert), replay, crash recovery via `resume()`. Owned by CommChannel. | `Journal.append(event)`, `Journal.upsert(key, event)`, `Journal.resume(snapshot_store) → ResumeResult` |
| **SnapshotStore** | Loop | State snapshots at Turn boundaries, turn results for idempotency. Owned by RunLoop. | `SnapshotStore.save(state)`, `SnapshotStore.load() → (state, seq)`, `SnapshotStore.has_turn_result(turn_id)` |
| **CommChannel** | Output | Abstracts communication back to the caller. Delivers events and responses. Owns Journal. May receive feedback. | `CommChannel.publish(event)`, `CommChannel.recv() → Feedback`, `CommChannel.on_state_change(state)` |
| **EventTransport** | Transport | Abstracts the wire protocol between RunLoop and external consumers. Enables language-agnostic protocol servers and MQ-based decoupling. | `EventTransport.publish(envelope)`, `EventTransport.subscribe() → envelope` |

**Thesis**: By decomposing the lifecycle into these six dimensions, all execution modes become configuration choices rather than architectural special cases. The Turn execution layer (RFC-0041's Run/Turn separation) remains identical across all modes. Durability, replay, and cross-language protocol support are composable concerns — not afterthoughts.

**Revision 4 key change**: The former monolithic `StateStore` is split into `Journal` (event layer, owned by CommChannel) and `SnapshotStore` (loop layer, owned by RunLoop). This separation aligns with Akka/Pekko's design where journal and snapshot-store are independent pluggable components. Journal supports both `append()` (for delta events) and `upsert()` (for entity-state events, mirroring ACP v2's ToolCallUpdate). `resume()` is a first-class operation on Journal that coordinates both layers for crash recovery.

### Execution Modes Covered

| Mode | TriggerSource | Journal | SnapshotStore | CommChannel | EventTransport | RunLoop |
|------|--------------|---------|----------------|-------------|---------------|---------|
| **Standalone single-turn** | `ImmediateTrigger` | `MemoryJournal` | `MemorySnapshotStore` | `DirectChannel` | `InProcessTransport` | 1 Turn then done |
| **Protocol session** (ACP/OpenCode/AG-UI) | `ProtocolTrigger` | `SQLJournal` | `SQLSnapshotStore` | `ProtocolChannel` | `InProcessTransport` | Multi-Turn |
| **Protocol session (remote)** | `MQTrigger` | `DurableJournal` | `DurableSnapshotStore` | `MQChannel` | `MessageQueueTransport` | Multi-Turn, protocol server in any language |
| **Long-running task** | `ScheduledTrigger` | `DurableJournal` | `DurableSnapshotStore` | `CallbackChannel` | `InProcessTransport` | Multi-Turn, crash-recoverable |
| **Channel wake-up** (openclaw/hermes) | `ChannelTrigger` | `DurableJournal` | `DurableSnapshotStore` | `GatewayChannel` | `MessageQueueTransport` | Dormant between wake-ups, crash-recoverable |

### Key Architectural Concepts

Beyond the six dimensions, this RFC introduces three cross-cutting concepts:

1. **Durability Model**: RunLoop supports snapshot/resume. Journal + SnapshotStore follow Akka/Pekko's clean separation (journal for events, snapshot for state). Journal supports `append()` (delta events) and `upsert()` (entity-state events, mirroring ACP v2's ToolCallUpdate). `resume()` is a first-class operation coordinating both layers. On crash, `journal.resume(snapshot_store)` recovers from the latest snapshot and replays the journal. This makes long-running and channel wake-up modes crash-safe.

2. **EventEnvelope**: A language-agnostic serialization format (JSON + schema versioning) for all events flowing through EventTransport. This enables protocol servers to be implemented in any language — they consume EventEnvelopes from the message queue, not Python objects.

3. **ProtocolBridge**: A translation layer for cross-version protocol bridging (e.g., ACP v2 server ↔ ACP v1 client). Inspired by ACP's `conversion.rs`. ProtocolBridge is a CommChannel decorator that translates between protocol versions at the boundary.

### Relationship to RFC-0041

RFC-0041 (Run vs Turn Separation) is a **prerequisite** for this RFC. It defines the Run/Turn separation within session context — restructuring `RunHandle` into a persistent idle/running/done state machine with unified steer/followup. This RFC extends that work to the **full lifecycle architecture**, covering:

1. **Standalone execution** without SessionPool (RFC-0041 mentions this as a goal but doesn't design it)
2. **Long-running tasks** (scheduled, background, multi-day)
3. **Channel wake-up** (gateway-driven, dormant between external stimuli)
4. **Cross-cutting concerns**: state persistence, event durability, communication patterns

RFC-0041's Run/Turn separation is **Phase 1** of this RFC's implementation plan. The TriggerSource/Journal/SnapshotStore/CommChannel abstractions are built **on top of** the RunLoop that RFC-0041 defines.

## Table of Contents

- Background & Context
- [Problem Statement](#problem-statement)
- Goals & Non-Goals
- [Evaluation Criteria](#evaluation-criteria)
- [Options Analysis](#options-analysis)
- [Recommendation](#recommendation)
- [Technical Design](#technical-design)
- Security Considerations
- Implementation Plan
- [Open Questions](#open-questions)
- Decision Record
- [References](#references)

---

## Background & Context

### Current State

AgentPool's execution entry points are scattered across multiple layers:

```
┌─────────────────────────────────────────────────────────────┐
│  Entry Points (5 different paths, no shared abstraction)     │
├─────────────────────────────────────────────────────────────┤
│  1. agent.run() / agent.run_stream()  — standalone          │
│  2. SessionController.receive_request()  — protocol session │
│  3. BackgroundTaskProvider  — long-running workers          │
│  4. WatchCommand  — file/system trigger                     │
│  5. (future) Channel gateway  — not yet designed            │
└─────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  Execution Layer (partially unified by RFC-0041)             │
│  RunHandle (idle/running/done) → Turn.execute()              │
│  NativeTurn | ACPTurn                                         │
└─────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  Output Layer (3 different delivery mechanisms)              │
│  1. AsyncIterator[RichAgentStreamEvent]  — standalone       │
│  2. EventBus → ProtocolEventConsumerMixin  — protocol       │
│  3. Storage write  — background workers                      │
└─────────────────────────────────────────────────────────────┘
```

**Key observation**: The execution layer (RunHandle → Turn) is being unified by RFC-0041. But the **input** (how prompts arrive) and **output** (how responses are delivered) layers remain fragmented. Each execution mode has its own input handling and output delivery, with no shared abstraction.

### Cross-Framework Research Summary

The full research is in `docs/design/lifecycle-analysis.md`. Key patterns relevant to this RFC:

| Framework | Input Pattern | Output Pattern | State Pattern |
|-----------|--------------|----------------|---------------|
| **pi** | Pure event-stream loop; steering+followup dual queue | Same event stream | In-memory |
| **hermes-agent** | Gateway adapters (Telegram/Discord/Slack); cron scheduler | Same gateway channel | In-memory + learning loop |
| **opencode** | SessionInput (admit→promote pipeline) | Durable EventV2 (SQL event store + replay) | SQL-backed, projector-based |
| **ACP v2** | `session/prompt` (fire-and-forget) + `session/inject` (steer/queue) | `StateUpdate` events (Running/Idle/RequiresAction) | Protocol-managed |
| **claw-code** | Rust ConversationRuntime; plugin lifecycle | Plugin event dispatch | Rust session persistence |
| **deer-flow** | LangChain middleware chain (26 middlewares) | LangGraph event stream | Checkpoint-based |

**Pattern synthesis**: Every framework decomposes into the same four concerns. The difference is which concern is hardcoded vs. pluggable:

- pi hardcodes input/output as event stream; state is in-memory only.
- hermes-agent hardcodes gateway adapters; no durable state.
- opencode makes output durable (EventV2); input is protocol-specific.
- ACP v2 makes input/output protocol-level; state is caller-managed.

**AgentPool's opportunity**: Make all six dimensions pluggable from the start, with sensible defaults.

### Historical Context

| Date | Change | Relevance |
|------|--------|-----------|
| 2026-04-26 | RFC-0029: `inject_prompt()`/`queue_prompt()` | First attempt at idle/wake — caller provides reactivation loop |
| 2026-06-15 | RFC-0037: Unify steer/followup | Recognized dual-system redundancy; mapped to pydantic-ai `enqueue()` |
| 2026-06-27 | RFC-0041: Run vs Turn separation | Restructured RunHandle to persistent idle/running/done; unified steer/followup at Run level |
| 2026-07-08 | Lifecycle analysis (`docs/design/lifecycle-analysis.md`) | Cross-framework research identifying 6 pluggable dimensions |

### Glossary

| Term | Definition |
|------|------------|
| **RunLoop** | The core execution loop that drives Turns. Owns the idle/running/done state machine. Built on RFC-0041's restructured RunHandle. Supports checkpoint/resume. |
| **Turn** | Single reactive cycle: prompt → model → tools → response. Agent-type-specific (NativeTurn, ACPTurn). Defined by RFC-0041. |
| **TriggerSource** | Pluggable abstraction for how prompts arrive at the RunLoop. Bridges external stimuli to internal message queue. |
| **Journal** | (Formerly "WAL") Pluggable abstraction for event persistence, crash recovery, and replay. Owned by CommChannel. Implements append + upsert semantics (Akka/Pekko journal model). Controls event-layer durability guarantees. Events are persisted to the journal before delivery to CommChannel, ensuring crash safety: if process dies after journal append but before delivery, the event is recoverable on restart. |
| **SnapshotStore** | Pluggable abstraction for state snapshots at Turn boundaries, turn results for idempotency, and crash recovery state. Owned by RunLoop. Implements save/load semantics (Akka/Pekko snapshot-store model). Controls loop-layer durability guarantees. |
| **CommChannel** | Pluggable abstraction for delivering events and responses back to the caller. May also receive feedback (steer/followup). |
| **EventTransport** | Pluggable abstraction for the wire protocol between RunLoop and external consumers. Enables language-agnostic protocol servers via MQ backends. |
| **EventEnvelope** | Language-agnostic serialization format (JSON + schema versioning) for all events flowing through EventTransport. |
| **ProtocolBridge** | CommChannel decorator that translates between protocol versions at the boundary (e.g., ACP v2↔v1). |
| **StateUpdate** | Protocol-agnostic state notification event: `Running | Idle(stop_reason) | RequiresAction`. Inspired by ACP v2. |
| **Feedback** | Messages flowing from CommChannel back to the RunLoop (e.g., user steering, channel replies). Distinct from TriggerSource prompts. |
| **Snapshot** | (Formerly "Checkpoint") Full state image at a Turn boundary. Combined with journal replay, enables efficient crash recovery without full history replay. Inspired by Akka/Pekko's `snapshot-store` (periodic state images). |
| **Recovery Point** | Logical sequence number in the journal where execution can resume after a crash. Defined by the last committed Turn boundary. Not a named primitive — it's an emergent property of the journal's sequence ordering. |
| **Committed** | Journal entries that have been fully processed and checkpointed. Committed entries are immutable and safe to compact. |
| **Inflight** | Journal entries written but not yet checkpointed. On crash, inflight entries must be replayed or rolled back. |
| **turn_id** | Unique identifier for a Turn. Used as idempotency key: on crash recovery, the RunLoop checks if the Turn was already completed before re-executing. |
| **Replay** | Traversing journal events from a given sequence number to rebuild state. Used for crash recovery, debugging, audit, and divergence detection. |
| **Compaction** | Discarding journal entries before a given sequence number after a snapshot has been taken. Prevents unbounded journal growth. |

#### Terminology Cross-Reference

This RFC's terminology is grounded in established frameworks:

| This RFC | Akka/Pekko | Temporal | Flink | Durable Functions |
|----------|-----------|----------|-------|-------------------|
| Journal | journal (EventAdaptor) | Event History | changelog | orchestration history |
| Snapshot | snapshot-store | (unified with history) | checkpoint | (unified with history) |
| Recovery Point | sequenceNr | WorkflowTask boundary | checkpoint ID | checkpoint |
| Committed | persisted + snapshotted | completed WorkflowTask | completed checkpoint | checkpointed |
| Inflight | persisted, not snapshotted | pending WorkflowTask | in-flight data | uncheckpointed |
| Replay | receiveRecover | Replay | (not applicable) | Replay |
| Compaction | (manual journal cleanup) | Continue-As-New | (automatic) | ContinueAsNew |

**Key insight from cross-framework research**: No framework names "crash point" as a primitive — it is always defined by the recovery mechanism. Temporal: WorkflowTask end. Akka: sequenceNr in journal. Flink: completed checkpoint ID. AgentPool follows Akka's clean separation: journal (event log) and snapshot (state image) are separate concepts, both pluggable via Journal + SnapshotStore.

---

## Problem Statement

### Problem 1: No Unified Execution Entry Point

AgentPool has 5 different entry points for agent execution, each with its own lifecycle management:

| Entry Point | File | Lifecycle Owner |
|-------------|------|-----------------|
| `agent.run()` | `agents/base_agent.py` | `BaseAgent._run_stream_once()` |
| `agent.run_stream()` | `agents/base_agent.py` | Same, but yields events |
| `SessionController.receive_request()` | `orchestrator/session_controller.py` | `RunHandle` (RFC-0041) |
| `BackgroundTaskProvider` | `tool_impls/workers/` | Manual task management |
| `WatchCommand` | `wolfharness_commands/` | CLI-managed loop |

No shared abstraction connects them. Adding a new execution mode (e.g., channel gateway) requires building an entirely new entry point.

### Problem 2: Input/Output Coupling

The input (how prompts arrive) and output (how responses are delivered) are tightly coupled to the execution mode:

- **Standalone**: Input is a function argument; output is an `AsyncIterator`.
- **Protocol session**: Input is `receive_request()`; output is `EventBus` subscription.
- **Background worker**: Input is a job definition; output is a storage write.

Changing the output mechanism (e.g., adding durable event logging to standalone mode) requires modifying the execution path itself.

### Problem 3: No Channel Wake-Up Support

The "channel wake-up" pattern — where an agent is dormant and wakes up in response to external messages (chat messages, webhooks, file changes) — has no first-class support. hermes-agent solves this with gateway adapters and a 7000-line loop; AgentPool has no solution.

This pattern is increasingly important as AgentPool is used for:
- Chat bot integrations (Telegram/Discord/Slack/WeCom)
- File watch triggers
- Webhook-driven automation
- Scheduled background tasks

### Problem 4: State Persistence is Mode-Specific

State persistence (message history, run state, event log) is handled differently per mode:
- Standalone: in-memory, lost on exit
- Protocol session: `SQLJournal + SQLSnapshotStore` (SQL-backed)
- Background worker: storage write only

There's no way to add durability to standalone mode without rewriting the execution path. opencode's durable EventV2 pattern (SQL event store + replay + divergence detection) is not available.

### Problem 5: Steer/Followup Scope is Limited

RFC-0041 unifies steer/followup at the Run level, but only within session context. Standalone execution has no steer/followup support. Channel wake-up requires steer/followup from external sources (e.g., a user sending a correction message to a chat bot mid-task).

### Problem 6: No Durable Execution / Crash Recovery

AgentPool has no crash recovery model. If the process dies mid-Turn:
- In-flight work is lost (no checkpoint)
- Message history may be partially persisted (inconsistent state)
- Long-running tasks have no resume capability
- Channel wake-up agents lose all context on restart

opencode solves this with durable EventV2 (SQL event store + projectors + replay with divergence detection). AgentPool's EventBus is in-memory with optional replay buffer — events are silently dropped on overflow. There is no journal, no snapshot, no crash recovery procedure.

**Specific gaps**:
- No checkpoint at Turn boundaries — a crash mid-Turn loses the entire Turn
- No write-ahead log — events are published fire-and-forget, not persisted before delivery
- No idempotency keys — re-execution after crash may duplicate side effects (tool calls, API calls)
- No recovery procedure — on restart, there's no way to know where to resume

### Problem 7: Protocol Layer is Python-Only

All protocol servers (ACP, OpenCode, AG-UI, OpenAI API) are implemented in Python and run in-process. There's no way to:
- Implement a protocol server in Rust/Go/TypeScript for performance
- Run the protocol server in a separate process for isolation
- Use a message broker (Redis/NATS/Kafka) as the event transport

This coupling means AgentPool can't be used in polyglot environments where the protocol layer needs to be in a different language than the execution layer.

### Problem 8: No Protocol Version Bridging

ACP has v1 and v2 with incompatible schemas. AgentPool currently supports both but has no general abstraction for bridging between protocol versions. The ACP reference implementation uses `conversion.rs` for cross-version translation, but AgentPool has no equivalent — each version is handled by separate code paths.

This problem extends beyond ACP: OpenCode, AG-UI, and other protocols will also evolve. A general `ProtocolBridge` abstraction is needed to handle version translation at the protocol boundary, supporting patterns like:
- ACP v2 server ↔ ACP v1 client (downgrade)
- ACP v1 server ↔ ACP v2 client (upgrade)
- Future protocol versions without code changes to RunLoop

### Impact of Inaction

- Each new execution mode requires a new entry point, lifecycle manager, and output handler
- Channel wake-up pattern remains unsupported, limiting AgentPool's applicability to chat bot / gateway scenarios
- State durability is all-or-nothing (either you use SessionPool or you don't)
- Cross-mode features (e.g., durable event logging for standalone runs) require mode-specific implementation
- **Crash during long-running tasks loses all work** — no checkpoint/resume
- **Protocol layer is locked to Python** — no polyglot support, no MQ-based decoupling
- **Protocol version migration requires code changes** — no bridging abstraction
- **No replay capability** — can't debug or audit past executions

---

## Goals & Non-Goals

### Goals (In Scope)

1. **Define six pluggable dimensions**: RunLoop, TriggerSource, Journal, SnapshotStore, CommChannel, EventTransport
2. **Cover all four execution modes**: standalone, protocol session, long-running task, channel wake-up
3. **Make execution mode a configuration choice**: Same RunLoop + Turn execution, different dimension implementations
4. **First-class channel wake-up**: Support dormant agents that wake on external stimuli
5. **Composable durability**: Any execution mode can use any Journal + SnapshotStore combination (in-memory to durable)
6. **Unified steer/followup across all modes**: RFC-0041's steer/followup extends to channel wake-up via CommChannel feedback
7. **StateUpdate event**: Protocol-agnostic state notification (Running/Idle/RequiresAction) across all modes
8. **Architecture-level design**: This RFC defines interfaces and dimension boundaries; implementation details deferred to sub-RFCs
9. **Durable execution**: Snapshot/resume at Turn boundaries, journal for in-flight events, crash recovery procedure, idempotency keys for side-effect safety
10. **Persistence & replay as first-class**: Event log + snapshot hybrid, state reconstruction from events, replay for debugging/audit, divergence detection
11. **Language-agnostic protocol layer**: EventTransport with MQ backends (Redis/NATS/Kafka), EventEnvelope serialization format, protocol servers implementable in any language
12. **Protocol version bridging**: ProtocolBridge abstraction for cross-version translation (ACP v2↔v1, future protocol evolution)

### Non-Goals (Out of Scope)

1. **Turn execution internals**: Deferred to RFC-0041 (Run/Turn separation)
2. **Specific gateway implementations**: Telegram/Discord/Slack adapters are implementation, not architecture
3. **Distributed RunLoop**: RunLoop is single-process; multi-node RunLoop coordination is future work (but EventTransport enables distributed consumers)
4. **Protocol server implementation**: ACP v2 / OpenCode / AG-UI protocol details are separate efforts
5. **Specific scheduler implementations**: Cron, interval, event-driven schedulers are implementations of TriggerSource
6. **Storage schema design**: Journal + SnapshotStore define the interface; SQL/document/event-log schemas are implementation choices
7. **Migration path for existing code**: Each sub-RFC will define its own migration
8. **Specific MQ broker selection**: Redis/NATS/Kafka are interchangeable implementations of EventTransport

---

## Evaluation Criteria

| Criterion | Weight | Measurement |
|-----------|--------|-------------|
| **Mode coverage** | High | All 4 execution modes expressible without new entry points |
| **Dimension orthogonality** | High | Changing one dimension doesn't require changing others |
| **Complexity reduction** | Medium | Net reduction in entry points and lifecycle management code |
| **Channel wake-up support** | High | Dormant agent can wake on external stimulus and respond via gateway |
| **Durability composability** | High | Any mode can use any Journal + SnapshotStore combination without code changes |
| **Crash recovery** | High | Process crash mid-Turn → restart → resume from checkpoint, no lost work |
| **Replay capability** | High | Full state reconstruction from event log; replay produces identical state |
| **Language-agnostic protocol** | Medium | Protocol servers implementable in any language via EventTransport |
| **Protocol version bridging** | Medium | ACP v2↔v1 bridging works without RunLoop changes |
| **Backward compatibility** | High | Existing YAML configs and Python APIs continue to work |
| **Implementation feasibility** | High | Each phase deliverable in 1-2 sprints |

---

## Options Analysis

### Option 1: Six Pluggable Dimensions (Recommended)

**Design**: Decompose lifecycle into RunLoop + TriggerSource + Journal + SnapshotStore + CommChannel + EventTransport. RunLoop is the core; the other five are injected dependencies.

```
┌─────────────────────────────────────────────────────┐
│                    RunLoop                           │
│  idle → running (Turn) → idle | done                │
│  steer() / followup() / close()                     │
│                                                     │
│  ┌─────────────┐  ┌──────────┐  ┌────────────┐  │
│  │TriggerSource│  │  Journal  │  │CommChannel │ │
│  │ (injected)  │  │(injected) │  │ (injected) │ │
│  └─────────────┘  └──────────┘  └────────────┘ │
│  ┌─────────────┐  ┌──────────────┐             │
│  │SnapshotStore│  │EventTransport│             │
│  │ (injected)  │  │  (injected)  │             │
│  └─────────────┘  └──────────────┘             │
└─────────────────────────────────────────────────────┘
```

**Pros**:
- Maximum orthogonality: each dimension independently swappable
- All 4 execution modes are pure configuration
- Channel wake-up is a natural composition (ChannelTrigger + GatewayChannel + DurableJournal + DurableSnapshotStore)
- StateUpdate event flows through CommChannel to any consumer
- Steer/followup works across all modes via CommChannel feedback loop
- Aligns with RFC-0041's RunLoop definition (RunLoop IS the restructured RunHandle)

**Cons**:
- 6 new abstractions to learn (but each is small and focused)
- Potential over-abstraction if most users only need 1-2 modes
- CommChannel and TriggerSource overlap for bidirectional channels (chat gateways)

**Mitigation for overlap**: For bidirectional channels (where input and output use the same medium, e.g., a Telegram chat), provide a `BidirectionalChannel` that implements both TriggerSource and CommChannel. This is a convenience, not a requirement.

### Option 2: Monolithic RunLoop with Mode Parameter

**Design**: Single RunLoop class with a `mode` parameter that selects internal behavior.

```python
class RunLoop:
    def __init__(self, mode: Literal["standalone", "session", "long_running", "channel"], ...):
        ...
```

**Pros**:
- Simpler mental model (one class, one parameter)
- No abstraction overhead
- Easy to understand for new users

**Cons**:
- Not orthogonal: changing output mechanism requires changing mode
- Adding a new mode requires modifying RunLoop
- Channel wake-up would be a 5th mode, requiring more RunLoop changes
- State durability can't be independently selected
- Violates open/closed principle
- Doesn't compose: can't have "standalone with durable state" or "channel with in-memory state"

**Verdict**: Rejected. Too rigid. The cross-framework research shows that all four concerns (input, execution, state, output) vary independently across frameworks. The six dimensions map to the four concerns with state split into Journal (events) and SnapshotStore (state).

### Option 3: Event-Sourced Architecture (opencode-style)

**Design**: Everything is an event. RunLoop is a projector over an event log. TriggerSource appends to the event log. CommChannel subscribes to the event log. Journal IS the event log.

```
EventLog (single source of truth)
  ↑ (append)        ↓ (subscribe)     ↓ (project)
TriggerSource       CommChannel       RunLoop (projector)
```

**Pros**:
- Maximum durability and replayability (opencode pattern)
- Single source of truth
- Natural audit trail
- Replay-based debugging
- Language-agnostic (event log can be MQ-backed)

**Cons**:
- Heavy: requires event log infrastructure even for standalone single-turn
- Latency: event serialization/deserialization on every operation
- Complexity: projectors, divergence detection, event versioning
- Not all modes need durability (standalone single-turn doesn't)
- Mismatch with ACP v2's push-based model (v2 pushes events, doesn't append to log)

**Verdict**: Rejected as the **default** architecture, but **fully absorbed** into the design:
- Journal + SnapshotStore IS event sourcing (Phase 2)
- EventTransport's EventEnvelope IS the serialized event format
- `EventLogJournal` + `EventLogSnapshotStore` is the maximum-durability Journal + SnapshotStore implementation
- MQ-backed EventTransport provides the distributed event log

This gives users the choice: `MemoryJournal + MemorySnapshotStore` for lightweight, `DurableJournal + DurableSnapshotStore` for crash recovery, `EventLogJournal + EventLogSnapshotStore` for full event sourcing, and `MessageQueueTransport` for distributed event log.

### Option 4: Plugin-Based Architecture

**Design**: RunLoop has a plugin system. Each lifecycle concern (input, state, output, scheduling) is a plugin. Plugins are discovered via entry points.

**Pros**:
- Extensible without code changes to RunLoop
- Community can contribute plugins (e.g., Slack gateway plugin)
- Natural for a framework that already uses entry points

**Cons**:
- Plugin discovery and lifecycle adds complexity
- Plugin ordering and dependencies are hard to manage
- Too loose: no clear contract between plugins
- Overlaps with existing ResourceProvider system

**Verdict**: Rejected as architecture. The six-dimension approach IS a plugin system, but with clear contracts (6 interfaces) rather than an open-ended plugin registry. Individual dimension implementations (e.g., `TelegramChannelTrigger`) can be registered via entry points.

### Comparison Matrix

| Criterion | Option 1 (6 Dimensions) | Option 2 (Mode Param) | Option 3 (Event-Sourced) | Option 4 (Plugins) |
|-----------|------------------------|----------------------|-------------------------|-------------------|
| Mode coverage | ✅ All 4 | ✅ All 4 | ✅ All 4 | ✅ All 4 |
| Orthogonality | ✅ Full | ❌ Coupled | ✅ Full | ⚠️ Partial |
| Complexity | Medium | Low | High | High |
| Channel wake-up | ✅ Natural | ❌ Needs new mode | ✅ Natural | ✅ Natural |
| Durability composability | ✅ Full | ❌ Mode-locked | ✅ Full | ✅ Full |
| Backward compatibility | ✅ Wrappers | ✅ Add mode param | ⚠️ Requires event log | ⚠️ Plugin migration |
| Implementation feasibility | ✅ Phased | ✅ Simple | ⚠️ Heavy infra | ⚠️ Plugin system first |

---

## Recommendation

**Option 1: Six Pluggable Dimensions** — RunLoop + TriggerSource + Journal + SnapshotStore + CommChannel + EventTransport.

This is the only option that achieves full orthogonality while remaining implementable in phases. The six dimensions map to the four concerns identified in cross-framework research, and each dimension has well-understood implementations from existing frameworks.

Key design decisions:

1. **RunLoop IS RFC-0041's restructured RunHandle** — not a new class. The six-dimension architecture is an extension of RFC-0041, not a replacement.
2. **TriggerSource and CommChannel may be the same object** for bidirectional channels. Provide `BidirectionalChannel` convenience base.
3. **Journal + SnapshotStore are opt-in for durability** — default is `MemoryJournal + MemorySnapshotStore`. But Journal and SnapshotStore interfaces are defined from the start; `MemoryJournal` and `MemorySnapshotStore` implement them in-memory.
4. **StateUpdate event** is published through CommChannel, not directly through EventBus. This allows non-EventBus consumers (e.g., webhook callbacks, MQ consumers) to receive state notifications.
5. **Steer/followup flows through CommChannel feedback** — for channel wake-up, the user's reply message IS the steer. The CommChannel receives it and calls `RunLoop.steer()`.
6. **EventTransport is the language-agnostic boundary** — EventEnvelope (JSON, schema-versioned) is the wire format. In-process by default; MQ-backed for polyglot/distributed setups.
7. **Persistence/replay is core, not optional** — Journal + SnapshotStore are part of the interface from Phase 1. `DurableJournal + DurableSnapshotStore` implementation in Phase 2 enables crash recovery for all non-standalone modes.
8. **ProtocolBridge handles version translation** — ACP v2↔v1 bridging is a CommChannel decorator. RunLoop is unaware of protocol versions. New protocol versions only need a new ProtocolBridge implementation.

---

## Technical Design

### Dimension 1: RunLoop

The RunLoop is the core execution loop. It is defined by RFC-0041 as the restructured `RunHandle` with idle/running/done states. This RFC adds the dimension injection points.

```python
from __future__ import annotations
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from wolfharness.orchestrator.turn import Turn
    from wolfharness.agents.events import RichAgentStreamEvent

class RunLoop:
    """Core execution loop. Built on RFC-0041's restructured RunHandle.

    The RunLoop owns:
    - idle/running/done state machine (from RFC-0041)
    - message queue (steer/followup, from RFC-0041)
    - Turn execution (delegated to Turn implementations, from RFC-0041)

    This RFC adds:
    - TriggerSource injection (how prompts arrive)
    - Journal injection (event persistence, passed to CommChannel)
    - SnapshotStore injection (state persistence, used by RunLoop)
    - CommChannel injection (how events are delivered, owns Journal)
    """

    def __init__(
        self,
        agent,  # MessageNode — the agent to execute
        trigger_source: TriggerSource | None = None,
        journal: Journal | None = None,                # Event layer
        snapshot_store: SnapshotStore | None = None,   # Loop layer
        comm_channel: CommChannel | None = None,
        session_id: str = "default",                   # For StateUpdate key derivation
    ) -> None:
        self._agent = agent
        self._session_id = session_id
        self._trigger = trigger_source or ImmediateTrigger()
        self._journal = journal or MemoryJournal()
        self._snapshots = snapshot_store or MemorySnapshotStore()
        # CommChannel receives Journal — it handles event persistence internally
        # When comm_channel is provided, inject our journal to ensure single
        # journal instance across RunLoop (resume) and CommChannel (append/upsert)
        if comm_channel is not None:
            comm_channel._journal = self._journal  # type: ignore[attr-defined]
            self._comm = comm_channel
        else:
            self._comm = DirectChannel(journal=self._journal)
        # idle/running/done state from RFC-0041
        self._status: RunStatus = "idle"
        self._message_queue: list[QueuedMessage] = []

    async def start(self, initial_prompt: str | None = None) -> None:
        """Start the RunLoop. Attempts resume first."""
        # 1. Attempt crash recovery via journal.resume() — first-class resume
        resumed = await self._journal.resume(self._snapshots)

        if resumed is None:
            # Fresh start — no prior state
            await self._snapshots.save(self._get_state_snapshot())
        elif resumed.is_inflight:
            # Crash during in-flight Turn — replay events to consumer
            # Set replay mode to prevent re-journaling (events already in journal)
            self._comm._replaying = True
            for event in resumed.events:
                await self._comm.publish(event)
            self._comm._replaying = False
            await self._comm.publish(
                StateUpdate(session_id=self._session_id, state=RunState.IDLE, stop_reason="crash_recovery")
            )
            self._state = resumed.state
        else:
            # Normal recovery — resume from snapshot
            self._state = resumed.state

        # 2. Continue normal execution
        if initial_prompt is not None:
            self._message_queue.append(QueuedMessage(content=initial_prompt, priority="normal"))
        await self._trigger.subscribe(self)
        await self._comm.attach(self)
        await self._run_loop()

    @property
    def is_running(self) -> bool:
        """Public read-only property for RunLoop running state.

        CommChannel implementations use on_state_change() callback instead
        of accessing this property directly. This property is provided for
        external consumers (e.g., health checks, monitoring).
        """
        return self._status == "running"

    async def steer(self, content: str) -> None:
        """Inject a steer message into active Turn (from RFC-0041)."""
        # RFC-0041 defines the actual implementation
        ...

    async def followup(self, content: str) -> None:
        """Queue a followup message (from RFC-0041)."""
        ...

    async def close(self) -> None:
        """Graceful shutdown. Drains pending, sets done."""
        ...

    async def _run_loop(self) -> None:
        """The idle → running → idle | done loop (from RFC-0041)."""
        while not self._closing:
            if not self._message_queue:
                self._status = "idle"
                await self._comm.on_state_change(RunState.IDLE)
                await self._comm.publish(StateUpdate(session_id=self._session_id, state=RunState.IDLE))
                await self._wait_for_wake()  # asyncio.Event from RFC-0041
                continue

            self._status = "running"
            await self._comm.on_state_change(RunState.RUNNING)
            await self._comm.publish(StateUpdate(session_id=self._session_id, state=RunState.RUNNING))
            turn_id = generate_turn_id()
            turn = self._agent.create_turn(prompts=self._message_queue, turn_id=turn_id, ...)

            # Check if this turn was already completed (crash recovery)
            if await self._snapshots.has_turn_result(turn_id):
                self._message_queue.clear()
                continue

            # Turn execution — CommChannel handles journaling internally
            # CommChannel calls journal.append() for deltas, journal.upsert() for entity state
            async for event in turn.execute():
                await self._comm.publish(event)

            # Snapshot after Turn completion (loop layer)
            await self._snapshots.save(self._get_state_snapshot())
            await self._snapshots.save_turn_result(turn_id, turn.result)
            self._message_queue.clear()

        self._status = "done"
        await self._comm.on_state_change(RunState.DONE)
        await self._comm.publish(StateUpdate(session_id=self._session_id, state=RunState.DONE))
```

**Key point**: The RunLoop code is **identical** regardless of execution mode. The mode is determined entirely by which TriggerSource, Journal, SnapshotStore, and CommChannel are injected.

### Dimension 2: TriggerSource

TriggerSource abstracts how prompts arrive at the RunLoop. It is the "input" dimension.

```python
@runtime_checkable
class TriggerSource(Protocol):
    """Abstracts how prompts arrive at the RunLoop."""

    async def subscribe(self, run_loop: RunLoop) -> None:
        """Attach to a RunLoop. Called once during start()."""
        ...

    async def poll(self) -> Prompt | None:
        """Poll for next prompt. Returns None if no prompt available."""
        ...

    async def close(self) -> None:
        """Cleanup resources."""
        ...
```

#### TriggerSource Implementations

**ImmediateTrigger** (standalone mode):
```python
class ImmediateTrigger(TriggerSource):
    """Single prompt, delivered immediately. For standalone execution."""

    def __init__(self, prompt: str) -> None:
        self._prompt = prompt
        self._delivered = False

    async def subscribe(self, run_loop: RunLoop) -> None:
        # No-op: prompt is already set
        pass

    async def poll(self) -> Prompt | None:
        if self._delivered:
            return None
        self._delivered = True
        return Prompt(content=self._prompt)
```

**ProtocolTrigger** (session mode):
```python
class ProtocolTrigger(TriggerSource):
    """Bridges protocol handler (ACP/OpenCode/AG-UI) to RunLoop.

    Wraps SessionController.receive_request() — prompts arrive
    via protocol messages and are forwarded to the RunLoop.
    """

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._queue: asyncio.Queue[Prompt] = asyncio.Queue()

    async def deliver(self, content: str, priority: str = "normal") -> None:
        """Called by protocol handler when a prompt arrives."""
        await self._queue.put(Prompt(content=content, priority=priority))

    async def poll(self) -> Prompt | None:
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
```

**ScheduledTrigger** (long-running task mode):
```python
class ScheduledTrigger(TriggerSource):
    """Triggers RunLoop on a schedule. For long-running tasks.

    Supports cron expressions, intervals, and one-shot delays.
    """

    def __init__(
        self,
        schedule: str | float,  # cron expression or interval seconds
        prompt_template: str,   # Jinja2 template for prompt generation
    ) -> None:
        self._schedule = schedule
        self._prompt_template = prompt_template
        self._next_run: datetime | None = None

    async def subscribe(self, run_loop: RunLoop) -> None:
        self._next_run = self._compute_next_run()

    async def poll(self) -> Prompt | None:
        if self._next_run is None:
            return None
        if datetime.now() >= self._next_run:
            self._next_run = self._compute_next_run()
            prompt = self._render_prompt()
            return Prompt(content=prompt)
        return None
```

**ChannelTrigger** (channel wake-up mode):
```python
class ChannelTrigger(TriggerSource):
    """Triggers RunLoop on external channel messages.

    For chat bot / gateway patterns. Messages from external sources
    (Telegram, Discord, Slack, webhooks) are delivered as prompts.

    Typically paired with GatewayChannel (which implements both
    TriggerSource and CommChannel for bidirectional communication).
    """

    def __init__(self, channel_config: ChannelConfig) -> None:
        self._config = channel_config
        self._queue: asyncio.Queue[Prompt] = asyncio.Queue()
        self._listener: asyncio.Task | None = None

    async def subscribe(self, run_loop: RunLoop) -> None:
        self._listener = asyncio.create_task(self._listen())

    async def _listen(self) -> None:
        """Listen on channel for incoming messages."""
        async for message in self._config.source:
            prompt = Prompt(
                content=message.content,
                metadata={"source": message.source, "channel": message.channel},
            )
            await self._queue.put(prompt)

    async def poll(self) -> Prompt | None:
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
```

### Dimension 3: Journal + SnapshotStore

**Revision 4 change**: The former monolithic `StateStore` is split into two separate dimensions, aligned with Akka/Pekko's clean separation of journal (event persistence) and snapshot-store (state persistence):

| Dimension | Layer | Controlled By | Responsibility |
|-----------|-------|--------------|----------------|
| **Journal** | Event layer | CommChannel | Event persistence (append + upsert), replay, compaction |
| **SnapshotStore** | Loop layer | RunLoop | State snapshots at Turn boundaries, turn results, crash recovery state |

**Why split**: Journal and snapshot are different concerns owned by different layers:
- Journal records **events** as they flow through CommChannel — it's an event-layer concern
- SnapshotStore records **state** at Turn boundaries — it's a loop-layer concern
- CommChannel knows event semantics (delta vs entity update → append vs upsert); RunLoop doesn't
- RunLoop knows when to snapshot (Turn boundaries); CommChannel doesn't
- They can independently evolve (swap journal to Kafka without touching snapshot implementation)

```python
@runtime_checkable
class Journal(Protocol):
    """Event-layer persistence. Controlled by CommChannel.

    The Journal records events as they flow through the system.
    It supports TWO write semantics:

    - append(): For delta events (each entry is a new record)
      Examples: PartDeltaEvent, StreamCompleteEvent, ToolCallStartEvent

    - upsert(key): For entity-state events (latest state per key replaces previous)
      Examples: ToolCallUpdateEvent (key=tool_call_id),
                StateUpdate (key=session_id),
                Message replacement (key=message_id)

    This mirrors ACP v2's ToolCallUpdate which IS an upsert operation.
    On replay, upsert keys return only the latest state per key.

    Journal owns event sequence numbers:
    - append() and upsert() both return seq
    - seq is the SOLE source of truth for event ordering
    - CommChannel uses this seq for EventEnvelope — never generates its own
    - Even MemoryJournal (non-durable) returns monotonically increasing seq
    """

    async def append(self, event: RichAgentStreamEvent) -> int:
        """Append a delta event to the journal.

        Each call creates a new journal entry.
        Used for events where every instance is meaningful (deltas, transitions).

        Returns the sequence number of the journal entry.
        """
        ...

    async def upsert(self, key: str, event: RichAgentStreamEvent) -> int:
        """Upsert an entity-state event by key.

        If an entry with the same key exists, it is replaced.
        If no entry exists, a new one is created.
        Used for events where only the latest state matters.

        Key examples:
        - tool_call_id for ToolCallUpdateEvent
        - session_id for StateUpdate
        - message_id for message replacement

        Returns the sequence number of the journal entry.
        """
        ...

    async def replay(
        self, from_seq: int = 0, to_seq: int | None = None
    ) -> AsyncIterator[RichAgentStreamEvent]:
        """Replay events from the journal.

        - Append entries: all returned, ordered by seq
        - Upsert entries: only the latest per key returned
        - Mixed: ordered by seq, upsert keys deduplicated to latest

        Used for:
        - Crash recovery (via resume())
        - Debugging: replay full session history
        - Audit: extract specific event range
        - Divergence detection: replay and compare with expected state
        """
        ...

    async def resume(
        self,
        snapshot_store: SnapshotStore,
    ) -> ResumeResult | None:
        """First-class resume operation. Coordinates Journal + SnapshotStore.

        This is the primary crash recovery entry point.
        Combines snapshot loading + journal replay into one operation.

        Internal flow:
        1. snapshot_store.load() → (state, last_journal_seq)
        2. If state is None → return None (no state to resume from)
        3. journal.replay(from_seq=state.snapshot_seq) → events since snapshot
        4. Determine if a Turn was in-flight (journal has entries but no turn_result)
        5. Return ResumeResult with state + events + in-flight flag

        Why this is on Journal (not RunLoop):
        - Resume logic requires cross-layer knowledge (snapshot + journal)
        - is_inflight determination needs both snapshot state and journal entries
        - Centralizing here avoids scattering recovery logic across RunLoop code
        - RunLoop just calls journal.resume(snapshot_store) — one call, one result

        Cross-layer dependency tradeoff:
        - Journal (event layer) receives SnapshotStore (loop layer) as a parameter
        - This is a method-level dependency, NOT a constructor-level ownership
        - Journal does NOT own SnapshotStore — it only uses it for resume()
        - Alternative: a separate RecoveryCoordinator component — rejected as
          unnecessary indirection (one method doesn't justify a new class)
        - The dependency is acceptable because resume() is the ONLY cross-layer
          operation; all other Journal/SnapshotStore methods are layer-independent
        """
        ...

    async def compact(self, before_seq: int) -> None:
        """Compact the journal by removing entries before the given sequence.

        Called after a successful snapshot to prevent unbounded journal growth.
        Entries before the latest snapshot are "committed" and safe to remove.
        """
        ...

    async def clear(self) -> None:
        """Clear all journal entries."""
        ...

    async def log_tool_execution(self, record: ToolExecutionRecord) -> None:
        """Log a tool execution for idempotent recovery."""
        ...

    async def get_tool_executions(self, turn_id: str) -> list[ToolExecutionRecord]:
        """Retrieve all tool executions for a Turn."""
        ...


@runtime_checkable
class SnapshotStore(Protocol):
    """Loop-layer persistence. Controlled by RunLoop.

    The SnapshotStore saves full state images at Turn boundaries.
    It is separate from the Journal (event-layer) because:
    - RunLoop controls WHEN to snapshot (Turn boundaries)
    - RunLoop controls WHAT state to snapshot (RunState)
    - Snapshot frequency is independent of event flow
    - Can swap snapshot implementation without touching journal

    Terminology (cross-framework alignment):
    - Snapshot = Akka snapshot-store, Flink checkpoint
    - Journal = Akka journal, Temporal Event History, Flink changelog
    - Recovery point = sequence number of last snapshot
    - Committed = journal entries backed by a snapshot
    - Inflight = journal entries not yet snapshotted
    """

    async def save(self, state: RunState) -> int:
        """Persist a full state snapshot at a Turn boundary.

        Returns the sequence number of the snapshot.
        Future journal entries after this seq are "inflight" until next snapshot.
        Entries before this seq become "committed" and are eligible for compaction.
        """
        ...

    async def load(self) -> tuple[RunState, int] | None:
        """Load latest snapshot + journal position.

        Returns (state, last_journal_seq) or None if no state exists.
        Called by Journal.resume() during crash recovery.
        """
        ...

    async def save_turn_result(self, turn_id: str, result: Any) -> None:
        """Save a completed Turn's result for idempotency."""
        ...

    async def has_turn_result(self, turn_id: str) -> bool:
        """Check if a Turn was already completed."""
        ...

    async def clear(self) -> None:
        """Clear all snapshots + turn results."""
        ...


@dataclass
class ResumeResult:
    """Result of Journal.resume() — snapshot state + events since snapshot."""
    state: RunState                    # Restored state from snapshot
    events: list[RichAgentStreamEvent]  # Journal events since snapshot
    snapshot_seq: int                  # Snapshot's journal position
    last_journal_seq: int              # Journal's last position
    is_inflight: bool                  # Whether a Turn was in-flight at crash
    inflight_turn_id: str | None       # Turn ID of in-flight Turn (None if not in-flight)
```

#### Journal + SnapshotStore Implementations

| Journal Implementation | SnapshotStore Implementation | Durability | Use Case |
|----------------------|------------------------------|-----------|----------|
| `MemoryJournal` | `MemorySnapshotStore` | None (in-process) | Standalone, testing, ephemeral |
| `SQLJournal` | `SQLSnapshotStore` | SQL-backed | Protocol sessions (current behavior, extended) |
| `DurableJournal` | `DurableSnapshotStore` | SQL + journal + snapshot | Long-running, channel wake-up (crash recovery) |
| `EventLogJournal` | `EventLogSnapshotStore` | Append-only event log | Maximum durability, audit, divergence detection |

**Key design**: Journal and SnapshotStore are independently composable. Any execution mode can use any combination. A standalone run can use `DurableJournal + DurableSnapshotStore` for crash recovery; a protocol session can use `MemoryJournal + MemorySnapshotStore` for low-latency. They MAY share the same underlying database but are separate interfaces.

#### Durability Model: Snapshot/Resume

The RunLoop snapshots at Turn boundaries — after each Turn completes, before processing the next prompt. This ensures:

1. **No lost Turns**: A completed Turn's state is always snapshotted
2. **Limited replay**: On crash, only the in-flight Turn needs replay (not full history)
3. **Idempotency**: Each Turn has a unique `turn_id`; on recovery, the RunLoop checks if the Turn was completed before replaying

**Revision 4 change**: RunLoop no longer calls `journal.append()` directly. CommChannel owns the Journal reference and handles event persistence internally — it knows whether to use `append()` or `upsert()` based on event type. RunLoop only calls `snapshot_store.save()` at Turn boundaries.

```python
class RunLoop:
    def __init__(
        self,
        agent,
        trigger_source: TriggerSource | None = None,
        journal: Journal | None = None,                # Event layer
        snapshot_store: SnapshotStore | None = None,   # Loop layer
        comm_channel: CommChannel | None = None,
        session_id: str = "default",                   # For StateUpdate key derivation
    ) -> None:
        self._agent = agent
        self._session_id = session_id
        self._trigger = trigger_source or ImmediateTrigger()
        self._journal = journal or MemoryJournal()
        self._snapshots = snapshot_store or MemorySnapshotStore()
        # CommChannel receives Journal — it handles event persistence
        # When comm_channel is provided, inject our journal to ensure single
        # journal instance across RunLoop (resume) and CommChannel (append/upsert)
        if comm_channel is not None:
            comm_channel._journal = self._journal  # type: ignore[attr-defined]
            self._comm = comm_channel
        else:
            self._comm = DirectChannel(journal=self._journal)
        self._status: RunStatus = "idle"
        self._message_queue: list[QueuedMessage] = []

    async def start(self, initial_prompt: str | None = None) -> None:
        """Start the RunLoop. Attempts resume first."""
        # 1. Attempt crash recovery via journal.resume() — first-class resume
        resumed = await self._journal.resume(self._snapshots)

        if resumed is None:
            # Fresh start — no prior state
            await self._snapshots.save(self._get_state_snapshot())
        elif resumed.is_inflight:
            # Crash during in-flight Turn — replay events to consumer
            # Set replay mode to prevent re-journaling (events already in journal)
            self._comm._replaying = True
            for event in resumed.events:
                await self._comm.publish(event)
            self._comm._replaying = False
            await self._comm.publish(
                StateUpdate(session_id=self._session_id, state=RunState.IDLE, stop_reason="crash_recovery")
            )
            self._state = resumed.state
        else:
            # Normal recovery — resume from snapshot
            self._state = resumed.state

        # 2. Continue normal execution
        if initial_prompt is not None:
            self._message_queue.append(QueuedMessage(content=initial_prompt, priority="normal"))
        await self._trigger.subscribe(self)
        await self._comm.attach(self)
        await self._run_loop()

    async def _run_loop(self) -> None:
        """The idle → running → idle | done loop (from RFC-0041)."""
        while not self._closing:
            if not self._message_queue:
                self._status = "idle"
                await self._comm.on_state_change(RunState.IDLE)
                await self._comm.publish(StateUpdate(session_id=self._session_id, state=RunState.IDLE))
                await self._wait_for_wake()
                continue

            self._status = "running"
            await self._comm.on_state_change(RunState.RUNNING)
            await self._comm.publish(StateUpdate(session_id=self._session_id, state=RunState.RUNNING))

            turn_id = generate_turn_id()
            turn = self._agent.create_turn(
                prompts=self._message_queue,
                turn_id=turn_id,  # Idempotency key
                ...
            )

            # Check if this turn was already completed (crash recovery)
            if await self._snapshots.has_turn_result(turn_id):
                self._message_queue.clear()
                continue

            # Turn execution — CommChannel handles journaling internally
            # CommChannel calls journal.append() for deltas, journal.upsert() for entity state
            async for event in turn.execute():
                await self._comm.publish(event)

            # Snapshot after Turn completion (loop layer)
            await self._snapshots.save(self._get_state_snapshot())
            await self._snapshots.save_turn_result(turn_id, turn.result)
            self._message_queue.clear()

        self._status = "done"
        await self._comm.on_state_change(RunState.DONE)
        await self._comm.publish(StateUpdate(session_id=self._session_id, state=RunState.DONE))
```

#### Crash Recovery Procedure

On process restart, the RunLoop calls `journal.resume(snapshot_store)` — a first-class resume operation that coordinates both layers:

```
1. Journal.resume(snapshot_store) internally:
   a. snapshot_store.load() → (snapshot_state, last_journal_seq)
   b. If snapshot_state is None → return None (fresh start)
   c. journal.replay(from_seq=snapshot_state.snapshot_seq) → events since snapshot
   d. Determine is_inflight:
      - journal has entries after snapshot (last_journal_seq > snapshot_seq)
      - AND snapshot_state.current_turn_id has no turn_result
      - If both true: Turn was in-flight at crash
   e. Return ResumeResult(state, events, is_inflight)

2. RunLoop.start() handles ResumeResult:
   a. If None → fresh start, no recovery needed
   b. If is_inflight=True:
      - Deliver replayed events to consumer (no gaps in event stream)
      - Publish StateUpdate(session_id=..., state=RunState.IDLE, stop_reason="crash_recovery")
      - Do NOT re-execute the Turn (LLM is non-deterministic)
   c. If is_inflight=False:
      - Resume from snapshot state normally
```

**Why resume() is on Journal (not RunLoop)**:
- Resume logic requires cross-layer knowledge (snapshot + journal)
- `is_inflight` determination needs both snapshot state and journal entries
- Centralizing in Journal avoids scattering recovery logic across RunLoop code
- RunLoop just calls `journal.resume(snapshot_store)` — one call, one result
- Aligns with Akka's `receiveRecover` (unified snapshot + event replay as one flow)

#### Tool Execution Log

**Oracle Critical Issue #1**: The journal stores event output (what the consumer saw), not tool execution records. On crash recovery, bash/API calls/file writes would re-execute, causing duplicate side effects.

**Solution**: Journal maintains a separate **tool execution log** alongside the journal:

```python
@dataclass
class ToolExecutionRecord:
    """Record of a tool execution within a Turn. Used for idempotent recovery."""
    turn_id: str           # Turn this execution belongs to
    tool_name: str         # Tool function name
    tool_input: dict       # Input arguments (JSON-serializable)
    status: str            # "completed" | "failed" | "interrupted"
    result: Any            # Tool output (JSON-serializable)
    seq: int               # Journal seq when this execution occurred
    timestamp: float       # When the execution started
    duration_ms: int       # Execution duration

# Tool execution log is on Journal (event layer) — see Journal interface above
# Journal.log_tool_execution() and Journal.get_tool_executions()
```

**Recovery behavior**:
1. On crash mid-Turn, `journal.resume(snapshot_store)` detects in-flight Turn
2. Tool execution records are loaded via `journal.get_tool_executions(turn_id)`
3. For each tool call in the replayed Turn:
   - If a record exists with `status="completed"`: replay the stored result (don't re-execute)
   - If a record exists with `status="interrupted"`: re-execute (was in-progress when crash occurred)
   - If no record exists: re-execute (hadn't started yet)
4. This is analogous to Temporal's activity-level idempotency, where each activity has an ID and results are cached

**Important limitation**: This only covers tools that go through the Journal. External side effects (e.g., an API call made directly in agent code, not through a tool) are NOT protected. Tool authors SHOULD use `turn_id` + `tool_name` as idempotency keys when calling external APIs.

#### LLM Replay Strategy

**Oracle Critical Issue #2**: Re-executing a Turn re-calls the LLM, which is non-deterministic — the same prompt produces different output. This means crash recovery can't simply re-execute interrupted Turns.

**Solution**: For in-flight Turns (crash during Turn execution), `journal.resume(snapshot_store)` replays journal events to the consumer up to the crash point, rather than re-executing the Turn from scratch.

```
Crash Recovery for In-Flight Turns:

1. journal.resume(snapshot_store) internally:
   a. snapshot_store.load() → (snapshot_state, last_journal_seq)
   b. last_journal_seq > snapshot_seq → Turn was in-flight
   c. journal.replay(from_seq=snapshot_seq, to_seq=last_journal_seq)
      → Yields all events that were already generated before the crash
      → Upsert keys return only latest state (efficient)
   d. Returns ResumeResult(is_inflight=True, events=[...])

2. RunLoop.start() handles ResumeResult:
   - Delivers replayed events to consumer (no gaps in event stream)
   - Publishes StateUpdate(session_id=..., state=RunState.IDLE, stop_reason="crash_recovery")
   - Does NOT re-execute the Turn (LLM is non-deterministic)

3. Alternative strategies (configurable):
   a. mark_interrupted (default): replay events, mark Turn interrupted
   b. re_invoke: replay events, then re-invoke LLM with accumulated context
      (preserves prior tool results via tool execution log)

Key distinction:
- Completed Turns: snapshot_store.has_turn_result(turn_id) → True → skip entirely
- In-flight Turns: journal has entries but no turn_result → replay events to consumer
```

**Design decision**: Option (a) — mark interrupted — is the default. This is safest and most predictable. Option (b) — re-invoke LLM — is available but requires explicit opt-in via `lifecycle.recover_strategy: replay | re_invoke | mark_interrupted`.

```yaml
agents:
  crash_safe_agent:
    type: native
    model: "openai:gpt-4o"
    lifecycle:
      journal: durable
      snapshot: durable
      recover_strategy: mark_interrupted  # default: safest
      # recover_strategy: replay          # replay events, don't re-invoke LLM
      # recover_strategy: re_invoke       # replay events, then re-invoke LLM with context
```

### Dimension 4: CommChannel

CommChannel abstracts communication back to the caller. It is the "output" dimension, but also supports bidirectional feedback.

```python
@runtime_checkable
class CommChannel(Protocol):
    """Abstracts event delivery and feedback reception.

    Revision 4 change: CommChannel owns the Journal reference (not RunLoop).
    This is because Journal is an event-layer concern:
    - CommChannel knows event semantics (delta vs entity update → append vs upsert)
    - CommChannel persists events BEFORE delivery (crash safety)
    - RunLoop no longer calls journal.append() — CommChannel does it internally

    Note: Journal ownership is an implementation convention, not a Protocol
    contract. CommChannel implementations SHOULD accept a Journal in their
    constructor, but the Protocol itself only defines methods.

    Decoupling from RunLoop internals:
    - CommChannel receives state changes via on_state_change() callback (observer pattern)
    - CommChannel does NOT access RunLoop._status directly
    - RunLoop calls on_state_change() whenever it transitions idle/running/done
    - CommChannel uses this to route feedback as steer (running) vs prompt (idle)

    Sequence number ownership:
    - CommChannel calls journal.append() or journal.upsert() internally
    - Journal returns seq — CommChannel uses it for EventEnvelope
    - CommChannel does NOT generate its own sequence numbers
    - All events are journaled (append or upsert); seq always comes from Journal
    - When replaying (_replaying=True), seq=0 (events already journaled)

    Replay mode:
    - During crash recovery, RunLoop sets CommChannel._replaying = True
    - When _replaying, CommChannel.publish() skips journaling (events already journaled)
    - This prevents duplicate journal entries for replayed events
    """

    _replaying: bool  # Set by RunLoop during crash recovery

    async def attach(self, run_loop: RunLoop) -> None:
        """Attach to a RunLoop. Enables feedback loop."""
        ...

    async def on_state_change(self, state: RunState) -> None:
        """Called by RunLoop when state transitions occur (observer pattern).

        CommChannel implementations use this to track RunLoop state
        without directly accessing RunLoop internals. This is critical
        for GatewayChannel, which routes incoming messages as steer
        (when running) or prompt (when idle) based on this state.

        This replaces the previous pattern of checking run_loop._status directly.
        """
        ...

    async def publish(self, event: RichAgentStreamEvent | StateUpdate) -> None:
        """Deliver an event to the consumer.

        Internally calls self._journal.append() or self._journal.upsert()
        BEFORE delivering to consumer (crash safety guarantee).
        SKIPS journaling when self._replaying is True (events already journaled).

        Event routing logic:
        - Delta events (PartDeltaEvent, StreamCompleteEvent): journal.append()
        - Entity events (ToolCallUpdateEvent, StateUpdate): journal.upsert(key)
        - Events not matching any entity pattern: fall through to append

        The key for upsert is derived from event type:
        - ToolCallUpdateEvent → key = event.tool_call_id
        - StateUpdate → key = f"state:{event.session_id}"
        - Message replacement → key = f"msg:{event.message_id}"
        """
        ...

    async def recv(self) -> Feedback | None:
        """Poll for feedback (steer/followup from consumer).

        Returns None if no feedback available.
        For unidirectional channels, always returns None.
        """
        ...

    async def close(self) -> None:
        """Cleanup resources."""
        ...
```

> **Note**: All CommChannel implementations MUST implement every method defined
> in the Protocol (`attach`, `on_state_change`, `publish`, `recv`, `close`) and
> the `_replaying` attribute. Code examples below show only the methods relevant
> to each implementation's key behavior; trivial methods (e.g., `attach()` storing
> a reference, `close()` cleaning up resources) are elided for brevity.
> `on_state_change()` may be a no-op for unidirectional channels (DirectChannel,
> CallbackChannel) that don't need state tracking.

#### CommChannel Implementations

**DirectChannel** (standalone mode):
```python
class DirectChannel:
    """Direct in-process delivery. Yields events to the caller.

    This is the default for agent.run() / agent.run_stream().
    Owns Journal reference for event persistence.
    """

    def __init__(self, journal: Journal | None = None) -> None:
        self._journal = journal or MemoryJournal()
        self._queue: asyncio.Queue[RichAgentStreamEvent | StateUpdate] = asyncio.Queue()
        self._run_loop: RunLoop | None = None
        self._replaying: bool = False  # Set by RunLoop during crash recovery

    async def publish(self, event) -> None:
        # Journal before delivery (crash safety)
        if not self._replaying:
            key = _get_upsert_key(event)
            if key:
                await self._journal.upsert(key, event)
            elif not isinstance(event, StateUpdate):
                await self._journal.append(event)
        await self._queue.put(event)

    async def recv(self) -> Feedback | None:
        return None  # Unidirectional: no feedback in standalone mode

    def events(self) -> AsyncIterator[RichAgentStreamEvent | StateUpdate]:
        """Consumer iterator. Used by agent.run_stream()."""
        ...
```

**ProtocolChannel** (session mode):
```python
class ProtocolChannel:
    """Delivers events via EventBus to protocol consumers.

    Wraps the existing ProtocolEventConsumerMixin pattern.
    Feedback arrives via SessionController.steer()/followup().
    Owns Journal reference for event persistence.
    """

    def __init__(self, event_bus: EventBus, session_id: str, journal: Journal | None = None) -> None:
        self._journal = journal or MemoryJournal()
        self._bus = event_bus
        self._session_id = session_id
        self._feedback_queue: asyncio.Queue[Feedback] = asyncio.Queue()
        self._replaying: bool = False  # Set by RunLoop during crash recovery

    async def publish(self, event) -> None:
        if not self._replaying:
            key = _get_upsert_key(event)
            if key:
                await self._journal.upsert(key, event)
            elif not isinstance(event, StateUpdate):
                await self._journal.append(event)
        await self._bus.publish(self._session_id, event)

    async def recv(self) -> Feedback | None:
        try:
            return self._feedback_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
```

**CallbackChannel** (long-running task mode):
```python
class CallbackChannel:
    """Delivers events via callback function. For long-running tasks.

    Optionally supports webhook delivery for state updates.
    Owns Journal reference for event persistence.
    """

    def __init__(
        self,
        callback: Callable[[RichAgentStreamEvent | StateUpdate], Awaitable[None]],
        webhook_url: str | None = None,
        journal: Journal | None = None,
    ) -> None:
        self._journal = journal or MemoryJournal()
        self._callback = callback
        self._webhook_url = webhook_url
        self._replaying: bool = False  # Set by RunLoop during crash recovery

    async def publish(self, event) -> None:
        if not self._replaying:
            key = _get_upsert_key(event)
            if key:
                await self._journal.upsert(key, event)
            elif not isinstance(event, StateUpdate):
                await self._journal.append(event)
        await self._callback(event)
        if self._webhook_url and isinstance(event, StateUpdate):
            await self._post_webhook(event)

    async def recv(self) -> Feedback | None:
        return None  # Long-running tasks typically don't receive feedback
```

**GatewayChannel** (channel wake-up mode):
```python
class GatewayChannel:
    """Bidirectional channel for gateway/chat-bot patterns.

    Implements both CommChannel (output) and TriggerSource (input)
    because chat gateways are bidirectional: messages come in as
    prompts, responses go out as events.

    The gateway (Telegram, Discord, Slack) is the transport.
    This class bridges between gateway messages and RunLoop.

    Owns Journal reference for event persistence.

    Decoupling from RunLoop internals:
    - GatewayChannel tracks RunLoop state via on_state_change() callback
    - Does NOT access run_loop._status directly
    - on_state_change() is called by RunLoop on every state transition
    - _is_running flag is set from the callback, not from direct access

    Feedback loop:
    1. User sends message → GatewayChannel receives → delivers as Prompt
    2. RunLoop processes → publishes events → GatewayChannel sends reply
    3. User sends correction mid-task → GatewayChannel receives →
       calls RunLoop.steer() (if running) or queues as followup (if idle)
    """

    def __init__(self, gateway: GatewayAdapter, journal: Journal | None = None) -> None:
        self._journal = journal or MemoryJournal()
        self._gateway = gateway
        self._run_loop: RunLoop | None = None
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._prompt_queue: asyncio.Queue[Prompt] = asyncio.Queue()
        self._is_running: bool = False  # Tracked via on_state_change()
        self._replaying: bool = False  # Set by RunLoop during crash recovery

    # CommChannel observer callback
    async def on_state_change(self, state: RunState) -> None:
        """Receive state changes from RunLoop (observer pattern).

        Replaces direct access to run_loop._status.
        GatewayChannel uses this to route incoming messages:
        - RUNNING: route as steer (inject into active Turn)
        - IDLE: route as new prompt (wake the RunLoop)
        """
        self._is_running = (state == RunState.RUNNING)

    # TriggerSource interface
    async def poll(self) -> Prompt | None:
        try:
            return self._prompt_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    # CommChannel interface
    async def publish(self, event) -> None:
        # Journal before delivery
        if not self._replaying:
            key = _get_upsert_key(event)
            if key:
                await self._journal.upsert(key, event)
            elif not isinstance(event, StateUpdate):
                await self._journal.append(event)

        if isinstance(event, StateUpdate):
            # Don't send state updates to chat; log internally
            return
        await self._gateway.send(event)

    async def _listen(self) -> None:
        """Listen on gateway for incoming messages."""
        async for message in self._gateway.incoming():
            if self._is_running:
                # Agent is active: route as steer
                await self._run_loop.steer(message.content)
            else:
                # Agent is idle: route as new prompt
                await self._prompt_queue.put(
                    Prompt(content=message.content, metadata={"source": message.source})
                )
```

**Helper function for upsert key derivation**:
```python
def _get_upsert_key(event: RichAgentStreamEvent | StateUpdate) -> str | None:
    """Derive the journal upsert key for an event, if applicable.

    Returns None for delta events (should use journal.append()).
    Returns a key string for entity-state events (should use journal.upsert()).

    This match/case is intentionally non-exhaustive for delta events.
    Event types not listed here return None (append semantics) via the
    fallback case. Add new entity-state event types here as they are
    introduced. Delta events (PartDeltaEvent, StreamCompleteEvent,
    ToolCallStartEvent, ToolCallCompleteEvent, RunStartedEvent,
    RunErrorEvent, RunFailedEvent, SubagentEvent, CompactionEvent,
    CustomEvent, etc.) are all append-only by default.
    """
    match event:
        case ToolCallUpdateEvent(tool_call_id=tcid):
            return f"tool_call:{tcid}"
        case StateUpdate(session_id=sid):
            return f"state:{sid}"
        case MessageReplacementEvent(message_id=mid):
            return f"msg:{mid}"
        case PlanUpdateEvent(plan_id=pid):
            return f"plan:{pid}"
        case _:
            return None  # Delta event — use append
```

### Dimension 5: EventTransport

EventTransport abstracts the wire protocol between RunLoop and external consumers. It is the "transport" dimension that enables language-agnostic protocol servers and MQ-based decoupling.

CommChannel and TriggerSource delegate to EventTransport for actual wire delivery. The default `InProcessTransport` uses asyncio queues (zero infra). MQ-backed transports (Redis/NATS/Kafka) enable protocol servers in any language.

```python
@runtime_checkable
class EventTransport(Protocol):
    """Abstracts the wire protocol between RunLoop and external consumers.

    EventTransport is the boundary between Python (RunLoop) and
    potentially-non-Python consumers (protocol servers, gateway adapters).

    All events are serialized as EventEnvelope (JSON + schema versioning)
    before transport. This ensures language-agnostic consumption.
    """

    async def publish(self, envelope: EventEnvelope) -> None:
        """Publish an event envelope to the transport.

        For MQ-backed transports, this writes to a message queue.
        For in-process, this pushes to an asyncio queue.
        """
        ...

    async def subscribe(
        self, topic: str, from_seq: int = 0
    ) -> AsyncIterator[EventEnvelope]:
        """Subscribe to events on a topic.

        For MQ-backed transports, this consumes from a message queue.
        from_seq enables replay (consumer requests events from a past position).

        For in-process, this iterates an asyncio queue with optional replay buffer.
        """
        ...

    async def ack(self, seq: int) -> None:
        """Acknowledge that an event has been processed.

        For MQ-backed transports, this commits the consumer offset.
        For in-process, this is a no-op.
        """
        ...

    async def close(self) -> None:
        """Cleanup transport resources."""
        ...
```

#### EventEnvelope Serialization

EventEnvelope is the language-agnostic serialization format. All events (RichAgentStreamEvent, StateUpdate, Feedback) are serialized to EventEnvelope before transport.

```python
@dataclass
class EventEnvelope:
    """Language-agnostic event envelope for cross-process communication.

    Schema versioned for forward/backward compatibility.
    Consumable by any language (JSON over MQ).
    """
    seq: int                    # Monotonic sequence number (from Journal, for ordering/replay)
    session_id: str             # Session/run identifier
    tenant_id: str              # Multi-tenant isolation key (filters at query/MQ level)
    turn_id: str | None         # Turn identifier (null for non-turn events)
    event_type: str             # Event type string (e.g., "part_delta", "tool_call_start")
    event_data: dict[str, Any]  # Event payload (JSON-serializable)
    schema_version: str         # Envelope schema version (e.g., "1.0")
    timestamp: float            # Unix timestamp
    metadata: dict[str, Any]    # Optional metadata (source, trace_id, etc.)
```

**Design decisions**:
- JSON serialization (not protobuf) for simplicity and debuggability. Protobuf can be added as a transport-level optimization later.
- Schema versioning enables protocol evolution without breaking consumers.
- `seq` field enables replay and ordering guarantees across MQ partitions. Seq is sourced from Journal — EventEnvelope never generates its own.
- `turn_id` enables idempotent consumption (consumer can deduplicate).
- `tenant_id` enables multi-tenant isolation at the MQ/query level.
- `metadata` field carries `trace_id` for distributed tracing across protocol servers, MQ transport, and external consumers.

#### EventTransport Implementations

**InProcessTransport** (default):
```python
class InProcessTransport(EventTransport):
    """In-process transport using asyncio queues.

    Zero infrastructure. Default for standalone and in-process protocol sessions.
    Events are never serialized — they pass as Python objects.
    Supports optional replay buffer for late subscribers.
    """

    def __init__(self, replay_buffer_size: int = 100) -> None:
        self._queues: dict[str, asyncio.Queue[EventEnvelope]] = {}
        self._replay: dict[str, list[EventEnvelope]] = defaultdict(list)
        self._replay_buffer_size = replay_buffer_size
```

**MessageQueueTransport** (MQ-backed):
```python
class MessageQueueTransport(EventTransport):
    """Message queue-backed transport for language-agnostic consumers.

    Supports multiple MQ backends (Redis Streams, NATS JetStream, Kafka).
    Protocol servers can be implemented in any language — they consume
    EventEnvelopes from the MQ, not Python objects.

    Key properties:
    - Durable: Events persist in MQ even if consumer is offline
    - Replayable: Consumer can request events from any past seq
    - Ordered: Events are ordered by seq within a session topic
    - Scalable: Multiple consumers can subscribe to the same topic

    MQ backend selection is configuration:
      transport:
        type: message_queue
        backend: redis_streams  # or nats_jetstream, kafka
        url: "redis://localhost:6379"
        topic_prefix: "wolfharness"
    """

    def __init__(
        self,
        backend: str,  # "redis_streams" | "nats_jetstream" | "kafka"
        url: str,
        topic_prefix: str = "wolfharness",
    ) -> None:
        self._backend = self._create_backend(backend, url)
        ...

    async def publish(self, envelope: EventEnvelope) -> None:
        # Serialize to JSON, write to MQ
        payload = json.dumps(envelope.to_dict())
        await self._backend.xadd(
            f"{self._topic_prefix}:{envelope.session_id}",
            {"data": payload, "seq": envelope.seq},
        )

    async def subscribe(
        self, topic: str, from_seq: int = 0
    ) -> AsyncIterator[EventEnvelope]:
        # Consume from MQ, deserialize from JSON
        async for entry in self._backend.xread(topic, from_seq):
            envelope = EventEnvelope.from_dict(json.loads(entry["data"]))
            yield envelope
```

#### MQ-Backed CommChannel and TriggerSource

With EventTransport, CommChannel and TriggerSource gain MQ-backed implementations:

```python
class MQChannel(CommChannel):
    """CommChannel backed by MessageQueueTransport.

    Events are published to MQ as EventEnvelopes.
    Feedback (steer/followup) arrives as EventEnvelopes on a feedback topic.

    Enables protocol servers in any language:
    - Protocol server subscribes to MQ topic for events
    - Protocol server publishes feedback to MQ topic for steer/followup
    - RunLoop and protocol server can be in different processes/machines

    Sequence number ownership:
    - MQChannel gets seq from Journal.append()/upsert() internally
    - MQChannel does NOT generate its own sequence numbers
    - MQChannel does NOT receive seq from RunLoop
    - This ensures Journal is the sole source of truth for ordering
    """

    def __init__(self, transport: MessageQueueTransport, session_id: str, journal: Journal | None = None) -> None:
        self._journal = journal or MemoryJournal()
        self._transport = transport
        self._session_id = session_id
        self._feedback_queue: asyncio.Queue[Feedback] = asyncio.Queue()
        self._is_running: bool = False
        self._replaying: bool = False

    async def on_state_change(self, state: RunState) -> None:
        self._is_running = (state == RunState.RUNNING)

    async def publish(self, event) -> None:
        # Journal before delivery (same as other CommChannel implementations)
        seq = 0
        if not self._replaying:
            key = _get_upsert_key(event)
            if key:
                seq = await self._journal.upsert(key, event)
            elif not isinstance(event, StateUpdate):
                seq = await self._journal.append(event)
        # When replaying or event not journaled: seq stays 0

        envelope = EventEnvelope(
            seq=seq,  # From Journal — NOT self-generated
            session_id=self._session_id,
            event_type=type(event).__name__,
            event_data=event.to_dict(),
            ...
        )
        await self._transport.publish(envelope)

    async def _listen_feedback(self) -> None:
        """Listen on feedback topic for steer/followup from consumers."""
        async for envelope in self._transport.subscribe(f"{self._session_id}:feedback"):
            feedback = Feedback.from_envelope(envelope)
            await self._feedback_queue.put(feedback)


class MQTrigger(TriggerSource):
    """TriggerSource backed by MessageQueueTransport.

    Prompts arrive as EventEnvelopes on a prompt topic.
    Enables external systems (schedulers, gateways, other agents) to
    trigger RunLoop execution via MQ.
    """

    def __init__(self, transport: MessageQueueTransport, session_id: str) -> None:
        self._transport = transport
        self._session_id = session_id
        self._queue: asyncio.Queue[Prompt] = asyncio.Queue()

    async def subscribe(self, run_loop: RunLoop) -> None:
        asyncio.create_task(self._listen())

    async def _listen(self) -> None:
        async for envelope in self._transport.subscribe(f"{self._session_id}:prompts"):
            prompt = Prompt.from_envelope(envelope)
            await self._queue.put(prompt)


class MQEndpoint(MQChannel, MQTrigger):
    """Convenience class implementing both CommChannel and TriggerSource over MQ.

    For bidirectional MQ setups where the same transport and session
    are used for both event delivery and prompt reception. This avoids
    creating separate MQChannel + MQTrigger instances that share the
    same transport.

    Usage:
        transport = MessageQueueTransport(backend="redis_streams", url="redis://...")
        endpoint = MQEndpoint(transport, session_id="sess_123")
        run_loop = RunLoop(
            agent=my_agent,
            trigger_source=endpoint,  # MQTrigger interface
            comm_channel=endpoint,    # MQChannel interface
        )

    Design rationale: MQChannel and MQTrigger are separate classes (not merged)
    because they serve different concerns — output (CommChannel) and input
    (TriggerSource). Most execution modes use only one of them. MQEndpoint
    is a convenience for the bidirectional case, following the adapter pattern.
    The separation prevents combinatorial explosion (ScheduledCallbackChannel,
    ScheduledMQChannel, etc.).
    """

    def __init__(self, transport: MessageQueueTransport, session_id: str) -> None:
        MQChannel.__init__(self, transport, session_id)
        MQTrigger.__init__(self, transport, session_id)
```

#### Protocol Server Decoupling

With EventTransport, the architecture becomes:

```
┌──────────────────────────────────────────────────────────────────┐
│              Protocol Layer (any language)                         │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ ACP Server   │  │ OpenCode     │  │ Custom Go/Rust Server │   │
│  │ (Python)     │  │ Server (TS)  │  │                       │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │
│         │                 │                     │                │
│  ProtocolBridge    ProtocolBridge         (no bridge needed)     │
│  (v2↔v1)           (version map)                                 │
└─────────┼─────────────────┼─────────────────────┼───────────────┘
          │                 │                     │
          ▼                 ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              EventTransport (Message Queue)                       │
│  Redis Streams │ NATS JetStream │ Kafka │ in-process              │
│                                                                 │
│  EventEnvelope (JSON, schema-versioned, seq-ordered)             │
│  Topics: {session_id}:events, {session_id}:feedback,             │
│          {session_id}:prompts                                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│              RunLoop (Python)                                     │
│  CommChannel.publish() → EventTransport.publish(envelope)        │
│  TriggerSource.poll() ← EventTransport.subscribe(prompts)        │
│  CommChannel.recv() ← EventTransport.subscribe(feedback)         │
└─────────────────────────────────────────────────────────────────┘
```

**Key benefit**: Protocol servers can be implemented in any language. They only need to:
1. Subscribe to `{session_id}:events` topic for RunLoop output
2. Publish to `{session_id}:feedback` topic for steer/followup
3. Publish to `{session_id}:prompts` topic for new prompts
4. Handle EventEnvelope JSON format

This enables polyglot architecture: RunLoop in Python (for pydantic-ai integration), protocol servers in Go/Rust (for performance), gateway adapters in TypeScript (for web ecosystem).

Inspired by ACP v2's `StateUpdate`, this is a protocol-agnostic state notification:

```python
from enum import Enum
from dataclasses import dataclass

class RunState(Enum):
    RUNNING = "running"
    IDLE = "idle"
    DONE = "done"
    REQUIRES_ACTION = "requires_action"

@dataclass
class StateUpdate:
    """Protocol-agnostic run state notification.

    Published through CommChannel whenever RunLoop transitions states.
    Consumers (protocol servers, webhook callbacks, chat gateways) use
    this to know when the agent is busy, idle, or needs input.
    """
    session_id: str  # Required for journal upsert key derivation
    state: RunState
    stop_reason: str | None = None  # For idle: why it stopped
    timestamp: float = 0.0
```

### Mode Composition Matrix

Each execution mode is a specific composition of the six dimensions:

```python
# Standalone single-turn
run_loop = RunLoop(
    agent=my_agent,
    trigger_source=ImmediateTrigger("What is 2+2?"),
    journal=MemoryJournal(),
    snapshot_store=MemorySnapshotStore(),
    comm_channel=DirectChannel(),
    session_id="standalone",  # Default for one-off execution
)
await run_loop.start()

# Protocol session (ACP/OpenCode/AG-UI) — in-process
run_loop = RunLoop(
    agent=my_agent,
    trigger_source=ProtocolTrigger(session_id="sess_123"),
    journal=SQLJournal(db=pool.db),
    snapshot_store=SQLSnapshotStore(db=pool.db),
    comm_channel=ProtocolChannel(event_bus=pool.event_bus, session_id="sess_123"),
    session_id="sess_123",
)
await run_loop.start()

# Protocol session (remote, polyglot) — MQ-backed, protocol server in Go
transport = MessageQueueTransport(backend="redis_streams", url="redis://...")
run_loop = RunLoop(
    agent=my_agent,
    trigger_source=MQTrigger(transport, session_id="sess_123"),
    journal=DurableJournal(db=pool.db),
    snapshot_store=DurableSnapshotStore(db=pool.db),
    comm_channel=BridgedCommChannel(
        MQChannel(transport, session_id="sess_123"),
        ACPv2toV1Bridge(),  # v2 RunLoop ↔ v1 client
    ),
    session_id="sess_123",
)
await run_loop.start()

# Long-running task (scheduled, crash-recoverable)
run_loop = RunLoop(
    agent=my_agent,
    trigger_source=ScheduledTrigger(
        schedule="0 9 * * 1-5",  # Weekdays 9am
        prompt_template="Generate daily report for {{ date }}",
    ),
    journal=DurableJournal(db=pool.db),       # Event-layer
    snapshot_store=DurableSnapshotStore(db=pool.db),  # Loop-layer
    comm_channel=CallbackChannel(
        callback=store_result,
        webhook_url="https://hooks.example.com/agent-status",
    ),
    session_id="daily_report",
)
await run_loop.start()

# Channel wake-up (chat bot, MQ-backed, crash-recoverable)
transport = MessageQueueTransport(backend="nats_jetstream", url="nats://...")
gateway = TelegramGateway(token="...")
channel = GatewayChannel(gateway=gateway)
run_loop = RunLoop(
    agent=my_agent,
    trigger_source=channel,  # GatewayChannel implements TriggerSource
    journal=DurableJournal(db=pool.db),       # Survives restart
    snapshot_store=DurableSnapshotStore(db=pool.db),
    comm_channel=channel,    # GatewayChannel implements CommChannel
    session_id="telegram_bot",
)
await run_loop.start()
```

### YAML Configuration

The six dimensions map naturally to YAML configuration:

```yaml
agents:
  standalone_agent:
    type: native
    model: "openai:gpt-4o"
    # Default: ImmediateTrigger + MemoryJournal + MemorySnapshotStore + DirectChannel + InProcessTransport

  session_agent:
    type: native
    model: "openai:gpt-4o"
    lifecycle:
      trigger: protocol
      journal: sql          # Event-layer persistence
      snapshot: sql         # Loop-layer persistence
      comm: protocol
      transport: in_process

  # Remote protocol session with ACP v1 client bridging
  remote_v1_agent:
    type: native
    model: "openai:gpt-4o"
    lifecycle:
      trigger:
        type: mq
        transport: message_queue
      journal: durable       # Event-layer: crash-safe event log
      snapshot: durable      # Loop-layer: crash-safe state snapshots
      comm:
        type: mq
        bridge:
          from: acp_v2
          to: acp_v1
      transport:
        type: message_queue
        backend: redis_streams
        url: "redis://localhost:6379"

  daily_reporter:
    type: native
    model: "openai:gpt-4o"
    system_prompt: "You are a report generator."
    lifecycle:
      trigger:
        type: scheduled
        schedule: "0 9 * * 1-5"
        prompt_template: "Generate daily report for {{ date }}"
      journal: durable  # Event-layer: crash-safe event log
      snapshot: durable  # Loop-layer: crash-safe state snapshots
      comm:
        type: callback
        webhook_url: "https://hooks.example.com/agent-status"
      transport: in_process

  slack_bot:
    type: native
    model: "openai:gpt-4o"
    system_prompt: "You are a helpful Slack bot."
    lifecycle:
      trigger:
        type: channel
        gateway:
          type: slack
          bot_token: "${SLACK_BOT_TOKEN}"
      journal: durable  # Survives restart, replays missed messages
      snapshot: durable
      comm:
        type: gateway  # Same gateway as trigger (bidirectional)
      transport:
        type: message_queue       # Events also published to MQ
        backend: nats_jetstream
        url: "nats://localhost:4222"
```

### Feedback Loop Architecture

The CommChannel feedback loop is what enables channel wake-up steer/followup:

```
User sends "stop" to Slack
  → SlackGateway receives message
  → GatewayChannel._listen() checks RunLoop status
  → If running: calls RunLoop.steer("stop") → injected into active Turn
  → If idle: queues as new Prompt → RunLoop wakes and processes

User sends "also check the API docs" to Slack
  → GatewayChannel._listen() checks RunLoop status
  → If running: calls RunLoop.followup("also check the API docs") → queued
  → If idle: queues as new Prompt → RunLoop wakes and processes
```

This is the same steer/followup mechanism from RFC-0041, but the **source** of the message is an external channel rather than a protocol handler. The RunLoop doesn't know or care where the steer came from.

### Protocol Version Bridging

AgentPool supports multiple protocol versions (ACP v1, ACP v2) and multiple protocols (ACP, OpenCode, AG-UI). Protocol version bridging enables cross-version communication without modifying the RunLoop.

Inspired by ACP's `conversion.rs` which translates between v1 and v2 schemas, this RFC proposes a general `ProtocolBridge` abstraction.

#### ProtocolBridge Interface

```python
@runtime_checkable
class ProtocolBridge(Protocol):
    """Translates between protocol versions at the boundary.

    A ProtocolBridge is a CommChannel decorator that:
    1. Intercepts events from RunLoop (internal format)
    2. Translates them to the target protocol version
    3. Delivers via the underlying CommChannel

    And in reverse:
    1. Receives feedback from the underlying CommChannel (target protocol version)
    2. Translates to internal format
    3. Delivers to RunLoop

    This enables patterns like:
    - ACP v2 RunLoop ↔ ACP v1 client (downgrade)
    - ACP v1 RunLoop ↔ ACP v2 client (upgrade)
    - Future protocol versions without RunLoop changes
    """

    def translate_event(
        self, event: RichAgentStreamEvent | StateUpdate
    ) -> RichAgentStreamEvent | StateUpdate | None:
        """Translate an outbound event to the target protocol version.

        Returns None to filter out events that have no equivalent
        in the target protocol version.
        """
        ...

    def translate_feedback(self, feedback: Feedback) -> Feedback:
        """Translate inbound feedback to the internal format."""
        ...

    def translate_prompt(self, prompt: Prompt) -> Prompt:
        """Translate an inbound prompt to the internal format."""
        ...


class BridgedCommChannel(CommChannel):
    """CommChannel decorator that applies ProtocolBridge translation.

    Wraps an underlying CommChannel (e.g., MQChannel or ProtocolChannel)
    and translates events/feedback between protocol versions.
    """

    def __init__(
        self,
        underlying: CommChannel,
        bridge: ProtocolBridge,
    ) -> None:
        self._underlying = underlying
        self._bridge = bridge
        self._replaying: bool = False  # Set by RunLoop during crash recovery

    async def publish(self, event) -> None:
        self._underlying._replaying = self._replaying  # propagate to underlying channel
        translated = self._bridge.translate_event(event)
        if translated is not None:
            await self._underlying.publish(translated)

    async def recv(self) -> Feedback | None:
        feedback = await self._underlying.recv()
        if feedback is None:
            return None
        return self._bridge.translate_feedback(feedback)
```

#### ACP v2 ↔ v1 Bridge

The ACP v2↔v1 bridge handles the specific differences identified in the cross-framework research:

| Concern | ACP v1 | ACP v2 | Bridge Strategy |
|---------|--------|--------|----------------|
| State notification | Implicit (stream end = done) | Explicit `StateUpdate` (Running/Idle/RequiresAction) | v2→v1: Drop StateUpdate, infer from stream events. v1→v2: Synthesize StateUpdate from stream start/end. |
| Message model | Whole-message replacement | Chunks + whole-message replacement | v2→v1: Accumulate chunks, emit as whole message. v1→v2: Split whole message into chunk + replacement. |
| Tool call updates | Separate start/progress/complete | Unified `ToolCallUpdate` (upsert) | v2→v1: Map upsert to start/progress/complete. v1→v2: Combine start/progress/complete into upsert. |
| Client I/O | Client handles fs/terminal | Removed (server-side only) | v2→v1: Server handles I/O, no client delegation. v1→v2: Intercept client I/O requests, handle server-side. |
| Diff changes | Unstructured | Structured `DiffChange` | v2→v1: Serialize structured diff to text. v1→v2: Parse text diff into structured format (best-effort). |
| Other variants | N/A | Forward-compatible `Other` variants | v2→v1: Drop unknown variants with warning. v1→v2: No-op (v1 has no unknown variants). |

```python
class ACPv2toV1Bridge(ProtocolBridge):
    """Bridges ACP v2 RunLoop output to ACP v1 client expectations.

    Used when: RunLoop speaks v2, client speaks v1.
    Pattern: ACP server v2 ↔ ACP client v1
    """

    def translate_event(self, event):
        match event:
            case StateUpdate(state=RunState.RUNNING):
                # v1 has no explicit running state; drop
                return None  # Filtered out
            case StateUpdate(state=RunState.IDLE):
                # v1: synthesize stream end
                return StreamCompleteEvent(...)
            case ToolCallUpdateEvent(action="upsert", ...):
                # v1: map to start or progress based on state
                if event.is_first_update:
                    return ToolCallStartEvent(...)
                return ToolCallProgressEvent(...)
            case DiffChangeEvent(structured=...):
                # v1: serialize to text
                return DiffChangeEvent(text=event.structured.to_unified_diff())
            case _:
                # Forward-compatible: drop unknown v2 variants
                if event.is_v2_only:
                    logger.warning(f"Dropping v2-only event: {event.event_type}")
                    return None
                return event  # Pass through compatible events


class ACPv1toV2Bridge(ProtocolBridge):
    """Bridges ACP v1 client output to ACP v2 RunLoop expectations.

    Used when: RunLoop speaks v2, client speaks v1 (inbound direction).
    Pattern: ACP server v2 ↔ ACP client v1
    """

    def translate_prompt(self, prompt):
        # v1 prompts are compatible with v2 (no translation needed)
        return prompt

    def translate_feedback(self, feedback):
        # v1 has no explicit steer/queue mode; infer from content
        # Default to "steer" for mid-turn, "queue" for post-turn
        if feedback.received_during_turn:
            feedback.mode = "steer"
        else:
            feedback.mode = "queue"
        return feedback
```

#### Bridging Configuration

Protocol bridging is configured at the CommChannel level:

```yaml
agents:
  acp_v1_agent:
    type: native
    model: "openai:gpt-4o"
    lifecycle:
      trigger: protocol
      journal: sql
      snapshot: sql
      comm:
        type: protocol
        bridge:
          from: acp_v2   # RunLoop speaks v2 internally
          to: acp_v1     # Client speaks v1
          # ACPv2toV1Bridge is auto-selected based on from/to
```

For MQ-based setups, the bridge sits between RunLoop and EventTransport:

```python
# ACP v2 RunLoop ↔ ACP v1 client via MQ
transport = MessageQueueTransport(backend="redis_streams", url="redis://...")
mq_channel = MQChannel(transport, session_id="sess_123")
bridge = ACPv2toV1Bridge()
bridged_channel = BridgedCommChannel(mq_channel, bridge)

run_loop = RunLoop(
    agent=my_agent,
    trigger_source=MQTrigger(transport, session_id="sess_123"),
    journal=DurableJournal(db=pool.db),
    snapshot_store=DurableSnapshotStore(db=pool.db),
    comm_channel=bridged_channel,  # Bridge translates before MQ delivery
)
```

The protocol server (in any language) consumes v1-format events from MQ and produces v1-format feedback. The bridge translates transparently.

Existing APIs continue to work through default dimension selection:

| Existing API | Default Dimensions | New Equivalent |
|-------------|-------------------|----------------|
| `agent.run("prompt")` | ImmediateTrigger + MemoryJournal + MemorySnapshotStore + DirectChannel | `RunLoop(agent, ImmediateTrigger("prompt")).start()` |
| `agent.run_stream("prompt")` | Same, DirectChannel.events() iterator | Same, iterate DirectChannel |
| `SessionController.receive_request()` | ProtocolTrigger + SQLJournal + SQLSnapshotStore + ProtocolChannel | `RunLoop(agent, ProtocolTrigger(...), ...).start()` |
| Background workers | ScheduledTrigger + DurableJournal + DurableSnapshotStore + CallbackChannel | `RunLoop(agent, ScheduledTrigger(...), ...).start()` |

The existing APIs become thin wrappers over RunLoop with default dimensions.

---

## Security Considerations

### Gateway Authentication

Channel wake-up mode introduces external input sources (chat gateways, webhooks). These must be authenticated:

- **GatewayChannel**: Delegates authentication to the `GatewayAdapter` implementation (e.g., Slack verifies webhook signatures, Telegram validates bot tokens)
- **CallbackChannel webhook**: Must validate incoming webhook signatures if feedback is supported
- **TriggerSource.poll()**: Must not trust prompt metadata blindly; metadata is advisory, not authoritative

### State Store Data Isolation

- `DurableJournal`, `DurableSnapshotStore`, `SQLJournal`, and `SQLSnapshotStore` must enforce per-session data isolation
- State snapshots must not leak between RunLoop instances
- `EventLogJournal` + `EventLogSnapshotStore` replay must be scoped to a single session
- Journal entries must include session_id for partition isolation
- **Multi-tenant isolation**: EventEnvelope MUST include `tenant_id` field; Journal and SnapshotStore queries MUST filter by `tenant_id`; MQ topics MUST be prefixed with `tenant_id` (e.g., `{tenant_id}:{session_id}:events`); cross-tenant data access must be prevented at the query level, not just at the application level

### CommChannel Output Sanitization

- `GatewayChannel.publish()` must sanitize events before sending to external channels (e.g., no internal tool paths, no API keys in responses)
- `CallbackChannel` webhook delivery must use HTTPS and validate certificates

### EventTransport Security

- `MessageQueueTransport` must support TLS for all MQ connections
- EventEnvelope may contain sensitive data — MQ access must be authenticated
- Consumer offset management must be per-consumer (one consumer can't advance another's offset)
- Replay capability has privacy implications — `Journal.clear() + SnapshotStore.clear()` must be enforceable (GDPR right to erasure)
- EventEnvelope `metadata` field must not contain credentials (use references to secret stores)
- **PII encryption at rest**: Journal entries and snapshots MAY contain PII (user messages, tool results with personal data). Journal and SnapshotStore implementations MUST support optional encryption at rest (e.g., AES-256-GCM with keys from a KMS). `MemoryJournal` and `MemorySnapshotStore` are exempt (in-process only). `DurableJournal`, `DurableSnapshotStore`, `EventLogJournal`, and `EventLogSnapshotStore` MUST encrypt by default when PII is detected.

### ProtocolBridge Security

- Version downgrade (v2→v1) may lose security-relevant metadata (e.g., v2's structured DiffChange could hide malicious content when serialized to text)
- Bridge implementations must validate translated content (e.g., text diffs must be parsed safely)
- Unknown v2 variants dropped during v1 bridging must be logged for audit

### Durable Execution Security

- Snapshot images contain full message history — must be encrypted at rest
- Journal entries may contain tool call results with sensitive data — must be encrypted at rest
- Idempotency keys (`turn_id`) must not be predictable (use UUID v4 or similar)
- Replay must not re-execute side effects (tool calls) — only reconstruct state
- Tool execution log entries may contain sensitive tool inputs/outputs — must be encrypted at rest

### Resource Exhaustion and Backpressure

- `ChannelTrigger._queue` must be bounded to prevent memory exhaustion from unprocessed messages
- `ScheduledTrigger` must support max consecutive runs to prevent runaway scheduling
- `DurableJournal` and `DurableSnapshotStore` must implement retention policies to prevent unbounded state growth
- `MessageQueueTransport` must enforce per-session topic limits
- Journal compaction must run automatically when size exceeds threshold
- **Backpressure**: All CommChannel implementations MUST support configurable `max_queue_size` (default: 1000) and `drop_policy` (default: `backpressure`, alternatives: `drop_oldest`, `drop_newest`). When `backpressure` is selected, `publish()` blocks until the consumer drains entries. When `drop_*` is selected, events are dropped with a logged warning. This replaces the current EventBus behavior of silently dropping events on overflow.

### Observability

- **Trace IDs**: EventEnvelope MUST include `trace_id` in metadata for distributed tracing. RunLoop generates a `trace_id` per Run (not per Turn). CommChannel implementations SHOULD propagate `trace_id` to external systems (e.g., HTTP headers for webhooks, MQ message attributes). This enables end-to-end tracing across protocol servers, MQ transport, and external consumers.
- **Structured logging**: All dimension implementations SHOULD emit structured log events with `session_id`, `turn_id`, `trace_id`, and `seq` for correlation.
- **Metrics**: RunLoop SHOULD expose metrics: turns_completed, turns_failed, journal_entries, snapshot_count, recovery_count, consumer_lag.

### Journal Schema Migration

- Journal entries have `schema_version` field (in EventEnvelope). When the schema evolves:
  1. **Forward-compatible**: New fields are optional; old consumers ignore unknown fields (JSON naturally supports this)
  2. **Breaking changes**: Journal and SnapshotStore implementations MUST support a `migrate_journal(from_version, to_version)` method. Migration runs on startup before any replay.
  3. **Versioned snapshots**: Snapshot images include their schema version. On load, if snapshot version < current, migration is applied.
  4. **Rollback safety**: Migration MUST be reversible or journaled. If migration fails, the original journal is preserved.

```yaml
lifecycle:
  journal: durable
  snapshot: durable
  journal_config:
    schema_version: "1.0"
    auto_migrate: true  # Automatically migrate on startup
    migrate_timeout: 30s
```

---

## Testing Strategy

Each dimension has well-defined interfaces, enabling mock fixtures per dimension. This section defines the testing strategy for the unified lifecycle architecture.

### Dimension-Level Mock Fixtures

| Dimension | Mock Implementation | Test Usage |
|-----------|--------------------|------------|
| `TriggerSource` | `MockTrigger` — programmable prompt injection | Inject prompts at specific times; test idle/wake transitions |
| `Journal` | `MemoryJournal` (already in-memory) | Fast, no SQL; use for unit tests; verify journal entries via `replay()` |
| `SnapshotStore` | `MemorySnapshotStore` (already in-memory) | Fast, no SQL; use for unit tests; verify snapshots via `load()` |
| `CommChannel` | `MockCommChannel` — captures all published events | Assert event sequences; simulate feedback (steer/followup) |
| `EventTransport` | `InProcessTransport` (already in-process) | No MQ infrastructure needed; verify EventEnvelope serialization |
| `ProtocolBridge` | `IdentityBridge` — no-op translation | Test without bridging; swap in real bridge for version-specific tests |

### Test Categories

1. **Unit tests** (`@pytest.mark.unit`): Each dimension interface tested in isolation with mock dependencies. Fast, no I/O.

2. **Integration tests** (`@pytest.mark.integration`): RunLoop with real dimension implementations (MemoryJournal, MemorySnapshotStore, DirectChannel, InProcessTransport). Verify end-to-end Turn execution, event delivery, and state transitions.

3. **Crash recovery tests** (`@pytest.mark.slow`): DurableJournal + DurableSnapshotStore with SQL backend. Simulate crash (kill process mid-Turn), restart, verify recovery. Test all three recovery strategies (`mark_interrupted`, `replay`, `re_invoke`).

4. **Protocol bridging tests** (`@pytest.mark.integration`): ACPv2toV1Bridge and ACPv1toV2Bridge. Verify event translation correctness for all event types. Test edge cases (unknown variants, structured diff parsing).

5. **MQ transport tests** (`@pytest.mark.slow`): MessageQueueTransport with Redis Streams (or mock MQ backend). Verify EventEnvelope serialization, consumer offset management, replay from seq, and multi-consumer fan-out.

### Test Fixtures

```python
@pytest.fixture
def mock_trigger():
    """Programmable trigger for testing."""
    return MockTrigger()

@pytest.fixture
def memory_journal():
    """In-memory Journal for testing."""
    return MemoryJournal()

@pytest.fixture
def memory_snapshot_store():
    """In-memory SnapshotStore for testing."""
    return MemorySnapshotStore()

@pytest.fixture
def mock_comm():
    """Mock CommChannel that captures all events."""
    return MockCommChannel()

@pytest.fixture
def run_loop(mock_trigger, memory_journal, memory_snapshot_store, mock_comm):
    """RunLoop with all-mock dimensions for unit testing."""
    return RunLoop(
        agent=mock_agent,
        trigger_source=mock_trigger,
        journal=memory_journal,
        snapshot_store=memory_snapshot_store,
        comm_channel=mock_comm,
    )

@pytest.fixture
def durable_run_loop(memory_journal, memory_snapshot_store):
    """RunLoop with DurableJournal + DurableSnapshotStore for crash recovery testing."""
    return RunLoop(
        agent=mock_agent,
        trigger_source=MockTrigger(),
        journal=DurableJournal(db=test_db),
        snapshot_store=DurableSnapshotStore(db=test_db),
        comm_channel=MockCommChannel(),
    )
```

### Crash Recovery Test Pattern

```python
@pytest.mark.slow
async def test_crash_recovery_mid_turn(durable_run_loop):
    """Test that a crash mid-Turn is recoverable."""
    # 1. Start RunLoop, begin a Turn
    await durable_run_loop.start("Process this data")
    
    # 2. Simulate crash: kill the RunLoop without graceful shutdown
    await durable_run_loop._force_crash()  # Internal test method
    
    # 3. Verify journal has entries but no turn_result
    state, last_seq = await durable_run_loop._snapshots.load()
    assert state is not None
    assert last_seq > state.snapshot_seq  # Turn was in-flight
    
    # 4. Create new RunLoop with same Journal + SnapshotStore (simulates restart)
    recovered_loop = RunLoop(
        agent=mock_agent,
        trigger_source=MockTrigger(),
        journal=durable_run_loop._journal,  # Same journal
        snapshot_store=durable_run_loop._snapshots,  # Same snapshot store
        comm_channel=MockCommChannel(),
    )
    
    # 5. Start recovery
    await recovered_loop.start()
    
    # 6. Verify consumer saw all events up to crash point
    events = recovered_loop._comm.captured_events
    assert len(events) > 0  # Events were replayed
    
    # 7. Verify Turn was marked interrupted (default strategy)
    state_updates = [e for e in events if isinstance(e, StateUpdate)]
    assert any(e.state == RunState.IDLE and e.stop_reason == "crash_recovery" 
               for e in state_updates)
```

This RFC defines the architecture. Implementation is phased, with each phase being independently shippable. Detailed implementation for each phase is deferred to sub-RFCs.

### Phase 0: Prerequisite — RFC-0041 (Run/Turn Separation)

**Status**: Draft (RFC-0041)
**Dependency**: None
**Sub-RFC**: RFC-0041

Restructure `RunHandle` into persistent idle/running/done state machine with unified steer/followup. This is the foundation — the RunLoop in this RFC IS RFC-0041's restructured RunHandle.

**Deliverables**:
- RunHandle with idle/running/done states
- Unified steer/followup (no native/non-native branching)
- Thin NativeTurn (~80 lines) and ACPTurn wrappers
- Elimination of ~415 lines compensating complexity
- Standalone execution without SessionPool

**Exit criteria**:
- `agent.run()` works without SessionPool
- Steer/followup works on standalone runs
- Orchestrator layer reduced from ~2500 to ~1000 lines

### Phase 1: Dimension Interfaces + Defaults + EventEnvelope

**Status**: This RFC
**Dependency**: Phase 0
**Sub-RFC**: TBD

Define the six dimension interfaces (`TriggerSource`, `Journal`, `SnapshotStore`, `CommChannel`, `EventTransport`, `RunLoop` injection points) as Python `Protocol` classes and implement default implementations. Define `EventEnvelope` serialization format.

**Deliverables**:
- `Protocol` classes for all six dimensions (structural subtyping, not ABC inheritance)
- `ImmediateTrigger`, `MemoryJournal`, `MemorySnapshotStore`, `DirectChannel`, `InProcessTransport` implementations
- `EventEnvelope` dataclass with JSON serialization + schema versioning
- `StateUpdate` event type defined and published on state transitions
- `RunLoop.__init__()` accepts dimension injections (journal + snapshot_store + comm_channel + trigger_source)
- CommChannel owns Journal reference, handles `append()` vs `upsert()` routing internally
- `_get_upsert_key(event)` helper for event-type-based upsert key derivation
- Journal interface: `append()`, `upsert(key, event)`, `replay()`, `resume(snapshot_store)`, `compact()`, `log_tool_execution()`, `get_tool_executions()`
- SnapshotStore interface: `save()`, `load()`, `save_turn_result()`, `has_turn_result()`, `clear()`
- `ResumeResult` dataclass

**Exit criteria**:
- All existing tests pass with default dimensions
- `agent.run()` uses `RunLoop(agent, ImmediateTrigger(...), MemoryJournal(), MemorySnapshotStore(), DirectChannel(journal=...), InProcessTransport())` internally
- StateUpdate events published on idle/running/done transitions
- EventEnvelope can serialize/deserialize all event types
- Journal `append()` + `upsert()` implemented (MemoryJournal backing)
- `journal.resume(snapshot_store)` returns None for fresh state

### Phase 2: Durable Execution + Persistence/Replay

**Status**: Future sub-RFC
**Dependency**: Phase 1
**Sub-RFC**: TBD

Implement `DurableJournal` + `DurableSnapshotStore` with journal + snapshot + crash recovery. Implement replay. This is the foundation for long-running tasks and channel wake-up.

**This phase is split into two sub-phases** to manage complexity and enable earlier delivery of basic crash recovery:

#### Phase 2a: Journal + Snapshot + Basic Recovery

**Deliverables**:
- `DurableJournal` with SQL-backed journal table (supports both append and upsert entries)
- `DurableSnapshotStore` with SQL-backed snapshot table
- Snapshot at Turn boundaries (after Turn completion, before next prompt)
- Basic crash recovery: `journal.resume(snapshot_store)` → ResumeResult
- Turn idempotency: `turn_id` as deduplication key, `save_turn_result()` / `has_turn_result()`
- Tool execution log: `log_tool_execution()` / `get_tool_executions()` for idempotent tool recovery

**Exit criteria**:
- Process crash mid-Turn → restart → `journal.resume()` detects in-flight Turn
- Completed Turns not re-executed (turn_id idempotency)
- Tool calls not re-executed on recovery (tool execution log)
- Basic recovery strategy: `mark_interrupted` (default)

#### Phase 2b: Replay + Compaction + Divergence Detection

**Deliverables**:
- `journal.replay(from_seq, to_seq)` for debugging/audit/crash recovery (upsert keys return only latest)
- LLM replay strategy: `journal.resume()` replays journal events to consumer for in-flight Turns
- Recovery strategy options: `mark_interrupted` (default), `replay`, `re_invoke`
- `journal.compact(before_seq)` for journal maintenance
- Hybrid compaction (time + size based)
- `EventLogJournal` + `EventLogSnapshotStore` (append-only event log variant) for maximum durability
- Divergence detection during replay (compare replayed state with expected state)

**Exit criteria**:
- In-flight Turn crash: journal events replayed to consumer up to crash point
- Full session state reconstructable from journal + snapshots
- Replay produces identical state to live execution
- Journal compaction prevents unbounded growth
- All three recovery strategies work: `mark_interrupted`, `replay`, `re_invoke`

**Phase 2 overall exit criteria** (2a + 2b combined):
- Process crash mid-Turn → restart → resume from snapshot, no lost work
- Tool calls are idempotent via tool execution log
- LLM is not re-invoked on crash recovery (events replayed instead)

### Phase 3: Protocol Dimensions + Version Bridging

**Status**: Future sub-RFC
**Dependency**: Phase 1
**Sub-RFC**: TBD

Implement `ProtocolTrigger`, `SQLJournal` + `SQLSnapshotStore`, `ProtocolChannel` to replace the current SessionController/EventBus/ProtocolEventConsumerMixin pattern. Implement `ProtocolBridge` for cross-version translation.

**Deliverables**:
- `ProtocolTrigger` wraps `SessionController.receive_request()`
- `SQLJournal` wraps existing SQL-backed journal storage (extended with append/upsert)
- `SQLSnapshotStore` wraps existing SQL-backed snapshot storage
- `ProtocolChannel` wraps EventBus + ProtocolEventConsumerMixin
- `ProtocolBridge` abstract base class
- `ACPv2toV1Bridge` implementation (v2 RunLoop ↔ v1 client)
- `ACPv1toV2Bridge` implementation (v1 RunLoop ↔ v2 client)
- `BridgedCommChannel` decorator
- YAML `lifecycle.comm.bridge` section
- Migration path: existing session agents use ProtocolTrigger + SQLJournal + SQLSnapshotStore + ProtocolChannel

**Exit criteria**:
- ACP/OpenCode/AG-UI/OpenAI API servers use RunLoop with Protocol dimensions
- ProtocolEventConsumerMixin refactored to consume from ProtocolChannel
- ACP v2↔v1 bridging works: v2 RunLoop serves v1 clients transparently
- No behavioral changes in existing protocol servers

### Phase 4: EventTransport + MQ Backends

**Status**: Future sub-RFC
**Dependency**: Phase 1
**Sub-RFC**: TBD

Implement `MessageQueueTransport` with multiple MQ backends. Implement `MQChannel`, `MQTrigger`, and `MQEndpoint`. Enable language-agnostic protocol servers.

**Note**: Phase 4 depends on Phase 1 only, NOT Phase 2. MQ backends (Redis Streams, NATS JetStream, Kafka) are already durable — they provide their own persistence, replay, and crash recovery. The Journal is for RunLoop internal state; MQ durability is for event transport. These are separate concerns.

**Deliverables**:
- `MessageQueueTransport` with pluggable backend interface
- `RedisStreamsBackend` (reference implementation)
- `NATSJetStreamBackend` (reference implementation)
- `KafkaBackend` (community contribution)
- `MQChannel` (CommChannel over MQ)
- `MQTrigger` (TriggerSource over MQ)
- EventEnvelope JSON serialization for all event types
- Consumer offset management (`ack()`, replay from seq)
- YAML `lifecycle.transport` section
- Reference protocol server in Go or Rust ( consuming EventEnvelopes from MQ)

**Exit criteria**:
- Protocol server in non-Python language can consume events from MQ
- Feedback (steer/followup) flows from MQ consumer back to RunLoop
- Events survive MQ broker restart (durable streams)
- Replay from any past sequence number works via MQ consumer offset

### Phase 5: Long-Running Task Dimensions

**Status**: Future sub-RFC
**Dependency**: Phase 1, Phase 2
**Sub-RFC**: TBD

Implement `ScheduledTrigger`, `CallbackChannel` for long-running task support. Built on DurableJournal + DurableSnapshotStore from Phase 2.

**Deliverables**:
- `ScheduledTrigger` with cron expression support (via `croniter` or similar)
- `CallbackChannel` with async callback + optional webhook delivery
- YAML `lifecycle.trigger.type: scheduled` support
- `WatchCommand` refactored to use ScheduledTrigger
- Crash recovery: scheduled task survives process restart

**Exit criteria**:
- Scheduled agents run on cron schedules
- Agent state survives process restart (DurableJournal + DurableSnapshotStore recovery)
- Webhook notifications sent on StateUpdate
- Missed schedules handled gracefully (catch-up or skip, configurable)

### Phase 6: Channel Wake-Up Dimensions

**Status**: Future sub-RFC
**Dependency**: Phase 1, Phase 2
**Sub-RFC**: TBD

Implement `ChannelTrigger`, `GatewayChannel`, `BidirectionalChannel` base, and `GatewayAdapter` interface for chat gateway integration.

**Note**: Phase 6 depends on Phase 1 and Phase 2 only, NOT Phase 5. Channel wake-up needs durable state (DurableJournal + DurableSnapshotStore from Phase 2) to survive restarts, but does NOT need the scheduled task infrastructure from Phase 5. Gateway triggers are event-driven (external messages), not schedule-driven (cron).

**Deliverables**:
- `GatewayAdapter` abstract base class (transport-agnostic)
- `GatewayChannel` bidirectional implementation (TriggerSource + CommChannel)
- `TelegramGatewayAdapter` (reference implementation)
- `SlackGatewayAdapter` (reference implementation)
- YAML `lifecycle.trigger.type: channel` support
- Feedback loop: gateway messages → steer/followup → RunLoop
- Crash recovery: missed channel messages replayed on restart (via DurableJournal + DurableSnapshotStore + MQ)

**Exit criteria**:
- Agent wakes on incoming Telegram/Slack message
- Steer works: user correction mid-task is injected into active Turn
- Followup works: user message while idle triggers new Turn
- Agent goes dormant after responding (idle state, no CPU usage)
- Agent survives restart: DurableJournal + DurableSnapshotStore restores conversation context

### Phase Summary

| Phase | Name | Dependency | Effort | Shippable? |
|-------|------|-----------|--------|------------|
| 0 | Run/Turn Separation (RFC-0041) | None | Large | ✅ Yes |
| 1 | Dimension Interfaces + Defaults + EventEnvelope | Phase 0 | Medium | ✅ Yes |
| 2a | Journal + Snapshot + Basic Recovery | Phase 1 | Medium | ✅ Yes |
| 2b | Replay + Compaction + Divergence Detection | Phase 2a | Medium | ✅ Yes |
| 3 | Protocol Dimensions + Version Bridging | Phase 1 | Medium | ✅ Yes |
| 4 | EventTransport + MQ Backends | Phase 1 | Large | ✅ Yes |
| 5 | Long-Running Task Dimensions | Phase 1, 2 | Medium | ✅ Yes |
| 6 | Channel Wake-Up Dimensions | Phase 1, 2 | Large | ✅ Yes |

**Parallelization**:
- Phases 2a, 3, 4 can all proceed in parallel after Phase 1
- Phase 2b depends on Phase 2a (replay builds on journal+snapshot)
- Phase 5 depends on Phase 2 (long-running needs DurableJournal + DurableSnapshotStore)
- Phase 6 depends on Phase 2 only, NOT Phase 5 (channel wake-up is event-driven, not schedule-driven)
- Phase 4 does NOT depend on Phase 2 (MQ backends provide their own durability)

**Key changes from revision 2**:
- Persistence/replay (Phase 2) split into 2a (basic recovery) and 2b (replay+compaction) for earlier delivery
- Phase 4 dependency reduced: Phase 1 only (MQ backends are already durable, don't need Journal/SnapshotStore)
- Phase 6 dependency reduced: Phase 1, 2 only (channel wake-up doesn't need scheduled task infrastructure)
- Three phases (2a, 3, 4) can now run in parallel after Phase 1, up from two (2, 3)

---

## Open Questions

### Q1: Should RunLoop support concurrent Turns?

**Context**: The current design assumes serial Turn execution (one Turn at a time per RunLoop). But some use cases (e.g., parallel analysis of multiple inputs) might benefit from concurrent Turns.

**Options**:
1. **Serial only** (current design) — simpler, matches ACP v2's session model
2. **Concurrent Turns** — add `max_concurrent_turns` parameter; use anyio CapacityLimiter
3. **Multiple RunLoops** — user creates N RunLoops, each serial, for parallelism

**Recommendation**: Serial only (Option 1). Parallelism is achieved through multiple RunLoops or team orchestration. Keeps the model simple.

### Q2: How does StateUpdate interact with ACP v2's state_change?

**Context**: ACP v2 has its own `StateUpdate` event (`Running/Idle/RequiresAction`). AgentPool's `StateUpdate` is protocol-agnostic.

**Options**:
1. **Direct mapping** — ACP v2's `state_change` IS AgentPool's `StateUpdate`, just with protocol-specific serialization
2. **Translation layer** — ProtocolChannel translates between internal StateUpdate and ACP v2's state_change

**Recommendation**: Direct mapping (Option 1). The semantics are identical. ProtocolChannel handles serialization, not translation.

### Q3: Should CommChannel support backpressure?

**Context**: If a consumer is slow (e.g., a webhook that takes 5 seconds to respond), should the RunLoop block?

**Options**:
1. **Unbounded queue** — events are never dropped, but memory grows
2. **Bounded queue with backpressure** — RunLoop blocks if consumer is slow
3. **Bounded queue with drop** — events dropped if consumer is slow (current EventBus behavior)

**Recommendation**: Bounded queue with configurable policy (Option 2 with escape valve). Default to backpressure; allow drop policy for fire-and-forget channels.

### Q4: Should TriggerSource support multiple RunLoops?

**Context**: A single gateway (e.g., one Slack workspace) might route messages to multiple agents.

**Options**:
1. **One TriggerSource per RunLoop** — simple, but requires a router/dispatcher in front
2. **TriggerSource can fan-out** — one TriggerSource, multiple RunLoops, each gets filtered prompts
3. **Router pattern** — separate `TriggerRouter` that multiplexes

**Recommendation**: Router pattern (Option 3). TriggerSource is 1:1 with RunLoop. A `TriggerRouter` can sit in front, routing messages to appropriate RunLoops based on metadata (channel name, user ID, etc.).

### Q5: How does this interact with the existing graph/team architecture?

**Context**: Teams are compiled into pydantic-graph workflows with Fork/Join. How does RunLoop relate to graph execution?

**Options**:
1. **RunLoop wraps graph execution** — Team's graph runs inside a single Turn
2. **Each graph step is a Turn** — RunLoop drives graph steps as individual Turns
3. **Orthogonal** — Graph execution is within Turn; RunLoop is above graph

**Recommendation**: Orthogonal (Option 3). A Turn may internally execute a graph (for teams). RunLoop doesn't know about graphs. This matches RFC-0041's design where Turn execution is delegated to agent-type-specific implementations.

### Q6: Should GatewayChannel support message threading?

**Context**: Chat platforms (Slack, Discord) support threaded conversations. Should each thread be a separate RunLoop?

**Recommendation**: Yes. Each thread = one RunLoop. The `GatewayAdapter` maps threads to RunLoop instances. This gives per-thread conversation isolation and independent state.

### Q7: What happens when journal grows unbounded for long-running channel agents?

**Context**: A channel wake-up agent may run for months. Journal entries accumulate. When should compaction occur?

**Options**:
1. **Time-based compaction** — compact entries older than N days
2. **Size-based compaction** — compact when journal exceeds N MB
3. **Snapshot-based compaction** — compact after every N snapshots
4. **Hybrid** — time + size, whichever triggers first

**Recommendation**: Hybrid (Option 4). Default: compact after 7 days OR 100MB, whichever triggers first. Configurable per agent.

### Q8: Should EventEnvelope support schema evolution (forward/backward compatibility)?

**Context**: EventEnvelope has a `schema_version` field. But what happens when a consumer receives an envelope with a newer schema version than it supports?

**Options**:
1. **Strict** — reject envelopes with unknown schema versions
2. **Forward-compatible** — consume what you can, ignore unknown fields (protobuf-style)
3. **Versioned adapters** — ProtocolBridge-style translation between schema versions

**Recommendation**: Forward-compatible (Option 2) with versioned adapters (Option 3) as fallback. EventEnvelope uses JSON, so unknown fields are naturally ignorable. For breaking changes, ProtocolBridge handles translation.

### Q9: How does ProtocolBridge handle bidirectional version translation?

**Context**: In an ACP v2↔v1 bridge, events flow v2→v1 (downgrade) and feedback flows v1→v2 (upgrade). Does one bridge handle both directions?

**Options**:
1. **One bridge, both directions** — `ACPv2toV1Bridge` handles both event translation (v2→v1) and feedback translation (v1→v2)
2. **Two bridges, composed** — `ACPv2toV1Bridge` for events, `ACPv1toV2Bridge` for feedback, composed in `BridgedCommChannel`

**Recommendation**: Option 1. A single bridge class handles both directions for a given version pair. The bridge knows both versions; direction is determined by whether it's translating an event (outbound) or feedback (inbound). This avoids bridge composition complexity.

### Q10: Should MQ transport support multiple consumers (fan-out)?

**Context**: Multiple protocol servers might want to consume the same RunLoop's events (e.g., an ACP server and a logging service).

**Options**:
1. **Single consumer** — one consumer per session topic; RunLoop only talks to one protocol server
2. **Multiple consumers** — MQ naturally supports fan-out; each consumer has its own offset
3. **Consumer groups** — Kafka-style consumer groups for load balancing

**Recommendation**: Multiple consumers (Option 2). MQ backends (Redis Streams, NATS, Kafka) naturally support multiple consumers with independent offsets. This enables patterns like: ACP server consumes events, logging service consumes same events for audit, analytics service consumes for monitoring. No extra design needed — it's an MQ property.

### Q11: How does crash recovery interact with MQ transport?

**Context**: If RunLoop crashes and restarts, does it re-read from MQ? Or does it use Journal/SnapshotStore recovery?

**Options**:
1. **Journal/SnapshotStore recovery only** — RunLoop recovers from journal/snapshot; MQ is just transport
2. **MQ replay** — RunLoop re-reads events from MQ from last acked offset
3. **Both** — Journal/SnapshotStore for internal state, MQ for consumer offset

**Recommendation**: Journal/SnapshotStore recovery only (Option 1). MQ is the transport, not the source of truth. Journal (events) and SnapshotStore (state) are the sources of truth. On restart: `journal.resume(snapshot_store)` → resume. MQ consumer offset is managed by the protocol server, not RunLoop. This separates concerns: RunLoop owns state, protocol server owns its own consumption position.

---

## Decision Record

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-07-08 | Adopt Option 1 (Six Pluggable Dimensions) | Maximum orthogonality, covers all 4 execution modes, aligns with cross-framework research |
| 2026-07-08 | RunLoop IS RFC-0041's restructured RunHandle | Avoids creating a new class; extends existing work |
| 2026-07-08 | Journal + SnapshotStore interfaces defined from Phase 1 | Durability is not an afterthought; MemoryJournal and MemorySnapshotStore implement them in-memory |
| 2026-07-08 | Persistence/replay is Phase 2 (core, not optional) | Foundation for long-running, channel wake-up, and MQ transport |
| 2026-07-08 | CommChannel and TriggerSource may be same object | Supports bidirectional channels without duplication |
| 2026-07-08 | Serial Turns only | Simplicity; parallelism through multiple RunLoops or teams |
| 2026-07-08 | StateUpdate maps directly to ACP v2 state_change | Identical semantics; no translation needed |
| 2026-07-08 | Phase 0 (RFC-0041) is prerequisite | Run/Turn separation is the foundation for dimension injection |
| 2026-07-08 | EventTransport as 6th dimension | Decouples protocol layer from Python; enables polyglot servers via MQ |
| 2026-07-08 | EventEnvelope is JSON with schema_version | Language-agnostic, debuggable; protobuf can be added later as optimization |
| 2026-07-08 | ProtocolBridge as CommChannel decorator | Version translation at boundary; RunLoop unaware of protocol versions |
| 2026-07-08 | Single ProtocolBridge handles both directions | Avoids bridge composition complexity for a given version pair |
| 2026-07-08 | Journal is source of truth for events, SnapshotStore for state, not MQ | MQ is transport; crash recovery uses journal/snapshot, not MQ replay |
| 2026-07-08 | MQ supports multiple consumers (fan-out) | Natural MQ property; enables audit/logging/analytics alongside protocol server |
| 2026-07-08 | Journal compaction is hybrid (time + size) | Prevents unbounded growth for long-running channel agents |
| 2026-07-08 | Adopt Akka/Pekko terminology (journal/snapshot/recovery_point) | Grounded in 7-framework research; cleanest separation of event log vs state image |
| 2026-07-08 | CommChannel.on_state_change() observer callback | Decouples GatewayChannel from RunLoop internals; no direct _status access |
| 2026-07-08 | Journal is sole sequence number owner | seq originates from Journal.append()/upsert(); CommChannel/EventTransport never generate their own |
| 2026-07-08 | MQEndpoint convenience class for bidirectional MQ | Adapter pattern; prevents combinatorial explosion of channel+trigger combinations |
| 2026-07-08 | Tool execution log for idempotent recovery | Oracle critical issue #1: prevents duplicate side effects on crash recovery |
| 2026-07-08 | LLM replay strategy: replay journal events, don't re-invoke LLM | Oracle critical issue #2: LLM is non-deterministic; replay events to consumer instead |
| 2026-07-08 | Phase 2 split into 2a (basic recovery) and 2b (replay+compaction) | Earlier delivery of crash recovery; replay is complex and can follow later |
| 2026-07-08 | Phase 4 depends on Phase 1 only, not Phase 2 | MQ backends provide their own durability; Journal/SnapshotStore is separate concern |
| 2026-07-08 | Phase 6 depends on Phase 2 only, not Phase 5 | Channel wake-up is event-driven, not schedule-driven; doesn't need scheduled task infra |
| 2026-07-08 | tenant_id in EventEnvelope | Multi-tenant isolation at MQ/query level |
| 2026-07-08 | Backpressure with configurable drop_policy | Replaces silent EventBus overflow; default backpressure, alternatives drop_oldest/drop_newest |
| 2026-07-08 | trace_id in EventEnvelope metadata | Distributed tracing across protocol servers, MQ transport, and external consumers |
| 2026-07-08 | PII encryption at rest for DurableJournal, DurableSnapshotStore, EventLogJournal, and EventLogSnapshotStore | GDPR compliance; AES-256-GCM with KMS-managed keys |
| 2026-07-08 | Journal schema migration with auto_migrate | Forward-compatible JSON + migrate_journal() for breaking changes; rollback safety |
| 2026-07-08 | Python Protocol + tagged unions + capability composition | Rust-like type safety with Python ergonomics; structural subtyping for dimension interfaces |

---

## References

### Internal

- [RFC-0041: Run vs Turn Separation](RFC-0041-loop-run-separation.md) — Prerequisite. Defines RunLoop (restructured RunHandle) with idle/running/done states.
- [RFC-0037: Unify Steer and Followup](RFC-0037-unify-steer-followup.md) — Subsumed by TriggerSource dimension. Steer/followup now works across all modes via CommChannel feedback.
- [RFC-0029: Agent Reactivation](RFC-0029-agent-reactivation-pending-prompt-queue.md) — Legacy mechanism, superseded by RunLoop's steer/followup.
- [Lifecycle Analysis](../../adr/lifecycle-analysis.md) — Cross-framework research basis for this RFC.
- [OpenSpec: introduce-anyio-structured-concurrency](../../../openspec/changes/archive/introduce-anyio-structured-concurrency/) — CancelScope hierarchy used by RunLoop.

### External

- [ACP v2 Schema](https://github.com/wey-gu/agent-client-protocol-schema) — StateUpdate (Running/Idle/RequiresAction), session/inject (steer/queue modes).
- [ACP conversion.rs](https://github.com/wey-gu/agent-client-protocol-schema/blob/main/src/v2/conversion.rs) — Cross-version bridging pattern (v1↔v2 translation).
- [pydantic-ai AgentRun](https://ai.pydantic.dev/) — `Agent.iter()` → `next(node)` → `End` cycle; `PendingMessageDrainCapability` for asap/when_idle drain.
- [opencode EventV2](https://github.com/sst/opencode) — Durable event store with SQL + projectors + replay.
- [pi Agent Loop](https://github.com/anthropics/pi) — Pure event-stream loop with steering+followup dual queue.
- [hermes-agent](https://github.com/jasonkneen/hermes-agent) — Gateway adapters (Telegram/Discord/Slack), cron scheduler, 7000-line god-object loop (anti-pattern reference).
- [Redis Streams](https://redis.io/docs/data-types/streams/) — MQ backend with consumer groups, replay, and persistence.
- [NATS JetStream](https://docs.nats.io/nats-concepts/jetstream) — MQ backend with durable streams and consumer offsets.
- [Apache Kafka](https://kafka.apache.org/) — Distributed event streaming platform with partitioned topics and consumer groups.

---

## Appendix A: Python Type Patterns

This RFC uses modern Python type patterns to achieve Rust-like trait composition without departing from Python idioms. The three patterns below are the foundation for all dimension interfaces.

### Pattern 1: Protocol (Structural Subtyping)

Python `Protocol` provides structural subtyping — any class that implements the right methods automatically satisfies the protocol, without inheritance. This is analogous to Rust's implicit trait implementations.

```python
from typing import Protocol

class CanPublish(Protocol):
    """Capability protocol for event publication.

    Any class with a compatible publish() method satisfies this protocol.
    No inheritance required — structural subtyping.
    """
    async def publish(self, event: RichAgentStreamEvent) -> None: ...

class CanReceive(Protocol):
    """Capability protocol for feedback reception."""
    async def recv(self) -> Feedback | None: ...

class CanTrigger(Protocol):
    """Capability protocol for prompt delivery."""
    async def poll(self) -> Prompt | None: ...
```

**Why not ABC inheritance?** ABC forces explicit inheritance, creating tight coupling. Protocol allows any class to satisfy the interface by having the right methods. This enables:
- `GatewayChannel` satisfies `CanPublish + CanReceive + CanTrigger` without inheriting from 3 ABCs
- `MQEndpoint` satisfies `CanPublish + CanReceive + CanTrigger` via composition
- Third-party classes can satisfy protocols without importing AgentPool

### Pattern 2: Tagged Unions with Discriminator

All events use tagged unions with `discriminator="type"` for exhaustive matching. This is Python's equivalent of Rust enums.

```python
from pydantic import BaseModel, Field
from typing import Literal, Annotated

class PartDeltaEvent(BaseModel):
    type: Literal["part_delta"] = "part_delta"
    delta: str

class ToolCallStartEvent(BaseModel):
    type: Literal["tool_call_start"] = "tool_call_start"
    tool_name: str
    tool_input: dict

class StreamCompleteEvent(BaseModel):
    type: Literal["stream_complete"] = "stream_complete"
    message: ChatMessage

# Tagged union — enables exhaustive match
RichAgentStreamEvent = Annotated[
    PartDeltaEvent | ToolCallStartEvent | StreamCompleteEvent,
    Field(discriminator="type")
]

# Usage: exhaustive match (mypy enforces all cases)
def handle_event(event: RichAgentStreamEvent) -> None:
    match event:
        case PartDeltaEvent(delta=text):
            print(text, end="")
        case ToolCallStartEvent(tool_name=name):
            print(f"\n[Tool: {name}]")
        case StreamCompleteEvent(message=msg):
            print(f"\n\nDone: {msg.content}")
```

**Benefits**: mypy enforces exhaustive matching (like Rust `match`), new event types require updating all handlers (compile-time safety), serialization is automatic via Pydantic.

### Pattern 3: Capability-Based Composition

Instead of monolithic interfaces (one ABC with all methods), compose capabilities from small Protocol fragments. This solves the GatewayChannel problem — it needs publish + receive + trigger, but shouldn't force all CommChannels to implement TriggerSource.

```python
# Small capability protocols
class CanPublish(Protocol): ...
class CanReceive(Protocol): ...
class CanTrigger(Protocol): ...

# Monolithic ABCs are composed FROM capabilities (not the reverse)
class CommChannel(CanPublish, CanReceive, Protocol):
    """CommChannel = CanPublish + CanReceive + lifecycle methods."""
    async def attach(self, run_loop: RunLoop) -> None: ...
    async def on_state_change(self, state: RunState) -> None: ...
    async def close(self) -> None: ...

class TriggerSource(CanTrigger, Protocol):
    """TriggerSource = CanTrigger + lifecycle methods."""
    async def subscribe(self, run_loop: RunLoop) -> None: ...
    async def close(self) -> None: ...

# GatewayChannel composes ALL capabilities
class GatewayChannel(CommChannel, TriggerSource):
    """Satisfies CanPublish + CanReceive + CanTrigger.
    No false inheritance — each capability is a separate Protocol.
    """
    ...

# DirectChannel only satisfies CommChannel capabilities (no CanTrigger)
class DirectChannel(CommChannel):
    """Satisfies CanPublish + CanReceive. Does NOT satisfy CanTrigger."""
    ...
```

**Why this matters**: The dimension overlap problem (CommChannel vs TriggerSource for bidirectional channels) is resolved by capability composition. A class can satisfy 0-N capability protocols. No "BidirectionalChannel" base class needed — it's just a class that happens to satisfy both protocols.

### Pattern Summary

| Pattern | Python Feature | Rust Equivalent | Used For |
|---------|---------------|-----------------|----------|
| Protocol | `typing.Protocol` | Implicit trait impl | Dimension interfaces |
| Tagged unions | Pydantic discriminator | Enum + match | Event types |
| Capability composition | Protocol intersection | Trait composition | GatewayChannel, MQEndpoint |

**Design decision**: Use Protocol + tagged unions as the base. Capability composition (CanPublish/CanReceive/CanTrigger) for the GatewayChannel problem. This gives us Rust-like type safety with Python ergonomics.

---

## Appendix B: Glossary Addendum — Cross-Framework Terminology

This appendix maps AgentPool's terminology to established frameworks, grounded in research from 7 systems (Temporal, Durable Functions, Akka/Pekko, Erlang/OTP, Ray, Flink, CAF).

### Three Recovery Model Categories

| Category | Frameworks | Key Property |
|----------|-----------|-------------|
| Event sourcing + replay | Temporal, Durable Functions, Akka/Pekko | Full execution logged to append-only log; state rebuilt by replaying |
| Snapshot + rollback | Flink, Ray | Periodic state snapshots; recovery resets to nearest snapshot |
| Process restart | Erlang/OTP, CAF | No persisted state; process killed and restarted fresh |

AgentPool follows **Event sourcing + replay** (Akka/Pekko model), with clean separation between journal (event log) and snapshot (state image).

### Terminology Cross-Reference

| AgentPool | Akka/Pekko | Temporal | Durable Functions | Flink | Erlang/OTP |
|-----------|-----------|----------|-------------------|-------|------------|
| Journal | journal | Event History | orchestration history | changelog | N/A |
| Snapshot | snapshot-store | (unified) | (unified) | checkpoint | N/A |
| Recovery point | sequenceNr | WorkflowTask boundary | checkpoint | checkpoint ID | N/A |
| Committed | persisted + snapshotted | completed WorkflowTask | checkpointed | completed checkpoint | N/A |
| Inflight | persisted, not snapshotted | pending WorkflowTask | uncheckpointed | in-flight data | N/A |
| Replay | receiveRecover | Replay | Replay | N/A | N/A |
| Compaction | manual journal cleanup | Continue-As-New | ContinueAsNew | automatic | N/A |
| Supervisor | N/A | N/A | N/A | N/A | supervisor (one_for_one, etc.) |

**Key insight**: No framework names "crash point" as a primitive — it is always defined by the recovery mechanism. AgentPool follows Akka's clean separation: journal (event persistence) and snapshot (state image) are separate concepts, both pluggable via Journal + SnapshotStore.

### Notable Patterns from Research

1. **Akka/Pekko**: Cleanest separation — journal and snapshot store are separate pluggable components. `receiveRecover` / `RecoveryCompleted` / `SnapshotOffer` lifecycle. `PersistenceId` + `highestSequenceNr` for recovery positioning.

2. **Temporal**: Unified Event History (no separate snapshot). `IsReplaying` flag is internal, not exposed to user code. `Continue-As-New` truncates history. `Reset` copies history to a point and discards the rest.

3. **Durable Functions**: `IsReplaying` is a first-class API — user code can check `IsReplaying` to skip side effects during replay. AgentPool's tool execution log serves the same purpose.

4. **Flink**: Distinguishes checkpoint (auto, system-managed) vs savepoint (manual, portable). State Backend (working state) vs Checkpoint Storage (snapshots) are separate. Chandy-Lamport barrier for distributed snapshots.

5. **Erlang/OTP**: No state persistence. Supervisor restart strategies (one_for_one, one_for_all, rest_for_one) with intensity/period anti-oscillation. AgentPool's `RunLoopSupervisor` (future work) would adopt this pattern.

---

## Appendix C: Derived Lifecycles

The six dimensions (RunLoop, TriggerSource, Journal, SnapshotStore, CommChannel, EventTransport) define the **primary lifecycle**. However, AgentPool has several **derived lifecycles** that are managed by the primary lifecycle but have their own state machines, cleanup procedures, and failure modes. This appendix maps these derived lifecycles and identifies gaps that the unified architecture must address.

### C.1: Session Lifecycle

The session lifecycle is the most complex derived lifecycle. It manages the creation, tracking, and cleanup of agent sessions.

**State Machine**:
```
CREATED → IDLE → RUNNING → (CANCELLED | FAILED) → IDLE/CHECKPOINTED → CLOSING → CLOSED
```

**Key Phases**:

| Phase | Implementation | Key Files |
|-------|---------------|-----------|
| Creation | `SessionController.get_or_create_session()` | `orchestrator/session_controller.py:261-395` |
| Agent instantiation | `get_or_create_session_agent()` — 3 branches: fresh native, child inheriting parent, per-session ACP | `session_controller.py:397-685` |
| Run tracking | `RunHandle` — idle/wake/turn loop, steer/followup injection | `orchestrator/run.py:197-433` |
| Close | 7-stage process: signal → timeout → cancel → snapshot → cascade children → MCP cleanup → agent exit → EventBus teardown | `session_controller.py:848-972` |
| Recovery | `resume_session()` — in-process elicitation resume, crash recovery rehydration | `orchestrator/session_pool.py:601-817` |
| TTL | Background sweeper every 30 min | `session_controller.py:1382-1395` |

**Gap**: The close process is fragile — a crash in any of the 7 stages leaves partial state. The unified architecture's Journal + SnapshotStore addresses this by making each stage's completion durable.

### C.2: MCP Lifecycle

AgentPool currently has **4 separate MCP managers**, each with its own lifecycle:

| Manager | Scope | Key Files | Cleanup |
|---------|-------|-----------|---------|
| `GlobalConnectionPool` | Pool-level shared stdio connections | `mcp/global_pool.py` | `shutdown_all(timeout=10s)` — no ref counting |
| `MCPManager._SessionContext` | Per-session MCP context, toolset cache | `mcp/manager.py:187-208` | `cleanup_session()` at session close |
| `SessionConnectionPool` | Per-session isolated MCP transports | `mcp/session_pool.py` | Owner-task for stdio |
| `SkillMcpManager` | Per-skill lazy MCP connections | `skills/skill_mcp_manager.py` | 5-min idle timeout, exponential backoff retry |

**MCP State Machine**:
```
REGISTERED → CONNECTING → ACTIVE → (IDLE → DISCONNECTED → RECONNECTING) → SHUTTING_DOWN → TERMINATED
```

**Gap 1 (Critical): MCP process leak on agent init crash**
- Location: `session_controller.py:575-579`
- `agent.__aenter__()` spawns MCP subprocesses, then `agent.load_session()` is called unsafely
- If `load_session()` fails, all subprocess references are dropped — processes dangle
- **Fix**: The unified architecture should treat MCP startup as part of the RunLoop's `start()` method, with MCP connections tracked in the Journal. On crash recovery, MCP connections are re-established from the Journal, not from in-memory state.

**Gap 2 (Critical): Skill MCP cleanup not wired into session close**
- `SkillMcpManager.cleanup(session_id)` depends on `on_run_ended()` being called by `SkillCapability`
- If `SkillCapability` is not loaded or `on_run_ended()` is not called (e.g., crash), skill MCP connections leak
- **Fix**: The unified architecture should register MCP cleanup as a RunLoop lifecycle hook, not as a capability callback. `RunLoop.close()` should cascade to all MCP managers deterministically.

**Gap 3 (Critical): `MCPManager.cleanup_session()` tied to deprecated API**
- If `MCPManager` is removed in v0.5.0 (as planned), per-session MCP state cleanup needs a new home
- **Fix**: The unified architecture should define a `MCPLifecycleManager` that coordinates all 4 MCP managers. This manager is injected into RunLoop as a lifecycle dependency, not as an ad-hoc cleanup call.

**Additional gaps**:
- No unifying lifecycle orchestrator for 4 MCP managers
- Two independent subprocess kill paths (owner-task vs ProcessManager)
- EventBus cleanup not crash-safe (stale queues if close crashes)
- Shared deps between pool and session have no lifecycle enforcement

### C.3: Resource Lifecycle

The resource lifecycle manages the async context manager chain that spans the entire AgentPool hierarchy.

**Context Manager Chain**:
```
AgentPool → SessionPool → SessionController → MCPManager → agent → ProcessManager → AsyncExitStack
```

**Cleanup order**: MCP before agent exit (correct), but fragile — crash in `agent.__aenter__()` after MCP spawn leaves processes dangling.

**Structured concurrency**: Per-session CancelScope, asyncio.Lock, asyncio.Task, async generators. No anyio.TaskGroup usage (TaskManager deprecated).

**Gap**: The current resource lifecycle has no crash safety. The unified architecture addresses this by:
1. Making resource acquisition part of the Journal (each resource acquisition is a journal entry)
2. On crash recovery, the RunLoop re-acquires resources from the journal
3. Resource cleanup is a RunLoop lifecycle hook, not an ad-hoc context manager exit

### C.4: Derived Lifecycle Gaps Summary

| Gap | Severity | Affected Lifecycle | Unified Architecture Fix |
|-----|----------|-------------------|------------------------|
| MCP process leak on init crash | Critical | MCP, Resource | Journal-tracked MCP startup; RunLoop manages MCP lifecycle |
| Skill MCP cleanup not guaranteed | Critical | MCP | MCP cleanup as RunLoop lifecycle hook, not capability callback |
| MCPManager.cleanup_session() deprecated | Critical | MCP | New MCPLifecycleManager coordinating all 4 managers |
| Close process not crash-safe | High | Session | Journal + SnapshotStore makes each close stage durable |
| No unifying MCP orchestrator | High | MCP | MCPLifecycleManager injected into RunLoop |
| EventBus cleanup not crash-safe | Medium | Session | EventBus replaced by CommChannel + EventTransport |
| Resource acquisition not crash-safe | Medium | Resource | Journal-tracked resource acquisition |

**Recommendation**: These derived lifecycle gaps should be addressed in sub-RFCs that build on the unified architecture. The primary fix is making resource acquisition and cleanup part of the RunLoop's durable lifecycle, not ad-hoc operations scattered across the codebase.

---

## Appendix D: ToolTransport — Cross-Layer Transport for Remote Tool Calls

### D.1: The Cross-Layer Access Problem

The six dimensions handle the **execution model** (how Turns run, how events flow). ResourceProviders handle the **capability model** (what tools are available). These are orthogonal — until a tool call needs to reach a remote service (e.g., MCP-over-ACP).

**Problem**: When tools are remote, ResourceProvider needs network transport. But EventTransport is owned by CommChannel (Dimension 6). ResourceProvider is a sibling of CommChannel under RunLoop — there is no defined path for ResourceProvider to access EventTransport.

This is not a hypothetical concern. MCP-over-ACP, remote MCP servers, and any distributed tool invocation all require the capability layer to send/receive messages over transport. Connection lifecycle (expire, timeout, reconnect) must be handled at the transport level, not leaked into Turn execution logic.

### D.2: Design Principle

**ToolTransport is part of the capability model, not a seventh dimension.**

The six dimensions handle Turn lifecycle. ToolTransport handles tool invocation transport. They are orthogonal but may share underlying infrastructure (e.g., the same MQ connection).

```
┌──────────────────────────────────────────────────────────┐
│                        RunLoop                             │
│                                                           │
│  ┌─────────────────────┐  ┌──────────────────────────┐   │
│  │  Execution Model    │  │  Capability Model         │   │
│  │  (6 dimensions)     │  │                           │   │
│  │                     │  │  ResourceProvider         │   │
│  │  CommChannel ───────┼──┼─► ToolTransport           │   │
│  │   └─ EventTransport │  │    ├─ LocalToolTransport  │   │
│  │       (MQ conn)     │  │    ├─ ACPToolTransport    │   │
│  │                     │  │    └─ MQToolTransport ────┼───┼──► shared MQ connection
│  │                     │  │                           │   │
│  └─────────────────────┘  └──────────────────────────┘   │
│                                                           │
│  Shared infrastructure:                                   │
│    CommChannel.EventTransport ──────┐                    │
│    ToolTransport.MQ ────────────────┘ (same MQ client)   │
└──────────────────────────────────────────────────────────┘
```

### D.3: ToolTransport Protocol

```python
from __future__ import annotations
from typing import Protocol, runtime_checkable, Any, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
import time
import json
import hashlib


class TransportState(Enum):
    """State of a ToolTransport connection."""
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    DEGRADED = "degraded"       # Partial connectivity (some tools unavailable)
    ERROR = "error"             # No connectivity


class ToolTimeoutError(Exception):
    """Tool call exceeded timeout."""


class ToolConnectionError(Exception):
    """Tool transport connection lost and could not be re-established."""


@dataclass
class ToolDefinition:
    """Discovered tool metadata."""
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolResult:
    """Successful tool execution result."""
    tool_name: str
    output: Any
    is_error: bool = False


@dataclass
class ToolError:
    """Failed tool execution result."""
    tool_name: str
    error_type: str          # "timeout" | "connection" | "execution" | "not_found"
    error_message: str
    retryable: bool = False


@runtime_checkable
class ToolTransport(Protocol):
    """Transport for remote tool calls.

    Implementations MAY share underlying connection with EventTransport
    (e.g., MQToolTransport shares the same MQ client as MessageQueueTransport),
    but MUST NOT depend on CommChannel internals.

    Connection lifecycle (expire, timeout, reconnect) is handled internally.
    Implementations SHOULD catch transport errors and return ToolError
    rather than raising exceptions.
    """

    async def call_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        turn_id: str,
        timeout: float = 30.0,
    ) -> ToolResult | ToolError:
        """Call a remote tool and return the result.

        Idempotent via turn_id + call_id: if the remote already processed
        this tool call (e.g., after reconnect), returns the cached result
        instead of re-executing.

        Implementations SHOULD catch ToolTimeoutError and
        ToolConnectionError internally and return ToolError instead
        of propagating exceptions. This ensures errors are data
        (journalable, replayable) rather than control flow.
        """
        ...

    async def discover_tools(self) -> list[ToolDefinition]:
        """Discover available remote tools.

        Implementations MAY cache results. Cache invalidates on
        TransportState changes (RECONNECTING → CONNECTED triggers
        re-discovery).
        """
        ...

    def add_state_listener(
        self,
        callback: Callable[[TransportState], None],
    ) -> None:
        """Register a callback for transport state changes.

        ResourceProvider uses this to mark tools as unavailable during
        RECONNECTING/ERROR and re-discover tools on CONNECTED.
        """
        ...

    async def close(self) -> None:
        """Clean up transport resources (connections, subscriptions)."""
        ...
```

### D.4: Implementations

#### LocalToolTransport

No transport — direct function call. For built-in tools (bash, read, grep, etc.).

```python
class LocalToolTransport:
    """No-op transport for local tools. Calls execute in-process."""

    def __init__(self) -> None:
        self._state: TransportState = TransportState.CONNECTED

    async def call_tool(
        self, tool_name: str, tool_input: dict, turn_id: str, timeout: float = 30.0
    ) -> ToolResult | ToolError:
        # Local tools are called directly by the agent, not via transport.
        # This transport exists only for interface uniformity; call_tool()
        # is never invoked on it in practice.
        return ToolError(tool_name, "not_found", "Local tools are called directly", retryable=False)

    async def discover_tools(self) -> list[ToolDefinition]:
        return []  # Local tools are registered statically.

    def add_state_listener(self, callback: Callable[[TransportState], None]) -> None:
        pass  # Always CONNECTED, no state changes.

    async def close(self) -> None:
        pass
```

#### ACPToolTransport

Direct ACP JSON-RPC connection to a remote agent (stdio or websocket). Connection managed by MCPLifecycleManager.

```python
class ACPToolTransport:
    """Direct ACP transport for remote tool calls.

    Manages its own connection (stdio subprocess or websocket) to a
    remote ACP agent. Connection lifecycle is handled by
    MCPLifecycleManager (exponential backoff reconnection).
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        reconnect_backoff: float = 1.0,  # Exponential backoff base
        reconnect_max_attempts: int = 5,
    ) -> None:
        self._command = command
        self._args = args or []
        self._cwd = cwd
        self._backoff = reconnect_backoff
        self._max_attempts = reconnect_max_attempts
        self._state: TransportState = TransportState.ERROR
        self._state_callbacks: list[Callable[[TransportState], None]] = []
        self._tool_cache: list[ToolDefinition] | None = None

    async def _ensure_connected(self) -> None:
        """Connect or reconnect if needed. Raises ToolConnectionError on failure."""
        ...

    async def call_tool(
        self, tool_name: str, tool_input: dict, turn_id: str, timeout: float = 30.0
    ) -> ToolResult | ToolError:
        try:
            await self._ensure_connected()
            # Send ACP tool/call request, await response
            result = await self._send_request(
                "tools/call", {"name": tool_name, "arguments": tool_input},
                timeout=timeout,
            )
            return ToolResult(tool_name=tool_name, output=result)
        except ToolTimeoutError:
            return ToolError(tool_name, "timeout", f"Tool call timed out after {timeout}s", retryable=True)
        except ToolConnectionError as e:
            return ToolError(tool_name, "connection", str(e), retryable=True)

    async def discover_tools(self) -> list[ToolDefinition]:
        if self._tool_cache is not None and self._state == TransportState.CONNECTED:
            return self._tool_cache
        await self._ensure_connected()
        result = await self._send_request("tools/list", {})
        self._tool_cache = [ToolDefinition(**t) for t in result]
        return self._tool_cache

    def add_state_listener(self, callback: Callable[[TransportState], None]) -> None:
        self._state_callbacks.append(callback)

    def _set_state(self, state: TransportState) -> None:
        if self._state != state:
            self._state = state
            self._tool_cache = None  # Invalidate cache on state change
            for cb in self._state_callbacks:
                cb(state)

    async def close(self) -> None:
        self._set_state(TransportState.ERROR)
        # Terminate subprocess or close websocket
        ...
```

#### MQToolTransport

Shares the underlying MQ connection with `MessageQueueTransport` (EventTransport). Uses separate topics for tool calls and results.

```python
class MQToolTransport:
    """MQ-based transport for remote tool calls.

    Shares the underlying MQ client (Redis/NATS/Kafka) with
    MessageQueueTransport (EventTransport), but uses separate topics.
    This avoids duplicating MQ connections while keeping tool call
    traffic isolated from agent event traffic.

    Connection lifecycle is delegated to the shared MQ client,
    which handles reconnection internally. State changes propagate
    to both CommChannel and MQToolTransport.
    """

    def __init__(
        self,
        mq_client: Any,  # Shared MQ client (same instance as MessageQueueTransport)
        session_id: str,
        tool_call_topic: str | None = None,    # Default: {session_id}:tool_calls
        tool_result_topic: str | None = None,  # Default: {session_id}:tool_results
    ) -> None:
        self._mq = mq_client
        self._session_id = session_id
        self._call_topic = tool_call_topic or f"{session_id}:tool_calls"
        self._result_topic = tool_result_topic or f"{session_id}:tool_results"
        self._state: TransportState = TransportState.CONNECTED
        self._state_callbacks: list[Callable[[TransportState], None]] = []
        self._pending: dict[str, asyncio.Future[ToolResult | ToolError]] = {}
        self._tool_cache: list[ToolDefinition] | None = None
        self._consumer_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start consuming tool results from MQ."""
        self._consumer_task = asyncio.create_task(self._consume_results())

    async def call_tool(
        self, tool_name: str, tool_input: dict, turn_id: str, timeout: float = 30.0
    ) -> ToolResult | ToolError:
        # call_id includes input hash to avoid collisions when the same
        # tool is called multiple times within a single turn.
        input_hash = hashlib.md5(
            json.dumps(tool_input, sort_keys=True).encode()
        ).hexdigest()[:8]
        call_id = f"{turn_id}:{tool_name}:{input_hash}"
        future: asyncio.Future[ToolResult | ToolError] = asyncio.get_running_loop().create_future()
        self._pending[call_id] = future

        envelope = EventEnvelope(
            seq=0,  # Tool calls don't use Journal seq; idempotency via call_id
            session_id=self._session_id,
            tenant_id=self._session_id,  # Per-session isolation
            turn_id=turn_id,
            event_type="tool_call_request",
            event_data={"call_id": call_id, "tool_name": tool_name, "tool_input": tool_input},
            schema_version="1.0",
            timestamp=time.time(),
            metadata={},
        )
        await self._mq.publish(self._call_topic, json.dumps(envelope.to_dict()))

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            self._pending.pop(call_id, None)
            return ToolError(tool_name, "timeout", f"Tool call timed out after {timeout}s", retryable=True)

    async def _consume_results(self) -> None:
        """Consume tool results from MQ and resolve pending futures."""
        async for message in self._mq.subscribe(self._result_topic):
            envelope = EventEnvelope.from_dict(json.loads(message))
            data = envelope.event_data
            call_id = data.get("call_id")
            if call_id and call_id in self._pending:
                future = self._pending.pop(call_id)
                if data.get("is_error"):
                    future.set_result(ToolError(
                        tool_name=data["tool_name"],
                        error_type=data.get("error_type", "execution"),
                        error_message=data.get("error_message", data.get("output", "")),
                        retryable=data.get("retryable", False),
                    ))
                else:
                    future.set_result(ToolResult(
                        tool_name=data["tool_name"],
                        output=data.get("output"),
                        is_error=False,
                    ))

    def add_state_listener(self, callback: Callable[[TransportState], None]) -> None:
        self._state_callbacks.append(callback)

    def _on_mq_state_change(self, connected: bool) -> None:
        """Called by shared MQ client on connection state change."""
        new_state = TransportState.CONNECTED if connected else TransportState.RECONNECTING
        if self._state != new_state:
            self._state = new_state
            self._tool_cache = None
            for cb in self._state_callbacks:
                cb(new_state)

    async def close(self) -> None:
        if self._consumer_task:
            self._consumer_task.cancel()
        self._state = TransportState.ERROR
        for future in self._pending.values():
            future.set_exception(ToolConnectionError("Transport closed"))
        self._pending.clear()
```

### D.5: MQ Topic Extension

The MQ topic scheme extends to support tool calls:

```
MQ Topics (per session):
  {session_id}:events         ← CommChannel (agent events to consumer)
  {session_id}:feedback       ← CommChannel (consumer → RunLoop: steer, followup)
  {session_id}:prompts        ← TriggerSource (prompts to RunLoop)
  {session_id}:tool_calls     ← ToolTransport (tool call requests to remote agents)    [NEW]
  {session_id}:tool_results   ← ToolTransport (tool results from remote agents)       [NEW]
```

Tool call and result messages use the same `EventEnvelope` format:

```json
// tool_calls topic
{
  "seq": 0,
  "session_id": "coder_session",
  "tenant_id": "coder_session",
  "turn_id": "turn_abc123",
  "event_type": "tool_call_request",
  "event_data": {
    "call_id": "turn_abc123:filesystem.read_file:a1b2c3d4",
    "tool_name": "filesystem.read_file",
    "tool_input": {"path": "/src/main.py"}
  },
  "schema_version": "1.0",
  "timestamp": 1720435200.0,
  "metadata": {}
}

// tool_results topic (success)
{
  "seq": 0,
  "session_id": "coder_session",
  "tenant_id": "coder_session",
  "turn_id": "turn_abc123",
  "event_type": "tool_call_result",
  "event_data": {
    "call_id": "turn_abc123:filesystem.read_file:a1b2c3d4",
    "tool_name": "filesystem.read_file",
    "output": "file contents here...",
    "is_error": false
  },
  "schema_version": "1.0",
  "timestamp": 1720435201.0,
  "metadata": {}
}

// tool_results topic (error)
{
  "seq": 0,
  "session_id": "coder_session",
  "tenant_id": "coder_session",
  "turn_id": "turn_abc123",
  "event_type": "tool_call_result",
  "event_data": {
    "call_id": "turn_abc123:filesystem.read_file:a1b2c3d4",
    "tool_name": "filesystem.read_file",
    "is_error": true,
    "error_type": "connection",
    "error_message": "MCP server connection expired",
    "retryable": true
  },
  "schema_version": "1.0",
  "timestamp": 1720435201.0,
  "metadata": {}
}
```

**Note**: `seq=0` for tool calls because they are not journaled by the Journal (they are not agent events). Idempotency is handled via `call_id` (composed of `turn_id` + `tool_name` + input hash) in the tool execution log.

### D.6: Connection Lifecycle Handling

#### MQ Connection Drop (shared by EventTransport + MQToolTransport)

```
1. MQ client detects connection loss
2. MQ client internally reconnects (built-in to Redis/NATS/Kafka clients)
3. State change propagated to both consumers:
   - MessageQueueTransport → CommChannel._on_mq_state_change()
   - MQToolTransport._on_mq_state_change()
4. Both transition to RECONNECTING:
   - CommChannel: pauses event publishing, queues events
   - MQToolTransport: marks remote tools as "degraded", new tool calls queue
5. Reconnection succeeds → both transition to CONNECTED:
   - CommChannel: flushes queued events
   - MQToolTransport: re-discovers tools, resumes pending tool calls
6. Reconnection fails (timeout) → both transition to ERROR:
   - CommChannel: reports error to RunLoop
   - MQToolTransport: fails pending tool calls with ToolConnectionError
```

#### MCP Server Connection Expire (managed by MCPLifecycleManager)

```
1. Tool call arrives at remote MCP proxy agent
2. MCP proxy's MCPLifecycleManager detects MCP connection expired
3. MCPLifecycleManager attempts reconnection (exponential backoff)
4. Reconnection succeeds:
   - Tool call executes normally
   - Result flows back: MCP → ResourceProvider → ToolTransport → MQ → main RunLoop
5. Reconnection fails:
   - ToolError(error_type="connection", retryable=True) flows back
   - Main RunLoop receives error, decides: retry, skip, or fail Turn
```

#### Tool Call In-Flight + Crash

```
1. ToolCallStartEvent written to Journal (turn_id=abc, tool=filesystem.read_file)
2. Tool execution log entry: {turn_id=abc, tool=filesystem.read_file, status=interrupted}
3. Crash occurs
4. Recovery: Journal.resume() → load snapshot → replay journal
5. Replay finds ToolCallStartEvent → check tool execution log:
   - status=completed → use cached result from ToolCallCompleteEvent (already in journal)
   - status=interrupted → re-send tool call (idempotent via call_id)
   - no record exists → remote likely didn't receive, re-send
6. Remote receives re-sent tool call with same call_id:
   - Already processed → returns cached result (idempotent)
   - Not processed → executes normally
7. Tool execution log updated: status=completed, result stored
```

### D.7: MCP-over-ACP Composition Pattern

Under this architecture, MCP-over-ACP is a **composition of two RunLoops** connected via MQ:

```
Main RunLoop                           MCP Proxy RunLoop
┌──────────────────────┐               ┌──────────────────────────┐
│ Trigger: MQTrigger   │               │ Trigger: MQTrigger        │
│ Journal: DurableJour │               │ Journal: MemoryJournal    │
│ Snapshot: DurableSnap│               │ Snapshot: MemorySnapshot  │
│ Comm: MQChannel      │◄─────────────►│ Comm: MQChannel           │
│ Transport: NATS      │  MQ topics    │ Transport: NATS           │
│                      │               │                           │
│ Capabilities:        │               │ Capabilities:             │
│  - bash (local)      │               │  - MCP filesystem         │
│  - read (local)      │  tool_calls → │  - MCP git                │
│  - remote_tools ─────┼──────────────►│  - MCP slack              │
│    (ToolTransport:   │  tool_results │                           │
│     MQToolTransport) │◄──────────────┤ ResourceProvider:         │
│                      │               │  MCPResourceProvider      │
└──────────────────────┘               │  (ToolTransport: ACP)     │
                                       └──────────────────────────┘
```

**Key points**:
- The MCP proxy is a full RunLoop with its own six dimensions (lightweight: MemoryJournal, MemorySnapshotStore)
- The main RunLoop's ResourceProvider uses `MQToolTransport` to reach the proxy
- The proxy RunLoop's ResourceProvider uses `ACPToolTransport` (or direct MCP) to reach MCP servers
- Connection lifecycle is handled at each layer: MQ reconnection (shared), ACP/MCP reconnection (MCPLifecycleManager)
- Tool calls are idempotent via `call_id` (turn_id + tool_name + input hash) — safe to retry after any connection failure
- The proxy can be in any language — it just needs to consume `tool_calls` topic and publish to `tool_results` topic

### D.8: YAML Configuration

```yaml
agents:
  coder:
    type: native
    model: "openai:gpt-4o"
    lifecycle:
      journal: durable
      snapshot: durable
      comm: mq
      transport:
        type: message_queue
        backend: nats_jetstream
        url: "nats://localhost:4222"
      session_id: coder
    capabilities:
      tools:
        - type: local
          name: bash
        - type: local
          name: read
        - type: remote
          transport: mq          # Uses MQToolTransport (shares NATS connection)
          agent: mcp_proxy       # Routes to mcp_proxy's tool_calls topic
          tools: [filesystem, git, slack]

  mcp_proxy:
    type: acp
    command: acp-mcp-proxy
    args: ["--mcp-config", "mcp_servers.yml"]
    lifecycle:
      journal: memory
      snapshot: memory
      comm: mq
      transport:
        type: message_queue
        backend: nats_jetstream
        url: "nats://localhost:4222"
      session_id: mcp_proxy
    capabilities:
      tools:
        - type: mcp
          server: filesystem
          transport: stdio
        - type: mcp
          server: git
          transport: stdio
        - type: mcp
          server: slack
          transport: websocket
```

### D.9: Impact on RFC-0042 Dimensions

ToolTransport does **not** add a seventh dimension. The six dimensions remain unchanged:

| Dimension | Impact | Change |
|---|---|---|
| RunLoop | None | RunLoop still owns CommChannel, Journal, SnapshotStore |
| TriggerSource | None | TriggerSource still handles prompt arrival |
| Journal | None | Journal still handles event persistence; tool calls tracked via tool execution log |
| SnapshotStore | None | SnapshotStore still handles state snapshots |
| CommChannel | None | CommChannel still owns EventTransport for agent events |
| EventTransport | Minor | MessageQueueTransport may share MQ client with MQToolTransport |

**What changes**: ResourceProvider (capability model, orthogonal to dimensions) gains an optional `ToolTransport` dependency for remote tool access. This is a capability model concern, not an execution model concern.

### D.10: Decision Record

| Decision | Rationale |
|---|---|
| ToolTransport is a Protocol, not ABC | Consistent with all other dimension interfaces in RFC-0042 |
| ToolTransport is NOT a dimension | It belongs to the capability model, orthogonal to the execution model (6 dimensions) |
| MQToolTransport shares MQ client with EventTransport | Avoids duplicating connections; single reconnection logic |
| Tool calls use separate MQ topics | Isolates tool call traffic from agent event traffic; allows independent scaling |
| `seq=0` for tool call envelopes | Tool calls are not agent events; idempotency via `turn_id` + `call_id` in tool execution log |
| `call_tool()` returns `ToolResult \| ToolError` (not raises) | Errors are data, not exceptions — enables journaling and replay |
| State change via `add_state_listener()` (observer registration) | Disambiguated from `CommChannel.on_state_change()` which IS the callback; ToolTransport REGISTERS callbacks (observer subject vs observer callback) |
| ACPToolTransport manages own connection | ACP is a different protocol from MQ; connection lifecycle is independent |
| LocalToolTransport returns `ToolError(not_found)` | Local tools are called directly; `call_tool()` returns error if accidentally invoked, maintaining Protocol contract |
