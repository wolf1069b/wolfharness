---
rfc_id: RFC-0001
title: Unified Run Tracking with RunHandle and PydanticAI Queue Adoption
status: APPROVED
author: AgentPool Architecture Team
reviewers:
  - Metis - Plan Consultant (PASSED)
  - Oracle - Read-Only High-IQ Consultant (PASSED)
created: 2025-06-03
last_updated: 2025-06-03
decision_date: 2025-06-03
---

## Overview

This RFC proposes introducing `RunHandle` as a first-class ephemeral runtime object managed directly by `SessionPool`, alongside migrating native agents to PydanticAI's native pending message queue (`enqueue()`). The core problem is that AgentPool currently lacks a centralized way to track, enumerate, and cancel active agent runs across all sessions — a capability that xeno-agent's `BackgroundTaskManager` already provides.

The proposal introduces `RunHandle` as a first-class ephemeral runtime object tracked in `SessionPool._runs`, enabling operations like `list_active_runs()`, `cancel_run(run_id)`, and `get_run(run_id)`. For native agents, this is combined with a migration to PydanticAI's `PendingMessageDrainCapability` to replace manual follow-up prompt queuing. Non-native agents (ACP, ClaudeCode, AGUI) retain their existing manual queue system.

## Background & Context

### Current Architecture

AgentPool's session orchestration lives in `src/wolfharness/orchestrator/core.py`, combining `SessionController` (session CRUD) and `TurnRunner` (turn execution + manual queue management). Key components:

- **`SessionState`**: Long-lived session metadata holding `turn_lock` (asyncio.Lock) and `active_run_ctx` (manually-synchronized pointer to `AgentRunContext`)
- **`AgentRunContext`**: Ephemeral per-run container with `injection_manager` (`PromptInjectionManager`)
- **`TurnRunner`**: Manages `_post_turn_injections`, `_post_turn_prompts`, `_injection_locks`, `inject_prompt()`, `queue_prompt()`, `_process_queued_work()`, `_trigger_auto_resume()`
- **`PromptInjectionManager`**: Two distinct responsibilities:
  1. `inject()`/`consume()`: Tool result augmentation via `after_tool_execute`
  2. `queue()`/`pop_queued()`: Follow-up prompt queuing after a turn ends

### Why This Matters Now

1. **No pool-level visibility**: To find active runs, one must iterate all sessions and check `active_run_ctx`. No `list_active_runs()` or `cancel_run_by_id()` exists.
2. **Fragile `active_run_ctx` pointer**: Manually synchronized in `finally` blocks; race-prone.
3. **PydanticAI v1.101.0+** (already at 1.102.0 in this project) introduced native `AgentRun.enqueue(*content, priority='asap'|'when_idle')` and `PendingMessageDrainCapability`, which can replace manual follow-up prompt queuing for native agents.
4. **xeno-agent precedent**: `BackgroundTaskManager` provides unified task tracking — AgentPool should have equivalent capability.

## Problem Statement

### Specific Problems

1. **Scattered run tracking**: Run state is fragmented across `SessionState.active_run_ctx`, `SessionState.turn_lock`, and `TurnRunner` internal dicts. No single authority knows about all active runs.
2. **Cancellation requires session ID**: To cancel a run, callers must know which session owns it. There's no `cancel_run(run_id)`.
3. **Manual queue duplication**: AgentPool re-implements PydanticAI's pending message queue for native agents with ~200 lines of bespoke code (`_post_turn_prompts`, `_process_queued_work`, `_trigger_auto_resume`).
4. **Concurrency fragility**: The `active_run_ctx` / `turn_lock` combination is used for both turn serialization and graceful close-session waiting, creating coupling between unrelated concerns.

### Impact of Not Solving

- Operational blind spots: Cannot enumerate active runs, monitor run health, or enforce `max_concurrent_runs`
- Technical debt: Manual queue system must be maintained despite upstream providing equivalent functionality
- Race conditions: `active_run_ctx` pointer synchronization is error-prone
- Barrier to new features: Pool-level orchestration (load balancing, circuit breakers) requires unified run tracking

## Goals & Non-Goals

### Goals

- Introduce `RunHandle` as a first-class ephemeral object with lifecycle tracking, managed directly by `SessionPool`
- Eliminate the fragile `SessionState.active_run_ctx` pointer
- Replace native agents' manual **follow-up prompt queue** with PydanticAI's `PendingMessageDrainCapability`
- Preserve `PromptInjectionManager.inject()`/`consume()` for **tool result augmentation** across all agents
- Refactor `SessionController` into a unified request router with agent-type-aware dispatch
- Keep `SessionState.turn_lock` for **non-native agents** (they still need turn serialization)

### Non-Goals

- Changing non-native agents' queueing/injection behavior (they keep existing manual system)
- Changing `PromptInjectionManager.inject()`/`consume()` behavior
- Historical run tracking / persistence (runs are ephemeral; historical tracking may be added later)
- Changes to `EventBus` API shape
- Changes to `SessionData` persistence schema

## Evaluation Criteria

| Criterion | Weight | Threshold |
|-----------|--------|-----------|
| **Unified visibility** | High | Must provide pool-level `list_active()`, `cancel_run()`, `get_run()` |
| **Native agent queue correctness** | High | Must match or exceed current manual queue behavior for follow-up prompts |
| **Non-native agent compatibility** | High | Must not break any non-native agent behavior |
| **Concurrency safety** | High | No race conditions, deadlocks, or duplicate runs |
| **Event stream compatibility** | High | Protocol handlers must receive identical events before/after migration |
| **Implementation risk** | Medium | Prefer lower-risk incremental approach |
| **Maintainability** | Medium | Reduce code duplication; clear separation of concerns |
| **Performance** | Low | No significant regression in throughput or latency |

## Options Analysis

### Option A: Unified Run Tracking + PydanticAI Queue (Two-Phase)

**Description**: 
- Phase 1: Build `RunHandle` managed by `SessionPool._runs` for all agent types, keeping existing manual queues
- Phase 2: Migrate native agents to PydanticAI `enqueue()`, unify non-native agents into `TurnRunner`

**Advantages**:
- **Lower risk**: Phase 1 validates run tracking without changing execution semantics
- **Incremental validation**: Each phase can be tested and deployed independently
- **Clear rollback**: Phase 1 is additive; Phase 2 can be reverted while keeping Phase 1
- **Addresses core problem**: Pool-level tracking is achieved in Phase 1
- **Aligns with upstream**: Uses PydanticAI's native capability instead of re-implementing
- **Simpler than separate registry**: `SessionPool._runs` is just a dict — no new class lifecycle

**Disadvantages**:
- **Two-phase complexity**: Requires maintaining both old and new paths during transition
- **Timeline**: Takes longer than a single-phase approach
- **Legacy code**: Non-native agent queue logic persists in `TurnRunner` (necessary; non-native agents cannot use PydanticAI)

**Evaluation Against Criteria**:

| Criterion | Score | Notes |
|-----------|-------|-------|
| Unified visibility | 5/5 | `SessionPool._runs` provides full visibility from Phase 1 |
| Native agent queue correctness | 4/5 | Prototype required to verify event mapping parity |
| Non-native agent compatibility | 5/5 | No changes to non-native paths in either phase |
| Concurrency safety | 4/5 | `_request_lock` + `current_run_id` prevents races; needs careful cleanup ordering |
| Event stream compatibility | 3/5 | `RunExecutor` must replicate current `_stream_events()`; prototype blocks Phase 2 |
| Implementation risk | 4/5 | Phase 1 is low risk; Phase 2 is medium risk but isolated |
| Maintainability | 4/5 | Eliminates native-agent manual queue; minimal new abstractions |
| Performance | 5/5 | `SessionPool._runs` dict ops are O(1); no registry overhead |

**Effort Estimate**: Medium-High (2-3 weeks)

**Risk Assessment**: 
- **Medium**: Phase 2 event mapping may not match current behavior exactly
- **Mitigation**: Prototype event mapping (task 1.3) blocks Phase 2; extensive tests for protocol handler compatibility

---

### Option B: Session-Scoped Run Tracking Only (No Pool Tracking)

**Description**: 
- Keep run tracking session-scoped only
- Replace `active_run_ctx` with `current_run_id` in `SessionState`
- Do NOT introduce `RunHandle` or `_runs` dict
- Migrate native agents to PydanticAI queue directly

**Advantages**:
- **Simpler**: No pool-level tracking; less code to maintain
- **Faster implementation**: Single phase, no run handle abstraction
- **Less complexity**: No new objects to reason about

**Disadvantages**:
- **Doesn't solve core problem**: No pool-level visibility; still must iterate all sessions to find active runs
- **No cross-session operations**: Cannot implement `cancel_run(run_id)` or `max_concurrent_runs`
- **Future work blocked**: Pool-level orchestration features require later refactoring anyway
- **Missed opportunity**: Addresses PydanticAI queue migration but not the architectural gap that prompted this RFC

**Evaluation Against Criteria**:

| Criterion | Score | Notes |
|-----------|-------|-------|
| Unified visibility | 1/5 | No pool-level visibility; session-scoped only |
| Native agent queue correctness | 4/5 | Same PydanticAI migration as Option A |
| Non-native agent compatibility | 5/5 | No changes to non-native paths |
| Concurrency safety | 4/5 | Similar locking strategy, just without pool tracking |
| Event stream compatibility | 3/5 | Same event mapping challenge as Option A |
| Implementation risk | 3/5 | Simpler but doesn't solve the actual problem |
| Maintainability | 3/5 | Doesn't reduce architectural debt |
| Performance | 4/5 | No `_runs` dict overhead |

**Effort Estimate**: Medium (1-2 weeks)

**Risk Assessment**: 
- **Low**: Simpler change with less code
- **But**: Fails to address the core architectural gap; may require another refactor later

---

### Option C: Keep Current Architecture + Add PydanticAI Queue Only

**Description**: 
- Minimal change: only migrate native agents to PydanticAI `enqueue()`
- Keep `active_run_ctx` and `turn_lock` as-is
- Do NOT introduce `RunHandle` or `_runs`

**Advantages**:
- **Minimal risk**: Smallest surface area of change
- **Fastest**: Single focused change
- **No new abstractions**: No learning curve for new concepts

**Disadvantages**:
- **Core problem unsolved**: No pool-level run tracking at all
- **Fragile pointer persists**: `active_run_ctx` remains a manually-synchronized pointer
- **Technical debt accumulates**: Manual queue removed but no structural improvement
- **Future work blocked**: Any pool-level feature requires yet another refactor

**Evaluation Against Criteria**:

| Criterion | Score | Notes |
|-----------|-------|-------|
| Unified visibility | 0/5 | No change to tracking; still session-scoped and fragmented |
| Native agent queue correctness | 4/5 | PydanticAI queue migration works |
| Non-native agent compatibility | 5/5 | No changes |
| Concurrency safety | 2/5 | `active_run_ctx` pointer remains fragile |
| Event stream compatibility | 3/5 | Same event mapping challenge |
| Implementation risk | 5/5 | Very low risk |
| Maintainability | 2/5 | Doesn't improve architecture |
| Performance | 4/5 | No `_runs` overhead |

**Effort Estimate**: Low-Medium (3-5 days)

**Risk Assessment**: 
- **Very Low**: Minimal change
- **But**: Doesn't address why we're here; wastes opportunity for architectural improvement

## Recommendation

**Recommended**: **Option A** — Unified Run Tracking with Two-Phase PydanticAI Queue Adoption.

**Justification**:
- Directly addresses the core problem (no pool-level run tracking) with `SessionPool._runs`
- `RunHandle` is lightweight; `SessionPool._runs` is just a dict — no separate registry class lifecycle
- Two-phase approach de-risks implementation: Phase 1 is additive and safe to keep; Phase 2 is isolated and can be rolled back independently
- Aligns with xeno-agent's `BackgroundTaskManager` pattern, which has proven effective
- Eliminates manual queue duplication for native agents while preserving non-native compatibility
- Sets foundation for future pool-level orchestration features (load balancing, circuit breakers, monitoring)

**Acknowledged trade-offs**:
- Non-native agent queue logic persists in `TurnRunner` (necessary; non-native agents cannot use PydanticAI)
- Two queue systems create some cognitive overhead (mitigated by clear documentation and agent-type-aware routing)
- Phase 2 requires a successful event mapping prototype before proceeding

## Technical Design

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AgentPool                                │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                    SessionPool                             │ │
│  │                                                            │ │
│  │  _sessions: dict[str, SessionState]                        │ │
│  │  _runs: dict[str, RunHandle]     ◄── pool-level tracking   │ │
│  │                                                            │ │
│  │  active_runs → list[RunHandle]    (O(n), n = #sessions)    │ │
│  │  cancel_run(run_id) → bool                                 │ │
│  │  get_run(run_id) → RunHandle | None                        │ │
│  │                                                            │ │
│  │  ┌─────────────────────────────────────────────────────┐   │ │
│  │  │          SessionController (Unified Router)          │   │ │
│  │  │                                                      │   │ │
│  │  │  receive_request(session_id, content, priority)      │   │ │
│  │  │    ├─ Native agent? ──► _create_run() or enqueue()   │   │ │
    │  │  │    └─ Non-native? ───► TurnRunner.inject_prompt()    │   │ │
│  │  │                                                      │   │ │
│  │  │  _create_run() → RunHandle → add to _runs            │   │ │
│  │  │  _cleanup_run() → remove from _runs                  │   │ │
│  │  └─────────────────────────────────────────────────────┘   │ │
│  │                          ▲                                 │ │
│  │              ┌───────────┴───────────┐                     │ │
│  │              ▼                       ▼                     │ │
│  │    ┌──────────────────┐   ┌────────────────────┐          │ │
│  │    │   RunExecutor    │   │   TurnRunner       │          │ │
│  │    │   (Phase 2)      │   │   (Phase 1)        │          │ │
│  │    └──────────────────┘   └────────────────────┘          │ │
│  │              ▼                       ▼                     │ │
│  │    ┌──────────────────┐   ┌────────────────────┐          │ │
│  │    │   Native Agent   │   │   Non-native Agent │          │ │
│  │    │   (PydanticAI)   │   │   (ACP/Claude/AGUI)│          │ │
│  │    └──────────────────┘   └────────────────────┘          │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Data Models

#### RunHandle
```python
@dataclass
class RunHandle:
    run_id: str
    status: RunStatus  # pending | running | completed | failed
    run_ctx: AgentRunContext
    agent_type: str    # "native" | "acp" | "claude" | "agui"
    session_id: str
    agent_run_ref: Any  # PydanticAI AgentRun for native, Task/Runner for non-native
    created_at: datetime
    completed_at: datetime | None
    complete_event: asyncio.Event
    
    def start(self) -> None: ...
    def complete(self) -> None: ...
    def fail(self, exception: Exception) -> None: ...
    def cancel(self) -> None: ...
```

#### SessionPool Changes
```python
class SessionPool:
    def __init__(self, max_concurrent_runs: int | None = None):
        self._sessions: dict[str, SessionState] = {}
        self._runs: dict[str, RunHandle] = {}  # NEW: pool-level run tracking
        self._max_concurrent_runs: int | None = max_concurrent_runs  # NEW: optional limit
    
    @property
    def active_runs(self) -> list[RunHandle]:
        return [r for r in self._runs.values() if r.status == "running"]
    
    def cancel_run(self, run_id: str) -> bool:
        if run := self._runs.get(run_id):
            run.cancel()
            return True
        return False
    
    def get_run(self, run_id: str) -> RunHandle | None:
        return self._runs.get(run_id)
```

#### SessionState Changes
```python
@dataclass
class SessionState:
    # REMOVED: active_run_ctx: AgentRunContext | None
    # KEPT for non-native: turn_lock: asyncio.Lock
    current_run_id: str | None = None
    _request_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    closing: bool = False  # NEW: set by close_session(), checked by receive_request()
    # ... other fields unchanged
```

#### AgentRunContext Changes
```python
@dataclass
class AgentRunContext:
    # Made optional: injection_manager for tool result augmentation
    injection_manager: PromptInjectionManager | None = None
    # ... other fields unchanged
```

### Event Mapping (Native Agents — Phase 2)

| PydanticAI Source | AgentPool Event | Notes |
|-------------------|-----------------|-------|
| `AgentRun` created | `RunStartedEvent` | Once per run |
| `ModelResponseNode` start | `PartStartEvent` | Model begins responding |
| `ModelResponseNode` stream | `PartDeltaEvent` | Per text chunk |
| `ModelResponseNode` end | `PartEndEvent` | Response complete |
| `FunctionToolNode` start | `ToolCallStartEvent` | Tool execution begins |
| `FunctionToolNode` end | `ToolCallCompleteEvent` | Tool execution ends |
| `EndNode` | `StreamCompleteEvent` | Normal termination |
| Run cancelled | `StreamCompleteEvent(cancelled=True)` | Cancellation signaling — adds `cancelled: bool = False` to `StreamCompleteEvent` |

**Critical**: `RunExecutor` must use `node.stream()` to preserve current event granularity. The mapping above is conceptual; actual implementation must match `_stream_events()` behavior.

### Error Propagation

- Run failures publish `RunFailedEvent` on EventBus with `run_id`, `session_id`, `exception`
- Protocol handlers subscribe to EventBus and handle errors
- `RunHandle.fail()` sets status, publishes event, sets `complete_event`

### Concurrency Model

```
receive_request():
  acquire _request_lock
  if current_run_id is None:
    current_run_id = new_run_id
    create RunHandle
    add to SessionPool._runs
    start run task (releases lock after start)
  else:
    enqueue message (native) or delegate to TurnRunner (non-native)
  release _request_lock

run task finally block:
  acquire _request_lock
  current_run_id = None
  remove from SessionPool._runs
  release _request_lock
  set complete_event
```

## Implementation Plan

### Phase 1: Run Tracking Foundation (Lower Risk)

**Duration**: 1-2 weeks
**Goal**: Add pool-level run tracking without changing execution semantics

1. **Prototype** (already have pydantic-ai 1.102.0)
   - [x] pydantic-ai already at 1.102.0
   - [ ] Create prototype script testing `agent.iter()` + `next()` with `enqueue(priority='when_idle')`
   - [ ] Document event mapping from prototype findings

2. **RunHandle**
   - [ ] Create `RunHandle` dataclass in `orchestrator/run.py`
   - [ ] Add lifecycle methods: `start()`, `complete()`, `fail()`, `cancel()`

3. **SessionPool Refactor**
   - [ ] Add `_runs: dict[str, RunHandle]` to `SessionPool`
   - [ ] Add `active_runs`, `cancel_run()`, `get_run()` properties/methods
   - [ ] Add thread-safe access patterns
   - [ ] Add tests for pool-level operations

4. **SessionState Refactor**
   - [ ] Remove `active_run_ctx`; add `current_run_id`
   - [ ] Add `_request_lock` (per-session)
   - [ ] Add `closing: bool = False` (set by close_session, checked by receive_request)
   - [ ] Keep `turn_lock` for non-native agents

5. **SessionController Router**
   - [ ] Implement `receive_request()` with agent-type-aware routing
   - [ ] Implement `_create_run()` and `_cleanup_run()`
   - [ ] Add `SessionPool._runs` integration
   - [ ] Add `max_concurrent_runs` enforcement
   - [ ] Add concurrency tests

6. **AgentRunContext + BaseAgent Cleanup**
   - [ ] Make `injection_manager` optional
   - [ ] Update instantiations (keep for all agents in Phase 1)
   - [ ] Handle `None` gracefully in hook manager
   - [ ] Update `BaseAgent._get_session_run_ctx()` to use `SessionPool._runs`

7. **Metrics + Error Propagation**
   - [ ] Update `MetricsCollector` to use `SessionPool.active_runs`
   - [ ] Add `RunFailedEvent` to EventBus
   - [ ] Update `RunHandle.fail()` to publish events
   - [ ] Audit all call sites that catch exceptions from `process_prompt()`

8. **Protocol Handlers + SessionPool Facade**
   - [ ] Update native-agent paths to use `receive_request()`
   - [ ] Redesign `SessionPool.run_stream()` for fire-and-forget semantics
   - [ ] Add `AgentPool` facade null-safety (`session_pool is None`)
   - [ ] Verify non-native paths unchanged

9. **Tests**
   - [ ] SessionPool._runs tests (list, cancel, cleanup)
   - [ ] SessionController tests (create, enqueue, concurrent requests)
   - [ ] Close session tests (graceful, forceful, race, closing guard)
   - [ ] Max concurrent runs enforcement test
   - [ ] BaseAgent._get_session_run_ctx() updated path test
   - [ ] Full test suite: `uv run pytest`

### Phase 2: Native Agent PydanticAI Queue (Higher Risk)

**Duration**: 1-2 weeks
**Blocked by**: Phase 1 completion + successful event mapping prototype

1. **RunExecutor**
   - [ ] Create `RunExecutor` driving `agent.iter()` + `next()` loop
   - [ ] Map PydanticAI events to AgentPool EventBus
   - [ ] Preserve isolated `agent_iteration_task` pattern

2. **TurnRunner Unification**
   - [x] Non-native queue logic already in `TurnRunner`; verified integration with `SessionPool._runs`
   - [x] `turn_lock` preserved for turn serialization

3. **Native Agent Queue Migration**
   - [ ] Remove manual follow-up prompt queue for native agents
   - [ ] Remove `_run_stream_once()` internal loop for native agents
   - [ ] Delegate `inject_prompt()`/`queue_prompt()` to PydanticAI `enqueue()`
   - [ ] Preserve `inject()`/`consume()` for tool result augmentation

4. **Tests**
   - [ ] PydanticAI-native auto-resume tests
   - [ ] `enqueue()` drain tests (asap, when_idle)
   - [ ] Tool result augmentation still works
   - [ ] Event stream parity test (RunExecutor vs current `_stream_events()`)
   - [ ] Full test suite: `uv run pytest`

### Rollback Strategy

- **Phase 1**: Safe to keep. `SessionPool._runs` is additive; doesn't change execution paths.
- **Phase 2**: Revert `RunExecutor` to use manual queues while keeping `SessionPool._runs`. Switch `SessionController` routing back to `TurnRunner` for native agents.

## Open Questions

1. **Subsequent `ModelRequestNode` events from `when_idle` drains**: Should these emit any event? Currently leaning toward "emit nothing" (silent), but protocol handlers may need to reset state.
2. **`PendingMessageDrainCapability` interaction with `NativeAgentHookManager` capabilities**: Need to verify no ordering conflicts since both inject capabilities. `PendingMessageDrainCapability` is auto-injected outermost; AgentPool's own capabilities must wrap inside it.
3. **`enqueue()` from Temporal activities**: Known upstream limitation where messages may be dropped. Document workaround (enqueue from workflow context, not activities).
4. **`SystemPromptPart` mid-run behavior**: Differs across providers (Anthropic/Google hoist to top). Avoid using in `enqueue()`.
5. **`max_concurrent_runs` queuing strategy**: When limit is reached, should requests be rejected immediately or queued? Current design rejects; consider adding an optional queue if needed.

## Decision Record

### Final Decision

**Approved**: Implement Option A — Unified Run Tracking with Two-Phase PydanticAI Queue Adoption.

### Approvers

- Metis - Plan Consultant (PASSED after 6 review rounds)
- Oracle - Read-Only High-IQ Consultant (PASSED after 6 review rounds)

### Key Discussion Points

1. **Scope clarification**: PydanticAI queue only available to native agents; non-native agents keep manual queues.
2. **PromptInjectionManager dual purpose**: `inject()`/`consume()` (tool result augmentation) is NOT replaced by PydanticAI; only `queue()`/`pop_queued()` (follow-up prompts) is replaced.
3. **Two-phase strategy**: Phase 1 (Run tracking) is lower risk and can stand alone; Phase 2 (PydanticAI queue) requires prototype validation.
4. **API naming**: PydanticAI uses `enqueue()` not `enqueue_message()`.
5. **Lock strategy**: Per-session `_request_lock` for check-and-create; `turn_lock` retained for non-native turn serialization.
6. **Run tracking approach**: `SessionPool._runs` directly, not a separate `RunRegistry` class — simpler, same capability.
7. **Close guard**: `SessionState.closing` flag prevents new runs during graceful shutdown, avoiding races between `close_session()` and `receive_request()`.
8. **Pool-level limits**: Optional `max_concurrent_runs` on `SessionPool` provides backpressure without requiring external rate limiting.
9. **BaseAgent compatibility**: `_get_session_run_ctx()` updated to use `SessionPool._runs` instead of `session.active_run_ctx` to preserve tool access to run context.

### Conditions on Approval

- Event mapping prototype (task 1.3) must pass before Phase 2 begins
- Full test suite must pass after each phase
- Non-native agent compatibility tests must pass in both phases
- Tool result augmentation must continue working for native agents after Phase 2

### Related Documents

- `openspec/changes/adopt-pydantic-ai-pending-message-queue/proposal.md`
- `openspec/changes/adopt-pydantic-ai-pending-message-queue/design.md`
- `openspec/changes/adopt-pydantic-ai-pending-message-queue/tasks.md`
- `openspec/changes/adopt-pydantic-ai-pending-message-queue/specs/pending-message-queue/spec.md`
