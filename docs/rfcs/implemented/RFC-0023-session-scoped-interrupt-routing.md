# RFC-0023: Session-Scoped Interrupt Routing

## Header Metadata

---
rfc_id: RFC-0023
title: Session-Scoped Interrupt Routing for Concurrent Agent Safety
status: IMPLEMENTED
author: yuchen.liu
reviewers: []
created: 2026-04-18
last_updated: 2026-04-23
decision_date: 2026-04-23
related_documents:
  - RFC-0021-agent-concurrent-execution-safety.md (Parent RFC — per-run context isolation)
  - RFC-0021-PRE-FLIGHT-ANALYSIS.md (State inventory and audit)
  - PR #17: fix/title-gen-nonblocking (Trigger — review exposed regression)
review_notes:
  - Metis review: self.session_id instance variable has same concurrent bug; run_in_background/stop break without _cancelled
  - Oracle review: session_id key mismatch (run_ctx.session_id UUID vs OpenCode session_id); is_cancelled() caller audit needed
  - 2026-04-23 review: OpenCode abort path also has a route-layer deadlock because abort_session() acquires agent_lock through get_or_load_session()
---

## 1. Overview

### 1.1 Summary

This RFC proposes a **session-scoped interrupt routing** mechanism that aligns OpenCode server's interrupt flow with RFC-0021's per-run context isolation design. The current `interrupt()` implementation stores `AgentRunContext` as an instance variable (`self._active_run_ctx`) to work around ContextVar's cross-task limitation, re-introducing shared mutable state that RFC-0021 was designed to eliminate. This RFC replaces the single instance variable with a `dict[session_id, AgentRunContext]` registry, enabling `interrupt()` to route cancellation to the correct per-session run context. It also fixes the OpenCode server's current abort deadlock by removing the lock-taking session load from `abort_session()`.

### 1.2 Why This Matters Now

PR #17 (title-gen-nonblocking) introduced `self._active_run_ctx` and `self._iteration_task` as instance variables to support cross-task interrupt. Code review identified that these instance variables create race conditions under concurrent sessions — the same class of bugs that RFC-0021 was implemented to solve. Shipping these variables as-is would regress the concurrent safety guarantees established by RFC-0021.

### 1.3 Expected Outcome

After implementation:
- **`interrupt(session_id=...)`** routes cancellation to the correct `AgentRunContext` even when called from a different async task
- **No instance-level mutable streaming state**: `_active_run_ctx` and `_iteration_task` are removed from instance scope, while `_cancelled` is retained only for background-run compatibility
- **Concurrent safety preserved**: Multiple sessions can run `run_stream()` on the same agent without overwriting each other's interrupt handles
- **OpenCode abort path becomes deadlock-free**: `abort_session()` no longer waits on `agent_lock` before calling `interrupt()`
- **Backward compatible**: Serial single-session usage continues to work identically

## 2. Background & Context

### 2.1 RFC-0021 Achievement

RFC-0021 migrated the following instance-level mutable state into `AgentRunContext` (per-run scope):

| State | Old Location | New Location | Status |
|-------|-------------|--------------|--------|
| `cancelled` | `self._cancelled` | `run_ctx.cancelled` | ✅ Migrated |
| `current_task` | `self._current_stream_task` | `run_ctx.current_task` | ✅ Migrated |
| `event_queue` | `self._event_queue` | `run_ctx.event_queue` | ✅ Migrated |
| `injection_manager` | `self._injection_manager` | `run_ctx.injection_manager` | ✅ Migrated |

The `AgentRunContext` is created fresh in each `run_stream()` call and accessed within the same async task via `_current_run_ctx_var` (a `contextvars.ContextVar`).

### 2.2 The Cross-Task Interrupt Problem

OpenCode server's `abort_session()` runs in an HTTP request handler task (Task B), while `run_stream()` runs in a different task (Task A). When Task B calls `interrupt()`, it cannot access `_current_run_ctx_var` because ContextVars are task-scoped — reading from Task B returns `None`.

```
Task A (run_stream)          Task B (abort_session)
─────────────────────        ─────────────────────
_current_run_ctx_var.set(ctx)
  ... agent runs ...
                              _current_run_ctx_var.get() → None!
                              await agent.interrupt()
                              → How to find ctx?
```

### 2.3 PR #17's Workaround (The Regression)

PR #17 solved the cross-task problem by storing `run_ctx` as an instance variable:

```python
# base_agent.py — PR #17 approach
self._active_run_ctx = run_ctx          # Instance variable: single slot
self._iteration_task = iteration_task   # Instance variable: single slot
```

This works for single-session use but re-introduces shared mutable state:

- Session A calls `run_stream()` → `self._active_run_ctx = ctx_A`
- Session B calls `run_stream()` → `self._active_run_ctx = ctx_B` (overwrites A)
- `abort_session(A)` calls `interrupt()` → finds `ctx_B` instead of `ctx_A`
- Session A becomes a zombie (uncancellable), Session B is incorrectly cancelled

### 2.4 OpenCode's Abort Flow

```python
# session_routes.py: current flow
@router.post("/{session_id}/abort")
async def abort_session(session_id: str, state: StateDep) -> bool:
    session = await get_or_load_session(state, session_id)  # acquires state.agent_lock
    # ...
    await state.agent.interrupt()   # ← No session_id passed, and often never reached
    # ...
```

Key observations:

1. `abort_session()` **already has `session_id`** in its handler signature but does not pass it to `interrupt()`. The `session_id` is the natural routing key for interrupt.
2. `get_or_load_session()` acquires `state.agent_lock`, while `_process_message_locked()` holds that same lock for the entire stream. During an active stream, `abort_session()` blocks before it can call `interrupt()`. This route-layer deadlock must be fixed together with session-scoped routing.

### 2.5 Glossary

| Term | Definition |
|------|-----------|
| **Run Registry** | A `dict[session_id, AgentRunContext]` mapping maintained by `BaseAgent` to route interrupts |
| **Cross-Task Interrupt** | Calling `interrupt()` from a different async task than the one running `run_stream()` |
| **Session-Scoped Routing** | Using `session_id` to identify which `AgentRunContext` to target for cancellation |
| **Route-Layer Deadlock** | `abort_session()` blocks on `agent_lock` before it can call `interrupt()` |
| **Zombie Run** | An active `run_stream()` that has lost its interrupt handle and cannot be cancelled |

## 3. Problem Statement

### 3.1 Specific Problem

Three instance variables introduced by PR #17 violate RFC-0021's per-run isolation:

| Variable | Location | Problem |
|----------|----------|---------|
| `self._active_run_ctx` | `base_agent.py` | Overwritten by concurrent `run_stream()` calls; interrupt targets wrong session |
| `self._iteration_task` | `native_agent/agent.py` | Same overwrite problem; LLM API call cancellation targets wrong task |
| `self._cancelled` | `base_agent.py` | Dual-track: both `self._cancelled` and `run_ctx.cancelled` exist; `is_cancelled()` checks instance flag, not per-run flag |

In addition, the current OpenCode abort route has a **server-side deadlock** independent of the registry problem:

| Component | Location | Problem |
|----------|----------|---------|
| `abort_session()` → `get_or_load_session()` | `session_routes.py` | Acquires `agent_lock` before calling `interrupt()` |
| `_process_message_locked()` | `message_routes.py` | Holds `agent_lock` for the full streaming duration |

These two problems stack: even a perfect registry does not help if the abort route never reaches `interrupt()`.

### 3.2 Evidence

**Code Review (PR #17, 11 comments, 6 high-priority)**:

1. `_active_run_ctx` race condition: concurrent sessions overwrite each other (3 separate review comments)
2. `_iteration_task` race condition: same overwrite pattern (2 review comments)
3. `conversation` history pollution: aborted message added to shared agent instance (2 review comments)
4. `_cancelled` not reset on new `run_stream()` (1 comment — actual bug even in single-session)
5. Redundant `_current_stream_task` (1 comment)
6. Stale comment referencing ContextVar (1 comment)
7. Duplicate fixture (1 comment)

**OpenCode abort investigation (2026-04-23)**:

1. `abort_session()` calls `get_or_load_session()`, which acquires `state.agent_lock`
2. `_process_message_locked()` already holds `state.agent_lock` while streaming
3. `abort_session()` therefore blocks before `interrupt()` and cannot cancel the active run
4. A later patch proposal that only waited for the stream task after `interrupt()` was insufficient, because the code path still never reached `interrupt()` when the lock was held

**Test Coverage Gap**: No test validates concurrent session interrupt routing.

### 3.3 Impact of Not Solving

- **Architectural regression**: RFC-0021's isolation guarantees are undermined
- **Production risk**: If multi-session support is ever added (e.g., web UI with multiple tabs), concurrent sessions will corrupt each other's state
- **Technical debt**: Dual-track `_cancelled` + `run_ctx.cancelled` creates confusion; future maintainers may accidentally rely on the wrong flag
- **Review blocker**: PR #17 cannot be merged with known race conditions

## 4. Goals & Non-Goals

### 4.1 Goals (In Scope)

1. **Primary**: Enable cross-task interrupt that routes to the correct `AgentRunContext` by `session_id`
2. **Primary**: Remove instance-level mutable streaming state (`_active_run_ctx`, `_iteration_task`) and stop using `_cancelled` for streaming runs
3. **Primary**: Make OpenCode `abort_session()` reach `interrupt()` without acquiring `agent_lock`
4. **Secondary**: Unify streaming cancellation to a single source of truth (`run_ctx.cancelled`)
5. **Secondary**: Update `is_cancelled()` to check per-run context, not instance flag, for streaming flows
6. **Secondary**: Add concurrent session interrupt routing and abort deadlock regression tests

### 4.2 Non-Goals (Out of Scope)

1. **Not**: Adding per-session agent instances (separate architectural decision)
2. **Not**: Solving `conversation` history isolation (requires per-session agent — noted as future work)
3. **Not**: Modifying the OpenCode TUI client (server-side change only)
4. **Not**: Addressing SSE disconnect-triggered cancellation (separate bug, deferred)
5. **Not**: Replacing the background-run control flow (`run_in_background()` / `stop()`) with a new registry mechanism in this RFC

### 4.3 Success Criteria

- [ ] `interrupt(session_id="ses_A")` cancels only `run_ctx` for session A, even when session B is also active
- [ ] `self._active_run_ctx` and `self._iteration_task` instance variables are removed
- [ ] `self._cancelled` is no longer used for streaming cancellation; `is_cancelled()` uses per-run context for streaming flows
- [ ] `abort_session()` bypasses `get_or_load_session()` and passes `session_id` to `interrupt()`
- [ ] All 9 existing concurrent safety tests pass
- [ ] All 8 existing interrupt tests pass
- [ ] New test: concurrent sessions can be interrupted independently
- [ ] New test: `abort_session()` does not block on `agent_lock` during an active stream
- [ ] Serial single-session behavior unchanged

## 5. Evaluation Criteria

| Criterion | Weight | Description | Measurement |
|-----------|--------|-------------|-------------|
| **Concurrent Safety** | Critical | No shared state between concurrent runs | Test: 2+ concurrent sessions, each interruptable independently |
| **RFC-0021 Alignment** | High | No regression of per-run isolation | No instance-level mutable streaming state remains; `_cancelled` stays background-only |
| **Backward Compatibility** | High | Serial execution works unchanged | All existing tests pass |
| **Implementation Complexity** | Medium | Reasonable effort and risk | Estimated dev days |
| **Debuggability** | Medium | Easy to trace interrupt routing | Clear logging of session_id → run_ctx resolution |
| **Performance** | Low | No significant overhead | Dict lookup < 1µs |

## 6. Options Analysis

### Option 1: Session-ID Run Registry (Recommended)

**Description**: Maintain a `dict[str, AgentRunContext]` in `BaseAgent` keyed by `session_id`. `interrupt()` accepts an optional `session_id` parameter and looks up the correct `run_ctx` from the registry. `run_stream()` registers its context on entry and removes it on exit.

```python
class BaseAgent:
    def __init__(self, ...):
        self._active_runs: dict[str, AgentRunContext] = {}

    async def run_stream(self, ..., session_id: str | None = None):
        run_ctx = AgentRunContext(deps=deps)
        effective_session_id = session_id or generate_session_id()
        run_ctx.session_id = effective_session_id
        self._active_runs[effective_session_id] = run_ctx
        try:
            ...
        finally:
            self._active_runs.pop(effective_session_id, None)

    async def interrupt(
        self,
        run_ctx: AgentRunContext | None = None,
        *,
        session_id: str | None = None,
    ):
        effective = run_ctx or (self._active_runs.get(session_id) if session_id else None)
        if effective:
            effective.cancelled = True
        await self._interrupt(effective)
```

**Advantages**:
- Full concurrent safety: each session's `run_ctx` is tracked independently
- Natural routing via `session_id` — the key that `abort_session()` already has
- No new data structures — just a dict replacing single instance variable
- Backward compatible at the API level: `interrupt(session_id=...)` is additive and explicit
- Aligns with RFC-0021 design: state stays in `AgentRunContext`, agent only holds a lookup table

**Disadvantages**:
- `interrupt()` API gains a new parameter (backward compatible — optional with default)
- All meaningful cross-task interrupt callers must pass `session_id` (caller audit required)
- `session_id` must be passed from `run_stream()` callers (already available in most call sites)

**Evaluation Against Criteria**:
| Criterion | Score | Notes |
|-----------|-------|-------|
| Concurrent Safety | ✅ | Per-session isolation |
| RFC-0021 Alignment | ✅ | No instance-level mutable streaming state |
| Backward Compatibility | ✅ | Optional parameter, explicit caller contract |
| Implementation Complexity | ✅ | 1-2 days |
| Debuggability | ✅ | Dict is inspectable, log session_id resolution |
| Performance | ✅ | Dict O(1) lookup |

**Effort Estimate**: 1-2 days

**Risk Assessment**: Low — replacing instance variable with dict is a small, well-scoped change

---

### Option 2: ContextVar with `as_task` Propagation

**Description**: Use Python 3.12+ `contextvars.copy_context()` to explicitly propagate `_current_run_ctx_var` from the `run_stream()` task to the `interrupt()` task.

```python
# Store context when run_stream starts
self._stream_context = contextvars.copy_context()

# In interrupt(), read from the stored context
async def interrupt(self, ...):
    ctx = self._stream_context or contextvars.copy_context()
    run_ctx = ctx.get(_current_run_ctx_var)
    ...
```

**Advantages**:
- No new registry data structure needed
- Uses Python's built-in context isolation mechanism
- Preserves the ContextVar-based access pattern

**Disadvantages**:
- `copy_context()` creates a snapshot — stale if `run_ctx` is mutated after copy
- Context objects are not async-aware: they don't auto-update across `await` points
- Only solves single-session case: multiple concurrent runs need multiple contexts
- Requires storing `Context` as instance variable — same overwrite problem as current `_active_run_ctx`
- Not a general solution; still need session routing for concurrent runs

**Evaluation Against Criteria**:
| Criterion | Score | Notes |
|-----------|-------|-------|
| Concurrent Safety | ❌ | Single context snapshot — can't handle multiple concurrent runs |
| RFC-0021 Alignment | ⚠️ | Still uses instance variable for context reference |
| Backward Compatibility | ✅ | No API change |
| Implementation Complexity | ⚠️ | Context propagation subtleties |
| Debuggability | ❌ | Context snapshots are opaque |
| Performance | ✅ | Negligible overhead |

**Effort Estimate**: 1 day (for single-session only)

**Risk Assessment**: Medium — context snapshot staleness bugs are hard to diagnose

---

### Option 3: Interrupt Handle Pattern

**Description**: Return an `InterruptHandle` object from `run_stream()` that callers store and pass to `interrupt()`. The handle encapsulates the `run_ctx` reference.

```python
@dataclass
class InterruptHandle:
    session_id: str
    run_ctx: AgentRunContext
    iteration_task: asyncio.Task | None = None

class BaseAgent:
    async def run_stream(self, ...):
        run_ctx = AgentRunContext(deps=deps)
        handle = InterruptHandle(session_id=session_id, run_ctx=run_ctx)
        self._interrupt_handles[session_id] = handle
        try:
            ...
        finally:
            del self._interrupt_handles[session_id]

    async def interrupt(self, handle: InterruptHandle | None = None, session_id: str | None = None):
        effective = handle or self._interrupt_handles.get(session_id)
        ...
```

**Advantages**:
- Most explicit: callers hold a typed reference to their interrupt handle
- Type-safe: `InterruptHandle` is a proper dataclass, not a bare dict value
- Extensible: can add `iteration_task` and other per-run handles to the dataclass
- Clean separation: `BaseAgent` doesn't need to know about `session_id` semantics

**Disadvantages**:
- More code: new dataclass, new instance variable, API change
- Callers must store the handle (lifecycle management burden)
- Higher migration surface: all `interrupt()` call sites need updating
- `InterruptHandle` lifetime must be managed carefully (dangling references)

**Evaluation Against Criteria**:
| Criterion | Score | Notes |
|-----------|-------|-------|
| Concurrent Safety | ✅ | Per-handle isolation |
| RFC-0021 Alignment | ✅ | No instance-level mutable run state |
| Backward Compatibility | ⚠️ | New type, API change |
| Implementation Complexity | ⚠️ | 2-3 days |
| Debuggability | ✅ | Typed handle is inspectable |
| Performance | ✅ | Dict lookup |

**Effort Estimate**: 2-3 days

**Risk Assessment**: Medium — higher migration surface, handle lifetime management

---

### Option 4: asyncio.Event Signal Pattern

**Description**: Each `run_stream()` creates an `asyncio.Event` as a cancel signal. `interrupt()` sets the event. The `run_stream()` loop checks the event on each iteration.

```python
class BaseAgent:
    async def run_stream(self, ...):
        cancel_signal = asyncio.Event()
        self._cancel_signals[session_id] = cancel_signal
        try:
            while not cancel_signal.is_set() and not run_ctx.cancelled:
                ...
        finally:
            del self._cancel_signals[session_id]

    async def interrupt(self, session_id: str | None = None):
        signal = self._cancel_signals.get(session_id)
        if signal:
            signal.set()
```

**Advantages**:
- Decouples cancellation from task cancellation — no `CancelledError` exceptions
- Event can be awaited (poll-free waiting)
- Simple mental model: set event = stop

**Disadvantages**:
- Does not cancel the asyncio.Task itself — long-running LLM API calls continue
- Requires cooperative checking: `run_stream()` must poll the event
- Does not address `iteration_task` cancellation
- Redundant with `run_ctx.cancelled` — two signals for the same concept
- Adds complexity without solving the full problem (task cancellation still needed)

**Evaluation Against Criteria**:
| Criterion | Score | Notes |
|-----------|-------|-------|
| Concurrent Safety | ⚠️ | Safe routing, but incomplete cancellation |
| RFC-0021 Alignment | ⚠️ | Adds new signal alongside run_ctx.cancelled |
| Backward Compatibility | ✅ | No API change to interrupt() |
| Implementation Complexity | ⚠️ | Cooperative checking throughout run_stream |
| Debuggability | ⚠️ | Two cancellation mechanisms to understand |
| Performance | ✅ | Event check is O(1) |

**Effort Estimate**: 2 days

**Risk Assessment**: Medium — incomplete solution, does not cancel stuck LLM calls

---

## 7. Recommendation

### 7.1 Recommended Option: Option 1 (Session-ID Run Registry)

**Justification**:

1. **Solves the concurrency problem at the correct abstraction**: Routes interrupt by `session_id`, cancels the correct `run_ctx` and `iteration_task`, and removes instance-level mutable streaming state
2. **Minimal API change**: One new optional parameter (`session_id`) on `interrupt()` — fully backward compatible
3. **Aligns with RFC-0021**: State stays in `AgentRunContext`; agent holds only a lookup table, not mutable state
4. **Natural fit**: `abort_session()` already has `session_id` in its handler signature
5. **Pairs cleanly with the required route-layer deadlock fix**: the server can validate the session in memory, then call `interrupt(session_id=...)` without touching `agent_lock`

**Trade-offs Accepted**:
- Callers that need cross-task interrupt must pass `session_id`; silent fallback guessing is explicitly rejected
- `session_id` must flow from `run_stream()` callers — but it's already available in all server-side call sites

**Alternatives Rejected**:
- Option 2 (ContextVar propagation): Does not solve concurrent case
- Option 3 (Interrupt Handle): More code for the same outcome; higher migration burden
- Option 4 (Event Signal): Incomplete — does not cancel stuck LLM tasks

### 7.2 Decision Rationale

| Criterion | Option 1 | Option 2 | Option 3 | Option 4 |
|-----------|----------|----------|----------|----------|
| Concurrent Safety | ✅ | ❌ | ✅ | ⚠️ |
| RFC-0021 Alignment | ✅ | ⚠️ | ✅ | ⚠️ |
| Backward Compatibility | ✅ | ✅ | ⚠️ | ✅ |
| Implementation Complexity | ✅ | ⚠️ | ⚠️ | ⚠️ |
| Debuggability | ✅ | ❌ | ✅ | ⚠️ |
| **Overall** | ✅ | ❌ | ⚠️ | ❌ |

## 8. Technical Design

### 8.1 Architecture

```
┌──────────────────────────────────────────────────────┐
│                   BaseAgent (Instance)               │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │  Shared State (Immutable/Per-Instance)         │  │
│  │  - name, model_name, tools, conversation       │  │
│  │  - _background_run_ctx (single background run) │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │  Run Registry (NEW)                            │  │
│  │  _active_runs: dict[session_id, AgentRunContext]│  │
│  │                                                │  │
│  │  "ses_A" → run_ctx_A (cancelled=False)         │  │
│  │  "ses_B" → run_ctx_B (cancelled=False)         │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘

interrupt(session_id="ses_A")
  → self._active_runs["ses_A"] → run_ctx_A
  → run_ctx_A.cancelled = True
  → _interrupt(run_ctx_A) → cancels run_ctx_A.current_task
  → NativeAgent iteration task via run_ctx (see 8.3)
```

### 8.2 API Changes

#### `BaseAgent.interrupt()` — New `session_id` Parameter

```python
# Before
async def interrupt(self, run_ctx: AgentRunContext | None = None) -> None:

# After
async def interrupt(
    self,
    run_ctx: AgentRunContext | None = None,
    *,
    session_id: str | None = None,
) -> None:
```

Resolution order:
1. Explicit `run_ctx` parameter (highest priority — for programmatic callers)
2. Registry lookup by `session_id` (for OpenCode abort flow)
3. No target found: no-op (caller must pass `run_ctx` or `session_id`; no heuristic guessing)

Important behavioral rule: `interrupt()` **must not** set `self._cancelled`. Only `run_ctx.cancelled` is updated for streaming runs; `self._cancelled` remains reserved for `stop()` / background-run behavior.

Note: Background run cancellation is handled separately via `self._cancelled` and
`_background_run_ctx`. See Section 10.3 for rationale.

#### `BaseAgent.is_cancelled()` — Per-Run Check with Background Fallback

`self._cancelled` is retained for background run compatibility (Decision 2 from review).
For streaming runs, cancellation is checked via per-run context.

```python
# Before
def is_cancelled(self) -> bool:
    background_cancelled = (
        self._background_run_ctx.cancelled if self._background_run_ctx else False
    )
    return self._cancelled or background_cancelled

# After — self._cancelled retained for background runs only
def is_cancelled(self, run_ctx: AgentRunContext | None = None) -> bool:
    # Check per-run context first (concurrent-safe for streaming runs)
    if run_ctx is not None:
        return run_ctx.cancelled
    # Fallback: check current ContextVar (works within run_stream task)
    current = _current_run_ctx_var.get(None)
    if current is not None:
        return current.cancelled
    # Legacy: background run context or instance flag
    # self._cancelled is ONLY for run_in_background()/stop() compatibility
    background_cancelled = (
        self._background_run_ctx.cancelled if self._background_run_ctx else False
    )
    return self._cancelled or background_cancelled
```

#### `NativeAgent._interrupt()` — Iteration Task via `run_ctx`

Move `iteration_task` tracking into `AgentRunContext` (or a companion field) so it's per-run, not per-instance:

```python
@dataclass
class AgentRunContext:
    # ... existing fields ...
    iteration_task: asyncio.Task[Any] | None = None  # NEW
```

```python
# native_agent/agent.py
async def _stream_events(self, run_ctx, ...):
    iteration_task = asyncio.create_task(agent_iteration_task())
    run_ctx.iteration_task = iteration_task  # Per-run, not self._iteration_task
    try:
        ...
    finally:
        run_ctx.iteration_task = None

async def _interrupt(self, run_ctx: AgentRunContext | None = None) -> None:
    task = run_ctx.current_task if run_ctx else None
    if task and not task.done():
        task.cancel()
    # Cancel iteration task via run_ctx (per-run, not per-instance)
    iteration_task = run_ctx.iteration_task if run_ctx else None
    if iteration_task is not None and not iteration_task.done():
        iteration_task.cancel()
```

### 8.3 Data Model Changes

#### `AgentRunContext` — New Fields

```python
@dataclass(kw_only=True)
class AgentRunContext:
    # ... existing fields unchanged ...
    iteration_task: asyncio.Task[Any] | None = None
    """The asyncio.Task running the LLM iteration for this run (NativeAgent only)."""
    parent_session_id: str | None = None
    """Per-run parent session reference used by OpenCode/subagent event metadata."""
    prompt_task: asyncio.Task[Any] | None = None
    """Per-run ACP prompt task used for cancellation without shared instance state."""
    # session_id is overridden in run_stream() to match the caller's session_id
    # (not the auto-generated UUID). See Section 10.2.
```

#### `BaseAgent.__init__` — Replace Instance Variables with Registry

```python
# Remove:
self._current_stream_task: asyncio.Task[Any] | None = None  # Already removed in PR #17 review
self._active_run_ctx: AgentRunContext | None = None

# Add:
self._active_runs: dict[str, AgentRunContext] = {}

# Keep (Decision 2 — background run compatibility only):
self._cancelled: bool = False  # ONLY for run_in_background()/stop(); streaming runs use run_ctx.cancelled
```

#### `NativeAgent.__init__` — Remove Instance Variable

```python
# Remove:
self._iteration_task: asyncio.Task[Any] | None = None
```

### 8.4 Server-Side Change

```python
# session_routes.py — abort_session
@router.post("/{session_id}/abort")
async def abort_session(session_id: str, state: StateDep) -> bool:
    # IMPORTANT: do not call get_or_load_session() here.
    # It acquires state.agent_lock and deadlocks against an active stream.
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    await state.cancel_session_background_tasks(session_id)

    try:
        await state.agent.interrupt(session_id=session_id)  # ← Pass session_id
        await asyncio.sleep(0.1)
    except Exception:
        pass
    state.session_status[session_id] = SessionStatus(type="idle")
    await state.broadcast_event(SessionStatusEvent.create(session_id, SessionStatus(type="idle")))
    return True
```

If the server later chooses to wait for stream completion before broadcasting idle, that wait must be tracked **per session** (for example, `active_stream_tasks[session_id]`) and treated as a sequencing enhancement, not as a substitute for the deadlock fix above.

### 8.5 Security Considerations

1. **Session isolation**: Per-session `run_ctx` prevents one session from cancelling another's agent run
2. **Registry cleanup**: `finally` block in `run_stream()` guarantees removal, preventing stale entries
3. **No privilege escalation**: `abort_session()` only interrupts sessions already active in server memory; persistent-only sessions are not loaded during abort because there is nothing live to interrupt

## 9. Implementation Plan

### Phase 1: Core Registry (Day 1)

**Tasks**:
1. Add `iteration_task` field to `AgentRunContext`
2. Add `_active_runs: dict[str, AgentRunContext]` to `BaseAgent.__init__`
3. Remove `_active_run_ctx` from `BaseAgent.__init__`; retain `_cancelled` only for background-run compatibility
4. Remove `_iteration_task` from `NativeAgent.__init__`
5. Update `run_stream()` to register/unregister in `_active_runs` and override `run_ctx.session_id`
6. Update `interrupt()` with `session_id` parameter and resolution logic
7. Update `is_cancelled()` to check per-run context for streaming runs

**Files Modified**:
- `src/wolfharness/agents/context.py` (add `iteration_task` field)
- `src/wolfharness/agents/base_agent.py` (registry, interrupt routing)
- `src/wolfharness/agents/native_agent/agent.py` (use `run_ctx.iteration_task`)

**Deliverable**: Core registry works, existing tests updated

**Rollback**: `git revert` on single commit

### Phase 2: Subclass Migration (Day 1-2)

**Tasks**:
1. Update `ACPAgent._interrupt()` to use `run_ctx.current_task` / `run_ctx.prompt_task`
2. Update `AGUIAgent._interrupt()` to use `run_ctx.current_task`
3. Migrate all subclass streaming-time `self.session_id` / `self.parent_session_id` reads to `run_ctx`
4. Audit `ClaudeCodeAgent`, `CodexAgent`, and converter/helper call graphs for removed instance-state reads

**Files Modified**:
- `src/wolfharness/agents/acp_agent/acp_agent.py`
- `src/wolfharness/agents/agui_agent/agui_agent.py`
- Other subclasses as needed

**Deliverable**: All subclasses use per-run context

**Rollback**: `git revert` on subclass commits

### Phase 3: Server Integration (Day 2)

**Tasks**:
1. Update `abort_session()` to pass `session_id` to `interrupt()`
2. Remove `get_or_load_session()` from `abort_session()` to avoid `agent_lock` deadlock
3. Update `message_routes.py` CancelledError handler to not depend on `_cancelled` for streaming runs
4. Verify `send_message_async` background tasks interact correctly
5. Optional follow-up: if waiting for stream completion is implemented, use a per-session task map instead of a global task slot

**Files Modified**:
- `src/wolfharness_server/opencode_server/routes/session_routes.py`
- `src/wolfharness_server/opencode_server/routes/message_routes.py`

**Deliverable**: OpenCode abort flow uses session-scoped routing

**Rollback**: `git revert` on server commits

### Phase 4: Test Coverage (Day 2)

**Tasks**:
1. Update existing interrupt tests to use new API
2. Add test: `test_concurrent_sessions_interrupt_independently`
3. Add test: `test_interrupt_by_session_id`
4. Add test: `test_interrupt_without_session_id_single_run_fallback`
5. Run full test suite for regression check

**Files Modified**:
- `tests/agents/native_agent/test_interrupt.py`
- `tests/agents/test_concurrent_safety.py` (add new tests)
- `tests/servers/opencode_server/test_cancelled_message.py` (update if needed)

**Deliverable**: All tests passing, concurrent interrupt validated

**Rollback**: `git revert` on test commits

### Dependencies

- RFC-0021 implementation (completed) — `AgentRunContext` dataclass exists
- PR #17 (open) — will be rebased on top of this RFC's implementation

## 10. Review Findings (Metis + Oracle)

### 10.0 Post-Review Status

First review (Metis + Oracle): 7 issues identified, all resolved by 3 confirmed decisions.
Second review (Oracle final): 2 new critical issues + 1 high issue found. Updated below.

### 10.1 CRITICAL: `self.session_id` Instance Variable Overwrite

**Source**: Metis pre-planning analysis

In `base_agent.py:642-652`, `run_stream()` sets `self.session_id` as an instance variable:

```python
if self.session_id is None:
    self.session_id = session_id or generate_session_id()
elif session_id and self.session_id != session_id:
    self.session_id = session_id
```

Under concurrent sessions, this means:
- Session A calls `run_stream(session_id="ses_A")` → `self.session_id = "ses_A"`
- Session B calls `run_stream(session_id="ses_B")` → `self.session_id = "ses_B"` (overwrites A)
- Session A's `ChatMessage` creation at line 762-766 now uses `session_id="ses_B"`
- Session A's `RunStartedEvent` at `native_agent/agent.py:839` emits `session_id=self.session_id` → "ses_B"

The registry will correctly route `interrupt(session_id="ses_A")` to the right `run_ctx`, but the messages produced by session A will already be corrupted with session B's ID. This is the same class of bug as `_active_run_ctx` overwrite.

**Decision required**: Should this RFC also migrate `self.session_id` reads inside `run_stream()` / `_run_stream_once()` / `_stream_events()` to read from `run_ctx` instead? Or is this explicitly out of scope?

**Recommended**: In scope. Add `effective_session_id` to `AgentRunContext` (set from the `run_stream()` parameter) and update internal message creation to read from `run_ctx` instead of `self.session_id`. The `self.session_id` instance variable can remain for external API compatibility but should not be read during concurrent execution.

### 10.2 BUG: `session_id` Key Mismatch

**Source**: Both Metis and Oracle identified this independently

`AgentRunContext.session_id` (line 60 of `context.py`) is auto-generated via `uuid.uuid4().hex`. The `session_id` parameter passed to `run_stream()` is the OpenCode session ID (different format). The RFC's proposed code:

```python
effective_session_id = session_id or run_ctx.session_id  # BUG: run_ctx.session_id is a random UUID
```

If `session_id` is not passed to `run_stream()`, the fallback to `run_ctx.session_id` uses the wrong key — `abort_session()` will never find the entry because it passes the OpenCode session ID.

**Fix**: Always derive the registry key from the `session_id` parameter, not from `run_ctx.session_id`:

```python
effective_session_id = session_id or generate_session_id()
self._active_runs[effective_session_id] = run_ctx
# Also update run_ctx so internal code reads the correct ID
run_ctx.session_id = effective_session_id
```

### 10.3 HIGH: `run_in_background()` / `stop()` Break Without `self._cancelled`

**Source**: Metis pre-planning analysis

The RFC proposes removing `self._cancelled` entirely but doesn't account for:
- `run_in_background()` line 506: `self._cancelled = False` — resets for new background run
- `stop()` line 515: `self._cancelled = True` — signals cancellation to the background loop

When `stop()` is called with no active `run_stream()`, `_current_run_ctx_var.get()` returns `None` and `_active_runs` is empty. There's no `session_id` to look up. Without `self._cancelled`, `stop()` has no way to signal the background loop.

**Fix options**:
1. Keep `self._cancelled` as a backward-compat flag for background runs only (simplest, minimal risk)
2. Route background runs through `_active_runs` with a reserved key like `"__background__"` (cleaner, more work)
3. Merge `_background_run_ctx` into `_active_runs` (full unification)

**Recommended**: Option 1 for this RFC. Add a clear docstring: `self._cancelled` is ONLY for background run cancellation; all streaming runs use `run_ctx.cancelled`.

### 10.4 HIGH: `is_cancelled()` Caller Audit Required

**Source**: Oracle architecture review

Before removing `self._cancelled`, all callers of `is_cancelled()` must be audited. Callers outside `run_stream()` task scope (where ContextVar is `None`) would get `False` from the new `is_cancelled()` even when the agent was interrupted.

**Action**: `grep -r "is_cancelled" src/` before removing `self._cancelled`. For any callers outside `run_stream()` scope, provide a migration path (e.g., pass `run_ctx` explicitly, or use a deprecation warning).

### 10.5 MEDIUM: Fallback Heuristic Needs Liveness Check

**Source**: Both Metis and Oracle

The `len(self._active_runs) == 1` fallback should verify the candidate entry's task is still running:

```python
if effective is None and len(self._active_runs) == 1:
    candidate = next(iter(self._active_runs.values()))
    if candidate.current_task is None or not candidate.current_task.done():
        effective = candidate
```

This prevents routing an interrupt to a stale entry from a crashed run.

### 10.6 MEDIUM: ACP Session Has Its Own `_cancelled` Flag

**Source**: Metis pre-planning analysis

`acp_server/session.py` has its own `_cancelled` flag (lines 203, 370, 379, 390, 422, 476) that's checked independently of the agent's. If the agent's `_cancelled` is removed without updating ACP session's `cancel()` flow, the two systems could get out of sync.

**Action**: Update ACP `session.cancel()` to call `self.agent.interrupt(session_id=self.session_id)` instead of relying on agent-level `_cancelled`.

### 10.7 LOW: `self._event_queue` Instance Variable Still Exists

**Source**: Metis pre-planning analysis

`base_agent.py:221` still has `self._event_queue = asyncio.Queue()` as a fallback in `AgentContext.report_progress()`. This is an inconsistency (instance-level queue alongside per-run queue) but not a concurrent safety issue since it's only used when `run_ctx` is `None`.

**Action**: Document as known limitation. Can be removed in a future cleanup.

### 10.8 CRITICAL: `self.session_id` Reads in ALL Subclasses Not in Migration Checklist

**Source**: Oracle final review

Decision 1 says "internal reads within `run_stream()` / `_run_stream_once()` / `_stream_events()` use `run_ctx`." But the Appendix A migration checklist only lists `base_agent.py` and `native_agent/agent.py`. Grep reveals **34 reads of `self.session_id` across 8 files**:

| File | Lines | Context |
|------|-------|---------|
| `acp_agent/acp_agent.py` | 437, 439, 517, 554 | `_stream_events()` — event/message creation |
| `acp_agent/acp_converters.py` | 509, 572 | Converter — event construction |
| `agui_agent/agui_agent.py` | 225, 315, 325, 326, 353, 439, 467 | `_stream_events()` — SDK thread mapping, event creation |
| `claude_code_agent/claude_code_agent.py` | 359, 889, 891, 1202, 1275 | `_stream_events()` — event creation |
| `codex_agent/codex_agent.py` | 373 | `_stream_events()` — event creation |

Every one of these reads `self.session_id` during streaming execution and will produce wrong session IDs under concurrent access.

**Action**: Add all files to migration checklist. Apply rule: "All `self.session_id` reads within `_stream_events()` and its call graph → `run_ctx.session_id`." Since `run_ctx` is already passed to `_stream_events()` in all agent types, this is a straightforward search-and-replace.

### 10.9 CRITICAL: `self.parent_session_id` Has the Same Concurrent-Overwrite Bug

**Source**: Oracle final review

In `base_agent.py:644,652`:
```python
self.session_id = session_id or generate_session_id()
self.parent_session_id = parent_session_id  # ← Same overwrite bug
```

Read in `native_agent/agent.py:843` (RunStartedEvent), `agui_agent/agui_agent.py:326-353`, and other subclasses. Under concurrent sessions, `self.parent_session_id` gets overwritten by the most recent `run_stream()` call, corrupting event metadata.

**Action**: Add `parent_session_id: str | None = None` field to `AgentRunContext`. Set it at `run_stream()` entry alongside `effective_session_id`. Migrate all internal reads to `run_ctx.parent_session_id`. Follow same pattern as `self.session_id` migration (Section 10.8).

### 10.10 HIGH: `interrupt()` Must NOT Set `self._cancelled` — Not Specified

**Source**: Oracle final review

Current `interrupt()` at `base_agent.py:1088` sets `self._cancelled = True` unconditionally. After the RFC, `interrupt()` is session-scoped. Setting `self._cancelled = True` from `interrupt(session_id="ses_A")` would incorrectly signal cancellation to background runs and the `is_cancelled()` fallback chain.

**Decision**: `interrupt()` does NOT set `self._cancelled`. It only sets `run_ctx.cancelled = True` on the target context. Only `stop()` sets `self._cancelled = True`.

This is a **behavioral change** from the current code. An implementer who copies the current `interrupt()` pattern would reintroduce cross-session contamination.

**Action**: Explicitly state in Section 8.2 that `interrupt()` does not set `self._cancelled`. Add to migration checklist.

### 10.11 MEDIUM: ACP `_prompt_task` Instance Variable Also Per-Session

**Source**: Oracle final review

`acp_agent.py` has `self._prompt_task` which is cancelled in `_interrupt()` (line 597-599). Under concurrent sessions, this instance variable has the same overwrite problem as `_active_run_ctx`. It should either be tracked per-session (via `run_ctx`) or cancelled through the `_interrupt()` interface.

**Action**: Add `prompt_task: asyncio.Task[Any] | None = None` to `AgentRunContext` for ACP agent use. Migrate `self._prompt_task` references in `acp_agent.py` to `run_ctx.prompt_task`.

### 10.12 CRITICAL: OpenCode `abort_session()` Deadlocks Before `interrupt()`

**Source**: 2026-04-23 local debugging of OpenCode ESC abort flow

Current route flow:

```python
abort_session()
  -> get_or_load_session()      # acquires state.agent_lock
  -> await state.agent.interrupt()

_process_message_locked()
  -> async with state.agent_lock:
       async for event in agent.run_stream(...):
           ...
```

During an active stream, `_process_message_locked()` already holds `state.agent_lock`. `abort_session()` therefore blocks inside `get_or_load_session()` and never reaches `interrupt()`. This is not a registry bug; it is a routing-layer deadlock.

**Action**: `abort_session()` must validate session existence from in-memory state and call `interrupt(session_id=...)` without acquiring `agent_lock`.

### 10.13 MEDIUM: Waiting for Stream Completion Is Secondary, Not Primary

**Source**: Review of PR #23 (`fix: opencode cancal`)

Waiting for the active stream task after `interrupt()` may improve idle-event sequencing, but it does **not** fix the primary bug if `abort_session()` still acquires `agent_lock` first. Any such waiting logic must come after the deadlock fix and must be tracked per session, not via a single global `active_stream_task` field.

**Action**: Keep stream-completion waiting out of the core RFC requirements. If adopted later, specify a per-session task map and treat it as a follow-up server sequencing enhancement.

## 11. Remaining Open Questions

1. **`conversation` isolation**: This RFC does not address conversation history pollution between sessions. The correct long-term solution is per-session agent instances. This should likely become a follow-up RFC because it remains the most impactful concurrent-safety gap after interrupt routing is fixed.

2. **Subagent interrupt semantics**: When a subagent runs within a parent session, should interrupting the parent also explicitly traverse child session IDs, or should parent-agent `_interrupt()` implementations remain solely responsible? Current recommendation: register each run under its own `session_id`, and keep subagent cancellation inside the parent agent's implementation.

3. **Idle-event sequencing**: After the deadlock fix lands, do we also want `abort_session()` to wait for per-session stream completion before broadcasting idle, or is the existing stream-side cleanup sufficient? This is intentionally deferred because it is a sequencing refinement, not part of the core deadlock/routing fix.

## 12. Decision Record

**Status**: DRAFT (post-review round 2 — all critical issues addressed)

**Decision**: Option 1 (Session-ID Run Registry) — approved with modifications

**Date**: 2026-04-23

**Approvers**: yuchen.liu

**Confirmed Decisions**:
1. **`self.session_id` migration**: IN SCOPE — store `effective_session_id` in `run_ctx`, internal reads use `run_ctx` instead of `self.session_id`. Applies to ALL agent types (ACP, AGUI, Claude Code, Codex), not just NativeAgent. (Decision 1, expanded per Section 10.8)
2. **`self._cancelled` retention**: KEPT for background run compatibility only, with docstring; streaming runs use `run_ctx.cancelled`. `interrupt()` does NOT set `self._cancelled`. (Decision 2, clarified per Section 10.10)
3. **Fallback heuristic removal**: `len()==1` fallback removed; `interrupt()` without `run_ctx` or `session_id` is a no-op (Decision 3)
4. **Abort route deadlock fix**: `abort_session()` does NOT call `get_or_load_session()`; it validates in-memory session presence and calls `interrupt(session_id=...)` without acquiring `agent_lock`.

**Additional Decisions (from Oracle final review)**:
5. **`self.parent_session_id` migration**: IN SCOPE — add `parent_session_id` to `AgentRunContext`, migrate internal reads to `run_ctx` (Section 10.9)
6. **ACP `_prompt_task` migration**: IN SCOPE — add `prompt_task` to `AgentRunContext` for ACP agent per-session tracking (Section 10.11)

**Key Discussion Points**:
- Option 1 selected for alignment with RFC-0021 and minimal API surface change
- `conversation` isolation deferred to future work
- Review identified `self.session_id` as additional concurrent overwrite bug (Section 10.1)
- Review identified `session_id` key mismatch bug in proposed code (Section 10.2)
- `self._cancelled` retained for background run compatibility (Section 10.3)
- Local debugging identified the OpenCode abort deadlock in `abort_session()` / `get_or_load_session()` (Section 10.12)

**Conditions on Implementation**:
- [ ] All existing tests pass
- [ ] New concurrent interrupt tests demonstrate session-scoped isolation
- [ ] No remaining instance-level mutable streaming state in `BaseAgent` or `NativeAgent` (except `self._cancelled` for background runs)
- [ ] `session_id` key mismatch bug fixed: `run_ctx.session_id` overridden with `effective_session_id` from `run_stream()` parameter
- [ ] `self.session_id` internal reads migrated to `run_ctx` in ALL agent types (ACP, AGUI, Claude Code, Codex) — not just NativeAgent
- [ ] `self.parent_session_id` migrated to `AgentRunContext` alongside `session_id`
- [ ] `interrupt()` does NOT set `self._cancelled` — only `stop()` does
- [ ] ACP `self._prompt_task` migrated to `run_ctx.prompt_task`
- [ ] `abort_session()` no longer acquires `agent_lock` through `get_or_load_session()` before calling `interrupt()`
- [ ] If idle sequencing waits for stream completion, the wait is tracked per session rather than via a global active-task slot
- [ ] All `interrupt()` callers audited: must pass `run_ctx` or `session_id` (no fallback heuristic)

---

## Appendix A: Migration Checklist

### Instance Variables to Remove

| Variable | File | Replacement |
|----------|------|-------------|
| `self._current_stream_task` | `base_agent.py` | Already removed in PR #17 review fix |
| `self._active_run_ctx` | `base_agent.py` | `self._active_runs[session_id]` |
| `self._iteration_task` | `native_agent/agent.py` | `run_ctx.iteration_task` |

### Instance Variables to Keep (with Scope Limitation)

| Variable | File | Scope | Notes |
|----------|------|-------|-------|
| `self._cancelled` | `base_agent.py` | Background runs only | ONLY set by `stop()`; `interrupt()` does NOT set it |
| `self.session_id` | `base_agent.py` | External API only | Internal reads migrated to `run_ctx.session_id`; write reflects most recent run |
| `self.parent_session_id` | `base_agent.py` | External API only | Internal reads migrated to `run_ctx.parent_session_id` |

### `self.session_id` Reads to Migrate (ALL Agent Types)

| File | Lines | Migration |
|------|-------|-----------|
| `base_agent.py` | 765, 799, 847 | `self.session_id` → `run_ctx.session_id` |
| `native_agent/agent.py` | 838, 840, 908, 919 | `self.session_id` → `run_ctx.session_id` |
| `acp_agent/acp_agent.py` | 437, 439, 517, 554 | `self.session_id` → `run_ctx.session_id` |
| `acp_agent/acp_converters.py` | 509, 572 | `self.session_id` → `run_ctx.session_id` |
| `agui_agent/agui_agent.py` | 225, 315, 325, 326, 353, 439, 467 | `self.session_id` → `run_ctx.session_id` |
| `claude_code_agent/claude_code_agent.py` | 359, 889, 891, 1202, 1275 | `self.session_id` → `run_ctx.session_id` |
| `codex_agent/codex_agent.py` | 373 | `self.session_id` → `run_ctx.session_id` |

### `self.parent_session_id` Reads to Migrate

| File | Lines | Migration |
|------|-------|-----------|
| `native_agent/agent.py` | 843 | `parent_session_id` → `run_ctx.parent_session_id` |
| Other subclasses | grep for `parent_session_id` | Same pattern |

### `AgentRunContext` New Fields

| Field | Type | Purpose |
|-------|------|---------|
| `iteration_task` | `asyncio.Task[Any] \| None = None` | Per-run LLM iteration task (NativeAgent) |
| `parent_session_id` | `str \| None = None` | Per-run parent session reference |
| `prompt_task` | `asyncio.Task[Any] \| None = None` | Per-run prompt task (ACPAgent) |

### Call Sites to Update

| Call Site | Current | Updated |
|-----------|---------|---------|
| `abort_session()` | `session = await get_or_load_session(...); await state.agent.interrupt()` | `if session_id not in state.sessions: raise 404; await state.agent.interrupt(session_id=session_id)` |
| `ACP session.cancel()` | `await self.agent.interrupt()` | `await self.agent.interrupt(session_id=self.session_id)` |
| `ACP._interrupt()` | `self._active_run_ctx.current_task` / `self._prompt_task` | `run_ctx.current_task` if `run_ctx` / `run_ctx.prompt_task` |
| `AGUI._interrupt()` | `self._active_run_ctx.current_task` | `run_ctx.current_task` if `run_ctx` |
| `interrupt()` body | `self._cancelled = True` | Do NOT set `self._cancelled` — only set `run_ctx.cancelled = True` |
| `is_cancelled()` | `return self._cancelled` | `return run_ctx.cancelled` with ContextVar fallback, then `self._cancelled` for background |
| `_stream_events()` | `self._iteration_task = task` | `run_ctx.iteration_task = task` |

## Appendix B: Test Plan

### New Tests

```python
# test_concurrent_safety.py — additions

async def test_concurrent_sessions_interrupt_independently():
    """Two concurrent sessions; interrupting one must not affect the other."""
    agent = Agent(name="test", model=SlowTestModel(pre_stream_delay=2.0))

    async def run(session_id: str) -> list:
        events = []
        async for event in agent.run_stream("prompt", session_id=session_id):
            events.append(event)
        return events

    task_a = asyncio.create_task(run("ses_A"))
    task_b = asyncio.create_task(run("ses_B"))
    await asyncio.sleep(0.2)  # Both streams started

    # Interrupt only session A
    await agent.interrupt(session_id="ses_A")
    await asyncio.sleep(0.3)

    # Session A should be cancelled
    assert agent._active_runs.get("ses_A") is None or agent._active_runs["ses_A"].cancelled

    # Session B should still be running
    assert "ses_B" in agent._active_runs
    assert not agent._active_runs["ses_B"].cancelled

    # Clean up
    await agent.interrupt(session_id="ses_B")
    await asyncio.gather(task_a, task_b, return_exceptions=True)


async def test_interrupt_by_session_id():
    """interrupt(session_id=...) routes to the correct run_ctx."""
    agent = Agent(name="test", model=SlowTestModel(pre_stream_delay=1.0))

    async def run(session_id: str):
        async for event in agent.run_stream("prompt", session_id=session_id):
            pass

    task = asyncio.create_task(run("ses_X"))
    await asyncio.sleep(0.1)

    await agent.interrupt(session_id="ses_X")

    # Verify run_ctx was found and cancelled
    assert "ses_X" not in agent._active_runs  # cleaned up by finally

    await asyncio.gather(task, return_exceptions=True)


async def test_abort_session_does_not_wait_on_agent_lock():
    """OpenCode abort must reach interrupt() while a stream still holds agent_lock."""
    ...
```

### Existing Tests to Update

| Test | Change |
|------|--------|
| `test_interrupt_without_run_ctx_*` | Pass `session_id` explicitly or update expectations to no-op |
| `test_subsequent_run_after_interrupt` | Remove `fast_agent._cancelled = False` manual reset |
| `test_interrupt_then_run_stream` | Verify `_cancelled` flag no longer needed |

---

**End of RFC-0023**
