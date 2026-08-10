---
rfc_id: RFC-0034
title: "Per-Session TodoTracker Isolation"
status: DRAFT
author: yuchen.liu
reviewers:
  - name: Oracle
    status: pending
  - name: Metis
    status: pending
created: 2026-05-26
last_updated: 2026-05-26
decision_date:
related_prds: []
related_rfcs:
  - RFC-0026 (Per-Session Agent Instances for OpenCode Server)
  - RFC-0031 (ACP Per-Session Agent Isolation)
  - RFC-0021 (Agent Concurrent Execution Safety)    
---

# RFC-0034: Per-Session TodoTracker Isolation

## Overview

This RFC proposes migrating the `TodoTracker` from a single global instance on `AgentPool` to per-session isolated instances. Currently, `AgentPool.todos` is shared across all sessions and protocols, causing cross-session todo contamination, inability to clean up per-session todos on session deletion, and wasteful broadcast of todo updates to all sessions. This RFC adapts the proven per-session isolation pattern established in [RFC-0026](../implemented/RFC-0026-per-session-agent-isolation.md) and [RFC-0031](../implemented/RFC-0031-acp-per-session-agent-isolation.md) to the `TodoTracker` domain.

## Table of Contents

- Background & Context
- [Problem Statement](#problem-statement)
- Goals & Non-Goals
- [Evaluation Criteria](#evaluation-criteria)
- [Options Analysis](#options-analysis)
- [Recommendation](#recommendation)
- [Technical Design](#technical-design)
- Implementation Plan
- [Open Questions](#open-questions)
- Decision Record
- [References](#references)

---

## Background & Context

### Current State

The todo/plan system has three layers:

1. **`TodoTracker`** (`utils/todos.py`): A simple dataclass holding a flat list of `TodoEntry` items with an `on_change` callback. Created as `AgentPool.todos = TodoTracker()` — one instance per pool.

2. **`PlanProvider`** (`resource_providers/plan_provider.py`): Provides plan management tools (`get_plan`, `set_plan`, `add_plan_entry`, etc.) to agents. Resolves the tracker via `agent_ctx.pool.todos` — the global instance.

3. **Protocol adapters**:
   - **OpenCode server**: `on_todo_change()` callback broadcasts `TodoUpdatedEvent` to **all** active sessions. The `get_session_todos()` route reads from `state.pool.todos` — the global tracker.
   - **ACP server**: `ACPEventConverter` converts `PlanUpdateEvent` → `AgentPlanUpdate` (protocol-level only). The `client_handler.py` has an unresolved TODO: *"AgentPlanUpdate handling is complex... Options: 1. Update pool.todos - requires merging with existing todos, 2. Pass through to UI, 3. Switch to agent-owned todos instead of pool-owned"*.

4. **Other consumers**: `CodexAgent` syncs plan updates to `self.agent_pool.todos.replace_all()`. `XenoPlanProvider` (in xeno-agent) extends the base provider with custom fields.

### Data Flow (Current)

```
Agent Tool Call
      │
      ▼
PlanProvider._get_tracker(agent_ctx) → agent_ctx.pool.todos (GLOBAL)
      │
      ├── tracker.add() / tracker.update() / tracker.replace_all()
      │
      ▼
TodoTracker._notify_change()
      │
      ▼
OpenCode: on_todo_change(tracker) → broadcast TodoUpdatedEvent to ALL sessions
ACP: PlanUpdateEvent → AgentPlanUpdate (per-session, but reads from global tracker)
```

### Historical Context

The global `TodoTracker` was introduced as a simple pool-level store for agent plans. At the time, the system had a single session model and no concurrent multi-session support. RFC-0026 and RFC-0031 introduced per-session agent isolation but did not address the shared todo state — which remains a contamination vector.

### Glossary

| Term | Definition |
|------|------------|
| `TodoTracker` | A dataclass holding a list of `TodoEntry` items with change notification support |
| `AgentPool.todos` | The single global `TodoTracker` instance shared across all sessions |
| `PlanProvider` | Resource provider that exposes plan management tools to agents |
| `PlanUpdateEvent` | Native event emitted when the plan changes, consumed by protocol adapters |
| `TodoUpdatedEvent` | OpenCode SSE event carrying the full todo list to clients |
| `AgentPlanUpdate` | ACP protocol notification carrying plan entries to clients |
| Per-session tracker | A dedicated `TodoTracker` instance created for a single session |

---

## Problem Statement

### The Problem

When multiple sessions exist (ACP, OpenCode, or mixed), they all share a single `TodoTracker` via `AgentPool.todos`. This causes:

1. **Cross-session todo contamination**: Session A's plan entries appear in Session B's `get_plan` results. If two users work on different tasks simultaneously, they see each other's todo items.

2. **No per-session cleanup path**: When a session is deleted, `state.todos.pop(session_id, None)` removes the in-memory cache (OpenCode), but the actual `TodoTracker` entries persist globally because `state.pool.todos` is not cleaned up.

3. **Wasteful broadcast**: `on_todo_change()` iterates all active sessions and sends `TodoUpdatedEvent` to every session, even though only one session's todos changed.

4. **ACP `AgentPlanUpdate` unresolved design**: The `client_handler.py` TODO at line 192-197 explicitly acknowledges: *"Update pool.todos - requires merging with existing todos"*. The current workaround (falling through to stream data) means ACP plan updates from remote agents are not centrally managed.

5. **Session resume/restore todo loss**: When a session is loaded from storage, its todos are not restored because `pool.todos` contains the todos from whichever session last wrote to it.

### Evidence

- `pool.py:182`: `self.todos = TodoTracker()` — single global instance
- `plan_provider.py:62-64`: `_get_tracker()` returns `agent_ctx.pool.todos` — global
- `server.py:146-158`: `on_todo_change()` broadcasts to ALL sessions
- `session_routes.py:1059-1074`: `get_session_todos()` reads from `state.pool.todos` — global
- `session_routes.py:795`: `state.todos.pop(session_id, None)` — removes cache but not tracker entries
- `client_handler.py:192-197`: Unresolved TODO about `AgentPlanUpdate` handling
- `codex_agent.py:427`: `self.agent_pool.todos.replace_all()` — overwrites global tracker

### Impact of Inaction

- **Risk**: Teams sharing an wolfharness server see each other's task lists. This is a functional correctness issue, not just a UX concern.
- **Risk**: Session restore (`load_session`, `resume_session`) cannot reconstruct the user's task list because todos are not persisted per-session.
- **Cost**: ACP `AgentPlanUpdate` integration remains incomplete, blocking proper plan management for ACP-connected clients.
- **Cost**: Memory waste — deleted sessions' todo entries accumulate in the global tracker with no eviction.

---

## Goals & Non-Goals

### Goals (In Scope)

1. Each session gets its own isolated `TodoTracker` instance
2. Protocol adapters (OpenCode, ACP) emit todo events only to the relevant session
3. Session deletion cleans up the session's `TodoTracker`
4. `PlanProvider` resolves the correct per-session tracker through `AgentContext`
5. Resolve the ACP `AgentPlanUpdate` TODO — properly sync per-session plan updates
6. Support session resume/restore of todo state
7. Maintain backward compatibility for single-session usage

### Non-Goals (Out of Scope)

1. **Not**: Persisting todos to storage (can be added independently)
2. **Not**: Changing the `TodoTracker` API surface (it works well internally)
3. **Not**: Implementing session cleanup/eviction policies
4. **Not**: Adding todo persistence in ACP protocol (beyond existing `AgentPlanUpdate`)
5. **Not**: Refactoring `BaseAgent` internals (that's RFC-0024)

### Success Criteria

- [ ] Two concurrent sessions can maintain independent todo lists without contamination
- [ ] Session deletion removes the session's `TodoTracker` and its entries
- [ ] `on_todo_change` only broadcasts to the session that owns the changed tracker
- [ ] ACP `AgentPlanUpdate` from remote agents updates the correct per-session tracker
- [ ] All existing tests pass without modification (backward compatibility)
- [ ] `PlanProvider` tools work correctly with per-session tracker resolution

---

## Evaluation Criteria

| Criterion | Weight | Description | Minimum Threshold |
|-----------|--------|-------------|-------------------|
| Isolation Completeness | Critical | Sessions fully isolated, no cross-contamination | Must pass multi-session test |
| Backward Compatibility | High | Single-session usage unchanged | All existing tests pass |
| Minimality | High | Smallest change that solves the problem | ≤ 8 files modified |
| Protocol Coverage | High | All protocols (OpenCode, ACP, CLI) benefit | Both OpenCode and ACP covered |
| ACP PlanUpdate Resolution | Medium | Resolves the long-standing client_handler TODO | AgentPlanUpdate updates per-session tracker |
| Testability | Medium | Easy to verify per-session isolation | Unit-testable without integration setup |

---

## Options Analysis

### Option 1: Per-Session TodoTracker Registry in AgentPool (Recommended)

Add a `TodoTrackerRegistry` to `AgentPool` that manages per-session `TodoTracker` instances. `PlanProvider` resolves the tracker via `agent_ctx.pool.get_session_todos(session_id)`. Protocol servers register `on_change` callbacks on session-specific trackers.

**Advantages**:
- Centralized management — all protocols benefit from a single registry
- Clean lifecycle — registry handles creation, lookup, and cleanup
- Minimal tool changes — `PlanProvider._get_tracker()` needs a one-line change
- Backward compatible — global `pool.todos` remains as fallback for sessionless contexts
- Natural fit — `AgentPool` already owns the global `TodoTracker`

**Disadvantages**:
- Adds session awareness to `AgentPool` (conceptually session-agnostic)
- Each protocol server must wire up per-session `on_change` callbacks
- Need to thread `session_id` through `AgentContext` to `PlanProvider`

**Evaluation Against Criteria**:

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Isolation Completeness | ✅ Excellent | Each session has independent tracker |
| Backward Compatibility | ✅ Good | `pool.todos` remains as fallback |
| Minimality | ✅ Good | ~150 lines across 6 files |
| Protocol Coverage | ✅ Excellent | Both protocols benefit from centralized registry |
| ACP PlanUpdate Resolution | ✅ Good | Per-session tracker enables proper handling |
| Testability | ✅ Good | Registry is a simple dict, easy to test |

**Effort Estimate**:
- Complexity: Medium
- Resources: 1 engineer, 2–3 days
- Dependencies: None (self-contained)

**Risk Assessment**:

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Session ID not available in `AgentContext` | Medium | High | Add `session_id` to `AgentContext` or resolve from agent |
| Protocol servers forget to register/cleanup callbacks | Low | Medium | Registry auto-creates trackers; add lifecycle docs |
| Memory leak from uncleaned trackers | Low | Medium | Protocol servers must call `remove_session_todos()` on session close |

---

### Option 2: Per-Session TodoTracker in Protocol Servers

Each protocol server (OpenCode, ACP) manages its own `dict[str, TodoTracker]`. Protocol servers create and clean up trackers as part of session lifecycle. `PlanProvider` resolves the tracker via a protocol-level hook.

**Advantages**:
- Session awareness stays in the protocol layer where it belongs
- Each protocol can customize todo handling independently
- No changes to `AgentPool` (stays session-agnostic)

**Disadvantages**:
- Duplicated logic — each protocol must implement the same tracker lifecycle
- `PlanProvider` has no way to access protocol-level tracker dicts (tools run inside agents, not protocol servers)
- ACP server has no existing session-scoped state management for todos
- Breaking change — tools cannot resolve per-session tracker without protocol-level injection

**Evaluation Against Criteria**:

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Isolation Completeness | ⚠️ Moderate | Requires each protocol to implement correctly |
| Backward Compatibility | ⚠️ Moderate | `PlanProvider` needs injection mechanism |
| Minimality | ❌ Poor | Duplicated logic per protocol + injection mechanism |
| Protocol Coverage | ❌ Poor | Each protocol must implement independently |
| ACP PlanUpdate Resolution | ⚠️ Moderate | Possible but requires ACP-specific implementation |
| Testability | ⚠️ Moderate | Must test per-protocol integration |

**Effort Estimate**:
- Complexity: High
- Resources: 1 engineer, 4–5 days
- Dependencies: Requires injection mechanism design

---

### Option 3: Agent-Owned TodoTracker

Attach a `TodoTracker` to each `BaseAgent` instance. Since RFC-0026 and RFC-0031 established per-session agents, each session's agent would naturally have its own tracker. `PlanProvider` resolves via `agent_ctx.agent.todos` instead of `agent_ctx.pool.todos`.

**Advantages**:
- Natural ownership — tracker follows agent lifecycle automatically
- Automatic cleanup — when per-session agent is cleaned up, its tracker dies too
- No registry needed — agent IS the tracker container
- Works for all protocols — per-session agents exist in both OpenCode and ACP

**Disadvantages**:
- Requires adding `todos` field to `BaseAgent` — changes core agent model
- Shared agents (single-session mode) would still share a tracker — no improvement for CLI/headless mode
- `BaseAgent` is already complex — adding more state increases surface area
- CodexAgent and ACPAgent use `pool.todos` directly — need separate migration
- Does not help for non-native agents (ACP agent type falls back to shared agent)

**Evaluation Against Criteria**:

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Isolation Completeness | ⚠️ Moderate | Only works for per-session agents |
| Backward Compatibility | ⚠️ Moderate | Changes `BaseAgent` public interface |
| Minimality | ❌ Poor | Touches core agent model |
| Protocol Coverage | ⚠️ Moderate | Depends on per-session agent adoption |
| ACP PlanUpdate Resolution | ⚠️ Moderate | Needs agent-level todo bridge |
| Testability | ⚠️ Moderate | Must test agent lifecycle interaction |

**Effort Estimate**:
- Complexity: High
- Resources: 1 engineer, 3–4 days
- Dependencies: Requires per-session agents for all protocols

---

### Options Comparison Summary

| Criterion | Option 1: Pool Registry | Option 2: Protocol Servers | Option 3: Agent-Owned |
|-----------|------------------------|---------------------------|----------------------|
| Isolation Completeness | ✅ Excellent | ⚠️ Moderate | ⚠️ Moderate |
| Backward Compatibility | ✅ Good | ⚠️ Moderate | ⚠️ Moderate |
| Minimality | ✅ Good | ❌ Poor | ❌ Poor |
| Protocol Coverage | ✅ Excellent | ❌ Poor | ⚠️ Moderate |
| ACP PlanUpdate Resolution | ✅ Good | ⚠️ Moderate | ⚠️ Moderate |
| Testability | ✅ Good | ⚠️ Moderate | ⚠️ Moderate |
| **Overall** | **Recommended** | Rejected | Deferred |

---

## Recommendation

**Option 1: Per-Session TodoTracker Registry in AgentPool.**

Option 2 is impractical because `PlanProvider` runs inside agent tool execution context and has no access to protocol-level state. The injection mechanism required would be more complex than the registry itself. Option 3 is architecturally sound but increases `BaseAgent` surface area and depends on per-session agent adoption — it should be considered as a future refinement if `BaseAgent` becomes stateless (RFC-0024).

Option 1 is the minimal, centralized solution. It follows the same pattern as `AgentPool._session_agents` (per-session registry) and resolves the long-standing ACP `AgentPlanUpdate` TODO.

### Accepted Trade-offs

1. **Session awareness in `AgentPool`**: Acceptable because `AgentPool` already has session-adjacent state (`SessionManager`, `StorageManager`). The registry is a lightweight dict that does not change the pool's core semantics.
2. **Protocol servers must wire callbacks**: Acceptable because the wiring is simple (one callback per session) and follows the existing `on_todo_change` pattern.
3. **`AgentContext.session_id` resolution**: Requires threading session ID through the context chain. Acceptable because `session_id` is already available on `BaseAgent` and `AgentContext` has agent access.

### Conditions

- `AgentContext` must expose `session_id` (either directly or via agent reference)
- Protocol servers must call `pool.remove_session_todos(session_id)` on session close/deletion
- `pool.todos` (global) must remain as fallback for sessionless contexts (CLI, tests)

---

## Technical Design

### Architecture Overview

```
BEFORE (Current):
┌─────────────────────────────────────────────┐
│              AgentPool                        │
│  todos: TodoTracker  ← GLOBAL, SHARED       │
└─────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
┌──────────────┐    ┌──────────────┐
│ Session A    │    │ Session B    │
│ reads pool   │    │ reads pool   │
│ .todos (SAME)│    │ .todos (SAME)│
└──────────────┘    └──────────────┘
  (contaminated)      (contaminated)

AFTER (Per-Session Registry):
┌─────────────────────────────────────────────┐
│              AgentPool                        │
│  todos: TodoTracker  ← GLOBAL FALLBACK      │
│  _session_todos: dict[str, TodoTracker]      │
│    ├─ "sess-A" → TodoTracker(instance A)    │
│    └─ "sess-B" → TodoTracker(instance B)    │
│  _session_todo_locks: dict[str, Lock]        │
└─────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
┌──────────────┐    ┌──────────────┐
│ Session A    │    │ Session B    │
│ reads own    │    │ reads own    │
│ tracker      │    │ tracker      │
└──────────────┘    └──────────────┘
  (isolated)          (isolated)
```

### Key Components

#### 1. `AgentPool` — Per-Session TodoTracker Registry (NEW)

**Add to `AgentPool` (`pool.py`)**:

```python
class AgentPool[TPoolDeps = None](BaseRegistry[NodeName, MessageNode[Any, Any]]):
    def __init__(self, ...):
        # ... existing init ...
        self.todos = TodoTracker()  # Global fallback for sessionless contexts
        # NEW: Per-session todo tracker registry
        self._session_todos: dict[str, TodoTracker] = {}
        self._session_todo_locks: dict[str, asyncio.Lock] = {}

    def get_session_todos(self, session_id: str | None) -> TodoTracker:
        """Get the TodoTracker for a specific session.

        Returns the per-session tracker if one exists, otherwise
        returns the global ``self.todos`` as fallback.

        Args:
            session_id: The session ID to get the tracker for.
                If None, returns the global tracker.

        Returns:
            TodoTracker for the given session, or the global tracker.
        """
        if session_id is None or session_id not in self._session_todos:
            return self.todos
        return self._session_todos[session_id]

    def get_or_create_session_todos(self, session_id: str) -> TodoTracker:
        """Get or create a per-session TodoTracker.

        Uses double-checked locking for concurrent access safety.

        Args:
            session_id: The session ID to get or create a tracker for.

        Returns:
            A dedicated TodoTracker for the given session.
        """
        if session_id in self._session_todos:
            return self._session_todos[session_id]

        if session_id not in self._session_todo_locks:
            self._session_todo_locks[session_id] = asyncio.Lock()

        # Note: This is a synchronous method. For truly concurrent creation,
        # use the async version below. In practice, tracker creation is
        # triggered by protocol server session creation which is already
        # serialized per session.
        if session_id not in self._session_todos:
            tracker = TodoTracker()
            self._session_todos[session_id] = tracker
        return self._session_todos[session_id]

    async def aget_or_create_session_todos(self, session_id: str) -> TodoTracker:
        """Async version of get_or_create_session_todos with proper locking.

        Args:
            session_id: The session ID to get or create a tracker for.

        Returns:
            A dedicated TodoTracker for the given session.
        """
        if session_id in self._session_todos:
            return self._session_todos[session_id]

        if session_id not in self._session_todo_locks:
            self._session_todo_locks[session_id] = asyncio.Lock()

        async with self._session_todo_locks[session_id]:
            if session_id in self._session_todos:
                return self._session_todos[session_id]
            tracker = TodoTracker()
            self._session_todos[session_id] = tracker
            return tracker

    def remove_session_todos(self, session_id: str) -> None:
        """Remove a session's TodoTracker.

        Called by protocol servers when a session is closed or deleted.

        Args:
            session_id: The session ID whose tracker should be removed.
        """
        self._session_todos.pop(session_id, None)
        self._session_todo_locks.pop(session_id, None)

    def cleanup_all_session_todos(self) -> None:
        """Remove all per-session TodoTrackers.

        Called during pool swap or shutdown.
        """
        self._session_todos.clear()
        self._session_todo_locks.clear()
```

#### 2. `AgentContext` — Session ID Resolution

**Add `session_id` property to `AgentContext` (`agents/context.py`)**:

```python
@dataclass
class AgentContext:
    # ... existing fields ...

    @property
    def session_id(self) -> str | None:
        """Get the session ID from the agent."""
        return self.agent.session_id if self.agent else None
```

This allows `PlanProvider` and other tools to resolve the per-session tracker without changing the `AgentContext` constructor.

#### 3. `PlanProvider` — Per-Session Tracker Resolution

**Modify `_get_tracker()` (`resource_providers/plan_provider.py`)**:

```python
def _get_tracker(self, agent_ctx: AgentContext) -> TodoTracker | None:
    """Get the TodoTracker for the current session."""
    if agent_ctx.pool is not None:
        return agent_ctx.pool.get_session_todos(agent_ctx.session_id)
    return None
```

**This is a one-line change**: `agent_ctx.pool.todos` → `agent_ctx.pool.get_session_todos(agent_ctx.session_id)`. When `session_id` is None (sessionless context like CLI), the global `pool.todos` is returned — preserving backward compatibility.

#### 4. OpenCode Server — Per-Session Callback Wiring

**Modify `create_app()` (`opencode_server/server.py`)**:

```python
# BEFORE: Global callback
async def on_todo_change(tracker: TodoTracker) -> None:
    todos = [
        Todo(id=e.id, content=e.content, status=e.status, priority=e.priority)
        for e in tracker.entries
    ]
    for session_id in state.sessions:
        event = TodoUpdatedEvent.create(session_id=session_id, todos=todos)
        await state.broadcast_event(event)

state.pool.todos.on_change = on_todo_change

# AFTER: Per-session callback registration during session creation
# Remove the global on_todo_change callback entirely.
# Instead, register per-session callbacks in ensure_session / _create_and_persist_session.
```

**Add per-session callback in `ServerState.ensure_session()` (`opencode_server/state.py`)**:

```python
async def _create_and_persist_session(self, session_id, parent_id):
    # ... existing session creation logic ...

    # NEW: Create per-session TodoTracker and wire callback
    tracker = await self.pool.aget_or_create_session_todos(session_id)

    async def on_session_todo_change(t: TodoTracker) -> None:
        """Broadcast todo updates to this session only."""
        from wolfharness_server.opencode_server.models.events import Todo, TodoUpdatedEvent
        todos = [
            Todo(id=e.id, content=e.content, status=e.status, priority=e.priority)
            for e in t.entries
        ]
        event = TodoUpdatedEvent.create(session_id=session_id, todos=todos)
        await self.broadcast_event(event)

    tracker.on_change = on_session_todo_change

    # ... rest of session creation ...
```

**Update `get_session_todos()` route (`session_routes.py`)**:

```python
# BEFORE: Reads from global tracker
tracker = state.pool.todos

# AFTER: Reads from per-session tracker
tracker = state.pool.get_session_todos(session_id)
```

**Update session deletion (`session_routes.py`)**:

```python
# Add after state.todos.pop(session_id, None):
state.pool.remove_session_todos(session_id)
```

**Update `ensure_runtime_session_state()` (`state.py`)**:

```python
# The state.todos dict is no longer needed for todo storage
# (it was a cache that never synced with pool.todos anyway).
# Keep it for backward compatibility but mark as deprecated.
```

#### 5. ACP Server — AgentPlanUpdate Resolution

**Modify `ACPSession.__post_init__()` or session creation to create per-session tracker**:

In `ACPSessionManager.create_session()` or `AgentPoolACPAgent.new_session()`:

```python
# NEW: Create per-session TodoTracker for this ACP session
tracker = await self.agent_pool.aget_or_create_session_todos(session_id)

# Wire the tracker's on_change to emit AgentPlanUpdate
async def on_acp_todo_change(t: TodoTracker) -> None:
    """Emit ACP AgentPlanUpdate when todos change."""
    from acp.schema import AgentPlanUpdate, PlanEntry as ACPPlanEntry
    from wolfharness.utils.todos import PlanEntry

    entries = [
        ACPPlanEntry(content=e.content, priority=e.priority, status=e.status)
        for e in t.entries
    ]
    # Schedule the notification (ACP notifications are async)
    # This integrates with the session's update notification system

tracker.on_change = on_acp_todo_change
```

**Resolve `client_handler.py` TODO**:

```python
# BEFORE (line 192-197):
# TODO: AgentPlanUpdate handling is complex and needs design work.

# AFTER:
case AgentPlanUpdate(entries=entries):
    # Update per-session tracker with remote plan entries
    tracker = self._agent.agent_pool.get_session_todos(self._agent.session_id)
    if tracker is not None:
        from wolfharness.utils.todos import PlanEntry
        native_entries = [
            PlanEntry(content=e.content, priority=e.priority, status=e.status)
            for e in entries
        ]
        tracker.replace_all(native_entries)
    self._update_event.set()
    return
```

**Update `ACPEventConverter` to read from per-session tracker**:

The `PlanUpdateEvent` → `AgentPlanUpdate` conversion already works correctly because `PlanProvider._emit_plan_update()` emits events that the converter processes. No change needed in the event converter.

**Update `ACPSession.close()` and `ACPSessionManager.close_session()`**:

```python
# In close_session():
if session.acp_agent:
    await session.acp_agent.remove_session_agent(session_id)
# NEW: Clean up per-session todo tracker
session.agent_pool.remove_session_todos(session_id)
```

#### 6. CodexAgent — Per-Session Tracker Sync

**Modify plan sync in `codex_agent.py`**:

```python
# BEFORE:
if self.agent_pool and self.agent_pool.todos:
    self.agent_pool.todos.replace_all(...)

# AFTER:
tracker = self.agent_pool.get_session_todos(self.session_id)
if tracker is not None:
    tracker.replace_all(...)
```

#### 7. Cleanup on Pool Swap

**Modify `swap_pool()` / `cleanup()` in relevant protocol handlers**:

```python
# In OpenCode server lifespan shutdown:
state.pool.cleanup_all_session_todos()

# In ACP AgentPoolACPAgent.swap_pool():
await self.cleanup_all_session_agents()
self.agent_pool.cleanup_all_session_todos()
```

---

## Implementation Plan

### Phase 1: Core Registry & Context Threading

**Scope**: Add per-session tracker registry to `AgentPool`, add `session_id` to `AgentContext`, update `PlanProvider`.

**Files**:

| File | Changes |
|------|---------|
| `pool.py` | Add `_session_todos`, `_session_todo_locks`, `get_session_todos()`, `get_or_create_session_todos()`, `aget_or_create_session_todos()`, `remove_session_todos()`, `cleanup_all_session_todos()` |
| `agents/context.py` | Add `session_id` property |
| `resource_providers/plan_provider.py` | Change `_get_tracker()` to use `pool.get_session_todos(agent_ctx.session_id)` |

**Duration**: 0.5 day

### Phase 2: OpenCode Server Integration

**Scope**: Wire per-session callbacks, update routes, remove global callback.

**Files**:

| File | Changes |
|------|---------|
| `opencode_server/server.py` | Remove global `on_todo_change` callback |
| `opencode_server/state.py` | Add per-session tracker creation + callback wiring in `_create_and_persist_session()` and session-store-hit path |
| `opencode_server/routes/session_routes.py` | Update `get_session_todos()` to read from per-session tracker, add `remove_session_todos()` on session deletion |

**Duration**: 0.5 day

### Phase 3: ACP Server Integration

**Scope**: Create per-session tracker for ACP sessions, resolve `AgentPlanUpdate` TODO, add cleanup.

**Files**:

| File | Changes |
|------|---------|
| `acp_server/session.py` or `acp_server/session_manager.py` | Create per-session tracker on session creation |
| `agents/acp_agent/client_handler.py` | Resolve `AgentPlanUpdate` TODO — update per-session tracker |
| `acp_server/session_manager.py` | Add `remove_session_todos()` on session close |

**Duration**: 0.5 day

### Phase 4: CodexAgent & Cleanup

**Scope**: Update `CodexAgent` plan sync, add pool swap cleanup.

**Files**:

| File | Changes |
|------|---------|
| `agents/codex_agent/codex_agent.py` | Use `pool.get_session_todos(self.session_id)` |
| Protocol-specific pool swap handlers | Add `cleanup_all_session_todos()` |

**Duration**: 0.5 day

### Phase 5: Testing & Validation

**Tests**:
1. Two concurrent sessions maintain independent todo lists
2. Session deletion removes per-session tracker entries
3. `on_todo_change` callback only fires for the relevant session
4. ACP `AgentPlanUpdate` updates per-session tracker correctly
5. Sessionless contexts (CLI, tests) fall back to global `pool.todos`
6. `PlanProvider` tools work correctly with per-session tracker
7. Pool swap cleans up all per-session trackers
8. Session resume/restore gets a fresh empty tracker
9. All existing tests pass without modification

**Duration**: 0.5–1 day

### Rollback Strategy

Self-contained change. Revert by:
1. Restoring `PlanProvider._get_tracker()` to use `agent_ctx.pool.todos`
2. Removing `_session_todos` registry from `AgentPool`
3. Restoring global `on_todo_change` callback in OpenCode server
4. Reverting `client_handler.py` `AgentPlanUpdate` handling to TODO comment

---

## Open Questions

1. **`AgentContext.session_id` availability**
   - Context: `BaseAgent.session_id` is set by protocol servers during session binding. For sessionless contexts (CLI, direct `agent.run_stream()`), it is `None`. The `get_session_todos(None)` fallback to global `pool.todos` handles this correctly.
   - Owner: Implementer
   - Status: **RESOLVED** — `None` session_id falls back to global tracker

2. **Per-session tracker persistence**
   - Context: This RFC does not address persisting todos to storage. When a session is loaded from storage after server restart, its tracker will be empty. This is the current behavior (todos are not persisted today).
   - Owner: Future RFC
   - Status: **DEFERRED** — out of scope, can be added independently

3. **ACP `AgentPlanUpdate` notification delivery**
   - Context: The `tracker.on_change` callback for ACP needs to schedule an `AgentPlanUpdate` notification through the session's update system. The exact integration point depends on how `ACPSession._update_callbacks` work.
   - Owner: Implementer
   - Status: Open — verify during ACP integration

4. **Child/subagent sessions**
   - Context: OpenCode child sessions (parent_id is set) currently share the parent's todo state. With per-session trackers, child sessions get their own tracker. Is this the desired behavior, or should child sessions inherit the parent's tracker?
   - Owner: Implementer
   - Status: Open — likely child sessions should have their own tracker (subagents often have different task lists)

5. **XenoPlanProvider compatibility**
   - Context: `XenoPlanProvider` (in xeno-agent) extends `PlanProvider` with custom fields. The `_get_tracker()` override should work correctly with per-session resolution since it calls `super()._get_tracker()`.
   - Owner: Implementer
   - Status: **RESOLVED** — `XenoPlanProvider` inherits the fix automatically

---

## Decision Record

> To be completed after RFC review.

---

## References

### Related RFCs

- [RFC-0026: Per-Session Agent Instances](../implemented/RFC-0026-per-session-agent-isolation.md) — OpenCode server per-session agent isolation (proven pattern)
- [RFC-0031: ACP Per-Session Agent Isolation](../implemented/RFC-0031-acp-per-session-agent-isolation.md) — ACP server per-session agent isolation (proven pattern)
- [RFC-0021: Agent Concurrent Execution Safety](../implemented/RFC-0021-agent-concurrent-execution-safety.md) — Per-run isolation via `AgentRunContext`
- [RFC-0024: Agent Stateless Refactor](./RFC-0024-agent-stateless-refactor.md) — Future: Make `BaseAgent` stateless (may enable agent-owned todos)

### Key Source Files

- `src/wolfharness/utils/todos.py` — `TodoTracker`, `TodoEntry`, `PlanEntry`
- `src/wolfharness/delegation/pool.py` — `AgentPool.todos` (global instance)
- `src/wolfharness/resource_providers/plan_provider.py` — `PlanProvider` (tool implementation)
- `src/wolfharness_server/opencode_server/server.py` — Global `on_todo_change` callback
- `src/wolfharness_server/opencode_server/state.py` — `ServerState.todos` (in-memory cache)
- `src/wolfharness_server/opencode_server/routes/session_routes.py` — `get_session_todos()` route
- `src/wolfharness_server/acp_server/event_converter.py` — `PlanUpdateEvent` → `AgentPlanUpdate`
- `src/wolfharness/agents/acp_agent/client_handler.py` — Unresolved `AgentPlanUpdate` TODO
- `src/wolfharness/agents/codex_agent/codex_agent.py` — `pool.todos.replace_all()` sync
