---
rfc_id: RFC-0028
title: Delegation Provider Session Adaptation — Unify Child Session Lifecycle
status: DRAFT
author: yuchen.liu
reviewers: []
created: 2026-04-24
last_updated: 2026-04-24 (Rev 7)
decision_date:
related_rfcs:
  - RFC-0001 (Workers and Teams Session Management)
  - RFC-0013 (Subagent Event Stream Unification)
  - RFC-0014 (SpawnSessionStart Event)
  - RFC-0026 (Per-Session Agent Isolation)
---

# RFC-0028: Delegation Provider Session Adaptation — Unify Child Session Lifecycle

## Overview

This RFC proposes adapting all delegation providers (SubagentTools, WorkersTools, Team, TeamRun) to use `SessionManager.create_child_session()` as the canonical path for child session creation, instead of the current ad-hoc pattern of manual ID generation + `SpawnSessionStart` emission without persistence. The result will be a single source of truth for session hierarchy, consistent project_id/cwd inheritance, and durable parent-child relationships that survive restarts.

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

AgentPool has a `SessionManager` class (`src/wolfharness/sessions/manager.py`) with a `create_child_session()` method that:

1. Generates a unique session ID
2. Loads the parent session and inherits `project_id` / `cwd`
3. Persists `SessionData` with `parent_id` to `SessionStore`
4. Returns the child session ID

**This method is never called by any delegation provider.** Instead, each provider independently:

1. Generates a child session ID via `identifier.ascending("session")`
2. Manually constructs and emits `SpawnSessionStart`
3. Passes `session_id` / `parent_session_id` as kwargs to `run_stream()`
4. **Never persists** the child session to `SessionStore`

The server-side `EventProcessor` or `ServerState.ensure_session()` reactively materializes child sessions upon receiving `SubAgentEvent` — duplicating the project_id/cwd inheritance logic that already exists in `SessionManager.create_child_session()`.

### Historical Context

- **RFC-0001** proposed adding session management to Workers/Teams, recommending "Incremental Enhancement" (copy SubagentTools pattern). This was partially implemented for WorkersTools but never for Team/TeamRun.
- **RFC-0013** unified event processing with `EventProcessor`, improving subagent streaming but leaving the session creation gap unaddressed.
- **RFC-0014** added `SpawnSessionStart` as an explicit lifecycle signal but did not connect it to `SessionManager`.
- **RFC-0026** introduced per-session agent instances, making session isolation more important but not addressing session hierarchy persistence.

### Glossary

| Term | Definition |
|------|------------|
| **Delegation Provider** | Any mechanism that spawns child agent execution: SubagentTools (`task`), WorkersTools, Team (parallel), TeamRun (sequential) |
| **SessionManager** | Central class for session CRUD and child session creation (`src/wolfharness/sessions/manager.py`) |
| **SessionStore** | Protocol for session persistence (`SessionData` objects) — memory or SQL-backed |
| **SessionData** | Pydantic model with `session_id`, `parent_id`, `project_id`, `cwd`, `agent_name`, `agent_type` |
| **SpawnSessionStart** | Event emitted before child content starts; carries `child_session_id`, `parent_session_id`, `depth`, metadata |
| **SubAgentEvent** | Wrapper event that propagates child agent events to parent stream |
| **ensure_session()** | OpenCode server-side method that reactively creates session objects from event data |

### Current Provider Behavior Matrix

| Provider | Session ID Generation | Calls `create_child_session()` | Persists to SessionStore | Emits SpawnSessionStart | SubAgentEvent has session IDs |
|---|---|---|---|---|---|
| **SubagentTools** | `identifier.ascending("session")` | ❌ | ❌ | ✅ | ✅ |
| **WorkersTools** | `identifier.ascending("session")` | ❌ | ❌ | ✅ | ✅ |
| **Team** (parallel) | None | ❌ | ❌ | ❌ | ❌ (all None) |
| **TeamRun** (sequential) | None | ❌ | ❌ | ❌ | ❌ (all None) |
| **ACPSessionManager** | Direct `store.save()` | ❌ | ✅ (but skips inheritance) | N/A | N/A |

---

## Problem Statement

### The Problem

Five specific gaps exist in the current delegation provider session handling:

1. **No Session Persistence**: `SubagentTools` and `WorkersTools` generate child session IDs but never persist `SessionData` to `SessionStore`. On server restart, all child session metadata is lost. `pool.sessions.get_child_sessions()` returns empty.

2. **`create_child_session()` is Dead Code**: The canonical API for child session creation exists and is tested, but zero production code calls it. The project_id/cwd inheritance logic it contains is duplicated in `ServerState.ensure_session()`.

3. **Team/TeamRun Have Zero Session Awareness**: Per RFC-0001 (never fully implemented for Teams), parallel and sequential teams do not propagate session IDs, emit `SpawnSessionStart`, or track parent-child relationships. Their `SubAgentEvent` instances have `child_session_id=None`.

4. **AgentContext Lacks Session Helpers**: Tools that need child session IDs must reach into `ctx.node.session_id` and call `identifier.ascending("session")` manually. No ergonomic API exists.

5. **`AgentRunContext.session_id` is Dead Code**: `AgentRunContext.session_id` defaults to `uuid.uuid4().hex` (context.py:60) but is **never read** — `run_ctx.session_id` has zero references in the codebase. `BaseAgent.run_stream()` creates `AgentRunContext(deps=deps)` without passing `session_id`, so the default value is entirely ignored. The actual session ID used is `BaseAgent.session_id`, which already uses `generate_session_id()` (base_agent.py:644). The format inconsistency is therefore moot — the field should be removed or connected to `BaseAgent.session_id`.

### Evidence

- `pool.sessions.get_child_sessions(parent_id)` returns `[]` for any parent session that spawned subagents/workers — the data was never persisted
- `SessionData` rows in SQL storage have no `parent_id` set for child sessions created by delegation providers
- The OpenCode TUI's session tree view cannot show parent-child hierarchy because the data doesn't exist
- `EventProcessor._process_subagent_event()` line ~737 contains a reactive fallback: `if child_ctx is None and child_session_id: await ctx.state.ensure_session(...)` — this compensates for the missing proactive persistence

### Impact of Inaction

- **Cost**: Every new delegation provider must reimplement session creation (3 out of 5 already do it differently)
- **Risk**: The reactive fallback in `EventProcessor` is fragile — it depends on event ordering and creates sessions on the OpenCode server side only, not in the core `SessionManager`
- **Opportunity Loss**: Cannot build features that rely on session hierarchy (cost tracking per subtree, session archival, analytics by delegation depth, session tree UI)

---

## Goals & Non-Goals

### Goals (In Scope)

1. All delegation providers use `SessionManager.create_child_session()` for child session creation
2. Child sessions are persisted to `SessionStore` with correct `parent_id`, `project_id`, `cwd`
3. Team and TeamRun propagate session IDs and emit `SpawnSessionStart`
4. `AgentContext` exposes a `create_child_session()` convenience method
5. `SpawnSessionStart` emission remains the responsibility of delegation providers (not `SessionManager`)
6. `AgentRunContext.session_id` dead code is removed or properly connected

### Non-Goals (Out of Scope)

1. **Changing `BaseAgent` internals** — that's RFC-0024/0025 territory (the `depth` param addition is a minimal signature change, not an internal refactor)
2. **Adding `SpawnSessionEnd` event** — `StreamCompleteEvent` already serves this purpose (per RFC-0014)
3. **Session cleanup/eviction** — orthogonal concern; can be addressed independently
4. **MCP server session isolation** — addressed in RFC-0026 verification
5. **Changing the OpenCode protocol** — this is an internal refactoring

### Success Criteria

- [ ] `pool.sessions.get_child_sessions(parent_id)` returns correct child session IDs after subagent/worker execution
- [ ] Team and TeamRun emit `SpawnSessionStart` with valid session IDs
- [ ] `SessionData` persisted to store has correct `project_id` and `cwd` inherited from parent
- [ ] `AgentRunContext.session_id` is either removed or properly connected to `BaseAgent.session_id`
- [ ] All existing tests pass without modification
- [ ] `ServerState.ensure_session()` no longer needs to duplicate project_id/cwd inheritance logic

---

## Evaluation Criteria

| Criterion | Weight | Description | Minimum Threshold |
|-----------|--------|-------------|-------------------|
| **Consistency** | High | All providers use the same session creation path | Must use `create_child_session()` |
| **Backward Compatibility** | High | Existing code continues to work | 100% backward compatible |
| **Minimality** | High | Smallest change that achieves goals | No unnecessary refactoring |
| **Persistence** | High | Child sessions survive in SessionStore | Must persist SessionData |
| **Implementation Cost** | Medium | Development effort required | ≤ 1 week |
| **Performance** | Medium | Overhead of session persistence | < 5% latency increase on delegation calls |

---

## Options Analysis

### Option 1: Adapt Providers to Call `create_child_session()` + Emit Events

**Description**

Refactor each delegation provider to call `ctx.pool.sessions.create_child_session()` for ID generation and persistence, then separately emit `SpawnSessionStart` and `SubAgentEvent` as before. Add `AgentContext.create_child_session()` convenience method. Fix `AgentRunContext.session_id` format.

**Advantages**

- Single source of truth for session creation — `SessionManager` handles ID generation, inheritance, and persistence
- Project_id/cwd inheritance is guaranteed correct (one implementation, not N)
- `get_child_sessions()` works for all delegation types
- Minimal architectural change — same event flow, just different session creation source
- `ServerState.ensure_session()` can be simplified (session already persisted)

**Disadvantages**

- Still requires changes to each delegation provider (SubagentTools, WorkersTools, Team, TeamRun)
- `create_child_session()` is async — adds `await` in provider code paths
- `SpawnSessionStart` emission is decoupled from session creation (two separate calls)

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Consistency | 5/5 | All providers call same API |
| Backward Compatibility | 5/5 | No breaking changes — same events emitted |
| Minimality | 4/5 | Focused changes, no new abstractions |
| Persistence | 5/5 | SessionData persisted by SessionManager |
| Implementation Cost | 4/5 | ~3-4 days, targeted changes |
| Performance | 5/5 | One extra store.save() per child — negligible |

**Effort Estimate**

- Complexity: Low-Medium
- Resources: 1 developer, 3-4 days
- Dependencies: None

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `create_child_session()` returns ID in different format | Low | Medium | It uses same `generate_session_id()` → `ascending("session")` |
| Store not available in some contexts | Low | High | `create_child_session()` already handles `store=None` (returns ID without persisting) |
| Async call in team execution path | Low | Low | Teams already use async |

---

### Option 2: Move Session Creation into `SpawnSessionStart` Handler

**Description**

Instead of having providers call `create_child_session()`, have the `EventProcessor` / `EventManager` handle session creation when it receives `SpawnSessionStart`. The provider still generates the ID and emits the event, but persistence happens in the event handler.

**Advantages**

- Session creation is triggered by the event — natural lifecycle coupling
- No async call needed in provider code (session created reactively)
- Single emission point for both creation signal and persistence trigger

**Disadvantages**

- Session persistence depends on event handling — if event is dropped or handler not attached, no persistence
- OpenCode server creates sessions via `EventProcessor`, but ACP/AG-UI servers have their own handlers — need to add persistence to each
- Core `SessionManager` is bypassed; persistence happens at server layer, not framework layer
- Tests that only run providers (without server) would not persist sessions

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Consistency | 3/5 | Each server must implement its own handler |
| Backward Compatibility | 4/5 | Events unchanged, but behavior changes on handler side |
| Minimality | 3/5 | Must add handlers to multiple servers |
| Persistence | 3/5 | Only persists when event handler runs |
| Implementation Cost | 3/5 | ~5 days, multiple server changes |
| Performance | 5/5 | Reactive, no extra call in provider |

**Effort Estimate**

- Complexity: Medium
- Resources: 1 developer, 5 days
- Dependencies: All protocol servers

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Event handler not attached | Medium | High | Must ensure all servers register handler |
| Session not persisted in test contexts | Medium | Medium | Need test infrastructure that attaches handler |
| Race condition: content events arrive before session created | Low | High | SpawnSessionStart must be processed first |

---

### Option 3: Unified DelegationProvider Base Class

**Description**

Create an abstract `DelegationProvider` base class that encapsulates the complete child session lifecycle: ID generation via `SessionManager`, `SpawnSessionStart` emission, `SubAgentEvent` wrapping, and depth tracking. Refactor SubagentTools and WorkersTools to inherit from it. Add Team/TeamRun support via the base class.

**Advantages**

- Maximum code reuse — one implementation for all delegation patterns
- Future delegation mechanisms automatically get session support
- Enforces consistent behavior across all providers

**Disadvantages**

- Significant refactoring of working code (SubagentTools, WorkersTools)
- SubagentTools and WorkersTools have different execution patterns (streaming vs background) — base class may be awkward
- Higher risk of regressions
- Team/TeamRun are `MessageNode` subclasses, not tools — different integration point

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Consistency | 5/5 | Perfect via shared base class |
| Backward Compatibility | 3/5 | Refactoring risk |
| Minimality | 2/5 | Large change surface |
| Persistence | 5/5 | Centralized |
| Implementation Cost | 2/5 | ~7-10 days |
| Performance | 5/5 | No additional overhead |

**Effort Estimate**

- Complexity: High
- Resources: 1-2 developers, 7-10 days
- Dependencies: Full regression testing required

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SubagentTools regressions | Medium | High | Extensive test coverage |
| Awkward abstraction for different execution models | Medium | Medium | Careful interface design |
| Team/TeamRun don't fit tool-based abstraction | High | Medium | May need separate base or mixin |

---

### Options Comparison Summary

| Criterion | Option 1: Adapt Providers | Option 2: Event-Driven | Option 3: Base Class |
|-----------|--------------------------|----------------------|---------------------|
| Consistency | 5/5 | 3/5 | 5/5 |
| Backward Compatibility | 5/5 | 4/5 | 3/5 |
| Minimality | 4/5 | 3/5 | 2/5 |
| Persistence | 5/5 | 3/5 | 5/5 |
| Implementation Cost | 4/5 | 3/5 | 2/5 |
| Performance | 5/5 | 5/5 | 5/5 |
| **Overall** | **28/30** | **21/30** | **22/30** |

---

## Recommendation

### Recommended Option

**Option 1: Adapt Providers to Call `create_child_session()` + Emit Events**

### Justification

Option 1 provides the best balance of consistency, backward compatibility, and implementation cost. It makes `SessionManager.create_child_session()` the canonical path without introducing new abstractions or architectural changes. The key insight is that session creation (ID generation + persistence + inheritance) is a **framework concern** that belongs in `SessionManager`, while event emission (`SpawnSessionStart`, `SubAgentEvent`) is a **protocol concern** that belongs in the provider. These two responsibilities are correctly separated.

Option 2 ties persistence to event handling, which creates a fragile coupling — sessions would only be persisted when a server is attached and handling events. Option 3 has the right long-term vision but the wrong timing — the refactoring risk is not justified when a focused adaptation achieves the same functional outcome.

### Accepted Trade-offs

1. **Two separate calls** (`create_child_session()` + `SpawnSessionStart` emission) instead of a single atomic operation
   - Acceptable because the two serve different purposes (persistence vs. UI notification) and the session ID links them
   - If `create_child_session()` succeeds but `SpawnSessionStart` emit fails, the persisted orphan session is benign but should be logged
   - If atomicity becomes critical, a future RFC can merge them

2. **Code duplication between providers** remains until a future base class is extracted
   - Acceptable because the duplicated pattern is simple (3-4 lines per provider) and stable

3. **`create_child_session()` is async** — adds `await` in provider hot paths
   - Acceptable because `store.save()` is the only async operation and is near-instant for MemorySessionStore

4. **`ensure_session()` double-persist risk** — after adaptation, `ensure_session()` could overwrite `SessionData` created by `create_child_session()`
    - Mitigated by: `ensure_session()` loads from store first and skips `store.save()` when session was already persisted
    - The `_session_from_session_data()` mapping creates only the UI `Session` object, without re-persisting

5. **Two-tier persistence model** — providers operating within an `AgentPool` persist proactively via `create_child_session()`, while providers operating outside a pool (e.g., a Team instantiated directly without a pool) cannot persist and fall back to reactive `ensure_session()` when a server is attached
    - Acceptable because out-of-pool usage has no `SessionStore` to persist to
    - The reactive `ensure_session()` fallback in `EventProcessor` still works for these cases
    - Documented explicitly so implementers don't assume the reactive path is fully eliminated

6. **`pool_id` field inconsistency** — `create_child_session()` uses `manifest.name` (e.g., `"my-pool"`) for `pool_id`, while `ensure_session()` (fallback path) uses `manifest.config_file_path` (e.g., `"/path/to/config.yml"`)
    - After adaptation, the store-first path in `ensure_session()` prevents overwrite, so the `manifest.name` value from `create_child_session()` is preserved
    - However, existing sessions in the store created before adaptation will have `config_file_path` format
    - **Recommendation**: Standardize `pool_id` to `manifest.name` in a follow-up; for this RFC, the store-first path prevents new inconsistencies

### Conditions

- Implementation must not break the OpenCode server's reactive fallback in `EventProcessor` — it should still work for any provider that hasn't been adapted yet
- Tests must verify `get_child_sessions()` returns correct results after delegation

---

## Technical Design

### Architecture Overview

```
BEFORE:
┌──────────────────┐     ┌──────────────────┐
│  SubagentTools   │     │  WorkersTools    │
│  ┌────────────┐  │     │  ┌────────────┐  │
│  │ ID gen     │  │     │  │ ID gen     │  │
│  │ SpawnEvent │  │     │  │ SpawnEvent │  │
│  │ NO persist │  │     │  │ NO persist │  │
│  └────────────┘  │     │  └────────────┘  │
└──────────────────┘     └──────────────────┘
┌──────────────────┐     ┌──────────────────┐
│  Team            │     │  TeamRun         │
│  (no sessions)   │     │  (no sessions)   │
└──────────────────┘     └──────────────────┘

AFTER:
                 ┌─────────────────────────┐
                 │    SessionManager        │
                 │  create_child_session()  │
                 │  ├─ ID generation        │
                 │  ├─ project_id/cwd inherit│
                 │  └─ SessionStore.persist │
                 └─────────┬───────────────┘
                           │ called by
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│SubagentTools │  │WorkersTools  │  │Team/TeamRun  │
│ ├─create_cs()│  │ ├─create_cs()│  │ ├─create_cs()│
│ ├─SpawnEvent │  │ ├─SpawnEvent │  │ ├─SpawnEvent │
│ └─SubAE wrap │  │ └─SubAE wrap │  │ └─SubAE wrap │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Key Components

#### 1. `AgentContext.create_child_session()` — Convenience Method

Add a convenience method to `AgentContext` that wraps `pool.sessions.create_child_session()`:

```python
# src/wolfharness/agents/context.py

class AgentContext:
    # ... existing code ...

    async def create_child_session(
        self,
        agent_name: str,
        agent_type: str = "native",
        parent_session_id: str | None = None,
    ) -> str:
        """Create a child session inheriting project_id/cwd from current session.

        Returns the child_session_id for use with run_stream() and SpawnSessionStart.
        Falls back to generate_session_id() if agent_pool is not available.

        Args:
            agent_name: Name of the child agent.
            agent_type: Type of the child agent (e.g., "native", "team", "acp").
            parent_session_id: Override for the parent session ID. If not provided,
                uses self.node.session_id or generates a new ID. This parameter
                prevents double-generation when the caller already knows the
                parent session ID.
        """
        parent_id = parent_session_id or self.node.session_id or generate_session_id()
        if self.node.agent_pool is not None:
            return await self.node.agent_pool.sessions.create_child_session(
                parent_session_id=parent_id,
                agent_name=agent_name,
                agent_type=agent_type,
            )
        else:
            # Agent created outside a pool — generate ID without persistence.
            # See "Two-Tier Persistence" in Accepted Trade-offs.
            return generate_session_id()
```

#### 2. SubagentTools Adaptation

```python
# src/wolfharness_toolsets/builtin/subagent_tools.py

# BEFORE:
child_session_id = identifier.ascending("session")
parent_session_id = ctx.node.session_id or identifier.ascending("session")

# AFTER:
child_session_id = await ctx.create_child_session(
    agent_name=agent_or_team,
    agent_type=source_type,
)
parent_session_id = ctx.node.session_id or generate_session_id()
```

The `SpawnSessionStart` emission remains unchanged — it already uses `child_session_id` and `parent_session_id`.

**Dual emission fix**: SubagentTools has two code paths — `task()` (async, ~line 349) and `_stream_task()` (streaming, ~line 96). Both currently emit `SpawnSessionStart` independently. After adaptation, the fix is:

1. `task()` calls `create_child_session()`, emits `SpawnSessionStart`, then calls `_stream_task()` 
2. `_stream_task()` **removes** its `SpawnSessionStart` emission entirely — the event is already emitted by `task()`
3. `_stream_task()` no longer needs `skip_spawn_event` — the single-emission principle is enforced structurally

This simplifies the design: each delegation boundary emits exactly one `SpawnSessionStart`, and the emission always happens at the call site (`task()`), not in the streaming helper. When `_stream_task()` is used in standalone streaming mode (called directly without `task()`), the caller is responsible for emitting `SpawnSessionStart` before entering the stream.

```python
# _stream_task() AFTER — no SpawnSessionStart emission
async def _stream_task(self, ctx, agent_or_team, message, child_session_id, parent_session_id, child_depth, **kwargs):
    # Stream with session context — no SpawnSessionStart here
    async for event in node.run_stream(
        message,
        session_id=child_session_id,
        parent_session_id=parent_session_id,
        depth=child_depth,
        **kwargs,
    ):
        yield SubAgentEvent(...)
```

**Depth propagation in SubagentTools**: The `getattr(ctx, "current_depth", 0)` anti-pattern must be replaced, but SubagentTools functions are PydanticAI tools — they don't receive `**kwargs` from `run_stream()`. The depth must be accessed through `AgentRunContext`:

```python
# Mechanism: BaseAgent.run_stream(depth=N) passes depth to AgentRunContext(depth=depth),
# which SubagentTools can then read via ctx.run_ctx.depth.

# In SubagentTools:
# BEFORE:
depth=getattr(ctx, "current_depth", 0) + 1,
# AFTER:
depth=(ctx.run_ctx.depth if ctx.run_ctx is not None else 0) + 1,
```

**CRITICAL**: `ctx.deps` resolves to user-provided `TDeps | None`, NOT `AgentRunContext`. The correct access path is `ctx.run_ctx.depth`, where `ctx` is `AgentContext`, and `ctx.run_ctx` is `AgentRunContext | None`. A None guard is required because `run_ctx` may be `None` in edge cases (e.g., tools called outside the `run_stream()` context).

This requires adding `depth: int = 0` to `AgentRunContext` as a field, populated from the `run_stream(depth=...)` parameter via `AgentRunContext(deps=deps, depth=depth)`. The field is then accessible via `ctx.run_ctx.depth` in tool functions.

#### 3. WorkersTools Adaptation

Same pattern as SubagentTools, applied in both `_create_agent_tool()` and `_create_node_tool()`:

```python
# BEFORE:
child_session_id = identifier.ascending("session")
parent_session_id = ctx.node.session_id or identifier.ascending("session")
# ... depth=1 (hardcoded) ...

# AFTER:
child_session_id = await ctx.create_child_session(
    agent_name=worker_name,
    agent_type=source_type,
)
parent_session_id = ctx.node.session_id or generate_session_id()
# depth must be computed from ctx.run_ctx, not hardcoded to 1
child_depth = (ctx.run_ctx.depth if ctx.run_ctx is not None else 0) + 1
```

**CRITICAL**: WorkersTools currently hardcodes `depth=1` in four locations (workers.py:130, 151, 221, 241). After adaptation, all four must use `child_depth = (ctx.run_ctx.depth if ctx.run_ctx is not None else 0) + 1` instead. The hardcoded `depth=1` breaks nested delegation (A→Worker→Worker would show depth=1 for both levels).

#### 4. `BaseAgent.run_stream()` Signature Update

**Critical**: `BaseAgent.run_stream()` currently has explicit typed parameters and **no `**kwargs` catch-all**. Passing `depth=current_depth + 1` from Team/TeamRun would raise `TypeError: run_stream() got an unexpected keyword argument 'depth'`.

**Solution**: Add `depth: int = 0` as an explicit parameter to `BaseAgent.run_stream()`:

```python
# src/wolfharness/agents/base_agent.py

# BEFORE (actual 12-param signature — simplified for illustration):
async def run_stream(
    self,
    *prompts: str | Prompt | Sequence[str | Prompt],
    session_id: str | None = None,
    parent_session_id: str | None = None,
    parent_id: str | None = None,
    message_history: list | None = None,
    model: str | Model | None = None,
    deps: AgentContext | None = None,
    # ... additional params
) -> AsyncIterator[AgentStreamEvent]:
    # ... creates AgentRunContext(deps=deps) without depth
    run_ctx = AgentRunContext(deps=deps)

# AFTER:
async def run_stream(
    self,
    *prompts: str | Prompt | Sequence[str | Prompt],
    depth: int = 0,  # NEW: depth tracking for delegation hierarchies
    session_id: str | None = None,
    parent_session_id: str | None = None,
    parent_id: str | None = None,
    message_history: list | None = None,
    model: str | Model | None = None,
    deps: AgentContext | None = None,
    # ... additional params
) -> AsyncIterator[AgentStreamEvent]:
    """Run agent with optional depth tracking for delegation hierarchies.

    Args:
        depth: Reserved for delegation depth tracking — used by Team/TeamRun
               wrappers and SubagentTools. Stored on AgentRunContext for
               tool access via ctx.run_ctx.depth.
    """
    run_ctx = AgentRunContext(deps=deps, depth=depth)  # Wire depth through
```

**IMPORTANT**: `BaseAgent.run_stream()` accepts `depth: int = 0`. No agent subclass (ACP, Claude, AG-UI, Codex) overrides `run_stream()` — they use the template method pattern via `_run_stream_once()`. Therefore, only `BaseAgent.run_stream()` needs the `depth` parameter.

This is the **only correct approach** — alternatives like stripping `depth` from kwargs before passing to `run_stream()` would require every call site to know about depth, defeating the purpose of kwargs forwarding.

#### 5. Team Adaptation (`team.py`)

Add session propagation to parallel team execution:

```python
# src/wolfharness/delegation/team.py

async def run_stream(self, *prompts, depth: int = 0, **kwargs):
    all_nodes = list(self.nodes)
    # Read session_id from kwargs first (set by SubagentTools caller),
    # then fall back to self.session_id (set by BaseAgent), then generate.
    # This ensures the session hierarchy chain is preserved when
    # SubagentTools delegates to a Team.
    parent_session_id = kwargs.pop("session_id", None) or self.session_id or generate_session_id()

    async def wrap_stream(node: MessageNode):
        # Create child session for this team member
        # NOTE: self.agent_pool (not self._pool) — see MessageNode.agent_pool
        if self.agent_pool is not None:
            child_session_id = await self.agent_pool.sessions.create_child_session(
                parent_session_id=parent_session_id,
                agent_name=node.name,
                agent_type=node.agent_type,  # Implementation type: "native", "acp", "team"
            )
        else:
            # Team created outside a pool — generate ID without persistence.
            # Session will be persisted reactively by ensure_session() when
            # a server is attached. See "Two-Tier Persistence" in trade-offs.
            child_session_id = generate_session_id()

        # Extract model_id for raw event wrapping (current code: team.py:200-202)
        node_model_id: str | None = None
        if isinstance(node, BaseAgent):
            node_model_id = node.model_name

        # Emit SpawnSessionStart
        yield SpawnSessionStart(
            child_session_id=child_session_id,
            parent_session_id=parent_session_id,
            tool_call_id=None,
            spawn_mechanism="spawn",
            source_name=node.name,
            source_type=get_source_type(node),  # Delegation type: "agent", "team_parallel", "team_sequential"
            depth=depth + 1,
            description=f"Run team member {node.name}",
            metadata={},
        )

        # Stream with session context
        async for event in node.run_stream(
            *prompts,
            session_id=child_session_id,
            parent_session_id=parent_session_id,
            depth=depth + 1,
            **kwargs,
        ):
            # CRITICAL: Distinguish already-wrapped SubAgentEvents from raw events.
            # SubAgentEvents from nested teams already carry correct depth from their
            # own context — incrementing it again would produce incorrect depth values.
            # Raw events from direct agents need depth from THIS team's context.
            # See current code: team.py:205-222, teamrun.py:296-312.
            if isinstance(event, SubAgentEvent):
                yield SubAgentEvent(
                    source_name=event.source_name,
                    source_type=event.source_type,
                    event=event.event,
                    depth=event.depth + 1,  # Increment nested team's depth
                    child_session_id=event.child_session_id,    # Preserve inner session IDs
                    parent_session_id=event.parent_session_id,  # Preserve inner session IDs
                    model_id=event.model_id,
                    mode=event.mode,
                )
            else:
                yield SubAgentEvent(
                    source_name=node.name,
                    source_type=get_source_type(node),  # Delegation type for this node
                    event=event,
                    depth=depth + 1,  # Use team's delegation depth
                    child_session_id=child_session_id,
                    parent_session_id=parent_session_id,
                    model_id=node_model_id,
                )

    async for event in as_generated(wrap_stream(n) for n in all_nodes):
        yield event
```

#### 6. TeamRun Adaptation (`teamrun.py`)

Sequential team — each step's child session is created independently:

```python
# src/wolfharness/delegation/teamrun.py

async def run_stream(self, *prompts, depth: int = 0, require_all: bool = True, **kwargs):
    # Read session_id from kwargs first (set by SubagentTools caller),
    # then fall back to self.session_id, then generate.
    parent_session_id = kwargs.pop("session_id", None) or self.session_id or generate_session_id()
    current_message = prompts

    for node in self.nodes:
        # NOTE: self.agent_pool (not self._pool) — see MessageNode.agent_pool
        if self.agent_pool is not None:
            child_session_id = await self.agent_pool.sessions.create_child_session(
                parent_session_id=parent_session_id,
                agent_name=node.name,
                agent_type=node.agent_type,  # Implementation type: "native", "acp", "team"
            )
        else:
            # TeamRun created outside a pool — generate ID without persistence.
            # See "Two-Tier Persistence" in Accepted Trade-offs.
            child_session_id = generate_session_id()

        # Extract model_id for raw event wrapping (current code: teamrun.py:291-293)
        node_model_id: str | None = None
        if isinstance(node, BaseAgent):
            node_model_id = node.model_name

        # Emit SpawnSessionStart
        yield SpawnSessionStart(
            child_session_id=child_session_id,
            parent_session_id=parent_session_id,
            tool_call_id=None,
            spawn_mechanism="spawn",
            source_name=node.name,
            source_type=get_source_type(node),  # Delegation type: "agent", "team_parallel", "team_sequential"
            depth=depth + 1,
            description=f"Run chain member {node.name}",
            metadata={},
        )

        # Stream with session context
        async for event in node.run_stream(
            *current_message,
            session_id=child_session_id,
            parent_session_id=parent_session_id,
            depth=depth + 1,
            **kwargs,
        ):
            # CRITICAL: Same SubAgentEvent depth distinction as Team (see Component #5).
            # Already-wrapped SubAgentEvents from nested teams increment their own depth;
            # raw events from direct agents use this TeamRun's delegation depth.
            # See current code: teamrun.py:296-312.
            if isinstance(event, SubAgentEvent):
                yield SubAgentEvent(
                    source_name=event.source_name,
                    source_type=event.source_type,
                    event=event.event,
                    depth=event.depth + 1,  # Increment nested team's depth
                    child_session_id=event.child_session_id,    # Preserve inner session IDs
                    parent_session_id=event.parent_session_id,  # Preserve inner session IDs
                    model_id=event.model_id,
                    mode=event.mode,
                )
            else:
                yield SubAgentEvent(
                    source_name=node.name,
                    source_type=get_source_type(node),  # Delegation type for this node
                    event=event,
                    depth=depth + 1,  # Use TeamRun's delegation depth
                    child_session_id=child_session_id,
                    parent_session_id=parent_session_id,
                    model_id=node_model_id,
                )
                # Collect final output for next step
                if isinstance(event, StreamCompleteEvent):
                    current_message = (str(event.message.content),)
```

#### 7. `AgentRunContext.session_id` Dead Code Fix

**Problem**: `AgentRunContext.session_id` defaults to `uuid.uuid4().hex` but is **never read** — `run_ctx.session_id` has zero references. `BaseAgent.run_stream()` creates `AgentRunContext(deps=deps)` without passing `session_id`, so the default is ignored. The actual session ID is `BaseAgent.session_id`, which already uses `generate_session_id()`.

**Solution**: Remove the dead field or connect it to `BaseAgent.session_id`:

```python
# src/wolfharness/agents/base_agent.py

# BEFORE:
run_ctx = AgentRunContext(deps=deps)

# AFTER (Option A — pass session_id):
run_ctx = AgentRunContext(deps=deps, session_id=self.session_id)

# AFTER (Option B — remove the field entirely):
# Remove session_id from AgentRunContext dataclass
# All consumers already use self.session_id (BaseAgent) or ctx.node.session_id (AgentContext)
```

**Recommendation**: Option C (deprecate) — mark the field as deprecated using a descriptor that emits `DeprecationWarning` on access. Removing it is a public API break for external consumers. In a future major version, it can be removed entirely.

```python
# src/wolfharness/agents/context.py

import warnings
from dataclasses import dataclass, field

class _DeprecatedSessionId:
    """Descriptor that emits DeprecationWarning on access to session_id.

    IMPORTANT: This descriptor must be attached to the class AFTER @dataclass
    processing, NOT used as a field default. Using it as a default would cause
    the descriptor instance itself to be stored as the field value, rather than
    None or a string. The correct pattern is:

        @dataclass
        class AgentRunContext:
            session_id: str | None = field(default=None)

        # After class definition:
        AgentRunContext.session_id = _DeprecatedSessionId()  # type: ignore[assignment]

    This preserves the deprecation warning on access while keeping the default
    behavior (None) correct.
    """

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name
        self._storage_name = f"_{name}"

    def __get__(self, obj: object, objtype: type | None = None) -> str | None:
        if obj is None:
            return None  # Class-level access
        warnings.warn(
            "AgentRunContext.session_id is deprecated. Use ctx.node.session_id instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(obj, self._storage_name, None)

    def __set__(self, obj: object, value: str | None) -> None:
        setattr(obj, self._storage_name, value)


@dataclass
class AgentRunContext:
    deps: AgentContext
    depth: int = 0  # Set by BaseAgent.run_stream(depth=...)
    # session_id is deprecated — use ctx.node.session_id instead.
    # Default is None; descriptor is attached after class definition (below).
    session_id: str | None = field(default=None)


# Attach descriptor AFTER @dataclass processing — NOT as the field default.
# If _DeprecatedSessionId() were the default, the dataclass __init__ would
# pass the descriptor instance as the value, and __set__ would store the
# descriptor instance in _session_id. Then __get__ would return it (not None).
AgentRunContext.session_id = _DeprecatedSessionId()  # type: ignore[assignment]
```

#### 8. ACPSessionManager Adaptation

`ACPSessionManager.create_session()` currently creates `SessionData` directly and calls `store.save()`, bypassing `create_child_session()` and losing project_id/cwd inheritance. This must be adapted in Phase 1.

**Key challenge**: ACP sessions are typically **top-level** (no parent session). When an ACP client connects, it creates a fresh session — there is no parent to inherit from. However, when a subagent within an ACP session delegates to another ACP agent, the child session SHOULD have a parent.

**Solution**: Add `parent_session_id` parameter to `create_session()`. When provided, call `create_child_session()` (with inheritance). When not provided, create a top-level session with the existing project_id computation from cwd.

```python
# src/wolfharness_server/acp_server/session_manager.py

# BEFORE:
async def create_session(
    self,
    agent: BaseAgent[Any, Any],
    cwd: str,
    client: Client,
    acp_agent: AgentPoolACPAgent,
    mcp_servers: Sequence[McpServer] | None = None,
    session_id: str | None = None,
    client_capabilities: ClientCapabilities | None = None,
    client_info: Implementation | None = None,
    subagent_display_mode: Literal["inline", "tool_box"] = "tool_box",
) -> str:
    # ... generates session_id, computes project_id from cwd ...
    data = SessionData(
        session_id=session_id,
        agent_name=agent.name,
        cwd=cwd,
        project_id=project_id,
        metadata={"protocol": "acp", "mcp_server_count": len(mcp_servers or [])},
    )
    if self.session_store:
        await self.session_store.save(data)
    # ... creates ACPSession runtime object ...

# AFTER:
async def create_session(
    self,
    agent: BaseAgent[Any, Any],
    cwd: str,
    client: Client,
    acp_agent: AgentPoolACPAgent,
    mcp_servers: Sequence[McpServer] | None = None,
    session_id: str | None = None,
    parent_session_id: str | None = None,  # NEW: for ACP subagent delegation
    client_capabilities: ClientCapabilities | None = None,
    client_info: Implementation | None = None,
    subagent_display_mode: Literal["inline", "tool_box"] = "tool_box",
) -> str:
    async with self._lock:
        if session_id is None:
            session_id = self.storage.generate_session_id()

        if session_id in self._active:
            raise ValueError(f"Session {session_id} already exists")

        # Use create_child_session() when parent is known — inherits project_id/cwd
        if parent_session_id is not None:
            session_id = await self._pool.sessions.create_child_session(
                parent_session_id=parent_session_id,
                agent_name=agent.name,
                agent_type="acp",
            )
        else:
            # Top-level ACP session — compute project_id from cwd
            from wolfharness_storage.opencode_provider.helpers import compute_project_id
            project_id = compute_project_id(cwd)
            data = SessionData(
                session_id=session_id,
                agent_name=agent.name,
                cwd=cwd,
                project_id=project_id,
                agent_type="acp",
                metadata={"protocol": "acp", "mcp_server_count": len(mcp_servers or [])},
            )
            if self.session_store:
                await self.session_store.save(data)

        # Create ACPSession runtime object (unchanged)
        session = ACPSession(
            session_id=session_id,
            agent=agent,
            cwd=cwd,
            client=client,
            mcp_servers=mcp_servers,
            acp_agent=acp_agent,
            client_capabilities=client_capabilities or ClientCapabilities(),
            client_info=client_info,
            manager=self,
            subagent_display_mode=subagent_display_mode,
        )
        session.register_update_callback(self._on_commands_updated)
        await session.initialize()
        await session.initialize_mcp_servers()
        self._active[session_id] = session
        return session_id
```

**Caller updates**: `acp_agent.py` calls `create_session()` in five places (lines 335, 399, 499, 527, 589). For top-level sessions (most calls), `parent_session_id` is left as `None` (default) — no change needed. For subagent delegation within ACP, the caller passes the current session's ID as `parent_session_id`.

**Note on session_id discard**: When `parent_session_id is not None`, `create_child_session()` generates its own session ID internally, which overwrites any `session_id` value (either caller-provided or generated on line 846). This means a caller-provided `session_id` is silently discarded in the child path. This is acceptable because: (1) `create_child_session()` is the canonical session ID generator, (2) ACP callers rarely provide explicit session IDs when delegating, and (3) the returned `session_id` always reflects the actual ID used. If needed in the future, `create_child_session()` could accept an optional `session_id` override.

#### 9. `ServerState.ensure_session()` Simplification

**Critical Issue**: After adaptation, `ensure_session()` would **overwrite** the `SessionData` persisted by `create_child_session()`. The two paths create `SessionData` objects with different fields:

- `create_child_session()` creates `SessionData(agent_type="native", pool_id="my-pool", cwd="/foo")`
- `ensure_session()` creates a `Session` UI object, then converts it via `opencode_to_session_data()` which may produce `SessionData(agent_type=None, pool_id="path/to/config.yml", cwd=None)`

The second `store.save()` would **overwrite** the correct data from step 1.

**Solution**: `ensure_session()` must skip `store.save()` when the session was already persisted:

```python
# src/wolfharness_server/opencode_server/state.py

async def ensure_session(self, session_id: str, parent_id: str | None = None) -> Session:
    # Check if session already exists in memory
    if session_id in self.sessions:
        session = self.sessions[session_id]
        # Broadcast update so TUI SolidJS store stays in sync
        # (matches current behavior in state.py:607-612)
        await self.broadcast_event(SessionUpdatedEvent.create(session))
        return session

    # 1. Try loading from SessionStore first (persisted by create_child_session)
    if self.pool.sessions.store:
        session_data = await self.pool.sessions.store.load(session_id)
        if session_data:
            # Session already persisted — create UI Session from SessionData
            session = self._session_from_session_data(session_data)
            self.sessions[session_id] = session
            self.ensure_runtime_session_state(session_id)
            await self.mark_session_idle(session_id)
            # CRITICAL: Broadcast events so the TUI session sidebar shows the child.
            # Without these broadcasts, the child session is persisted but invisible
            # in the TUI until the next session.sync() REST call.
            await self.broadcast_event(SessionCreatedEvent.create(session))
            await self.broadcast_event(SessionUpdatedEvent.create(session))
            return session

    # 2. Fallback: create new session (for sessions not created via create_child_session)
    # ... existing creation logic with project_id/cwd inheritance ...
    # This path still calls store.save() because no prior SessionData exists
```

**New method**: `_session_from_session_data()` maps `SessionData` to OpenCode `Session`:

```python
def _session_from_session_data(self, session_data: SessionData) -> Session:
    """Create an OpenCode Session UI object from a persisted SessionData."""
    now = now_ms()
    return Session(
        id=session_data.session_id,
        project_id=session_data.project_id or helpers.compute_project_id(self.working_dir),
        directory=session_data.cwd or self.working_dir,
        title=session_data.title or "New Session",
        version=session_data.version,
        time=TimeCreatedUpdated(
            created=int(session_data.created_at.timestamp() * 1000) if session_data.created_at else now,
            updated=now,
        ),
        parent_id=session_data.parent_id,
    )
```

**Key constraint**: This mapping must NOT call `store.save()` — the `SessionData` is already persisted and correct.

**Known limitation**: The store-first path in `ensure_session()` skips `bind_agent_to_session()`. For child sessions created by delegation providers, this is correct — the child agent is already bound by the delegation provider. However, for top-level sessions recovered from the store (e.g., after server restart), the agent will NOT be re-bound. This is acceptable for the current scope because: (1) server restart recovery is a separate concern, (2) the TUI can still display the session (it has a `Session` object), and (3) the session's interaction history is preserved in the store. If agent re-binding is needed on restart, it should be addressed in a future RFC.

#### 10. Type-Safe Agent Type Resolution — Two Separate Concepts

**Problem**: The RFC previously used `getattr(node, "agent_type", "native")` which violates the project's type-safety rules (AGENTS.md: "never resort to getattr"). Additionally, the original `get_source_type()` helper conflated two distinct domains:

| Concept | Domain | Values | Used For |
|---------|--------|--------|----------|
| **Agent implementation type** | `AgentTypeLiteral` | `"native"`, `"acp"`, `"agui"`, `"claude"`, `"codex"` | `SessionData.agent_type`, `create_child_session(agent_type=...)` |
| **Delegation source type** | `Literal["agent", "team_parallel", "team_sequential"]` | `"agent"`, `"team_parallel"`, `"team_sequential"` | `SpawnSessionStart.source_type`, `SubAgentEvent.source_type` |

A native agent's implementation type is `"native"`, but its delegation source type is `"agent"`. A Team's implementation type could be `"team"`, but its source type is `"team_parallel"`. Conflating these produces **runtime type errors** — `"native"` is not a valid `source_type` value.

**Solution**: Provide two separate, type-safe functions with distinct return types:

```python
# src/wolfharness/messaging/messagenode.py

class MessageNode:
    # ... existing code ...

    @property
    def agent_type(self) -> str:
        """Return the agent implementation type string for this node.

        Used for SessionData.agent_type and create_child_session(agent_type=...).
        Returns values from AgentTypeLiteral domain.
        """
        # Import locally to avoid circular imports
        from wolfharness.agents.base_agent import BaseAgent
        from wolfharness.delegation.team import Team
        from wolfharness.delegation.teamrun import TeamRun

        if isinstance(self, Team | TeamRun):
            return "team"
        if isinstance(self, BaseAgent):
            return self.AGENT_TYPE  # ClassVar on BaseAgent subclasses
        return "native"
```

```python
# src/wolfharness/agents/helpers.py

from typing import Literal
from wolfharness.messaging.messagenode import MessageNode

# The return type matches SpawnSessionStart.source_type and SubAgentEvent.source_type
SourceType = Literal["agent", "team_parallel", "team_sequential"]

def get_source_type(node: MessageNode) -> SourceType:
    """Return the delegation source type for this node.

    Used for SpawnSessionStart.source_type and SubAgentEvent.source_type.
    Returns values from the source_type Literal domain, NOT AgentTypeLiteral.

    This is a DIFFERENT concept from node.agent_type:
    - agent_type = implementation type (native, acp, claude, codex, team)
    - source_type = delegation type (agent, team_parallel, team_sequential)
    """
    from wolfharness.delegation.team import Team
    from wolfharness.delegation.teamrun import TeamRun
    from wolfharness.agents.base_agent import BaseAgent

    match node:
        case Team():
            return "team_parallel"
        case TeamRun():
            return "team_sequential"
        case BaseAgent():
            return "agent"
        case _:
            return "agent"
```

**Usage rules**:
- `get_source_type(node)` → used for `SpawnSessionStart.source_type`, `SubAgentEvent.source_type` (returns `"agent"` / `"team_parallel"` / `"team_sequential"`)
- `node.agent_type` → used for `SessionData.agent_type`, `create_child_session(agent_type=...)` (returns `"native"` / `"acp"` / `"team"` etc.)

**Pre-existing codebase bug**: `team.py:32` and `teamrun.py:28` import `SubAgentType` from `wolfharness.agents.events.events` inside `TYPE_CHECKING` blocks, but `SubAgentType` does **not exist** in that module. This is a type-checking error that's silently ignored at runtime. The new `SourceType` type alias and `get_source_type()` function in `helpers.py` replace this broken import. **Implementation action**: Update the `TYPE_CHECKING` imports in `team.py` and `teamrun.py` to use `from wolfharness.agents.helpers import SourceType` instead of the broken `SubAgentType` import.

**`AgentTypeLiteral` domain note**: `AgentTypeLiteral = Literal["native", "acp", "agui", "claude", "codex"]` does NOT include `"team"`. The `MessageNode.agent_type` property returns `"team"` for Team/TeamRun instances, which is outside the `AgentTypeLiteral` domain. This is acceptable because `SessionData.agent_type` is typed as `str | None` (not `AgentTypeLiteral`), so storing `"team"` is valid at runtime and in the persistence layer. The `"team"` value provides useful semantic information about which kind of node created the session. If stricter typing is desired in the future, `AgentTypeLiteral` could be extended to include `"team"`.

**Behavioral change — wildcard case**: The current production code in `team.py:197` raises `ValueError` for unexpected node types: `raise ValueError(f"Unexpected node type: {type(node)}")`. The RFC's `get_source_type()` uses `case _: return "agent"` as a defensive default instead. This is a deliberate trade-off: crashing on unexpected types vs. gracefully handling future node types. The defensive default is preferred for robustness, but it means new `MessageNode` subclasses will silently be treated as agents until `get_source_type()` is updated. **Implementation action**: Add a `logging.warning(f"Unknown node type {type(node).__name__} in get_source_type(), defaulting to 'agent'")` in the `case _` branch to aid debugging.

This ensures all `getattr(node, "agent_type", "native")` calls across SubagentTools, WorkersTools, Team, and TeamRun are replaced with the type-safe, domain-appropriate function. The current codebase already uses this `match` pattern correctly in team.py:189-197, teamrun.py:278-284, subagent_tools.py:318-326, and workers.py:110-116.

#### 11. Parent-Child ID Relationship Clarification

`BaseAgent.run_stream()` accepts BOTH `parent_id: str | None` and `parent_session_id: str | None`. These are **independent parameters** serving different purposes:

| Parameter | Purpose | Set By | Used In |
|-----------|---------|--------|---------|
| `parent_session_id` | Delegation hierarchy — identifies the parent in the session tree | Delegation providers (SubagentTools, Team) | `SpawnSessionStart.parent_session_id`, `SubAgentEvent.parent_session_id` |
| `parent_id` | ChatMessage threading — links a child conversation message to a parent message | Callers that need explicit message-level threading (rare) | ChatMessage `parent_id` field, conversation history |

**Key distinction**: `parent_session_id` is a **session-level** concept (which session spawned this session?), while `parent_id` is a **message-level** concept (which message does this message reply to?). Zero code in `BaseAgent.run_stream()` connects them — they are not auto-propagated.

**Convention for delegation providers**: Providers set `parent_session_id` on the `run_stream()` call. The `parent_id` parameter is left as `None` (default) for delegation scenarios, because delegated agents start a new conversation thread. If a provider needs explicit message-level threading, it would set `parent_id` independently.

**How `Session.parent_id` (TUI tree) is populated**: The OpenCode `ensure_session()` method maps `SessionData.parent_id` → `Session.parent_id`. Since `create_child_session()` stores `parent_session_id` as `SessionData.parent_id`, the TUI tree view correctly shows the parent-child relationship through the `SessionData` persistence path, not through `BaseAgent.run_stream()` parameter propagation.

#### 12. Team Flat Hierarchy Design Decision

**Decision**: Team and TeamRun create child sessions for their members **directly under the calling session**. The Team/TeamRun itself does NOT get a `SessionData` entry. This produces a **flat hierarchy** rather than a nested one:

```
Calling Session (parent)
  ├─ Member A (child)    ← parent_id = calling session
  ├─ Member B (child)    ← parent_id = calling session
  └─ Member C (child)    ← parent_id = calling session
```

vs. nested hierarchy (NOT chosen):
```
Calling Session (parent)
  └─ Team Session (intermediate)  ← Would need its own SessionData
      ├─ Member A (child)
      ├─ Member B (child)
      └─ Member C (child)
```

**Rationale**: Teams are ephemeral coordination constructs, not first-class execution units. Creating an intermediate `SessionData` for the team would add complexity without clear benefit — the team has no model, no tokens, no interaction history of its own. Members' events are already grouped via `SubAgentEvent.source_name`. A flat hierarchy also simplifies `get_child_sessions()` queries.

#### 13. Depth Overflow Guard

Providers must enforce a maximum delegation depth to prevent unbounded recursion:

```python
# src/wolfharness/agents/base_agent.py or constants

MAX_DELEGATION_DEPTH = 10  # Hard cap on delegation nesting

# In delegation providers (SubagentTools, Team, TeamRun):
current_depth = ctx.run_ctx.depth if ctx.run_ctx is not None else 0
if current_depth >= MAX_DELEGATION_DEPTH:
    raise DelegationDepthError(
        f"Maximum delegation depth ({MAX_DELEGATION_DEPTH}) exceeded. "
        f"Current depth: {current_depth}"
    )
child_depth = current_depth + 1
```

`DelegationDepthError` is defined in `src/wolfharness/agents/exceptions.py` alongside other agent-specific exceptions. `MAX_DELEGATION_DEPTH` is a module-level constant in the same file.

```python
# src/wolfharness/agents/exceptions.py

class DelegationDepthError(Exception):
    """Raised when delegation depth exceeds MAX_DELEGATION_DEPTH."""
    pass

MAX_DELEGATION_DEPTH: int = 10
```

The server-side depth cap (currently 5 in OpenCode's `EventProcessor`) remains as an independent safety net. The provider-side cap is higher (10) because it counts logical delegation depth, while the server cap counts streaming nesting depth.

### Ambiguity Resolutions

#### AR-1: Incomplete SessionData from store.load()

When `store.load()` returns a `SessionData` with `project_id=None` or `cwd=None`, the `_session_from_session_data()` mapping falls back to the server's defaults:

```python
# In _session_from_session_data():
project_id=session_data.project_id or compute_project_id(self.working_dir),
directory=session_data.cwd or self.working_dir,
```

This matches the current behavior of `ensure_session()` when creating new sessions. The fallback is safe because `working_dir` is always available on the server.

#### AR-2: Depth Propagation Across Protocol Boundaries

ACP and AG-UI agents have their own `_run_stream_once()` implementations but do NOT override `run_stream()`. The `depth: int = 0` parameter is only needed on `BaseAgent.run_stream()` (which all subclasses inherit). ACP/AG-UI agents cannot propagate depth to their remote processes — the remote agent has no concept of AgentPool's depth tracking. Depth is only tracked within the AgentPool framework layer. Events from remote agents always appear at the depth set by the Team/TeamRun wrapper.

#### AR-3: Orphaned Child Session Cleanup

If `create_child_session()` succeeds but the delegation fails before emitting `SpawnSessionStart`, the persisted `SessionData` becomes an orphan — a session entry with no corresponding events. This is **benign**: the session appears as an empty session in the TUI sidebar. Cleanup of orphan sessions is deferred to a future RFC. The rationale: orphan detection requires correlating session creation with event emission, which adds complexity not justified by the low impact of empty sessions.

#### AR-4: ACPSessionManager Top-Level vs Child Sessions

Resolved in Component #8: `ACPSessionManager.create_session()` now accepts `parent_session_id`. When provided, it calls `create_child_session()` (with inheritance). When not provided (top-level ACP sessions), it creates `SessionData` directly with `project_id` computed from `cwd`. This two-path approach mirrors the ACP protocol's distinction between new client sessions and subagent delegation.

### Edge Cases

#### EC-1: Team with 0 Members

`Team.run_stream()` with `all_nodes = list(self.nodes)` producing an empty list results in `as_generated([])` yielding no events. This is correct — no child sessions are created, no `SpawnSessionStart` emitted. No special handling needed.

#### EC-2: Concurrent ensure_session() with Same session_id

If two `ensure_session()` calls race for the same `session_id`, the first one wins (persists to store or loads from store). The second call finds the session already in `self.sessions` and returns the cached instance with a `SessionUpdatedEvent` broadcast. This is idempotent and safe.

#### EC-3: create_child_session() During Pool Shutdown

If `create_child_session()` is called while the pool is shutting down, `store.save()` may fail or be a no-op (depending on the store implementation). This is an edge case that should be handled by the pool's shutdown sequence — pool shutdown should wait for in-flight delegations to complete before closing the store. No special handling in `create_child_session()` itself.

#### EC-4: Depth Overflow Bypass When run_ctx is None

The depth overflow guard `(ctx.run_ctx.depth if ctx.run_ctx is not None else 0) + 1` defaults to depth=1 when `run_ctx` is None. This means a tool called outside `run_stream()` (where `run_ctx` is None) starts at depth=1, not depth=0. This is acceptable because: (1) tools called outside `run_stream()` are rare, (2) depth=1 is a safe starting point (the tool IS delegating), and (3) the `MAX_DELEGATION_DEPTH` cap still applies on subsequent delegations.

#### EC-5: TeamRun First Member Fails with require_all=False

When `require_all=False` and the first member fails, `TeamRun` continues to the next member. Each member independently creates its child session. The failed member's child session is orphaned (has `SessionData` but no events). This matches the orphan handling described in AR-3 — benign, no cleanup needed.

#### EC-6: Team Member Raises Exception Mid-Stream

If a Team member's `run_stream()` raises an exception after emitting some events, the `SpawnSessionStart` was already emitted but no `StreamCompleteEvent` follows. Since Team is parallel, other members continue executing. The partial child session is orphaned — it has `SessionData` (from `create_child_session()`) and some events, but no clean termination. This is benign per AR-3. The `as_generated()` utility handles the exception and continues yielding from other members.

#### EC-7: create_child_session() with Non-Existent parent_session_id

`create_child_session()` loads the parent via `store.load(parent_session_id)`. If this returns `None` (parent not in store — e.g., parent is in-memory only), the child inherits `project_id=None` and `cwd=None`. This is handled by the `_session_from_session_data()` fallback (`project_id or compute_project_id(working_dir)`) when the session is later loaded by `ensure_session()`. The fallback is safe because `working_dir` is always available. See Open Question 6 for soft validation recommendation.

#### EC-8: AgentContext.create_child_session() with agent_pool=None AND session_id=None

When a tool is called outside a pool context (`node.agent_pool is None`) and the node has no `session_id`, the fallback path uses `generate_session_id()` for both the `parent_session_id` and the `child_session_id` returned by `create_child_session()`. This creates two disconnected new IDs with no persistence. The generated `child_session_id` is still valid for event emission (`SpawnSessionStart`, `SubAgentEvent`). The generated `parent_session_id` may differ from any actual parent session, but since no persistence occurs, the inconsistency is confined to the event stream. The session hierarchy can be reconstructed by `ensure_session()` when a server is attached.

### Data Model Changes

No schema changes to `SessionData`, `SpawnSessionStart`, or `SubAgentEvent`. The enhancement is in how these are populated and when they are persisted.

### API Changes

No public API changes. New `AgentContext.create_child_session()` method is additive.

---

## Security Considerations

### Threat Analysis

| Threat | Impact | Likelihood | Mitigation |
|--------|--------|------------|------------|
| Session ID collision | Medium | Very Low | `identifier.ascending()` uses monotonic counter + random component |
| Session store corruption | Medium | Low | `SessionStore.save()` overwrites by session_id; ensure_session() must NOT re-save SessionData that was already persisted by create_child_session() to avoid data loss |
| Unauthorized parent_id access | Low | Low | `create_child_session()` loads parent but does NOT validate it exists — silently inherits `project_id=None`/`cwd=None` if parent missing. See Open Question 6 |
| Orphan child sessions | Low | Medium | If create_child_session() succeeds but SpawnSessionStart emission fails, the persisted SessionData has no corresponding event. Benign — appears as empty session. No cleanup needed (future RFC) |

### Security Measures

- [x] Session IDs generated using `identifier.ascending()` with secure random component
- [x] No sensitive data in `SessionData.metadata`
- [ ] `create_child_session()` should validate parent session exists — currently silently proceeds with None inheritance if parent not in store (see Open Question 6)

---

## Implementation Plan

### Phases

#### Phase 1: Core Adaptation (2 days)

- **Scope**: Add `AgentContext.create_child_session()`, adapt SubagentTools, WorkersTools, and ACPSessionManager
- **Deliverables**:
  - `AgentContext.create_child_session()` method
  - SubagentTools uses `create_child_session()` instead of `identifier.ascending()`
  - WorkersTools uses `create_child_session()` in both code paths
  - ACPSessionManager uses `create_child_session()` instead of direct `store.save()`
  - Unit tests verifying `SessionStore` now contains child sessions with correct project_id/cwd
- **Dependencies**: None

#### Phase 2: Team/TeamRun Session Support (2 days)

- **Scope**: Add session propagation to Team and TeamRun
- **Deliverables**:
  - Team.run_stream() creates child sessions and emits SpawnSessionStart
  - TeamRun.run_stream() creates child sessions and emits SpawnSessionStart
  - SubAgentEvent instances from teams carry session IDs
  - Depth passed via `run_stream(depth=...)` kwargs (not instance attribute)
  - None guard for `self.agent_pool` (team outside pool → generate_session_id() without persistence)
  - Unit tests for team session hierarchy
- **Dependencies**: Phase 1 (for pattern validation)

#### Phase 3: Cleanup & Consistency (1 day)

- **Scope**: Fix dead code, simplify ensure_session(), add consistency tests
- **Deliverables**:
  - Remove or connect `AgentRunContext.session_id` dead field
  - `ensure_session()` loads from SessionStore first (skipping redundant store.save())
  - Add `_session_from_session_data()` mapping method to ServerState
  - Remove `identifier.ascending("session")` calls from providers (now in SessionManager)
   - Replace `getattr(ctx, "current_depth", 0)` with `(ctx.run_ctx.depth if ctx.run_ctx is not None else 0) + 1` in SubagentTools
  - Test asserting `RunStartedEvent.session_id == SpawnSessionStart.child_session_id`
  - Integration tests with nested delegation
- **Dependencies**: Phase 1, Phase 2

### Milestones

| Milestone | Description | Target | Status |
|-----------|-------------|--------|--------|
| Core Adaptation | SubagentTools + WorkersTools + ACPSessionManager use create_child_session() | Day 2 | Not Started |
| Team Support | Team/TeamRun propagate sessions with depth kwargs | Day 4 | Not Started |
| Cleanup & Consistency | Dead code removal, ensure_session() fix, consistency tests | Day 5 | Not Started |

### Rollback Strategy

Each phase is independently revertible:

1. **Phase 1**: Revert SubagentTools/WorkersTools/ACPSessionManager to original session creation
2. **Phase 2**: Revert Team/TeamRun changes (they currently work without session propagation)
3. **Phase 3**: Revert ensure_session() changes, restore AgentRunContext.session_id

No data migration required — `SessionData` schema is unchanged. Extra persisted child sessions are benign if rollback occurs.

---

## Open Questions

1. **~~How does Team/TeamRun access `pool.sessions`?~~**
   - Context: Team and TeamRun are `MessageNode` subclasses, not tools with `AgentContext`. They may need a reference to `pool` or `SessionManager`.
   - **RESOLVED**: `MessageNode` has `self.agent_pool: AgentPool | None` (messagenode.py:80). Teams use `self.agent_pool.sessions.create_child_session()` with a `None` guard. When `agent_pool is None` (team created outside a pool), fall back to `generate_session_id()` without persistence.

2. **Should `create_child_session()` also emit `SpawnSessionStart`?**
   - Context: Currently session creation and event emission are separate. Merging them would guarantee they're always paired, but couples persistence to event system.
   - Owner: Architecture
   - Status: Open — recommend keeping separate for now; `SpawnSessionStart` is a protocol-level concern

3. **~~Depth tracking for Team/TeamRun~~**
   - Context: SubagentTools tracks depth via `getattr(ctx, "current_depth", 0)`. Teams don't have `AgentContext`. How should depth be incremented?
   - **RESOLVED**: Depth is stored on `AgentRunContext.depth` (set by `BaseAgent.run_stream(depth=...)`). SubagentTools reads it via `ctx.run_ctx.depth` (with None guard: `(ctx.run_ctx.depth if ctx.run_ctx is not None else 0) + 1`). Team/TeamRun receive depth via `run_stream(depth=...)` param. This avoids per-instance state (which would be incorrect for concurrent team execution), eliminates the `getattr` anti-pattern, and uses the correct access path (`ctx.run_ctx`, not `ctx.deps` which resolves to user-provided `TDeps`).

4. **ACPSessionManager adaptation**
   - Context: `ACPSessionManager` directly calls `pool.sessions.store.save()` instead of `create_child_session()`, losing project_id/cwd inheritance. Should it be adapted too?
   - Owner: Architecture
   - Status: Open — **must be adapted as part of Phase 1, not deferred**. After this RFC, there will be two session creation paths: `create_child_session()` (correct, with inheritance) and `ACPSessionManager` (incorrect, without inheritance). Leaving this inconsistency creates a maintenance burden where developers must remember which path handles inheritance correctly. **Recommendation**: Adapt `ACPSessionManager` to call `create_child_session()` in Phase 1. This is a small change (replace direct `store.save()` call with `create_child_session()`).

5. **`RunStartedEvent` consistency after adaptation**
    - Context: `BaseAgent.run_stream()` emits `RunStartedEvent(session_id=..., parent_session_id=...)` independently from `SpawnSessionStart`. After adaptation, the `session_id` kwarg passed to `run_stream()` is the `child_session_id` from `create_child_session()`, and `RunStartedEvent` will carry this same ID. This is consistent but should be verified: `RunStartedEvent.session_id` should always equal `SpawnSessionStart.child_session_id` for the same child run.
    - Owner: Implementation
    - Status: Open — add explicit assertion in tests (see TG-4)

6. **Should `create_child_session()` validate parent existence?**
    - Context: Currently, `create_child_session()` loads the parent session but does NOT validate it exists. If `store.load(parent_session_id)` returns `None`, the child inherits `project_id=None` and `cwd=None`. This contradicts the original security claim in this RFC. The OpenCode server sometimes creates sessions in-memory (`self.sessions` dict) without persisting them to `SessionStore` — so `store.load()` returns `None` even though the parent exists. Adding a hard validation would break this case.
    - Owner: Architecture
    - Status: Open — recommend: add **soft** validation (log warning when parent not found, don't raise). Hard validation should be deferred until all server-side sessions are guaranteed to be persisted.

7. **~~ACP top-level session creation semantics~~**
   - Context: `ACPSessionManager.create_session()` creates **top-level** sessions (no `parent_id`). The RFC says to adapt it to use `create_child_session()`, but that method requires `parent_session_id`. ACP sessions without a parent should still be created as top-level.
   - **RESOLVED**: Component #8 handles this with a two-path approach: when `parent_session_id` is provided, call `create_child_session()` (with inheritance); when not provided, create `SessionData` directly with `project_id` computed from `cwd`. No need for a separate `create_top_level_session()` method.

## Test Specifications

### TG-1: SessionData Field Value Assertions

Unit tests must verify that child sessions persisted via `create_child_session()` have correct field values:

```python
async def test_create_child_session_inherits_parent_fields():
    # Setup parent session
    parent_data = SessionData(session_id="parent_1", project_id="proj_abc", cwd="/work")
    await store.save(parent_data)

    # Create child
    child_id = await pool.sessions.create_child_session(
        parent_session_id="parent_1", agent_name="coder", agent_type="native"
    )
    child_data = await store.load(child_id)

    assert child_data is not None
    assert child_data.parent_id == "parent_1"
    assert child_data.project_id == "proj_abc"  # inherited
    assert child_data.cwd == "/work"             # inherited
    assert child_data.agent_name == "coder"
    assert child_data.agent_type == "native"
    assert child_data.pool_id is not None
```

Edge cases to test:
- Parent has `project_id=None` → child gets `None` (no fallback in `create_child_session()`)
- Parent not in store → child gets `project_id=None, cwd=None` (silent None inheritance)
- `agent_name` matches the parameter passed

### TG-2: `ensure_session()` Skip-Store-Save Path

Verify that `ensure_session()` does NOT overwrite data created by `create_child_session()`:

```python
async def test_ensure_session_skips_save_when_already_persisted():
    # Create via create_child_session (correct data)
    child_id = await pool.sessions.create_child_session(
        parent_session_id="parent_1", agent_name="coder", agent_type="native"
    )
    original = await store.load(child_id)
    assert original.agent_type == "native"
    assert original.pool_id == pool.manifest.name

    # Call ensure_session
    session = await state.ensure_session(child_id, parent_id="parent_1")

    # Verify original data preserved (not overwritten)
    after = await store.load(child_id)
    assert after.agent_type == "native"       # NOT overwritten to None
    assert after.pool_id == pool.manifest.name  # NOT overwritten to config path
```

### TG-3: None Guard for Out-of-Pool Team

Verify that a Team created outside a pool still works:

```python
async def test_team_without_pool_generates_session_without_persistence():
    team = Team([agent_a, agent_b], name="orphan_team")
    assert team.agent_pool is None

    events = []
    async for event in team.run_stream("test"):
        events.append(event)

    # Assert SpawnSessionStart was emitted
    spawn_events = [e for e in events if isinstance(e, SpawnSessionStart)]
    assert len(spawn_events) >= 1
    # Assert child_session_id was generated (not None)
    assert spawn_events[0].child_session_id is not None
    # Session will be persisted reactively by ensure_session() if a server is attached
```

### TG-4: `RunStartedEvent` Consistency

Verify that `RunStartedEvent.session_id` matches `SpawnSessionStart.child_session_id`:

```python
async def test_run_started_matches_spawn_session():
    async with AgentPool("config.yml") as pool:
        agent = pool.get_agent("coordinator")
        events = []
        async for event in agent.run_stream("delegate to worker"):
            events.append(event)

        spawn_events = [e for e in events if isinstance(e, SpawnSessionStart)]
        for spawn in spawn_events:
            # Find corresponding RunStartedEvent in SubAgentEvents
            sub_events = [
                e for e in events
                if isinstance(e, SubAgentEvent)
                and e.child_session_id == spawn.child_session_id
            ]
            for sub in sub_events:
                if isinstance(sub.event, RunStartedEvent):
                    assert sub.event.session_id == spawn.child_session_id
```

### TG-5: `opencode_to_session_data()` Overwrite Prevention

The specific data loss scenario that the store-first path prevents:

```python
async def test_ensure_session_does_not_overwrite_create_child_session_data():
    # Create via create_child_session (correct data)
    child_id = await pool.sessions.create_child_session(
        parent_session_id="parent_1", agent_name="coder", agent_type="native"
    )
    original = await store.load(child_id)
    assert original.agent_type == "native"
    assert original.pool_id == pool.manifest.name

    # Old ensure_session() would call opencode_to_session_data() + store.save()
    # which overwrites with different pool_id and agent_type.
    # New ensure_session() loads from store first and skips save.
    await state.ensure_session(child_id, parent_id="parent_1")

    # Verify original data preserved
    after = await store.load(child_id)
    assert after.agent_type == "native"
    assert after.pool_id == pool.manifest.name
```

### TG-6: Nested Delegation Depth Propagation

Verify that depth increments correctly across A→B→C delegation chain:

```python
async def test_nested_delegation_depth():
    # Agent A delegates to Agent B which delegates to Agent C
    # Expected depth: A=0, B=1, C=2
    events = []
    async for event in coordinator.run_stream("delegate to B, who delegates to C"):
        events.append(event)

    spawn_events = sorted(
        [e for e in events if isinstance(e, SpawnSessionStart)],
        key=lambda e: e.depth,
    )
    assert len(spawn_events) >= 2
    assert spawn_events[0].depth == 1  # B spawned at depth 1
    assert spawn_events[1].depth == 2  # C spawned at depth 2
```

### TG-7: SubagentTools→Team→Member Delegation Chain

Verify session hierarchy is preserved when SubagentTools delegates to a Team:

```python
async def test_subagent_to_team_delegation_chain():
    # Coordinator (SubagentTools) → Team → Member A, Member B
    events = []
    async for event in coordinator.run_stream("run team_x"):
        events.append(event)

    # Verify SpawnSessionStart for team members
    spawn_events = [e for e in events if isinstance(e, SpawnSessionStart)]
    assert len(spawn_events) >= 2  # One per team member

    # All members should share the same parent_session_id (the coordinator's session)
    parent_ids = {e.parent_session_id for e in spawn_events}
    assert len(parent_ids) == 1  # Flat hierarchy — all under coordinator
```

### TG-8: Single SpawnSessionStart Per Delegation

Catch dual emission bug in SubagentTools sync mode:

```python
async def test_single_spawn_per_delegation():
    """Regression test: SubagentTools must emit exactly one SpawnSessionStart
    per delegation, not one in task() and another in _stream_task()."""
    events = []
    async for event in coordinator.run_stream("delegate to worker"):
        events.append(event)

    spawn_events = [e for e in events if isinstance(e, SpawnSessionStart)]
    # Each child agent should produce exactly ONE SpawnSessionStart
    session_ids = [e.child_session_id for e in spawn_events]
    assert len(session_ids) == len(set(session_ids)), (
        f"Duplicate SpawnSessionStart detected: {session_ids}"
    )
```

### TG-9: ctx.run_ctx None Fallback in SubagentTools

Verify depth defaults to 0 when `run_ctx` is None:

```python
async def test_depth_fallback_when_run_ctx_none():
    """If ctx.run_ctx is None (tool called outside run_stream context),
    depth should default to 0, not raise AttributeError."""
    ctx = AgentContext(node=mock_node)  # run_ctx not set
    depth = (ctx.run_ctx.depth if ctx.run_ctx is not None else 0) + 1
    assert depth == 1  # Defaults to 0 + 1
```

### TG-10: ACPSessionManager Adaptation Test

Verify ACPSessionManager uses `create_child_session()` with inheritance:

```python
async def test_acp_session_manager_uses_create_child_session():
    parent_data = SessionData(session_id="acp_parent", project_id="proj_x", cwd="/work")
    await store.save(parent_data)

    session_id = await acp_session_manager.create_session(
        agent_name="acp_agent",
        parent_session_id="acp_parent",
    )

    child_data = await store.load(session_id)
    assert child_data is not None
    assert child_data.parent_id == "acp_parent"
    assert child_data.project_id == "proj_x"  # Inherited
    assert child_data.cwd == "/work"           # Inherited
```

### TG-11: ensure_session() Store-Miss But Memory-Hit

Verify no duplicate Session objects when session is in memory but not in store:

```python
async def test_ensure_session_store_miss_memory_hit():
    """If session exists in self.sessions dict but not in SessionStore,
    ensure_session() should return the in-memory version without error."""
    session = Session(id="test_1", project_id="proj", directory="/work")
    state.sessions["test_1"] = session  # In memory only

    result = await state.ensure_session("test_1")
    assert result.id == "test_1"
    # Should not attempt store.load() or create new session
```

### TG-12: pool_id Consistency in ensure_session() Create-Fresh Path

When `ensure_session()` creates a new session (store-first path misses), verify `pool_id` uses `manifest.name`:

```python
async def test_ensure_session_fresh_uses_manifest_name():
    """When ensure_session() creates a new session (not in store),
    the resulting SessionData should use manifest.name as pool_id,
    not config_file_path."""
    session = await state.ensure_session("new_session_1")
    session_data = await store.load("new_session_1")

    # pool_id should come from manifest.name, not config_file_path
    assert session_data.pool_id == pool.manifest.name
    assert session_data.pool_id != str(pool.manifest.config_file_path)
```

### TG-13: Depth Overflow Exception

Verify `DelegationDepthError` is raised when depth exceeds `MAX_DELEGATION_DEPTH`:

```python
async def test_depth_overflow_raises_error():
    """When delegation depth exceeds MAX_DELEGATION_DEPTH,
    DelegationDepthError must be raised."""
    # Simulate a context at max depth
    run_ctx = AgentRunContext(deps=ctx, depth=MAX_DELEGATION_DEPTH)
    ctx.run_ctx = run_ctx

    with pytest.raises(DelegationDepthError, match="Maximum delegation depth"):
        # SubagentTools/Team/TeamRun should check depth before delegating
        child_depth = (ctx.run_ctx.depth if ctx.run_ctx is not None else 0) + 1
        if ctx.run_ctx.depth >= MAX_DELEGATION_DEPTH:
            raise DelegationDepthError(...)
```

### TG-14: _stream_task() Single Emission After Fix

Verify that after removing `SpawnSessionStart` from `_stream_task()`, only one is emitted per delegation:

```python
async def test_stream_task_single_emission():
    """After removing SpawnSessionStart from _stream_task(),
    task() emits exactly one SpawnSessionStart."""
    events = []
    async for event in coordinator.run_stream("delegate to worker"):
        events.append(event)

    spawn_events = [e for e in events if isinstance(e, SpawnSessionStart)]
    assert len(spawn_events) == 1  # Exactly one per delegation
```

### TG-15: WorkersTools Depth Propagation

Verify WorkersTools uses `ctx.run_ctx.depth` instead of hardcoded `depth=1`:

```python
async def test_workers_tools_depth_propagation():
    """WorkersTools must compute depth from ctx.run_ctx, not hardcode 1."""
    # Setup coordinator at depth 2
    run_ctx = AgentRunContext(deps=ctx, depth=2)
    ctx.run_ctx = run_ctx

    events = []
    async for event in coordinator.run_stream("run worker_x"):
        events.append(event)

    spawn_events = [e for e in events if isinstance(e, SpawnSessionStart)]
    assert len(spawn_events) >= 1
    assert spawn_events[0].depth == 3  # 2 + 1, not hardcoded 1
```

### TG-16: TeamRun require_all with Depth

Verify `require_all` parameter is preserved alongside `depth`:

```python
async def test_teamrun_require_all_with_depth():
    """TeamRun.run_stream() must accept both require_all and depth params."""
    teamrun = TeamRun([agent_a, agent_b], name="chain")
    events = []
    # Should not raise TypeError
    async for event in teamrun.run_stream("test", depth=1, require_all=False):
        events.append(event)
    # Depth should propagate
    spawn_events = [e for e in events if isinstance(e, SpawnSessionStart)]
    for spawn in spawn_events:
        assert spawn.depth == 2  # 1 + 1
```

### TG-17: ensure_session() Store-Miss Negative Case

When `store.load(session_id)` returns `None` but session_id is also not in memory, the fallback creation path must execute:

```python
async def test_ensure_session_store_miss_fallback():
    """When session is not in store AND not in memory,
    ensure_session() must create a new session via fallback path."""
    session_id = "unknown_session_xyz"
    assert session_id not in state.sessions
    assert await store.load(session_id) is None

    session = await state.ensure_session(session_id)
    assert session is not None
    assert session.id == session_id
    # Should have been persisted by fallback store.save()
    session_data = await store.load(session_id)
    assert session_data is not None
```

### TG-18: Nested SubAgentEvent Depth Preservation in Team

Verify that Team correctly handles already-wrapped SubAgentEvents from nested teams — depth must increment, not flatten:

```python
async def test_nested_subagent_event_depth_preservation():
    """When Team contains a nested TeamRun, SubAgentEvents from the inner
    team must have depth incremented, NOT overwritten to the outer team's depth."""
    # Team [TeamRun [A, B], C]
    inner_run = TeamRun([agent_a, agent_b], name="Sequential")
    outer_team = Team([inner_run, agent_c], name="Parallel")

    events = []
    async for event in outer_team.run_stream("test"):
        events.append(event)

    sub_events = [e for e in events if isinstance(e, SubAgentEvent)]
    # Inner team members should have depth > outer team members
    # (outer wraps with depth+1, inner already had its own depth)
    depths = [e.depth for e in sub_events]
    assert max(depths) > min(depths), f"Depth was flattened: {depths}"
```

### TG-19: Concurrent ensure_session() for Same session_id

Verify idempotency when two calls race:

```python
async def test_concurrent_ensure_session_same_id():
    """Two concurrent ensure_session() calls for the same ID
    must not produce duplicate Session objects."""
    # Pre-create session in store
    child_id = await pool.sessions.create_child_session(
        parent_session_id="parent_1", agent_name="coder", agent_type="native"
    )

    # Call ensure_session concurrently
    results = await asyncio.gather(
        state.ensure_session(child_id, parent_id="parent_1"),
        state.ensure_session(child_id, parent_id="parent_1"),
    )

    # Both should return the same Session object
    assert results[0].id == results[1].id
    assert len([s for s in state.sessions.values() if s.id == child_id]) == 1
```

### TG-20: Agent Subclass run_stream() Depth Param Acceptance

Verify that `BaseAgent.run_stream()` accepts the `depth` parameter (no subclass overrides exist):

```python
async def test_agent_subclass_run_stream_accepts_depth():
    """All agent types must accept depth= in run_stream() without TypeError."""
    for agent in [native_agent, acp_agent, claude_agent, agui_agent, codex_agent]:
        try:
            async for _ in agent.run_stream("test", depth=1):
                break  # Just verify first event, no TypeError
        except TypeError as e:
            if "depth" in str(e):
                pytest.fail(f"{type(agent).__name__}.run_stream() does not accept depth: {e}")
```

### TG-21: Orphaned Child Session Scenario

Verify that a child session persisted by `create_child_session()` but never receiving events appears as empty in the TUI:

```python
async def test_orphaned_child_session_is_benign():
    """If create_child_session() succeeds but delegation fails before events,
    the persisted SessionData should exist but have no events."""
    child_id = await pool.sessions.create_child_session(
        parent_session_id="parent_1", agent_name="coder", agent_type="native"
    )

    # Verify SessionData exists in store
    data = await store.load(child_id)
    assert data is not None
    assert data.parent_id == "parent_1"

    # Session appears in TUI as empty — this is benign
    session = await state.ensure_session(child_id)
    assert session.id == child_id
```

### TG-22: Mixed Agent Type Team (Native + ACP + TeamRun)

Verify session creation works for teams with mixed agent types:

```python
async def test_mixed_agent_type_team_sessions():
    """Team with Native, ACP, and TeamRun members must create
    correct SessionData for each type."""
    mixed_team = Team([native_agent, acp_agent, teamrun_member], name="Mixed")
    events = []
    async for event in mixed_team.run_stream("test"):
        events.append(event)

    spawn_events = [e for e in events if isinstance(e, SpawnSessionStart)]
    assert len(spawn_events) == 3

    # Verify each child session has correct agent_type
    for spawn in spawn_events:
        data = await store.load(spawn.child_session_id)
        assert data is not None
        # agent_type should match the node type
        assert data.agent_type in ("native", "acp", "team")
```

### TG-23: create_child_session() with Parent Having project_id=None

Verify that None inheritance is handled correctly (separate test, not just a comment):

```python
async def test_create_child_session_parent_project_id_none():
    """When parent session has project_id=None, child should inherit None
    (no fallback in create_child_session())."""
    parent_data = SessionData(session_id="parent_no_proj", project_id=None, cwd="/work")
    await store.save(parent_data)

    child_id = await pool.sessions.create_child_session(
        parent_session_id="parent_no_proj", agent_name="coder", agent_type="native"
    )
    child_data = await store.load(child_id)

    assert child_data.project_id is None  # Inherited None, not computed
    assert child_data.cwd == "/work"       # cwd still inherited
```

### TG-24: Depth Overflow When run_ctx is None

Verify depth overflow guard works even when `run_ctx` is None:

```python
async def test_depth_overflow_when_run_ctx_none():
    """Even when run_ctx is None (tool outside run_stream context),
    the depth overflow guard should prevent unbounded delegation."""
    ctx = AgentContext(node=mock_node)  # run_ctx not set
    current_depth = ctx.run_ctx.depth if ctx.run_ctx is not None else 0
    # At depth 0, delegation should proceed (0 < MAX_DELEGATION_DEPTH)
    assert current_depth < MAX_DELEGATION_DEPTH
    child_depth = current_depth + 1
    assert child_depth == 1  # Starts at 1 when run_ctx is None
```

### TG-25: MessageNode.agent_type Property and get_source_type() Return Correct Values

Verify both `agent_type` and `get_source_type()` return domain-appropriate values:

```python
async def test_messagenode_agent_type_and_source_type():
    """MessageNode.agent_type and get_source_type() must return
    values from their respective domains."""
    from wolfharness.agents.base_agent import BaseAgent
    from wolfharness.delegation.team import Team
    from wolfharness.delegation.teamrun import TeamRun

    native = Agent(name="native", model="test")
    team = Team([native], name="team")
    teamrun = TeamRun([native], name="chain")

    # agent_type returns implementation type (AgentTypeLiteral domain)
    assert native.agent_type == native.AGENT_TYPE  # e.g., "native"
    assert team.agent_type == "team"  # Implementation type
    assert teamrun.agent_type == "team"  # Implementation type

    # get_source_type returns delegation source type (source_type domain)
    from wolfharness.agents.helpers import get_source_type
    assert get_source_type(native) == "agent"       # NOT "native"!
    assert get_source_type(team) == "team_parallel"  # NOT "team"!
    assert get_source_type(teamrun) == "team_sequential"  # NOT "team"!
```

### TG-26: get_source_type() Type Compatibility with SpawnSessionStart

Verify that `get_source_type()` return values are valid for `source_type` fields:

```python
async def test_get_source_type_compatible_with_source_type_literal():
    """get_source_type() must return values accepted by
    SpawnSessionStart.source_type and SubAgentEvent.source_type."""
    from wolfharness.agents.helpers import get_source_type, SourceType

    for node in [native_agent, acp_agent, team, teamrun]:
        result = get_source_type(node)
        # SourceType is Literal["agent", "team_parallel", "team_sequential"]
        assert result in ("agent", "team_parallel", "team_sequential"), (
            f"get_source_type({type(node).__name__}) returned {result!r}, "
            f"which is not a valid source_type value"
        )
```

### TG-27: ACPSessionManager Top-Level Session Creation

Verify ACPSessionManager top-level path (without `parent_session_id`):

```python
async def test_acp_session_manager_top_level_session():
    """When parent_session_id is None, ACPSessionManager should
    create a top-level session with project_id computed from cwd."""
    session_id = await acp_session_manager.create_session(
        agent_name="acp_agent",
        parent_session_id=None,  # Top-level session
    )

    session_data = await store.load(session_id)
    assert session_data is not None
    assert session_data.parent_id is None  # No parent
    assert session_data.agent_type == "acp"
    assert session_data.project_id is not None  # Computed from cwd
```

### TG-28: get_source_type() Wildcard Path Behavior

Verify the `case _` branch in `get_source_type()` returns `"agent"` with a warning:

```python
async def test_get_source_type_wildcard_defaults_to_agent():
    """Unknown MessageNode subclasses should default to 'agent'
    with a warning log."""
    from wolfharness.agents.helpers import get_source_type
    from wolfharness.messaging.messagenode import MessageNode

    class CustomNode(MessageNode):
        name: str = "custom"

    node = CustomNode()
    with pytest.warns(UserWarning, match="Unknown node type"):
        result = get_source_type(node)
    assert result == "agent"
```

### TG-29: _DeprecatedSessionId Descriptor with dataclasses.asdict()

Verify the descriptor works correctly with `dataclasses.asdict()`:

```python
async def test_deprecated_session_id_asdict():
    """dataclasses.asdict() should work with the _DeprecatedSessionId
    descriptor, triggering deprecation warning on access."""
    from wolfharness.agents.context import AgentRunContext

    ctx = AgentRunContext(deps=None, depth=0)
    with pytest.warns(DeprecationWarning, match="session_id"):
        d = dataclasses.asdict(ctx)
    assert "session_id" in d
```

### TG-30: MessageNode.agent_type Circular Import Safety

Verify that importing `MessageNode.agent_type` doesn't trigger circular import errors:

```python
def test_messagenode_agent_type_no_circular_import():
    """MessageNode.agent_type property with local imports should not
    cause circular import errors at module load time."""
    import importlib
    import wolfharness.messaging.messagenode

    # Force re-import to detect circular dependency
    importlib.reload(wolfharness.messaging.messagenode)
    # Should not raise ImportError
```

### TG-31: create_child_session() pool_id Matches manifest.name

Verify that child sessions get the correct `pool_id`:

```python
async def test_child_session_pool_id_matches_manifest():
    """Child session pool_id should match pool.manifest.name,
    not config_file_path."""
    async with AgentPool(manifest) as pool:
        parent_id = pool.sessions.create_session("parent")
        child_id = await pool.sessions.create_child_session(
            parent_session_id=parent_id,
            agent_name="child_agent",
            agent_type="native",
        )
        child_data = await pool.sessions.store.load(child_id)
        assert child_data.pool_id == pool.manifest.name
```

### TG-32: ensure_session() Store-First Path Skips bind_agent_to_session

Verify that the store-first path in `ensure_session()` does NOT call `bind_agent_to_session`:

```python
async def test_ensure_session_store_first_skips_bind():
    """When ensure_session() loads a session from store, it should
    NOT call bind_agent_to_session (agent binding is handled by
    the delegation provider)."""
    # Pre-populate store with child session data
    await store.save(SessionData(
        session_id="child_from_store",
        parent_id="parent_1",
        agent_name="delegated_agent",
        agent_type="native",
    ))
    session = await state.ensure_session("child_from_store")
    assert session is not None
    # bind_agent_to_session should NOT have been called
    assert "child_from_store" not in state.bound_agents
```

---

## Decision Record

### Review Revision Log

**Rev 1** (2026-04-24): Dialectical codebase verification of Oracle/Metis review findings

| # | Original Claim | Verification Result | Revision Applied |
|---|---------------|--------------------|---------|
| 1 | Team/TeamRun use `self._pool` | ❌ They use `self.agent_pool` (from `MessageNode`). Fixed all pseudocode. | ✅ |
| 2 | `Session.from_session_data()` needed for ensure_session() | ❌ Insufficient — `Session` ≠ `SessionData`, and ensure_session() has 6 side-effects beyond mapping. Replaced with `_session_from_session_data()` that skips store.save(). | ✅ |
| 3 | `AgentRunContext.session_id` format needs fixing | ❌ The field is **dead code** — never read. Should be removed or connected, not reformatted. | ✅ |
| 4 | Double-save in ensure_session() is idempotent | ❌ Not idempotent — `opencode_to_session_data()` produces different field values than `create_child_session()`, causing data loss on overwrite. | ✅ |
| 5 | Depth on MessageNode as instance attribute | ⚠️ Wrong for concurrent execution. Changed to kwargs pattern (`depth` passed via `run_stream(**kwargs)`). | ✅ |
| 6 | ACPSessionManager can be deferred | ❌ Leaving two session creation paths creates maintenance burden. Moved to Phase 1. | ✅ |
| 7 | RunStartedEvent consistency not mentioned | Missing concern. Added Open Question 5 with test assertion requirement. | ✅ |

**Rev 2** (2026-04-24): Oracle/Metis Round 2 review — 3 must-fix issues + 5 hidden risks

| # | Finding | Source | Revision Applied |
|---|---------|--------|---------|
| 1 | `depth` kwarg crashes `BaseAgent.run_stream()` — no `**kwargs` catch-all | Oracle I-1/R-2 | ✅ Added `depth: int = 0` param to `BaseAgent.run_stream()` signature + `AgentRunContext.depth` field |
| 2 | `AgentContext.create_child_session()` calls `self.pool` which doesn't exist | Oracle I-4 | ✅ Changed to `self.node.agent_pool` with None guard |
| 3 | Team/TeamRun ignore `kwargs["session_id"]` — session hierarchy chain broken at Team boundaries | Metis HR-1 | ✅ Changed to `kwargs.pop("session_id", None) or self.session_id or generate_session_id()` |
| 4 | `_session_from_session_data()` skips broadcast events — TUI won't show child sessions | Oracle I-3, Metis HR-2 | ✅ Added `SessionCreatedEvent`/`SessionUpdatedEvent` broadcasts |
| 5 | `create_child_session()` does NOT validate parent exists — contradicts security claim | Metis HR-4 | ✅ Fixed security claim + added Open Question 6 for soft validation |
| 6 | `pool_id` mismatch + two-tier persistence not documented | Metis HR-3, HR-5 | ✅ Added trade-offs #5 and #6 |
| 7 | Depth propagation broken across SubagentTools ↔ Team boundary | Metis AMB-2 | ✅ Added `AgentRunContext.depth` field + mechanism specification |
| 8 | Missing test specifications | Metis TG-1–5 | ✅ Added Test Specifications section with 5 concrete tests |
| 9 | Removing `AgentRunContext.session_id` is public API break | Metis PE-1 | ✅ Changed to deprecation (Option C) instead of removal |

**Rev 3** (2026-04-24): Oracle/Metis Round 3 review — 2+3 must-fix, 5+4 should-fix

| # | Finding | Source | Revision Applied |
|---|---------|--------|---------|
| 1 | `ctx.deps.depth` is WRONG — must be `ctx.run_ctx.depth` with None guard (deps resolves to TDeps, not AgentRunContext) | Oracle M-1, Metis HR-1 | ✅ Fixed all pseudocode to use `(ctx.run_ctx.depth if ctx.run_ctx is not None else 0) + 1` |
| 2 | `AgentRunContext` construction never passes `depth` | Oracle M-2 | ✅ Added `depth=depth` to `AgentRunContext(deps=deps, depth=depth)` in run_stream AFTER pseudocode |
| 3 | `_stream_task()` in SubagentTools emits dual `SpawnSessionStart` | Metis HR-2 | ✅ Added `skip_spawn_event` parameter documentation |
| 4 | RFC "BEFORE" pseudocode wrong — actual signature has 12 explicit params, not `**kwargs` | Metis HR-3 | ✅ Showed actual parameter signature with depth inserted |
| 5 | `getattr(node, "agent_type")` violates type-safety rules | Oracle S-4, Metis HR-4 | ✅ Added `MessageNode.agent_type` property + `get_source_type()` helper with isinstance checks |
| 6 | `parent_session_id` double-generation when `node.session_id` is None | Oracle S-1 | ✅ Added `parent_session_id` param to `AgentContext.create_child_session()` |
| 7 | `parent_session_id_from_kwargs` dead code | Oracle S-2 | ✅ Removed from Team/TeamRun pseudocode |
| 8 | Team flat vs nested hierarchy not documented | Metis AMB-1 | ✅ Added Component #10 with design decision and rationale |
| 9 | `parent_id` vs `parent_session_id` relationship unexplained | Metis AMB-2 | ✅ Added Component #9 with relationship table and convention |
| 10 | All `run_stream()` overrides need `depth` param | Oracle S-5 | ✅ Added note in Component #4 AFTER pseudocode |
| 11 | Depth overflow — no max depth guard | Metis EC-1 | ✅ Added Component #11 with MAX_DELEGATION_DEPTH=10 |
| 12 | Open Question 3 text contradicts actual mechanism | Oracle S-3 | ✅ Fixed to reference `ctx.run_ctx.depth` |
| 13 | Section numbering error (two "#### 6.") | Oracle N-1 | ✅ Renumbered to 7, 8 |
| 14 | `get_source_type()` used but never defined | Oracle N-2 | ✅ Added Component #8 with full definition |
| 15 | Missing TG-6 through TG-12 test specifications | Metis | ✅ Added 7 new test specifications |

**Rev 4** (2026-04-24): Oracle/Metis Round 4 review — APPROVE_WITH_MINOR / 3 MUST-FIX

| # | Finding | Source | Revision Applied |
|---|---------|--------|---------|
| 1 | `self._agent_type` doesn't exist — should be `self.AGENT_TYPE` (ClassVar on BaseAgent) | Oracle MUST, Metis HR-6 | ✅ Fixed `MessageNode.agent_type` property to use `self.AGENT_TYPE` |
| 2 | Component #9 factual error — `BaseAgent.run_stream()` does NOT propagate `parent_session_id` → `parent_id`; they are independent parameters (session-level vs message-level) | Metis HR-5 | ✅ Rewrote Component #10 (renumbered) with correct distinction and explanation of how `Session.parent_id` is populated |
| 3 | `_stream_task()` pseudocode missing — `skip_spawn_event` described in text but no pseudocode; simpler approach: remove SpawnSessionStart from `_stream_task()` entirely | Oracle SHOULD, Metis HR-7 | ✅ Replaced `skip_spawn_event` with structural single-emission: `_stream_task()` never emits `SpawnSessionStart` |
| 4 | Duplicate "#### 8." numbering still broken — added Component #8 without renumbering | Oracle SHOULD | ✅ Renumbered 8→8, 8→9, 9→10, 10→11, 11→12 |
| 5 | `DelegationDepthError` referenced but undefined — no module location specified | Oracle SHOULD | ✅ Added definition in `src/wolfharness/agents/exceptions.py` with `MAX_DELEGATION_DEPTH` constant |
| 6 | `TeamRun.run_stream()` pseudocode drops `require_all: bool = True` parameter | Metis AMB-3 | ✅ Added `require_all` to TeamRun signature |
| 7 | `ensure_session()` in-memory path returns without broadcasting `SessionUpdatedEvent` — TUI regression vs current code | Metis AMB-4 | ✅ Added `SessionUpdatedEvent` broadcast to in-memory return path |
| 8 | WorkersTools hardcodes `depth=1` in four locations — breaks nested delegation | Metis AMB-6 | ✅ Added depth propagation specification: `(ctx.run_ctx.depth if ctx.run_ctx is not None else 0) + 1` |
| 9 | Added TG-13 through TG-17 test specifications | Metis | ✅ Added 5 new tests |

**Rev 5** (2026-04-24): Oracle/Metis Round 5 review — APPROVE_WITH_MINOR / 2 MUST-FIX

| # | Finding | Source | Revision Applied |
|---|---------|--------|------------------|
| 1 | Team/TeamRun pseudocode wraps ALL events uniformly with `depth=depth + 1` — flattens nested SubAgentEvent depth. Actual code distinguishes already-wrapped (increment existing) vs raw events (set from context). | Metis HR-1 | ✅ Added `isinstance(event, SubAgentEvent)` branch to Team and TeamRun pseudocode |
| 2 | ACPSessionManager adaptation has NO pseudocode — just says "must be adapted" | Metis HR-2 | ✅ Added Component #8 with full ACPSessionManager pseudocode showing top-level vs child session paths |
| 3 | Phase 3 deliverable says `kwargs.get("depth", 0)` but Component #2 specifies `ctx.run_ctx.depth` — contradiction | Oracle SHOULD | ✅ Fixed to `(ctx.run_ctx.depth if ctx.run_ctx is not None else 0) + 1` |
| 4 | Deprecation needs descriptor, not just docstring — dataclass field deprecation requires special handling | Oracle SHOULD | ✅ Added `_DeprecatedSessionId` descriptor class with `DeprecationWarning` |
| 5 | `task()` labeled "(sync)" but is async | Oracle SHOULD | ✅ Fixed to "(async)" |
| 6 | Metis ambiguities: incomplete SessionData, depth across protocols, orphan cleanup, ACPSessionManager top-level vs child | Metis AMB-1–4 | ✅ Added Ambiguity Resolutions section (AR-1 through AR-4) |
| 7 | Metis edge cases: 0-member team, concurrent ensure_session, pool shutdown, None run_ctx bypass, TeamRun partial fail | Metis EC-1–5 | ✅ Added Edge Cases section (EC-1 through EC-5) |
| 8 | Added TG-18 through TG-25 test specifications | Metis | ✅ Added 8 new tests |
| 9 | Component renumbering after adding Component #8 (ACPSessionManager) | N/A | ✅ Renumbered 9→10, 10→11, 11→12, 12→13 |

**Rev 6** (2026-04-24): Oracle/Metis Round 6 review — APPROVE_WITH_MINOR / 1 MUST-FIX

| # | Finding | Source | Revision Applied |
|---|---------|--------|------------------|
| 1 | `agent_type` / `get_source_type()` domain mismatch — `MessageNode.agent_type` returns values from `AgentTypeLiteral` ("native", "acp", "team"), but `source_type` on `SpawnSessionStart`/`SubAgentEvent` requires `Literal["agent", "team_parallel", "team_sequential"]`. Conflating two domains produces runtime type errors. | Metis MUST-FIX | ✅ Split into two separate functions: `node.agent_type` for implementation type, `get_source_type()` with `match` statement returning `SourceType` Literal. Added domain comparison table and usage rules. |
| 2 | `_DeprecatedSessionId` descriptor as dataclass field default is broken — stores descriptor instance instead of None. Dataclass `__init__` passes the descriptor instance as the default value, which `__set__` stores in `_session_id`. | Oracle S-1 | ✅ Changed to `field(default=None)` with descriptor attached after class definition. Added detailed comment explaining why. |
| 3 | Team/TeamRun SubAgentEvent branch overwrites inner `child_session_id`/`parent_session_id` — current Team/TeamRun code **drops** these fields (they default to `None`). RFC's pseudocode now **preserves** them (improvement over current code, matching `_stream_task` pattern in subagent_tools.py). | Oracle S-2 | ✅ Changed to `child_session_id=event.child_session_id` and `parent_session_id=event.parent_session_id` for nested SubAgentEvents. |
| 4 | `node_model_id` referenced in Team/TeamRun pseudocode but never extracted | Metis SHOULD-FIX-1 | ✅ Added `node_model_id` extraction (from `node.model_name` when `isinstance(node, BaseAgent)`) to both Team and TeamRun pseudocode |
| 5 | ACPSessionManager child path discards caller-provided `session_id` | Metis SHOULD-FIX-2 | ✅ Documented as accepted behavior with rationale |
| 6 | `ensure_session()` store-first path skips `bind_agent_to_session` | Metis SHOULD-FIX-3 | ✅ Documented as known limitation with scope boundary |
| 7 | TG-25 tests wrong expectations — expects `get_source_type()` returns `"team"` for Team, but it should return `"team_parallel"` | Metis | ✅ Rewrote TG-25 to test both `agent_type` and `get_source_type()` with correct domain values. Added TG-26 (type compatibility) and TG-27 (ACP top-level). |
| 8 | Missing edge cases: Team member exception, non-existent parent, agent_pool=None + session_id=None | Metis | ✅ Added EC-6, EC-7, EC-8 |
| 9 | Open Question 7 should be marked RESOLVED (Component #8 addresses it) | Metis | ✅ Marked RESOLVED with two-path solution description |

#### Rev 7 (2026-04-24) — Round 7 Oracle/Metis Review

| # | Finding | Source | Resolution |
|---|---------|--------|------------|
| 1 | `SubAgentType` broken import in team.py/teamrun.py — imported from events.py but doesn't exist there | Oracle B, Metis | ✅ Documented as pre-existing bug. Implementation must replace with `SourceType` from `helpers.py`. |
| 2 | `AgentTypeLiteral` doesn't include "team" — `MessageNode.agent_type` returns "team" which is outside the Literal domain | Metis HR-1 | ✅ Documented: `SessionData.agent_type` is `str | None`, so "team" is valid. Future extension possible. |
| 3 | `get_source_type()` wildcard `case _` returns "agent" — behavioral change from current `ValueError` in team.py:197 | Oracle C, Metis HR-2 | ✅ Documented as deliberate trade-off. Implementation should add `logging.warning()` in wildcard branch. |
| 4 | Finding #3 description said "current code preserves them" — current Team/TeamRun code actually **drops** inner session IDs (they default to None) | Oracle A | ✅ Corrected: RFC's pseudocode improves on current code (matches `_stream_task` pattern). |
| 5 | False claim: "All `run_stream()` overrides in agent subclasses MUST accept `depth: int = 0`" — verified no subclass overrides `run_stream()`, they use `_run_stream_once()` | Metis HR-3 | ✅ Corrected: only `BaseAgent.run_stream()` needs `depth` param. |
| 6 | TG-25 test ordering error — references `team`/`teamrun` before creation | Oracle D | ✅ Reordered: create instances before assertions. |
| 7 | Missing tests: wildcard path, descriptor asdict, circular import, pool_id match, bind_agent skip | Metis | ✅ Added TG-28 through TG-32. |

> To be completed after RFC review.

### Decision

**Status**: PENDING REVIEW

**Date**:

**Approvers**:

### Decision Summary

### Key Discussion Points

### Conditions of Approval

### Dissenting Opinions

---

## References

### Related Documents

- [RFC-0001: Workers and Teams Session Management](../implemented/RFC-0001-workers-teams-session-management.md)
- [RFC-0013: Subagent Event Stream Unification](../implemented/RFC-0013-subagent-event-unification.md)
- [RFC-0014: SpawnSessionStart Event](../implemented/RFC-0014-spawn-session-events.md)
- [RFC-0026: Per-Session Agent Isolation](../implemented/RFC-0026-per-session-agent-isolation.md)

### Key Source Files

- `src/wolfharness/sessions/manager.py` — `SessionManager.create_child_session()`
- `src/wolfharness/sessions/store.py` — `SessionStore` protocol
- `src/wolfharness/sessions/models.py` — `SessionData` model
- `src/wolfharness_toolsets/builtin/subagent_tools.py` — SubagentTools
- `src/wolfharness_toolsets/builtin/workers.py` — WorkersTools
- `src/wolfharness/delegation/team.py` — Team (parallel)
- `src/wolfharness/delegation/teamrun.py` — TeamRun (sequential)
- `src/wolfharness/agents/context.py` — `AgentContext`
- `src/wolfharness/agents/events/events.py` — `SpawnSessionStart`, `SubAgentEvent`
- `src/wolfharness_server/opencode_server/state.py` — `ServerState.ensure_session()`
- `src/wolfharness_server/opencode_server/event_processor.py` — `EventProcessor`
- `src/wolfharness_server/acp_server/session_manager.py` — `ACPSessionManager`
