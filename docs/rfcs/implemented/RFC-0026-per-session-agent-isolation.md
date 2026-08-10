---
rfc_id: RFC-0026
title: "Per-Session Agent Instances — Remove agent_lock"
status: IMPLEMENTED
author: yuchen.liu
reviewers: []
created: 2026-04-20
last_updated: 2026-04-23
decision_date: 2026-04-23
related_rfcs:
  - RFC-0021 (Agent Concurrent Execution Safety)
  - RFC-0024 (Agent Stateless Refactor — Phase 2)
  - RFC-0025 (Shared Agent Architecture — Phase 3)
---

# RFC-0026: Per-Session Agent Instances — Remove agent_lock

> **Phase 1 of the Multi-Session Isolation Roadmap.** See [RFC-0024](../draft/RFC-0024-agent-stateless-refactor.md) for Phase 2 and [RFC-0025](../draft/RFC-0025-shared-agent-architecture.md) for Phase 3.

## Overview

This RFC proposes the minimal change to enable concurrent multi-client access: replace the single shared `BaseAgent` instance and global `agent_lock` with a per-session agent registry. Each session gets its own `BaseAgent` instance, eliminating shared mutable state between sessions and making the global lock unnecessary. `BaseAgent` internals are **not modified** — this is a server-layer change only.

## Problem Statement

When multiple OpenCode clients connect to the same `wolfharness serve-opencode` server, only the first can process messages. All others are blocked by a global `asyncio.Lock` held for the entire duration of LLM inference (10–120 seconds per response).

**Root cause**: `ServerState.agent` is a single `BaseAgent` instance shared across all sessions. `bind_agent_to_session()` mutates `agent.session_id` and `agent._input_provider` per request. A global `agent_lock` serializes access, but blocks everything during inference.

**Code evidence** (`state.py:94-98`):
```python
# Global lock for the shared OpenCode agent instance.
# The base agent mutates per-run state (session_id, input provider,
# active run context, model/mode overrides), so cross-session access must
# be serialized until the server moves to per-session agent instances.
agent_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
```

**Impact**: Teams cannot share an wolfharness server. Each user must run their own instance.

## Goals & Non-Goals

### Goals

1. Enable multiple OpenCode clients to create sessions and send messages concurrently on the same server
2. Eliminate the global `agent_lock`
3. Ensure session state isolation (no conversation contamination, input provider cross-talk, or session_id clobbering)
4. Maintain backward compatibility for single-client usage

### Non-Goals

1. **Not**: Refactoring `BaseAgent` internals (that's [RFC-0024](../draft/RFC-0024-agent-stateless-refactor.md))
2. **Not**: Implementing session cleanup/eviction (acceptable for 2–5 concurrent users; can be added later)
3. **Not**: Lazy MCP initialization at tool-call level (agents init MCP on first `run_stream()`; ~50ms session creation + MCP init on first message)
4. **Not**: Migrating ACP server (separate follow-up)
5. **Not**: Changing the OpenCode client-side protocol

### Success Criteria

- [ ] Two OpenCode clients can create sessions and send messages simultaneously without blocking
- [ ] No conversation contamination between sessions
- [ ] No `agent_lock` remains in `ServerState`
- [ ] All existing OpenCode server tests pass
- [ ] First-message latency for session creation ≤ 5 seconds (including MCP init on first run_stream)

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Concurrent Safety | Critical | Sessions fully isolated, no cross-contamination |
| Backward Compatibility | High | Single-client usage unchanged |
| Minimality | High | Smallest change that solves the problem |
| Resource Cost | Medium | Memory and MCP subprocess cost per additional session |

## Options Analysis

### Option 1: Per-Session Agent Registry (Recommended)

Replace `ServerState.agent` with `ServerState.agents: dict[str, BaseAgent]`. Each session creates its own agent via the existing `NativeAgentConfig.get_agent()` path. Remove `agent_lock` and `bind_agent_to_session()`.

**Advantages**:
- Complete isolation by construction — each session's agent has its own `conversation`, `session_id`, `_input_provider`, `_active_run_ctx`
- `agent_lock` removed entirely — per-session locks (`session_locks`) are sufficient
- `interrupt()` works naturally — each agent has its own `_active_run_ctx`
- Codebase comments identify this as the intended direction
- Minimal change: ~115 lines across 5 files

**Disadvantages**:
- MCP subprocess cost per session (~1–4s creation, ~10–50MB memory)
- Model switching becomes per-session (correct behavior, but a behavior change)
- No session cleanup (acceptable for 2–5 concurrent users)

### Option 2: Parameterize `run_stream` Calls

Pass `message_history`, `input_provider`, `session_id` as parameters to `run_stream()`. Keep a single shared agent.

**Advantages**:
- No additional instances, no MCP cost
- `run_stream()` already accepts these parameters

**Disadvantages**:
- `_active_run_ctx`, `_current_stream_task`, `_cancelled` **cannot** be parameterized — `interrupt()` reads `_active_run_ctx` from a different async task; `ContextVar` returns `None` outside the originating task
- A session→run_ctx registry is per-session state with extra steps
- Every new per-session state on `BaseAgent` requires a new parameter/registry
- Does not solve the root cause (shared mutable instance)

### Comparison

| Criterion | Option 1: Per-Session Agents | Option 2: Parameterize |
|-----------|------------------------------|----------------------|
| Concurrent Safety | ✅ Complete | ❌ `_active_run_ctx` not parameterizable |
| Backward Compatibility | ✅ | ✅ |
| Minimality | ✅ ~115 lines | ❌ Needs session→run_ctx registry |
| Resource Cost | ⚠️ ~10–50MB/session | ✅ No overhead |
| Architecture | ✅ Solves root cause | ❌ Fragile, incomplete |

## Recommendation

**Option 1: Per-Session Agent Registry.**

Option 2 is a dead end because `_active_run_ctx` cannot be parameterized — `interrupt()` reads it from a different async task where `ContextVar` returns `None`. A session→run_ctx registry is just per-session state stored outside the agent, with more indirection and bookkeeping. Option 1 solves the root cause with less total complexity.

### Accepted Trade-offs

1. **MCP subprocess per session**: Acceptable for 2–5 concurrent users. If this becomes a bottleneck, [RFC-0025](../draft/RFC-0025-shared-agent-architecture.md) addresses it by sharing a single agent with pool-level MCP.
2. **No session cleanup**: Acceptable because sessions are typically long-lived (hours). Can be added independently.
3. **Model switching per-session**: This is the correct behavior — different sessions should be able to use different models.

## Technical Design

### Architecture

```
BEFORE:
┌──────────────────────────────────────┐
│              ServerState              │
│  agent: BaseAgent        ← SHARED    │
│  agent_lock: asyncio.Lock ← GLOBAL   │
└──────────────────────────────────────┘
        │           │           │
    Client A    Client B    Client C    ← B, C blocked

AFTER:
┌──────────────────────────────────────┐
│              ServerState              │
│  agents: dict[str, BaseAgent]         │
│    ├─ "sess-1" → BaseAgent           │
│    ├─ "sess-2" → BaseAgent           │
│    └─ "sess-3" → BaseAgent           │
│  (agent_lock removed)                │
└──────────────────────────────────────┘
    Client A    Client B    Client C    ← All active
```

### ServerState Changes (`state.py`)

```python
# REMOVED:
agent: BaseAgent
agent_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

# ADDED:
agents: dict[str, BaseAgent] = field(default_factory=dict)
_agent_config: NativeAgentConfig  # Stored at init for creating new instances

# REMOVED:
def bind_agent_to_session(self, session_id, agent=None): ...

# ADDED:
async def get_or_create_agent(self, session_id: str) -> BaseAgent:
    """Get the per-session agent, creating one if needed."""
    if session_id not in self.agents:
        agent = await self._create_agent_for_session(session_id)
        self.agents[session_id] = agent
    return self.agents[session_id]

async def _create_agent_for_session(self, session_id: str) -> BaseAgent:
    """Create a new BaseAgent instance for a session."""
    agent = self._agent_config.get_agent()  # Existing config→agent path
    agent.session_id = session_id
    agent._input_provider = self.ensure_input_provider(session_id)
    await agent.__aenter__()  # Initialize MCP servers, tools, etc. (deferred to first run_stream)
    return agent
```

### Route Changes

**Pattern: Replace `state.agent` + `state.agent_lock` with `state.get_or_create_agent(session_id)`**

```python
# BEFORE (repeated 15+ times across routes):
async with state.agent_lock:
    agent = state.bind_agent_to_session(session_id)
    # ... use agent ...

# AFTER:
agent = state.get_or_create_agent(session_id)
# ... use agent ...
```

**`session_routes.py: create_session()`**:
```python
# BEFORE:
async with state.agent_lock:
    agent = state.bind_agent_to_session(session_id)
    agent.conversation.chat_messages.clear()

# AFTER:
agent = await state.get_or_create_agent(session_id)
# conversation is already empty on a new agent instance
```

**`message_routes.py: _process_message_locked()`**:
```python
# BEFORE:
async with state.agent_lock:
    agent = state.bind_agent_to_session(session_id, agent=agent)
    iterator = agent.run_stream(*user_prompt, session_id=session_id)
    async for oc_event in adapter.process_stream(iterator):
        await state.broadcast_event(oc_event)

# AFTER:
agent = state.get_or_create_agent(session_id)
iterator = agent.run_stream(*user_prompt, session_id=session_id)
async for oc_event in adapter.process_stream(iterator):
    await state.broadcast_event(oc_event)
```

### `set_model()` Behavior Change

Model switching becomes per-session (correct behavior):

```python
# BEFORE: set_model() on shared agent affects ALL sessions
# AFTER:  set_model() on per-session agent affects ONLY that session

@router.post("/config/model")
async def set_model(session_id: str, model: str, state: StateDep):
    agent = state.get_or_create_agent(session_id)
    agent.set_model(model)  # Only affects this session
```

The current `set_model()` + restore pattern in message_routes and session_routes (4 call sites) should be reviewed — with per-session agents, model overrides are naturally scoped and don't need restoration.

### Fork Session Handling

```python
async def fork_session(source_session_id: str, ...) -> str:
    source_agent = state.agents[source_session_id]
    new_agent = await state._create_agent_for_session(new_session_id)
    # Copy conversation history from source
    new_agent.conversation.chat_messages = list(source_agent.conversation.chat_messages)
    state.agents[new_session_id] = new_agent
    return new_session_id
```

## Pre-Implementation Verification

The following MUST be verified before starting implementation. Failure on any item blocks the RFC.

### V1: `NativeAgentConfig.get_agent()` Creates New Instances

`_create_agent_for_session()` relies on `NativeAgentConfig.get_agent()` returning a **new instance** each call, not a singleton. If it returns the same instance, per-session isolation is broken.

**Verification**: Read `NativeAgentConfig.get_agent()` source. Confirm it creates a new instance. If it delegates to `AgentPool.get_agent()`, that path must also be checked — `AgentPool.get_agent()` may return a cached singleton.

**Fallback**: If `get_agent()` returns a singleton, use the config to construct a new `Agent` directly (e.g., `Agent.from_config(config)`) or call `get_agent()` with a flag that forces new instance creation.

### V2: Per-Session MCP Subprocess Isolation

Each per-session agent spawns its own MCP subprocesses. Verify that MCP subprocess state (scratchpad content, DB connections) is isolated per-subprocess, not shared via a central coordinator.

**Verification**: Check if node-level MCP servers use shared state (e.g., a shared database file, shared socket). If scratchpad MCP writes to a shared store, per-session agents may still cross-contaminate.

**Fallback**: If MCP servers share state, configure each session's MCP server with a unique data directory or namespace.

### V3: `AgentRunContext.session_id` Disconnect

Currently `AgentRunContext.session_id` is auto-generated as `uuid.uuid4().hex` (random), NOT connected to the `session_id` parameter passed to `run_stream()`. This doesn't affect Phase 1 (each session has its own agent), but should be noted as a known issue for Phase 2.

**Action**: Document as a known issue. Do NOT fix in Phase 1 (fixing belongs to Phase 2 scope).

## Implementation Plan

### Scope

~115 lines of changes across 5 files:

| File | Changes |
|------|---------|
| `state.py` | Replace `agent` + `agent_lock` with `agents` dict + `get_or_create_agent()`. Remove `bind_agent_to_session()`. Store `_agent_config` at init. |
| `session_routes.py` | Remove ~10 `async with state.agent_lock:` blocks. Replace `state.agent` with `state.get_or_create_agent(session_id)`. Update `create_session()`, `fork_session()`. |
| `message_routes.py` | Remove `agent_lock` in `_process_message_locked()`. Replace `state.bind_agent_to_session()` with `state.get_or_create_agent()`. Review `set_model()` restore pattern. |
| `config_routes.py` | Remove 2 `agent_lock` blocks. Update `set_model()` to use per-session agent. |
| `global_routes.py` | Update any remaining `state.agent` references. |

### Duration

1–2 days.

### Rollback

Self-contained change. Revert by restoring `state.agent` + `state.agent_lock` + `bind_agent_to_session()` and updating route references back.

## Open Questions

1. **`NativeAgentConfig.get_agent()` instance creation**: Does it return a new instance each call? If it goes through `AgentPool.get_agent()`, does the pool cache or return new instances?
   - Context: Per-session isolation depends on new instances. If `get_agent()` returns a singleton, we need an alternative path.
   - Status: **BLOCKS IMPLEMENTATION** — must verify before coding

2. **MCP server shared state**: Do node-level MCP servers (scratchpad, knowledge_base) share state across subprocess instances?
   - Context: If scratchpad writes to a shared database, per-session isolation at the agent level may not be sufficient.
   - Status: Open — verify during implementation

3. **Shared `ToolManager`**: Tools with state (bash session, file editor) are per-session by construction. Should stateless tools be shared across sessions?
   - Context: Negligible memory savings for 2–5 sessions.
   - Status: Open — likely not worth the complexity

## Decision Record

> Complete after RFC review.

## Review Notes

### Oracle + Metis Review (2026-04-20)

- **Phase 1 is the right first step**: 1–2 days, removes immediate pain, 100–250MB overhead is trivial for 2–5 users
- **MCP lazy-init timing**: Init on first `run_stream()`, not on session creation. Session creation ~50ms, MCP init deferred to first message where user already expects latency
- **MCP tool isolation is a blind spot**: Stateful MCP servers (scratchpad, DB connections) may share state across sessions even with per-session subprocesses. Verify V2 above.
- **Phase 2+3 decisions recorded** for future implementation:
  - Model override → `AgentRunContext` storage (NOT save/restore — broken under concurrency)
  - `SessionState` → Consolidate ServerState's 7+ per-session dicts into single dataclass
  - `conversation` migration → Hard cut, no dual-path deprecation
  - `_active_runs` → Per-instance dict, NOT ClassVar (avoids cross-agent-type collision)

---

## References

- [RFC-0021: Agent Concurrent Execution Safety](./RFC-0021-agent-concurrent-execution-safety.md) — Per-run isolation via `AgentRunContext`
- [RFC-0024: Agent Stateless Refactor](../draft/RFC-0024-agent-stateless-refactor.md) — Phase 2: Make `BaseAgent` stateless
- [RFC-0025: Shared Agent Architecture](../draft/RFC-0025-shared-agent-architecture.md) — Phase 3: Single agent, per-session state

### Key Source Files

- `packages/wolfharness/src/wolfharness_server/opencode_server/state.py` — `ServerState`, `agent_lock`, `bind_agent_to_session()`
- `packages/wolfharness/src/wolfharness_server/opencode_server/routes/session_routes.py` — Session CRUD
- `packages/wolfharness/src/wolfharness_server/opencode_server/routes/message_routes.py` — Message handling
- `packages/wolfharness/src/wolfharness/agents/base_agent.py` — `BaseAgent`, shared mutable state
