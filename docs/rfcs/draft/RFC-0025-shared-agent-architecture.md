---
rfc_id: RFC-0025
title: "Shared Agent Architecture — Single Agent, Per-Session State"
status: DRAFT
author: yuchen.liu
reviewers: []
created: 2026-04-20
last_updated: 2026-04-20
decision_date:
related_rfcs:
  - RFC-0021 (Agent Concurrent Execution Safety)
  - RFC-0026 (Per-Session Agent Instances — Phase 1)
  - RFC-0024 (Agent Stateless Refactor — Phase 2)
---

# RFC-0025: Shared Agent Architecture — Single Agent, Per-Session State

> **Phase 3 of the Multi-Session Isolation Roadmap.** Depends on [RFC-0024](./RFC-0024-agent-stateless-refactor.md) (Phase 2). This is the convergence point that aligns wolfharness's server architecture with pydantic-ai's stateless agent model.

## Overview

This RFC proposes reverting from per-session `BaseAgent` instances (Phase 1) to a single shared agent that serves all sessions. After [RFC-0024](./RFC-0024-agent-stateless-refactor.md) makes `BaseAgent` stateless, a shared agent becomes safe — all session-scoped state is passed as parameters at call time and held by `ServerState`. This eliminates the MCP subprocess cost per session (~10–50MB, ~1–4s init per session), achieving the resource efficiency that pydantic-ai's stateless design enables.

## Background & Context

### Phase 1→2→3 Progression

| Phase | Agent Model | Session State | MCP Cost | Lock |
|-------|-------------|---------------|----------|------|
| **Before** | 1 shared agent | On agent instance | 1× | Global `agent_lock` |
| **Phase 1** (RFC-0026) | Per-session agents | On agent instance | N× | None |
| **Phase 2** (RFC-0024) | Per-session agents | Parameters + `AgentRunContext` | N× | None |
| **Phase 3** (RFC-0025) | 1 shared agent | `ServerState.session_states[session_id]` | 1× | None |

Phase 1 solves the immediate problem (remove `agent_lock`) by creating per-session agents. Phase 2 makes the agent stateless so it can be shared. Phase 3 realizes the sharing, eliminating N-1 agent instances and their MCP subprocesses.

### pydantic-ai's Intended Model

pydantic-ai's `Agent` is designed as a **reusable, stateless worker**:

```python
agent = Agent("openai:gpt-4o")  # One agent, many runs

# Each run passes its own state
result1 = await agent.run("hello", message_history=[], deps=deps1)
result2 = await agent.run("hello", message_history=[], deps=deps2)  # Different session
```

Phase 3 makes wolfharness's server match this model exactly: one `BaseAgent` instance, many `run_stream()` calls, each with its own `message_history` and `input_provider`.

### Resource Savings

With Phase 1 (per-session agents), each concurrent session adds:
- Node-level MCP subprocesses: ~1–4s init, ~10–50MB memory (per MCP server)
- ToolManager, CommandStore, IsolatedMemoryFileSystem, StagedContent: ~10ms init, ~5MB memory

With Phase 3 (shared agent), these costs are **amortized to 1×** regardless of session count:

| Metric | Phase 1 (5 sessions × 2 MCP) | Phase 3 (1 agent × 2 MCP) | Savings |
|--------|-------------------------------|---------------------------|---------|
| MCP subprocesses | 10 | 2 | 80% |
| Memory (MCP) | ~50–250MB | ~10–50MB | 80% |
| Session creation time | ~1–4s | ~5ms | 99% |

## Problem Statement

### The Problem

Phase 1's per-session agents solve the concurrency problem but introduce resource proportional to session count. For teams of 5–10 concurrent users with 2+ MCP servers each, this becomes significant:

- 10 sessions × 2 MCP × ~25MB = ~500MB just for MCP subprocesses
- Session creation takes 1–4s (MCP init), degrading user experience
- Each `BaseAgent` instance carries its own `ToolManager`, `CommandStore`, `IsolatedMemoryFileSystem`

### Impact

- **Resource ceiling**: Server memory limits the number of concurrent sessions
- **Latency**: New session creation is slow due to MCP init
- **Operational cost**: More resources needed per user, reducing server density

## Goals & Non-Goals

### Goals

1. Reduce MCP subprocess cost from N× to 1× (where N = concurrent sessions)
2. Reduce session creation time from ~1–4s to ~5ms
3. Align server architecture with pydantic-ai's stateless agent model
4. Maintain full session isolation (no conversation contamination, no input provider cross-talk)

### Non-Goals

1. **Not**: Distributed multi-process server (single-process asyncio only)
2. **Not**: MCP connection pooling (pool-level MCP is already shared; node-level MCP becomes shared via single agent)
3. **Not**: Changing the client-side protocol
4. **Not**: ACP server migration (follow-up, same pattern)

### Success Criteria

- [ ] Single `BaseAgent` instance serves all sessions
- [ ] MCP subprocess count is constant regardless of session count
- [ ] Session creation time ≤ 10ms (no MCP init)
- [ ] No conversation contamination between sessions
- [ ] All existing tests pass
- [ ] Memory per additional session ≤ 5MB (state storage only)

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Resource Efficiency | Critical | MCP subprocess and memory cost per session |
| Session Isolation | Critical | No cross-session data leakage |
| Latency | High | Session creation and first-message time |
| pydantic-ai Alignment | High | Agent model matches framework design |
| Migration Risk | Medium | Complexity of reverting from per-session to shared agent |

## Options Analysis

### Option 1: Shared Agent + Per-Session State in ServerState (Recommended)

Replace `ServerState.agents: dict[str, BaseAgent]` with a single `ServerState.agent: BaseAgent` plus `ServerState.session_states: dict[str, SessionState]`. Each `run_stream()` call passes `message_history` and `input_provider` from `SessionState`.

**Advantages**:
- MCP cost reduced to 1× — single agent's MCP servers serve all sessions
- Session creation is ~5ms — no agent construction or MCP init
- Full pydantic-ai alignment — agent is a stateless worker, state is caller's responsibility
- Memory per additional session is only the `SessionState` dataclass (~1–5MB)
- Pool-level MCP is already shared; node-level MCP is now shared via single agent

**Disadvantages**:
- **Requires Phase 2 to be complete** — agent must be fully stateless
- **Model switching becomes global**: If `set_model()` is called on the shared agent, it affects all sessions. Need per-session model override mechanism.
- **`_internal_fs` and `staged_content`**: These are per-session state currently on the agent. After Phase 2 they move to `AgentRunContext`/`SessionState`, so this is resolved.
- **Tool state**: Tools with session state (bash session, file editor) need per-session scoping. Currently they're on the agent's `ToolManager`. Need tool state isolation mechanism.

### Option 2: Hybrid — Shared Agent + Per-Session MCP

Share a single agent for LLM inference and pool-level MCP, but create per-session node-level MCP managers. Tools that need MCP are routed to the session-specific MCP manager.

**Advantages**:
- Solves tool state isolation (per-session MCP managers provide per-session tool state)
- Model switching can be per-session (each session's MCP manager can have its own model config)

**Disadvantages**:
- **Still has N× MCP cost for node-level servers** — the main resource problem is not solved
- **More complex architecture** — hybrid shared/per-session routing adds indirection
- **Defeats the purpose** — if we're creating per-session MCP anyway, Phase 1's per-session agents are simpler

### Comparison

| Criterion | Option 1: Shared Agent | Option 2: Hybrid |
|-----------|----------------------|------------------|
| Resource Efficiency | ✅ 1× MCP | ❌ Still N× for node MCP |
| Session Isolation | ✅ Via `SessionState` | ✅ Via per-session MCP |
| Latency | ✅ ~5ms creation | ⚠️ Node MCP init on first use |
| pydantic-ai Alignment | ✅ Stateless agent | ❌ Mixed model |
| Migration Risk | Medium | High |
| Tool State Isolation | ⚠️ Needs mechanism | ✅ Per-session MCP |

## Recommendation

**Option 1: Shared Agent + Per-Session State in ServerState.**

Option 2 doesn't solve the core resource problem (N× MCP subprocesses). The tool state isolation concern is real but solvable via a lighter mechanism than per-session MCP — tools can read session-scoped state from `AgentRunContext.deps` or `AgentContext`, which Phase 2 already makes available.

### Accepted Trade-offs

1. **Global model switching**: `set_model()` on the shared agent affects all sessions. Per-session model overrides are stored in `AgentRunContext.model_override` and checked before each LLM call. This avoids the broken save/restore pattern (race condition under concurrency). Long-term: contribute `run_stream(model=...)` to pydantic-ai.

2. **Tool state isolation**: Tools with session state (bash, file editor) need to be aware of session context. After Phase 2, `AgentRunContext` carries `_internal_fs` and `staged_content` — tools access these via `ctx.agent_context.run_ctx`. For bash sessions, a per-session bash state can be stored in `SessionState` and passed via `AgentRunContext.deps`.

3. **Migration from per-session to shared**: This is a server-layer change only — `BaseAgent` is unchanged (Phase 2 already made it stateless). The server switches from `state.agents[session_id]` to `state.agent` + `state.session_states[session_id]`.

## Technical Design

### Architecture

```
AFTER PHASE 1 (current):
┌────────────────────────────────────────────────────────┐
│                     ServerState                         │
│  agents: dict[str, BaseAgent]                           │
│    ├─ "sess-1" → BaseAgent (own MCP, own conversation) │
│    ├─ "sess-2" → BaseAgent (own MCP, own conversation) │
│    └─ "sess-3" → BaseAgent (own MCP, own conversation) │
│  sessions: dict[str, Session]                           │
│  messages: dict[str, list]                              │
└────────────────────────────────────────────────────────┘

AFTER PHASE 3 (proposed):
┌────────────────────────────────────────────────────────┐
│                     ServerState                         │
│  agent: BaseAgent  ← SINGLE SHARED (stateless)         │
│  session_states: dict[str, SessionState]                │
│    ├─ "sess-1" → SessionState(session, messages,       │
│    │              input_prov, internal_fs, staged,      │
│    │              model, lock, last_active)             │
│    ├─ "sess-2" → SessionState(...)                     │
│    └─ "sess-3" → SessionState(...)                     │
└────────────────────────────────────────────────────────┘
         │            │            │
     Client A     Client B     Client C
     (run_stream  (run_stream  (run_stream
      + sess-1     + sess-2     + sess-3
      state)       state)       state)
```

### SessionState Dataclass

`SessionState` consolidates ServerState's 7+ per-session dictionaries into a single container:

**Current ServerState per-session dicts (to be consolidated)**:

| Dict | Key Type | Purpose |
|---|---|---|
| `sessions` | `dict[str, Session]` | Session metadata |
| `messages` | `dict[str, list[ChatMessage]]` | Conversation history |
| `session_locks` | `dict[str, asyncio.Lock]` | Per-session locks |
| `input_providers` | `dict[str, OpenCodeInputProvider]` | Per-session input providers |
| `agents` | `dict[str, BaseAgent]` | Per-session agents (Phase 1) |
| `_agent_last_active` | `dict[str, float]` | Idle tracking |
| (other per-session state) | | |

**Consolidated SessionState**:

```python
@dataclass
class SessionState:
    """Per-session state held by the server, passed to run_stream() on each call.
    
    Consolidates ServerState's 7+ per-session dictionaries into a single container.
    """

    # Session metadata (was state.sessions[session_id])
    session: Session

    # Conversation (was state.messages[session_id] + agent.conversation)
    messages: list[ChatMessage] = field(default_factory=list)

    # Input provider (was state.input_providers[session_id] + agent._input_provider)
    input_provider: InputProvider | None = None

    # Session filesystem (was agent._internal_fs)
    internal_fs: IsolatedMemoryFileSystem = field(default_factory=IsolatedMemoryFileSystem)

    # Staged content (was agent.staged_content)
    staged_content: StagedContent = field(default_factory=StagedContent)

    # Per-session model override (was global set_model)
    model_override: str | None = None

    # Session lock (was state.session_locks[session_id])
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Last active timestamp (for cleanup)
    last_active: float = field(default_factory=time.monotonic)
```

### ServerState Changes

```python
class ServerState:
    agent: BaseAgent  # Single shared instance
    session_states: dict[str, SessionState] = field(default_factory=dict)
    
    # REMOVED (consolidated into SessionState):
    # sessions: dict[str, Session]
    # messages: dict[str, list[ChatMessage]]
    # session_locks: dict[str, asyncio.Lock]
    # input_providers: dict[str, OpenCodeInputProvider]
    # agents: dict[str, BaseAgent]
    # _agent_last_active: dict[str, float]
    
    def get_session_state(self, session_id: str) -> SessionState:
        """Get or create per-session state."""
        if session_id not in self.session_states:
            self.session_states[session_id] = SessionState(
                session=Session(id=session_id),
                input_provider=self._create_input_provider(session_id),
            )
        return self.session_states[session_id]
```

This consolidation simplifies ServerState from 7+ dicts to 1, with all per-session state co-located.

### Route Changes

**`message_routes.py: send_message()`**:
```python
# Phase 1 (current):
agent = state.get_or_create_agent(session_id)
iterator = agent.run_stream(*user_prompt, session_id=session_id)

# Phase 3 (proposed):
session_state = state.get_session_state(session_id)
agent = state.agent  # Shared instance

# Per-session model override is handled via AgentRunContext, not set_model()

iterator = agent.run_stream(
    *user_prompt,
    session_id=session_id,
    message_history=session_state.messages,
    input_provider=session_state.input_provider,
    model=session_state.model_override,  # Per-session model via parameter
)
```

**`session_routes.py: create_session()`**:
```python
# Phase 1 (current):
agent = await state.get_or_create_agent(session_id)
# MCP init happens here (~1–4s)

# Phase 3 (proposed):
session_state = state.get_session_state(session_id)
# No agent creation, no MCP init — ~5ms
```

**`config_routes.py: set_model()`**:
```python
# Phase 1 (current): per-session agent
agent = state.get_or_create_agent(session_id)
agent.set_model(model)

# Phase 3 (proposed): per-session override
session_state = state.get_session_state(session_id)
session_state.model_override = model
# Model override is applied via AgentRunContext, not set_model()
```

### Model Override Mechanism

Since the shared agent has a single model, per-session model switching requires an override mechanism. The save/restore pattern is **broken under concurrency** — asyncio yields between `set_model()` and the LLM call, allowing another session to mutate the model before the first session's inference begins.

Instead, per-session model overrides are stored in `AgentRunContext`:

```python
@dataclass
class AgentRunContext:
    # ... existing fields ...
    model_override: str | None = None  # Per-session model override
```

In `_run_stream_once()`, the model override is checked before the LLM call:

```python
async def _run_stream_once(self, run_ctx: AgentRunContext, ...):
    model = run_ctx.model_override or self.model  # Use override if set
    # ... use `model` for LLM call ...
```

In the server, the override is passed via `run_stream(deps=...)` or a dedicated parameter:

```python
# Server route:
session_state = state.get_session_state(session_id)
iterator = agent.run_stream(
    *user_prompt,
    session_id=session_id,
    message_history=session_state.messages,
    input_provider=session_state.input_provider,
    model=session_state.model_override,  # Per-session model
)
```

**Long-term fix**: Contribute `run_stream(model=...)` parameter upstream to pydantic-ai. This is the cleanest approach but requires upstream coordination.

### Session Cleanup

With a shared agent, session cleanup is lightweight — just drop the `SessionState`:

```python
async def cleanup_idle_sessions(self, timeout: float = 1800.0):
    """Evict session states idle beyond timeout."""
    now = time.monotonic()
    for session_id, state in list(self.session_states.items()):
        if now - state.last_active > timeout:
            del self.session_states[session_id]
            # No agent cleanup needed — shared agent persists
```

No MCP subprocess teardown needed (shared agent keeps its MCP servers alive).

### Tool State Isolation

Tools that maintain per-session state (bash, file editor) access it via `AgentRunContext`:

```python
# After Phase 2, AgentRunContext carries per-session state:
class AgentRunContext:
    session_id: str
    internal_fs: IsolatedMemoryFileSystem  # From SessionState
    staged_content: StagedContent           # From SessionState
    deps: Any                              # Business deps

# Bash tool accesses per-session state:
async def bash_handler(ctx: AgentContext, command: str) -> str:
    run_ctx = ctx._run_ctx
    # Bash session is scoped to internal_fs (per-session)
    return await run_shell(command, cwd=run_ctx.internal_fs.cwd)
```

If a tool needs persistent state across runs (e.g., a bash session that persists between messages), the state is stored in `SessionState` and passed to each `AgentRunContext` via `run_stream(deps=...)`.

### MCP Tool State Isolation (Critical Risk)

With a shared agent, all sessions share the same MCP subprocess connections. This creates a **critical isolation risk** for stateful MCP servers:

| MCP Server | Risk | Isolation Mechanism |
|---|---|---|
| Scratchpad | Data leakage — session A reads session B's notes | MCP server must namespace by session_id |
| Knowledge base | Cross-session search results | Query scoping or per-session indices |
| Database connections | Transaction visibility across sessions | Connection-level isolation or per-session connections |

**Mitigation options**:
1. Pass `session_id` as a parameter to every MCP tool call — MCP servers namespace their state by session_id
2. Use MCP server instances per session (defeats Phase 3's resource savings for those servers)
3. Require stateless MCP servers only (may not be realistic for scratchpad/knowledge base)

This risk does not exist in Phase 1 (per-session agents have isolated MCP subprocesses).

## Implementation Plan

### Duration: 2–3 days

| Phase | Scope | Duration |
|-------|-------|----------|
| **P3.1** | Create `SessionState` dataclass. Replace `state.agents[session_id]` with `state.agent` + `state.session_states[session_id]`. Update all route call sites. | 1 day |
| **P3.2** | Implement model override mechanism. Update `set_model()` routes. | 0.5 day |
| **P3.3** | Session cleanup (lightweight — no MCP teardown). Update `fork_session()`. | 0.5 day |
| **P3.4** | Integration tests for shared-agent concurrent sessions. Regression tests. | 1 day |

### Dependencies

- **Requires** [RFC-0024](./RFC-0024-agent-stateless-refactor.md) (Phase 2) to be complete — `BaseAgent` must be fully stateless
- **Requires** `SessionState` integration from Phase 2's open question resolution

### Rollback

Revert to Phase 1's per-session agents by restoring `ServerState.agents: dict[str, BaseAgent]` and `get_or_create_agent()`. Since `BaseAgent` is stateless (after Phase 2), per-session agents also work correctly — rollback is safe.

## Open Questions

1. **Model override timing**: ~~Should per-session model overrides be applied via save/restore on the shared agent, or should `run_stream()` accept a `model` parameter?~~
    - Context: Save/restore has a race condition if two sessions with different models run concurrently (one restores the wrong model). A `model` parameter is cleaner but requires pydantic-ai changes.
    - Status: Resolved — use `AgentRunContext` model_override field. Save/restore is broken under concurrency (Oracle identified race condition). Long-term: contribute `run_stream(model=...)` to pydantic-ai.

2. **Node-level MCP tool isolation**: If two sessions use the same MCP tool (e.g., scratchpad), are MCP tool calls automatically isolated by session, or does the MCP server need to be session-aware?
   - Context: MCP servers are typically stateless (stateless HTTP) or use connection-scoped state (stdio subprocess). With a shared agent, all sessions share the same MCP subprocess connections.
   - Status: Open — depends on MCP server implementation

3. **Should Phase 3 be optional?**: ~~If Phase 1's per-session agents work well for 2–5 users, is Phase 3 worth the additional complexity?~~
    - Context: Phase 3's main benefit is resource savings (1× vs N× MCP). For small teams, this may not matter.
    - Status: Resolved — Phase 3 is deferred until resource pressure justifies it. Phase 1 is sufficient for 2-5 concurrent users (100-250MB overhead). Phase 3's primary benefit (1× vs N× MCP) becomes relevant at 5+ concurrent sessions or when MCP server count increases.

4. **ACP convergence**: Should the ACP server be migrated to the shared-agent pattern simultaneously?
   - Context: ACP server has the same shared-agent problem. Migration is simpler after Phase 3 (just follow the same pattern).
   - Status: Open — likely separate PR

## Decision Record

> Complete after RFC review.

---

## References

- [RFC-0026: Per-Session Agent Instances](../implemented/RFC-0026-per-session-agent-isolation.md) — Phase 1: Remove `agent_lock`
- [RFC-0024: Agent Stateless Refactor](./RFC-0024-agent-stateless-refactor.md) — Phase 2: Make `BaseAgent` stateless
- [RFC-0021: Agent Concurrent Execution Safety](../implemented/RFC-0021-agent-concurrent-execution-safety.md) — Per-run isolation via `AgentRunContext`
- pydantic-ai `Agent` — Stateless worker pattern

### Key Source Files

- `packages/wolfharness/src/wolfharness_server/opencode_server/state.py` — `ServerState`, per-session agent registry
- `packages/wolfharness/src/wolfharness_server/opencode_server/routes/session_routes.py` — Session CRUD
- `packages/wolfharness/src/wolfharness_server/opencode_server/routes/message_routes.py` — Message handling
- `packages/wolfharness/src/wolfharness/agents/base_agent.py` — `BaseAgent` (stateless after Phase 2)

## Review Notes

### Oracle + Metis Review (2026-04-20)

- **Save/restore model override is BROKEN under concurrency** — asyncio yields between `set_model()` and LLM call, allowing cross-session mutation. Must use `AgentRunContext.model_override` or `run_stream(model=...)` instead.
- **MCP tool isolation is a critical blind spot** — stateful MCP servers (scratchpad, knowledge_base) share state across sessions when using a single agent. Phase 1 is safe (per-session MCP); Phase 3 must address this.
- **Phase 2 is the irreducible enabler** — without stateless refactoring, shared agent is unsafe
- **Lazy MCP init** (defer to first tool call) reduces session creation from ~1-4s to ~50ms in Phase 1; Phase 3 eliminates this entirely
- **SessionState consolidation** — user decided to merge ServerState's 7+ per-session dicts into single `SessionState` dataclass

### User Decisions (2026-04-20)

- **Route**: Only Phase 1 will be implemented now. Phase 2/3 are deferred to future demand.
- **Model override**: `AgentRunContext` storage (NOT save/restore)
- **SessionState**: Consolidate — merge all per-session dicts
- **conversation migration**: Hard cut (applies to Phase 2)
- **MCP lazy-init**: First `run_stream()` (applies to Phase 1)
