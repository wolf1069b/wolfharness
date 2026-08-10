---
rfc_id: RFC-0038
title: "Eliminate Pool-Level Agent Instances — Config-Only AgentPool"
status: IMPLEMENTED
author: yuchen.liu
reviewers: []
created: 2026-06-25
last_updated: 2026-06-26
decision_date: 2026-06-26
related_rfcs:
  - RFC-0024 (Agent Stateless Refactor — Phase 2)
  - RFC-0025 (Shared Agent Architecture — Phase 3)
  - RFC-0026 (Per-Session Agent Instances — Phase 1)
---

# RFC-0038: Eliminate Pool-Level Agent Instances — Config-Only AgentPool

> **Natural extension of the Multi-Session Isolation Roadmap (RFC-0024/0025/0026).**
> After agents become stateless and session-scoped, pool-level agent instances serve no purpose.

## Overview

This RFC proposes removing all pool-level `Agent`/`ACPAgent` instances from `AgentPool`. The pool becomes a **pure config store**: it parses YAML into Pydantic config models (`NativeAgentConfig`, `ACPAgentConfig`) and provides a metadata query API. Actual agent instances are created exclusively by `SessionPool` on a per-session basis.

Currently, `AgentPool.__init__()` eagerly creates all agent instances (line 230-239 of `pool.py`), then `AgentPool.__aenter__()` initializes their MCP subprocesses and tool providers (lines 303-319). These pool-level agents serve as:
1. A metadata lookup table for protocol servers (name, description, display_name)
2. A fallback template for `SessionPool` when MCP process limits are hit
3. A shared instance for child/tool sessions to preserve `internal_fs` consistency

All three uses can be replaced with config-based metadata or MCP connection pooling.

## Background & Context

### The Multi-Session Isolation Roadmap

| Phase | RFC | What It Does |
|-------|-----|--------------|
| Phase 1 | RFC-0026 | Per-session agent instances (already implemented) |
| Phase 2 | RFC-0024 | Make `BaseAgent` stateless — move session state off agent |
| Phase 3 | RFC-0025 | Single shared agent serves all sessions |
| **Phase 4** | **RFC-0038** | **Eliminate pool-level agents entirely — config only** |

RFC-0024 and RFC-0025 move toward a model where a single agent instance is shared across sessions. This RFC takes the next logical step: if agents are stateless and per-session, why does the pool need agent instances at all?

### pydantic-ai Already Supports This Model

pydantic-ai's `Agent` is designed for "config-first, instantiate-later":

```python
# Deferred model binding — no provider connection at construction
agent = Agent(model=None, defer_model_check=True)

# From YAML/JSON spec — pure config, no runtime resources
agent = Agent.from_spec({"model": "openai:gpt-4o", "instructions": "..."})

# AgentSpec is a standalone Pydantic model for serializable agent definitions
spec = AgentSpec.from_file("agent.yaml")
```

AgentPool does not leverage these capabilities. It calls `cfg.get_agent()` eagerly in `__init__`, creating `ToolManager`, `MCPManager`, `MessageHistory`, and other heavy infrastructure for every agent — even agents that may never be used in a session.

### Historical Precedent in AgentPool

- `2026-06-17-thin-wolfharness-core` (archived OpenSpec change): Already thinned agent types from 5 → 2 (native + acp), removing ~16K LOC. The `file` agent type was deliberately kept as a "config-only" mechanism — proving this pattern is viable.
- `2026-06-03-thin-pydantic-ai-wrappers` (archived): Established the "Complement, Don't Wrap" vision. Event stream thinning was deferred (Phase 2g, never implemented).
- `refactor-skills-as-capabilities` (active OpenSpec change): Already implements lazy MCP server connections — servers connect on first tool call, not activation.

### Glossary

| Term | Definition |
|------|------------|
| **Pool-level agent** | An `Agent`/`ACPAgent` instance created by `AgentPool.__init__()` and held in the pool's `BaseRegistry` |
| **Session-level agent** | An `Agent`/`ACPAgent` instance created by `SessionPool.get_or_create_session_agent()` for a specific session |
| **Config store** | A component that holds parsed YAML config models (`NativeAgentConfig`, `ACPAgentConfig`) but no runtime agent instances |
| **MCP connection pooling** | Sharing MCP subprocess connections across sessions without sharing entire agent instances |

## Problem Statement

### What's Wrong

`AgentPool` creates and holds heavyweight agent instances that serve almost no purpose:

1. **Pool-level agents are never used for execution.** All actual agent runs go through `SessionPool`, which creates its own per-session agent instances via `get_or_create_session_agent()` (core.py:599-770). The pool-level agents sit idle.

2. **Pool-level agents are only used as metadata containers.** Protocol servers access `pool.all_agents` to get agent names, descriptions, and display names for listing/discovery. These are all config properties — no runtime agent instance is needed.

3. **The fallback use case is a workaround for MCP resource limits.** When MCP process limits are hit, `SessionPool` falls back to reusing the pool-level agent (core.py:698-710). This is MCP connection sharing disguised as agent instance sharing.

4. **Pool-level agents create unnecessary startup cost.** `cfg.get_agent()` resolves model providers, creates `ToolManager`, `MCPManager`, `MessageHistory`, `SystemPrompts`, `HookManager`, `EventManager`, `CommandStore`, and `ExecutionEnvironment` — for every agent, even if never used.

### Evidence

Code audit of all `pool.all_agents` / `pool.get_agent()` / `pool.main_agent` usages:

| Location | What It Accesses | Needs Agent Instance? |
|----------|-----------------|----------------------|
| `agent_routes.py:137` | `agent.name`, `agent.description` | **No** — config properties |
| `acp_agent.py:172` | `len(pool.all_agents)` | **No** — just count |
| `acp_agent.py:181` | `a.name`, `a.display_name` | **No** — config properties |
| `session_routes.py:89` | `name in pool.all_agents` | **No** — name lookup |
| `core.py:586` | `pool.main_agent.name` | **No** — name string |
| `core.py:627` | `pool.get_agent(name)` | **Yes** — but only as MCP fallback |
| `core.py:638-639` | Pool agent for child sessions | **Yes** — but should be session-scoped |
| `pool.py:303-307` | Provider injection into all agents | **Yes** — but should be per-session |
| `pool.py:947,965` | Team/graph building | **Yes** — but resolvable lazily |

**Result: 5 out of 9 usages need zero agent runtime — only config metadata. The remaining 4 are resolvable through MCP connection pooling, lazy graph building, and session-scoped state transfer.**

### Impact of Not Solving

- **Startup latency**: For configs with N agents, `AgentPool.__init__()` spends O(N) time creating heavy objects. With 10+ agents, this is measurable seconds.
- **Memory waste**: Unused agents hold `MCPManager`, `ConnectionManager`, `EventManager`, `ToolManager` — each tens of KB to MB.
- **Architectural confusion**: Pool-level agents blur the boundary between "configuration" and "runtime". New contributors must understand two separate agent lifecycles (pool-level + session-level).
- **Blocks future work**: Config hot-reload, agent eviction, and cross-pool sharing all require clear config/runtime separation.

## Goals & Non-Goals

### Goals

1. **Remove eager agent creation from `AgentPool.__init__()`** — pool stores config models only
2. **Replace `pool.all_agents` with config-based metadata API** — protocol servers query config, not agent instances
3. **Remove pool-level agent fallback in `SessionPool`** — always create per-session agents from config, use MCP connection pooling for resource sharing
4. **Move pool-level provider injection to session level** — MCP/skills providers injected when session agent is created, not on all pool agents
5. **Preserve public API compatibility** where feasible — `pool.get_agent("name")` may change semantics

### Non-Goals

- Making `BaseAgent` fully stateless (that's RFC-0024)
- Single shared agent for all sessions (that's RFC-0025)
- Config hot-reload (future RFC)
- Agent instance eviction/GC (future RFC)
- Changing the YAML config schema
- Removing `MessageNode` abstraction

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Startup time reduction** | High | Eliminate O(N) agent construction in `__init__` |
| **API compatibility** | High | Minimize breaking changes to public API |
| **Implementation complexity** | Medium | Lines changed, files touched, risk of regression |
| **Architectural clarity** | Medium | Does the resulting design have clear boundaries? |
| **SessionPool compatibility** | High | Must not break per-session agent creation |
| **Protocol server compatibility** | High | All 6 protocol servers must continue to work |

## Options Analysis

### Option A: Status Quo (No Change)

**Description**: Keep pool-level agent instances as-is.

**Advantages**:
- Zero implementation effort
- No risk of regression
- Existing tests continue to pass

**Disadvantages**:
- Startup latency persists
- Memory waste persists
- Architectural confusion persists
- Blocks future work (hot-reload, eviction)

**Evaluation Against Criteria**:

| Criterion | Score | Notes |
|-----------|-------|-------|
| Startup time reduction | ✗ | No improvement |
| API compatibility | ✓ | No changes |
| Implementation complexity | ✓ | Zero effort |
| Architectural clarity | ✗ | Status quo |
| SessionPool compatibility | ✓ | No changes |
| Protocol server compatibility | ✓ | No changes |

**Effort Estimate**: None.

**Risk Assessment**: Carries the ongoing cost of eager agent creation. Low risk to keep, but blocks architectural improvements.

---

### Option B: Progressive Lazy Loading (Pool Holds Config + Lazy Proxies)

**Description**: Keep pool-level agent registry but make creation lazy. `AgentPool.__init__()` stores configs; `_ensure_agent(name)` creates the agent on first access. `pool.get_agent()` / `pool.all_agents` trigger lazy creation transparently.

**Advantages**:
- Minimal API changes — `pool.get_agent("name")` still returns an agent
- Graph/team building works as before (lazy creation triggered on access)
- Lower implementation risk than full config-only
- Startup time reduced (agents created on demand)

**Disadvantages**:
- Still holds agent instances after creation — no eviction
- Pool-level agents still exist as a concept — architectural confusion persists
- Two code paths for agent creation (pool-level lazy + session-level)
- `pool.all_agents` access materializes ALL agents (can be slow)
- Thread safety for `_ensure_agent()` requires per-name locks

**Evaluation Against Criteria**:

| Criterion | Score | Notes |
|-----------|-------|-------|
| Startup time reduction | ◐ | Deferred, not eliminated |
| API compatibility | ✓ | `get_agent()` still works |
| Implementation complexity | ◐ | ~100 lines in pool.py, plus lock management |
| Architectural clarity | ✗ | Still has pool-level agents |
| SessionPool compatibility | ✓ | No changes needed |
| Protocol server compatibility | ✓ | Transparent lazy creation |

**Effort Estimate**: Short (1-4 hours). Changes concentrated in `pool.py`.

**Risk Assessment**: Low technical risk. Main risk is race conditions on `_ensure_agent()` — mitigated by per-name `asyncio.Lock`. `pool.all_agents` triggering all lazy creations is a performance gotcha but behaviorally correct.

---

### Option C: Pure Config Store (Eliminate Pool-Level Agents Entirely)

**Description**: `AgentPool` becomes a config parser + metadata provider. No agent instances at the pool level. `SessionPool` is the sole creator of agent instances. Protocol servers query `pool.manifest.agents` for metadata instead of `pool.all_agents`.

```
Before:                              After:

AgentPool                            AgentPool
├─ MCPManager (shared)               ├─ MCPManager (shared)
├─ SkillsManager (shared)            ├─ SkillsManager (shared)
├─ StorageManager (shared)           ├─ StorageManager (shared)
├─ Agent "coder" (instance)     →    ├─ NativeAgentConfig "coder"
├─ Agent "reviewer" (instance)  →    └─ NativeAgentConfig "reviewer"
└─ ACPAgent "goose" (instance)  →
                                     SessionPool
                                     └─ get_or_create_session_agent()
                                          └─ cfg.get_agent()  ← per-session
```

**Advantages**:
- Cleanest architecture — clear config/runtime boundary
- Maximum startup time reduction — zero agent construction in pool
- Maximum memory reduction — no idle agent instances
- Enables future work (hot-reload, eviction, cross-pool sharing)
- Single path for agent creation (SessionPool only)
- Aligns with pydantic-ai's `AgentSpec` pattern

**Disadvantages**:
- More files touched — protocol servers, SessionPool, pool.py, graph building
- `pool.get_agent(name)` semantics change (may need to return config or be deprecated)
- Graph/team building must work with config references, not agent instances
- MCP fallback in SessionPool needs redesign (connection pooling)
- Child session state sharing needs redesign (pass from parent session agent)

**Evaluation Against Criteria**:

| Criterion | Score | Notes |
|-----------|-------|-------|
| Startup time reduction | ✓ | Eliminates all pool-level agent construction |
| API compatibility | ◐ | `pool.get_agent()` may change; `pool.all_agents` replaced |
| Implementation complexity | ◐ | ~10 files, ~200-400 lines changed |
| Architectural clarity | ✓ | Single creation path, clear config/runtime boundary |
| SessionPool compatibility | ◐ | MCP fallback and child session sharing need redesign |
| Protocol server compatibility | ◐ | Metadata access pattern changes (config-based) |

**Effort Estimate**: Medium (1-3 days). Touches `pool.py`, `core.py`, all protocol servers, graph building, team creation.

**Risk Assessment**: Medium technical risk. Protocol servers and SessionPool are the highest-risk areas. Mitigation: implement incrementally, keep pool-level agents behind a feature flag during transition, run full test suite after each step.

---

## Recommendation

**Option C: Pure Config Store** is recommended.

### Justification

1. **It's the natural endpoint of the Multi-Session Isolation Roadmap.** RFC-0024 makes agents stateless. RFC-0025 shares one agent across sessions. RFC-0038 eliminates the pool-level agent entirely — the logical conclusion.

2. **The "progressive lazy loading" (Option B) is a half-measure.** It defers creation but doesn't eliminate the architectural confusion of having two agent lifecycles. It would likely be followed by Option C anyway — incurring the cost twice.

3. **pydantic-ai already models this correctly.** `AgentSpec` is a pure config model; `Agent.from_spec()` is the factory. AgentPool should mirror this pattern: `AgentsManifest` is the config, `SessionPool` is the factory.

4. **The protocol server impact is minimal.** All protocol servers access `pool.all_agents` only for name/description/display_name — these are available from `pool.manifest.agents`.

### Acknowledged Trade-offs

- **Breaking API change for `pool.get_agent()`**: Callers that expect a runtime agent instance from the pool will need to adapt. In practice, the only consumer is `SessionPool.get_or_create_session_agent()`, which already calls `cfg.get_agent()` directly and uses `pool.get_agent()` only as a fallback.
- **MCP connection pooling needed**: The current fallback (reuse pool agent when MCP limits hit) must be replaced with proper MCP connection pooling. This is a net improvement — MCP connections are the resource being conserved, not agent instances.
- **Graph building must work with config references**: Teams and graphs currently reference pool agents. They must instead reference agent configs and resolve to session agents at execution time.

## Technical Design

### Architecture: Before vs After

```
┌─── BEFORE ─────────────────────────┐  ┌─── AFTER ──────────────────────────┐
│                                     │  │                                     │
│  AgentPool.__init__()               │  │  AgentPool.__init__()               │
│  ├─ parse YAML → AgentsManifest     │  │  ├─ parse YAML → AgentsManifest     │
│  ├─ for each agent:                 │  │  ├─ MCPManager (shared singleton)   │
│  │   └─ cfg.get_agent() → Agent    │  │  ├─ SkillsManager (shared)          │
│  │      ├─ ToolManager              │  │  └─ StorageManager (shared)         │
│  │      ├─ MCPManager               │  │                                     │
│  │      ├─ MessageHistory           │  │  AgentPool.__aenter__()             │
│  │      ├─ SystemPrompts            │  │  ├─ start MCP subprocesses          │
│  │      ├─ HookManager              │  │  ├─ discover skills                 │
│  │      ├─ EventManager             │  │  └─ start storage                   │
│  │      ├─ CommandStore             │  │                                     │
│  │      └─ ExecutionEnvironment     │  │  SessionPool                        │
│  │                                  │  │  └─ get_or_create_session_agent()   │
│  │  AgentPool.__aenter__()          │  │     ├─ cfg = manifest.agents[name]  │
│  │  ├─ start MCP subprocesses       │  │     ├─ agent = cfg.get_agent(...)   │
│  │  ├─ inject providers → agents    │  │     ├─ inject MCP/skills providers  │
│  │  ├─ agent.__aenter__() for all   │  │     └─ await agent.__aenter__()     │
│  │  └─ build graph                  │  │                                     │
│                                     │  │  Protocol Servers                   │
│  Protocol Servers                   │  │  └─ pool.manifest.agents[name]      │
│  └─ pool.all_agents[name]           │  │     .name, .description, ...        │
│                                     │  │                                     │
└─────────────────────────────────────┘  └─────────────────────────────────────┘
```

### Key API Changes

#### AgentPool

```python
# REMOVED: pool.get_agent(name) → BaseAgent
#   Replaced by: pool.manifest.agents[name] → NativeAgentConfig | ACPAgentConfig

# REMOVED: pool.all_agents → dict[str, MessageNode]
#   Replaced by: pool.manifest.agents → dict[str, AnyAgentConfig]

# REMOVED: pool.main_agent → BaseAgent
#   Replaced by: pool.main_agent_config → AnyAgentConfig (property)

# NEW: pool.main_agent_name → str
#   Returns the name of the main agent from config

# NEW: pool.get_agent_metadata(name) → AgentMetadata
#   Returns {name, description, display_name, type} without creating an instance

# KEPT: pool.manifest → AgentsManifest
#   Already provides full config access
```

#### SessionPool

```python
# CHANGED: get_or_create_session_agent()
#   Old: Falls back to pool.get_agent(name) when MCP limits hit
#   New: Always calls cfg.get_agent(); uses MCP connection pool for resource sharing
#   Child sessions: receive parent session's agent reference, not pool agent

# NEW: MCP connection pool (internal)
#   Shared MCP subprocess connections across sessions
#   Session agents reference shared connections instead of owning their own
```

#### Protocol Servers

```python
# CHANGED: Agent listing
#   Old: for name, agent in pool.all_agents.items():
#            Agent(name=name, description=agent.description)
#   New: for name, cfg in pool.manifest.agents.items():
#            Agent(name=name, description=cfg.description)

# CHANGED: Agent existence check
#   Old: if name not in pool.all_agents:
#   New: if name not in pool.manifest.agents:

# CHANGED: Agent role switching (ACP)
#   Old: for a in pool.all_agents.values():
#            SessionConfigSelectOption(value=a.name, name=a.display_name)
#   New: for name, cfg in pool.manifest.agents.items():
#            SessionConfigSelectOption(value=name, name=cfg.display_name or name)
```

### Data Flow

```
                    ┌──────────────────┐
                    │   agents.yml     │
                    └────────┬─────────┘
                             │ parse (eager, lightweight)
                             ▼
                    ┌──────────────────┐
                    │  AgentsManifest  │  ← Pydantic model
                    │  ├─ agents:      │     No runtime resources
                    │  │  coder:       │
                    │  │    type: native│
                    │  │    model: ... │
                    │  │    description│
                    │  └─ ...          │
                    └───┬──────────┬───┘
                        │          │
              ┌─────────┘          └─────────┐
              │ (metadata queries)           │ (session creation)
              ▼                              ▼
        Protocol Servers               SessionPool
        "what agents exist?"           get_or_create_session_agent()
        pool.manifest.agents                │
              │                              ▼
              │                       cfg.get_agent(pool=...)
              │                       → Agent instance
              │                       → inject providers
              │                       → await __aenter__()
              │
              ▼
        [{"name": "coder", "description": "...", "type": "native"}, ...]
```

## Implementation Plan

### Phase 1: Add Config Metadata API (Day 1, ~2h)

**Goal**: Protocol servers can query agent metadata without touching agent instances. No behavior changes yet.

1. Add `AgentPool.get_agent_metadata(name) → AgentMetadata` method
2. Add `AgentPool.main_agent_name → str` property
3. Add `AgentPool.main_agent_config → AnyAgentConfig` property
4. Migrate protocol servers to use new metadata API (one server at a time)
5. Run protocol server tests

**Files**: `pool.py`, `acp_server/acp_agent.py`, `opencode_server/routes/agent_routes.py`, `opencode_server/routes/session_routes.py`, `opencode_server/routes/message_routes.py`, `agui_server/server.py`, `openai_api_server/server.py`, `a2a_server/server.py`, `mcp_server/server.py`

### Phase 2: Remove Pool-Level Agent Creation (Day 1-2, ~3h)

**Goal**: `AgentPool.__init__()` no longer creates agent instances.

1. Remove `cfg.get_agent()` loop from `AgentPool.__init__()` (pool.py:230-239)
2. Store `self._agent_configs = dict(self.manifest.agents)`
3. Move provider injection (pool.py:303-307) to `SessionPool.get_or_create_session_agent()`
4. Move agent `__aenter__` (pool.py:313-319) to `SessionPool.get_or_create_session_agent()`
5. Make graph building work with config references (resolve lazily)
6. Make team creation work with config references
7. Remove `pool.all_agents`, `pool.get_agent()`, `pool.main_agent` (or deprecate)

**Files**: `pool.py`, `core.py`

### Phase 3: MCP Connection Pooling (Day 2-3, ~4h)

**Goal**: Replace pool-agent MCP fallback with proper connection pooling.

1. Implement `MCPConnectionPool` — shares MCP subprocesses across sessions
2. Wire into `SessionPool.get_or_create_session_agent()`
3. Remove pool-agent fallback path (core.py:698-710)
4. Fix child session state sharing — pass parent session agent reference
5. Run full test suite

**Files**: `core.py`, new `mcp_server/connection_pool.py`

### Phase 4: Cleanup & Verification (Day 3, ~2h)

1. Remove deprecated `pool.get_agent()`, `pool.all_agents`, `pool.main_agent`
2. Update type hints throughout
3. Run full test suite (`uv run pytest`)
4. Run type checking (`uv run --no-group docs mypy src/`)
5. Run linting (`uv run ruff check src/`)
6. Manual integration test with each protocol server

### Rollback Strategy

Each phase is independently revertible via git. Phase 1 can ship alone (adds API, no behavior change). Phase 2 is the critical cutover — if issues arise, revert Phase 2 and keep Phase 1.

## Open Questions

1. **Should `pool.get_agent()` be deprecated with a warning or removed immediately?**
   - Deprecation with warning allows gradual migration. But the only known caller (`SessionPool`) will be updated in Phase 2 anyway. Recommend: remove in Phase 2, document migration path.

2. **MCP connection pooling: per-pool or per-session-group?**
   - Per-pool is simpler (one pool of MCP connections shared by all sessions). Per-session-group is more flexible but complex. Recommend: start with per-pool, optimize later.

3. **What about programmatic `Agent` construction (not from config)?**
   - `AgentPool.__init__()` currently supports `manifest=None` (empty config) and programmatic `add_agent(agent_instance)`. The programmatic path should remain for direct agent construction use cases.

4. **Does this affect `wolfharness run <agent_name> "prompt"` CLI?**
   - Yes. The CLI currently calls `pool.get_agent(name).run(prompt)`. It should instead create a session via SessionPool or directly call `cfg.get_agent().run(prompt)`.

5. **Should `AgentPool` still extend `BaseRegistry[NodeName, MessageNode]`?**
   - If the pool no longer holds agent instances, the registry type constraint must change. Options: (a) remove `BaseRegistry` inheritance, (b) change type param to config models, (c) keep as a thin wrapper. Recommend: remove `BaseRegistry` inheritance — the pool is no longer a registry of runtime nodes.

## Decision Record

| Field | Value |
|-------|-------|
| Decision | IMPLEMENTED |
| Date | 2026-06-26 |
| Approvers | yuchen.liu |
| Key discussion points | SessionPool is the exclusive execution path; AgentPool is a config store + service manager |
| Conditions/constraints | All phases completed — pool-level agent instances eliminated, config metadata API in place, protocol servers migrated to config-based queries |
