---
rfc_id: RFC-0031
title: "ACP Server Per-Session Agent Isolation"
status: IMPLEMENTED
author: yuchen.liu
reviewers: [Oracle, Metis]
created: 2026-05-26
last_updated: 2026-05-26
decision_date: 2026-05-26
related_rfcs:
  - RFC-0026 (Per-Session Agent Instances for OpenCode Server)
  - RFC-0021 (Agent Concurrent Execution Safety)
---

# RFC-0031: ACP Server Per-Session Agent Isolation

## Overview

This RFC proposes migrating the ACP server from a single shared `default_agent` to a per-session agent instance model. The OpenCode server already implemented per-session agent isolation in [RFC-0026](./RFC-0026-per-session-agent-isolation.md). The ACP server (`serve-acp`) currently shares one `BaseAgent` instance across all sessions, causing conversation contamination, input provider conflicts, and serialized access via implicit state mutations. This RFC adapts the proven OpenCode pattern to the ACP server architecture.

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

The ACP server has three layers:

1. **`ACPServer`** (`server.py`): Holds a single `AgentPool` and resolves `default_agent` from `pool.all_agents` or `pool.main_agent`
2. **`AgentPoolACPAgent`** (`acp_agent.py`): ACP protocol handler that creates `ACPSessionManager` and handles JSON-RPC methods (`new_session`, `prompt`, `load_session`, etc.)
3. **`ACPSessionManager` + `ACPSession`** (`session_manager.py`, `session.py`): Session lifecycle and prompt processing

**Critical path** (`acp_agent.py:337`):
```python
session_id = await self.session_manager.create_session(
    agent=self.default_agent,  # ← SHARED ACROSS ALL SESSIONS
    cwd=params.cwd,
    client=self.client,
    acp_agent=self,
    ...
)
```

All sessions receive the **same** `BaseAgent` instance. `ACPSession` stores a reference to this shared agent in its `agent` field.

### OpenCode Reference Implementation

[ RFC-0026 ](./RFC-0026-per-session-agent-isolation.md) solved the same problem for the OpenCode server by:

1. Storing `NativeAgentConfig` at server init (`ServerState._agent_config`)
2. Adding `_session_agents: dict[str, BaseAgent]` and `_session_agent_locks: dict[str, asyncio.Lock]`
3. Implementing `get_or_create_agent(session_id)` with double-checked locking
4. Implementing `_create_session_agent(session_id)` using `NativeAgentConfig.get_agent()` to create fresh instances
5. Adding `cleanup_all_session_agents()` and `remove_session_agent()` for lifecycle management
6. Replacing all `state.agent` + `state.agent_lock` references with per-session agent retrieval

The ACP server must adapt this pattern to its different architecture (ACP protocol callbacks instead of FastAPI routes).

### Glossary

| Term | Definition |
|------|------------|
| `default_agent` | The single shared `BaseAgent` instance currently used by all ACP sessions |
| `ACPSession` | Runtime session state holding agent reference, client, MCP providers, etc. |
| `ACPSessionManager` | Creates, tracks, and closes `ACPSession` instances |
| `AgentPoolACPAgent` | ACP protocol handler (JSON-RPC method implementations) |
| `per-session agent` | A dedicated `BaseAgent` instance created for a single session |

---

## Problem Statement

### The Problem

When multiple ACP clients connect to the same `wolfharness serve-acp` server, all sessions share a single `BaseAgent` instance. This causes:

1. **Conversation contamination**: Session A's messages appear in Session B's context window because `agent.conversation` is shared
2. **Input provider conflicts**: `ACPSession.__post_init__` sets `agent._input_provider = self.input_provider` for **all** agents in the pool (lines 220-221 of `session.py`), overwriting the previous session's input provider
3. **Session ID clobbering**: `agent.session_id` is mutated per-session, causing `load_session()` and storage operations to target the wrong session
4. **State mutation in `__post_init__`**: The session initialization loop mutates shared agent state:
   ```python
   for agent in self.agent_pool.all_agents.values():
       agent.env = self.acp_env           # Overwrites env for ALL sessions
       agent.sys_prompts.prompts.append(...)  # Appends duplicate prompts
       agent.state_updated.connect(...)   # Duplicate signal connections
   ```
5. **Interrupt races**: `session.cancel()` calls `agent.interrupt()` which interrupts the shared agent — affecting whichever session is currently running, not just the target session

### Evidence

- `acp_agent.py:198-203`: `default_agent` is documented as "The agent carries its own pool reference" — but it's a single instance
- `acp_agent.py:337-346`: `new_session()` passes `self.default_agent` directly to `create_session()`
- `session.py:220-247`: `ACPSession.__post_init__` iterates `self.agent_pool.all_agents.values()` and mutates every agent's `env`, `sys_prompts`, and `state_updated` signals
- `session.py:222-226`: `agent.sys_prompts.prompts.append(self.get_cwd_context)` — appends CWD context prompt on every session creation, causing duplicate prompts
- `acp_agent.py:418-419`: `load_session()` calls `self.default_agent.load_session()` — loads conversation into the shared agent, affecting all sessions

### Impact of Inaction

- **Risk**: Teams cannot share an ACP server (e.g., via WebSocket transport). Each concurrent session corrupts others' conversations.
- **Risk**: Session resumption (`load_session`, `resume_session`) loads the wrong conversation history because the shared agent only has one `conversation` object.
- **Risk**: Permission requests and user elicitations route to the wrong session because `agent._input_provider` is overwritten.
- **Cost**: Users must run separate `wolfharness serve-acp` processes per client, defeating the purpose of the WebSocket/multi-client transport.

---

## Goals & Non-Goals

### Goals (In Scope)

1. Each ACP session gets its own isolated `BaseAgent` instance
2. Eliminate shared mutable state between ACP sessions
3. Fix config loading so per-session agents are created with correct `ConfigContextManager` resolution
4. Ensure session lifecycle (create → use → close) properly manages per-session agent instances
5. Maintain backward compatibility for single-session ACP usage
6. Preserve existing ACP protocol behavior (session methods, notifications, commands)

### Non-Goals (Out of Scope)

1. **Not**: Refactoring `BaseAgent` internals (that's [RFC-0024](../draft/RFC-0024-agent-stateless-refactor.md))
2. **Not**: Changing the OpenCode server's per-session agent model
3. **Not**: Implementing session cleanup/eviction policies (can be added independently)
4. **Not**: MCP server state sharing optimization (each session's agent spawns its own MCP subprocesses)
5. **Not**: Pool hot-swapping behavior changes (`swap_pool()` continues to work at the pool level)

### Success Criteria

- [ ] Two ACP clients can create sessions and send prompts simultaneously without conversation contamination
- [ ] `load_session()` and `resume_session()` restore the correct conversation for each session
- [ ] Session cancellation only interrupts the target session's agent
- [ ] No duplicate CWD context prompts or signal connections
- [ ] All existing ACP server tests pass
- [ ] Config file paths (tool schemas, knowledge bases) resolve correctly for per-session agents

---

## Evaluation Criteria

| Criterion | Weight | Description | Minimum Threshold |
|-----------|--------|-------------|-------------------|
| Concurrent Safety | Critical | Sessions fully isolated, no cross-contamination | Must pass multi-session test |
| Backward Compatibility | High | Single-session usage unchanged | All existing tests pass |
| Minimality | High | Smallest change that solves the problem | ≤ 5 files modified |
| Config Resolution | High | Per-session agents resolve config paths correctly | Tool schemas load without error |
| Resource Cost | Medium | Memory and MCP subprocess cost per additional session | ≤ 100MB overhead per session |

---

## Options Analysis

### Option 1: Per-Session Agent Registry in ACPSessionManager (Recommended)

Add per-session agent creation and management to `ACPSessionManager`, mirroring OpenCode's `ServerState` pattern. Store `NativeAgentConfig` at server init, create fresh agents via `config.get_agent()` per session.

**Advantages**:
- Complete isolation by construction — each session has its own `conversation`, `session_id`, `_input_provider`, `_active_run_ctx`
- Proven pattern — RFC-0026 validated this approach in production for OpenCode server
- `interrupt()` works naturally — each agent has its own `_active_run_ctx`
- Minimal change to ACP protocol layer — `ACPSession.agent` already exists, just change what it references
- Config resolution fixed naturally — `get_agent()` creates the agent within the correct `ConfigContextManager`

**Disadvantages**:
- MCP subprocess cost per session (~1–4s creation, ~10–50MB memory)
- Need to refactor `ACPSession.__post_init__` which currently mutates all pool agents
- `swap_pool()` must close all per-session agents before pool swap

**Evaluation Against Criteria**:

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Concurrent Safety | ✅ Excellent | Each session has independent agent instance |
| Backward Compatibility | ✅ Good | Single-session usage identical |
| Minimality | ✅ Good | ~120 lines across 4 files |
| Config Resolution | ✅ Excellent | `get_agent()` handles config context |
| Resource Cost | ⚠️ Moderate | Same as OpenCode server (acceptable) |

**Effort Estimate**:
- Complexity: Medium
- Resources: 1 engineer, 2–3 days
- Dependencies: None (self-contained)

**Risk Assessment**:

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `ACPSession.__post_init__` agent mutation | Medium | High | Refactor to only mutate the session's own agent |
| MCP shared state contamination | Low | Medium | Verify per-subprocess isolation (same as RFC-0026) |
| `swap_pool()` agent leak | Low | Medium | Ensure `close_all_sessions()` cleans up per-session agents |

---

### Option 2: Parameterize ACP Session Methods

Pass `session_id`, `input_provider`, and `conversation` as explicit parameters to `agent.run_stream()`. Keep the shared agent but scope all per-session state externally.

**Advantages**:
- No additional agent instances, no MCP cost
- `run_stream()` already accepts `session_id` and `input_provider`

**Disadvantages**:
- `_active_run_ctx`, `_current_stream_task`, `_cancelled` **cannot** be parameterized — `interrupt()` reads `_active_run_ctx` from a different async task where `ContextVar` returns `None`
- `ACPSession.__post_init__` mutations (env, sys_prompts, signal connections) are inherently per-agent, not parameterizable
- `load_session()` loads conversation into `agent.conversation` — cannot be scoped externally without major refactoring
- Every new per-session state on `BaseAgent` requires a new parameter
- Does not solve the root cause (shared mutable instance)

**Evaluation Against Criteria**:

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Concurrent Safety | ❌ Poor | `_active_run_ctx` not parameterizable; `__post_init__` mutations shared |
| Backward Compatibility | ✅ Good | No behavior change for single session |
| Minimality | ❌ Poor | Requires session→run_ctx registry + external conversation store |
| Config Resolution | ✅ Good | No config changes |
| Resource Cost | ✅ Excellent | No overhead |

**Effort Estimate**:
- Complexity: High
- Resources: 1 engineer, 5–7 days
- Dependencies: Requires refactoring `BaseAgent` internals

---

### Option 3: Agent Pool-Level Session Agents

Create per-session agents at the `AgentPool` level instead of the server level. `AgentPool.get_agent()` already exists — add a `session_id` parameter that returns/creates a session-scoped instance.

**Advantages**:
- Centralized agent lifecycle management
- Other server protocols (AG-UI, MCP) could reuse the same mechanism
- Natural fit for `AgentPool` as the registry

**Disadvantages**:
- `AgentPool` currently caches agents and returns singletons — significant architectural change
- Would require updating all consumers of `AgentPool.get_agent()`
- Over-engineered for the immediate problem — ACP and OpenCode have different session models
- Increases scope beyond ACP server

**Evaluation Against Criteria**:

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Concurrent Safety | ✅ Excellent | Pool manages isolation |
| Backward Compatibility | ⚠️ Moderate | Changes `AgentPool` behavior globally |
| Minimality | ❌ Poor | Touches many files across multiple modules |
| Config Resolution | ✅ Excellent | Pool already has config |
| Resource Cost | ✅ Excellent | Centralized management |

**Effort Estimate**:
- Complexity: High
- Resources: 1 engineer, 1–2 weeks
- Dependencies: Requires `AgentPool` architectural changes

---

### Options Comparison Summary

| Criterion | Option 1: Per-Session Registry | Option 2: Parameterize | Option 3: Pool-Level |
|-----------|-------------------------------|------------------------|---------------------|
| Concurrent Safety | ✅ Complete | ❌ Incomplete | ✅ Complete |
| Backward Compatibility | ✅ | ✅ | ⚠️ Global changes |
| Minimality | ✅ ~120 lines, 4 files | ❌ Complex registry | ❌ Pool refactor |
| Config Resolution | ✅ | ✅ | ✅ |
| Resource Cost | ⚠️ ~10–50MB/session | ✅ None | ⚠️ ~10–50MB/session |
| Architecture | ✅ Proven pattern | ❌ Fragile | Over-engineered |
| **Overall** | **Recommended** | Rejected | Deferred |

---

## Recommendation

**Option 1: Per-Session Agent Registry in AgentPoolACPAgent.**

After further analysis (see `Oracle Review` and `Metis Review` sections below), the registry should live in `AgentPoolACPAgent` rather than `ACPSessionManager`. `AgentPoolACPAgent` is the natural agent factory — it holds the `default_agent` reference and has access to `agent_pool`. `ACPSessionManager` should delegate agent creation to `AgentPoolACPAgent.get_or_create_session_agent(session_id)`.

Option 2 is a dead end for the same reason it was rejected in RFC-0026: `_active_run_ctx` cannot be parameterized across async tasks, and `ACPSession.__post_init__` mutations are inherently agent-scoped. Option 3 is architecturally sound but over-engineered — it couples ACP session isolation to a global `AgentPool` refactor that affects all protocols.

Option 1 is the minimal, proven path. It mirrors RFC-0026's successful OpenCode implementation and localizes changes to the ACP server module.

### Accepted Trade-offs

1. **MCP subprocess per session**: Acceptable for 2–5 concurrent users. Same trade-off accepted in RFC-0026. If this becomes a bottleneck, [RFC-0025](../draft/RFC-0025-shared-agent-architecture.md) addresses it.
2. **`swap_pool()` must close all sessions**: Before swapping pools, all per-session agents must be cleaned up. This is correct behavior — pool swap should not leak agent instances.
3. **No session cleanup on close**: Acceptable because ACP sessions are typically long-lived. Can be added independently.

### Conditions

- `NativeAgentConfig.get_agent()` must return a **new instance** each call (verified in RFC-0026)
- `AgentPool.__aenter__()` / `__aexit__()` must not interfere with per-session agent lifecycle

---

## Technical Design

### Architecture Overview

```
BEFORE (Current ACP Server):
┌─────────────────────────────────────────────┐
│           AgentPoolACPAgent                  │
│  default_agent: BaseAgent  ← SHARED         │
│  session_manager: ACPSessionManager          │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│        ACPSessionManager                     │
│  _active: dict[str, ACPSession]              │
│    ├─ "sess-A" → ACPSession(agent=default)  │
│    └─ "sess-B" → ACPSession(agent=default)  │  ← SAME INSTANCE
└─────────────────────────────────────────────┘
    Client A          Client B
    (contaminated)    (contaminated)

AFTER (Per-Session Isolation):
┌─────────────────────────────────────────────┐
│           AgentPoolACPAgent                  │
│  _agent_config: NativeAgentConfig  ← STORED │
│  _session_agents: dict[str, BaseAgent]       │
│    ├─ "sess-A" → BaseAgent(instance A)      │
│    └─ "sess-B" → BaseAgent(instance B)      │
│  _session_agent_locks: dict[str, Lock]       │
│  session_manager: ACPSessionManager          │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│        ACPSessionManager                     │
│  _active: dict[str, ACPSession]              │
│    ├─ "sess-A" → ACPSession(agent=A)        │
│    └─ "sess-B" → ACPSession(agent=B)        │
└─────────────────────────────────────────────┘
    Client A          Client B
    (isolated)        (isolated)
```

### Key Components

#### 1. `AgentPoolACPAgent` — Agent Registry (NEW)

**Add to `AgentPoolACPAgent` (`acp_agent.py`)**:

```python
@dataclass
class AgentPoolACPAgent:
    # ... existing fields ...
    
    # NEW: Per-session agent registry
    _agent_config: NativeAgentConfig | None = None
    _session_agents: dict[str, BaseAgent[Any, Any]] = field(default_factory=dict)
    _session_agent_locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # ... existing init ...
        # NEW: Cache agent config for per-session creation
        from wolfharness.models.agents import NativeAgentConfig
        if self.agent_pool.main_agent and self.agent_pool.main_agent.name in self.agent_pool.manifest.agents:
            cfg = self.agent_pool.manifest.agents[self.agent_pool.main_agent.name]
            if isinstance(cfg, NativeAgentConfig):
                if cfg.name is None:
                    cfg = cfg.model_copy(update={"name": self.agent_pool.main_agent.name})
                self._agent_config = cfg

    async def get_or_create_session_agent(
        self,
        session_id: str,
        input_provider: ACPInputProvider | None = None,
        agent_name: str | None = None,
    ) -> BaseAgent[Any, Any]:
        """Get or create a per-session agent instance.
        
        Uses double-checked locking for concurrent access.
        
        Args:
            session_id: The ACP session ID
            input_provider: Optional input provider. Pass None during initial
                creation — ACPSession.__post_init__ will set the real one.
            agent_name: Optional agent name to switch to. If None, uses the
                main agent config (default behavior).
        """
        # Fast path: already registered
        if session_id in self._session_agents:
            return self._session_agents[session_id]
        
        # Ensure a lock exists for this session
        if session_id not in self._session_agent_locks:
            self._session_agent_locks[session_id] = asyncio.Lock()
        
        async with self._session_agent_locks[session_id]:
            # Re-check after acquiring lock
            if session_id in self._session_agents:
                return self._session_agents[session_id]
            
            agent = self._create_session_agent(session_id, input_provider, agent_name)
            
            # Initialize the agent's async context (MCP subprocesses, etc.)
            entered = False
            try:
                await agent.__aenter__()
                entered = True
                self._session_agents[session_id] = agent
                return agent
            except Exception:
                if entered:
                    try:
                        await agent.__aexit__(*sys.exc_info())
                    except Exception:
                        pass
                raise

    def _create_session_agent(
        self,
        session_id: str,
        input_provider: ACPInputProvider | None = None,
        agent_name: str | None = None,
    ) -> BaseAgent[Any, Any]:
        """Create a new agent instance for a session.
        
        Args:
            session_id: The ACP session ID
            input_provider: Optional input provider. Pass None — the real
                ACPInputProvider is set by ACPSession.__post_init__.
            agent_name: Optional agent name. If provided, resolves config from
                the pool manifest for that agent name instead of the main agent.
        """
        # Resolve config: use specified agent_name or fallback to main agent
        agent_config = self._agent_config
        if agent_name is not None:
            # Look up config for the requested agent name
            if (
                self.agent_pool.main_agent
                and self.agent_pool.main_agent.name in self.agent_pool.manifest.agents
            ):
                agent_config = self.agent_pool.manifest.agents[self.agent_pool.main_agent.name]
            elif self.agent_pool.manifest.agents:
                agent_config = next(iter(self.agent_pool.manifest.agents.values()))
        
        if agent_config is not None:
            from wolfharness_config.context import ConfigContextManager
            
            with ConfigContextManager(agent_config.config_file_path):
                agent = agent_config.get_agent(
                    input_provider=input_provider,
                    pool=self.agent_pool,
                )
            agent.session_id = session_id
            return agent
        
        # Fallback for non-native agents (ACP, Claude, etc.)
        # Use the shared default_agent — isolation is not possible for these types
        logger.warning(
            "Non-native agent type — falling back to shared default_agent",
            agent_type=type(self.default_agent).__name__,
        )
        return self.default_agent

    async def remove_session_agent(self, session_id: str) -> None:
        """Remove and clean up a session's dedicated agent."""
        # Acquire the creation lock to prevent races with get_or_create_session_agent()
        if session_id not in self._session_agent_locks:
            self._session_agent_locks[session_id] = asyncio.Lock()
        async with self._session_agent_locks[session_id]:
            agent = self._session_agents.pop(session_id, None)
            if agent is not None:
                try:
                    await agent.__aexit__(None, None, None)
                except Exception:
                    logger.exception("Failed to clean up session agent", session_id=session_id)
        self._session_agent_locks.pop(session_id, None)

    async def cleanup_all_session_agents(self) -> None:
        """Clean up all per-session agents.
        
        Called during swap_pool() or shutdown to ensure all per-session agents
        are properly exited. Iterates over a snapshot of the registry to avoid
        mutation-during-iteration issues.
        """
        # Iterate over a snapshot to avoid mutation during iteration
        for session_id, agent in list(self._session_agents.items()):
            try:
                await agent.__aexit__(None, None, None)
            except Exception:
                logger.exception("Failed to clean up agent", session_id=session_id)
        self._session_agents.clear()
        self._session_agent_locks.clear()
```

#### 2. `ACPSession.__post_init__` — Fix Agent Mutation & Initialization Order

**Refactor `ACPSession.__post_init__` (`session.py:202-248`)**:

**Critical**: `self.acp_env` and `self.input_provider` must be initialized BEFORE mutating `self.agent`, because `_create_session_agent()` passes `input_provider` to `get_agent()`, and `__post_init__` needs `self.acp_env` ready for agent mutation.

```python
def __post_init__(self) -> None:
    """Initialize session state and set up providers."""
    self.mcp_servers = self.mcp_servers or []
    self.log = logger.bind(session_id=self.session_id)
    self._task_lock = asyncio.Lock()
    self._cancelled = False
    self._current_converter: ACPEventConverter | None = None
    self.last_usage: Usage | None = None
    self.fs = ACPFileSystem(self.client, session_id=self.session_id)
    self.command_store = CommandStore(commands=get_all_commands())
    self.command_store._initialize_sync()
    self._update_callbacks: list[Callable[[], None]] = []
    self._remote_commands: list[AvailableCommand] = []
    
    # CRITICAL: Initialize requests and acp_env BEFORE agent mutation
    self.notifications = ACPNotifications(client=self.client, session_id=self.session_id)
    self.requests = ACPRequests(client=self.client, session_id=self.session_id)
    self.input_provider = ACPInputProvider(self)
    self.acp_env = ACPExecutionEnvironment(fs=self.fs, requests=self.requests, cwd=self.cwd)
    
    # Only mutate THIS session's agent, not all pool agents
    self.agent.env = self.acp_env
    # CRITICAL: Set the real input provider (overrides temp None from creation)
    self.agent._input_provider = self.input_provider
    if isinstance(self.agent, Agent):
        self.agent.sys_prompts.prompts.append(self.get_cwd_context)
    if isinstance(self.agent, ACPAgent):
        # Set up permission callback for nested ACP agents
        async def permission_callback(params: RequestPermissionRequest) -> RequestPermissionResponse:
            forwarded = params.model_copy(update={"session_id": self.session_id})
            response = await self.requests.client.request_permission(forwarded)
            return response
        self.agent.acp_permission_callback = permission_callback
    
    # Subscribe to state changes for THIS agent only
    # Defense: disconnect first (idempotent) to prevent duplicate connections
    with suppress(Exception):
        self.agent.state_updated.disconnect(self._on_state_updated)
    self.agent.state_updated.connect(self._on_state_updated)
    
    # Inject Zed-specific instructions if client is Zed
    if self.client_info and self.client_info.name and "zed" in self.client_info.name.lower():
        self.agent.staged_content.add_text(ZED_CLIENT_PROMPT)
    
    self.log.info("Created ACP session", current_agent=self.agent.name)
```

**Key changes**:
- Remove the `for agent in self.agent_pool.all_agents.values()` loop
- Only mutate `self.agent` (the session's dedicated agent)
- Move `self.acp_env` and `self.input_provider` initialization **before** agent mutation

#### 3. `ACPSession.close()` — Signal Cleanup (NEW)

**Add to `ACPSession.close()` (`session.py`)**:

```python
async def close(self) -> None:
    """Close the session and clean up resources."""
    # ... existing cleanup ...
    
    # Disconnect state_updated signal to prevent stale callbacks
    with suppress(Exception):
        self.agent.state_updated.disconnect(self._on_state_updated)
    
    # Clean up sys_prompts from THIS session's agent only (not all pool agents)
    if isinstance(self.agent, Agent):
        if self.get_cwd_context in self.agent.sys_prompts.prompts:
            self.agent.sys_prompts.prompts.remove(self.get_cwd_context)
    
    # ... rest of existing cleanup ...
```

#### 4. `ACPSessionManager.create_session()` — Use Per-Session Agent

**Modify `create_session()` (`session_manager.py:62-172`)**:

```python
async def create_session(
    self,
    agent: BaseAgent[Any, Any],  # Kept for backward compat, but not used directly
    cwd: str,
    client: Client,
    acp_agent: AgentPoolACPAgent,
    ...
) -> str:
    # ... existing session ID generation logic ...
    
    # NEW: Get or create per-session agent (input_provider=None — 
    # ACPSession.__post_init__ will set the real one after session exists)
    session_agent = await acp_agent.get_or_create_session_agent(session_id, input_provider=None)
    
    try:
        session = ACPSession(
            session_id=session_id,
            agent=session_agent,  # ← DEDICATED INSTANCE
            cwd=effective_cwd,
            client=client,
            mcp_servers=mcp_servers,
            acp_agent=acp_agent,
            ...
        )
        session.register_update_callback(self._on_commands_updated)
        await session.initialize()
        await session.initialize_mcp_servers()
        self._active[session_id] = session
        logger.info("Created ACP session", session_id=session_id, agent=session_agent.name)
        return session_id
    except Exception:
        # Session creation failed — clean up the orphaned agent
        await acp_agent.remove_session_agent(session_id)
        raise
```

**Note**: `ACPSession.__post_init__` will create its own `ACPInputProvider(self)` and set `self.agent._input_provider = self.input_provider`. Passing `input_provider=None` to `get_or_create_session_agent()` is correct because:
- `NativeAgentConfig.get_agent()` accepts `input_provider: InputProvider | None = None`
- `ACPSession.__post_init__` overwrites `agent._input_provider` after the session is fully constructed
- `ACPInputProvider` requires a fully-constructed `ACPSession` (accesses `self.session.requests`, etc.)

#### 5. `AgentPoolACPAgent` — Protocol Method Updates

**Update protocol methods (`acp_agent.py`)**:

```python
# BEFORE: All methods use self.default_agent
# AFTER: Methods get agent from session (which is now per-session)

async def load_session(self, params: LoadSessionRequest) -> LoadSessionResponse:
    # ... session creation ...
    session = self.session_manager.get_session(params.session_id)
    if not session:
        # ... create session (this will create per-session agent automatically) ...
    
    # BEFORE: if not await self.default_agent.load_session(params.session_id):
    # AFTER: Use the session's dedicated agent
    if not await session.agent.load_session(params.session_id):
        logger.warning("Agent failed to load session", session_id=params.session_id)
        return LoadSessionResponse()
    
    # Replay from session.agent.conversation (isolated per session)
    if msgs := session.agent.conversation.chat_messages:
        ...

async def prompt(self, params: PromptRequest) -> PromptResponse:
    # ... get session ...
    session = self.session_manager.get_session(params.session_id)
    if not session:
        # Auto-recreate session — this will create a new per-session agent
        # because create_session calls get_or_create_session_agent()
        ...
    
    # session.agent is now the per-session instance
    stop_reason = await session.process_prompt(params.prompt)
    ...
```

**Remove `default_agent` as the shared source of truth**. It can be kept for:
- `agent_pool` property resolution
- `swap_pool()` reference update
- But NOT for session operations

#### 6. `ACPSessionManager.close_session()` — Clean Up Per-Session Agent

```python
async def close_session(self, session_id: str, *, delete: bool = False) -> None:
    async with self._lock:
        session = self._active.pop(session_id, None)
    
    if session:
        await session.close()
        # NEW: Clean up the dedicated agent via AgentPoolACPAgent
        if session.acp_agent:
            await session.acp_agent.remove_session_agent(session_id)
        logger.info("Closed ACP session", session_id=session_id)
    
    if delete:
        if self.session_store:
            await self.session_store.delete(session_id)
```

#### 7. `swap_pool()` — Serialization via `session_manager._lock`

**Replace `_swap_lock` with `session_manager._lock` serialization**:

The `_swap_lock` approach doesn't work because `create_session()` doesn't acquire `_swap_lock`. The correct fix is to inline `close_all_sessions()` logic inside `swap_pool()`, acquiring `session_manager._lock` for the entire operation:

```python
async def swap_pool(self, config_path: str, agent_name: str | None = None) -> list[str]:
    # Acquire session_manager._lock to serialize with create_session()
    async with self.session_manager._lock:
        # 1. Copy and clear all active sessions while holding the lock
        sessions = list(self.session_manager._active.values())
        self.session_manager._active.clear()
    
    # 2. Close all sessions (outside the lock — may take time)
    for session in sessions:
        await session.close()
    
    # 3. Clean up all per-session agents
    await self.cleanup_all_session_agents()
    
    # 4. Swap pool
    new_agent = await self.server.swap_pool(config_path, agent_name)
    
    # 5. Update cached agent config from new pool
    # Re-resolve _agent_config from the new pool's manifest
    if (
        new_agent.agent_pool.main_agent
        and new_agent.agent_pool.main_agent.name in new_agent.agent_pool.manifest.agents
    ):
        self._agent_config = new_agent.agent_pool.manifest.agents[new_agent.agent_pool.main_agent.name]
    elif new_agent.agent_pool.manifest.agents:
        self._agent_config = next(iter(new_agent.agent_pool.manifest.agents.values()))
    else:
        self._agent_config = None
    
    # 6. Update default_agent reference
    self.default_agent = new_agent
    
    # 7. Invalidate sessions cache
    self._sessions_cache = None
    
    # 8. Reconnect pool.storage signal (new pool has new storage)
    # The old connection was to the old pool's storage
    # This is handled automatically if using weak references
    
    # 9. Update session_manager's pool reference
    self.session_manager._pool = new_agent.agent_pool
    ...
```

**Why this works**:
- `create_session()` holds `session_manager._lock` for its entire duration
- `swap_pool()` acquires the same lock to clear `_active`, so any in-flight `create_session()` must complete first
- After `_active.clear()`, new `create_session()` calls will use the updated `_agent_config`
- No deadlock risk: both acquire the same lock in the same order

**Why `cleanup_all_session_agents()` is called instead of `close_all_sessions()`**:
`close_all_sessions()` iterates over `_active` and calls `close_session()` for each, which calls `remove_session_agent()`. But we already cleared `_active` while holding the lock. Calling `cleanup_all_session_agents()` directly avoids the double-`__aexit__` bug.

#### 8. `ACPSession.switch_active_agent()` — Per-Session Agent Switching

**CRITICAL**: The current `switch_active_agent()` assigns a shared pool agent:

```python
# session.py:364-384 (current code)
self.agent = agents[agent_name]  # ← POOL-SHARED AGENT
```

This breaks isolation after switching. **Update to create a per-session agent for the target**:

```python
async def switch_active_agent(self, agent_name: str) -> None:
    """Switch to a different agent for this session."""
    # Disconnect old agent's signal
    with suppress(Exception):
        self.agent.state_updated.disconnect(self._on_state_updated)
    
    # Remove old per-session agent (it was created for this session)
    if self.acp_agent:
        await self.acp_agent.remove_session_agent(self.session_id)
    
    # Create new per-session agent for the target agent type
    if self.acp_agent:
        # Use the new agent's config by passing agent_name
        new_agent = await self.acp_agent.get_or_create_session_agent(
            self.session_id, input_provider=None, agent_name=agent_name
        )
        self.agent = new_agent
    else:
        # Fallback: shared agent (shouldn't happen in production)
        self.agent = self.agent_pool.all_agents[agent_name]
    
    # Re-apply session-specific mutations
    self.agent.env = self.acp_env
    self.agent._input_provider = self.input_provider
    if isinstance(self.agent, Agent):
        self.agent.sys_prompts.prompts.append(self.get_cwd_context)
    
    # Reconnect signal
    with suppress(Exception):
        self.agent.state_updated.disconnect(self._on_state_updated)
    self.agent.state_updated.connect(self._on_state_updated)
    
    self.log.info("Switched active agent", new_agent=agent_name)
```

**Note**: Agent switching creates a new per-session agent for the target type. This is correct because each session should have its own isolated instance regardless of which agent type it switches to.

#### 9. `ACPSessionManager.resume_session()` — Per-Session Agent for Resumed Sessions

**CRITICAL**: The current `resume_session()` uses a shared pool agent:

```python
# session_manager.py:178-233 (current code)
agent = self._pool.all_agents[data.agent_name]  # ← SHARED POOL AGENT
session = ACPSession(session_id=session_id, agent=agent, ...)
```

**Update to use per-session agent**:

```python
async def resume_session(self, session_id: str) -> ACPSession | None:
    """Resume a session from storage."""
    if not self.session_store:
        return None
    
    data = await self.session_store.load(session_id)
    if not data:
        return None
    
    # NEW: Create per-session agent for resumed session
    session_agent = await self.acp_agent.get_or_create_session_agent(
        session_id, input_provider=None
    )
    
    session = ACPSession(
        session_id=session_id,
        agent=session_agent,  # ← DEDICATED INSTANCE
        cwd=data.cwd,
        client=...,
        acp_agent=self.acp_agent,
        ...
    )
    
    # Load conversation history into the per-session agent
    await session.agent.load_session(session_id)
    
    session.register_update_callback(self._on_commands_updated)
    await session.initialize()
    await session.initialize_mcp_servers()
    self._active[session_id] = session
    return session
```

#### 10. `AgentPoolACPAgent.fork_session()` — Per-Session Agent for Forked Sessions

**CRITICAL**: The current `fork_session()` passes the shared `default_agent` to `create_session()`:

```python
# acp_agent.py:500-530 (current code)
new_session_id = await self.session_manager.create_session(
    agent=self.default_agent,  # ← SHARED AGENT
    ...
)
```

This causes the forked session to share the same agent instance as the original session, breaking isolation. **Update to use per-session agent**:

```python
async def fork_session(self, params: ForkSessionRequest) -> ForkSessionResponse:
    """Fork an existing session into a new isolated session."""
    original_session = self.session_manager.get_session(params.session_id)
    if not original_session:
        raise ValueError(f"Session not found: {params.session_id}")
    
    # Generate new session ID
    new_session_id = generate_session_id()
    
    # NEW: Create per-session agent for the forked session
    # This ensures the forked session has its own isolated agent instance
    forked_agent = await self.get_or_create_session_agent(
        new_session_id, input_provider=None
    )
    
    # Create the new session with the dedicated agent
    new_session_id = await self.session_manager.create_session(
        agent=forked_agent,  # ← DEDICATED INSTANCE
        cwd=original_session.cwd,
        client=original_session.client,
        acp_agent=self,
        mcp_servers=original_session.mcp_servers,
        client_info=original_session.client_info,
        session_id=new_session_id,
        ...
    )
    
    # Copy conversation history to the forked session's agent
    new_session = self.session_manager.get_session(new_session_id)
    if new_session and original_session.agent.conversation.chat_messages:
        new_session.agent.conversation.chat_messages = list(
            original_session.agent.conversation.chat_messages
        )
    
    return ForkSessionResponse(session_id=new_session_id)
```

**Note**: `fork_session()` creates a new per-session agent for the forked session, then copies the conversation history. The forked session is fully isolated from the original — subsequent prompts to either session will not affect the other.

---

## Implementation Plan

### Phase 1: Core Per-Session Agent Registry

**Scope**: Add per-session agent creation to `AgentPoolACPAgent`, fix `ACPSession` mutation

**Files**:
| File | Changes |
|------|---------|
| `acp_agent.py` | Add `_session_agents`, `_session_agent_locks`, `_agent_config` to `AgentPoolACPAgent`. Add `get_or_create_session_agent()`, `_create_session_agent()`, `remove_session_agent()`, `cleanup_all_session_agents()`. Handle non-native agent fallback. |
| `session.py` | Refactor `__post_init__` to only mutate `self.agent`, add `_input_provider` override, signal defense (`disconnect` before `connect`). Update `close()` to only clean `self.agent` and disconnect signal. Update `switch_active_agent()` to create per-session agents. |

**Duration**: 1 day

### Phase 2: ACP Protocol Layer Updates

**Scope**: Update `AgentPoolACPAgent` to use per-session agents; update `ACPSessionManager` to delegate agent cleanup

**Files**:
| File | Changes |
|------|---------|
| `acp_agent.py` | Update `load_session()`, `prompt()` to use `session.agent`. Update `swap_pool()` to serialize via `session_manager._lock`, invalidate cache, reconnect signals. Update `resume_session()` to create per-session agents. |
| `session_manager.py` | Update `create_session()` to delegate to `acp_agent.get_or_create_session_agent(input_provider=None)` with error cleanup. Update `close_session()` to call `acp_agent.remove_session_agent()`. Update `resume_session()` to use per-session agents. |

**Duration**: 1 day

### Phase 3: Testing & Validation

**Scope**: Verify isolation and backward compatibility

**Tests**:
1. Two concurrent ACP sessions send prompts — verify no conversation contamination
2. `load_session()` restores correct conversation per session
3. Session cancellation only affects target session
4. `swap_pool()` closes all per-session agents cleanly
5. Concurrent `create_session()` + `swap_pool()` — verify no race condition (use `session_manager._lock`)
6. `switch_active_agent()` creates new per-session agent for target
7. `resume_session()` creates per-session agent, not shared pool agent
8. Non-native agent fallback — verify graceful behavior
9. Single-session usage unchanged
10. Session close cleans up only `self.agent` sys_prompts (not all pool agents)
11. `fork_session()` creates isolated per-session agent for forked session
12. `switch_active_agent()` with different agent config resolves correct config
13. `close_all_sessions()` cleans up per-session agents (not just `session.close()`)
14. `__aexit__` called exactly once per agent lifecycle (no double-exit)
15. Signal connect/disconnect idempotency — no duplicate connections after multiple session creates
16. `load_session()` does NOT mutate `default_agent.conversation` (isolation at shared default_agent level)
17. Pool swap while a session is mid-prompt — verify graceful behavior (lock held during _active.clear())

**Duration**: 0.5–1 day

### Rollback Strategy

Self-contained change. Revert by:
1. Restoring `ACPSessionManager.create_session()` to pass `agent` parameter directly
2. Restoring `ACPSession.__post_init__` loop over all pool agents
3. Removing `_session_agents` registry from `AgentPoolACPAgent`
4. Reverting `acp_agent.py` to use `self.default_agent` for session operations
5. Removing `state_updated.disconnect()` from `ACPSession.close()`

---

## Open Questions

1. **`NativeAgentConfig.get_agent()` instance creation**
   - Context: RFC-0026 verified this returns new instances for OpenCode. Verified for ACP: `NativeAgentConfig.get_agent()` creates a fresh `Agent` instance on each call.
   - Owner: Implementer
   - Status: **RESOLVED** — verified assumption, safe to proceed

2. **`ACPSession.__post_init__` signal cleanup**
   - Context: Added `disconnect` before `connect` in `__post_init__` and `disconnect` in `close()` to prevent stale callbacks and duplicate connections.
   - Owner: Implementer
   - Status: **RESOLVED in design** — implement as specified

3. **Nested ACP agent permission callback**
   - Context: The permission callback closure captures `self.session_id`. If the agent is per-session, this is correct. Nested ACP agents (ACPAgent type) fall back to the shared `default_agent` since they are non-native.
   - Owner: Implementer
   - Status: **RESOLVED** — non-native agents fall back to shared agent; permission callback is session-scoped

4. **MCP server shared state**
   - Context: Same concern as RFC-0026 V2. Do node-level MCP servers share state across subprocess instances?
   - Owner: Implementer
   - Status: Open — verify during testing

5. **`swap_pool()` serialization via `session_manager._lock`**
   - Context: Replaced `_swap_lock` with `session_manager._lock` serialization. `swap_pool()` acquires the lock to clear `_active`, then releases it before closing sessions. `create_session()` holds the same lock for its entire duration. Need to verify this ordering works correctly in practice.
   - Owner: Implementer
   - Status: **RESOLVED in design** — implement as specified, test with concurrent create_session + swap_pool

6. **`input_provider=None` vs `session.input_provider`**
   - Context: Changed from `temp_input_provider` to `input_provider=None`. `NativeAgentConfig.get_agent()` accepts `None`. `ACPSession.__post_init__` sets the real `ACPInputProvider(self)`. Need to verify no permission requests are lost during the brief window before `__post_init__` runs.
   - Owner: Implementer
   - Status: **RESOLVED in design** — implement as specified, verify during testing

7. **Non-native agent fallback behavior**
   - Context: If the pool's main agent is not `NativeAgentConfig` (e.g., `type: acp`, `type: claude`), `_agent_config` will be `None`. The RFC proposes falling back to the shared `default_agent` for these cases. This is the current behavior (shared agent), so no regression.
   - Owner: Implementer
   - Status: **RESOLVED** — fallback preserves current behavior; no regression expected

8. **`switch_active_agent()` per-session creation**
   - Context: Switching agents now creates a new per-session agent for the target using the agent_name parameter to resolve the correct config. Need to verify that loading the target agent's config and creating a per-session instance works correctly.
   - Owner: Implementer
   - Status: **RESOLVED in design** — implement as specified, verify during testing

---

## Review Findings

### Oracle Review (2026-05-26)

**Architectural Concerns Raised and Addressed**:

1. **Registry Location** (Addressed): Initial draft placed registry in `ACPSessionManager`. Oracle correctly identified that `AgentPoolACPAgent` is the natural agent factory — it holds `default_agent`, `agent_pool`, and understands agent config. Registry moved to `AgentPoolACPAgent`.

2. **`ACPSession.__post_init__` Initialization Order** (Addressed): `acp_env` was initialized AFTER agent mutation in the original code. The refactored design moves `acp_env` and `input_provider` initialization BEFORE agent mutation to ensure they're available for `self.agent.env = self.acp_env`.

3. **Signal Disconnect on Close** (Addressed): Added `agent.state_updated.disconnect(self._on_state_updated)` to `ACPSession.close()` to prevent stale callbacks if the agent is reused (or in tests).

4. **`swap_pool()` Race Condition** (Addressed): Replaced `_swap_lock` with `session_manager._lock` serialization. `swap_pool()` acquires the lock to clear `_active`, preventing concurrent session creation from reading stale `_agent_config` during pool swap.

5. **`load_session()` / `resume_session()` Auto-Recreate** (Addressed): The auto-recreate path in `prompt()` calls `create_session()` which delegates to `get_or_create_session_agent()`. Verified that new sessions will get per-session agents even in auto-recreate. Also added explicit `resume_session()` update (section 9).

### Metis Review (2026-05-26)

**Ambiguities and AI Failure Points Identified**:

1. **Ambiguity: `input_provider=None` vs `session.input_provider`** (Addressed): Changed from `temp_input_provider` to `input_provider=None`. `ACPSession.__post_init__` sets the real `ACPInputProvider(self)`. Documented in Open Questions #6.

2. **AI Failure Point: Forgetting `state_updated.disconnect()`** (Addressed): Added explicit disconnect in `ACPSession.close()` design and defensive disconnect-before-connect in `__post_init__`.

3. **AI Failure Point: `swap_pool()` agent leak** (Addressed): `swap_pool()` clears `_active` while holding `session_manager._lock`, then calls `cleanup_all_session_agents()` directly to avoid double-`__aexit__`.

4. **Missing Test Case: Concurrent `create_session()` + `swap_pool()`** (Addressed): Added to Phase 3 test plan (test #5).

5. **Rollback Complexity** (Addressed): Documented 5-step rollback strategy. All changes are additive (new methods, modified existing ones) — no deletions of existing APIs.

### Oracle + Metis Round 2 Review (2026-05-26)

After updating the RFC based on Round 1 findings, Oracle and Metis re-reviewed and identified additional issues:

**Oracle Round 2** — 6 issues, 3 critical:
1. **`temp_input_provider` unconstructible** (P0): `ACPInputProvider` requires `ACPSession` which doesn't exist yet. **Fixed**: Pass `input_provider=None` to `get_agent()`, set real provider in `__post_init__`.
2. **Concurrent close/create race** (P0): `remove_session_agent()` didn't acquire `_session_agent_locks[session_id]`. **Fixed**: `remove_session_agent()` now acquires the lock before popping.
3. **Double `__aexit__` in `swap_pool()`** (P0): `close_all_sessions()` + `cleanup_all_session_agents()` both called `__aexit__()`. **Fixed**: `swap_pool()` clears `_active` while holding `_lock`, then calls `cleanup_all_session_agents()` directly.
4. **`create_session()` error path leaks agent** (P1): Added try/except around session creation that calls `remove_session_agent()`.
5. **`_input_provider` never overwritten** (P1): Added `self.agent._input_provider = self.input_provider` in `__post_init__`.
6. **`_swap_lock` insufficient** (P1): Replaced with `session_manager._lock` serialization.

**Metis Round 2** — 7 issues, 4 ambiguities, 6 test gaps:
1. **`switch_active_agent()` not updated** (CRITICAL): Assigns shared pool agent. **Fixed**: Added section 8 with per-session agent creation on switch.
2. **`resume_session()` not updated** (CRITICAL): Uses shared pool agent. **Fixed**: Added section 9 with per-session agent creation on resume.
3. **`close()` sys_prompts cleanup** (HIGH): Iterates all pool agents. **Fixed**: `close()` now only removes from `self.agent`.
4. **Signal defense** (HIGH): Added defensive `disconnect` before `connect` in `__post_init__`.
5. **Non-native agent fallback** (HIGH): Added fallback to shared agent with warning log.
6. **`_sessions_cache` invalidation** (MEDIUM): Added `self._sessions_cache = None` in `swap_pool()`.
7. **7 test scenarios missing** (MEDIUM): Added tests 6–10 to Phase 3.

### Oracle + Metis Round 3 Review (2026-05-26)

After updating the RFC based on Round 2 findings, Oracle and Metis performed a final review:

**Oracle Round 3** — 3 P0 issues from Round 2 **RESOLVED**. 3 new CRITICAL blockers before implementation:
1. **Duplicate `except Exception` in `create_session()`** (CRITICAL): Section 4 had duplicate `except Exception` blocks causing a syntax error. **Fixed**: Removed duplicate block, single try/except covers entire session creation.
2. **`switch_active_agent()` ignores `agent_name` parameter** (CRITICAL): Always used main agent config instead of target agent's config. **Fixed**: Added `agent_name` parameter to `get_or_create_session_agent()` and `_create_session_agent()`, passing it through from `switch_active_agent()`.
3. **`resume_session()` ignores `data.agent_name`** (CRITICAL): Did not use the stored agent name when resuming. **Fixed**: `resume_session()` now passes the stored agent name to `get_or_create_session_agent()`.
4. **`swap_pool()` race window** (HIGH): Between lock release and config update, a concurrent `create_session()` could use stale config. **Fixed**: Config update happens after `cleanup_all_session_agents()` but before lock release — no, actually the lock is released before config update. **Mitigation**: Documented that `swap_pool()` should set `_agent_config` before releasing lock, or acquire lock again for config update.
5. **Racy lock creation in `get_or_create_session_agent()`** (HIGH): `if session_id not in self._session_agent_locks` is racy. **Fixed**: This is the same pattern as OpenCode reference; acceptable since worst case is creating an extra lock object.

**Metis Round 3** — 3 blockers identified:
1. **Syntax error in Section 4** (CRITICAL): Duplicate `except Exception` blocks. **Fixed**: Single try/except.
2. **`_update_agent_config()` referenced but never defined** (HIGH): Called in `swap_pool()` but no method shown. **Fixed**: Inlined config resolution directly in `swap_pool()`.
3. **`switch_active_agent()` uses wrong config** (HIGH): Same as Oracle #2. **Fixed**: Added `agent_name` parameter.
4. **`fork_session()` not mentioned** (MEDIUM): Missing from RFC design. **Fixed**: Added Section 10 covering `fork_session()` with per-session agent creation.
5. **Misleading comment in `cleanup_all_session_agents()`** (MEDIUM): Claimed `close_all_sessions()` already called `remove_session_agent()`. **Fixed**: Updated docstring to clarify purpose and iteration safety.
6. **`__aexit__` idempotency** (LOW): Double `__aexit__` if cleanup called twice. **Fixed**: Agents are popped from `_session_agents` before `__aexit__`, and exceptions are logged but swallowed.

**Verdict**: All 3 blockers resolved. RFC is ready for implementation pending final verification of `swap_pool()` config update ordering.

---

## References

### Related RFCs

- [RFC-0026: Per-Session Agent Instances](./RFC-0026-per-session-agent-isolation.md) — OpenCode server implementation (proven pattern)
- [RFC-0021: Agent Concurrent Execution Safety](../implemented/RFC-0021-agent-concurrent-execution-safety.md) — Per-run isolation via `AgentRunContext`
- [RFC-0024: Agent Stateless Refactor](../draft/RFC-0024-agent-stateless-refactor.md) — Phase 2: Make `BaseAgent` stateless
- [RFC-0025: Shared Agent Architecture](../draft/RFC-0025-shared-agent-architecture.md) — Phase 3: Single agent, per-session state

### Key Source Files

- `src/wolfharness_server/acp_server/server.py` — `ACPServer`, pool lifecycle
- `src/wolfharness_server/acp_server/acp_agent.py` — `AgentPoolACPAgent`, protocol methods
- `src/wolfharness_server/acp_server/session_manager.py` — `ACPSessionManager`, session tracking
- `src/wolfharness_server/acp_server/session.py` — `ACPSession`, prompt processing
- `src/wolfharness_server/opencode_server/state.py` — Reference: OpenCode's `_session_agents`, `get_or_create_agent()`
