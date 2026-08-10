# MCP Provider Lifecycle Architecture

**Date**: 2026-07-01
**Status**: Draft
**Author**: Sisyphus (via brainstorming session)
**Branch**: `refactor/migrate-to-mcptoolset`
**Worktree**: `/Users/yuchen.liu/src/yilab/iroot-llm/packages/wolfharness-fix-mcp`

## Problem Statement

AgentPool's MCP tool provider lifecycle has a fundamental architectural flaw: it conflates three orthogonal concerns — MCP server configs, MCP provider instances, and MCP toolset materialization — into a single mutable pipeline (`agent.tools.providers`). This causes:

1. **Race condition**: Session-level MCP providers are registered after the agentlet is already built, so subagents lose access to MCP-over-ACP tools (e.g., `workspace-fs`).
2. **Cross-task CancelScope errors**: Cached `MCPToolset` instances shared across asyncio tasks trigger `RuntimeError: Attempted to exit cancel scope in a different task`.
3. **Whack-a-mole fixes**: Each patch fixes one symptom but creates new problems because the root cause — mutable provider list + timing coupling — remains.

### Additional Requirements (Industrial-Grade)

- **Stateful MCP support**: Some MCP servers maintain per-session state (conversation history, resource locks, workflow state). Connections to these servers must not be shared across sessions.
- **Multi-user isolation**: Different users must have isolated connection state for session-level MCP servers.
- **Skill MCP integration**: Skills can declare their own MCP servers (`SkillMcpManager`). The architecture must include these.
- **One-time comprehensive fix**: Not another patch — a principled redesign.

## Design: Three-Tier Connection Scoping

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TIER 1: McpConfigSnapshot                              │
│                     (immutable config, inheritable)                         │
│                                                                             │
│  Created at session creation time. Contains ALL MCP server configs:        │
│  - Pool-level (from YAML mcp_servers)                                      │
│  - Agent-level (from agent's mcp_servers)                                  │
│  - Session-level (inherited from parent or from session/new)               │
│  - Skill MCP (populated dynamically by SkillMcpManager)                    │
│                                                                             │
│  Frozen after initialization for pool/agent/session configs.               │
│  During init, snapshot replaced via with_*_configs() (new instance).       │
│  Skill configs remain dynamic throughout session lifetime.                 │
│  Child sessions inherit parent's session-level configs at creation.        │
│  session/load on restore → replace snapshot; on active → ignore.           │
└───────────────────────────┬─────────────────────────────────────────────────┘
                            │ read
┌───────────────────────────▼─────────────────────────────────────────────────┐
│                     TIER 2: Connection Pools                               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2a: GlobalConnectionPool (pool-level singleton)                    │   │
│  │                                                                    │   │
│  │  Scope: shared across ALL sessions + users                         │   │
│  │  Key: client_id                                                    │   │
│  │  Pattern: owner-task (deer-flow) for stdio                         │   │
│  │           direct transport for HTTP/SSE                            │   │
│  │  Ref-counted: N sessions share 1 connection                        │   │
│  │  Servers: pool-level + agent-level configs only                    │   │
│  │  Assumption: stateless (shared infrastructure)                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2b: SessionConnectionPool (per-session)                            │   │
│  │                                                                    │   │
│  │  Scope: per-session, isolated                                      │   │
│  │  Key: (session_id, server_name, skill_name?)                       │   │
│  │  Pattern: owner-task for stdio; per-session for HTTP/SSE           │   │
│  │  NOT shared across sessions (supports stateful MCP)                │   │
│  │  Servers: session-level (ACP client-provided) + skill MCP          │   │
│  │  Lifecycle: created lazily, cleaned up on session close            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────────────────┘
                            │ borrow
┌───────────────────────────▼─────────────────────────────────────────────────┐
│                     TIER 3: Toolset Materialization                        │
│                     (in get_agentlet())                                     │
│                                                                             │
│  Reads McpConfigSnapshot → determines which servers needed                 │
│  For pool/agent configs → borrow from GlobalConnectionPool                 │
│  For session/skill configs → borrow from SessionConnectionPool             │
│  Creates MCPToolset per agentlet (fresh, no cross-task issues)             │
│  Bypasses agent.tools.providers — MCP has its own dedicated path           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Principles

1. **Config is data, not live objects.** MCP server configs are plain data models (`BaseMCPServerConfig` subclasses). They are frozen into an immutable snapshot at session creation time. No live `MCPResourceProvider` instances are added to a mutable list.

2. **Connections are task-affine.** anyio's `CancelScope` is task-local. The owner-task pattern (dedicated `asyncio.Task` enters/exits the context manager) ensures cancel scopes are always entered and exited in the same task. This applies to all stdio connections. HTTP/SSE connections use direct transports (no cancel scope issues).

3. **Pool-level connections are shared; session-level connections are isolated.** Pool-level YAML `mcp_servers` are assumed stateless and shared across all sessions via `GlobalConnectionPool`. Session-level MCP servers (ACP client-provided) and skill MCP servers are per-session via `SessionConnectionPool`, supporting stateful MCP and multi-user isolation.

4. **Child sessions inherit parent's session-level configs at creation time.** No ACP round-trip needed. The `session/load` `mcpServers` parameter is used only when restoring a session from storage, not when the session is already active.

5. **Toolset materialization is lazy and per-agentlet.** `get_agentlet()` reads the frozen snapshot and borrows connections from the appropriate pool. Each agentlet gets fresh `MCPToolset` instances (no caching, no cross-task issues).

6. **MCP tools bypass `agent.tools.providers`.** MCP has its own dedicated path in `get_agentlet()`, decoupled from the `ResourceProvider` pipeline. This eliminates the mutable-list problem entirely.

## Components

### 1. `McpConfigEntry`

A single MCP server configuration entry within a snapshot.

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from wolfharness_config.mcp_server import BaseMCPServerConfig

@dataclass(frozen=True)
class McpConfigEntry:
    """Single MCP server configuration entry."""

    server_config: BaseMCPServerConfig
    source: Literal["pool", "agent", "session", "skill"]
    skill_name: str | None = None  # If source="skill", which skill
```

### 2. `McpConfigSnapshot`

Immutable snapshot of all MCP server configs for a session.

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

@dataclass(frozen=True)
class McpConfigSnapshot:
    """Immutable snapshot of all MCP server configs for a session.

    Created at session creation time. Pool/agent/session configs are frozen.
    Skill configs start empty and are populated dynamically by SkillMcpManager.
    Child sessions inherit parent's session-level configs at creation.
    """

    pool_configs: tuple[McpConfigEntry, ...]
    agent_configs: tuple[McpConfigEntry, ...]
    session_configs: tuple[McpConfigEntry, ...]
    skill_configs: tuple[McpConfigEntry, ...]  # Dynamic — see SkillMcpManager

    @property
    def all_configs(self) -> tuple[McpConfigEntry, ...]:
        """All configs, in priority order."""
        return (*self.pool_configs, *self.agent_configs,
                *self.session_configs, *self.skill_configs)

    @property
    def global_configs(self) -> tuple[McpConfigEntry, ...]:
        """Configs served by GlobalConnectionPool (pool + agent level)."""
        return (*self.pool_configs, *self.agent_configs)

    @property
    def session_scoped_configs(self) -> tuple[McpConfigEntry, ...]:
        """Configs served by SessionConnectionPool (session + skill level)."""
        return (*self.session_configs, *self.skill_configs)

    def with_skill_configs(self, skills: tuple[McpConfigEntry, ...]) -> McpConfigSnapshot:
        """Create a new snapshot with updated skill configs.

        Since the dataclass is frozen, this returns a new instance.
        Only skill_configs can change — pool/agent/session are immutable.
        """
        return McpConfigSnapshot(
            pool_configs=self.pool_configs,
            agent_configs=self.agent_configs,
            session_configs=self.session_configs,
            skill_configs=skills,
        )

    def with_session_configs(self, sessions: tuple[McpConfigEntry, ...]) -> McpConfigSnapshot:
        """Create a new snapshot with updated session configs.

        Used during initialization (e.g., initialize_mcp_servers() adds
        session-level configs). Returns a new frozen instance.
        After initialization completes, this should not be called.
        """
        return McpConfigSnapshot(
            pool_configs=self.pool_configs,
            agent_configs=self.agent_configs,
            session_configs=sessions,
            skill_configs=self.skill_configs,
        )
```

**Lifecycle:**

| Event | Action |
|-------|--------|
| Parent session created (`session/new`) | Snapshot created with pool + agent + session configs |
| Child session created (`create_child_session`) | Snapshot created, inheriting parent's `session_configs` |
| `session/load` on stored session | Snapshot replaced with one from `mcpServers` param |
| `session/load` on active session | Snapshot NOT replaced (existing wins) |
| Skill loaded during run | `skill_configs` updated via `with_skill_configs()` |

### 3. `GlobalConnectionPool`

Pool-level singleton for sharing connections across all sessions.

```python
class GlobalConnectionPool:
    """Pool-level singleton for sharing MCP connections across sessions.

    Manages connections for pool-level and agent-level MCP servers.
    Assumes servers are stateless (safe to share across sessions/users).

    Uses the owner-task pattern (deer-flow) for stdio servers:
    - A dedicated asyncio.Task enters and exits the transport context manager.
    - Callers signal via Events; the owner task handles lifecycle.
    - This eliminates cross-task CancelScope errors.

    HTTP/SSE servers use direct transports (no cancel scope issues).
    """

    def __init__(self) -> None:
        self._connections: dict[str, _PooledConnection] = {}
        self._lock = threading.Lock()  # Thread-safe (deer-flow pattern)

    async def get_transport(
        self, config: BaseMCPServerConfig
    ) -> ClientTransport:
        """Get or create a shared transport for the given config.

        For stdio: uses owner-task pattern (dedicated asyncio.Task).
        For HTTP/SSE: creates direct transport (no pooling overhead,
                     but still cached by client_id for reuse).
        For ACP: raises NotImplementedError (handled by SessionConnectionPool).

        Returns a ClientTransport that can be used to construct MCPToolset.
        """
        ...

    async def release(self, client_id: str) -> None:
        """Decrement ref count. When 0, signal owner-task to shut down."""
        ...

    async def shutdown_all(self, timeout: float = 10.0) -> None:
        """Clean shutdown of all connections. Called on pool shutdown."""
        ...
```

**Owner-task pattern for stdio:**

```
Caller (any asyncio task)              Owner Task (dedicated)
───────────────────────                ──────────────────────
get_transport(stdio_config)
  → acquire _lock
  → check _connections cache
  → if miss:
    → create _PooledConnection
    → create owner_task = asyncio.create_task(_run_session())  ───→  _run_session():
    → await ready_event.wait() (with timeout)                         async with transport.connect_session():
    ← ready_event.set() (from owner)                                      ready_event.set()
  → ref_count += 1                                                        await close_event.wait()
  → release _lock                                                     ← close_event.set() (from release)
  → return transport                                                  → transport.__aexit__() (same task!)
                                                                      → done_event.set()
release(client_id)
  → acquire _lock
  → ref_count -= 1
  → if ref_count == 0:
    → close_event.set() (signals owner to shut down)
    → await done_event.wait() (with timeout)
    → remove from _connections
  → release _lock
```

**Key safety properties:**
- `threading.Lock` (not `asyncio.Lock`) — safe from both async and sync paths
- `asyncio.shield(ready_event.wait())` prevents caller cancellation from leaking mid-init
- Owner task always runs `__aexit__` in the same task that ran `__aenter__`
- LRU eviction with `MAX_SESSIONS` limit (configurable, default 256)

### 4. `SessionConnectionPool`

Per-session connection pool for session-level and skill MCP servers.

```python
class SessionConnectionPool:
    """Per-session connection pool for session-level and skill MCP servers.

    Each session gets its own isolated connections.
    Supports stateful MCP servers (no cross-session sharing).
    Supports multi-user isolation (session = user context boundary).

    Uses owner-task pattern for stdio (same as GlobalConnectionPool).
    HTTP/SSE connections are per-session (not shared, but lightweight).
    """

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._connections: dict[str, _SessionConnection] = {}
        self._lock = threading.Lock()

    async def get_transport(
        self,
        config: BaseMCPServerConfig,
        skill_name: str | None = None,
    ) -> ClientTransport:
        """Get or create a session-scoped transport.

        Key: (config.client_id, skill_name) — ensures skill MCP servers
        are isolated even within the same session.
        """
        ...

    async def cleanup(self, timeout: float = 5.0) -> None:
        """Clean up all connections for this session.

        Called when session is closed. Signals all owner-tasks to shut down,
        waits with timeout, then force-cancels any remaining.
        """
        ...
```

### 5. Modified `MCPManager`

```python
class MCPManager:
    """Manages pool-level MCP servers + global connection pool.

    Changes from current:
    - as_capability() now accepts an optional McpConfigSnapshot
    - Uses GlobalConnectionPool for pool/agent configs
    - No _toolset_cache (already removed)
    - MCPResourceProvider no longer added to agent.tools.providers
    """

    def __init__(self, pool: AgentPool) -> None:
        self._pool = pool
        self._global_pool = GlobalConnectionPool()  # NEW
        # ... existing fields ...

    async def as_capability(
        self,
        snapshot: McpConfigSnapshot | None = None,
        session_pool: SessionConnectionPool | None = None,
    ) -> list[MCP]:
        """Create MCP capabilities from config snapshot.

        If snapshot is provided (new path):
        - For global configs: borrow from GlobalConnectionPool
        - For session configs: borrow from SessionConnectionPool
        - Create fresh MCPToolset per call (no caching)

        If snapshot is None (legacy path):
        - Falls back to old behavior (pool.mcp servers only)
        """
        ...

    async def disconnect_all(self) -> None:
        """Shut down GlobalConnectionPool."""
        await self._global_pool.shutdown_all()
        # ... existing cleanup ...
```

### 6. Modified `get_agentlet()`

```python
async def get_agentlet(self, ...) -> PydanticAgent:
    # ... existing code for tools, hooks, deferred bridge, approval bridge ...

    # 4. MCP servers — NEW PATH
    # Read from frozen config snapshot, borrow from connection pools.
    # This bypasses agent.tools.providers for MCP — MCP has its own path.
    mcp_snapshot = self._mcp_snapshot  # Set at agent creation time
    session_pool = self._session_connection_pool  # Set at agent creation time
    if mcp_snapshot is not None:
        mcp_capabilities = await self.mcp.as_capability(
            snapshot=mcp_snapshot,
            session_pool=session_pool,
        )
        tool_capabilities.extend(mcp_capabilities)
    else:
        # Legacy fallback (standalone agents without pool)
        mcp_capabilities = await self.mcp.as_capability()
        tool_capabilities.extend(mcp_capabilities)

    # 5. Skill capabilities — unchanged
    # ...
```

### 7. Modified `get_or_create_session_agent()` (child session path)

```python
# In SessionController.get_or_create_session_agent(), child session path:
if session.parent_session_id:
    parent_state = self._sessions.get(session.parent_session_id)
    parent_agent = parent_state.agent if parent_state else None

    # ... create child agent from config ...

    await agent.__aenter__()

    # NEW: Build MCP config snapshot for child session
    parent_snapshot: McpConfigSnapshot | None = (
        parent_agent._mcp_snapshot if parent_agent is not None else None
    )

    # Pool configs: same as parent (shared pool)
    pool_configs = parent_snapshot.pool_configs if parent_snapshot else (
        self._build_pool_configs()
    )

    # Agent configs: from child's own YAML config (NOT inherited)
    agent_configs = self._build_agent_configs(cfg)

    # Session configs: INHERITED from parent (key fix!)
    session_configs = parent_snapshot.session_configs if parent_snapshot else ()

    child_snapshot = McpConfigSnapshot(
        pool_configs=pool_configs,
        agent_configs=agent_configs,
        session_configs=session_configs,  # Inherited!
        skill_configs=(),  # Populated lazily by SkillMcpManager
    )
    agent._mcp_snapshot = child_snapshot
    agent._session_connection_pool = SessionConnectionPool(session_id)

    # Add pool-level NON-MCP providers (skills instruction, skills tools, etc.)
    # MCP providers are NOT added here — they go through the snapshot path
    if self.pool is not None:
        if self.pool.skills_instruction_provider:
            agent.tools.add_provider(self.pool.skills_instruction_provider)
        agent.tools.add_provider(self.pool.skills_tools_provider)

    # ... rest of existing code ...
```

### 8. Modified `ACPSession.initialize_mcp_servers()`

```python
async def initialize_mcp_servers(self) -> None:
    """Initialize session-level MCP servers.

    Changes:
    - No longer creates live MCPResourceProvider instances
    - Instead, converts mcpServers to McpConfigEntry and adds to snapshot
    - For ACP-transport servers: creates AcpMcpTransport and stores in SessionConnectionPool
    - Provider registration on agent.tools is REMOVED (MCP bypasses providers)
    """
    if not self.mcp_servers:
        return

    new_entries: list[McpConfigEntry] = []

    async def _init_server(server: Any) -> None:
        try:
            with anyio.fail_after(30):
                cfg = convert_acp_mcp_server_to_config(server)

                if isinstance(server, AcpMcpServer):
                    # ACP-transport: create transport and store in SessionConnectionPool
                    connection_id = await self.acp_agent.connect_acp_mcp_server(server)
                    conn = self.acp_agent._mcp_manager.get_connection(connection_id)
                    transport = AcpMcpTransport(conn, timeout=...)
                    await self.session_connection_pool.add_transport(
                        cfg.client_id, transport, skill_name=None
                    )

                # Convert to config entry
                entry = McpConfigEntry(
                    server_config=cfg,
                    source="session",
                    skill_name=None,
                )
                new_entries.append(entry)

        except TimeoutError:
            self.log.warning("MCP server init timed out", server_name=server.name)
        except Exception:
            self.log.exception("Failed to setup MCP server", server_name=server.name)

    await asyncio.gather(*[_init_server(s) for s in self.mcp_servers])

    # Update session's MCP config snapshot with new session-level entries
    if new_entries:
        existing = self.agent._mcp_snapshot
        if existing is not None:
            # Merge: add new session configs (dedup by client_id)
            existing_ids = {e.server_config.client_id for e in existing.session_configs}
            merged = (*existing.session_configs,
                      *(e for e in new_entries if e.server_config.client_id not in existing_ids))
            self.agent._mcp_snapshot = existing.with_session_configs(merged)
        # Register MCP prompts as commands (unchanged)
        await self._register_mcp_prompts_as_commands()
```

Note: `McpConfigSnapshot.with_session_configs()` is defined in the component section above. It follows the same pattern as `with_skill_configs()` — returns a new frozen instance. This is used during initialization only; after initialization completes, session configs are immutable.

### 9. Modified `resume_session()` (session/load semantics)

```python
async def resume_session(self, session_id, ..., mcp_servers=None) -> ACPSession | None:
    # ... load from store ...

    # Check if session is already active
    if session_id in self._acp_sessions:
        # ALREADY ACTIVE: ignore mcpServers from session/load
        # Existing snapshot wins — no overwrite
        logger.info("Session already active, ignoring session/load mcpServers",
                     session_id=session_id)
        return self._acp_sessions[session_id]

    # RESTORING from storage: use session/load's mcpServers
    session = ACPSession(
        session_id=session_id,
        agent=session_agent,
        mcp_servers=mcp_servers,  # Used to build snapshot
        ...
    )
    await session.initialize()
    await session.initialize_mcp_servers()  # Builds snapshot from mcpServers
    self._acp_sessions[session_id] = session
    return session
```

## Data Flow

### Normal Session Creation (Parent)

```
1. ACP client sends session/new with mcpServers=[workspace-fs]
2. ACPSessionManager.create_session()
   → creates ACPSession with mcp_servers=[workspace-fs]
   → session.initialize()
   → session.initialize_mcp_servers()
     → converts to McpConfigEntry(source="session")
     → builds McpConfigSnapshot
3. SessionPool.get_or_create_session_agent()
   → creates Agent from config
   → agent.__aenter__()
   → sets agent._mcp_snapshot = snapshot
   → sets agent._session_connection_pool = SessionConnectionPool(session_id)
   → adds pool-level NON-MCP providers (skills, etc.) to agent.tools
4. User sends prompt
5. _run_stream_run_turn() → get_agentlet()
   → reads _mcp_snapshot
   → for pool/agent configs: borrow from GlobalConnectionPool
   → for session configs: borrow from SessionConnectionPool
   → create MCPToolset per server (fresh, no cache)
   → return agentlet with ALL tools
```

### Child Session Creation (Subagent)

```
1. Parent agent's LLM calls task() tool with agent="librarian"
2. subagent_tools.py:
   → ctx.create_child_session(agent_name="librarian", parent_session_id=parent_sid)
     → SessionPool.create_session(parent_session_id=parent_sid)
     → get_or_create_session_agent(child_sid, "librarian")
       → creates child Agent from config
       → agent.__aenter__()
       → builds McpConfigSnapshot:
           pool_configs = parent's pool_configs (same pool)
           agent_configs = child's own YAML config (NOT inherited)
           session_configs = parent's session_configs (INHERITED!)
           skill_configs = () (populated lazily)
       → sets agent._mcp_snapshot = child_snapshot
       → sets agent._session_connection_pool = SessionConnectionPool(child_sid)

   → SpawnSessionStart event emitted
   → ACP client receives, sends session/load with mcpServers=[workspace-fs]
     → resume_session()
     → session ALREADY ACTIVE → IGNORES session/load mcpServers
     → (snapshot already has workspace-fs from inheritance)

   → session_pool.run_stream(child_sid, prompt)
     → get_or_create_session_agent(child_sid) → returns cached agent
     → get_agentlet()
       → reads _mcp_snapshot (has workspace-fs INHERITED!)
       → borrows from SessionConnectionPool (creates NEW connection for child)
       → ALL tools available, including workspace-fs
       → NO RACE — snapshot was set at agent creation, before run_stream
```

### Session Restore (session/load on stored session)

```
1. ACP client sends session/load with mcpServers=[workspace-fs, ...]
2. resume_session()
   → session NOT in _acp_sessions (restoring from storage)
   → loads session data from store
   → creates fresh ACPSession
   → session.initialize()
   → session.initialize_mcp_servers()
     → converts mcpServers to McpConfigEntry(source="session")
     → builds McpConfigSnapshot with these configs
   → get_or_create_session_agent()
     → agent._mcp_snapshot = snapshot (with load's mcpServers)
   → All tools available immediately
```

### Skill MCP Integration

```
1. Agent starts, SkillMcpManager discovers skill MCP configs
2. When skill is loaded/invoked during a run:
   → SkillMcpManager registers MCP config
   → Creates McpConfigEntry(source="skill", skill_name="my_skill")
   → Updates agent._mcp_snapshot via with_skill_configs()
   → Connection created lazily in SessionConnectionPool
3. get_agentlet() (next call)
   → reads snapshot including skill_configs
   → borrows from SessionConnectionPool with skill_name key
   → skill MCP tools available
```

### MCP-over-ACP Bridge

```
1. ACP-transport MCP server (e.g., workspace-fs) provided by ACP client
2. initialize_mcp_servers()
   → For AcpMcpServer: calls acp_agent.connect_acp_mcp_server(server)
   → Gets AcpMcpConnection
   → Creates AcpMcpTransport (wraps ACP JSON-RPC tunnel)
   → Stores transport in SessionConnectionPool
   → Adds McpConfigEntry to snapshot
3. get_agentlet()
   → Reads snapshot, sees workspace-fs entry
   → Borrows transport from SessionConnectionPool
   → Creates MCPToolset(client=transport, ...)
   → workspace-fs tools available to agent
```

## Error Handling

| Scenario | Handling |
|----------|----------|
| Pool-level MCP server down | `GlobalConnectionPool.get_transport()` raises after timeout (30s). `get_agentlet()` catches per-server, logs warning, continues with remaining servers. |
| Session-level MCP server down | Same — per-server try/catch, graceful degradation. Agent runs with partial tools. |
| Owner-task crashes (stdio) | `ready_event` never set → timeout (30s) → `get_transport()` raises. Connection entry marked as error, future calls fail fast. |
| Connection pool exhausted | Global: no hard limit (ref-counted, shared). Session: no hard limit (per-session). LRU eviction at MAX_SESSIONS=256 for global stdio pool. |
| Session closed with active connections | `SessionConnectionPool.cleanup()` signals all owner-tasks via `close_event`, waits with 5s timeout, force-cancels remaining. |
| Pool shutdown | `GlobalConnectionPool.shutdown_all()` signals all owner-tasks, waits with 10s timeout, force-cancels remaining. |
| Cross-task CancelScope | Eliminated by design: owner-task pattern (stdio) + no caching (HTTP/SSE) + fresh MCPToolset per agentlet. |
| Config snapshot conflict (session/load on active session) | Ignored — existing snapshot wins. Logged as info. |
| Skill MCP server fails to connect | `SkillMcpManager` catches, logs, skill tools not available. Skill instructions still injected (tools fail at call time). |
| ACP transport MCP disconnect | `AcpMcpTransport` raises on next tool call. `MCPToolset` propagates error to agent. Agent can retry or report to user. |

## Testing Strategy

### Unit Tests

| Component | Tests |
|-----------|-------|
| `McpConfigSnapshot` | Immutability (frozen=True), `with_skill_configs()` returns new instance, `global_configs` and `session_scoped_configs` partition correctly, dedup by `client_id` |
| `McpConfigEntry` | Frozen dataclass, `source` field validation, `skill_name` optional |
| `GlobalConnectionPool` | Owner-task lifecycle (start, ready, release, shutdown), ref counting (N acquire → 1 release → still alive, N acquire → N release → shutdown), concurrent access (threading.Lock), LRU eviction, stdio vs HTTP/SSE path selection |
| `SessionConnectionPool` | Per-session isolation (two sessions → two connections), skill_name key isolation, lazy creation, cleanup with timeout, cleanup force-cancel |

### Integration Tests

| Test | What it verifies |
|------|-----------------|
| `test_child_session_inherits_parent_session_mcp` | Child agent's `_mcp_snapshot.session_configs` contains parent's session configs |
| `test_child_session_has_own_agent_configs` | Child agent's `_mcp_snapshot.agent_configs` come from child's YAML, not parent's |
| `test_child_session_has_pool_configs` | Child agent's `_mcp_snapshot.pool_configs` match parent's |
| `test_session_load_on_active_ignored` | `resume_session()` on active session does NOT replace snapshot |
| `test_session_load_on_restore_uses_mcpServers` | `resume_session()` on stored session DOES use `mcpServers` param |
| `test_get_agentlet_has_all_mcp_tools` | `get_agentlet()` returns agentlet with pool + agent + session MCP tools |
| `test_cross_task_no_cancel_scope_error` | Parent task enters, child task exits — no RuntimeError |
| `test_stateful_mcp_isolation` | Session A's MCP state not visible to Session B |
| `test_skill_mcp_in_snapshot` | After skill load, `skill_configs` in snapshot, tools available in `get_agentlet()` |
| `test_mcp_bypasses_providers_list` | `agent.tools.providers` does NOT contain MCP providers |
| `test_global_pool_ref_counting` | 3 sessions sharing 1 pool server → 1 connection, 3 refs |
| `test_session_pool_cleanup` | Session close → all session connections cleaned up |
| `test_global_pool_shutdown` | Pool shutdown → all global connections cleaned up |
| `test_owner_task_same_task_enter_exit` | Owner task enters and exits CM in same asyncio task |
| `test_streaming_adapter_no_cancel_scope` | Streaming adapter shutdown does not trigger CancelScope error |

### Manual QA

| Test | What it verifies |
|------|-----------------|
| Run `diag-agent-ng.yaml` config | Parent agent (engineer) has pool + agent + session MCP tools |
| Spawn subagent (librarian) | Subagent has pool + inherited session MCP tools (workspace-fs) |
| Multiple subagents in parallel | Each subagent has isolated SessionConnectionPool, no CancelScope errors |
| Session restore via session/load | Restored session has MCP tools from `mcpServers` param |
| Skill with MCP server | Skill MCP tools available after skill load |

## Migration Path

### Phase 1: Config Snapshot + Inheritance (eliminates race condition)

1. Create `McpConfigEntry` and `McpConfigSnapshot` dataclasses
2. Modify `get_or_create_session_agent()` to build snapshot at agent creation
3. Child sessions inherit parent's `session_configs`
4. Modify `resume_session()` — ignore `session/load` on active sessions
5. Modify `initialize_mcp_servers()` — build config entries instead of live providers
6. `get_agentlet()` reads from snapshot (fallback to legacy if no snapshot)

**Result**: Race condition eliminated. Subagents have inherited MCP tools.

### Phase 2: Connection Pools (eliminates CancelScope errors)

7. Implement `GlobalConnectionPool` with owner-task pattern
8. Implement `SessionConnectionPool`
9. Modify `MCPManager.as_capability()` to use pools
10. Remove MCP providers from `agent.tools.providers` path

**Result**: Cross-task CancelScope eliminated. Connections shared efficiently.

### Phase 3: Skill MCP Integration

11. Modify `SkillMcpManager` to register configs in snapshot via `with_skill_configs()`
12. Skill MCP connections go through `SessionConnectionPool` with `skill_name` key

**Result**: Skill MCP fully integrated into the architecture.

### Phase 4: Cleanup + Testing

13. Remove old `session_mcp_providers` list from `ACPSession`
14. Remove old `agent.tools.add_provider(mcp_provider)` calls for MCP
15. Remove debug logging added during debugging
16. Write comprehensive test suite
17. Run full test suite + manual QA

**Result**: Clean, production-ready architecture.

## OpenSpec Change

This design should be captured as a new OpenSpec change: **`mcp-provider-lifecycle-architecture`**.

Existing changes that overlap:
- `fix-subagent-mcp-inheritance` (37/37 complete, didn't fully fix) — superseded by this design
- `session-pool-architecture` (0/66) — partially overlaps, should be reconciled
- `unify-tool-interception-to-pydantic-ai-capabilities` (0/41) — partially overlaps, should be reconciled

## References

- **deer-flow** (`bytedance/deer-flow`): Owner-task pattern, `threading.Lock`, LRU eviction, transport-aware materialization
- **Zed Editor**: Shared registry by reference for subagents, `ContextServerStore` lifecycle manager
- **oh-my-opencode**: Per-session connection key, three-tier config merge, single gateway tool, race prevention mechanisms
- **pydantic-ai issue #2818**: Known MCPToolset cross-task issue
- **Oracle analysis**: Three-layer architecture recommendation (config snapshot + connection pool + lazy materialization)
