# Agent Pool Backdoor Cleanup — Pre-M4 Baseline

## Problem

`MessageNode.agent_pool` is a direct reference to the full `AgentPool` god-class on every agent. RFC #135 identifies this as the **#1 architectural risk**: it spans Layers 2–5, blocks tenant isolation, makes layer boundaries permeable, and prevents clean dependency injection.

M1 introduced `HostContext` as the replacement — a frozen dataclass carrying only infrastructure service references. M2 Task 11 was supposed to migrate all call sites but was marked `[x]` while only completing Phase 1 (core agents). Protocol servers (ACP, OpenCode) were never migrated. Additionally, `HostContext.pool: AgentPool | None` was added as a back-reference, re-creating the exact backdoor the architecture was designed to eliminate.

### Current State

- **~64 external `.agent_pool` property accesses** across 15 files (AGENTS.md's "18 references" is stale)
- **`HostContext.pool` back-reference** defeats immutability — any consumer can reach the full mutable pool
- **`AgentFactory` holds its own `self._pool`** and also reads `host_context.pool` (redundant)
- **Skill orchestration logic** (`is_skill_visible_to_node`, `get_skill_instructions_for_node`, `skill_capabilities`, `skill_provider`, `skill_commands`) lives on `AgentPool` with no service abstraction
- **Protocol servers** (ACP, OpenCode) directly receive `AgentPool` in constructors, bypassing the DI design

### Root Cause

The original M1 design was correct: `HostContext` = immutable DI bundle, `AgentPool` = lifecycle owner. The implementation drifted in three ways:

1. `pool` back-reference added to `HostContext` as temporary escape hatch, never removed
2. Skill business logic not extracted into a service, left on `AgentPool`
3. Protocol servers never migrated to receive `HostContext` (M2 Phase 2 deferred, never completed)

## Design

### Architecture Alignment (RFC #135)

The six-layer architecture defines clear boundaries:

```
Layer 1: ConfigRegistry    — versioned config storage
Layer 2: AgentHost          — owns mutable infrastructure, constructs HostContext
Layer 3: AgentFactory       — compiles (manifest, host_context) → AgentRegistry
Layer 4: RunLoop            — drives idle → running → idle cycle
Layer 5: Agent Core         — MessageNode, receives AgentContext at runtime
Layer 6: ProtocolServer     — translates wire protocols ↔ RunLoop
```

`HostContext` is the **Layer 2 → Layer 3/4/5 dependency injection bundle**. It carries service references (not raw pool attributes). Each field is an independent service object with its own responsibilities.

`AgentContext` (Layer 4 → 5) carries per-run state. `AgentContext.host: HostContext | None` is "rarely needed, for advanced tools" per RFC #135.

### Changes

#### 1. Extract `SkillService` Protocol

Skill orchestration logic on `AgentPool` forms a cohesive cluster that should be a service, not pool attributes:

**State**: `skill_capabilities`, `skill_provider`, `skill_commands`, `_skill_resolver`
**Query methods**: `is_visible_to_node(skill, node_name)`, `get_instructions_for_node(skill_name, node_name)`

```python
# capabilities/skill_service.py — NEW FILE

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from wolfharness_toolsets.builtin.combined_toolset import CombinedToolsetCapability
    from wolfharness.skills.command_registry import SkillCommandRegistry

@runtime_checkable
class SkillService(Protocol):
    """Pool-scoped skill operations service.

    Encapsulates skill capability management, scope-based visibility,
    and instruction loading without exposing the full AgentPool.
    AgentPool implements this protocol; HostContext carries the reference.
    """

    @property
    def capabilities(self) -> list[Any]:
        """Pool-scoped SkillCapability instances created during __aenter__."""
        ...

    @property
    def provider(self) -> CombinedToolsetCapability | None:
        """Combined toolset for skill URI resolution."""
        ...

    @property
    def commands(self) -> SkillCommandRegistry | None:
        """Skill command registry for slash commands."""
        ...

    def is_visible_to_node(self, skill: Any, node_name: str | None) -> bool:
        """Check if a skill is visible to a node's package scope."""
        ...

    async def get_instructions_for_node(self, skill_name: str, node_name: str) -> str:
        """Load skill instructions using a target node's package scope."""
        ...
```

Write operations (`register_skill_provider`, `unregister_skill_provider`) are **excluded** — they only run during pool `__aenter__`, never through `HostContext` at runtime.

#### 2. Extend `HostContext`

Add two fields:

```python
# host/context.py — MODIFIED

@dataclass(frozen=True)
class HostContext:
    # ... existing 17 fields unchanged ...
    main_agent_name: str | None = None          # NEW: from pool constructor param
    skill_service: SkillService | None = None   # NEW: skill orchestration service
    # pool: AgentPool[Any] | None = None        # REMOVED in Phase 4
```

- `main_agent_name`: resolved from pool constructor param or `manifest.default_agent`. Simple string, same category as `config_file_path`.
- `skill_service`: `AgentPool` implements `SkillService`. HostContext carries the reference. Same pattern as `mcp: MCPManager`, `storage: StorageManager`.

Config-derived data that does **not** need new fields:
- `agent_configs` → callers use `ctx.manifest.agents`
- `compaction_pipeline` → callers use `ctx.manifest.get_compaction_pipeline()`
- `main_agent_config` → callers use `ctx.manifest.agents[ctx.main_agent_name]`

#### 3. Update `AgentPool.get_context()`

```python
def get_context(self) -> HostContext:
    self._host_context = HostContext(
        # ... existing fields ...
        main_agent_name=self.main_agent_name,
        skill_service=self,  # AgentPool implements SkillService
        # pool=self,  # REMOVED in Phase 4
    )
    return self._host_context
```

#### 4. Add `MessageNode._bind_pool()` for internal wiring

```python
# messaging/messagenode.py — ADDED

def _bind_pool(self, pool: AgentPool[Any] | None) -> None:
    """Internal: bind node to pool for host_context access.

    Used by Talk wiring to propagate pool reference to callback-created nodes.
    This is the only legitimate writer of _agent_pool outside __init__.
    """
    self._agent_pool = pool
```

#### 5. Migrate all `.agent_pool` callers to `.host_context`

Complete access mapping:

| Old access | New access | Phase |
|---|---|---|
| `agent_pool.manifest` | `host_context.manifest` | 2-3 |
| `agent_pool.session_pool` | `host_context.session_pool` | 2-3 |
| `agent_pool.storage` | `host_context.storage` | 2-3 |
| `agent_pool.mcp` | `host_context.mcp` | 2-3 |
| `agent_pool.skills` | `host_context.skills_registry` | 2-3 |
| `agent_pool.prompt_manager` | `host_context.prompt_manager` | 2-3 |
| `agent_pool.main_agent_name` | `host_context.main_agent_name` | 2-3 |
| `agent_pool.agent_configs` | `host_context.manifest.agents` | 2-3 |
| `agent_pool.compaction_pipeline` | `host_context.manifest.get_compaction_pipeline()` | 2-3 |
| `agent_pool.skill_capabilities` | `host_context.skill_service.capabilities` | 2 |
| `agent_pool.skill_provider` | `host_context.skill_service.provider` | 2 |
| `agent_pool.skill_commands` | `host_context.skill_service.commands` | 3 |
| `agent_pool.is_skill_visible_to_node()` | `host_context.skill_service.is_visible_to_node()` | 2 |
| `agent_pool.get_skill_instructions_for_node()` | `host_context.skill_service.get_instructions_for_node()` | 2 |

#### 6. Migrate `ACPProtocolHandler` to receive `HostContext`

```python
# handler.py — MODIFIED

class ACPProtocolHandler(ProtocolEventConsumerMixin):
    def __init__(
        self,
        host_context: HostContext,  # was: agent_pool: AgentPool[Any]
        ...
    ) -> None:
        super().__init__()
        self._host_context = host_context
        ...

    @property
    def event_bus(self) -> EventBus:
        session_pool = self._host_context.session_pool
        ...
```

#### 7. Migrate `ACPSession.agent_pool` property

```python
# session.py — MODIFIED

@property
def host_context(self) -> HostContext:  # was: def agent_pool
    ctx = self.agent.host_context
    if ctx is None:
        raise RuntimeError("Agent has no associated pool")
    return ctx
```

#### 8. Remove `pool` from `HostContext` (Phase 4)

- `AgentFactory`: 3 uses of `host_context.pool` → `self._pool` (already exists)
- `talk/talk.py`: 2 uses of `ctx.pool` → `self.source._agent_pool` + `other._bind_pool(pool)`
- `host/context.py`: remove `pool` field
- `delegation/pool.py`: remove `pool=self` from `get_context()`

#### 9. Remove `MessageNode.agent_pool` property (Phase 5)

- Remove `agent_pool` property getter + setter from `messagenode.py`
- Keep `_agent_pool` private field (needed for `host_context` property)
- Update `storage` property to go through `host_context`
- Migrate all test files referencing `.agent_pool`

## Phased Implementation

### Dependency Graph

```
Phase 1 (Foundation) ── no behavior change, new structures
  ├──→ Phase 2 (Core migration) ── agents stop using agent_pool ──┐
  └──→ Phase 3 (Server migration) ── servers stop using agent_pool ┤
                                                                     ↓
                                          Phase 4 (Backdoor removal)
                                                     │
                                                     ↓
                                          Phase 5 (Property removal)
```

Phase 2 and Phase 3 can run in **parallel** — both depend only on Phase 1.

### Phase 1: Foundation

**Goal**: Create new structures alongside existing ones. Zero behavior change.

| Task | File | Type |
|---|---|---|
| Create `SkillService` Protocol | `capabilities/skill_service.py` (new) | New file |
| AgentPool implements SkillService | `delegation/pool.py` | No code change (duck-typed) |
| Add `main_agent_name` + `skill_service` to HostContext | `host/context.py` | Add 2 fields |
| Update `get_context()` to populate new fields | `delegation/pool.py` | Add 2 lines |
| Add `_bind_pool()` method | `messaging/messagenode.py` | Add method |

**Verification**: All tests pass. New fields default to `None`, no behavior change.

### Phase 2: Core Migration

**Goal**: Core agent code (~9 refs, 4 files) stops using `agent_pool` property.

| File | Refs | Key changes |
|---|---|---|
| `agents/native_agent/agent.py` | 3 | `skill_capabilities` → `skill_service.capabilities`; `is_skill_visible_to_node` → `skill_service.is_visible_to_node` |
| `delegation/base_team.py` | 3 | `skill_provider` → `skill_service.provider`; `get_skill_instructions_for_node` → `skill_service.get_instructions_for_node` |
| `wolfharness_commands/utils.py` | 2 | `manifest.config_file_path` → `ctx.config_file_path` |
| `shared/model_utils.py` | 1 | `agent.agent_pool` → `agent.host_context` |

**Verification**: Core agent code no longer triggers `DeprecationWarning`. Unit tests pass.

### Phase 3: Server Migration

**Goal**: Protocol servers (~46 refs, 9 files) stop using `agent_pool`.

**3a: ACPProtocolHandler signature** (handler.py, 7 refs)
- Constructor: `agent_pool: AgentPool` → `host_context: HostContext`
- All `self.agent_pool.session_pool` → `self._host_context.session_pool`

**3b: AgentPoolACPAgent** (acp_agent.py, 28 refs)
- `self.agent_pool.manifest.X` → `ctx.manifest.X`
- `self.agent_pool.main_agent_name` → `ctx.main_agent_name`
- `self.agent_pool.session_pool` → `ctx.session_pool`
- `agent.agent_pool` on other objects → `agent.host_context`
- Constructor arg `agent_pool=self.agent_pool` → `host_context=self.host_context`

**3c: ACPSession** (session.py, 11 refs)
- `agent_pool` property → `host_context` property (delegates to `self.agent.host_context`)
- `self.agent_pool.skills` → `ctx.skills_registry`
- `self.agent_pool.agent_configs` → `ctx.manifest.agents`
- `self.agent_pool.skill_commands` → `ctx.skill_service.commands`
- `self.agent_pool.prompt_manager` → `ctx.prompt_manager`

**3d: OpenCode server** (4 files, 8 refs)
- `state.py`: `agent.agent_pool` → `agent.host_context`
- `server.py`: `agent.agent_pool.session_pool` → `ctx.session_pool`
- `session_routes.py`: `agent.agent_pool.compaction_pipeline` → `ctx.manifest.get_compaction_pipeline()`
- `agent_routes.py`: `state.agent.agent_pool` → `state.agent.host_context`

**3e: Debug commands** (debug_commands.py, 1 ref)
- `session.agent_pool.manifest.agents` → `ctx.manifest.agents`

**Verification**: All server code no longer triggers `DeprecationWarning`. ACP snapshot tests + OpenCode integration tests pass.

### Phase 4: Backdoor Removal

**Goal**: `HostContext.pool` removed. No path back to `AgentPool` through `HostContext`.

| File | Change |
|---|---|
| `host/factory.py` (3 refs) | `host_context.pool` → `self._pool` |
| `talk/talk.py` (2 refs) | `ctx.pool` → `self.source._agent_pool`; `other.agent_pool = ...` → `other._bind_pool(...)` |
| `host/context.py` | Remove `pool` field |
| `delegation/pool.py` | Remove `pool=self` from `get_context()` |

**Verification**: `grep -r 'host_context.pool' src/` returns 0. All tests pass.

### Phase 5: Property Removal

**Goal**: `MessageNode.agent_pool` property removed.

| File | Change |
|---|---|
| `messaging/messagenode.py` | Remove `agent_pool` property + setter; update `storage` property |
| `AGENTS.md` | Remove deprecation section |
| Test files | Migrate remaining `.agent_pool` references |

**Verification**: `grep -r '\.agent_pool\b' src/ tests/` returns 0 (excluding `_agent_pool` private field and constructor `agent_pool=` kwargs).

## Pre-M4 Baseline Definition

**Phases 1–4 = clean baseline**:

- ✅ All code accesses infrastructure through `host_context`
- ✅ `HostContext` has no `pool` back-reference
- ✅ `AgentFactory` uses own `self._pool` (not through `HostContext`)
- ✅ `SkillService` extracted as independent Protocol
- ✅ Protocol servers receive `HostContext`
- ⏸ `agent_pool` property still exists (Phase 5 optional, can defer to M4)

Phase 5 is **nice-to-have**: the property remaining doesn't affect M4 development since all internal code has migrated. But removing it is the final clean state.

## Risks

| Risk | Mitigation |
|---|---|
| `SkillService` is mutable (capabilities list rebuilt at runtime) | Acceptable — `HostContext` freezes the reference, not the referenced object's state. Same pattern as `MCPManager`. |
| ACPProtocolHandler constructed before pool `__aenter__` | Not possible — handler is created in `AgentPoolACPAgent.__post_init__`, after pool initialization. |
| Test files reference `.agent_pool` | Phase 5 includes test migration. Phases 1–4 leave property in place, so tests still pass. |
| `AgentFactory` still holds `self._pool` | Acknowledged. Full removal of pool reference from Factory requires config model refactoring (`cfg.get_agent(pool=...)`), which is M4 scope (config split). Pre-M4 baseline accepts Factory holding pool via constructor, not through HostContext. |
