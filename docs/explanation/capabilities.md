# Capabilities (M3 — Replaces Resource Providers)

In M3, the old `ResourceProvider` hierarchy was replaced with native pydantic-ai `AbstractCapability` / `AbstractToolset` implementations. Each `AbstractCapability` produces tools, instructions, change notifications, and optionally implements `ResourceSource` for read-only data access. The old `src/wolfharness/resource_providers/` directory (14 files, ~3860 LOC) was physically deleted after migration.

## Capability Registry

| Capability | Replaces | Key File |
|---|---|---|
| `MCPCapability` | `MCPResourceProvider` | `capabilities/mcp_capability.py` |
| `SkillCapability` | `LocalResourceProvider` | `skills/capability.py` |
| `SubagentCapability` | `PoolResourceProvider` | `capabilities/subagent_capability.py` |
| `FunctionToolsetCapability` | `StaticResourceProvider` | `capabilities/function_toolset.py` |
| `CombinedToolsetCapability` | `AggregatingResourceProvider` | `capabilities/combined_toolset.py` |
| `FilteredToolsetCapability` | `FilteringResourceProvider` | `capabilities/filtered_toolset.py` |
| `CodeModeCapability` | `CodeModeResourceProvider` | `capabilities/code_mode_capability.py` |

## Supporting Types

- `ResourceSource` (`capabilities/resource_source.py`) — `@runtime_checkable Protocol` for read-only data access (`list()`, `read(uri)`, `exists(uri)`, `on_change()`). Orthogonal to `AbstractCapability` — same object can implement both.
- `AggregatedResourceSource` — Composes multiple `ResourceSource` instances, routes by URI scheme.
- `AgentContext` (`capabilities/agent_context.py`) — Frozen dataclass carrying `agent_registry`, `delegation`, `session`, `scope`, `resources`, `host`. Constructed by RunLoop per-turn.
- `DelegationService` (`capabilities/delegation.py`) — Protocol exposing `spawn_subagent(name, prompt)` and `get_available_agents()`. Limits tools to operations they need without exposing `AgentPool`.
- `ChangeEvent` (`capabilities/change_event.py`) — Frozen dataclass for capability change notifications (`on_change()` stream).
- Entry-point registry (`capabilities/registry.py`) — Discovers custom capabilities via `wolfharness.capabilities` entry-point group.
- `ExtensionRegistry` (`capabilities/extension_registry.py`) — Unified capability registry with 4-level scope storage. See [ExtensionRegistry and Scope Hierarchy](#extensionregistry-and-scope-hierarchy) below.

## ExtensionRegistry and Scope Hierarchy

**`ExtensionRegistry`** (`capabilities/extension_registry.py`) is the unified capability registry with 4-level scope storage. It replaces fragmented infrastructure (SkillURIResolver._providers, AggregatedResourceSource) with a single registry that supports pool, agent, session, and turn-level capability scoping.

**Scope hierarchy (outer → inner):**

```
POOL → AGENT → SESSION → TURN
```

| Level | Visibility | Key |
|---|---|---|
| `POOL` | Visible to all agents and sessions | — (global list) |
| `AGENT` | Visible to a specific named agent across all sessions | `agent_name` |
| `SESSION` | Visible within a specific session (e.g. MCP connections) | `session_id` |
| `TURN` | Visible for one turn (guarded by `asyncio.Lock`) | `session_id` + `agent_name` + `turn_id` |

**`Scope`** is an immutable frozen dataclass identifying where a capability is visible:

```python
@dataclass(frozen=True, slots=True)
class Scope:
    level: ScopeLevel
    agent_name: str = ""    # required for AGENT/TURN
    session_id: str = ""    # required for SESSION/TURN
    turn_id: str = ""       # required for TURN
```

**Query semantics** (`get_visible_capabilities(scope)`): walks the hierarchy from outer to inner, collecting all capabilities at each level:

| Query Scope | Visible Capabilities |
|---|---|
| `POOL` | POOL |
| `AGENT` | POOL + AGENT |
| `SESSION` | POOL + AGENT + SESSION |
| `TURN` | POOL + AGENT + SESSION + TURN |

**Session teardown:** `clear_session(session_id)` removes SESSION and TURN level entries for a given session. AGENT and POOL level caps are NOT affected (they outlive sessions). Called during `_close_session_unlocked()` as step 6b.

### Migration: ScopeLevel Hierarchy Reorder

The `ScopeLevel` enum has been reordered from `POOL > SESSION > AGENT > TURN` to `POOL > AGENT > SESSION > TURN`. This reflects the true lifecycle: agents outlive sessions.

**Breaking changes for Scope construction:**

- `Scope(level=ScopeLevel.AGENT, session_id=..., agent_name=...)` → `Scope(level=ScopeLevel.AGENT, agent_name=...)` (session_id no longer needed for AGENT scope)
- `Scope` dataclass field order changed: `agent_name` is now the 2nd field (after `level`), `session_id` is 3rd
- SESSION scope queries now include AGENT scope caps (POOL + AGENT + SESSION)
- AGENT scope queries no longer include SESSION scope caps (POOL + AGENT only)

## Deleted Alongside ResourceProviders

- `src/wolfharness/tools/factory.py` (194 LOC, 6 `ToolsetFactory` classes) — became dead code after all providers migrated.
- `src/wolfharness/tools/manager.py` (364 LOC, `ToolManager`) — all `agent.tools.X` access migrated to direct capability references.
