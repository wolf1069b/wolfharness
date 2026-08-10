# capabilities — AbstractCapability System

## Where to Look

| Task | File |
|---|---|
| AgentContext frozen dataclass | `agent_context.py` |
| DelegationService Protocol | `delegation.py` |
| ResourceSource Protocol + AggregatedResourceSource | `resource_source.py` |
| Entry-point capability discovery | `registry.py` |
| RunLoopDelegationService (M3) | `runloop_delegation.py` |
| MCPCapability | `mcp_capability.py` |
| SubagentCapability | `subagent_capability.py` |
| FunctionToolsetCapability | `function_toolset.py` |
| CombinedToolsetCapability | `combined_toolset.py` |
| FilteredToolsetCapability | `filtered_toolset.py` |
| CodeModeCapability | `code_mode_capability.py` |
| ExtensionRegistry (4-level scope) | `extension_registry.py` |
| ModalityFilterCapability | `modality_filter.py` |
| Modality classification utils | `modality_utils.py` |

## Conventions

- **One MCP server per capability**: Each `MCPCapability` wraps exactly one server. Use `CombinedToolsetCapability` to combine them.
- **ResourceSource is orthogonal to AbstractCapability**: Same object can implement both. `ResourceSource` is for read-only data access (list, read, exists, on_change).
- **AgentContext is frozen**: Constructed by RunLoop per-turn. Carries `agent_registry`, `delegation`, `session`, `scope`, `resources`, `host`.
- **DelegationService limits exposure**: `spawn_subagent(name, prompt)` and `get_available_agents()` only. Does not expose full `AgentPool`.
- **SkillCapability injection order**: In `get_agentlet()`, skill capabilities are injected at position 5 (after MCP, deferred bridge, approval bridge, and hook capabilities).
- **Entry-point registry**: Custom capabilities discovered via `wolfharness.capabilities` entry-point group.
- **ExtensionRegistry scope hierarchy**: `POOL → AGENT → SESSION → TURN`. Agents outlive sessions (AGENT scope keyed by `agent_name` only, no `session_id`). `clear_session()` removes SESSION and TURN entries during teardown. Factory registers config-derived caps at AGENT scope; `get_agentlet()` registers session-specific caps (MCP, SkillManagerCap, ResourceCapability) at SESSION scope.
- **ScopeLevel reorder is breaking**: `Scope` field order changed — `agent_name` is now 2nd (after `level`), `session_id` is 3rd. AGENT scope queries no longer include SESSION caps.

## Anti-Patterns

- **Accessing `agent_pool` read-only**: Use `host_context` (immutable `HostContext`) instead. The `agent_pool` property emits `DeprecationWarning` as of M2.
- **Direct tool code in `tools/`**: Tool framework goes in `tools/`. Concrete implementations go in `tool_impls/`. Capabilities wrap toolsets, not individual tools.
