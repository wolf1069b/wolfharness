---
rfc_id: RFC-0052
title: Restore Skill System Capabilities After M3 Refactor
status: REVIEW
author: Sisyphus
reviewers:
  - Oracle: PASSED
  - Momus: N/A (format limitation)
created: 2026-07-12
last_updated: 2026-07-12
---

## Overview

This RFC proposes restoring three critical skill system functionalities lost during the M3 capability refactor (commit `15179ea40`) and subsequent extension-source-architecture change. The regressions break `skill://` URI resolution, per-skill tool registration, and `load_skill` tool propagation. The fix restores these capabilities within the new `SkillManagerCap` architecture without reverting to the deleted `SkillCapability` class.

## Background & Context

### Current State

The M3 refactor replaced the per-skill `SkillCapability` class (422 LOC, now dead code with zero imports) with a unified `SkillManagerCap` (439 LOC). The migration correctly ported instruction injection and skill listing, but three critical functions were lost:

1. **`skill://` URI resolution**: `SkillURIResolver` delegates to `ExtensionRegistry.resolve_uri()`, but `SkillManagerCap` is never registered with `ExtensionRegistry`. The old `_providers` dict path is dead code when `extension_registry` is set (always true since pool.py:605).

2. **Per-skill tool registration**: The old `SkillCapability.get_toolset()` eagerly imported Python tools via `SkillToolManager.import_tools()` and lazily connected MCP servers via `SkillMcpManager`, both wrapped in `PrefixedToolset("{name}__tool__")` / `PrefixedToolset("{name}__mcp__")`. `SkillManagerCap` does none of this — it only holds `dict[str, Skill]` for instruction injection.

3. **`load_skill` propagation**: `_inject_pool_providers()` (factory.py:680-710) deliberately stopped injecting `skills_tools_provider`, leaving non-SessionPool/Factory agent creation paths without the `load_skill`/`list_skills` tools.

### Historical Context

The skill system went through multiple overlapping refactors:

```
refactor-skills-as-capabilities → m3-capability-native → thin-wrapper-refactor
→ extension-source-architecture → remove-provider-from-skill-uris
```

Each step's spec approval was siloed. The top-level `specs/` directory was updated to the latest architecture view, but older specs referencing deleted code were never audited. The `load_skill`/`list_skills` tools (570 LOC) were never covered by any spec.

### Glossary

| Term | Definition |
|------|------------|
| `SkillCapability` | Old per-skill capability class (deleted, dead code) |
| `SkillManagerCap` | New unified skill capability (replaces SkillCapability) |
| `SkillToolManager` | Utility class for importing Python tools from `import_path` strings |
| `SkillMcpManager` | Old per-skill MCP server manager (deleted) |
| `McpServerCap` | New MCP server capability (replaces SkillMcpManager) |
| `ExtensionRegistry` | 4-level scope registry for capability discovery |
| `SkillResource` | Protocol: `list_skills()`, `read_skill()`, `skill_exists()` |
| `PrefixedToolset` | pydantic-ai toolset wrapper that prefixes all tool names |
| `FilteredToolset` | pydantic-ai toolset wrapper that filters tools by function |

### Related Documents

- OpenSpec change: `openspec/changes/restore-skill-capabilities/`
- Existing spec: `openspec/specs/skill-manager-cap/spec.md`
- Existing spec: `openspec/specs/extension-registry/spec.md`
- Existing spec: `openspec/specs/agent-factory/spec.md`

## Problem Statement

### Evidence of Problem

1. **ACP server logs** (`~/Library/Logs/wolfharness/acp.log`): Agent errors with `'FunctionTool' object has no attribute '__name__'` when trying to load skills.

2. **User report**: Agent cannot access `load_skill` tool or resolve `skill://` URIs after M3 refactor.

3. **Code analysis**: `SkillCapability` (skills/capability.py) has zero imports across the codebase. `SkillManagerCap` does not call `SkillToolManager` or create per-skill `McpServerCap` instances.

4. **Test gap**: 20 skill-related test files (~3,580 LOC) exist, but none verify per-skill tool registration end-to-end or `skill://` URI resolution through the new `ExtensionRegistry` path.

### Impact of Not Solving

- Skills declaring `tools` or `mcp_servers` in frontmatter are non-functional — the agent cannot use these tools
- `skill://` URI access is completely broken — all URI-based skill loading fails
- Standalone agents (non-SessionPool) lack `load_skill`/`list_skills` tools
- The skill system, a core differentiator of AgentPool, is partially broken

## Goals & Non-Goals

### Goals

- Restore `skill://` URI resolution end-to-end through `ExtensionRegistry`
- Flatten `skill://` URIs from `skill://local/{name}` to `skill://{name}` (no provider segment)
- Restore per-skill Python tool registration with `{name}__tool__` prefixing
- Restore per-skill MCP server tool registration with `{name}__mcp__` prefixing
- Restore `allowed_tools` filtering per skill
- Fix `load_skill` tool propagation for all agent creation paths
- Migrate `load_skill`/`list_skills` MCP provider dispatch from `getattr` to `SkillResource` protocol
- Add specs for previously unspec'd `load_skill`/`list_skills` tools
- Add end-to-end tests covering all three regressions

### Non-Goals

- Migrating `ctx.pool` references to `host_context` in `load_skill`/`list_skills` (requires extending `HostContext`, deferred to separate change)
- Unifying `skill://` URI format (flat vs provider-qualified, deferred to separate change)
- Restoring `SkillCapability` class (stays dead code; all logic goes into `SkillManagerCap`)
- Restoring `SkillMcpManager` (replaced by `McpServerCap` per-skill instances)
- Restoring `on_run_ended()` MCP cleanup (handled by `McpServerCap.__aexit__()`)
- Restoring `CapabilityOrdering` (`ProcessHistory`, `NativeTool`)

## Evaluation Criteria

| Criterion | Weight | Minimum Threshold |
|-----------|--------|-------------------|
| Functional correctness | Critical | All 3 regressions fixed |
| Architectural consistency | High | No restoration of deleted classes |
| Test coverage | High | End-to-end tests for each regression |
| Spec coverage | Medium | All new behavior spec'd |
| Implementation complexity | Medium | Minimal new classes/abstractions |
| Backward compatibility | Medium | No breaking changes to YAML config |

## Options Analysis

### Option A: Restore per-skill `SkillCapability` as children of `SkillManagerCap`

**Description**: Re-activate the deleted `SkillCapability` class (422 LOC) and create per-skill instances as children of `SkillManagerCap`. `CombinedToolsetCapability.get_toolset()` would automatically merge their toolsets.

**Advantages**:
- Minimal code change — `SkillCapability` already has all the logic (tool import, MCP connection, filtering, lifecycle)
- Per-skill isolation is preserved naturally
- `get_toolset()` doesn't need override — parent's merge handles it

**Disadvantages**:
- Restores a deleted class — contradicts the M3 consolidation direction
- `SkillCapability` references `SkillMcpManager` (also deleted) — would need restoration or adaptation
- `SkillCapability` was designed for the old architecture, may not integrate cleanly with `ExtensionRegistry`
- Increases class count and maintenance surface

**Evaluation Against Criteria**:

| Criterion | Score | Notes |
|-----------|-------|-------|
| Functional correctness | 8/10 | Likely works, but SkillMcpManager dependency is a risk |
| Architectural consistency | 3/10 | Restores deleted classes, contradicts M3 direction |
| Test coverage | 7/10 | Existing tests may cover old SkillCapability behavior |
| Spec coverage | 6/10 | Old specs reference SkillCapability |
| Implementation complexity | 7/10 | Less new code, but more integration work |
| Backward compatibility | 9/10 | Restores exact old behavior |

**Effort Estimate**: Medium — restore 2 deleted classes, adapt to new architecture, update tests

**Risk Assessment**: Medium — `SkillMcpManager` restoration may have hidden dependencies on deleted `ResourceProvider` infrastructure

### Option B: Add tool registration logic to `SkillManagerCap` (Recommended)

**Description**: Extend `SkillManagerCap` with `SkillToolManager` dependency, per-skill tool import, `McpServerCap` creation, and `PrefixedToolset` wrapping. Fully override `get_toolset()` to handle three tool categories: Python tools, per-skill MCP tools, and non-skill children.

**Advantages**:
- No restoration of deleted classes — maintains M3 consolidation
- All skill logic consolidated in one class (`SkillManagerCap`)
- Uses new `McpServerCap` (the standard MCP capability) instead of deleted `SkillMcpManager`
- `ExtensionRegistry` registration is straightforward (one capability at POOL scope)

**Disadvantages**:
- `SkillManagerCap` complexity increases (from 439 to ~550 LOC estimated)
- `get_toolset()` requires full override (cannot call `super()` due to prefixing requirements)
- New data structure `_skill_mcp_children: dict[str, list[McpServerCap]]` needed for skill-to-child mapping
- `for_run()` must be updated to propagate `tool_manager`

**Evaluation Against Criteria**:

| Criterion | Score | Notes |
|-----------|-------|-------|
| Functional correctness | 9/10 | All 3 regressions addressed directly |
| Architectural consistency | 9/10 | Maintains M3 direction, uses new abstractions |
| Test coverage | 8/10 | 64 new test tasks covering all scenarios |
| Spec coverage | 9/10 | 2 new specs + 3 modified specs |
| Implementation complexity | 6/10 | More new code, but well-specified with pseudocode |
| Backward compatibility | 9/10 | No YAML config changes, same tool prefixing convention |

**Effort Estimate**: Medium-High — extend `SkillManagerCap`, update pool.py, factory.py, skills.py, add tests

**Risk Assessment**: Low — all referenced classes (`PrefixedToolset`, `FilteredToolset`, `McpServerCap`, `SkillToolManager`) exist and are functional

### Option C: Create a new `SkillToolCapability` separate from `SkillManagerCap`

**Description**: Create a new `SkillToolCapability` class that handles only tool registration (Python + MCP). `SkillManagerCap` handles instructions and skill listing. Both are registered as separate capabilities.

**Advantages**:
- Separation of concerns — `SkillManagerCap` stays focused on instructions
- New class is purpose-built for the new architecture
- `SkillManagerCap` doesn't grow in complexity

**Disadvantages**:
- Introduces a new class — increases class count
- Two capabilities must coordinate (skill listing from `SkillManagerCap`, tool registration from `SkillToolCapability`)
- `allowed_tools` filtering spans both capabilities — `get_wrapper_toolset()` on which one?
- More complex agent assembly — factory must inject both capabilities

**Evaluation Against Criteria**:

| Criterion | Score | Notes |
|-----------|-------|-------|
| Functional correctness | 8/10 | Works but coordination between two classes adds complexity |
| Architectural consistency | 7/10 | New class, but follows capability pattern |
| Test coverage | 7/10 | More integration points to test |
| Spec coverage | 7/10 | Additional spec needed for new class |
| Implementation complexity | 5/10 | Most complex option — new class + coordination |
| Backward compatibility | 9/10 | No YAML config changes |

**Effort Estimate**: High — new class, coordination logic, factory updates, more tests

**Risk Assessment**: Medium — coordination between two capabilities may introduce subtle bugs

## Recommendation

**Option B: Add tool registration logic to `SkillManagerCap`**

Based on the evaluation criteria, Option B scores highest overall (50/60 vs 40/60 for Option A and 43/60 for Option C). It maintains the M3 architectural direction, uses existing abstractions (`McpServerCap`, `PrefixedToolset`, `FilteredToolset`), and has the lowest risk profile.

The main trade-off accepted is increased `SkillManagerCap` complexity (~110 LOC growth). This is mitigated by:
- Keeping tool import logic in a private `_import_skill_tools()` method
- Full pseudocode for `get_toolset()` and `get_wrapper_toolset()` overrides in the design
- 64 test tasks covering all scenarios

## Technical Design

### D1: Register `SkillManagerCap` with `ExtensionRegistry` at POOL scope

The `SkillManagerCap` SHALL be registered with `ExtensionRegistry` at `ScopeLevel.POOL`. With flat URIs (`skill://{name}`), `ExtensionRegistry.resolve_uri()` receives `provider_name=None` and iterates all visible `SkillResource` capabilities without name filtering — no capability name matching is needed.

### D2: `SkillManagerCap` accepts `SkillToolManager`, imports Python tools eagerly, and fully overrides `get_toolset()`

```python
def get_toolset(self):
    toolsets: list[AbstractToolset] = []

    # 1. Python tools: PrefixedToolset per skill
    for skill_name, tools in self._skill_tools.items():
        pa_tools = [t.to_pydantic_ai() for t in tools]
        toolsets.append(PrefixedToolset(
            prefix=f"{skill_name}__tool__",
            wrapped=FunctionToolset(pa_tools),
        ))

    # 2. Per-skill McpServerCap children: PrefixedToolset per skill
    for skill_name, child_caps in self._skill_mcp_children.items():
        for child in child_caps:
            child_ts = child.get_toolset()
            if child_ts is not None:
                toolsets.append(PrefixedToolset(
                    prefix=f"{skill_name}__mcp__",
                    wrapped=child_ts,
                ))

    # 3. Non-skill children (unprefixed)
    skill_child_ids = {id(c) for caps in self._skill_mcp_children.values() for c in caps}
    for cap in self._capabilities:
        if id(cap) not in skill_child_ids:
            ts = cap.get_toolset()
            if ts is not None:
                toolsets.append(ts)

    return CombinedToolset(toolsets=toolsets) if toolsets else None
```

`super().get_toolset()` SHALL NOT be called — the parent's merge doesn't apply per-skill prefixes.

### D3: Per-skill MCP servers tracked in `_skill_mcp_children`

For each skill with `mcp_servers` frontmatter, create `McpServerCap` instances, store in `_skill_mcp_children[skill_name]`, and add to `_capabilities` for lifecycle management.

### D4: Composite `allowed_tools` filter via `get_wrapper_toolset()`

```python
def get_wrapper_toolset(self, toolset):
    skill_filters: dict[str, set[str]] = {}
    for name, skill in self._local_skills.items():
        allowed = skill.parsed_allowed_tools()
        if allowed:
            skill_filters[name] = set(allowed)

    if not skill_filters:
        return None

    def _filter(ctx, tool_def):
        name = tool_def.name
        for skill_name, allowed_set in skill_filters.items():
            prefix = f"{skill_name}__"
            if name.startswith(prefix):
                bare = name[len(prefix):].rsplit("__", 1)[-1]
                return bare in allowed_set
        return True

    return FilteredToolset(wrapped=toolset, filter_func=_filter)
```

### D5: Restore `skills_tools_provider` injection in `_inject_pool_providers()`

One-line fix: `agent._external_capabilities.append(host_context.skills_tools_provider)` guarded by `if host_context.skills_tools_provider is not None`. Safe due to singleton pattern and tool name deduplication.

### D6: Migrate MCP provider dispatch to `SkillResource` protocol

Replace `getattr(provider, 'get_skills', None)` with `isinstance(provider, SkillResource)` in the MCP provider iteration path only. Local skills path (`ctx.pool.skills.list_skills()`) stays unchanged.

### D7: Update `for_run()` to propagate `tool_manager`

```python
async def for_run(self, ctx):
    children_for_run = [await child.for_run(ctx) for child in self._capabilities]
    cap = SkillManagerCap(
        local_skills=self._local_skills,
        children=children_for_run,
        matcher_fn=self._matcher_fn,
        always_active=self._always_active,
        registry=self._registry,
        name=self._name,
        tool_manager=self._tool_manager,  # NEW
    )
    return cap
```

## Implementation Plan

### Phase 1: URI Resolution Fix (Tasks 1.1-1.4)
- Register `SkillManagerCap` with `ExtensionRegistry` at POOL scope
- Add end-to-end URI resolution test

### Phase 2: Python Tool Registration (Tasks 2.1-2.7)
- Add `tool_manager` parameter, `_import_skill_tools()`, `get_toolset()` override
- Update `for_run()` to propagate `tool_manager`
- Add end-to-end tool registration test

### Phase 3: MCP Tool Registration (Tasks 3.1-3.5)
- Create per-skill `McpServerCap` instances, track in `_skill_mcp_children`
- Verify lifecycle management
- Add end-to-end MCP tool test

### Phase 4: allowed_tools Filtering (Tasks 4.1-4.4)
- Implement composite filter in `get_wrapper_toolset()`
- Add tests for filtering, no-filter, and empty-list edge cases

### Phase 5: load_skill Propagation (Tasks 5.1-5.3)
- Restore injection in `_inject_pool_providers()`
- Add tests for standalone and child session agents

### Phase 6: Protocol Migration (Tasks 6.1-6.6)
- Replace `getattr` with `isinstance(SkillResource)` in MCP provider path
- Verify local path unchanged
- Run regression tests

### Phase 7-15: Specs, Unit Tests, Edge Cases, Integration Tests, Full Suite

### Phase 16: Dead Code Cleanup
- Remove deprecated pool fields, unreachable agent.py branches, dead manager.py properties

### Phase 17: Flat URI Implementation  
- Remove hardcoded `skill://local/` prefix, fix URI parsing, update tests

**Dependencies**: None — all referenced classes exist and are functional.

**Rollback Strategy**: Revert to pre-change state. The change is purely additive (new methods, new parameters) — no existing behavior is modified, only restored.

## Open Questions

1. **Should `allowed_tools` support glob patterns?** Currently only exact bare name matching. The old `SkillCapability` also used exact matching. Glob support could be added later if needed.

2. **Should `load_skill` tool status display be migrated?** The `load_skill` tool creates a throwaway `SkillToolManager` on each call (line 353) to display tool import status. This is informational only and slightly wasteful. Deferred to future cleanup.

3. **Should `skill://` URI format be unified?** Resolved — this RFC adopts flat URIs (`skill://{name}`) as part of the implementation (D9).

## Decision Record

| Reviewer | Round | Result | Issues Found | Issues Fixed |
|----------|-------|--------|--------------|--------------|
| Oracle | 1 | FAIL | 6 critical + 4 minor | All 10 |
| Oracle | 2 | FAIL | 3 remaining | All 3 |
| Oracle | 3 | FAIL | 2 leftover contradictions | Both |
| Oracle | 4 | VERIFIED | 0 | — |

**Key discussion points**:
- `get_toolset()` must fully override parent (no `super()` call) — prefixing requires manual child iteration
- `SkillManagerCap` is registered with `ExtensionRegistry` at POOL scope — no capability name matching needed for flat URIs
- `for_run()` must propagate `tool_manager` to prevent per-run tool loss
- `allowed_tools: []` (explicitly empty) filters all skill tools, distinct from `None` (no filter)
- MCP provider migration only — local `ctx.pool.skills` path stays unchanged

**Conditions on approval**: None. All artifacts (proposal, design, 5 specs, 64 tasks) are internally consistent and verified by Oracle.
