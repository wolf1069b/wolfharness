---
rfc_id: RFC-0058
title: "Unified MCP Server Registration: Eliminating the Dual-Object Problem for Tool and Resource Access"
status: DRAFT
author: pinjun.mo
reviewers:
  - name: yuchen.liu
    status: pending
created: 2026-08-14
last_updated: 2026-08-14 (v4: RFC-0051→0052→0058 decision lineage added to Historical Context; open question #7 — top-level vs skill-MCP prefix convention divergence)
decision_date:
related_rfcs:
  - RFC-0051 (Extension Source Architecture — Resource Protocols and Client Injection)
  - RFC-0052 (Restore Skill Capabilities — SkillManagerCap children wiring)
related_specs:
  - docs/specs/mcp-resource-technical-report.md (MCP Resource consumption architecture)
---

# RFC-0058: Unified MCP Server Registration — Eliminating the Dual-Object Problem for Tool and Resource Access

## Table of Contents

- [Overview](#overview)
- [Background & Context](#background--context)
- [Problem Statement](#problem-statement)
- [Goals & Non-Goals](#goals--non-goals)
- [Evaluation Criteria](#evaluation-criteria)
- [Options Analysis](#options-analysis)
- [Recommendation](#recommendation)
- [Technical Design](#technical-design)
- [Security Considerations](#security-considerations)
- [Implementation Plan](#implementation-plan)
- [Open Questions](#open-questions)
- [Decision Record](#decision-record)
- [References](#references)

---

## Overview

AgentPool currently maintains two independent object representations for each top-level MCP server: a `McpServerCap` instance (created by `MCPManager.setup_server()`) and a pydantic-ai `MCP` capability (built on-demand by `MCPManager.get_capabilities()`). The former handles resource/skill protocol access; the latter handles tool exposure. This dual-object design causes `@` mention resource access to fail for top-level MCP servers, creates state synchronization risk, and complicates the tool injection path.

This RFC proposes **unifying on `McpServerCap` as the single object representation** for each MCP server — responsible for both tool exposure (via `get_toolset()`) and resource access (via `ResourceAccess` protocol). The `MCPManager.get_capabilities()` path for top-level servers is retired in favor of direct `McpServerCap` injection into agent tool capabilities. `SkillManagerCap` gains `ResourceAccess` delegation for skill-level MCP children.

**Expected outcome**: `@` mention works for all MCP servers, tool and resource access share a single object per server, and the `MCPManager.get_capabilities()` legacy path is removed for top-level servers.

---

## Background & Context

### Current State

The MCP server lifecycle in AgentPool involves two parallel paths:

**Path A — McpServerCap (resource/skill path)**:
- `pool.py:175` creates `MCPManager(servers=top_level_servers)`
- `MCPManager.__aenter__()` (manager.py:320) calls `setup_server()` per server
- `setup_server()` (manager.py:420) creates `McpServerCap(config, client=pre_created_client)`, appends to `self.providers`
- `_rebuild_skill_capabilities()` (pool.py:656) filters providers by `isinstance(provider, SkillResource)`, stuffs them into `SkillManagerCap.children`
- `SkillManagerCap` registers at POOL scope (pool.py:686)
- `SkillManagerCap.get_toolset()` (skill_manager_cap.py:448) **fully overrides** `CombinedToolsetCapability.get_toolset()`, does NOT call `super()`, does NOT iterate `self._children` for toolset purposes — children serve only `SkillResource`/`CommandResource` protocol queries

**Path B — pydantic-ai MCP capability (tool path)**:
- `NativeAgent.__init__()` (agent.py:358) calls `self.mcp.get_aggregating_provider()` → `CombinedToolsetCapability` over ACP-only providers → appended to `_external_capabilities`
- `get_agentlet()` (agent.py:1123) calls `self.mcp.get_capabilities(session_id)` → builds fresh pydantic-ai `MCP` capabilities from config snapshot → `tool_capabilities.extend(mcp_capabilities)`
- `MCPManager.get_capabilities()` (manager.py:649) reads `McpConfigSnapshot`, creates `MCPToolset` per server via `GlobalConnectionPool.get_transport()`, wraps in `MCP(url, local=toolset)`

**Path C — ACP aggregating provider (ACP-only tool path)**:
- `get_aggregating_provider()` (manager.py:637) filters `isinstance(p.config, AcpMCPServerConfig)`, wraps in `CombinedToolsetCapability`
- Injected via `_inject_pool_providers()` (factory.py:883) into `_external_capabilities` for child sessions

### Historical Context

**Decision lineage: RFC-0051 designed unification → RFC-0052 split it → RFC-0058 restores it.**

- **RFC-0051 (Extension Source Architecture, 2026-07-11)** designed `McpServerCap` (then `McpResource`) to implement `ResourceAccess` and be **independently registered in `ExtensionRegistry` at POOL scope** — its lifecycle diagram (`register(McpServerCap(config), POOL)`, §Lifecycle Management) shows it as a standalone capability, not a child of `SkillManagerCap`. Tools were designed to flow through `AgentFactory.compile() → McpServerCap.get_toolset()`. The intent was **one server = one capability = tools + resources together**.
- **RFC-0052 (Restore Skill Capabilities, 2026-07-12)** split this design to fix three M3 skill regressions. Its **Option B split registration along the resource/tool boundary**: (a) top-level McpServerCap instances implementing `SkillResource` were stuffed into `SkillManagerCap.children` (pool.py `_rebuild_skill_capabilities()`) to restore `skill://` URI resolution for remote skills — silently removing the independent POOL-scope registration that RFC-0051 specified; (b) the **tool surface was left on the untouched `MCPManager.get_capabilities()` path**. RFC-0052's D2 decision (fully-rewritten `SkillManagerCap.get_toolset()`) explicitly declared non-skill children "unprefixed" (get_toolset() case 3): children in `_capabilities` other than per-skill MCP were appended to the combined toolset **without** a `PrefixedToolset` wrapper (implemented in commit `6f07fd36f`, `src/agentpool/capabilities/skill_manager_cap.py`). The implicit assumption was that non-skill children are independent, mutually-distinct capabilities that never collide. **That assumption breaks when multiple top-level MCP servers are configured** — their McpServerCap instances are all stuffed into `children`, and two servers exposing the same tool name (e.g., both have `search`) collide silently under the unprefixed path. RFC-0058's per-server prefixing corrects this false assumption.
- The `MCPManager.get_capabilities()` path predates `McpServerCap` and was **not retired when `McpServerCap` was introduced** — RFC-0051's implementation (commit `7d7cf9560`) assumed `get_capabilities()` would be narrowed to session/skill scope, but the top-level call in `get_agentlet()` was left in place. As a result, the McpServerCap tool path designed in RFC-0051 is effectively dead code today: the `SkillManagerCap.get_toolset()` override does not iterate `children` for tools, and tools for top-level servers come exclusively from `get_capabilities()`. The dual-object problem is the accumulated outcome of these two decisions, not a single deliberate split.
- **RFC-0058 restores RFC-0051's design intent**: independently register McpServerCap at POOL scope (Phase 2), migrate tool exposure back to `McpServerCap.get_toolset()` (Phase 3), and narrow `get_capabilities()` to session/skill scope. This is a restoration, not a novel architecture.

### Glossary

| Term | Definition |
|------|------------|
| McpServerCap | AgentPool capability wrapping a single MCP server connection; implements `ResourceAccess`, `SkillResource`, `CommandResource`, `ToolAccess`, `ChangeObservable` |
| MCPManager | Manages lifecycle of top-level and agent-level MCP servers; creates McpServerCap instances; provides `get_capabilities()` and `get_aggregating_provider()` |
| ExtensionRegistry | 4-level scope registry (POOL > AGENT > SESSION > TURN) for capability lookup |
| ResourceAccess | Protocol providing `list_resources()`, `read_resource()`, `resource_exists()` |
| SkillManagerCap | Capability managing local skills, per-skill MCP, and remote skill discovery; inherits `CombinedToolsetCapability` |
| Dual-object problem | Same MCP server represented by two objects: McpServerCap (resources) and pydantic-ai MCP (tools) |
| `@` mention | Editor-driven resource injection via `GET /experimental/resource` endpoint → `list_mcp_resources()` → `registry.get_resource_access()` |
| ResourceCapability | Agent-facing capability exposing 5 resource tools (`list_resources`, `read_resource`, etc.) for model-initiated resource access |

---

## Problem Statement

### The Problem

1. **`@` mention cannot access top-level MCP resources**: `list_mcp_resources()` (agent_routes.py:492) calls `registry.get_resource_access(scope)`, which returns capabilities implementing `ResourceAccess`. `McpServerCap` implements `ResourceAccess`, but it is registered as a child of `SkillManagerCap`, which does NOT implement `ResourceAccess`. The registry only sees directly-registered capabilities, not their children. Therefore `get_resource_access()` returns an empty list for top-level MCP servers.

2. **Dual-object state divergence**: The same MCP server has two object representations — `McpServerCap` with a pre-created `MCPClient` (in `MCPManager.providers`) and pydantic-ai `MCP` with a separately created `MCPToolset` (in `get_capabilities()`). Each maintains its own connection. Connection state, caching, and error handling can diverge.

3. **Unnecessary connection multiplicity**: For a single MCP server, two TCP connections may be established — one by `MCPClient` (in `setup_server()`) and one by `MCPToolset` (in `get_capabilities()` via `GlobalConnectionPool`). This doubles resource consumption and complicates connection lifecycle management.

4. **Tool name collisions across MCP servers**: Neither the `McpServerCap.get_toolset()` path nor the `MCPManager.get_capabilities()` path applies a server-level prefix to tool names. `MCPClient.convert_tool()` (client.py:557) sets `tool_callable.__name__ = tool.name` — the raw MCP tool name with no namespace. pydantic-ai's `MCPToolset` provides a `tool_name_conflict_hint` suggesting `.prefixed("...")` but does not auto-prefix. `load_mcp_toolsets()` (mcp.py:1754) demonstrates the intended pattern — `toolset.prefixed(name)` — but `get_capabilities()` does not follow it. If two MCP servers expose a tool with the same name (e.g., both have `search`), pydantic-ai's `CombinedToolset` will silently overwrite one with the other. Skill-level MCP already solves this via `PrefixedToolset(prefix=f"{skill_name}__mcp__")` (skill_manager_cap.py:548), but top-level MCP has no equivalent.

### Evidence

- `GET /experimental/resource` returns `200 OK` with empty `{}` body when only top-level MCP servers are configured (observed in `~/Library/Logs/agentpool/opencode.log`)
- `curl` to MCP server at `localhost:8002/mcp` confirms 5 resources are exposed via `resources/list` protocol
- Code trace confirms `SkillManagerCap` does not implement `ResourceAccess` (skill_manager_cap.py class declaration), and `get_resource_access()` (extension_registry.py:389) filters by `isinstance(cap, ResourceAccess)`
- `SkillManagerCap.get_toolset()` (skill_manager_cap.py:448-554) does not iterate `self._children` for toolset — confirmed by full method read

### Impact of Inaction

- **Cost**: `@` mention feature is broken for all top-level MCP servers, requiring users to manually paste resource content or use model-initiated tools (which may not always be appropriate)
- **Risk**: Dual connections increase the chance of connection exhaustion, stale state, and inconsistent error handling
- **Opportunity**: Without unification, every new MCP-related feature (e.g., resource subscription, change notification) must be implemented twice — once for McpServerCap, once for the pydantic-ai MCP path

---

## Goals & Non-Goals

### Goals (In Scope)

1. `@` mention works for top-level MCP servers registered at POOL scope
2. `@` mention works for skill-level MCP servers via `SkillManagerCap` ResourceAccess delegation
3. `McpServerCap` is the single object representation per MCP server — responsible for both tool exposure and resource access
4. `MCPManager.get_capabilities()` is retired for top-level servers (Path B is eliminated)
5. No duplicate tool exposure for any MCP server
6. **Tool names are namespaced per server — no silent collisions when multiple MCP servers expose same-named tools**
7. Model-initiated resource access via `ResourceCapability` continues to work unchanged
8. ACP transport MCP servers continue to work via the aggregating provider path (Path C)

### Non-Goals (Out of Scope)

1. MCP server-side resource exposure (AgentPool as MCP server) — documented as a gap in the technical report (§5.9), not addressed here
2. URI conflict detection across multiple MCP servers — deferred to a follow-up; custom scheme convention is assumed
3. Refactoring of session-scoped MCP config snapshot mechanism — the `McpConfigSnapshot` continues to serve session-scoped and skill-scoped configs
4. `ResourceCapability` redesign — it is already a registry consumer and requires no changes
5. **Tool Registry / Tool Selection / semantic routing** — the "1000 tools, show the model 10-30" problem (tool retrieval, `search_tools` tool, intent-based routing between same-named tools across servers). This RFC only solves registration and flat namespacing for the current config-driven scale (single-digit servers). Tool selection and retrieval are a separate layer to be designed in a follow-up RFC. OpenCode's direction (server namespace + permission/agent filtering + Code Mode for context control) is acknowledged as the target architecture but not implemented here.

### Success Criteria

- [ ] `GET /experimental/resource` returns resources from all configured top-level MCP servers
- [ ] `GET /experimental/resource` returns resources from skill-level MCP servers
- [ ] Model can call `list_resources` / `read_resource` tools and receive results from all MCP servers
- [ ] Model can call MCP tools (e.g., `search_database`) from top-level MCP servers
- [ ] No duplicate tools appear in the agent's tool list for any MCP server
- [ ] Tools from different MCP servers with the same raw name are distinguishable (no silent overwrite)
- [ ] Only one TCP connection per MCP server (verified by connection count)
- [ ] ACP transport MCP servers continue to expose tools via aggregating provider

---

## Evaluation Criteria

| Criterion | Weight | Description | Minimum Threshold |
|-----------|--------|-------------|-------------------|
| Resource access correctness | High | `@` mention and `ResourceCapability` can discover and read resources from all MCP server types | 100% of configured servers |
| Tool exposure correctness | High | All MCP tools reach the agent's `tool_capabilities` without duplication | 0 duplicates |
| Tool namespacing | High | Tools from different servers with same raw name are distinguishable | 0 silent collisions |
| Architectural simplicity | High | Number of object representations per MCP server | 1 |
| Backward compatibility | Medium | Existing YAML configs work without modification | All existing configs |
| Implementation effort | Medium | Lines of code changed, files touched, test updates needed | < 500 LOC changed |
| ACP transport support | Medium | ACP MCP servers continue to work through their existing path | No regression |
| Connection efficiency | Low | TCP connections per MCP server | 1 |

---

## Options Analysis

### Option 1: Unified McpServerCap Registration (Recommended)

**Description**

Register each top-level `McpServerCap` independently at POOL scope in `ExtensionRegistry`. Replace `MCPManager.get_capabilities()` call in `get_agentlet()` with direct injection of `McpServerCap` instances into `tool_capabilities`. Add `ResourceAccess` delegation to `SkillManagerCap` for skill-level MCP children.

Key changes:
- `pool.py:_rebuild_skill_capabilities()`: Stop stuffing top-level McpServerCap into `SkillManagerCap.children`. Register each at POOL scope independently.
- `agent.py:get_agentlet()`: Replace `mcp_capabilities = await self.mcp.get_capabilities(session_id)` with direct injection of `McpServerCap` instances into `tool_capabilities`. Wrap each in a `PrefixedToolset` using the server's `display_name` as prefix.
- `mcp_server_cap.py:get_toolset()`: Wrap the returned `CombinedToolset` in a `PrefixedToolset` using the server's `display_name` as prefix (via a new `tool_prefix` property), so tools are automatically namespaced per server.
- `mcp_server/manager.py:setup_server()`: De-duplicate `display_name` across servers (append `_2`, `_3`, ... on collision) so each McpServerCap's tool prefix is unique even when two configured servers share the same name.
- `skill_manager_cap.py`: Add `ResourceAccess` protocol implementation that delegates to `_skill_mcp_children`.
- `agent_routes.py:list_mcp_resources()`: No change needed — `get_resource_access()` will now find POOL-scoped McpServerCap instances directly.

**Advantages**

- Single object per MCP server — `McpServerCap` handles both tools (`get_toolset()`) and resources (`ResourceAccess`)
- Tool namespacing via `PrefixedToolset` prevents silent collisions when multiple servers expose same-named tools
- `@` mention works for top-level MCP servers without any endpoint changes
- Eliminates the `MCPManager.get_capabilities()` path for top-level servers, removing ~180 lines of complex snapshot/transport/cache logic from the hot path
- Single TCP connection per server (the `MCPClient` created in `setup_server()`)
- Consistent with RFC-0051's original design intent — McpServerCap as independently-registered capability
- Namespacing pattern aligns with pydantic-ai's own `load_mcp_toolsets()` example (mcp.py:1754: `toolset.prefixed(name)`)

**Disadvantages**

- `MCPManager.get_capabilities()` must remain for session-scoped and skill-scoped MCP configs (it cannot be fully removed)
- `get_agentlet()` must handle two tool injection paths: McpServerCap instances (top-level) and `get_capabilities()` (session-scoped) — though the latter is simplified
- Session-scoped MCP config isolation currently relies on `McpConfigSnapshot` + `get_capabilities()` partition; retiring it for top-level configs means top-level McpServerCap instances are shared across all sessions (which is already the case for `MCPClient` connections)
- `SkillManagerCap` gains `ResourceAccess` implementation, slightly increasing its responsibility surface

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Resource access correctness | Excellent | POOL-scope registration makes McpServerCap directly discoverable by `get_resource_access()` |
| Tool exposure correctness | Excellent | `get_toolset()` returns lazy ToolsetFunc wrapped in PrefixedToolset; no duplication risk since `get_capabilities()` path is removed for top-level |
| Tool namespacing | Excellent | PrefixedToolset with server display_name; follows pydantic-ai's own convention |
| Architectural simplicity | Excellent | Single object per server; RFC-0051 alignment |
| Backward compatibility | Good | YAML config unchanged; `get_capabilities()` retained for session/skill scope |
| Implementation effort | Medium | ~300 LOC across pool.py, agent.py, skill_manager_cap.py; test updates needed |
| ACP transport support | Excellent | ACP aggregating provider path (Path C) is unaffected |
| Connection efficiency | Excellent | Single MCPClient per server, no MCPToolset duplicate |

**Effort Estimate**

- Complexity: Medium
- Resources: 1 engineer, 2-3 days
- Dependencies: None (self-contained refactor)

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Session-scoped tool isolation breaks | Low | Medium | Top-level servers were always shared; session isolation only applies to session/skill configs which still use `get_capabilities()` |
| `get_toolset()` lazy client fails | Low | High | `setup_server()` pre-creates `MCPClient`, so `_ensure_client()` returns immediately |
| `for_run()` not overridden on McpServerCap | Low | Low | McpServerCap inherits default `for_run()` → returns `self`; connection is server-scoped, sharing across runs is intended |

---

### Option 2: SkillManagerCap ResourceAccess Proxy Only

**Description**

Keep top-level McpServerCap instances inside `SkillManagerCap.children` (no change to registration). Add `ResourceAccess` implementation to `SkillManagerCap` that delegates to all children implementing `ResourceAccess`. Keep `MCPManager.get_capabilities()` for tool exposure unchanged.

Key changes:
- `skill_manager_cap.py`: Add `ResourceAccess` protocol implementation delegating to `self._children`.
- No changes to `pool.py`, `agent.py`, or `manager.py`.

**Advantages**

- Minimal code change — only `skill_manager_cap.py` is modified
- No risk to tool exposure path — `get_capabilities()` continues as-is
- `SkillManagerCap` already delegates `SkillResource` and `CommandResource`; adding `ResourceAccess` follows the same pattern

**Disadvantages**

- Dual-object problem persists — `McpServerCap` for resources, pydantic-ai `MCP` for tools
- Two TCP connections per server remain
- `SkillManagerCap.children` semantically wrong — top-level MCP servers are not skills
- `get_capabilities()` complexity remains in the hot path
- Future features must still be implemented in two places

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Resource access correctness | Good | `@` mention works via delegation, but only for children that implement `ResourceAccess` |
| Tool exposure correctness | Good | No change to existing path; but dual-object problem means tools and resources may diverge |
| Tool namespacing | Poor | `get_capabilities()` does not prefix tools; collision risk remains |
| Architectural simplicity | Poor | Dual-object problem remains; SkillManagerCap semantically overloaded |
| Backward compatibility | Excellent | No changes to any other file |
| Implementation effort | Low | ~80 LOC in skill_manager_cap.py only |
| ACP transport support | Excellent | Unaffected |
| Connection efficiency | Poor | Two connections per server persist |

**Effort Estimate**

- Complexity: Low
- Resources: 1 engineer, 0.5 days
- Dependencies: None

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `SkillManagerCap` becomes too broad | Medium | Low | Acceptable trade-off for minimal-change approach |
| Future features require dual implementation | High | Medium | Document as known tech debt |

---

### Option 3: Hybrid — Independent Registration + Retain get_capabilities()

**Description**

Register top-level McpServerCap independently at POOL scope (like Option 1) for resource access, but retain `MCPManager.get_capabilities()` for tool exposure (like Option 2). Add `ResourceAccess` delegation to `SkillManagerCap` for skill-level MCP.

Key changes:
- `pool.py`: Register McpServerCap at POOL scope, AND keep them in `SkillManagerCap.children`.
- `agent.py`: No change to `get_capabilities()` call.
- `skill_manager_cap.py`: Add `ResourceAccess` delegation to children.
- `agent_routes.py`: Deduplicate `get_resource_access()` results (McpServerCap appears both at POOL scope and as SkillManagerCap child).

**Advantages**

- Resource access works immediately via POOL-scope registration
- Tool exposure path is untouched — zero risk of tool regression
- Gradual migration path — can retire `get_capabilities()` later

**Disadvantages**

- McpServerCap registered twice (POOL scope + SkillManagerCap child) — deduplication needed
- Dual-object problem persists
- `SkillManagerCap.children` still semantically wrong
- Most complex of the three options — adds registration without removing the old path

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Resource access correctness | Good | Works but requires deduplication in `get_resource_access()` consumers |
| Tool exposure correctness | Good | No change to existing path |
| Tool namespacing | Poor | `get_capabilities()` path unchanged; collision risk remains |
| Architectural simplicity | Poor | Adds a path without removing the old one; highest complexity |
| Backward compatibility | Excellent | All existing paths preserved |
| Implementation effort | Medium | ~200 LOC but with deduplication complexity |
| ACP transport support | Excellent | Unaffected |
| Connection efficiency | Poor | Two connections per server persist |

**Effort Estimate**

- Complexity: Medium
- Resources: 1 engineer, 1-2 days
- Dependencies: None

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Duplicate resources in `@` mention list | High | Low | Deduplicate by URI in `list_mcp_resources()` |
| Confusion from dual registration | Medium | Low | Document as transitional state |

---

### Options Comparison Summary

| Criterion | Option 1: Unified | Option 2: Proxy Only | Option 3: Hybrid |
|-----------|-------------------|---------------------|-----------------|
| Resource access correctness | Excellent | Good | Good |
| Tool exposure correctness | Excellent | Good | Good |
| Tool namespacing | Excellent | Poor | Poor |
| Architectural simplicity | Excellent | Poor | Poor |
| Backward compatibility | Good | Excellent | Excellent |
| Implementation effort | Medium | Low | Medium |
| ACP transport support | Excellent | Excellent | Excellent |
| Connection efficiency | Excellent | Poor | Poor |
| **Overall** | **Best** | Acceptable (short-term) | Not recommended |

---

## Recommendation

### Recommended Option

**Option 1: Unified McpServerCap Registration**

### Justification

Based on the evaluation criteria, Option 1 scores highest on architectural simplicity (single object per server, RFC-0051 alignment) and connection efficiency (single TCP connection). The implementation effort is moderate (~300 LOC) and the risk profile is manageable — the primary risk (session-scoped tool isolation) does not apply because top-level servers were always shared across sessions via `MCPClient`.

Option 2 is viable as a short-term stopgap if implementation time is constrained, but it leaves the dual-object problem unresolved and accumulates tech debt. Option 3 adds complexity without removing the old path, making it the worst long-term option.

### Accepted Trade-offs

1. **`MCPManager.get_capabilities()` retained for session/skill scope**: The full retirement of `get_capabilities()` is not feasible in this RFC because session-scoped and skill-scoped MCP configs rely on the snapshot mechanism. This is acceptable — the dual-object problem only affects top-level servers, which are the common case.
2. **`SkillManagerCap` gains `ResourceAccess` responsibility**: This slightly broadens `SkillManagerCap`'s surface area, but the delegation pattern is identical to existing `SkillResource` and `CommandResource` delegation — no new architectural pattern is introduced.
3. **Top-level McpServerCap shared across all sessions**: This is already the behavior for `MCPClient` connections (created once in `setup_server()`). The change makes tool exposure consistent with this existing sharing semantics.

### Conditions

- ACP transport MCP servers must continue to work via the aggregating provider path without regression
- Existing tests for `MCPManager.get_capabilities()` must continue to pass (they exercise session-scoped configs)
- The `ResourceCapability` (model-initiated resource access) must work unchanged

---

## Technical Design

### Architecture Overview

```
                              ┌─────────────────────────────┐
                              │     ExtensionRegistry        │
                              │     (POOL scope)             │
                              │                              │
                              │  ┌────────────────────┐     │
                              │  │ SkillManagerCap    │     │
                              │  │  ├─ SkillResource  │     │
                              │  │  ├─ CommandResource│     │
                              │  │  ├─ ResourceAccess │ ← NEW delegation
                              │  │  │   to _skill_mcp │     │
                              │  │  │   _children     │     │
                              │  │  └─ get_toolset()  │     │
                              │  │    (builtin +     │     │
                              │  │     skill tools +  │     │
                              │  │     skill MCP)     │     │
                              │  └────────────────────┘     │
                              │                              │
                              │  ┌────────────────────┐     │
                              │  │ McpServerCap A     │ ← NEW independent
                              │  │  ├─ ToolAccess     │   registration
                              │  │  ├─ ResourceAccess │     │
                              │  │  ├─ SkillResource  │     │
                              │  │  ├─ CommandResource│     │
                              │  │  └─ get_toolset()  │     │
                              │  └────────────────────┘     │
                              │  ┌────────────────────┐     │
                              │  │ McpServerCap B     │ ← NEW independent
                              │  │  └─ ...            │   registration
                              │  └────────────────────┘     │
                              └─────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
              ┌─────▼─────┐     ┌──────▼──────┐    ┌──────▼──────┐
              │ @ mention │     │ Resource    │    │ get_agentlet│
              │ endpoint  │     │ Capability  │    │ tool_caps   │
              │           │     │ (model tools)│   │             │
              └───────────┘     └─────────────┘    └─────────────┘
              get_resource_     list_resources     pool.mcp.providers
              access(scope)     read_resource      → McpServerCap
                                                  .get_toolset()
```

### Key Changes

#### 1. pool.py — `_rebuild_skill_capabilities()`

**Before**:
```python
mcp_children = [
    provider for provider in self.mcp.providers
    if isinstance(provider, SkillResource)
]
cap = SkillManagerCap(local_skills=..., children=mcp_children, ...)
self._extension_registry.register(cap, pool_scope)
```

**After**:
```python
# Register each top-level McpServerCap independently at POOL scope
for provider in self.mcp.providers:
    self._extension_registry.register(provider, pool_scope)

# SkillManagerCap only manages local skills and per-skill MCP
cap = SkillManagerCap(local_skills=..., children=[], ...)
self._extension_registry.register(cap, pool_scope)
```

Note: `SkillManagerCap` still needs `SkillResource` access to top-level MCP providers for remote skill listing. This is handled via `ExtensionRegistry.get_skill_resources(scope)` which returns all POOL-scoped `SkillResource` implementations — including the independently registered McpServerCap instances. The `SkillURIResolver` registration at pool.py:610 is unchanged.

**Dead code elimination**: With top-level McpServerCap no longer in `SkillManagerCap._capabilities`, RFC-0052's D2 case 3 ("non-skill children unprefixed") becomes dead code — `_capabilities` now only holds per-skill MCP children (already handled by case 2). The unprefixed branch in `skill_manager_cap.py:get_toolset()` SHOULD be removed to prevent future confusion and to ensure no capability silently escapes namespacing.

#### 2. agent.py — `get_agentlet()`

**Before**:
```python
# 4. MCP servers
mcp_capabilities = await self.mcp.get_capabilities(
    session_id=run_ctx.session_id if run_ctx else None
)
tool_capabilities.extend(mcp_capabilities)
```

**After**:
```python
# 4. MCP servers — top-level: inject McpServerCap directly
pool = self._agent_pool
if pool is not None:
    # Non-ACP top-level providers: inject as capabilities (tools via get_toolset())
    for provider in pool.mcp.providers:
        if not isinstance(provider.config, AcpMCPServerConfig):
            tool_capabilities.append(provider)
    # Session-scoped configs (session + skill): still use get_capabilities()
    session_mcp_caps = await self.mcp.get_capabilities(
        session_id=run_ctx.session_id if run_ctx else None,
        exclude_global=True,  # ← NEW param to skip pool+agent configs
    )
    tool_capabilities.extend(session_mcp_caps)
```

Note: `get_capabilities()` gains an `exclude_global` parameter to skip pool-level and agent-level configs (already handled by McpServerCap injection). It continues processing session-scoped and skill-scoped configs. ACP providers continue through the aggregating provider path (Path C, unchanged).

#### 3. mcp_server_cap.py — `get_toolset()` with PrefixedToolset

**Before**:
```python
def get_toolset(self) -> Any:
    async def _build_toolset(ctx):
        client = await self._ensure_client()
        tools = await client.list_tools()
        if not tools:
            return None
        converted = [client.convert_tool(t) for t in tools]
        pydantic_tools = [wrap_tool_for_pydantic_ai(tool) for tool in converted]
        toolsets = [FunctionToolset[Any]([tool]) for tool in pydantic_tools]
        return CombinedToolset(toolsets)
    return _build_toolset
```

**After**:
```python
def get_toolset(self) -> Any:
    async def _build_toolset(ctx):
        client = await self._ensure_client()
        tools = await client.list_tools()
        if not tools:
            return None
        converted = [client.convert_tool(t) for t in tools]
        pydantic_tools = [wrap_tool_for_pydantic_ai(tool) for tool in converted]
        toolsets = [FunctionToolset[Any]([tool]) for tool in pydantic_tools]
        combined = CombinedToolset(toolsets)
        # Namespace tools by server name to prevent cross-server collisions.
        # Follows pydantic-ai's own convention: load_mcp_toolsets() uses
        # toolset.prefixed(name) at mcp.py:1754.
        # Skill-level MCP already uses PrefixedToolset(prefix=f"{skill}__mcp__").
        return PrefixedToolset(wrapped=combined, prefix=self._tool_prefix)
    return _build_toolset
```

Naming convention: the prefix is derived from the server's `display_name` only — the manager prefix (`pool_mcp_`) is **not** included, as it carries no semantic information for the model. Aligns with OpenCode's `<server>_<tool>` convention (e.g., `github_search`, `slack_send_message`). For a pool-level server with `display_name` `xeno-kb`, the prefix is `xeno-kb`. A tool named `search_database` becomes `xeno-kb_search_database` in the model's tool list. `McpServerCap` gains a `_tool_prefix` property:

```python
class McpServerCap(...):
    def __init__(self, config, *, name=None, ...):
        # `name` retains the manager-qualified identifier (pool_mcp_xeno-kb)
        # for status/logging/internal identity; `_tool_prefix` is the
        # model-visible namespace (xeno-kb).
        self._name = name or config.client_id
        self._tool_prefix = config.display_name

    @property
    def tool_prefix(self) -> str:
        return self._tool_prefix
```

Design note: The prefix uses `_` (underscore) as separator, consistent with `PrefixedToolset`'s implementation (`f'{self.prefix}_{name}'`, prefixed.py:32), with skill-level MCP's `__mcp__` convention, and with OpenCode's current `<server>_<tool>` naming (per OpenCode MCP docs). The model sees fully-qualified names; the `PrefixedToolset.call_tool()` method (prefixed.py:38) strips the prefix before dispatching to the original tool, so `MCPClient.call_tool()` receives the raw tool name as before.

#### 3. skill_manager_cap.py — Add ResourceAccess delegation

**New methods**:
```python
async def list_resources(self) -> Sequence[ResourceEntry]:
    """Delegate to skill-level MCP children implementing ResourceAccess."""
    entries: list[ResourceEntry] = []
    for caps in self._skill_mcp_children.values():
        for cap in caps:
            if isinstance(cap, ResourceAccess):
                try:
                    entries.extend(await cap.list_resources())
                except Exception:
                    continue
    return entries

async def read_resource(self, uri: str) -> list[TextResourceContent | BlobResourceContent] | None:
    """Delegate to skill-level MCP children."""
    for caps in self._skill_mcp_children.values():
        for cap in caps:
            if isinstance(cap, ResourceAccess):
                try:
                    result = await cap.read_resource(uri)
                except Exception:
                    continue
                if result is not None:
                    return result
    return None

async def resource_exists(self, uri: str) -> bool:
    """Delegate to skill-level MCP children."""
    for caps in self._skill_mcp_children.values():
        for cap in caps:
            if isinstance(cap, ResourceAccess) and await cap.resource_exists(uri):
                return True
    return False
```

#### 4. manager.py — `get_capabilities()` adjustment

Add `exclude_global: bool = False` parameter. When `True`, skip `snap.global_configs` processing (pool + agent configs) since those are handled by McpServerCap injection. Session-scoped configs continue to be processed.

```python
async def get_capabilities(
    self,
    session_id: str | None = None,
    *,
    exclude_global: bool = False,
) -> list[MCP]:
    ...
    if ctx is not None and ctx.snapshot is not None:
        if not exclude_global:
            await _process_global_configs(ctx.snapshot, self._toolset_cache)
        if ctx.connection_pool is not None:
            await _process_session_configs(...)
    else:
        if not exclude_global:
            # Legacy path: process self.servers
            for server in self.servers:
                ...
    return capabilities
```

#### 5. Unchanged components

| Component | Why unchanged |
|-----------|---------------|
| `agent_routes.py:list_mcp_resources()` | Already calls `registry.get_resource_access(scope)` — now finds POOL-scoped McpServerCap directly |
| `resource_capability.py` | Already calls `registry.get_resource_access(scope)` — benefits automatically |
| `resource_resolver.py` | Already iterates `resource_caps` from registry — benefits automatically |
| `mcp_server_cap.py` | `get_toolset()` and `ResourceAccess` implementations are already correct |
| `factory.py:_inject_pool_providers()` | ACP aggregating provider injection (Path C) is unchanged |
| `MCPManager.setup_server()` | McpServerCap creation logic is unchanged |
| `MCPManager.get_aggregating_provider()` | ACP-only filtering is unchanged |

### Data Flow After Changes

**`@` mention** (editor → resource list):
```
GET /experimental/resource
  → list_mcp_resources()
  → registry.get_resource_access(SESSION scope)
  → returns: [McpServerCap_A, McpServerCap_B, SkillManagerCap]
  → McpServerCap_A.list_resources() → MCP resources/list → 5 entries
  → McpServerCap_B.list_resources() → MCP resources/list → 3 entries
  → SkillManagerCap.list_resources() → delegates to _skill_mcp_children → 2 entries
  → Total: 10 resources, aggregated, returned to editor
```

**Model tool call** (model → MCP tool):
```
Model calls search_database(query="...")
  → pydantic-ai resolves tool from capabilities list
  → McpServerCap.get_toolset() returned ToolsetFunc
  → ToolsetFunc calls _ensure_client() → MCPClient (pre-created)
  → client.call_tool("search_database", {"query": "..."})
  → Result returned to model
```

**Model resource access** (model → ResourceCapability):
```
Model calls list_resources tool
  → ResourceCapability.list_resources()
  → registry.get_resource_access(scope)
  → returns: [McpServerCap_A, McpServerCap_B, SkillManagerCap]
  → Aggregated results formatted as text table
  → Returned to model as tool result
```

---

## Security Considerations

### Threat Analysis

| Threat | Impact | Likelihood | Mitigation |
|--------|--------|------------|------------|
| POOL-scope McpServerCap visible to all agents | Medium | Low | Intended behavior — top-level MCP servers are pool-wide resources. Agent-level isolation is maintained by agent-scope registration. |
| Resource URI leakage across agents | Medium | Low | Same as above — top-level server resources are intentionally shared. Session-scoped resources remain isolated via `get_capabilities()` snapshot. |
| `get_capabilities()` session scope bypass | High | Low | `exclude_global` flag only skips global configs; session-scoped configs are still processed through the snapshot mechanism. |

### Security Measures

- [ ] Verify that session-scoped MCP configs (session + skill) are NOT affected by the `exclude_global` flag
- [ ] Confirm that `get_resource_access(SESSION scope)` does not leak TURN-scoped capabilities from other sessions
- [ ] Ensure McpServerCap connection sharing across sessions does not expose per-session state (e.g., MCP session headers)

---

## Implementation Plan

### Phases

#### Phase 1: SkillManagerCap ResourceAccess Delegation

- **Scope**: Add `ResourceAccess` implementation to `SkillManagerCap` for `_skill_mcp_children`
- **Deliverables**: Updated `skill_manager_cap.py`, unit tests for delegation
- **Dependencies**: None
- **Risk**: Low — additive change, no existing behavior modified

#### Phase 2: Independent POOL-scope Registration

- **Scope**: Stop stuffing top-level McpServerCap into `SkillManagerCap.children`; register independently at POOL scope. Add `display_name` de-duplication in `MCPManager.setup_server()`.
- **Deliverables**: Updated `pool.py` `_rebuild_skill_capabilities()`, updated `manager.py` `setup_server()`, updated tests
- **Dependencies**: Phase 1 (SkillManagerCap no longer needs children for ResourceAccess)
- **Risk**: Medium — `SkillManagerCap` loses direct access to top-level MCP SkillResource providers; must rely on `ExtensionRegistry.get_skill_resources()` instead. Verify that `list_skills` / `read_skill` / `list_commands` still work via registry queries.

**display_name de-duplication** (in `manager.py:setup_server()`): the tool prefix derives from `display_name`, so two configured servers sharing a name would produce colliding prefixes. Resolve at server-setup time by tracking used names per manager and appending a numeric suffix on collision:

```python
# manager.py — inside __aenter__, before creating providers:
used_names: set[str] = set()
for server in self.servers:
    base = server.display_name
    candidate = base
    n = 2
    while candidate in used_names:
        candidate = f"{base}_{n}"
        n += 1
    used_names.add(candidate)
    # Pass `display_name=candidate` when constructing the McpServerCap
```

Behavior: two servers both named `github` become tool prefixes `github` and `github_2`. The `McpServerCap._name` (manager-qualified internal id) remains unique and unchanged; only the model-visible `tool_prefix` is de-duplicated.

#### Phase 3: Tool Exposure Migration

- **Scope**: Replace `get_capabilities()` for top-level servers with direct McpServerCap injection in `get_agentlet()`
- **Deliverables**: Updated `agent.py`, `manager.py` (`exclude_global` param), integration tests
- **Dependencies**: Phase 2 (McpServerCap already at POOL scope)
- **Risk**: Medium — must ensure no duplicate tools and no missing tools. ACP path must be unaffected.

#### Phase 4: Cleanup and Documentation

- **Scope**: Update RFC-0051 references, technical report, AGENTS.md; remove dead code paths
- **Deliverables**: Documentation updates, dead code removal
- **Dependencies**: Phase 3 complete and tested
- **Risk**: Low

### Milestones

| Milestone | Description | Target | Status |
|-----------|-------------|--------|--------|
| M1 | Phase 1 + 2: `@` mention works for top-level MCP | Day 2 | Not Started |
| M2 | Phase 3: Tool exposure via McpServerCap | Day 3 | Not Started |
| M3 | Phase 4: Documentation and cleanup | Day 4 | Not Started |

### Rollback Strategy

Each phase is independently revertible:
- Phase 1: Remove `ResourceAccess` methods from `SkillManagerCap`
- Phase 2: Restore `children=mcp_children` in `_rebuild_skill_capabilities()`
- Phase 3: Restore `get_capabilities()` call without `exclude_global`

Full rollback: revert all three phases in reverse order. No data migration is involved.

---

## Open Questions

1. **Should `SkillManagerCap.list_skills()` query the registry instead of `self._children`?**
   - Context: After Phase 2, `SkillManagerCap` no longer has top-level McpServerCap in `self._children`. Remote skill listing must either query `registry.get_skill_resources(scope)` or accept that only local + per-skill skills are listed.
   - Owner: pinjun.mo
   - Status: Open — leaning toward registry query for consistency with `ResourceAccess`

2. **Should `_setup_skills_provider()` (pool.py:610) also use the registry instead of `self.mcp.providers`?**
   - Context: `SkillURIResolver.register_provider()` currently iterates `self.mcp.providers` directly. After Phase 2, these providers are in the registry, but the resolver doesn't query the registry.
   - Owner: pinjun.mo
   - Status: Open — may defer to keep the resolver's direct registration path

3. **Does `get_capabilities()` need to handle the case where a top-level server is both in `self.providers` AND in session-scoped configs?**
   - Context: An agent could override a pool-level MCP server with an agent-level config of the same name. Currently `get_capabilities()` handles this via snapshot partitioning. After the change, the pool-level McpServerCap is injected directly, and the agent-level config goes through `get_capabilities()`.
   - Owner: yuchen.liu
   - Status: Open — needs verification that no duplicate tools result from this scenario

4. **Should `MCPManager.get_aggregating_provider()` be expanded to include non-ACP providers?**
   - Context: Currently ACP-only. After this RFC, non-ACP providers are injected via `pool.mcp.providers` in `get_agentlet()`. The aggregating provider could be a single injection point for all providers, simplifying `get_agentlet()`.
   - Owner: pinjun.mo
   - Status: Open — deferred to a follow-up; current design separates ACP (Path C) from non-ACP (direct injection) for clarity

5. **Should internal identity be separated from the model-visible tool name?**
   - Context: Currently `McpServerCap._name` (manager-qualified id, e.g. `pool_mcp_xeno-kb`) serves as the internal identity AND is used in status keys (`get_server_status`), resource `source_uri` (`mcp://{name}`), and logging. The new `tool_prefix` (model-visible, e.g. `xeno-kb`) is separate. The industry pattern (per ChatGPT/OpenCode discussion) is to fully separate `internal_id` (`mcp_01:tool_17`) from `modelName` (`github_search`), so server renames/reconnects never break tool identity. Deferring — current config-driven scale has stable display_names, and `_name` is documented as identity.
   - Owner: pinjun.mo
   - Status: Open — deferred to a follow-up; revisit when dynamic server config (add/remove servers at runtime) is introduced

6. **Should namespace use `.` (dot) instead of `_` (underscore) separators?**
   - Context: ChatGPT's original suggestion used dot notation (`github.search`), but its own follow-up confirmed OpenCode uses underscore (`github_search`) and pydantic-ai's `PrefixedToolset` is underscore-native (`f'{prefix}_{name}'`). This RFC adopts underscore. A dot-based separator would require bypassing `PrefixedToolset` and custom-prefixing tool definitions.
   - Owner: pinjun.mo
   - Status: Resolved — underscore; aligns with OpenCode + PrefixedToolset. Documented here to record the deliberation.

7. **Should top-level and skill-MCP tool prefix conventions be unified?**
   - Context: this RFC introduces `{display_name}` as the tool prefix for top-level MCP servers (e.g. `xeno-kb_search_database`), while skill-level MCP keeps the RFC-0052-established `{skill_name}__mcp__` prefix (e.g. `python-expert__mcp__read_code`). Both solve collision avoidance, but the two conventions coexist. Options: (a) keep them separate (skill prefix carries the `__mcp__` transport hint, top-level prefix is a pure server namespace); (b) unify skill MCP to `{skill_name}_{server_name}` dropping `__mcp__`; (c) establish a general `{namespace}__{subtype}__` scheme for all MCP prefixes. This RFC deliberately keeps the existing `__mcp__` convention to minimize blast radius; unification is a naming-polish follow-up.
   - Owner: pinjun.mo
   - Status: Open — deferred to a follow-up; current dual convention is internally consistent (both prevent collision) and backward compatible

---

## Decision Record

> Complete this section after RFC review is concluded.

### Decision

**Status**: [PENDING REVIEW]

**Date**: 

**Approvers**:
- [ ]

### Decision Summary

[TBD]

### Key Discussion Points

[TBD]

### Conditions of Approval

[TBD]

### Dissenting Opinions

[TBD]

---

## References

### Related Documents

- [RFC-0051: Extension Source Architecture](RFC-0051-extension-source-architecture.md) — Original design for Resource Protocols and McpServerCap
- [RFC-0052: Restore Skill Capabilities](RFC-0052-restore-skill-capabilities.md) — SkillManagerCap children wiring that caused the ResourceAccess gap
- [MCP Resource Technical Report](../../specs/mcp-resource-technical-report.md) — Full MCP resource consumption architecture documentation

### External Resources

- [MCP Specification 2026-07-28 — Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [RFC 3986 — Uniform Resource Identifier](https://www.rfc-editor.org/rfc/rfc3986)
- [RFC 6570 — URI Template](https://www.rfc-editor.org/rfc/rfc6570)

### Appendix

#### A. File-to-Change Mapping

| File | Phase | Change |
|------|-------|--------|
| `src/wolfharness/capabilities/skill_manager_cap.py` | 1 | Add `ResourceAccess` delegation methods |
| `src/wolfharness/capabilities/skill_manager_cap.py` | 2 | Remove RFC-0052 D2 case-3 "non-skill children unprefixed" dead code |
| `src/wolfharness/capabilities/mcp_server_cap.py` | 3 | Wrap `get_toolset()` result in `PrefixedToolset` for tool namespacing |
| `src/wolfharness/delegation/pool.py` | 2 | Independent POOL-scope registration; remove `children=mcp_children` |
| `src/wolfharness/agents/native_agent/agent.py` | 3 | Replace `get_capabilities()` with direct McpServerCap injection |
| `src/wolfharness/mcp_server/manager.py` | 2 | De-duplicate `display_name` in `setup_server()` for unique tool prefixes |
| `src/wolfharness/mcp_server/manager.py` | 3 | Add `exclude_global` param to `get_capabilities()` |
| `tests/capabilities/test_skill_manager_cap.py` | 1-2 | Test ResourceAccess delegation; test without top-level children |
| `tests/delegation/test_pool.py` | 2 | Test independent POOL-scope registration |
| `tests/agents/test_native_agent.py` | 3 | Test tool exposure via McpServerCap; test no duplicate tools |
| `tests/mcp_server/test_manager_capability.py` | 3 | Test `exclude_global` parameter |
| `tests/servers/opencode_server/test_resource_resolution.py` | 2 | Test `@` mention with top-level MCP resources |
