---
rfc_id: RFC-0060
title: Agent Mode Declaration for OpenCode At-Mention Visibility
status: DRAFT
author: pinjun.mo
reviewers: []
created: 2026-08-17
last_updated: 2026-08-17
decision_date:
related_prds: []
related_rfcs:
  - RFC-0034 (BackgroundTask Architecture Redesign for AgentPool)
  - RFC-0013 (Subagent Event Stream Unification for OpenCode Protocol)
---

# RFC-0060: Agent Mode Declaration for OpenCode At-Mention Visibility

## Overview

This RFC proposes adding a `mode` field to the wolfharness agent manifest so that users can declare whether each agent is a **primary** agent (shown in the OpenCode switcher), a **subagent** (shown in the at-mention `@` popup), or **both** (`all`). Today, wolfharness's OpenCode server hardcodes every agent as `mode="primary"`, which causes OpenCode clients to filter all custom agents out of the at-mention popup — making `@visionary`-style delegation impossible.

This change is scoped to the OpenCode protocol server in `packages/agentpool`. It does not alter agent execution semantics, delegation behavior, or other protocols (ACP, AG-UI, MCP).

## Table of Contents

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

## Background & Context

### Current State

wolfharness exposes agents to OpenCode clients through `GET /agent` in `src/wolfharness_server/opencode_server/routes/agent_routes.py`. The current implementation hardcodes `mode="primary"` for every agent in the manifest:

```python
agents = [
    Agent(
        name=name,
        display_name=agent.display_name,
        description=agent.description or f"Agent: {name}",
        mode="primary",          # hardcoded
        default=(name == default_name),
    )
    for name, agent in ctx.manifest.agents.items()
]
```

The wolfharness OpenCode server model already declares the full mode vocabulary:

```python
# src/wolfharness_server/opencode_server/models/agent.py
AgentMode = Literal["subagent", "primary", "all"]
class Agent(OpenCodeBaseModel):
    ...
    mode: AgentMode = "primary"
```

The manifest agent config (`NativeAgentConfig`) has `display_name`, `description`, `tools`, `capabilities`, `model`, and other runtime fields — but **no mode declaration field**.

### Historical Context

OpenCode (upstream client) determines at-mention visibility client-side in two places:

1. **Switcher** (`packages/app/src/context/local.tsx:71`):
   ```ts
   const list = createMemo(() =>
     sync().data.agent.filter((item) => item.mode !== "subagent" && !item.hidden)
   )
   ```
   The switcher shows every agent whose `mode !== "subagent"`.

2. **At-mention popup** (`packages/app/src/components/prompt-input-v2.tsx:275`):
   ```ts
   ...props.controls.agents.available
     .filter((agent) => !agent.hidden && agent.mode !== "primary")
     .map((agent) => ({ id: `agent:${agent.name}`, kind: "agent", label: `@${agent.name}`, ... }))
   ```
   The at-mention popup shows every agent whose `mode !== "primary"`.

Native OpenCode derives `mode` from per-agent config files (`{agent,agents}/*.md` markdown with frontmatter). An agent with `mode: subagent` appears in the at-mention popup only; `mode: primary` appears in the switcher only; `mode: all` appears in both.

Earlier RFC-0013 unified subagent event streaming for the OpenCode protocol, and RFC-0034 redesigned background-task delegation — but neither addressed how *static* multi-agent manifests surface their agent hierarchy to the client.

### Glossary

| Term | Definition |
|------|------------|
| **primary** | Agent mode shown in the OpenCode switcher (main agent tabs). |
| **subagent** | Agent mode shown in the at-mention (`@`) popup for delegation. |
| **all** | Agent mode visible in both switcher and at-mention popup. |
| **Agent** | The wolfharness OpenCode server response type (`AgentMode = Literal["subagent","primary","all"]`). |
| **AgentPart** | OpenCode message part created when a user types `@agent-name`; the server converts it into a prompt instructing the model to call the `task` tool. |
| **task tool** | Background-delegation tool (from `BackgroundTaskCapability`) that spawns a subtask for another agent. |

---

## Problem Statement

### The Problem

Users of the OpenCode integration cannot delegate to custom agents via at-mention. When a user configures multiple agents (e.g. a `viking_tester` main agent plus a `visionary` multimodal analyst) and types `@visionary`, the at-mention popup does not list `visionary` — it only lists files and built-ins.

### Evidence

- `agent_routes.py:154` hardcodes `mode="primary"` for every manifest agent.
- OpenCode client `prompt-input-v2.tsx:275` filters at-mention candidates to `mode !== "primary"`.
- Observed behavior on a running server (`GET /agent` returns `visionary` with `"mode":"primary","default":false`), while `@visionary` produces no suggestion in the client popup.
- The `task`/`background_task` capability and the `AgentPart` handling in `converters.py:216` are already implemented — the only missing piece is agent visibility metadata flowing from manifest to client.

### Impact of Inaction

- **Cost**: Every user who wants multi-agent delegation in OpenCode must work around the gap — either by relying on the model to self-select a `task` tool call (fragile, poor UX) or by forking the server.
- **Risk**: The gap between "agents configured" and "agents usable" creates a misleading DX: configs advertise agents that cannot actually be reached by the user.
- **Opportunity**: Without a fix, wolfharness's multi-agent story is invisible in the primary client UX; with it, `@`-driven delegation becomes a first-class interaction that matches native OpenCode's agent model.

---

## Goals & Non-Goals

### Goals (In Scope)

1. Add a `mode` field to the manifest agent config (`NativeAgentConfig`) with vocabulary `subagent | primary | all`, defaulting to `primary` for backward compatibility.
2. Surface the declared mode through `GET /agent` so the OpenCode client can render switcher and at-mention correctly.
3. Preserve existing behavior when `mode` is not declared (all agents remain `primary` — status quo).
4. Keep the change confined to the OpenCode protocol server layer + manifest config validation.

### Non-Goals (Out of Scope)

1. Changing agent **execution** semantics — a `subagent` is still a first-class pool agent; this RFC only changes *visibility metadata*.
2. Introducing server-side enforcement that a `subagent` cannot be used as the default/main agent — that decision is client-enforced today and out of scope.
3. Changing other protocols (ACP, AG-UI, MCP, A2A) — they have their own identity/visibility mechanisms.
4. Building a UI or documentation site for mode management.
5. Auto-deriving mode from graph/team structure (e.g. "agents referenced by `task` are subagents") — deferred as a future enhancement.

### Success Criteria

- [ ] A YAML config declaring `mode: subagent` on an agent results in that agent appearing in the at-mention popup (`/agent` returns `mode: "subagent"`).
- [ ] A YAML config declaring `mode: primary` on an agent results in that agent appearing only in the switcher (default, backward compatible).
- [ ] A YAML config declaring `mode: all` results in the agent appearing in both.
- [ ] Omitting `mode` on all agents reproduces current behavior exactly (no regression).
- [ ] Existing wolfharness tests pass without modification (unless a test explicitly asserts `mode="primary"` hardcoding, which would be updated).

---

## Evaluation Criteria

Weight: High = strongest driver for this decision.

| Criterion | Weight | Description | Minimum Threshold |
|-----------|--------|-------------|-------------------|
| **Backward Compatibility** | High | Existing configs without `mode` must behave identically | Must pass: zero behavior change when field absent |
| **Implementation Cost** | High | Development effort in days | Low (< 2 days) |
| **Fidelity to OpenCode semantics** | High | Matches native OpenCode `primary/subagent/all` model | Must align with client filter logic |
| **Maintainability** | Medium | Changes are small, local, and readable | Single server layer file + config field |
| **Flexibility** | Medium | Supports future extensions (e.g. auto-derive mode) | Field design does not preclude future work |
| **Testability** | Medium | Easy to write unit tests on route output | Tests on `list_agents` with mocked manifest |

---

## Options Analysis

### Option 1: Manifest-declared `mode` field (Recommended Candidate)

**Description**

Add `mode: Literal["subagent", "primary", "all"] = "primary"` to `NativeAgentConfig`. Update `list_agents` to read `agent.mode` instead of the hardcoded literal. Users declare per-agent visibility in YAML:

```yaml
agents:
  viking_tester:
    type: native
    mode: primary        # default; shown in switcher
    ...
  visionary:
    type: native
    mode: subagent       # shown in at-mention only
    ...
  logician:
    type: native
    mode: all            # shown in both
    ...
```

Validation is handled by Pydantic at config-parse time (invalid values rejected before server start). The `Agent` wire type already supports the full vocabulary.

**Advantages**

- Explicit, declarative, and matches native OpenCode's per-agent config semantics.
- Backward compatible by construction (default `primary` = today's behavior).
- Single, low-risk change surface: one config field + one route lookup.
- Pydantic gives free schema validation and JSON Schema documentation.
- `all` mode enables the useful "switcher + delegable" hybrid that upstream OpenCode supports.

**Disadvantages**

- Each agent's mode is static — a rename from "subagent" to "primary" requires an edit and a server restart.
- Adds a field users must learn about (mitigated by a clear default).
- No automatic consistency check (e.g. "an agent that only exists to be delegated should probably be `subagent`") — user responsibility.

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Backward Compatibility | 5 | Default `primary` reproduces exact current behavior |
| Implementation Cost | 5 | ~1 field + ~3-line route change + tests |
| Fidelity to OpenCode semantics | 5 | Mirrors upstream `primary/subagent/all` exactly |
| Maintainability | 5 | Tiny diff, no architectural change |
| Flexibility | 4 | Field design permits future auto-derivation |
| Testability | 5 | Trivial unit tests on `list_agents` |

**Effort Estimate**

- Complexity: Low
- Resources: 1 engineer, 0.5–1 day including tests
- Dependencies: None beyond wolfharness own manifest + OpenCode routes

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Configs with typo'd mode values fail at startup | Low | Medium | Pydantic Literal rejection error is descriptive; document in YAML reference |
| Tests asserting `mode="primary"` hardcoding break | Medium | Low | Update those specific assertions; they encode the bug, not a contract |
| Client caches old `/agent` response | Medium | Low | Standard server restart; no wire-format change for existing `mode` values |

---

### Option 2: Auto-derive mode from config structure

**Description**

Infer mode from existing manifest structure without a new field. For example: the agent named by `default_agent` (or `main_agent_name`) becomes `primary`; every other agent becomes `subagent`. Optionally, agents referenced in `graph:`/`connections:` or delegation instructions become `subagent`.

**Advantages**

- Zero config changes — existing YAML files gain at-mention visibility automatically.
- "Main agent is primary, everyone else is delegable" is a sensible default for team-style configs.

**Disadvantages**

- Hidden magic: users cannot express `all` (both switcher + at-mention) or override the heuristic.
- Breaks today's behavior *by default*: agent sets without `default_agent` would silently change mode; agents currently expected in the switcher would disappear from it.
- `diag-agent.yaml`-style configs with multiple "main" agents (e.g. `engineer`, `reviewer`) would all become subagents, collapsing the switcher.
- Less explicit and harder to reason about than a declared field.

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Backward Compatibility | 2 | Breaks current switcher content by default |
| Implementation Cost | 4 | Heuristic in `list_agents` only |
| Fidelity to OpenCode semantics | 3 | No way to express `all`; lossy mapping |
| Maintainability | 2 | Hidden rule, hard to discover or override |
| Flexibility | 2 | Heuristic blocks user control |
| Testability | 3 | Behavior depends on manifest shape |

**Effort Estimate**

- Complexity: Low-Medium
- Resources: 1 engineer, ~0.5 day
- Dependencies: None

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Existing multi-agent configs silently lose switcher agents | High | High | Auto-derivation is too aggressive; must add escape hatch, which becomes Option 1 anyway |
| Users cannot express `all` | High | Medium | No `all` in heuristic model |
| Confusing debug sessions ("why is my agent not in the switcher now?") | Medium | Medium | Requires docs + debugging |

---

### Option 3: Config-level opt-out flag (`hidden`) instead of mode

**Description**

Add a boolean `hidden: bool = False` to `NativeAgentConfig`. Pass it through to the wire `Agent.hidden` field. The client's at-mention filter is `!agent.hidden && agent.mode !== "primary"` — but since all wolfharness agents are `primary`, `hidden` alone still does not surface them in at-mention. To make delegation visible, this option would require **also** changing the server default mode to `"all"` or making `hidden`/mode interact.

**Advantages**

- Reuses an existing OpenCode field (`Agent.hidden`) already in the wire model.
- `hidden` is useful on its own (suppress an agent from all client surfaces).

**Disadvantages**

- **Does not solve the problem alone**: the at-mention filter's first clause is `mode !== "primary"`, so without mode changes, `hidden: false` does not add anything to the popup.
- Requires the same mode work as Option 1 to achieve visibility, making it strictly more complex.
- Two interacting fields (`hidden` + `mode`) create a confusing matrix for users.

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Backward Compatibility | 4 | New field absent = no behavior change |
| Implementation Cost | 4 | Two fields + route changes |
| Fidelity to OpenCode semantics | 3 | Requires mode change anyway; `hidden` alone insufficient |
| Maintainability | 3 | Two interacting flags |
| Flexibility | 3 | `hidden` orthogonal but incomplete |
| Testability | 3 | Two code paths to cover |

**Effort Estimate**

- Complexity: Medium
- Resources: 1 engineer, ~1 day (but incomplete without Option 1's mode wiring)
- Dependencies: Needs mode work from Option 1

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Ship `hidden`-only and discover at-mention still broken | High | High | `hidden` does not affect the `mode !== "primary"` clause |
| Field interaction confusion | Medium | Medium | Document matrix |

---

### Options Comparison Summary

| Criterion | Option 1 (mode field) | Option 2 (auto-derive) | Option 3 (hidden flag) |
|-----------|----------------------|------------------------|------------------------|
| Backward Compatibility | 5 | 2 | 4 |
| Implementation Cost | 5 | 4 | 4 |
| Fidelity to OpenCode semantics | 5 | 3 | 3 |
| Maintainability | 5 | 2 | 3 |
| Flexibility | 4 | 2 | 3 |
| Testability | 5 | 3 | 3 |
| **Weighted Total** (60% tech, 40% ops) | **4.80** | **2.70** | **3.20** |

---

## Recommendation

### Recommended Option

**Option 1: Manifest-declared `mode` field**

### Justification

Option 1 scores highest across every criterion. It is the only option that is fully backward compatible *and* fully faithful to OpenCode's native `primary/subagent/all` model. The implementation is minimal (single config field + route lookup + tests), the semantics are explicit and user-controlled, and `all` mode gives users the hybrid visibility that upstream OpenCode supports. Option 2 hides behavior in a heuristic that collapses multi-agent switchers by default — unacceptable for a library where configs already rely on multiple visible agents. Option 3 is strictly a subset of Option 1's work and cannot solve the stated problem alone because the at-mention filter's primary gate is `mode`, not `hidden`.

### Accepted Trade-offs

1. **Static mode requires config edits to change visibility**: Acceptable because agent role changes are infrequent (adding/removing a delegable specialist), and declarative YAML is the project's configuration philosophy.
2. **No automatic mode validation/consistency**: Acceptable because the field is soft metadata; a misconfigured agent is visible or hidden, never mis-executing. Pydantic rejects invalid enum values at parse time regardless.

### Conditions

- `mode` must default to `"primary"` to preserve existing behavior exactly.
- The wire `Agent.mode` field already exists; no client-visible wire format change is introduced (existing values are a subset of the Literal).
- Manifest `mode` should appear in the YAML configuration reference documentation with the three-value vocabulary.

---

## Technical Design

> Preliminary design for review; finalize after acceptance.

### Architecture Overview

```
┌─────────────────────┐      ┌──────────────────────────────┐      ┌──────────────────┐
│  Manifest (YAML)    │      │  OpenCode server (agentpool) │      │  OpenCode client │
│  mode: subagent     │─────▶│  list_agents reads mode      │─────▶│  switcher:       │
│  mode: all          │      │  Agent.mode = cfg.mode       │      │  mode!=subagent  │
│  mode: primary      │      │                              │      │  at-mention:     │
└─────────────────────┘      └──────────────────────────────┘      │  mode!=primary   │
                                                                   └──────────────────┘
```

### Key Components

#### 1. Manifest config field (`NativeAgentConfig`)

- Location: `src/wolfharness/models/agents.py`
- Technology: Pydantic v2 `Literal` field on `NativeAgentConfig`
- Interface:

```python
# On NativeAgentConfig (BaseAgentConfig or NativeAgentConfig)
mode: AgentMode = "primary"   # AgentMode = Literal["subagent", "primary", "all"]
```

Where `AgentMode` is imported from the OpenCode server model (or a shared constants module). To avoid a server→core import inversion, `AgentMode` should be defined in a core/shared location (e.g. `wolfharness_config` or a constants module) and re-exported by the OpenCode server model.

#### 2. Route lookup (`GET /agent`)

- Location: `src/wolfharness_server/opencode_server/routes/agent_routes.py`
- Change:

```python
agents = [
    Agent(
        name=name,
        display_name=agent.display_name,
        description=agent.description or f"Agent: {name}",
        mode=agent.mode,          # was: "primary"
        default=(name == default_name),
    )
    for name, agent in ctx.manifest.agents.items()
]
```

#### 3. Wire model (unchanged)

- Location: `src/wolfharness_server/opencode_server/models/agent.py`
- The `Agent.mode: AgentMode = "primary"` field already exists — no change.

### Data Model

```python
# wolfharness_config or shared core module
AgentMode = Literal["subagent", "primary", "all"]

# src/wolfharness/models/agents.py
class NativeAgentConfig(BaseAgentConfig):
    mode: AgentMode = "primary"   # NEW
    ...

# src/wolfharness_server/opencode_server/models/agent.py
AgentMode = AgentMode              # re-export from core
class Agent(OpenCodeBaseModel):
    mode: AgentMode = "primary"    # unchanged
```

### API Design

`GET /agent` response — no wire format change, only the `mode` value becomes configurable:

```
GET /agent
→ [{ "name": "viking_tester", "mode": "primary",  "default": true,  ... },
   { "name": "visionary",     "mode": "subagent", "default": false, ... },
   { "name": "logician",      "mode": "all",      "default": false, ... }]
```

---

## Security Considerations

### Threat Analysis

| Threat | Impact | Likelihood | Mitigation |
|--------|--------|------------|------------|
| A config author mistakenly marks a privileged agent `subagent`, hiding it from default switcher | Low | Low | Mode is declarative metadata; the agent remains fully functional by name; `all` mode exists for hybrid needs |
| Malformed `mode` value in YAML | Low | Low | Pydantic `Literal` rejects unknown values at config parse with a descriptive error |
| Delegation to a `subagent` without authorization | Low | Low | Mode does not change delegation authorization; `BackgroundTaskCapability` already governs `task` tool usage |

### Security Measures

- [x] Pydantic `Literal` validation rejects unknown `mode` values at config load.
- [ ] Document that `mode` affects client *visibility* only, not execution authorization.

### Compliance

No regulatory requirements are affected. This is a client-visibility metadata change.

---

## Implementation Plan

### Phases

#### Phase 1: Core manifest field

- **Scope**: Define `AgentMode` in a shared core location and add `mode` to `NativeAgentConfig`.
- **Deliverables**: `mode` field with default `"primary"`, validated by Pydantic.
- **Dependencies**: None.

#### Phase 2: Route wiring

- **Scope**: Read `agent.mode` in `list_agents`.
- **Deliverables**: `GET /agent` returns configured mode values.
- **Dependencies**: Phase 1.

#### Phase 3: Tests + docs

- **Scope**: Unit tests for `list_agents` (mode passthrough, default behavior, all three values). Update YAML reference docs.
- **Deliverables**: Green test suite; documented YAML field.
- **Dependencies**: Phase 2.

### Milestones

| Milestone | Description | Target | Status |
|-----------|-------------|--------|--------|
| M1 | Manifest field + route wiring (Phase 1+2) | Day 1 | Not Started |
| M2 | Tests + docs (Phase 3) | Day 1-2 | Not Started |

### Rollback Strategy

Single-commit revert (config field + route line). Since the change is additive with a backward-compatible default, rollback is a plain `git revert` with no data migration.

---

## Open Questions

1. **Where should `AgentMode` be defined?**
   - Context: The OpenCode server model currently owns `AgentMode`. Manifest (core) needs it too, which would invert core→server import. Options: define in `wolfharness_config` (config-layer types), core models, or a shared constants module.
   - Owner: wolfharness core maintainers
   - Status: Open

2. **Should ACP/other protocols also expose mode?**
   - Context: ACP has its own agent registry (`RegistryAgent`) with a different shape. Out of scope here, but should be tracked for parity.
   - Owner: team
   - Status: Open

3. **Does the client-popup filter also respect `hidden`?**
   - Context: We are not adding `hidden` now. If a future need arises, its interaction with `mode` should be documented.
   - Owner: team
   - Status: Open

---

## Decision Record

> Complete this section after review is concluded.

### Decision

**Status**: PENDING — DRAFT in review

**Date**:

**Approvers**:

### Decision Summary

### Key Discussion Points

### Conditions of Approval

### Dissenting Opinions

---

## References

### Related Documents

- `src/wolfharness_server/opencode_server/routes/agent_routes.py`
- `src/wolfharness_server/opencode_server/models/agent.py`
- `src/wolfharness/models/agents.py` (`NativeAgentConfig`)
- `docs/rfcs/RFC-0013-subagent-event-unification.md`
- `docs/rfcs/RFC-0034-background-task-redesign.md`

### External Resources

- OpenCode app at-mention filter: `packages/app/src/components/prompt-input-v2.tsx` (`filter: agent.mode !== "primary"`)
- OpenCode app switcher filter: `packages/app/src/context/local.tsx` (`filter: item.mode !== "subagent"`)
- OpenCode agent config model: `packages/opencode/src/agent/agent.ts` (`mode: Schema.Literals(["subagent", "primary", "all"])`)
- OpenCode SDK agent type: `@opencode-ai/sdk/v2` `Agent`

### Appendix

Config example for the motivating use case:

```yaml
# packages/xeno-agent/config/diag-agent-viking.yaml (proposed)
agents:
  viking_tester:
    type: native
    model: glm52
    mode: primary                 # main diagnostic agent (switcher)
    capabilities:
      - *viking_kb_capability
      - *background_task_capability
  visionary:
    type: native
    model: qwen3-vl-235b
    mode: subagent                # multimodal analyst (at-mention only)
    tools: []
    capabilities:
      - *viking_kb_capability

# User flow:
#   用户输入 "@visionary 分析这张电路图" →
#   客户端 at-mention 弹窗显示 @visionary（因为 mode=subagent）→
#   AgentPart(name="visionary") → 服务端注入 task 指令 → viking_tester 调用 task 工具委派给 visionary
```