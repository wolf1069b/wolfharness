---
rfc_id: RFC-0053
title: Pre-M4 Protocol Server Debt Cleanup
status: REVIEW
author: Leoyzen
reviewers:
  - name: pending
    status: pending
created: 2026-07-13
last_updated: 2026-07-13
decision_date:
related_prds: []
related_rfcs:
  - RFC-0050 (AgentWolf v1 Foundation Architecture)
  - RFC-0042 (Unified Lifecycle Architecture)
  - RFC-0041 (Loop/Run Separation)
---

# RFC-0053: Pre-M4 Protocol Server Debt Cleanup

## Overview

This RFC proposes a phased cleanup of 51 technical debt items identified across the ACP server, OpenCode server, and shared orchestration infrastructure on the `refactor/agentwolf_v1` branch. The cleanup establishes the architectural baseline required before M4 (multi-config) can begin. M1–M3.5 refactoring is complete — `HostContext` replaces `AgentPool` access, lifecycle dimensions decouple the RunLoop, and the `ResourceProvider` hierarchy is replaced by native pydantic-ai `AbstractCapability`. However, the protocol server layer accumulated dual execution paths, type safety erosion, encapsulation violations, and identity coupling that will compound under M4's multi-config model.

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

The `refactor/agentwolf_v1` branch (PR #144) contains 75 commits across M1–M3.5 refactoring milestones. The branch is OPEN and MERGEABLE with CI green. Key completions:

- `.agent_pool` references in source: 0 (backdoor eliminated)
- `ResourceProvider` hierarchy: deleted, replaced by `AbstractCapability`
- `ProtocolEventConsumerMixin`: adopted by all 4 event-streaming servers (ACP, OpenCode, AG-UI, OpenAI API)
- `EventBus`: clean (no `receive()`/`get()` issues)
- 4,385 tests collected, CI passing

### Historical Context

The debt accumulated during rapid M1–M3 iteration where architectural foundations were replaced while protocol servers continued to use the old patterns. Three parallel explore agents systematically catalogued the debt:

| Domain | Blocking | Severe | Moderate | Nice-to-have | Total |
|--------|----------|--------|----------|--------------|-------|
| ACP Server | 6 | 0 | 0 | 12 | 18 |
| OpenCode Server | 0 | 8 | 6 | 5 | 19 |
| Shared Infrastructure | 0 | 0 | 14 | 0 | 14 |
| **Total** | **6** | **8** | **20** | **17** | **51** |

### Glossary

| Term | Definition |
|------|------------|
| `HostContext` | Immutable dataclass providing access to MCP manager, storage, registry without exposing the full mutable `AgentPool` |
| `RunHandle` | The RunLoop implementation; modified in-place with dimension injection (TriggerSource, Journal, SnapshotStore, CommChannel, EventTransport) |
| `Turn.execute()` | Unified turn execution path via `HookAwareTurn`; the target single path for all agent types |
| `_run_stream_once()` | Legacy standalone execution path in `BaseAgent`; fires hooks inline for non-native agents |
| `hooks_fired` | `set[str]` on `AgentRunContext` preventing double hook firing when both paths execute |
| `CommChannel` | Protocol abstracting event delivery + feedback reception; owns the Journal |
| `RunScope` | Proposed cross-cutting routing context (config_id, tenant_id, session_id, user_id) for M4 |
| `ProtocolEventConsumerMixin` | Reusable event consumer lifecycle for protocol servers |

---

## Problem Statement

### The Problem

Four categories of technical debt block or complicate M4:

**1. Dual Execution Paths** — ACP agents have two code paths held together by a `hooks_fired` double-fire guard (21 refs across 4 files). `ACPTurn.execute()` is dead code because `ACPAgentAPI` is missing `stream_events()` and `get_messages()` methods. ACP standalone falls back to a 200-LOC inline `_stream_events()` implementation.

**2. Encapsulation Violations** — OpenCode routes access private attributes (`_all_capabilities`, `_sessions`, `_servers`) in 6 files. 68 `state.pool.*` accesses bypass `HostContext`. ACP server mutates `NativeAgent._mcp_snapshot` externally.

**3. Type Safety Erosion** — 8 `# type: ignore[attr-defined]` in `run.py` alone. `deliver_feedback` is duck-typed via `try/except AttributeError`. `_channel_publishes_to_event_bus` uses fragile `isinstance` check. `hasattr`/`getattr` patterns in ACP code.

**4. Identity Coupling** — OpenCode server hardcodes `state.agent.name` as session identity (5 files) and `config_file_path` as pool identity (3 files). `session_controller` hardcodes `self.pool.manifest.agents` (4 sites). Under M4's multi-config model, these assumptions break.

### Evidence

```mermaid
graph TD
    subgraph "Debt Distribution by File"
        run_py["orchestrator/run.py<br/>6 type:ignore, ~373 SLOC"]
        base_agent["base_agent.py<br/>dual paths, 21 hooks_fired refs"]
        acp_agent["acp_agent.py<br/>200 LOC inline, dead ACPTurn"]
        opencode_routes["opencode_server/routes/*.py<br/>68 state.pool accesses, 6 private access"]
        session_ctrl["session_controller.py<br/>4 hardcoded pool.manifest refs"]
        agent_py["native_agent/agent.py<br/>_mcp_snapshot, _session_connection_pool"]
    end

    style run_py fill:#f99,stroke:#c00
    style base_agent fill:#f99,stroke:#c00
    style acp_agent fill:#f99,stroke:#c00
    style opencode_routes fill:#f96,stroke:#c60
    style session_ctrl fill:#f96,stroke:#c60
    style agent_py fill:#f96,stroke:#c60
```

- `grep -rn 'hooks_fired' src/` → 21 results across 4 files
- `grep -rn '# type: ignore' src/wolfharness/orchestrator/run.py` → 8 results
- `grep -rn 'state\.pool\.' src/wolfharness_server/opencode_server/` → 68 results
- `grep -rn '_mcp_snapshot\|_session_connection_pool' src/` → results in agent.py + session.py
- `RunHandle.start()` → ~373 SLOC with `# noqa: PLR0915`

### Impact of Inaction

- **Cost**: M4 implementation must thread `RunScope` through both execution paths, doubling routing surface area. Identity coupling requires touching every OpenCode route during M4 rather than now.
- **Risk**: Type safety erosion compounds — each new `# type: ignore` makes the next one easier to add. Dual paths make hook behavior harder to debug under multi-config.
- **Opportunity**: M4's `ConfigRegistry`/`HostRegistry` requires a clean single-entry-point architecture. Fixing debt during M4 means debugging both layers simultaneously.

---

## Goals & Non-Goals

### Goals (In Scope)

1. Unify ACP execution through `Turn.execute()` as the sole path; remove `hooks_fired` guard (21 refs → 0)
2. Remove 6 `# type: ignore[attr-defined]` in `run.py` by typing `CommChannel` protocol completely
3. Remove `_mcp_snapshot`/`_session_connection_pool` from `NativeAgent`; consolidate on `MCPManager`
4. Remove legacy `RunStatus` enum; add `RunOutcome` to preserve terminal state distinction
5. Wire `McpToolsChangedEvent` and distinguish `StreamCompleteEvent(cancelled=True)` in `EventProcessor`
6. Refactor `RunHandle.start()` (~373 SLOC) into 5 sub-methods each < 100 SLOC
7. Add `deliver_feedback`, `publishes_to_event_bus`, `set_replaying` to `CommChannel` protocol

### Non-Goals (Out of Scope — Merged into M4)

1. Migrate 68 `state.pool.*` accesses to `state.host_context.*` in OpenCode server → M4 task group 18
2. Introduce `RunScope` dataclass with default values → M4 task groups 7-8
3. Add public API methods for private attribute access in OpenCode routes → M4 task group 18
4. Handle `RunStartedEvent` in `EventProcessor` → M4 task 18.3
5. Remove single-config hardcoding in `session_controller` → M4 task 18.9

### Non-Goals (Out of Scope — Deferred)

6. AgentWolf rename (`wolfharness` → `agentwolf`) — deferred to final phase
7. M4 implementation itself (ConfigRegistry, HostRegistry, RunScope routing)
8. ACP protocol version switch or proxy chain refactor
9. `session_pool_integration.py` file split (1,453 LOC) — optional, can defer
10. `_agent_pool` constructor threading removal (25+ refs) — deferred to rename

### Success Criteria

- [ ] `grep -rn 'hooks_fired' src/` returns 0
- [ ] `grep -rn 'type: ignore\[attr-defined\]' src/wolfharness/orchestrator/run.py` returns 0
- [ ] `grep -rn '_mcp_snapshot\|_session_connection_pool' src/` returns 0
- [ ] `grep -rn 'RunStatus' src/` returns 0
- [ ] `grep -rn 'host_context.pool' src/` returns 0
- [ ] `grep -rn '_replaying' src/wolfharness/orchestrator/run.py` returns 0
- [ ] `uv run pytest` — all tests pass
- [ ] `uv run ruff check src/` — no new lint errors
- [ ] `uv run --no-group docs mypy src/` — no new type errors
- [ ] ACP standalone streaming snapshot test passes (same events + metadata enrichment as before)

---

## Evaluation Criteria

| Criterion | Weight | Description | Minimum Threshold |
|-----------|--------|-------------|-------------------|
| M4 Readiness | High | Eliminates all debt items that would block or complicate M4 implementation | All 6 blocking + 8 severe items resolved |
| Type Safety | High | Removes `# type: ignore`, `hasattr`, `getattr`, `try/except AttributeError` patterns | 0 `# type: ignore` in `run.py` |
| Architectural Cleanliness | High | Eliminates dual paths, legacy enums, encapsulation violations | Single execution path for all agent types |
| Implementation Speed | Medium | Time from start to M4-ready baseline | ≤ 30 task-days |
| Risk of Regression | Medium | Likelihood of breaking existing functionality | All 4,385 tests pass; snapshot test green |
| Merge Conflict Risk | Low | Likelihood of conflicts with `develop/agentic` | Minimized by scoping to server/orchestrator layer |

---

## Options Analysis

### Option 1: Full Cleanup Before M4 (Phases 1–6)

**Description**

Execute all 6 phases (43 tasks) covering ACP path unification, legacy field cleanup, OpenCode hardening, type safety, M4 identity preparation, and event system gaps. Phase 7 (14 nice-to-have items) remains optional.

**Advantages**

- Establishes clean architectural baseline; M4 builds on solid foundation
- All 51 debt items addressed in a single focused effort
- Codebase context is fresh from M1–M3 work; lower cognitive cost than returning later
- Verification gates provide clear M4 readiness signal
- `RunScope` abstraction in Phase 5 makes M4's routing a configuration change, not a code change

**Disadvantages**

- Extended PR #144 lifecycle (~30 task-days)
- Potential merge conflicts with `develop/agentic` during the extended period
- Large changeset increases review burden
- `ACPTurn.execute()` is dead code — making it live may surface latent bugs

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| M4 Readiness | ★★★★★ | All 6 blocking + 8 severe + 14 shared items resolved |
| Type Safety | ★★★★★ | 0 `type: ignore` in `run.py`; CommChannel fully typed |
| Architectural Cleanliness | ★★★★★ | Single execution path; no legacy enums; no encapsulation violations |
| Implementation Speed | ★★★☆☆ | ~30 task-days; 6 phases |
| Risk of Regression | ★★★☆☆ | Large changeset; ACP path change is behavioral |
| Merge Conflict Risk | ★★☆☆☆ | Extended PR lifecycle increases exposure |

**Effort Estimate**

- Complexity: High
- Resources: 1 developer, ~30 task-days
- Dependencies: None (codebase is stable, CI green)

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ACP path unification surfaces latent bugs | Medium | High | Snapshot test before/after; incremental commits per task |
| Hook ordering changes break ACP tests | Medium | Medium | Tests that mock `_run_stream_once` updated in Phase 1 |
| Merge conflicts with `develop/agentic` | Medium | Low | Rebase frequently; scope changes to server/orchestrator layer |

---

### Option 2: Core Cleanup Only (Phases 1–4), Defer Identity & Events to M4

**Description**

Execute Phases 1–4 (22 tasks) covering ACP path unification, legacy field cleanup, OpenCode hardening, and type safety. Defer Phase 5 (M4 identity) and Phase 6 (event gaps) to be handled during M4 implementation.

**Advantages**

- Faster to M4 start (~20 task-days vs 30)
- Addresses the most architecturally significant debt (dual paths, type erosion)
- Smaller changeset reduces review burden and merge conflict window
- M4 team can handle identity abstraction as part of their feature work

**Disadvantages**

- M4 must handle identity coupling (`agent.name` → `RunScope.session_id`) alongside new ConfigRegistry/HostRegistry work
- Event gaps (`RunStartedEvent`, `McpToolsChangedEvent`) persist into M4
- `session_controller` hardcoding requires M4 to touch 4 additional sites
- Debugging M4 routing bugs is harder when identity abstraction is incomplete

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| M4 Readiness | ★★★☆☆ | Blocking items resolved, but identity coupling remains |
| Type Safety | ★★★★★ | Same as Option 1 for Phases 1–4 |
| Architectural Cleanliness | ★★★★☆ | Dual paths eliminated, but event gaps remain |
| Implementation Speed | ★★★★☆ | ~20 task-days; 4 phases |
| Risk of Regression | ★★★★☆ | Smaller changeset; less behavioral change |
| Merge Conflict Risk | ★★★☆☆ | Shorter PR lifecycle |

**Effort Estimate**

- Complexity: Medium-High
- Resources: 1 developer, ~20 task-days
- Dependencies: None

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| M4 identity work is harder without RunScope abstraction | High | Medium | M4 team creates RunScope as first task |
| Event gaps cause M4 test failures | Medium | Low | Fix events as M4 bugs are discovered |

---

### Option 3: ACP Unification Only (Phase 1), Defer Everything Else

**Description**

Execute only Phase 1 (6 tasks) to unify ACP execution through `Turn.execute()`. All other debt is deferred to M4 or later.

**Advantages**

- Minimal pre-M4 work (~6 task-days)
- Addresses the single most critical debt (dead `ACPTurn.execute()`, `hooks_fired` guard)
- Fastest path to M4 start

**Disadvantages**

- M4 development begins on a baseline with 45 unresolved debt items
- Type safety erosion (`type: ignore` cluster) persists and may grow during M4
- OpenCode server's 68 `state.pool.*` accesses make M4 routing changes touch every route
- Identity coupling requires M4 to change 12+ sites across OpenCode + session_controller
- Harder to isolate M4-introduced bugs from pre-existing debt

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| M4 Readiness | ★★☆☆☆ | Only 1 of 6 blocking items addressed; 45 items remain |
| Type Safety | ★★☆☆☆ | `type: ignore` cluster untouched |
| Architectural Cleanliness | ★★☆☆☆ | One path unified, rest remains |
| Implementation Speed | ★★★★★ | ~6 task-days; 1 phase |
| Risk of Regression | ★★★★★ | Minimal changeset |
| Merge Conflict Risk | ★★★★★ | Shortest PR lifecycle |

**Effort Estimate**

- Complexity: Medium
- Resources: 1 developer, ~6 task-days
- Dependencies: None

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| M4 bugs are hard to isolate from pre-existing debt | High | High | Accept and debug as encountered |
| Type safety continues eroding during M4 | Medium | Medium | Code review vigilance |

---

### Option 4: Hybrid — Orthogonal Phases Separately, Overlapping Phases with M4

**Description**

Execute Phases 1, 2, 4, 6 (19 tasks, ~15 days) as a separate pre-M4 cleanup since they touch files orthogonal to M4's scope. Merge Phases 3 and 5 (9 tasks) into the `m4-multi-config` change as task group 18, since they modify the same OpenCode route files that M4's RunScope routing touches.

```mermaid
flowchart LR
    subgraph Separate["Separate (pre-M4, ~15 days)"]
        P1["Phase 1: ACP Unification<br/>6 tasks"]
        P2["Phase 2: Legacy Cleanup<br/>5 tasks"]
        P4["Phase 4: Type Safety<br/>5 tasks"]
        P6["Phase 6: Event Gaps<br/>3 tasks"]
    end

    subgraph M4["Merged into M4"]
        P3["Phase 3: OpenCode Hardening<br/>5 tasks → Group 18"]
        P5["Phase 5: RunScope Identity<br/>4 tasks → Groups 7-8 + 18"]
    end

    Separate --> M4

    style Separate fill:#9f9,stroke:#0a0
    style M4 fill:#ff9,stroke:#c60
```

**Advantages**

- Orthogonal cleanup (Phases 1/2/4/6) is isolated and independently verifiable
- ACP path unification (highest risk) is debugged before M4 complexity is added
- OpenCode route files are changed only once (during M4), not twice
- RunScope abstraction is introduced alongside M4's routing implementation, not as a separate empty shell
- Shorter pre-M4 timeline (~15 days vs 30) with no loss of M4 readiness
- Type safety improvements make M4 code type-safe from the start

**Disadvantages**

- M4 carries additional cleanup tasks (9 tasks) alongside new feature work
- OpenCode hardening verification is interleaved with M4 verification
- Phase 3 and 5 task tracking spans two OpenSpec changes

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| M4 Readiness | ★★★★★ | All items resolved; overlapping items resolved within M4 |
| Type Safety | ★★★★★ | Phases 1/2/4 address all type:ignore and duck-typing |
| Architectural Cleanliness | ★★★★★ | Single execution path; no encapsulation violations |
| Implementation Speed | ★★★★☆ | ~15 days separate + 9 tasks folded into M4 (no extra time) |
| Risk of Regression | ★★★★☆ | Smaller separate changeset; ACP risk isolated |
| Merge Conflict Risk | ★★★★☆ | Shorter separate PR; M4 changes are additive |

**Effort Estimate**

- Complexity: Medium-High
- Resources: 1 developer, ~15 task-days (separate) + 9 tasks merged into M4
- Dependencies: Phase 1 must complete before M4 starts

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| M4 task group 18 adds scope creep | Medium | Low | Tasks are well-scoped from explore agent findings |
| OpenCode hardening bugs are harder to isolate in M4 | Low | Medium | Verification gates V6/V9 applied within M4 |

---

### Options Comparison Summary

| Criterion | Option 1: Full | Option 2: Core | Option 3: ACP Only | Option 4: Hybrid |
|-----------|----------------|----------------|---------------------|------------------|
| M4 Readiness | ★★★★★ | ★★★☆☆ | ★★☆☆☆ | ★★★★★ |
| Type Safety | ★★★★★ | ★★★★★ | ★★☆☆☆ | ★★★★★ |
| Architectural Cleanliness | ★★★★★ | ★★★★☆ | ★★☆☆☆ | ★★★★★ |
| Implementation Speed | ★★★☆☆ | ★★★★☆ | ★★★★★ | ★★★★☆ |
| Risk of Regression | ★★★☆☆ | ★★★★☆ | ★★★★★ | ★★★★☆ |
| Merge Conflict Risk | ★★☆☆☆ | ★★★☆☆ | ★★★★★ | ★★★★☆ |
| **Overall** | **28/30** | **25/30** | **21/30** | **29/30** |

---

## Recommendation

### Recommended Option

**Option 4: Hybrid — Orthogonal Phases Separately, Overlapping Phases with M4**

### Justification

Option 4 scores highest (29/30) by achieving the same M4 Readiness and Type Safety as Option 1 (full cleanup) while reducing the separate pre-M4 timeline from ~30 to ~15 days. The key insight is that Phases 3 (OpenCode hardening) and 5 (RunScope identity) touch the same OpenCode route files that M4 modifies — doing them separately means two rounds of changes to the same files, with merge conflict risk in between.

On Merge Conflict Risk (weighted Low), Option 4 scores ★★★★☆ vs Option 1's ★★☆☆☆. The separate cleanup is scoped to ACP server, orchestrator, and lifecycle files — none of which M4 modifies. The OpenCode route changes happen once, during M4.

On Risk of Regression (weighted Medium), Option 4 scores ★★★★☆ vs Option 1's ★★★☆☆. The separate changeset is smaller (19 tasks vs 43), and the highest-risk item (ACP path unification) is isolated and tested before M4's multi-config complexity is introduced.

### Accepted Trade-offs

1. **M4 carries 9 additional cleanup tasks (task group 18)**: Acceptable because these tasks are well-scoped from explore agent findings and touch files M4 modifies anyway.
2. **Phase 3 and 5 tracking spans two OpenSpec changes**: Acceptable because the `pre-m4-protocol-cleanup` tasks.md explicitly cross-references the moved tasks, and `m4-multi-config` task group 18 references the origin.
3. **ACP path unification (Phase 1) must complete before M4 starts**: Acceptable because Phase 1 is the highest-risk item and benefits from isolated debugging.
4. **`ACPTurn.execute()` becoming live may surface latent bugs**: Acceptable because snapshot tests provide before/after comparison, and each task is an incremental commit.

### Conditions

- Phase 1 (ACP unification) must complete and pass snapshot tests before M4 begins
- Phases 2, 4, 6 can proceed in parallel after Phase 1
- M4 task group 18 (OpenCode hardening + identity) should be executed early in M4, before RunScope routing implementation (task groups 8-9)
- Phase 7 (nice-to-have) is explicitly optional and does not block M4

---

## Technical Design

### Architecture Overview

#### Current State (Post-M3)

```mermaid
flowchart TB
    subgraph Servers["Protocol Servers"]
        ACP["ACP Server"]
        OC["OpenCode Server"]
        AGUI["AG-UI Server"]
        API["OpenAI API Server"]
    end

    subgraph Infra["Shared Infrastructure"]
        PEC["ProtocolEventConsumerMixin"]
        EB["EventBus"]
        SC["SessionController"]
        RH["RunHandle.start()<br/>~373 SLOC"]
    end

    subgraph Paths["Execution Paths"]
        RSO["_run_stream_once()<br/>ACP hook firing (legacy)"]
        TE["Turn.execute()<br/>via HookAwareTurn (target)"]
        HF["hooks_fired guard<br/>21 refs across 4 files"]
    end

    ACP --> PEC
    OC --> PEC
    AGUI --> PEC
    API --> PEC
    PEC --> EB
    EB --> SC
    SC --> RH
    RH --> RSO
    RH --> TE
    RSO -.->|"fires hooks first"| HF
    TE -.->|"checks guard, skips"| HF

    style RSO fill:#f99,stroke:#c00
    style HF fill:#f99,stroke:#c00
    style RH fill:#f96,stroke:#c60
```

#### Target State (Post-Cleanup)

```mermaid
flowchart TB
    subgraph Servers["Protocol Servers"]
        ACP["ACP Server"]
        OC["OpenCode Server<br/>via HostContext"]
        AGUI["AG-UI Server"]
        API["OpenAI API Server"]
    end

    subgraph Infra["Shared Infrastructure"]
        PEC["ProtocolEventConsumerMixin"]
        EB["EventBus"]
        SC["SessionController<br/>via RunScope"]
        RH["RunHandle<br/>decomposed: _idle_loop, _run_turn, _drain_events"]
    end

    subgraph Path["Single Execution Path"]
        TE["Turn.execute()<br/>via HookAwareTurn"]
        CC["CommChannel<br/>deliver_feedback + publishes_to_event_bus"]
    end

    subgraph Identity["M4 Identity Layer"]
        RS["RunScope<br/>config_id, tenant_id, session_id"]
    end

    ACP --> PEC
    OC --> PEC
    AGUI --> PEC
    API --> PEC
    PEC --> EB
    EB --> SC
    SC --> RH
    RH --> TE
    TE --> CC
    RS --> SC

    style TE fill:#9f9,stroke:#0a0
    style CC fill:#9f9,stroke:#0a0
    style RS fill:#9f9,stroke:#0a0
```

### Phase 1: ACP Execution Path Unification

```mermaid
sequenceDiagram
    participant Client as ACP Client
    participant Agent as ACPAgent
    participant API as ACPAgentAPI
    participant Turn as ACPTurn
    participant HAT as HookAwareTurn

    Note over Client,Turn: Current: Client → Agent._stream_events() → 200 LOC inline
    Note over Client,Turn: Target: Client → Agent._stream_events() → Turn.execute()

    Client->>Agent: run_stream(prompt)
    Agent->>API: create ACPTurn(api, context)
    Agent->>Turn: execute()
    Turn->>HAT: fire_pre_turn_hooks()
    HAT->>HAT: run hooks (no guard needed)
    Turn->>API: stream_events()
    API->>Client: ACP events stream
    Turn->>HAT: fire_post_turn_hooks()
    Turn-->>Agent: turn result
    Agent-->>Client: RichAgentStreamEvent stream
```

**Key changes:**
- `ACPAgentAPI` gains `stream_events()` and `get_messages()` (wrapping `ACPClient`)
- `ACPAgent._stream_events()` delegates to `ACPTurn.execute()` instead of inline implementation
- `AGENT_TYPE != "native"` hook firing branches deleted from `base_agent.py`
- `hooks_fired` set removed from `AgentRunContext`; all 21 references deleted

### Phase 2: CommChannel Protocol Typing

```mermaid
classDiagram
    class CommChannel {
        <<protocol>>
        +publish(event) void
        +recv() Feedback | None
        +deliver_feedback(feedback) void
        +publishes_to_event_bus bool
        +close() void
    }

    class DirectChannel {
        +publish(event) void
        +recv() None
        +deliver_feedback(feedback) void
        +publishes_to_event_bus bool = false
    }

    class ProtocolChannel {
        +publish(event) void
        +recv() Feedback | None
        +deliver_feedback(feedback) void
        +publishes_to_event_bus bool = true
    }

    CommChannel <|.. DirectChannel
    CommChannel <|.. ProtocolChannel
```

**Key changes:**
- `deliver_feedback(feedback: Feedback) -> None` added to `CommChannel` protocol
- `DirectChannel.deliver_feedback` = no-op; `ProtocolChannel.deliver_feedback` = enqueue to feedback queue
- `publishes_to_event_bus: bool` property replaces `isinstance(channel, ProtocolChannel)` check
- `RunHandle` holds direct `_journal` reference instead of accessing `self._comm_channel._journal`
- 8 `# type: ignore[attr-defined]` in `run.py` eliminated

### Phase 3: OpenCode Server — Private Access & state.pool Migration

```mermaid
flowchart LR
    subgraph Before["Before (68 accesses)"]
        R1["routes/*.py"] -->|"state.pool.session_pool"| P1["AgentPool"]
        R2["routes/*.py"] -->|"state.pool.manifest"| P1
        R3["routes/*.py"] -->|"state.pool.todos"| P1
        R4["routes/*.py"] -->|"state.pool.skills"| P1
        R5["routes/*.py"] -->|"state.pool.storage"| P1
        R6["routes/*.py"] -->|"agent._all_capabilities"| A1["Agent"]
        R7["routes/*.py"] -->|"session_ctrl._sessions"| SC1["SessionController"]
    end

    subgraph After["After (via HostContext)"]
        R1b["routes/*.py"] -->|"state.host_context.session_pool"| HC["HostContext"]
        R2b["routes/*.py"] -->|"state.host_context.manifest"| HC
        R3b["routes/*.py"] -->|"state.host_context.todos"| HC
        R4b["routes/*.py"] -->|"state.host_context.skills"| HC
        R5b["routes/*.py"] -->|"state.host_context.storage"| HC
        R6b["routes/*.py"] -->|"agent.get_capabilities()"| A2["Agent (public API)"]
        R7b["routes/*.py"] -->|"session_ctrl.get_session()"| SC2["SessionController (public API)"]
    end

    style Before fill:#f99,stroke:#c00
    style After fill:#9f9,stroke:#0a0
```

**New public API methods:**

| Method | Replaces |
|--------|----------|
| `Agent.get_capabilities() -> list[AbstractCapability]` | `agent._all_capabilities` |
| `Agent.get_all_tools() -> list[Tool]` | `agent._get_all_tools()` |
| `SessionController.get_session(id) -> SessionState \| None` | `session_controller._sessions[id]` |
| `LspManager.get_server(name) -> LspServer \| None` | `lsp_manager._servers[name]` |
| `SessionPool.get_runs(session_id) -> dict` | `session_pool.sessions._runs` |

### Phase 4: RunHandle.start() Decomposition

```mermaid
flowchart TB
    subgraph Before["Before: ~373 SLOC monolith"]
        start1["start()"] --> everything1["_idle + _run_turn + _drain + _recovery<br/>all in one method<br/># noqa: PLR0915"]
    end

    subgraph After["After: composable sub-methods"]
        start2["start()"] --> recover["_handle_recovery()<br/>crash recovery"]
        recover --> idle["_idle_loop()<br/>wait for prompts"]
        idle --> run["_run_turn()<br/>Turn.execute()"]
        run --> drain["_drain_events()<br/>event queue flush"]
        drain --> idle
    end

    style Before fill:#f99,stroke:#c00
    style After fill:#9f9,stroke:#0a0
```

### Phase 5: RunScope Identity Abstraction

```mermaid
flowchart LR
    subgraph Current["Current Identity Model"]
        session_id["session_id = agent.name"]
        pool_id["pool_id = config_file_path"]
        agents["agents = pool.manifest.agents"]
    end

    subgraph PreM4["Pre-M4 (RunScope with defaults)"]
        rs1["RunScope("] --> rs2["config_id='default'<br/>session_id=agent.name<br/>tenant_id='default'"]
    end

    subgraph M4["M4 (RunScope from protocol)"]
        rs3["RunScope("] --> rs4["config_id=from_header<br/>session_id=from_header<br/>tenant_id=from_header"]
    end

    Current --> PreM4 --> M4

    style Current fill:#f99,stroke:#c00
    style PreM4 fill:#ff9,stroke:#c60
    style M4 fill:#9f9,stroke:#0a0
```

```python
@dataclass(frozen=True)
class RunScope:
    config_id: str = "default"
    tenant_id: str = "default"
    session_id: str | None = None  # defaults to agent.name pre-M4
    user_id: str | None = None
```

---

## Implementation Plan

### Phases

```mermaid
gantt
    title Pre-M4 Cleanup Implementation Plan
    dateFormat YYYY-MM-DD
    axisFormat %m/%d

    section Phase 1: ACP Unification
    1.1 ACPAgentAPI adapter           :p11, 2026-07-14, 2d
    1.2 Refactor _stream_events       :p12, after p11, 2d
    1.3 Remove hook firing            :p13, after p12, 1d
    1.4 Remove hooks_fired            :p14, after p13, 1d
    1.5 Remove deprecated APIs        :p15, after p14, 1d
    1.6 Remove display_mode field     :p16, after p15, 1d

    section Phase 2: Legacy Cleanup
    2.1 MCP state consolidation       :p21, 2026-07-14, 3d
    2.2 CommChannel.deliver_feedback  :p22, 2026-07-14, 2d
    2.3 Remove pool escape hatch      :p23, after p22, 1d
    2.4 Remove RunStatus enum         :p24, after p22, 2d
    2.5 Remove dead code              :p25, after p24, 1d

    section Phase 3: OpenCode Hardening
    3.1 Public API methods            :p31, 2026-07-14, 3d
    3.2 state.pool migration          :p32, after p31, 4d
    3.3 RunStartedEvent handler       :p33, after p31, 1d
    3.4 Remove dual abort paths       :p34, after p32, 1d
    3.5 Remove legacy fallback        :p35, after p34, 1d

    section Phase 4: Type Safety
    4.1 RunHandle dimension refs      :p41, after p22, 2d
    4.2 publishes_to_event_bus        :p42, after p41, 1d
    4.3 Replace hasattr               :p43, after p13, 1d
    4.4 Fix ACPTurn except clauses    :p44, after p13, 1d
    4.5 Refactor start()              :p45, after p41, 3d
    4.6 Remove typing.Any             :p46, after p33, 1d

    section Phase 5: M4 Identity
    5.1 RunScope dataclass            :p51, after p45, 1d
    5.2 Session identity              :p52, after p51, 2d
    5.3 Pool identity                 :p53, after p51, 1d
    5.4 Remove hardcoded agents       :p54, after p52, 2d

    section Phase 6: Event Gaps
    6.1 McpToolsChangedEvent          :p61, after p33, 2d
    6.2 Cancelled distinction         :p62, after p33, 1d
    6.3 Remove deprecated adapter     :p63, after p62, 1d
```

### Phase Dependencies

```mermaid
flowchart TD
    p11["1.1 ACPAgentAPI"] --> p12["1.2 Refactor _stream_events"]
    p12 --> p13["1.3 Remove hook firing"]
    p13 --> p14["1.4 Remove hooks_fired"]
    p14 --> p15["1.5 Remove deprecated APIs"]
    p15 --> p16["1.6 Remove display_mode"]

    p21["2.1 MCP state consolidation"] --> p22["2.2 CommChannel.deliver_feedback"]
    p22 --> p23["2.3 Remove pool escape hatch"]
    p22 --> p24["2.4 Remove RunStatus enum"]
    p24 --> p25["2.5 Remove dead code"]

    p31["3.1 Public API methods"] --> p32["3.2 state.pool migration"]
    p31 --> p33["3.3 RunStartedEvent"]
    p32 --> p34["3.4 Remove dual abort"]
    p34 --> p35["3.5 Remove legacy fallback"]

    p22 --> p41["4.1 RunHandle dimension refs"]
    p41 --> p42["4.2 publishes_to_event_bus"]
    p13 --> p43["4.3 Replace hasattr"]
    p13 --> p44["4.4 Fix ACPTurn except"]
    p41 --> p45["4.5 Refactor start()"]
    p33 --> p46["4.6 Remove typing.Any"]

    p45 --> p51["5.1 RunScope dataclass"]
    p51 --> p52["5.2 Session identity"]
    p51 --> p53["5.3 Pool identity"]
    p52 --> p54["5.4 Remove hardcoded agents"]

    p33 --> p61["6.1 McpToolsChangedEvent"]
    p33 --> p62["6.2 Cancelled distinction"]
    p62 --> p63["6.3 Remove deprecated adapter"]

    p14 --> p714["7.14 Update test mocks"]

    style p11 fill:#f99
    style p14 fill:#f99
    style p22 fill:#f99
    style p45 fill:#f99
    style p51 fill:#ff9
```

### Milestones

| Milestone | Description | Target | Status |
|-----------|-------------|--------|--------|
| M-1 | Phase 1 complete: ACP unified, snapshot test green | Day 8 | Not Started |
| M-2 | Phases 2-4 complete: Type safety gates pass | Day 18 | Not Started |
| M-3 | Phase 5 complete: RunScope abstraction in place | Day 24 | Not Started |
| M-4 | Phase 6 complete: All verification gates green | Day 28 | Not Started |
| M-5 | M4 can begin | Day 30 | Not Started |

### Rollback Strategy

Each phase is a set of incremental commits. If a phase introduces regressions:

1. Revert the phase's commits via `git revert`
2. The phases are designed to be independent (except Phase 5 depends on 1–4)
3. Phase 1 can be reverted independently if ACP path unification surfaces critical bugs
4. `hooks_fired` guard can be restored if double-firing is detected after removal

---

## Open Questions

1. **`_agent_pool` constructor threading**
   - Context: 25+ refs thread `_agent_pool` through constructors for internal wiring. Removing this requires changing constructor signatures across `MessageNode` and 3 agent files.
   - Question: Should this be done now or deferred to the AgentWolf rename (where constructor signatures change anyway)?
   - Owner: Leoyzen
   - Status: Open — lean toward deferring to rename

2. **`session_pool_integration.py` file split**
   - Context: 1,453 LOC in one file with 4 responsibilities (event consumer, event adapter, session lifecycle, tool registration). Splitting is mechanical but touches many imports.
   - Question: Worth doing before M4 or during M4 when the file is already being modified for RunScope?
   - Owner: Leoyzen
   - Status: Open — lean toward during M4

3. **`RunScope` placement**
   - Context: `RunScope` needs to be consumed by both `RunHandle` (orchestrator layer) and protocol servers (server layer). RFC-0050 places it in the protocol layer.
   - Question: Should `RunScope` live in `orchestrator/` (runtime concern) or `lifecycle/` (infrastructure concern)?
   - Owner: Leoyzen
   - Status: Open — lean toward `orchestrator/` since it's a runtime routing concept

---

## Decision Record

> Complete this section after RFC review is concluded.

### Decision

**Status**: Pending Review

**Date**: TBD

**Approvers**:
- Leoyzen

### Decision Summary

TBD

### Key Discussion Points

TBD

### Conditions of Approval

TBD

### Dissenting Opinions

TBD

---

## References

### Related Documents

- OpenSpec Change: pre-m4-protocol-cleanup
- Design Document
- Task Breakdown
- [RFC-0050: AgentWolf v1 Foundation Architecture](./RFC-0050-agentwolf-v1-foundation-architecture.md)
- [RFC-0042: Unified Lifecycle Architecture](./RFC-0042-unified-lifecycle-architecture.md)
- [RFC-0041: Loop/Run Separation](./RFC-0041-loop-run-separation.md)

### External Resources

- [PR #144: refactor/agentwolf v1](https://github.com/Leoyzen/wolfharness/pull/144)

### Appendix

#### Debt Item Inventory (51 items)

| ID | Category | Severity | File | Description |
|----|----------|----------|------|-------------|
| B1 | ACP | Blocking | `acp_agent.py:632-662` | `ACPAgentAPI` missing `stream_events()`, `get_messages()` |
| B2 | ACP | Blocking | `base_agent.py:1504-1669` | Dual execution paths |
| B3 | ACP | Blocking | 4 files, 21 refs | `hooks_fired` double-fire guard |
| B4 | ACP | Blocking | `agent.py:337-338` | `_mcp_snapshot`, `_session_connection_pool` legacy fields |
| B5 | ACP | Blocking | `run.py:822-878` | `deliver_feedback` duck-typed, 4 `type: ignore` |
| B6 | ACP | Blocking | `event_converter.py:166-168` | Deprecated `subagent_display_mode` field |
| S1 | OpenCode | Severe | 6 route files | Private attribute access |
| S2 | OpenCode | Severe | OpenCode routes | 68 `state.pool.*` bypassing HostContext |
| S3 | OpenCode | Severe | `event_processor.py` | `RunStartedEvent` not handled |
| S4 | OpenCode | Severe | `session_routes.py:914-947` | Dual abort paths |
| S5 | OpenCode | Severe | `session_routes.py:660-665` | Legacy `list_sessions()` fallback |
| S6 | OpenCode | Severe | 5 files | Hardcoded `state.agent.name` as identity |
| S7 | OpenCode | Severe | 3 files | `config_file_path` as pool_id |
| S8 | OpenCode | Severe | `agent_routes.py`, `pty_routes.py` | Direct `state.agent.env` access |
| 1.1 | Shared | Moderate | `base_agent.py:1589-1670` | Hook firing dual path |
| 1.2 | Shared | Moderate | `run.py:286-303` | Dual event publishing |
| 1.3 | Shared | Moderate | `turn.py:96-301` | `hooks_fired` guard as migration artifact |
| 1.4 | Shared | Moderate | `session_controller.py:846-891` | `_consume_run` consumer dance |
| 2.1 | Shared | Moderate | `base_team.py:411` | `HostContext.pool` escape hatch (1 remaining) |
| 2.2 | Shared | Moderate | `run.py:130-998` | Legacy `RunStatus` enum coexists with `RunState` |
| 3.1 | Shared | Moderate | `session_controller.py:484-491` | Dead code after return |
| 3.2 | Shared | Moderate | `session_controller.py:436-964` | Single-config hardcoding |
| 3.3 | Shared | Moderate | `acp_agent/turn.py:180-240` | Generic `except Exception` clauses |
| 3.4 | Shared | Moderate | `run.py:401-779` | `RunHandle.start()` ~373 SLOC |
| 4.1 | Shared | Moderate | `run.py:286-303` | `_channel_publishes_to_event_bus` fragile isinstance |
| 4.2 | Shared | Moderate | `run.py` 8 sites | `# type: ignore[attr-defined]` cluster |
| 4.3 | Shared | Moderate | `session_controller.py:964` | `# type: ignore[arg-type]` |
| 4.4 | Shared | Moderate | `base.py:56` | `BaseServer.pool` weakly typed |
| M1-M6 | OpenCode | Moderate | Various | Duplicated logic, file size, unwired events |
| N1-N12 | ACP | Nice-to-have | Various | TODOs, deprecated code, `hasattr`/`getattr` patterns |
| N1-N5 | OpenCode | Nice-to-have | Various | Deprecated test artifacts, TODOs, typing |
